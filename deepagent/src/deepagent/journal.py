# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""The event journal: one append-only stream per job, everything consumes it.

A job's whole trace -- agent thinking and tool calls, step transitions,
verification runs, notifications -- lands in one `events.jsonl` under the job's
control dir, streamed as it happens (each append flushes). It is the single
observability substrate: `icu tail` follows it for a human, and the reentry
loop reads its work-event count (`progress`) as the liveness signal (a turn
that produced no result but emitted agent output/tool calls is alive -- a long
build still running -- and is retried for free, retiring the old per-turn
`.progress` file).

Harness-agnostic by construction: the journal is a plain file the application
owns and passes down; the langgraph harness writes agent events into it via a
callback, but any backend can append the same records. Events are typed
(constrained by a schema, not scraped) with a small set of common columns and
a `data` dict for kind-specific overflow, so a new event kind never migrates
the reader.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class EventKind(StrEnum):
    # agent-turn events (written by the harness callback)
    THINKING = "thinking"  # a reasoning block's text
    MESSAGE = "message"  # assistant response text
    TOOL_START = "tool_start"  # a tool call (name + truncated args)
    TOOL_END = "tool_end"  # a tool result (name + truncated output)
    REPORT = "report"  # the structured turn report (disposition + summary)
    FRICTION = "friction"  # agent-reported friction (broken env, wasted rounds)
    USAGE = "usage"  # per-turn token/call accounting
    TURN = "turn"  # a turn began (agent + index)
    # orchestration events (written by the machine / scheduler)
    STEP = "step"  # a workflow step transition
    VERIFY = "verify"  # a machine acceptance check outcome
    NOTIFY = "notify"  # a human-facing notification
    BUDGET = "budget"  # a revise/poll budget was consumed


# What counts as the agent doing work this turn, for the reentry loop's liveness
# signal. Deliberately excludes TURN and USAGE: run_once emits those two
# UNCONDITIONALLY every turn (at start and end), even on a timeout or an error
# that produced no result -- so counting them would make every turn look alive
# and defeat the dead-turn cap entirely. Orchestration events (STEP/VERIFY/
# NOTIFY/BUDGET) are written by the machine outside the turn and are excluded
# for the same reason: only the agent's own output (content + tool calls, and
# the structured report) proves the turn advanced.
_PROGRESS_KINDS = frozenset(
    {
        EventKind.THINKING,
        EventKind.MESSAGE,
        EventKind.TOOL_START,
        EventKind.TOOL_END,
        EventKind.REPORT,
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Event(BaseModel):
    kind: EventKind
    ts: str = Field(default_factory=_now)
    agent: str = ""  # which agent/step produced it, when applicable
    turn: int | None = None  # agent-turn index, when applicable
    # Common columns the reader displays directly; kind decides which are set.
    name: str = ""  # tool / budget / notify-event / verify-command name
    text: str = ""  # thinking / message / reason / detail body
    data: dict = Field(default_factory=dict)  # kind-specific overflow


class Journal:
    """Append-only writer over one job's events.jsonl. Each append flushes so a
    follower sees events live. `count` is the total appended; `progress` counts
    only agent work events (_PROGRESS_KINDS) and is the reentry loop's liveness
    cursor -- a turn that advanced `progress` did real work and is retried free,
    while `count` growth alone (which the harness's own TURN/USAGE bookkeeping
    guarantees every turn) does not prove liveness.

    Appends are serialized by a lock: the langgraph harness drives this from
    langchain callback hooks that run on a shared executor thread pool, so
    multiple hooks in one turn can append concurrently. Without the lock the
    per-append file writes interleave and tear records mid-line, and `read`
    only tolerates a torn *trailing* line -- a mid-file tear silently drops
    events from the sole crash-observability substrate."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.count = 0
        self.progress = 0
        self._lock = threading.Lock()

    def append(self, event: Event) -> None:
        # Serialize the JSON outside the lock (no shared state); hold the lock
        # only across the write + cursor updates so concurrent callback threads
        # cannot interleave a partial line or race the counters.
        line = event.model_dump_json() + "\n"
        is_progress = event.kind in _PROGRESS_KINDS
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as f:
                    f.write(line)
                    f.flush()
            except OSError as e:  # a broken journal must never crash the run
                log.warning("journal: append failed for %s: %s", self.path, e)
                return
            self.count += 1
            if is_progress:
                self.progress += 1

    def emit(self, kind: EventKind, **fields) -> None:
        """Construct and append in one call: journal.emit(EventKind.STEP, ...)."""
        self.append(Event(kind=kind, **fields))

    @staticmethod
    def read(path: Path | str) -> list[Event]:
        """Load a journal from disk, skipping any malformed trailing line (a
        crash mid-append). For `icu tail` and finalize."""
        out: list[Event] = []
        try:
            lines = Path(path).read_text().splitlines()
        except OSError:
            return out
        for line in lines:
            if not line.strip():
                continue
            try:
                out.append(Event.model_validate_json(line))
            except ValueError:
                continue  # partial write at the tail
        return out

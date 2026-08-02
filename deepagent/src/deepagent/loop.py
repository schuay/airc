# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""The reentry loop: re-invoke the harness until it signals termination.

This is the heart of the control structure. The caller runs the harness, reads
its `disposition`, and loops while it is CONTINUE; COMPLETE/ABANDON/BLOCKED are
terminal. An iteration cap guarantees the loop terminates even if the agent
never stops signalling CONTINUE -- termination is a property of this code, not a
property the agent has to be trusted for. Continuity between iterations lives in
the worktree and a scratchpad under the control dir; the backend decides whether
a turn resumes the prior context or starts fresh.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .harness import REPORT_TOOL_NAME, AgentResult, Disposition, Harness
from .journal import Journal

log = logging.getLogger(__name__)


# A mid-run reflection turn, injected once at caps.checkpoint_turn. It exists
# because "Continue from where you left off" every turn is pure momentum: a loop
# that cannot reach its goal keeps trying tactical variants instead of stepping
# back to reason about whether the goal is reachable at all. This forces that
# mode switch once, without capping effort -- "one specific new hypothesis" is a
# valid outcome, so a hard-but-reachable goal gets a sharper attempt, not an
# early abort. Deliberately generic (achievability, not domain specifics): the
# stage's own system prompt defines what a negative verdict means (for a repro,
# reproduced=false with evidence).
_REFLECT = (
    "Step back before continuing. You have made several attempts -- assess"
    " achievability, not tactics: is the goal actually reachable here, or are"
    " you accumulating evidence that it cannot be (an upstream guard, a flag"
    " default, a type check, a mistaken premise in the task)? Record that"
    " assessment in your casefile, then decide explicitly: either state one"
    " specific new hypothesis worth another turn, or report your terminal"
    " verdict now with the evidence. Repeating the same approach is not a valid"
    " choice."
)
# The last turn: the loop is about to synthesize an ABANDON, which discards the
# reasoned negative verdict the agent was likely converging toward. Force the
# verdict instead, so the ceiling yields an earned answer, not a non-answer.
_FINAL = (
    "This is your final turn -- do not continue. Commit to a terminal verdict"
    f" now via the `{REPORT_TOOL_NAME}` tool, backed by your evidence: your best"
    " result if you have one, or a reasoned negative verdict (why the goal could"
    " not be achieved) if you do not."
)


@dataclass
class LoopCaps:
    max_iters: int = 20
    timeout_s: float = 3600.0  # per harness invocation
    # Consecutive *dead* turns (no result AND no progress) tolerated before
    # abandoning. A turn that advanced its progress file is alive (e.g. a long
    # build still running) and is retried for free -- max_iters is the ceiling on
    # progressing work. This cap only catches a harness that produces nothing.
    no_result_cap: int = 3
    # Turn index at which to inject the reflection prompt (0-based). None keeps
    # the pre-checkpoint behavior (plain continue every turn). Applications that
    # want the mode switch set it, typically to max_iters // 2.
    checkpoint_turn: int | None = None


def _resume_prompt(i: int, caps: LoopCaps) -> str:
    """The instruction for resume turn `i` (i > 0): a plain continue with turn
    awareness, escalating to a reflection checkpoint and then a final-turn
    verdict force. Turn awareness lets the model pace itself against the cap
    instead of being surprised by it."""
    n = caps.max_iters
    if i >= n - 1:
        return _FINAL
    head = (
        f"Continue from where you left off (turn {i + 1} of {n}) -- your previous"
        " report was a checkpoint, not completion, so do the next concrete step of"
        f" work now. When you pause or finish, call the `{REPORT_TOOL_NAME}` tool."
    )
    if caps.checkpoint_turn is not None and i == caps.checkpoint_turn:
        return f"{head}\n\n{_REFLECT}"
    return head


# The per-attempt ledger, appended to on every turn. Its readers, in order of
# how much they can be trusted to exist: a human following the job, the agent's
# own next attempt (which reads conclusions rather than re-deriving them), and
# the resume path when a checkpoint is missing. That ordering is why the ledger
# is a plain file and the checkpoint is a cache -- correctness rests on the file.
ATTEMPTS_FILE = "ATTEMPTS.md"


def _append_attempt(casefile: Path | None, agent: str, turn: int, result) -> None:
    if casefile is None:
        return
    parts = [f"## {agent or 'agent'} attempt {turn + 1} -- {result.disposition.value}"]
    if result.summary:
        parts.append(result.summary.strip())
    if result.reason:
        parts.append(f"reason: {result.reason.strip()}")
    try:
        casefile.mkdir(parents=True, exist_ok=True)
        with (casefile / ATTEMPTS_FILE).open("a") as f:
            f.write("\n\n".join(parts) + "\n\n")
    except OSError as e:
        # The ledger is for the reader and the next attempt; losing an entry
        # costs context, never the run.
        log.warning("could not append to %s: %s", ATTEMPTS_FILE, e)


def _with_interjection(prompt: str, interject) -> str:
    """Append pending out-of-band news to a turn's prompt.

    Also applies to turn 0 (whose resume prompt is empty), so a goal that starts
    with news already waiting carries it into its first turn rather than only
    from the second.
    """
    if interject is None or not (news := interject()):
        return prompt
    return f"{prompt}\n\n{news}" if prompt else news


async def _resume_notice(workdir: Path) -> str:
    """What a turn needs to know when it resumes into a tree it did not leave.

    A restart leaves whatever the previous attempt was mid-way through: a
    half-applied edit, an interrupted build, a conflicted rebase. An agent that
    assumes a clean tree compounds that, so it is handed the actual state rather
    than left to check -- deterministic, cheap, and in front of it instead of
    dependent on it asking.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(workdir),
            "status",
            "--short",
            "--branch",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    except (OSError, asyncio.TimeoutError):
        return ""
    status = out.decode(errors="replace").strip()
    return (
        "You are RESUMING after an interruption -- this loop restarted, so work"
        " from an earlier attempt may be partial: a half-applied edit, an"
        " interrupted build, an unfinished rebase. Read your casefile"
        f" {ATTEMPTS_FILE} for what was already tried, and reconcile the tree"
        " before continuing.\n\nCurrent `git status --short --branch`:\n\n"
        f"{status or '(clean)'}"
    )


async def run_agent_loop(
    harness: Harness,
    *,
    prompt_path: Path,
    workdir: Path,
    control_dir: Path,
    caps: LoopCaps,
    agent: str = "",
    casefile: Path | None = None,
    journal: Journal | None = None,
    interject: Callable[[], str] | None = None,
) -> AgentResult:
    """Drive the harness to a terminal disposition (or synthesize an ABANDON).

    Returns the agent's terminal AgentResult. If the cap fires first, returns a
    synthetic ABANDON so the caller always sees a terminal result and never an
    open-ended CONTINUE. When a journal is given, the harness streams the turn's
    events into it and its work-event count (`progress`) is the liveness signal
    (a turn that produced no result but emitted agent output/tool calls is alive
    -- a long build still running).

    `interject` is consulted BETWEEN turns and its text, when non-empty, is
    appended to the next turn's prompt. That is the whole mechanism for telling a
    RUNNING goal something the orchestrator learned after it started (new review
    comments, say): the turn boundary already exists and already carries a
    per-turn string, so nothing new couples the caller to a live turn, and the
    agent is never interrupted mid-edit. The text rides the resume prompt rather
    than the system prompt on purpose -- the growing-prefix cache covers
    [system + tools], so mutating the system text re-caches mid-goal.
    """
    control_dir.mkdir(parents=True, exist_ok=True)
    # Turn 0 of a loop whose control dir already holds results is a RESUME: the
    # previous run of this same goal died. (A fresh goal gets a fresh control
    # dir, keyed by step and round, so this cannot false-positive on a revise.)
    resumed_notice = (
        await _resume_notice(workdir) if any(control_dir.glob("result.*.json")) else ""
    )
    if resumed_notice:
        log.info("%s: resuming into an existing control dir", agent or "agent")
    consecutive_empty = 0
    for i in range(caps.max_iters):
        result_path = control_dir / f"result.{i:03d}.json"
        before = journal.progress if journal is not None else 0
        run = await harness.run_once(
            prompt_path=prompt_path,
            workdir=workdir,
            result_path=result_path,
            timeout_s=caps.timeout_s,
            agent=agent,
            resume=i > 0,  # turn 0 sends the task; later turns continue the thread
            resume_prompt=_with_interjection(
                _resume_prompt(i, caps) if i > 0 else resumed_notice, interject
            ),
            casefile=casefile,
            journal=journal,
        )
        if run.result is None:
            # No valid result: a turn that was cut (timeout) or violated the
            # contract. If the turn advanced the journal it was alive (a long
            # build still running, the common case), so retry for free and let
            # the worktree's build resume; max_iters bounds it. Only a turn that
            # produced nothing at all -- no result, no journal growth -- counts
            # as a dead harness toward the cap.
            #
            # An exhausted empty candidate is the exception: dead whatever the
            # journal says. progress is measured against the START of the turn,
            # so one tool call minutes before the model went silent scores it
            # alive and resets the streak. Observed: a turn whose last tool call
            # preceded the silence by ~7 minutes retried free, and the identical
            # failure repeated. Excluding it lets no_result_cap bound the churn.
            alive = (
                journal is not None
                and journal.progress > before
                and not run.empty_candidate
            )
            if alive:
                log.info(
                    "%s turn %d: no result but journal advanced; retrying free",
                    agent or "agent",
                    i,
                )
                consecutive_empty = 0  # alive: breaks the dead-turn streak
                continue
            consecutive_empty += 1
            # finish_reason distinguishes a provider-side empty candidate (a
            # deterministic SAFETY/RECITATION/MALFORMED_FUNCTION_CALL block that
            # reproduces every resume) from a plain cut turn; empty_candidate names
            # the silent shape (a zero-part STOP). Name both in the log and the
            # abandon reason so a dead-turn abandon is not opaque.
            why = f"exit {run.exit_code}"
            if run.finish_reason:
                why += f", finish_reason={run.finish_reason}"
            if run.empty_candidate:
                why += " (empty candidate)"
            # Point at the journal, not run.log_path: the langgraph harness
            # derives that name (result_path.with_suffix(".log")) but nothing
            # ever writes it, so the old hint sent an operator to a file that
            # does not exist. events.jsonl is written and flushed per event.
            log.warning(
                "%s turn %d: no result and no progress (%s), retry %d/%d -- inspect %s",
                agent or "agent",
                i,
                why,
                consecutive_empty,
                caps.no_result_cap,
                journal.path if journal is not None else run.log_path,
            )
            if consecutive_empty >= caps.no_result_cap:
                return AgentResult(
                    disposition=Disposition.ABANDON,
                    reason=f"no valid result after {consecutive_empty} dead attempts "
                    f"({why})",
                )
            continue
        consecutive_empty = 0
        _append_attempt(casefile, agent, i, run.result)
        if run.result.disposition is not Disposition.CONTINUE:
            # Terminal: the stored conversation's only remaining value was
            # resuming it, so let the harness drop it rather than accumulate
            # every attempt of every job in a durable saver.
            if (forget := getattr(harness, "forget", None)) is not None:
                with contextlib.suppress(Exception):
                    await forget(control_dir)
            return run.result
    return AgentResult(
        disposition=Disposition.ABANDON,
        reason=f"did not converge in {caps.max_iters} iterations",
    )

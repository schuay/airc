"""Shared supervised poll loop for bus subscribers.

One loop per subscription: poll new messages, handle each, then ack -- so the
cursor advances only after a message is durably handled (at-least-once; handlers
must be idempotent). A failed poll is logged and the loop continues, so one
wedged handler never takes the others down. Topics are file-backed, so polling
on an interval is cheap.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from bus import Envelope, Subscription
from pydantic import ValidationError

from ..store import MessageKind

if TYPE_CHECKING:
    from ..room import Room

log = logging.getLogger(__name__)

POLL_INTERVAL = 30.0

# A handler returns this to defer its ack: the event was only batched in memory
# (not yet durably handled), so the cursor must not advance past it until the
# drain-end flush (on_drain_end) lands the batch. A crash or a failed flush then
# redelivers the batch next drain instead of losing it -- the flush must clear
# its batch up front so a redelivered pass rebuilds it without duplicates.
DEFER = object()

# Age helpers + a freshness gate. On restart a subscriber drains every event
# published since its cursor; for an ephemeral stream (commit commentary, perf)
# replaying a long backlog after downtime floods the room with stale chatter.
# The gate acks (skips) events older than a window so a short outage still
# catches up while a long one does not -- keyed on the event's own timestamp,
# never the envelope's publish time, because the watcher is in the same suite:
# a full-suite restart republishes the whole backlog with a fresh publish time,
# which would defeat a publish-time filter. Durable streams (findings) pass no
# window and so always replay.
AgeOf = Callable[[Envelope], "float | None"]
CaughtUp = Callable[[int, float], Awaitable[None]]


def age_seconds(iso: str) -> float | None:
    """Seconds between now and an ISO8601 timestamp, or None when empty or
    unparseable -- the caller treats None as 'cannot tell, do not skip'."""
    if not iso:
        return None
    try:
        t = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - t).total_seconds()


def fmt_age(seconds: float) -> str:
    s = int(seconds)
    if s < 3600:
        return f"{max(1, s // 60)}m"
    if s < 86400:
        return f"{s // 3600}h"
    return f"{s // 86400}d"


async def post_caught_up(
    room: "Room", noun: str, skipped: int, oldest: float, window: float
) -> None:
    """A single operational notice after a restart drained a stale backlog, so
    the room shows the gap was handled rather than going eerily silent. Its own
    thread (these events spawn a thread each, so there is no shared home), kind
    `notice` so the orchestrator does not route it to a persona."""
    thread = room.create_thread("[airc] caught up")
    await room.post(
        thread.id,
        "airc",
        MessageKind.NOTICE,
        f"caught up: skipped {skipped} {noun} older than {fmt_age(window)} "
        f"(oldest ~{fmt_age(oldest)} back)",
    )


async def _drain(
    name: str,
    sub: Subscription,
    handle: Callable[[Envelope], Awaitable[None]],
    max_age: float | None,
    age_of: AgeOf | None,
    on_caught_up: CaughtUp | None,
    on_drain_end: Callable[[], Awaitable[None]] | None = None,
) -> None:
    skipped = 0
    oldest = 0.0
    deferred_hwm = 0  # highest DEFERred seq; acked only once the flush lands

    async def flush_deferred() -> None:
        # Land the batch, then advance the cursor past it -- in that order, so
        # a crash or flush failure redelivers the batch instead of losing it.
        # Must run before any LATER ack: the cursor is monotonic, so acking a
        # later seq would silently advance past the unflushed batch. (In
        # practice seq order is publish order is age order, so deferred events
        # form a prefix and the mid-loop calls are correctness guards only.)
        nonlocal deferred_hwm
        if not deferred_hwm:
            return
        await on_drain_end()
        sub.ack(deferred_hwm)
        deferred_hwm = 0

    for seq, env in sub.poll():
        if max_age is not None and age_of is not None:
            # The age lambdas parse the payload, and they run before the
            # per-message guard below: a poison event raising ValidationError
            # here would escape _drain un-acked and wedge the topic on the same
            # seq forever. Treat it as "cannot tell, do not skip" and fall
            # through to handle, whose guard skips it loudly.
            try:
                age = age_of(env)
            except ValidationError:
                age = None
            if age is not None and age > max_age:
                skipped += 1
                oldest = max(oldest, age)
                await flush_deferred()
                sub.ack(seq)
                continue
        try:
            result = await handle(env)
        except ValidationError as e:
            # A poison event -- its payload no longer fits the schema (a producer
            # on a newer/older event shape). Deterministic, so retrying forever
            # would wedge the whole topic. Skip it loudly (ERROR:) and advance; a
            # transient handler error still propagates below and retries.
            log.error(
                "ERROR: %s: unparseable event at seq %d; skipping: %s", name, seq, e
            )
            result = None
        if result is DEFER:
            if on_drain_end is None:
                # Nothing will ever flush (and so ack) this event; better a loud
                # retry loop than a cursor silently pinned forever.
                raise RuntimeError(f"{name}: handler deferred without on_drain_end")
            deferred_hwm = seq
            continue
        await flush_deferred()
        sub.ack(seq)
    if skipped and on_caught_up is not None:
        await on_caught_up(skipped, oldest)
    await flush_deferred()


async def subscribe_loop(
    name: str,
    sub: Subscription,
    handle: Callable[[Envelope], Awaitable[None]],
    interval: float = POLL_INTERVAL,
    *,
    max_age: float | None = None,
    age_of: AgeOf | None = None,
    on_caught_up: CaughtUp | None = None,
    on_drain_end: Callable[[], Awaitable[None]] | None = None,
) -> None:
    while True:
        try:
            await _drain(name, sub, handle, max_age, age_of, on_caught_up, on_drain_end)
        except Exception:
            log.exception("%s: poll failed", name)
        await asyncio.sleep(interval)


# Focus-aware triage shared by subscribers that decide whether a world event is
# worth commentary. Moved here from the old watchers/base.py when the repo and
# perf watchers became separate publisher processes; the commentary subscriber is
# the remaining consumer (perf no longer triages -- its publisher gates upstream).

_SKIP_FLOOR = """\
Always answer SKIP for: dependency rolls, reverts and relands, version bumps,
auto-generated changes, trivial typo or rename changes, and anything fully
explained by its own description with nothing left for an expert to add."""

_GENERIC_CRITERIA = """\
Answer INTERESTING if it introduces a feature, API, or notable behaviour change,
fixes a notable (security, correctness, data-loss) bug, makes an architectural or
design decision, or has substantial performance or reliability impact."""

_FILTER_TEMPLATE = """\
You are a senior engineer triaging {kind} to decide if it warrants a short
expert commentary in a team chat.

{guidance}

{skip_floor}

Reply with EXACTLY one word: INTERESTING or SKIP."""


def filter_system_prompt(kind: str, focus: str) -> str:
    guidance = (
        f"What to surface for this source:\n{focus.strip()}"
        if focus.strip()
        else _GENERIC_CRITERIA
    )
    return _FILTER_TEMPLATE.format(kind=kind, guidance=guidance, skip_floor=_SKIP_FLOOR)


async def focus_interesting(
    model,
    kind: str,
    focus: str,
    text: str,
    *,
    tokens=None,
    model_id: str = "",
) -> bool:
    """Focus-aware binary triage: would this item warrant commentary?

    Triage runs before any thread exists, so usage is booked against thread 0
    with kind "triage" when a TokenLog is supplied.
    """
    try:
        reply = await model.ainvoke(
            [
                {"role": "system", "content": filter_system_prompt(kind, focus)},
                {"role": "user", "content": text},
            ]
        )
    except Exception:
        log.exception("filter model failed; skipping item")
        return False
    if tokens is not None and (usage := getattr(reply, "usage_metadata", None)):
        from airc_core import usage_counts

        tokens.add(0, "triage", "triage", *usage_counts(usage), model_id)
    return "INTERESTING" in str(reply.text).upper()

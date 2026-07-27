# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Agent-set timers: "wake me in X minutes" for real-world events.

An agent (typically in a DM) can schedule its own follow-up -- "check the
pinpoint job in 60 min and summarize" -- and end its turn. A single waiter fires
the wake at time T by driving a fresh turn for the SAME agent on the SAME thread,
so it resumes with its own context (its checkpoint plus a catch-up of anything
posted meanwhile) rather than a cold prompt.

Design notes:
- ONE waiter for any number of timers: a min-heap keyed by fire time plus a poke
  Event, so a 2-minute timer correctly preempts a 60-minute one. Never a task
  per timer.
- The timer tools (timer_create/timer_list/timer_cancel) are plain langchain
  tools (no MCP): each reads its thread and agent from the injected
  RunnableConfig, so they need no ambient wiring.
- Each pending timer has a stable integer id (the monotonic add sequence, never
  reused), returned by timer_create and used by timer_cancel. list/cancel are
  scoped to the calling thread, so an agent only ever sees and cancels its own
  chat's timers.
- Cancellation is lazy tombstoning: a live-timer dict is the source of truth, and
  a heap entry whose id is no longer live is skipped when it surfaces. This keeps
  the single-heap/single-waiter shape (a heap cannot cheaply remove an interior
  entry); the per-thread pending count is decremented exactly once, at whichever
  of cancel-or-fire happens first.
- Durable across restart when a store is wired: add/cancel/fire mirror to a
  `timers` table, and `restore()` repopulates the heap at startup. A timer whose
  fire time already passed while the daemon was down surfaces with a non-positive
  delay and so fires once, immediately, on the next run() tick. Without a store
  the scheduler is purely in-memory (the shape tests and a bare room use).
- Firing goes through the orchestrator (deliver), which runs the turn under the
  per-(thread, agent) lock -- never a raw run_turn, which would race a concurrent
  human-triggered turn for the same agent.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

if TYPE_CHECKING:
    from .store import Store

log = logging.getLogger(__name__)

# Guardrails: a timer is a scheduled unit of paid work, so bound both how far out
# it can reach and how many can pile up on one chat.
_MAX_HORIZON_S = 24 * 3600
_MAX_PENDING_PER_THREAD = 5

# deliver(thread_id, agent, note) -> runs the wake turn
Deliver = Callable[[int, str, str], Awaitable[None]]


@dataclass(order=True)
class _Wake:
    fire_at: float
    seq: int
    thread_id: int
    agent: str
    note: str


class TimerScheduler:
    """One waiter over a heap of pending wakes. `deliver` is set by the daemon
    before `run()` starts (it points at the orchestrator's locked turn path).

    When a `store` is passed, pending timers are mirrored to it and survive a
    restart (call `restore()` once before `run()`); without one the scheduler is
    in-memory only."""

    def __init__(self, store: "Store | None" = None) -> None:
        self._store = store
        self._heap: list[_Wake] = []
        self._seq = 0
        self._poke = asyncio.Event()
        self._pending: dict[int, int] = {}  # thread_id -> count, for the cap
        # id -> the still-pending wake. The heap may hold stale entries a cancel
        # left behind (a heap has no cheap interior delete), so this dict, not the
        # heap, is the truth about what will still fire.
        self._live: dict[int, _Wake] = {}
        self._tasks: set[asyncio.Task] = set()  # in-flight deliveries
        self.deliver: Deliver | None = None

    def restore(self) -> None:
        """Repopulate the heap from the store's persisted timers, and seed the id
        counter past the highest restored id so a new timer cannot reuse one. A
        timer already due fires on the first run() tick (its delay is <= 0); the
        24h horizon bounds how stale a restored timer can be. Idempotent-safe to
        call once at startup; a no-op without a store or with none persisted."""
        if self._store is None:
            return
        for seq, thread_id, agent, fire_at, note in self._store.all_timers():
            wake = _Wake(fire_at, seq, thread_id, agent, note)
            heapq.heappush(self._heap, wake)
            self._live[seq] = wake
            self._pending[thread_id] = self._pending.get(thread_id, 0) + 1
            self._seq = max(self._seq, seq + 1)
        if self._live:
            self._poke.set()

    def add(self, thread_id: int, agent: str, fire_at: float, note: str) -> int | None:
        """Enqueue a wake. Returns its stable id, or None if the per-thread cap is
        already hit."""
        if self._pending.get(thread_id, 0) >= _MAX_PENDING_PER_THREAD:
            return None
        wake = _Wake(fire_at, self._seq, thread_id, agent, note)
        self._seq += 1
        heapq.heappush(self._heap, wake)
        self._live[wake.seq] = wake
        self._pending[thread_id] = self._pending.get(thread_id, 0) + 1
        if self._store is not None:
            self._store.add_timer(wake.seq, thread_id, agent, fire_at, note)
        self._poke.set()  # re-arm the waiter in case this jumps the queue
        return wake.seq

    def list_for(self, thread_id: int) -> list[tuple[int, float, str]]:
        """The (id, fire_at, note) of this thread's still-pending timers, soonest
        first. Scoped to one thread: an agent lists only its own chat's timers."""
        wakes = [w for w in self._live.values() if w.thread_id == thread_id]
        wakes.sort(key=lambda w: w.fire_at)
        return [(w.seq, w.fire_at, w.note) for w in wakes]

    def cancel(self, thread_id: int, timer_id: int) -> bool:
        """Cancel a pending timer by id, scoped to the thread that owns it (so a
        chat can only cancel its own). Tombstones the heap entry -- run() skips it
        when it surfaces -- and frees the thread's cap slot now. Returns False if
        no matching pending timer exists (already fired, already cancelled, or
        another thread's id)."""
        wake = self._live.get(timer_id)
        if wake is None or wake.thread_id != thread_id:
            return False
        del self._live[timer_id]
        self._release_slot(thread_id)
        if self._store is not None:
            self._store.remove_timer(timer_id)
        return True

    def _release_slot(self, thread_id: int) -> None:
        """Free one of the thread's pending-cap slots (on cancel or on fire)."""
        remaining = self._pending.get(thread_id, 0) - 1
        if remaining > 0:
            self._pending[thread_id] = remaining
        else:
            self._pending.pop(thread_id, None)

    async def run(self) -> None:
        try:
            while True:
                if not self._heap:
                    await self._poke.wait()
                    self._poke.clear()
                    continue
                delay = self._heap[0].fire_at - time.time()
                if delay > 0:
                    try:
                        await asyncio.wait_for(self._poke.wait(), timeout=delay)
                        # A new (maybe earlier) timer arrived: recompute the head.
                        self._poke.clear()
                        continue
                    except TimeoutError:
                        pass  # the head is due
                wake = heapq.heappop(self._heap)
                # A cancelled timer leaves a tombstone in the heap (no cheap
                # interior delete); its cap slot was already freed at cancel, so
                # just drop it here.
                if self._live.get(wake.seq) is not wake:
                    continue
                del self._live[wake.seq]
                self._release_slot(wake.thread_id)
                # Consume the persisted row at fire time (one-shot), matching the
                # cap-slot release: a crash mid-delivery drops the timer rather
                # than re-firing a "check the job" note twice on the next restart.
                if self._store is not None:
                    self._store.remove_timer(wake.seq)
                self._fire(wake)
        finally:
            # A wake mid-delivery at shutdown: cancel and drain so it cannot be
            # unwinding into a store/checkpointer the daemon is about to close.
            for t in list(self._tasks):
                t.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)

    def _fire(self, wake: _Wake) -> None:
        if self.deliver is None:  # not wired (or shutting down)
            return
        t = asyncio.create_task(
            self._deliver_guarded(wake), name=f"timer-wake:{wake.thread_id}"
        )
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def _deliver_guarded(self, wake: _Wake) -> None:
        assert self.deliver is not None
        try:
            await self.deliver(wake.thread_id, wake.agent, wake.note)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "timer: wake delivery failed (thread %d, agent %s)",
                wake.thread_id,
                wake.agent,
            )


def _ctx_from_config(config: RunnableConfig | None) -> tuple[int | None, str]:
    """The (thread_id, agent) a turn runs under, from the config the runner sets
    as configurable.thread_id = "<thread_id>:<agent>"."""
    try:
        raw = (config or {}).get("configurable", {}).get("thread_id", "")
        tid, _, agent = str(raw).partition(":")
        return int(tid), agent
    except (ValueError, AttributeError):
        return None, ""


def _fmt_eta(fire_at: float) -> str:
    """A short human ETA for a fire time ("in ~12 min", "in ~2.0 h", "now")."""
    secs = fire_at - time.time()
    if secs <= 0:
        return "now"
    mins = secs / 60.0
    return f"in ~{mins:.0f} min" if mins < 90 else f"in ~{mins / 60:.1f} h"


def make_timer_tools(scheduler: TimerScheduler) -> list:
    """The three local (non-MCP) timer tools bound to a scheduler, returned as a
    list the runner appends to every chat persona's toolset. Each reads its
    (thread, agent) from the injected RunnableConfig, so list/cancel are scoped
    to the calling chat -- an agent can only see and cancel its own thread's
    timers. Ids are the scheduler's stable per-timer sequence."""

    @tool
    async def timer_create(minutes: float, note: str, config: RunnableConfig) -> str:
        """Schedule a single follow-up turn in this chat. After `minutes` you get
        a fresh turn here with `note` as the instruction, keeping this chat's
        context. Returns the timer's id, which timer_cancel takes.

        Use this RARELY, only when a later check genuinely adds value: to return
        to a real-world event that is not ready yet and that you actually need to
        act on -- a running pinpoint or CQ job, a long build, a CL you are waiting
        to land, a dish that needs turning. Do NOT use it for routine reminders,
        to keep a conversation going, or when the person could just ask again; if
        a check later would not add real value, do not schedule one. Prefer
        answering now over deferring. Pick a realistic delay. Example: minutes=60,
        note="check pinpoint job 1234 and summarize the result"."""
        thread_id, agent = _ctx_from_config(config)
        if thread_id is None:
            return "could not schedule: missing turn context."
        secs = float(minutes) * 60.0
        if secs <= 0:
            return "declined: minutes must be positive."
        if secs > _MAX_HORIZON_S:
            return f"declined: the maximum is {_MAX_HORIZON_S // 3600}h ahead."
        note = note.strip()
        if not note:
            return "declined: include a note describing what to do on wake."
        timer_id = scheduler.add(thread_id, agent, time.time() + secs, note)
        if timer_id is None:
            return (
                "declined: this chat already has the maximum"
                f" {_MAX_PENDING_PER_THREAD} pending timers."
            )
        return f"scheduled timer {timer_id}: i'll follow up in about {minutes:g} min."

    @tool
    async def timer_list(config: RunnableConfig) -> str:
        """List this chat's pending timers (id, when it fires, and its note), so
        you can tell the person what is scheduled or find an id to cancel. Only
        this chat's timers are visible."""
        thread_id, _ = _ctx_from_config(config)
        if thread_id is None:
            return "could not list: missing turn context."
        pending = scheduler.list_for(thread_id)
        if not pending:
            return "no pending timers in this chat."
        return "\n".join(
            f"{tid}: {_fmt_eta(fire_at)} -- {note}" for tid, fire_at, note in pending
        )

    @tool
    async def timer_cancel(timer_id: int, config: RunnableConfig) -> str:
        """Cancel a pending timer in this chat by its id (from timer_create or
        timer_list). Only this chat's timers can be cancelled."""
        thread_id, _ = _ctx_from_config(config)
        if thread_id is None:
            return "could not cancel: missing turn context."
        if scheduler.cancel(thread_id, int(timer_id)):
            return f"cancelled timer {timer_id}."
        return (
            f"no pending timer {timer_id} in this chat (already fired or cancelled?)."
        )

    return [timer_create, timer_list, timer_cancel]

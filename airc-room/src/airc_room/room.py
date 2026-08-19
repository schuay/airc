# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""The room: shared message bus connecting transports, orchestrator, watchers.

Every message — human input from a transport, agent replies, watcher
announcements — flows through Room.post(). The room persists it, fans it out
to all transports, and enqueues it for the orchestrator. The orchestrator is
the only consumer of the queue; transports are display-only sinks.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from .store import Message, MessageKind, Store, Thread

log = logging.getLogger(__name__)

# Per-transport deadline on a single deliver/typing call. These run while the
# orchestrator holds a turn-semaphore slot (and the "thinking..." card), so a
# transport hanging on network I/O here would otherwise pin that slot forever.
_TRANSPORT_TIMEOUT = 60.0


class UndeliveredError(RuntimeError):
    """Every transport failed to deliver a message the caller must not lose.

    Raised only by `post(require_delivery=True)`. The default stays
    fire-and-forget, because for a conversational turn the store IS the record
    and a failed frontend is the frontend's problem. A subscriber draining a
    durable queue is the opposite case: it acks the queue message on return, so
    a swallowed transport failure loses the item with no retry and no trace.
    """


class Transport(Protocol):
    """A chat frontend (console, Matrix, Google Chat, ...).

    Two required methods -- an inbound loop (`run`) that feeds human input into
    the room via `room.post`, and an outbound sink (`deliver`) the room calls for
    every message. Optional methods are duck-typed, not part of the protocol, so a
    transport implements only what it renders: `typing(thread_id, sender, active,
    budget)` for a "composing..." indicator, `on_event(agent, event, detail)` for
    tool-call traces, `aclose()` for shutdown cleanup.

    Thread routing is generic: `msg.thread_id` is the room's thread reference, and
    each transport maps it to its own native thread concept -- the console renders
    it flat, the Google Chat transport maps it to a space + server thread (via the
    store), and the Matrix transport maps it to an `m.thread` relation. Nothing in
    this interface is shaped to one frontend's thread model, so a new transport
    slots in by owning that one mapping.

    Message rendering -- and, for a future end-to-end-encrypted transport, payload
    encryption -- happens inside the transport's own send path (its `deliver` and
    whatever private post helper it uses), never in the room. That keeps E2E an
    implementation detail a transport can add without reshaping this protocol: the
    room hands over a plaintext `Message` and the transport decides how it reaches
    the wire.
    """

    name: str

    async def run(self) -> None:
        """Inbound loop: receive frontend input and post it into the room. Runs
        for the transport's lifetime (a REPL, a Pub/Sub pull, a Matrix sync
        loop); the interactive console owns the foreground, others run as
        background tasks."""
        ...

    async def deliver(self, msg: Message) -> None:
        """Display a message. Called for every message in the room, including
        ones this transport posted itself (filter on msg origin if needed).
        `msg.thread_id` is the generic thread reference this transport maps to its
        native thread concept."""
        ...


class Room:
    def __init__(self, store: Store) -> None:
        self._store = store
        self._transports: list[Transport] = []
        self.inbox: asyncio.Queue[Message] = asyncio.Queue()

    @property
    def store(self) -> Store:
        """The store behind this room, for a holder that needs more of it than
        Room re-exports.

        Room's own surface is deliberately the small set of operations the room
        loop performs, and growing it one delegating method at a time for each
        new caller is how a facade turns into a second copy of the store's API.
        A local tool built with `room` (build_local_tools) is the case that
        forced this: it posts through Room and reads its own plugin_state rows,
        and those two have to be the SAME connection or "is this proposal in the
        thread this message is in" becomes a cross-store join.
        """
        return self._store

    def add_transport(self, transport: Transport) -> None:
        self._transports.append(transport)

    # ── threads ──────────────────────────────────────────────────────────────

    def create_thread(self, title: str) -> Thread:
        return self._store.create_thread(title)

    def get_thread(self, thread_id: int) -> Thread | None:
        return self._store.get_thread(thread_id)

    def thread_for_commit(self, hash: str, title: str) -> tuple[Thread, bool]:
        """Get-or-create the single thread for a commit hash, returning
        (thread, created). Commentary and findings for one commit both route
        through this, so they converge on one thread whichever arrives first;
        `created` lets commentary post its announcement only on first sight (the
        dedup). A stale mapping (thread deleted) is recreated."""
        return self.thread_for_key(hash, title)

    def thread_for_key(self, key: str, title: str) -> tuple[Thread, bool]:
        """thread_for_commit generalized to any stable key: the store's mapping
        is just key -> thread, and non-commit homes (the perf backfill digest)
        need the same get-or-create so retries and later outages converge on one
        thread. Non-hash keys must not look like a revision, or they could
        collide with a real commit's mapping."""
        tid = self._store.commit_thread(key)
        if tid is not None and (t := self._store.get_thread(tid)) is not None:
            return t, False
        t = self._store.create_thread(title)
        self._store.set_commit_thread(key, t.id)
        return t, True

    def list_threads(self) -> list[Thread]:
        return self._store.list_threads()

    def thread_messages(self, thread_id: int) -> list[Message]:
        return self._store.thread_messages(thread_id)

    def record_announcement_meta(self, thread_id: int, meta: dict) -> None:
        """Persist a commit announcement's source identity so a later handover
        (which only has the thread, not the Announcement) can recover it."""
        self._store.set_announcement_meta(
            thread_id,
            meta.get("repo_name", ""),
            meta.get("repo_path", ""),
            meta.get("hash", ""),
        )

    def default_thread(self) -> Thread:
        threads = self._store.list_threads()
        return threads[0] if threads else self._store.create_thread("main")

    # ── messages ─────────────────────────────────────────────────────────────

    async def post(
        self,
        thread_id: int,
        sender: str,
        kind: MessageKind,
        text: str,
        follow_up: str = "",
        require_delivery: bool = False,
        sender_id: str = "",
    ) -> Message:
        """Persist a message, enqueue for orchestration, deliver to transports.

        Persist and enqueue happen back to back with no await, so the inbox
        receives a thread's messages in id order even when posters run
        concurrently -- the orchestrator's per-thread watermark depends on
        this. Delivery (which can block on network) comes after.

        follow_up names an app-registered response handler for a SYSTEM
        announcement (default "" = the room's plain forced-commentator turn); a
        subscriber sets it so the orchestrator can dispatch without knowing the
        domain.

        Delivery is best-effort by default: a transport that raises is logged
        and the post still succeeds, because the store is the record and one
        broken frontend must not fail a turn. `require_delivery` inverts that
        for a caller whose own retry depends on the answer -- a subscriber
        draining a durable queue acks on return, so for it a swallowed failure
        is a silently dropped item. It raises UndeliveredError only when EVERY
        transport failed; a partial failure is still a delivered message, and
        re-posting it would duplicate for the transports that worked.
        """
        msg = self._store.add_message(
            thread_id, sender, kind, text, follow_up, sender_id=sender_id
        )
        self.inbox.put_nowait(msg)
        delivered = 0
        for t in self._transports:
            try:
                async with asyncio.timeout(_TRANSPORT_TIMEOUT):
                    await t.deliver(msg)
                delivered += 1
            except Exception:
                log.exception("transport %s failed to deliver message", t.name)
        if require_delivery and self._transports and not delivered:
            raise UndeliveredError(
                f"no transport delivered message {msg.id} to thread {thread_id}"
            )
        return msg

    async def typing(
        self, thread_id: int, sender: str, active: bool, budget: float | None = None
    ) -> None:
        """Signal that an agent is (no longer) composing a reply. Transports may
        render it (e.g. a Chat "thinking..." placeholder); those without a
        ``typing`` method ignore it. ``budget`` is the turn's time allowance in
        seconds, letting a transport show a countdown."""
        for t in self._transports:
            fn = getattr(t, "typing", None)
            if fn is None:
                continue
            try:
                async with asyncio.timeout(_TRANSPORT_TIMEOUT):
                    await fn(thread_id, sender, active, budget)
            except Exception:
                log.exception("transport %s typing failed", t.name)

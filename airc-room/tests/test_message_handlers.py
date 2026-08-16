# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Plugin message handlers: the room's push seam for things a persona should not
have to be woken up to answer.

What is guarded here is the seam's properties rather than any one handler:
consuming stops orchestration and nothing else, a broken handler cannot silence
the room, and a replayed message reaches the chain -- which is the whole reason
the hook sits in the worker loop instead of in room.post.
"""

import asyncio
from contextlib import asynccontextmanager

import pytest
from airc_room.orchestrator import Disposition, MessageHandler, Orchestrator
from airc_room.store import MessageKind
from test_concurrency import drive, make_env, replies


class Recorder:
    """A handler that records what it saw and answers with a fixed verdict."""

    def __init__(self, name="recorder", verdict=Disposition.PASS):
        self.name = name
        self._verdict = verdict
        self.seen = []  # message ids, in arrival order

    async def handle(self, msg):
        self.seen.append(msg.id)
        return self._verdict


class Exploding:
    name = "exploding"

    def __init__(self):
        self.calls = 0

    async def handle(self, msg):
        self.calls += 1
        raise RuntimeError("this handler is broken")


def _with_handlers(tmp_path, monkeypatch, handlers, agents=("perf",)):
    """make_env, but with the handler chain wired into a fresh orchestrator."""
    store, room, runner, orch = make_env(tmp_path, monkeypatch, agents=agents)
    orch = Orchestrator(orch._cfg, room, runner, store, message_handlers=list(handlers))
    return store, room, runner, orch


@asynccontextmanager
async def running(orch):
    """The orchestrator running, with recovery already done.

    Posting before run() would put a message in the inbox AND leave it above the
    watermark, so it arrives twice -- an artefact of the test, not the room: cli
    creates the orchestrator task before any transport or watcher, precisely so
    recovery completes before anything can deliver. Tests that want the replay
    path use add_message instead, which never touches the inbox.
    """
    task = asyncio.create_task(orch.run())
    await asyncio.sleep(0)  # _recover is synchronous; one tick reaches it
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def until(pred, timeout=5.0):
    async with asyncio.timeout(timeout):
        while not pred():
            await asyncio.sleep(0.01)


async def test_a_consumed_message_wakes_nobody(tmp_path, monkeypatch):
    # The point of the seam: a mechanical act (a command) is answered without
    # spending a persona turn on it, even when it addresses one by name.
    h = Recorder(verdict=Disposition.CONSUMED)
    store, room, runner, orch = _with_handlers(tmp_path, monkeypatch, [h])
    t = room.create_thread("main")

    async with running(orch):
        m = await room.post(t.id, "alice", MessageKind.HUMAN, "perf: !submit abc")
        await until(lambda: h.seen == [m.id])

    assert runner.calls == []  # no turn, despite the address
    assert replies(store, t.id) == []


async def test_a_passed_message_routes_as_it_always_did(tmp_path, monkeypatch):
    h = Recorder(verdict=Disposition.PASS)
    store, room, runner, orch = _with_handlers(tmp_path, monkeypatch, [h])
    t = room.create_thread("main")

    async with running(orch):
        m = await room.post(t.id, "alice", MessageKind.HUMAN, "perf: what changed?")
        await until(lambda: len(replies(store, t.id)) == 1)

    # Not h.seen == [m.id]: the reply is a message too and reaches the chain on
    # the same worker (the idempotency contract _consumed states), so whether it
    # appears here is a race between routing it and the cancel above exiting the
    # block -- under a loaded machine the worker wins, and the assertion must
    # not bid on scheduling. Invariant either way: the chain saw m exactly once,
    # saw it first (per-thread id order), and the turn only ran because the
    # chain let it (PASS precedes routing).
    assert h.seen[0] == m.id
    assert h.seen.count(m.id) == 1
    assert runner.calls == [("perf", t.id)]


async def test_the_first_consumer_wins_and_the_rest_never_run(tmp_path, monkeypatch):
    first = Recorder(name="first", verdict=Disposition.PASS)
    second = Recorder(name="second", verdict=Disposition.CONSUMED)
    third = Recorder(name="third", verdict=Disposition.CONSUMED)
    _store, room, runner, orch = _with_handlers(
        tmp_path, monkeypatch, [first, second, third]
    )
    t = room.create_thread("main")

    async with running(orch):
        m = await room.post(t.id, "alice", MessageKind.HUMAN, "perf: go")
        await until(lambda: second.seen == [m.id])

    assert first.seen == [m.id]  # registration order, up to the consumer
    assert third.seen == []
    assert runner.calls == []


async def test_a_broken_handler_cannot_silence_the_room(tmp_path, monkeypatch):
    # A handler that raises is logged and read as PASS. The failure mode this
    # guards against is one plugin exception swallowing a message nobody answers.
    boom = Exploding()
    after = Recorder(name="after", verdict=Disposition.PASS)
    store, room, runner, orch = _with_handlers(tmp_path, monkeypatch, [boom, after])
    t = room.create_thread("main")

    async with running(orch):
        m = await room.post(t.id, "alice", MessageKind.HUMAN, "perf: go")
        await until(lambda: len(replies(store, t.id)) == 1)

    # boom may also run on the reply (same worker-loop placement as above), so
    # its call count is not this test's subject; that m reached it and the
    # chain continued past the raise is.
    assert boom.calls >= 1
    assert after.seen[0] == m.id  # the chain continued past the raise
    assert after.seen.count(m.id) == 1
    assert runner.calls == [("perf", t.id)]


async def test_consuming_leaves_the_message_in_the_store_untouched(
    tmp_path, monkeypatch
):
    # CONSUMED suppresses ORCHESTRATION, not the message. It is already persisted
    # and already delivered by the time a handler runs, and its kind is left
    # alone, so the thread's history stays honest about what was said.
    h = Recorder(verdict=Disposition.CONSUMED)
    store, room, _runner, orch = _with_handlers(tmp_path, monkeypatch, [h])
    t = room.create_thread("main")

    async with running(orch):
        m = await room.post(t.id, "alice", MessageKind.HUMAN, "perf: !submit abc")
        await until(lambda: h.seen == [m.id])

    stored = store.thread_messages(t.id)[-1]
    assert stored.id == m.id
    assert stored.kind is MessageKind.HUMAN
    assert stored.text == "perf: !submit abc"


async def test_a_consumed_message_still_advances_the_watermark(tmp_path, monkeypatch):
    # Consumption is completion: a consumed message must commit, or every restart
    # replays it forever and the handler answers it again each time.
    h = Recorder(verdict=Disposition.CONSUMED)
    store, room, _runner, orch = _with_handlers(tmp_path, monkeypatch, [h])
    t = room.create_thread("main")

    async with running(orch):
        m = await room.post(t.id, "alice", MessageKind.HUMAN, "perf: !submit abc")
        await until(lambda: (store.get_orchestrated(t.id) or 0) >= m.id)

    assert store.get_orchestrated(t.id) == m.id


async def test_a_replayed_message_reaches_the_chain(tmp_path, monkeypatch):
    # The reason the hook is here and not in room.post: recovery replays
    # persisted messages above the watermark straight into the workers, bypassing
    # post entirely. A hook there would silently skip exactly the messages that
    # most need re-processing -- and consumption would not be crash-durable.
    h = Recorder(verdict=Disposition.CONSUMED)
    store, room, runner, orch = _with_handlers(tmp_path, monkeypatch, [h])
    t = room.create_thread("main")
    # Persisted but never orchestrated, as if the process died right after
    # add_message. The inbox never sees it; only _recover can deliver it.
    m = store.add_message(t.id, "alice", MessageKind.HUMAN, "perf: !submit abc")

    await drive(orch, lambda: h.seen == [m.id])

    assert h.seen == [m.id]
    assert runner.calls == []


@pytest.mark.parametrize("kind", [MessageKind.NOTICE, MessageKind.PING])
async def test_operational_kinds_never_reach_the_chain(tmp_path, monkeypatch, kind):
    # These are never routed to a persona either. A handler seeing them would be
    # observing the room's own bookkeeping -- including the notices a handler
    # itself posts, which is a loop waiting to happen.
    h = Recorder(verdict=Disposition.CONSUMED)
    _store, room, _runner, orch = _with_handlers(tmp_path, monkeypatch, [h])
    t = room.create_thread("main")

    async with running(orch):
        await room.post(t.id, "perf", kind, "(perf is thinking)")
        m = await room.post(t.id, "alice", MessageKind.HUMAN, "perf: go")
        await until(lambda: h.seen == [m.id])

    assert h.seen == [m.id]


async def test_handlers_see_a_thread_in_order(tmp_path, monkeypatch):
    # Handlers inherit the worker's per-thread ordering, which is what lets one
    # reason about "the newest open packet" without re-reading the thread.
    h = Recorder(verdict=Disposition.CONSUMED)
    _store, room, _runner, orch = _with_handlers(tmp_path, monkeypatch, [h])
    t = room.create_thread("main")

    async with running(orch):
        ids = [
            (await room.post(t.id, "alice", MessageKind.HUMAN, f"!cmd {i}")).id
            for i in range(5)
        ]
        await until(lambda: len(h.seen) == 5)

    assert h.seen == ids


async def test_a_slow_handler_delays_only_its_own_thread(tmp_path, monkeypatch):
    # Handlers run inline in the per-thread worker, so a slow one is a per-thread
    # cost and never a room-wide one. The contract is that they stay fast; this
    # is what bounds the damage when one does not.
    class Slow:
        name = "slow"

        def __init__(self):
            self.done = asyncio.Event()

        async def handle(self, msg):
            if msg.thread_id == slow_thread:
                await asyncio.sleep(0.3)
                self.done.set()
            return Disposition.CONSUMED

    h = Slow()
    store, room, _runner, orch = _with_handlers(tmp_path, monkeypatch, [h])
    slow = room.create_thread("slow")
    fast = room.create_thread("fast")
    slow_thread = slow.id

    async with running(orch):
        await room.post(slow.id, "alice", MessageKind.HUMAN, "!cmd")
        m = await room.post(fast.id, "alice", MessageKind.HUMAN, "!cmd")
        # The fast thread commits while the slow handler is still sleeping.
        await until(lambda: (store.get_orchestrated(fast.id) or 0) >= m.id)
        assert not h.done.is_set()


async def test_a_bare_room_registers_no_handlers(tmp_path, monkeypatch):
    # The default: absent hook means absent behavior, like every other optional
    # part of the plugin contract.
    store, room, runner, orch = make_env(tmp_path, monkeypatch, agents=("perf",))
    t = room.create_thread("main")

    async with running(orch):
        await room.post(t.id, "alice", MessageKind.HUMAN, "perf: go")
        await until(lambda: len(replies(store, t.id)) == 1)

    assert orch._message_handlers == []
    assert runner.calls == [("perf", t.id)]


def test_the_protocol_matches_a_duck_typed_handler():
    # runtime_checkable, so a plugin can assert its own handler conforms without
    # importing a base class to inherit from.
    assert isinstance(Recorder(), MessageHandler)
    assert not isinstance(object(), MessageHandler)

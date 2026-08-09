# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

import pytest
from airc_room.room import Room, UndeliveredError
from airc_room.store import MessageKind, Store


class _Store:
    def add_message(self, *a):
        raise AssertionError("not used")


def _room():
    return Room(_Store())


class _Typing:
    name = "typing-capable"

    def __init__(self):
        self.calls = []

    async def deliver(self, msg): ...

    async def typing(self, thread_id, sender, active, budget=None):
        self.calls.append((thread_id, sender, active, budget))


class _Plain:
    name = "no-typing"

    async def deliver(self, msg): ...


class _Boom:
    name = "boom"

    async def deliver(self, msg): ...

    async def typing(self, thread_id, sender, active, budget=None):
        raise RuntimeError("transport blew up")


async def test_typing_fans_out_and_skips_unsupported():
    room = _room()
    cap, plain = _Typing(), _Plain()
    room.add_transport(cap)
    room.add_transport(plain)  # no typing method -> skipped, no error
    await room.typing(3, "perf", True, budget=900.0)
    await room.typing(3, "perf", False)
    assert cap.calls == [(3, "perf", True, 900.0), (3, "perf", False, None)]


async def test_typing_isolates_transport_failures():
    room = _room()
    cap = _Typing()
    room.add_transport(_Boom())
    room.add_transport(cap)
    await room.typing(1, "gc", True)  # _Boom raises; must not stop cap
    assert cap.calls == [(1, "gc", True, None)]


class _DeadTransport:
    """Every deliver fails -- an outage, a 503, an expired credential."""

    name = "dead"

    async def deliver(self, msg):
        raise RuntimeError("503 from the chat backend")


class _Recorder:
    name = "recorder"

    def __init__(self):
        self.seen = []

    async def deliver(self, msg):
        self.seen.append(msg)


def _real_room(tmp_path):
    return Room(Store(tmp_path / "airc.db"))


async def test_post_is_best_effort_by_default(tmp_path):
    # The conversational contract: the store is the record, so a broken frontend
    # must not fail a turn.
    room = _real_room(tmp_path)
    room.add_transport(_DeadTransport())
    t = room.create_thread("t")
    msg = await room.post(t.id, "u", MessageKind.NOTICE, "hello")
    assert [m.text for m in room.thread_messages(t.id)] == ["hello"]
    assert msg.text == "hello"


async def test_require_delivery_raises_when_no_transport_took_it(tmp_path):
    # The queue-drain contract: a subscriber acks its bus message when post
    # returns, so a swallowed transport failure drops the item with no retry and
    # no failed/ record. It has to hear about it.
    room = _real_room(tmp_path)
    room.add_transport(_DeadTransport())
    t = room.create_thread("t")
    with pytest.raises(UndeliveredError):
        await room.post(t.id, "u", MessageKind.NOTICE, "hi", require_delivery=True)
    # Still persisted: the raise is about delivery, and the caller's retry finds
    # the history intact.
    assert [m.text for m in room.thread_messages(t.id)] == ["hi"]


async def test_require_delivery_accepts_a_partial_delivery(tmp_path):
    # One transport working means the message reached a reader; re-posting would
    # duplicate it for them. Only a total failure is a lost message.
    room = _real_room(tmp_path)
    rec = _Recorder()
    room.add_transport(_DeadTransport())
    room.add_transport(rec)
    t = room.create_thread("t")
    await room.post(t.id, "u", MessageKind.NOTICE, "hi", require_delivery=True)
    assert [m.text for m in rec.seen] == ["hi"]


async def test_require_delivery_is_satisfied_with_no_transports(tmp_path):
    # Headless (tests, a room with no frontend attached): there is nothing to
    # fail, so demanding delivery must not turn every post into an error.
    room = _real_room(tmp_path)
    t = room.create_thread("t")
    await room.post(t.id, "u", MessageKind.NOTICE, "hi", require_delivery=True)
    assert len(room.thread_messages(t.id)) == 1


def test_the_room_exposes_the_store_it_was_built_on(tmp_path):
    """A local tool holding a Room must reach the same connection the room posts
    through -- otherwise its own state rows and the messages it compares them
    against are two stores with independent orderings."""
    store = Store(tmp_path / "airc.db")
    room = Room(store)
    assert room.store is store

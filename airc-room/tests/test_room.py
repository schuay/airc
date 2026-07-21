from airc_room.room import Room


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

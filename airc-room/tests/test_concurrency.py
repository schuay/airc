# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Concurrent orchestrator: parallel rounds, watermarks, recovery, ordering."""

import asyncio

from airc_room import orchestrator as orch_mod
from airc_room.config import Config, OrchestratorConfig
from airc_room.orchestrator import Orchestrator, _PendingMsg
from airc_room.room import Room
from airc_room.store import Store


class FakeRunner:
    """Turns sleep for `delay` and record overlap, keyed like the real runner."""

    def __init__(self, names, delay=0.05):
        self._names = list(names)
        self.delay = delay
        self.calls = []  # (agent, thread_id) in completion order
        self.addressed = []  # (agent, thread_id, addressed) in completion order
        self.task_prompts = []  # (agent, thread_id, task_prompt) in completion order
        self.active = 0
        self.max_active = 0

    @property
    def agents(self):
        from types import SimpleNamespace

        return {n: SimpleNamespace(description=f"{n} expert") for n in self._names}

    async def run_turn(self, name, thread_id, *, addressed=False, task_prompt=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.active -= 1
        self.calls.append((name, thread_id))
        self.addressed.append((name, thread_id, addressed))
        self.task_prompts.append((name, thread_id, task_prompt))
        return f"reply from {name}"


def make_env(tmp_path, monkeypatch, agents=("perf", "compiler"), cfg=None, delay=0.05):
    monkeypatch.setattr(orch_mod, "make_model", lambda mid: object())
    store = Store(tmp_path / "t.db")
    room = Room(store)
    runner = FakeRunner(agents, delay=delay)
    orch = Orchestrator(cfg or Config(), room, runner, store)
    return store, room, runner, orch


async def drive(orch, until, timeout=5.0):
    """Run the orchestrator until `until()` is true, then cancel it."""
    task = asyncio.create_task(orch.run())
    try:
        async with asyncio.timeout(timeout):
            while not until():
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def replies(store, thread_id):
    return [m for m in store.thread_messages(thread_id) if m.kind == "agent"]


async def test_responders_of_one_message_overlap(tmp_path, monkeypatch):
    store, room, runner, orch = make_env(tmp_path, monkeypatch)
    t = room.create_thread("main")
    await room.post(t.id, "alice", "human", "perf, compiler: go")
    await drive(orch, lambda: len(replies(store, t.id)) == 2)
    assert runner.max_active == 2


async def test_same_agent_overlaps_across_threads(tmp_path, monkeypatch):
    store, room, runner, orch = make_env(tmp_path, monkeypatch, agents=("perf",))
    t1, t2 = room.create_thread("a"), room.create_thread("b")
    await room.post(t1.id, "alice", "human", "perf: go")
    await room.post(t2.id, "bob", "human", "perf: go")
    await drive(
        orch,
        lambda: len(replies(store, t1.id)) == 1 and len(replies(store, t2.id)) == 1,
    )
    assert runner.max_active == 2


async def test_same_agent_same_thread_is_serialized(tmp_path, monkeypatch):
    store, room, runner, orch = make_env(tmp_path, monkeypatch, agents=("perf",))
    t = room.create_thread("main")
    await room.post(t.id, "alice", "human", "perf: first")
    await room.post(t.id, "alice", "human", "perf: second")
    await drive(orch, lambda: len(replies(store, t.id)) == 2)
    assert runner.max_active == 1


async def test_turn_semaphore_bounds_parallelism(tmp_path, monkeypatch):
    cfg = Config()
    cfg.orchestrator = OrchestratorConfig(max_concurrent_turns=1)
    store, room, runner, orch = make_env(
        tmp_path, monkeypatch, agents=("perf",), cfg=cfg
    )
    t1, t2 = room.create_thread("a"), room.create_thread("b")
    await room.post(t1.id, "alice", "human", "perf: go")
    await room.post(t2.id, "bob", "human", "perf: go")
    await drive(
        orch,
        lambda: len(replies(store, t1.id)) == 1 and len(replies(store, t2.id)) == 1,
    )
    assert runner.max_active == 1


async def test_watermark_advances_through_replies(tmp_path, monkeypatch):
    store, room, _runner, orch = make_env(tmp_path, monkeypatch, agents=("perf",))
    t = room.create_thread("main")
    await room.post(t.id, "alice", "human", "perf: go")
    await drive(orch, lambda: len(replies(store, t.id)) == 1)
    # Let the reply's own (responderless) round complete too. The first drive
    # may have cancelled mid-round, in which case the second run replays the
    # trigger (at-least-once) and posts a duplicate -- so the watermark must
    # reach at least `last`, not exactly it.
    last = store.thread_messages(t.id)[-1].id
    await drive(orch, lambda: (store.get_orchestrated(t.id) or 0) >= last)


async def test_recovery_replays_unorchestrated(tmp_path, monkeypatch):
    store, room, runner, orch = make_env(tmp_path, monkeypatch, agents=("perf",))
    t = room.create_thread("main")
    # Persisted but never orchestrated (as if the process died right after
    # add_message): bypass room.post so the inbox never sees it.
    store.add_message(t.id, "alice", "human", "perf: go")
    await drive(orch, lambda: len(replies(store, t.id)) == 1)
    assert runner.calls and runner.calls[0] == ("perf", t.id)


async def test_recovery_skips_pre_upgrade_threads(tmp_path, monkeypatch):
    store, room, runner, orch = make_env(tmp_path, monkeypatch, agents=("perf",))
    t = room.create_thread("main")
    m = store.add_message(t.id, "alice", "human", "perf: go")
    # Simulate a thread created before the watermark table existed.
    store._db.execute("DELETE FROM orchestrated WHERE thread_id = ?", (t.id,))
    store._db.commit()
    task = asyncio.create_task(orch.run())
    await asyncio.sleep(0.2)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert runner.calls == []  # history not replayed
    assert store.get_orchestrated(t.id) == m.id  # initialized to the tip


async def test_idle_worker_exits_and_revives(tmp_path, monkeypatch):
    monkeypatch.setattr(orch_mod, "_IDLE_S", 0.05)
    store, room, _runner, orch = make_env(
        tmp_path, monkeypatch, agents=("perf",), delay=0.01
    )
    t = room.create_thread("main")
    await room.post(t.id, "alice", "human", "perf: go")
    await drive(orch, lambda: len(replies(store, t.id)) == 1)
    # New orchestrator run: worker spawns, idles out, then a late message
    # must still be processed by a respawned worker.
    task = asyncio.create_task(orch.run())
    await asyncio.sleep(0.2)  # > _IDLE_S: worker for t exits
    assert t.id not in orch._workers
    await room.post(t.id, "alice", "human", "perf: again")
    try:
        async with asyncio.timeout(5.0):
            while len(replies(store, t.id)) < 2:
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_room_post_enqueues_in_id_order(tmp_path, monkeypatch):
    store = Store(tmp_path / "t.db")
    room = Room(store)

    class SlowTransport:
        name = "slow"
        delay = 0.05

        async def deliver(self, msg):
            # First poster suspends here AFTER its id is assigned and enqueued;
            # a second poster must not overtake it in the inbox.
            await asyncio.sleep(self.delay)

    room.add_transport(SlowTransport())
    t = room.create_thread("main")

    async def post(text):
        await room.post(t.id, "alice", "human", text)

    await asyncio.gather(post("one"), post("two"))
    ids = [room.inbox.get_nowait().id for _ in range(2)]
    assert ids == sorted(ids)


async def test_turn_timeout_cuts_off_stuck_turn(tmp_path, monkeypatch):
    cfg = Config()
    cfg.orchestrator = OrchestratorConfig(turn_timeout=0.05)
    store, room, _runner, orch = make_env(
        tmp_path, monkeypatch, agents=("perf",), cfg=cfg, delay=10.0
    )
    t = room.create_thread("main")
    await room.post(t.id, "alice", "human", "perf: go")

    def timed_out():
        msgs = store.thread_messages(t.id)
        return any(m.kind == "notice" and "gave up after" in m.text for m in msgs)

    await drive(orch, timed_out)
    assert replies(store, t.id) == []  # no reply ever landed


async def test_busy_agent_does_not_block_idle_agent(tmp_path, monkeypatch):
    # The regression: A finishes fast, B is slow on the same message. A new
    # message addressed to A must be answered while B still grinds -- A is not
    # parked behind B by the thread's worker.
    monkeypatch.setattr(orch_mod, "make_model", lambda mid: _FixedFilter("NOBODY"))
    store = Store(tmp_path / "t.db")
    room = Room(store)

    class PerAgentDelay(FakeRunner):
        async def run_turn(self, name, thread_id, *, addressed=False, task_prompt=None):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(2.0 if name == "compiler" else 0.01)
            finally:
                self.active -= 1
            self.calls.append((name, thread_id))
            return f"reply from {name}"

    runner = PerAgentDelay(("perf", "compiler"))
    orch = Orchestrator(Config(), room, runner, store)
    t = room.create_thread("main")
    await room.post(t.id, "alice", "human", "perf, compiler: go")  # both
    await room.post(t.id, "alice", "human", "perf: again")  # perf only

    def perf_answered_twice():
        return sum(1 for m in replies(store, t.id) if m.sender == "perf") >= 2

    # If A were blocked behind B (the bug), perf's second reply could not land
    # until compiler's 2s turn finished; the 1.5s budget would time out.
    await drive(orch, perf_answered_twice, timeout=1.5)
    compiler = [m for m in replies(store, t.id) if m.sender == "compiler"]
    assert compiler == []  # B still busy while A got through both


async def test_cancelled_turn_is_not_committed(tmp_path, monkeypatch):
    # A turn cancelled mid-flight (shutdown) must leave its message below the
    # watermark so recovery replays it -- the at-least-once guarantee. If the
    # decrement ran on cancellation, the message would commit and be lost.
    monkeypatch.setattr(orch_mod, "make_model", lambda mid: _FixedFilter("NOBODY"))
    store = Store(tmp_path / "t.db")
    room = Room(store)
    runner = FakeRunner(("perf",), delay=10.0)  # turn never finishes in time
    orch = Orchestrator(Config(), room, runner, store)
    t = room.create_thread("main")
    await room.post(t.id, "alice", "human", "perf: go")
    task = asyncio.create_task(orch.run())
    async with asyncio.timeout(2.0):
        while runner.active == 0:  # wait until perf's turn is in flight
            await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert (store.get_orchestrated(t.id) or 0) == 0  # uncommitted -> replays
    assert replies(store, t.id) == []  # perf never finished


async def test_finish_one_commits_only_contiguous_prefix(tmp_path, monkeypatch):
    # Out-of-order round completion must not advance the watermark past an
    # incomplete earlier message; the prefix commits contiguously.
    monkeypatch.setattr(orch_mod, "make_model", lambda mid: object())
    store = Store(tmp_path / "t.db")
    room = Room(store)
    orch = Orchestrator(Config(), room, FakeRunner(("perf",)), store)
    t = room.create_thread("main")
    m1 = store.add_message(t.id, "a", "human", "one")
    m2 = store.add_message(t.id, "a", "human", "two")
    pm1, pm2 = _PendingMsg(m1.id, remaining=1), _PendingMsg(m2.id, remaining=1)
    orch._pending[t.id] = [pm1, pm2]

    orch._finish_one(pm2, t.id)  # later message done first
    assert store.get_orchestrated(t.id) == 0  # still blocked on m1
    assert [p.msg_id for p in orch._pending[t.id]] == [m1.id, m2.id]

    orch._finish_one(pm1, t.id)  # now the whole prefix pops
    assert store.get_orchestrated(t.id) == m2.id
    assert t.id not in orch._pending


class _FixedFilter:
    """Filter model whose ainvoke always answers with a fixed string."""

    def __init__(self, answer):
        self._answer = answer
        self.prompts = []

    async def ainvoke(self, prompt):
        from types import SimpleNamespace

        self.prompts.append(prompt)
        return SimpleNamespace(text=self._answer, usage_metadata=None)


async def test_announcement_always_routes_to_one_agent(tmp_path, monkeypatch):
    # Watcher announcements skip the should-anyone-speak gate: the router must
    # name exactly one commentator (triage already decided it deserves one).
    fixed = _FixedFilter("perf")
    monkeypatch.setattr(orch_mod, "make_model", lambda mid: fixed)
    store = Store(tmp_path / "t.db")
    room = Room(store)
    runner = FakeRunner(("perf", "compiler"), delay=0.01)
    orch = Orchestrator(Config(), room, runner, store)
    t = room.create_thread("[v8] [maglev] thing")
    await room.post(t.id, "repo:v8", "system", "[v8] [maglev] thing\ndetails here")
    await drive(orch, lambda: len(replies(store, t.id)) == 1)
    assert runner.calls[0] == ("perf", t.id)
    system, user = fixed.prompts[0]
    assert "WHICH ONE" in system["content"]
    assert "[v8] [maglev] thing" in user["content"]


async def test_coordinator_prompt_is_cache_friendly(tmp_path, monkeypatch):
    # The stable prefix (intro + full roster + rules) must live in the system
    # message and the variable transcript/sender exclusion in the user message,
    # so the system prefix is byte-identical regardless of who sent the message.
    monkeypatch.setattr(orch_mod, "make_model", lambda mid: _FixedFilter("NOBODY"))
    store = Store(tmp_path / "t.db")
    room = Room(store)
    runner = FakeRunner(("perf", "compiler"))
    orch = Orchestrator(Config(), room, runner, store)
    t = room.create_thread("main")
    m = store.add_message(t.id, "perf", "agent", "is the regression real?")

    msgs = orch._coordinator_prompt(m, ["compiler"], streak=1)
    system, user = msgs[0]["content"], msgs[1]["content"]
    # Full roster + rules in the stable prefix; no transcript leaks into it.
    assert "perf: perf expert" in system and "compiler: compiler expert" in system
    assert "Default to NOBODY" in system
    assert "is the regression real?" not in system
    # Variable transcript and the sender exclusion live in the user message.
    assert "is the regression real?" in user
    assert "do not pick perf" in user

    # The system prefix does not depend on the sender (cache stability).
    other = store.add_message(t.id, "compiler", "agent", "maybe")
    assert orch._coordinator_prompt(other, ["perf"], streak=2)[0]["content"] == system


async def test_announcement_follow_up_dispatches_event_does_not(tmp_path, monkeypatch):
    # An announcement carrying a follow_up dispatches its response to the
    # registered handler (here a stub injecting a brief via the TurnContext); a
    # perf changepoint is an event, routes through the coordinator, carries no
    # follow_up, and gets a plain turn. Domain-neutral: the room knows only the
    # key, not what the handler does (the coding brief/digest lives in the plugin).
    async def brief_handler(ctx):
        text = await ctx.run_turn(task_prompt="TAG BRIEF")
        if text is not None:
            await ctx.post(text)

    fixed = _FixedFilter("perf")
    monkeypatch.setattr(orch_mod, "make_model", lambda mid: fixed)
    store = Store(tmp_path / "t.db")
    room = Room(store)
    runner = FakeRunner(("perf", "compiler"), delay=0.01)
    orch = Orchestrator(
        Config(), room, runner, store, follow_ups={"commit": brief_handler}
    )

    t1 = room.create_thread("[v8] thing")
    await room.post(
        t1.id, "repo:v8", "system", "[v8] thing\ndetails", follow_up="commit"
    )
    t2 = room.create_thread("skiz regression")
    await room.post(t2.id, "perf:skiz", "event", "JetStream3 down 4% on m3")
    await drive(orch, lambda: {c[1] for c in runner.calls} >= {t1.id, t2.id})

    assert ("perf", t1.id, "TAG BRIEF") in runner.task_prompts
    assert ("perf", t2.id, None) in runner.task_prompts
    # The event went through the coordinator (default-silence gate), not the
    # forced-one announcement router: its prompt tags the perf line [event].
    assert any("[event] perf:skiz" in user["content"] for _, user in fixed.prompts)


async def test_human_address_marks_turn_addressed(tmp_path, monkeypatch):
    # A human "handle:" address withdraws the agent's NOTHING_TO_ADD hatch:
    # run_turn must be told the turn is addressed.
    fixed = _FixedFilter("NOBODY")  # ends any follow-on round
    monkeypatch.setattr(orch_mod, "make_model", lambda mid: fixed)
    store = Store(tmp_path / "t.db")
    room = Room(store)
    runner = FakeRunner(("perf", "compiler"), delay=0.01)
    orch = Orchestrator(Config(), room, runner, store)
    t = room.create_thread("main")
    await room.post(t.id, "alice", "human", "compiler: why does this deopt?")
    await drive(orch, lambda: len(replies(store, t.id)) == 1)
    assert runner.addressed[0] == ("compiler", t.id, True)


async def test_coordinator_route_is_not_addressed(tmp_path, monkeypatch):
    # A reply the coordinator chose (no explicit address) keeps the hatch.
    fixed = _FixedFilter("perf")
    monkeypatch.setattr(orch_mod, "make_model", lambda mid: fixed)
    store = Store(tmp_path / "t.db")
    room = Room(store)
    runner = FakeRunner(("perf", "compiler"), delay=0.01)
    orch = Orchestrator(Config(), room, runner, store)
    t = room.create_thread("main")
    await room.post(t.id, "alice", "human", "the lowering looks slow here")
    await drive(orch, lambda: len(replies(store, t.id)) == 1)
    assert runner.addressed[0] == ("perf", t.id, False)


async def test_agent_address_forces_other_agent(tmp_path, monkeypatch):
    # An agent's reply that STARTS with another handle forces that agent --
    # even when it echoes its own name into the address ("perf, compiler: ..."),
    # which previously voided the whole match (sender not in candidates).
    class ChainingRunner(FakeRunner):
        async def run_turn(self, name, thread_id, *, addressed=False, task_prompt=None):
            await super().run_turn(name, thread_id, addressed=addressed)
            if name == "perf":
                return "perf, compiler: please double-check the lowering"
            return "compiler checked it"

    monkeypatch.setattr(orch_mod, "make_model", lambda mid: object())
    store = Store(tmp_path / "t.db")
    room = Room(store)
    runner = ChainingRunner(("perf", "compiler"), delay=0.01)
    orch = Orchestrator(Config(), room, runner, store)
    t = room.create_thread("main")
    await room.post(t.id, "alice", "human", "perf: go")
    await drive(
        orch, lambda: any("compiler checked" in m.text for m in replies(store, t.id))
    )
    assert ("compiler", t.id) in runner.calls
    # The address came from an agent, not a human, so the override is off.
    assert ("compiler", t.id, False) in runner.addressed


def test_reap_locks_keeps_held_locks(tmp_path, monkeypatch):
    # A timer wake holds a per-(thread, agent) lock without registering in
    # _pending; the idle-worker lock reaping must not delete a held lock, else a
    # later routed turn would get a fresh lock and run concurrently with the wake.
    _, _, _, orch = make_env(tmp_path, monkeypatch)
    held = asyncio.Lock()
    idle = asyncio.Lock()
    orch._agent_locks[(7, "perf")] = held  # a wake is running under this one
    orch._agent_locks[(7, "compiler")] = idle
    orch._agent_locks[(9, "perf")] = asyncio.Lock()  # other thread, untouched

    async def _run():
        async with held:  # simulate the in-flight wake turn
            orch._reap_locks(7)

    asyncio.run(_run())
    assert (7, "perf") in orch._agent_locks  # held -> survived
    assert (7, "compiler") not in orch._agent_locks  # idle -> reaped
    assert (9, "perf") in orch._agent_locks  # other thread -> untouched


async def test_run_structured_turn_serializes_per_persona(tmp_path):
    # Structured turns of one persona share a LangGraph thread id on purpose (the
    # growing prefix cache accumulates across the stream), so concurrent runs
    # would share cache state and contaminate each other's context. The runner
    # must serialize them per persona.
    from airc_room.personas import Persona
    from airc_room.runner import AgentRunner, _AgentEntry, _TurnUsage

    cfg = Config()
    cfg.token_db_path = tmp_path / "tokens.db"
    runner = AgentRunner(cfg, {}, object(), object())
    runner._agents = {
        "Sonic": _AgentEntry(
            persona=Persona(
                name="Sonic",
                display_name="Sonic",
                description="d",
                system_prompt="",
                key="perf",
            ),
            graph=None,
        )
    }
    runner._structured_agents["Sonic"] = object()  # skip graph construction
    active, max_active = 0, 0

    async def stream(graph, name, input, config):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return '{"tag": "SKIP", "summary": ""}', _TurnUsage()

    runner._stream = stream
    results = await asyncio.gather(
        runner.run_structured_turn("Sonic", "commit A", extra_system="x"),
        runner.run_structured_turn("Sonic", "commit B", extra_system="x"),
    )
    assert all(r is not None for r in results)
    assert max_active == 1  # never overlapped

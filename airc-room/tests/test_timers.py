# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

import asyncio
import time

from airc_room.store import Store
from airc_room.timers import (
    _MAX_PENDING_PER_THREAD,
    TimerScheduler,
    make_timer_tools,
)
from airc_room.turn_context import (
    AGENT_KEY,
    THREAD_KEY,
    TRIGGER_KEY,
    turn_config,
    turn_context,
    turn_trigger,
)


def _timer_tools(scheduler):
    """The three timer tools by name, for tests to invoke individually."""
    return {t.name: t for t in make_timer_tools(scheduler)}


def _cfg(thread_id=7, agent="perf", generation=0):
    """A turn config built the way the runner builds it. Tests must never hand-
    write this dict: the composite thread_id is the runner's private checkpoint
    key, and tests that spelled it out by hand are exactly why a change to its
    shape went unnoticed while every timer wake was being dropped in prod."""
    return {"configurable": turn_config(thread_id, agent, generation)}


def test_turn_context_reads_identity_from_the_runners_config():
    assert turn_context(_cfg(7, "perf")) == (7, "perf")
    # Nickname handles: the STABLE key travels, whatever the addressable name is.
    assert turn_context(_cfg(42, "sonic")) == (42, "sonic")


def test_turn_context_survives_a_bumped_context_generation():
    # The regression. A memory compaction bumps the generation, which folds into
    # the composite checkpoint id; the persona identity must be unaffected. The
    # old parser split the composite on its first colon and returned "perf:g3"
    # here, which matches no live persona, so every wake was dropped.
    assert turn_context(_cfg(7, "perf", generation=3)) == (7, "perf")


def test_turn_context_refuses_a_turn_with_no_identity():
    # The forced-JSON structured turn sets a bare thread_id and holds no local
    # tools. Reparsing it would invent a thread; refusing is the honest answer.
    assert turn_context({"configurable": {"thread_id": "structured:perf"}}) == (
        None,
        "",
    )
    assert turn_context({}) == (None, "")
    assert turn_context(None) == (None, "")


def test_turn_trigger_reads_the_message_that_caused_the_turn():
    cfg = {"configurable": turn_config(7, "perf", 0, trigger_id=42)}
    assert turn_trigger(cfg) == 42
    # And the identity is unaffected by carrying one.
    assert turn_context(cfg) == (7, "perf")


def test_turn_trigger_is_none_when_no_message_caused_the_turn():
    # A timer wake is driven by a note, not a message; a structured turn belongs
    # to no thread at all. Both are ordinary, so None is an answer rather than a
    # fault -- the tool reading it decides what to do without one.
    assert turn_trigger(_cfg(7, "perf")) is None
    assert turn_trigger({"configurable": {}}) is None
    assert turn_trigger({}) is None
    assert turn_trigger(None) is None


def test_a_triggerless_turn_still_has_an_identity():
    # The reason the trigger is read separately rather than as a third element
    # of turn_context: folding it into that all-or-nothing check would make
    # every timer wake read as "no identity" and silently disable every local
    # tool on it.
    cfg = _cfg(7, "perf")
    assert turn_context(cfg) == (7, "perf")
    assert turn_trigger(cfg) is None
    # bool is an int subclass; True must not read as message 1.
    assert turn_trigger({"configurable": {TRIGGER_KEY: True}}) is None


def test_turn_context_refuses_a_half_populated_identity():
    # All or nothing. Returning the half that is present is how this breaks
    # quietly: an empty agent still reads as "present" to timer_create (which
    # only checks the thread id), so it would report success, persist a timer no
    # persona can own, and have the wake dropped at fire time -- the exact
    # signature of the bug this module was written to kill.
    assert turn_context({"configurable": {THREAD_KEY: 7}}) == (None, "")
    assert turn_context({"configurable": {AGENT_KEY: "perf"}}) == (None, "")
    assert turn_context({"configurable": {THREAD_KEY: 7, AGENT_KEY: ""}}) == (None, "")
    # bool is an int subclass; True must not read as thread 1.
    assert turn_context({"configurable": {THREAD_KEY: True, AGENT_KEY: "perf"}}) == (
        None,
        "",
    )


async def test_run_turn_builds_a_config_its_own_tools_can_read(tmp_path, monkeypatch):
    """The contract, end to end: the config the REAL run_turn builds must be
    readable by the REAL turn_context.

    This is the test the original bug needed and did not have. Every other test
    here checks one side against a fixture, so the two halves could drift apart
    (the runner folded ":g<n>" into the composite id, the parser kept splitting
    on the first colon) with all of them still green. Asserting on the parsed
    identity rather than on the config's shape is the point -- it stays true
    however the runner chooses to key its checkpoints next.
    """
    from airc_room.config import Config
    from airc_room.personas import Persona
    from airc_room.runner import AgentRunner, _AgentEntry, _TurnUsage

    store = Store(tmp_path / "airc.db")
    thread = store.create_thread("t")
    store.add_message(thread.id, "human", "human", "perf: have a look")
    # A compacted thread, which is what exposed the bug: the generation is
    # non-zero and folds into the composite checkpoint id.
    store.bump_context_generation(thread.id)
    store.bump_context_generation(thread.id)

    cfg = Config()
    cfg.token_db_path = tmp_path / "tokens.db"
    runner = AgentRunner(cfg, {}, object(), store)
    persona = Persona(
        name="Sonic",  # addressable name differs from the stable key
        display_name="Sonic",
        description="d",
        system_prompt="",
        key="perf",
    )
    runner._agents = {"Sonic": _AgentEntry(persona=persona, graph=object())}

    seen: dict = {}

    async def _fake_stream(graph, agent_name, payload, config):
        seen.update(config)
        return "ok", _TurnUsage()

    monkeypatch.setattr(runner, "_stream", _fake_stream)
    await runner.run_turn("Sonic", thread.id, addressed=True, trigger_id=99)
    store.close()

    # The identity a local tool would recover, from what the runner actually built.
    assert turn_context(seen) == (thread.id, "perf")
    # And the provenance, through the same real-runner path.
    assert turn_trigger(seen) == 99
    # Under use_nicknames the addressable name is the nickname while the stable
    # key stays the folder handle; a timer stores the key, so the wake path must
    # translate key -> name. The key is never the name, and a dead persona is None.
    from airc_room.config import Config
    from airc_room.personas import Persona
    from airc_room.runner import AgentRunner, _AgentEntry

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
    assert runner.name_for_key("perf") == "Sonic"  # stable key -> live name
    assert runner.name_for_key("Sonic") is None  # the live name is not a key
    assert runner.name_for_key("ghost") is None  # not live -> dropped


async def test_deliver_wake_translates_stable_key_before_respond(monkeypatch):
    # Regression: with nicknames on, a timer persists the stable key ("perf") but
    # runner.agents is keyed by the nickname ("Sonic"), so the wake was dropped as
    # "agent gone". deliver_wake must translate the key to the live name.
    from airc_room import orchestrator as orch_mod
    from airc_room.config import Config
    from airc_room.orchestrator import Orchestrator

    monkeypatch.setattr(orch_mod, "make_model", lambda mid: object())

    class _FakeRunner:
        def name_for_key(self, key):
            return "Sonic" if key == "perf" else None

    class _FakeRoom:
        # deliver_wake now runs the execution bracket itself (typing + post);
        # stub the two room calls it makes so the translation path is exercised.
        async def typing(self, *a, **k):
            pass

        async def post(self, *a, **k):
            pass

    orch = Orchestrator(Config(), _FakeRoom(), runner=_FakeRunner(), store=object())

    calls = []

    async def _fake_guarded_turn(name, thread_id, **kw):
        calls.append((name, thread_id, kw))

    monkeypatch.setattr(orch, "_guarded_turn", _fake_guarded_turn)

    await orch.deliver_wake(5, "perf", "check pinpoint 1234")
    assert calls == [
        ("Sonic", 5, {"addressed": True, "task_prompt": "check pinpoint 1234"})
    ]

    calls.clear()
    await orch.deliver_wake(5, "ghost", "x")  # no live persona -> dropped
    assert calls == []


def test_add_respects_per_thread_cap():
    sch = TimerScheduler()
    now = time.time()
    for i in range(_MAX_PENDING_PER_THREAD):
        assert sch.add(1, "a", now + 100, f"n{i}") is not None
    assert sch.add(1, "a", now + 100, "overflow") is None  # cap hit on thread 1
    assert sch.add(2, "a", now + 100, "other") is not None  # a different thread is fine


def test_add_returns_unique_ids():
    sch = TimerScheduler()
    now = time.time()
    ids = [sch.add(1, "a", now + 100, f"n{i}") for i in range(3)]
    assert ids == [0, 1, 2]  # the monotonic sequence, never reused


def test_cancel_frees_slot_and_is_thread_scoped():
    sch = TimerScheduler()
    now = time.time()
    ids = [sch.add(1, "a", now + 100, f"n{i}") for i in range(_MAX_PENDING_PER_THREAD)]
    assert sch.add(1, "a", now + 100, "overflow") is None  # at cap
    # Cancelling one frees exactly one slot.
    assert sch.cancel(1, ids[0]) is True
    assert sch.add(1, "a", now + 100, "now fits") is not None
    # A second cancel of the same id is a no-op (already gone), and does not
    # over-free the cap.
    assert sch.cancel(1, ids[0]) is False
    # Another thread cannot cancel thread 1's timer.
    assert sch.cancel(2, ids[1]) is False
    assert sch.cancel(1, ids[1]) is True


def test_list_for_is_thread_scoped_and_sorted():
    sch = TimerScheduler()
    now = time.time()
    a = sch.add(1, "a", now + 200, "later")
    b = sch.add(1, "a", now + 50, "sooner")
    sch.add(2, "a", now + 10, "other thread")
    listed = sch.list_for(1)
    assert [tid for tid, _, _ in listed] == [b, a]  # soonest first, thread 1 only
    assert [note for _, _, note in listed] == ["sooner", "later"]


async def test_cancelled_timer_does_not_fire():
    fired: list[str] = []

    async def deliver(thread_id, agent, note):
        fired.append(note)

    sch = TimerScheduler()
    sch.deliver = deliver
    task = asyncio.create_task(sch.run())
    now = time.time()
    keep = sch.add(1, "a", now + 0.05, "keep")
    drop = sch.add(1, "a", now + 0.03, "drop")
    assert sch.cancel(1, drop) is True  # tombstoned before it fires
    await asyncio.sleep(0.15)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert fired == ["keep"]
    assert sch.list_for(1) == []  # keep fired, drop cancelled -> none pending
    assert keep is not None


async def test_scheduler_fires_in_fire_order_with_preemption():
    fired: list[str] = []

    async def deliver(thread_id, agent, note):
        fired.append(note)

    sch = TimerScheduler()
    sch.deliver = deliver
    task = asyncio.create_task(sch.run())
    now = time.time()
    sch.add(1, "a", now + 0.08, "late")
    sch.add(1, "a", now + 0.02, "early")  # added second, must fire first
    await asyncio.sleep(0.2)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert fired == ["early", "late"]


async def test_scheduler_decrements_cap_after_firing():
    sch = TimerScheduler()
    sch.deliver = lambda *a: asyncio.sleep(0)
    task = asyncio.create_task(sch.run())
    sch.add(1, "a", time.time() + 0.02, "one")
    await asyncio.sleep(0.1)
    # After firing, the thread's slot is freed, so we can schedule again.
    for i in range(_MAX_PENDING_PER_THREAD):
        assert sch.add(1, "a", time.time() + 100, f"n{i}")
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_timer_create_schedules_and_reads_context():
    sch = TimerScheduler()
    create = _timer_tools(sch)["timer_create"]
    cfg = _cfg()
    out = await create.ainvoke({"minutes": 60, "note": "check job 5"}, config=cfg)
    assert "scheduled timer 0" in out  # the id is surfaced for cancel
    # The wake was enqueued for the config's (thread, agent).
    assert sch.list_for(7) == [(0, sch._live[0].fire_at, "check job 5")]
    assert sch._live[0].agent == "perf"


async def test_timer_create_guardrails():
    create = _timer_tools(TimerScheduler())["timer_create"]
    cfg = _cfg()
    assert "declined" in await create.ainvoke({"minutes": -1, "note": "x"}, config=cfg)
    assert "declined" in await create.ainvoke(
        {"minutes": 9_999_999, "note": "x"}, config=cfg
    )
    assert "declined" in await create.ainvoke({"minutes": 5, "note": " "}, config=cfg)
    # No turn context -> cannot schedule.
    assert "context" in await create.ainvoke({"minutes": 5, "note": "x"}, config={})


async def test_timer_create_reports_cap_decline():
    sch = TimerScheduler()
    create = _timer_tools(sch)["timer_create"]
    cfg = _cfg()
    for _ in range(_MAX_PENDING_PER_THREAD):
        await create.ainvoke({"minutes": 5, "note": "x"}, config=cfg)
    assert "declined" in await create.ainvoke({"minutes": 5, "note": "x"}, config=cfg)


async def test_timer_list_and_cancel_tools_roundtrip():
    sch = TimerScheduler()
    tools = _timer_tools(sch)
    cfg = _cfg()
    assert "no pending timers" in await tools["timer_list"].ainvoke({}, config=cfg)
    await tools["timer_create"].ainvoke({"minutes": 30, "note": "stir"}, config=cfg)
    listed = await tools["timer_list"].ainvoke({}, config=cfg)
    assert "0:" in listed and "stir" in listed
    # Cancel by id, scoped to this chat.
    assert "cancelled timer 0" in await tools["timer_cancel"].ainvoke(
        {"timer_id": 0}, config=cfg
    )
    assert "no pending timer 0" in await tools["timer_cancel"].ainvoke(
        {"timer_id": 0}, config=cfg
    )
    assert "no pending timers" in await tools["timer_list"].ainvoke({}, config=cfg)


# ── persistence across restart ───────────────────────────────────────────────


def test_restore_rebuilds_heap_and_seeds_ids(tmp_path):
    # A fresh scheduler over the same store sees the prior run's pending timers,
    # and the id counter resumes past the highest restored id so a new timer
    # cannot collide with a restored one.
    store = Store(tmp_path / "s.db")
    now = time.time()
    sch = TimerScheduler(store)
    a = sch.add(1, "perf", now + 100, "later")
    b = sch.add(1, "perf", now + 50, "sooner")
    assert (a, b) == (0, 1)

    fresh = TimerScheduler(store)
    fresh.restore()
    assert fresh.list_for(1) == [(1, now + 50, "sooner"), (0, now + 100, "later")]
    assert fresh.add(1, "perf", now + 200, "new") == 2  # seq resumed past 1
    store.close()


def test_cancel_and_restore_drops_the_row(tmp_path):
    # A cancelled timer must not come back on restart.
    store = Store(tmp_path / "s.db")
    now = time.time()
    sch = TimerScheduler(store)
    keep = sch.add(1, "perf", now + 100, "keep")
    drop = sch.add(1, "perf", now + 100, "drop")
    assert sch.cancel(1, drop) is True

    fresh = TimerScheduler(store)
    fresh.restore()
    assert [note for _, _, note in fresh.list_for(1)] == ["keep"]
    assert keep == 0
    store.close()


async def test_fired_timer_is_not_re_delivered_after_restore(tmp_path):
    # Firing consumes the persisted row, so a restart does not replay a wake that
    # already ran.
    store = Store(tmp_path / "s.db")
    fired: list[str] = []

    async def deliver(thread_id, agent, note):
        fired.append(note)

    sch = TimerScheduler(store)
    sch.deliver = deliver
    task = asyncio.create_task(sch.run())
    sch.add(1, "perf", time.time() + 0.02, "ran")
    await asyncio.sleep(0.1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert fired == ["ran"]

    fresh = TimerScheduler(store)
    fresh.restore()
    assert fresh.list_for(1) == []  # nothing left to replay
    store.close()


async def test_stale_timer_fires_once_immediately_on_restore(tmp_path):
    # A timer whose fire time passed while the daemon was down surfaces with a
    # non-positive delay and fires on the first tick after restore.
    store = Store(tmp_path / "s.db")
    store.add_timer(0, 1, "perf", time.time() - 3600, "overdue")

    fired: list[str] = []

    async def deliver(thread_id, agent, note):
        fired.append(note)

    sch = TimerScheduler(store)
    sch.deliver = deliver
    sch.restore()
    task = asyncio.create_task(sch.run())
    await asyncio.sleep(0.1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert fired == ["overdue"]
    assert store.all_timers() == []  # consumed, so a further restart is clean
    store.close()


def test_scheduler_without_store_does_not_persist():
    # The bare (storeless) scheduler stays purely in-memory: restore is a no-op.
    sch = TimerScheduler()
    sch.add(1, "perf", time.time() + 100, "n")
    sch.restore()  # no store -> nothing to rebuild, no crash
    assert len(sch.list_for(1)) == 1

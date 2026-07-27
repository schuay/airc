# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

import asyncio
import time

from airc_room.store import Store
from airc_room.timers import (
    _MAX_PENDING_PER_THREAD,
    TimerScheduler,
    _ctx_from_config,
    make_timer_tools,
)


def _timer_tools(scheduler):
    """The three timer tools by name, for tests to invoke individually."""
    return {t.name: t for t in make_timer_tools(scheduler)}


def test_ctx_from_config_parses_thread_and_agent():
    assert _ctx_from_config({"configurable": {"thread_id": "7:perf"}}) == (7, "perf")
    # Nickname handles have no colon; still parses.
    assert _ctx_from_config({"configurable": {"thread_id": "42:sonic"}}) == (
        42,
        "sonic",
    )
    # Missing/malformed context degrades to (None, "").
    assert _ctx_from_config({}) == (None, "")
    assert _ctx_from_config(None) == (None, "")


def test_name_for_key_translates_stable_key_to_live_name(tmp_path):
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
        return None  # nothing to add -> no post

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
    cfg = {"configurable": {"thread_id": "7:perf"}}
    out = await create.ainvoke({"minutes": 60, "note": "check job 5"}, config=cfg)
    assert "scheduled timer 0" in out  # the id is surfaced for cancel
    # The wake was enqueued for the config's (thread, agent).
    assert sch.list_for(7) == [(0, sch._live[0].fire_at, "check job 5")]
    assert sch._live[0].agent == "perf"


async def test_timer_create_guardrails():
    create = _timer_tools(TimerScheduler())["timer_create"]
    cfg = {"configurable": {"thread_id": "7:perf"}}
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
    cfg = {"configurable": {"thread_id": "7:perf"}}
    for _ in range(_MAX_PENDING_PER_THREAD):
        await create.ainvoke({"minutes": 5, "note": "x"}, config=cfg)
    assert "declined" in await create.ainvoke({"minutes": 5, "note": "x"}, config=cfg)


async def test_timer_list_and_cancel_tools_roundtrip():
    sch = TimerScheduler()
    tools = _timer_tools(sch)
    cfg = {"configurable": {"thread_id": "7:perf"}}
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

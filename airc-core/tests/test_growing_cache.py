# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""_GrowingPrefixCache: per-conversation boundary growth, isolation, recovery.

Cache create/delete are mocked; these exercise the per-call logic (what to cache,
what tail to send, run-scoping, the window guard, LRU eviction, cooldown) plus a
real create_agent graph end to end. tools_tokens is set above the 4096 floor so
caching engages without padding the message content.
"""

import pytest
from airc_core.agent import (
    _GrowingPrefixCache,
    _is_cache_gone,
    _last_step_boundary,
    _recache_pays,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


async def _noop(r):
    return "ok"


SYS = SystemMessage("system prompt")


class _Req:
    """ModelRequest stand-in: messages, state, runtime.execution_info.thread_id,
    and override(model=, messages=)."""

    def __init__(self, messages, thread="t1", model="BASE", model_calls=None):
        self.messages = messages
        self.model = model
        # model_calls is what CallBudgetMiddleware puts in graph state; None
        # models a graph built without it (the key is simply absent).
        self.state = {} if model_calls is None else {"model_calls": model_calls}
        self.runtime = type(
            "RT", (), {"execution_info": type("EI", (), {"thread_id": thread})()}
        )()

    def override(self, *, model=None, messages=None):
        return _Req(
            messages if messages is not None else self.messages,
            thread=self.runtime.execution_info.thread_id,
            model=model if model is not None else self.model,
        )


def _step(i):
    cid = f"c{i}"
    return [
        AIMessage(content="", tool_calls=[{"name": "look", "args": {}, "id": cid}]),
        ToolMessage(content=f"result {i}", tool_call_id=cid),
    ]


def _history(n_steps):
    msgs = [HumanMessage("hello")]
    for i in range(n_steps):
        msgs += _step(i)
    return msgs


def _fat_history(n_steps, chars=40_000):
    """History whose tool results are large enough that growing the cache repays
    its creation -- the payback rule is about tokens, so a thin history of the
    same message count must NOT trigger a re-cache."""
    msgs = [HumanMessage("hello")]
    for i in range(n_steps):
        cid = f"c{i}"
        msgs += [
            AIMessage(content="", tool_calls=[{"name": "look", "args": {}, "id": cid}]),
            ToolMessage(content="x" * chars, tool_call_id=cid),
        ]
    return msgs


def _mw(tools_tokens=5000, max_calls=70):
    state = {"created": [], "deleted": []}

    async def create(prefix):
        name = f"c{len(state['created']) + 1}"
        state["created"].append((name, len(prefix)))
        return name

    async def delete(name):
        state["deleted"].append(name)

    def model_for(name):
        return f"M:{name}"

    mw = _GrowingPrefixCache(
        create, delete, model_for, SYS, tools_tokens, max_calls=max_calls
    )
    return mw, state


async def _run(mw, req):
    seen = {}

    async def handler(r):
        seen["model"], seen["messages"] = r.model, r.messages
        return "ok"

    await mw.awrap_model_call(req, handler)
    return seen


# ── _is_cache_gone ───────────────────────────────────────────────────────────


def test_is_cache_gone_matches_deleted_and_expired():
    assert _is_cache_gone(RuntimeError("404 CachedContent not found"))
    assert _is_cache_gone(Exception("400 Cache content 7808 is expired."))
    assert not _is_cache_gone(RuntimeError("429 rate limit exceeded"))


def test_is_cache_gone_matches_known_id_regardless_of_wording():
    assert _is_cache_gone(Exception("780 has been evicted, sorry"), name="780")
    assert not _is_cache_gone(Exception("400 malformed request"), name="780")
    assert not _is_cache_gone(Exception("400 missing a thought signature"))


def test_last_step_boundary():
    assert _last_step_boundary([HumanMessage("x")]) == 0
    h = _history(3)
    # The prefix ends on the last AIMessage holding a function call; its tool
    # response opens the tail.
    assert _last_step_boundary(h) == 6
    assert isinstance(h[5], AIMessage) and isinstance(h[6], ToolMessage)
    # Conversational rest-point: plain user text is a valid prefix ending.
    conv = [HumanMessage("q"), AIMessage("a"), HumanMessage("q2"), AIMessage("a2")]
    assert _last_step_boundary(conv) == 3
    # A human turn right after a tool response merges with it on the wire, so
    # it is no rest-point; fall back to the function-call ending.
    merged = [*_history(1), HumanMessage("go on"), AIMessage("a")]
    assert _last_step_boundary(merged) == 2


# ── boundary growth ──────────────────────────────────────────────────────────


async def test_first_call_caches_system_only_and_sends_full_history():
    # boundary 0 == the system+tools cache: full history is the tail.
    mw, state = _mw()
    seen = await _run(mw, _Req(_history(0)))
    assert state["created"] == [("c1", 1)]  # cached [system] only
    assert seen["model"] == "M:c1"
    assert seen["messages"] == _history(0)  # full history sent


async def test_grows_to_a_step_boundary_and_sends_only_the_tail():
    mw, state = _mw()
    await mw.awrap_model_call(_Req(_history(0)), _noop)  # cache gen c1 (boundary 0)
    msgs = _fat_history(4)  # tail large enough to repay a re-cache
    seen = await _run(mw, _Req(msgs))
    boundary = _last_step_boundary(msgs)  # 8
    assert state["created"][-1] == ("c2", boundary + 1)  # [system]+prefix
    assert state["deleted"] == ["c1"]  # superseded gen deleted
    assert seen["model"] == "M:c2"
    assert seen["messages"] == msgs[boundary:]
    # The tail opens on the tool response answering the cached function call.
    assert isinstance(seen["messages"][0], ToolMessage)


# ── re-cache payback ─────────────────────────────────────────────────────────


def test_recache_pays_follows_the_eoq_threshold():
    # delta*calls_since >= 2B, with a horizon long enough not to bind.
    assert not _recache_pays(100_000, 1_000, 4, calls_left=70)[0]  # 4k << 200k
    assert not _recache_pays(100_000, 10_000, 19, calls_left=70)[0]  # 190k < 200k
    assert _recache_pays(100_000, 10_000, 21, calls_left=70)[0]  # 210k >= 200k


def test_recache_pays_needs_actual_growth():
    # A boundary that moved no tokens can never repay: guards a divide-by-zero
    # style degenerate where calls_since alone would eventually trip any
    # threshold.
    assert not _recache_pays(100_000, 0, 1_000, calls_left=70)[0]
    assert not _recache_pays(100_000, -5_000, 1_000, calls_left=70)[0]


def test_recache_pays_refuses_near_the_end_of_a_turn():
    # Same delta and prefix, only the horizon differs: creation costs B now to
    # save (1-r)*delta per remaining call, so with few calls left it is loss.
    big_delta, prefix = 50_000, 100_000
    assert _recache_pays(prefix, big_delta, 20, calls_left=70)[0]
    assert not _recache_pays(prefix, big_delta, 20, calls_left=2)[0]


async def test_thin_growth_does_not_recache():
    # The regression this guards: a fixed message-count trigger re-cached every
    # 8 messages regardless of prefix size, re-buying a ~120k prefix to absorb a
    # ~2k tail -- recouping ~7% of its own cost before being superseded.
    mw, state = _mw(tools_tokens=120_000)
    await _run(mw, _Req(_history(0)))  # first cache
    for steps in range(1, 12):  # many calls, but tiny tool results
        await _run(mw, _Req(_history(steps)))
    assert len(state["created"]) == 1  # never re-cached on thin growth


async def test_horizon_blocks_recache_when_the_turn_is_nearly_over():
    # A delta that HAS satisfied the payback rule (it is read over enough calls
    # in steady state) but cannot repay over the 2 calls actually left. The
    # prefix must dominate the delta for the horizon to bind -- a delta large
    # relative to the prefix repays even in a couple of calls.
    mw, state = _mw(tools_tokens=200_000, max_calls=70)
    await _run(mw, _Req(_history(0)))
    msgs = _fat_history(4)
    for _ in range(20):  # accumulates calls_since past the payback threshold
        await _run(mw, _Req(msgs, model_calls=68))
    assert len(state["created"]) == 1  # horizon suppressed it


async def test_missing_call_count_falls_back_to_the_full_cap():
    # A graph without CallBudgetMiddleware has no model_calls key; the horizon
    # must not read that as "0 calls left" and disable caching outright. Same
    # shape as the horizon test above, differing only in the absent key.
    mw, state = _mw(tools_tokens=200_000, max_calls=70)
    await _run(mw, _Req(_history(0)))
    msgs = _fat_history(4)
    for _ in range(20):
        await _run(mw, _Req(msgs))  # state has no model_calls
    assert len(state["created"]) == 2  # re-cached on merit, horizon inert


async def test_recache_resets_the_payback_clock():
    # calls_since must restart per generation, else an old count keeps tripping
    # the threshold and every subsequent call re-caches.
    mw, state = _mw()
    await _run(mw, _Req(_history(0)))
    msgs = _fat_history(4)
    for _ in range(6):
        await _run(mw, _Req(msgs))
    assert mw._states["t1"].calls_since < 6  # reset by the new generation
    created_after_growth = len(state["created"])
    await _run(mw, _Req(msgs))  # immediately after: must not re-cache again
    assert len(state["created"]) == created_after_growth


# ── per-conversation isolation ───────────────────────────────────────────────


async def test_threads_get_independent_caches():
    mw, state = _mw()
    await _run(mw, _Req(_history(0), thread="A"))
    await _run(mw, _Req(_history(0), thread="B"))
    assert set(mw._states) == {"A", "B"}
    assert mw._states["A"].name != mw._states["B"].name
    assert len(state["created"]) == 2  # one cache per conversation


async def test_new_run_shrink_resets_and_drops_prior_cache():
    # Review keys all runs to one (None) state; a shorter list = a fresh run.
    mw, state = _mw()
    await _run(mw, _Req(_history(5), thread=None))
    first = mw._states[None].name
    await _run(mw, _Req(_history(0), thread=None))
    assert first in state["deleted"]


# ── recovery / failure ───────────────────────────────────────────────────────


async def test_mid_run_cache_gone_degrades_and_rebuilds():
    mw, _state = _mw()
    await _run(mw, _Req(_history(0)))  # cache c1
    calls = []

    async def handler(r):
        calls.append(r.model)
        if r.model == "M:c1" and len(calls) == 1:
            raise RuntimeError("400 Cache content c1 is expired.")
        return "ok"

    out = await mw.awrap_model_call(_Req(_history(0)), handler)
    assert out == "ok"
    assert calls[0] == "M:c1" and calls[1] == "BASE"  # tried cache, then uncached
    assert mw._states["t1"].name is None  # reset for rebuild next call


async def test_permanent_create_failure_cools_down_instance_wide():
    state = {"attempts": 0}

    async def create(prefix):
        state["attempts"] += 1
        raise RuntimeError("permission denied")

    async def delete(name):
        pass

    mw = _GrowingPrefixCache(create, delete, lambda n: n, SYS, 5000, max_calls=70)
    # Two conversations both try to create; a permanent failure holds the long
    # cooldown, capping the storm at one attempt regardless of thread.
    for thread in ("A", "B", "A", "B"):
        await _run(mw, _Req(_history(2), thread=thread))
    assert state["attempts"] == 1


async def test_transient_create_failure_recovers_after_short_cooldown(monkeypatch):
    from airc_core import agent

    # Collapse the transient cooldown to zero so the next turn is immediately
    # eligible to retry -- the regression this guards: a prefill overload used to
    # trip the 15m permanent blackout and run uncached for many minutes.
    monkeypatch.setattr(agent, "_CACHE_TRANSIENT_COOLDOWN_S", 0)
    state = {"attempts": 0}

    async def create(prefix):
        state["attempts"] += 1
        if state["attempts"] == 1:
            raise RuntimeError("429 resource_exhausted: overloaded prefill queue")
        return f"c{state['attempts']}"

    async def delete(name):
        pass

    mw = _GrowingPrefixCache(
        create, delete, lambda n: f"M:{n}", SYS, 5000, max_calls=70
    )
    # First turn's create fails transiently; the short cooldown lets the next
    # turn rebuild instead of staying uncached.
    first = await _run(mw, _Req(_history(2), thread="A"))
    assert first["model"] == "BASE"  # uncached this turn
    second = await _run(mw, _Req(_history(2), thread="A"))
    assert state["attempts"] == 2  # retried, not blacked out
    assert second["model"] == "M:c2"  # served from the rebuilt cache


# ── empty-candidate step-aside ───────────────────────────────────────────────


async def test_empty_candidate_retry_serves_uncached():
    from airc_core import agent

    # _EmptyCandidateRetry sets this for its one mutated retry: the cache must
    # stop serving the prefix that may be producing the empty, so the retry
    # tests a genuinely different input.
    mw, _state = _mw()
    first = await _run(mw, _Req(_history(2)))
    assert first["model"] == "M:c1"  # cached normally
    agent._empty_retry.set(1)
    try:
        seen = await _run(mw, _Req(_history(2)))
    finally:
        agent._empty_retry.set(0)
    assert seen["model"] == "BASE"  # uncached: the full request, no cached_content


async def test_one_empty_candidate_flake_does_not_cost_the_cache():
    """_EmptyCandidateRetry over the real cache: one flake, then a normal turn.

    Driven through both middlewares rather than by setting _empty_retry by
    hand, because the cost being guarded against comes from how the two
    compose -- a hand-set counter cannot see a mismatch between what the retry
    sets and what the cache keys on. A single flake must not tear down a warm
    prefix: on a long conversation that repurchases the whole thing at full
    input rate on every later call in the turn.
    """
    from airc_core import agent

    mw, state = _mw()
    empty = agent._EmptyCandidateRetry()
    box = [AIMessage(content=""), AIMessage(content="the answer")]

    async def inner(req):
        # The cache is nested inside the empty-candidate retry, so it serves
        # each attempt; pop one canned response per model call.
        await mw.awrap_model_call(req, _noop)
        return type("R", (), {"result": [box.pop(0)]})()

    agent._empty_retry.set(0)
    try:
        await empty.awrap_model_call(_Req(_history(2)), inner)  # flake, then ok
        assert not box, "both canned responses should have been consumed"
        assert state["deleted"] == []
        # The turn continues on the SAME cache: no delete, no second prefill.
        after = await _run(mw, _Req(_history(2)))
        assert after["model"] == "M:c1"
        assert [name for name, _ in state["created"]] == ["c1"]
    finally:
        agent._empty_retry.set(0)


# ── window guard ─────────────────────────────────────────────────────────────


async def test_window_guard_serves_uncached_when_prefix_plus_tail_too_big():
    # A prefix near the cap plus a large tail would exceed the window: the cache
    # must step aside and send the full request uncached for that call.
    mw, _ = _mw(tools_tokens=500_000)  # prefix ~500k, under the 0.6*1M cap
    await _run(mw, _Req(_history(0)))  # creates cache (prefix_tokens ~500k)
    big_tail = [*_history(0), HumanMessage("x" * 1_600_000)]  # ~400k-token tail
    seen = await _run(mw, _Req(big_tail))
    assert seen["model"] == "BASE"  # served uncached, not via the cache
    assert seen["messages"] == big_tail


async def test_window_guard_does_not_double_count_the_prefix():
    # A large prefix plus a SMALL tail whose recent AIMessage reports the last
    # call's FULL prompt (prefix + tail) must NOT trip the guard. The old code
    # added prefix_tokens to a tail estimate that already floored on that full
    # prompt, ~doubling the prefix and disabling the cache on exactly the large
    # contexts it exists for.
    mw, _ = _mw(tools_tokens=500_000)  # prefix_tokens ~500k
    await _run(mw, _Req(_history(0)))  # create the cache
    tail_msg = AIMessage(
        content="ok",  # tiny in chars...
        usage_metadata={
            "input_tokens": 550_000,  # ...but reports the full prefix+tail prompt
            "output_tokens": 1,
            "total_tokens": 550_001,
        },
    )
    seen = await _run(mw, _Req([*_history(0), tail_msg]))
    # ~550k total is under the 0.9M window, so the cache must serve it -- not step
    # aside to BASE as the double-counted 500k+550k would have forced.
    assert seen["model"] != "BASE"


async def test_prefix_size_corrected_from_reported_cache_read():
    # The serve guard's char estimate is replaced by the provider's exact
    # cache_read after a cached call, so a token-dense prefix the estimate
    # under-counted is bounded accurately on later, larger tails.
    from langchain.agents.middleware.types import ModelResponse

    mw, _ = _mw(tools_tokens=10_000)  # estimate ~10k
    st = mw._states  # populated on first call below

    async def handler(r):
        msg = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 700_000,
                "output_tokens": 1,
                "total_tokens": 700_001,
                "input_token_details": {"cache_read": 700_000},  # true prefix size
            },
        )
        return ModelResponse(result=[msg])

    await mw.awrap_model_call(_Req(_history(0)), handler)
    assert st["t1"].prefix_tokens == 700_000  # corrected from the estimate


# ── LRU eviction ─────────────────────────────────────────────────────────────


async def test_lru_eviction_deletes_the_evicted_cache(monkeypatch):
    from airc_core import agent

    monkeypatch.setattr(agent, "_GROWING_MAX_STATES", 2)
    mw, state = _mw()
    for thread in ("A", "B", "C"):  # third insertion evicts the LRU (A)
        await _run(mw, _Req(_history(0), thread=thread))
    assert "A" not in mw._states
    assert state["created"][0][0] in state["deleted"]  # A's cache was deleted


# ── end-to-end through a real create_agent graph ─────────────────────────────


class _ScriptModel(BaseChatModel):
    shared: dict
    label: str

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "script"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.shared["received"].append((self.label, list(messages)))
        i = self.shared["i"]
        msg = self.shared["script"][i]
        self.shared["i"] = min(i + 1, len(self.shared["script"]) - 1)
        return ChatResult(generations=[ChatGeneration(message=msg)])


async def test_end_to_end_graph_grows_cache_and_sends_tail():
    from airc_core.agent import CallBudgetMiddleware, base_middleware
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware
    from langchain_core.tools import tool

    @tool
    def look(x: str) -> str:
        """Look something up."""
        # Large enough that the growing history repays a re-cache; with a thin
        # result the payback rule correctly declines to grow the boundary and
        # the cache stays at gen 1 (covered by test_thin_growth_does_not_recache).
        return f"result for {x}: " + "y" * 40_000

    script = [
        AIMessage(
            content="",
            tool_calls=[{"name": "look", "args": {"x": str(k)}, "id": f"c{k}"}],
        )
        for k in range(6)
    ] + [AIMessage(content="NO_MAJOR_ISSUES")]
    shared = {"script": script, "i": 0, "received": []}
    base = _ScriptModel(shared=shared, label="base")
    cached = _ScriptModel(shared=shared, label="cached")

    state = {"created": [], "deleted": []}

    async def create(prefix):
        name = f"cache{len(state['created']) + 1}"
        state["created"].append(name)
        return name

    async def delete(name):
        state["deleted"].append(name)

    gc = _GrowingPrefixCache(
        create, delete, lambda n: cached, SystemMessage("SYS"), 5000, max_calls=70
    )
    mw = base_middleware("google_genai:fake", "SYS", [look])
    mw.append(gc)
    mw += [
        CallBudgetMiddleware([(100, "soft"), (200, "hard")]),
        ModelCallLimitMiddleware(run_limit=50, exit_behavior="end"),
    ]
    agent = create_agent(
        base, tools=[look], system_prompt="SYS", middleware=mw
    ).with_config({"recursion_limit": 500})

    result = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})

    assert result["messages"][-1].text == "NO_MAJOR_ISSUES"
    assert state["created"]  # caching engaged in the live graph
    # Once the boundary grows past 0, the cached model receives only the tail
    # (no original HumanMessage). create_agent prepends system_message, so the
    # tail begins at index 1.
    cached_calls = [m for label, m in shared["received"] if label == "cached"]
    assert cached_calls
    grown = [m for m in cached_calls if not any(isinstance(x, HumanMessage) for x in m)]
    assert grown, "expected at least one sub-tail call once the cache grew"
    assert all(isinstance(m[0], SystemMessage) for m in cached_calls)


def test_seed_vertex_cache_globals_never_touches_credentials(monkeypatch):
    # Location and project are seeded; credentials never are. The cached-content
    # client insists on TLS and so cannot use the sandbox's plaintext loopback
    # seam at all -- seeding one here would point a client we do not use at a
    # credential we do not have, and in a box there is none to find.
    from airc_core.agent import _seed_vertex_cache_globals
    from google.cloud.aiplatform import initializer

    sentinel = "ORIGINAL_SENTINEL"
    initializer.global_config._credentials = sentinel

    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-east4")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    assert _seed_vertex_cache_globals() == "us-east4"
    assert initializer.global_config._location == "us-east4"
    assert initializer.global_config._project == "test-project"
    assert initializer.global_config._credentials == sentinel


async def test_growing_cache_fns_create_and_delete_seed_globals(monkeypatch):
    import langchain_google_vertexai as lgv
    from airc_core.agent import _growing_cache_fns
    from google.cloud.aiplatform import initializer
    from vertexai.preview import caching

    captured_create = {}
    captured_delete = {}

    def fake_create_context_cache(model, messages, **kw):
        captured_create["creds"] = initializer.global_config._credentials
        captured_create["location"] = initializer.global_config._location
        captured_create["project"] = initializer.global_config._project
        return "cache-123"

    def fake_cached_content_init(self, name):
        captured_delete["creds"] = initializer.global_config._credentials
        captured_delete["location"] = initializer.global_config._location
        captured_delete["project"] = initializer.global_config._project
        self._name = name

    def fake_cached_content_delete(self):
        captured_delete["deleted"] = getattr(self, "_name", None)

    monkeypatch.setattr(lgv, "create_context_cache", fake_create_context_cache)
    monkeypatch.setattr(caching.CachedContent, "__init__", fake_cached_content_init)
    monkeypatch.setattr(caching.CachedContent, "delete", fake_cached_content_delete)

    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project-123")

    initializer.global_config._credentials = None
    initializer.global_config._location = None
    initializer.global_config._project = None

    create, delete, _, _ = _growing_cache_fns(
        "google_vertexai:gemini-2.5-flash", [], 10
    )

    cache_name = await create([SystemMessage("SYS")])
    assert cache_name == "cache-123"
    assert captured_create["location"] == "us-central1"
    assert captured_create["project"] == "test-project-123"
    assert captured_create["creds"] is None

    await delete("cache-123")
    assert captured_delete["location"] == "us-central1"
    assert captured_delete["project"] == "test-project-123"
    assert captured_delete["creds"] is None

    # Both paths run on ADC, whatever it resolves to: the seeding never supplies
    # a credential of its own, so a None here is the seam staying out of the way
    # rather than a token that failed to mint.


# -- genai-stack cache create/delete ------------------------------------------


async def test_genai_cache_create_builds_config_and_returns_full_name(monkeypatch):
    from airc_core import agent as agent_mod
    from langchain_core.tools import tool

    @tool
    def look(x: str) -> str:
        """Look something up."""
        return x

    captured = {}

    class _Caches:
        async def create(self, *, model, config):
            captured["model"], captured["config"] = model, config
            return type("C", (), {"name": "projects/p/locations/l/cachedContents/42"})()

        async def delete(self, *, name):
            captured["deleted"] = name

    client = type("Cl", (), {"aio": type("A", (), {"caches": _Caches()})()})()
    monkeypatch.setattr(agent_mod, "_genai_client", lambda: client)

    prefix = [
        SYS,
        HumanMessage("hello"),
        AIMessage(content="", tool_calls=[{"name": "look", "args": {}, "id": "c1"}]),
    ]
    name = await agent_mod._genai_cache_create(
        "google_vertexai:gemini-3.8-flash", prefix, [look], ttl_minutes=15
    )
    assert name == "projects/p/locations/l/cachedContents/42"
    assert captured["model"] == "gemini-3.8-flash"
    cfg = captured["config"]
    assert cfg.ttl == "900s"
    assert cfg.system_instruction is not None
    # The prefix ends on the model's function call (the growing-cache shape).
    assert cfg.contents[-1].role == "model"
    assert cfg.contents[-1].parts[-1].function_call.name == "look"
    assert cfg.tools[0].function_declarations[0].name == "look"

    await agent_mod._genai_cache_delete(name)
    # Bare id, not the number-name create returned: the client rebuilds the
    # path under the configured project NAME, which is what the sandbox
    # proxy's allowlist is anchored to.
    assert captured["deleted"] == "42"


# ── serve-time cache rejection ───────────────────────────────────────────────


async def _serve_then(mw, msgs, cached_exc, uncached_exc=None):
    """Drive one cached call whose cached attempt raises `cached_exc`, recording
    whether the uncached resend happened and what it was sent."""
    seen = {"cached": 0, "uncached": 0, "messages": None}

    async def handler(r):
        if r.model != "BASE":  # the cache-bound model
            seen["cached"] += 1
            raise cached_exc
        seen["uncached"] += 1
        seen["messages"] = r.messages
        if uncached_exc is not None:
            raise uncached_exc
        return "ok"

    return seen, await mw.awrap_model_call(_Req(msgs), handler)


async def test_serve_time_rejection_falls_back_and_drops_the_cache():
    # A model rejecting the cached prefix on its shape says nothing about the
    # cache: the turn must continue uncached instead of dying.
    mw, state = _mw()
    await mw.awrap_model_call(_Req(_history(0)), _noop)  # cache gen c1
    msgs = _fat_history(4)
    rejection = ValueError("INVALID_ARGUMENT: Requests ending with a model turn")
    seen, resp = await _serve_then(mw, msgs, rejection)

    assert (seen["cached"], seen["uncached"]) == (1, 1)
    assert seen["messages"] == msgs  # resent whole, not as a tail
    assert resp == "ok"
    assert state["created"][-1][0] in state["deleted"]  # rejected gen deleted
    assert mw._states[_Req(msgs).runtime.execution_info.thread_id].name is None
    # Backed off, so the next call does not immediately rebuild the same shape.
    assert mw._cooldown_until > 0


async def test_serve_time_rejection_surfaces_a_genuine_request_error():
    # The uncached resend failing too proves the request was at fault, not the
    # prefix: the original error stands and the cache is kept.
    mw, state = _mw()
    await mw.awrap_model_call(_Req(_history(0)), _noop)
    msgs = _fat_history(4)
    original = ValueError("400 missing a thought signature")
    seen, exc = None, None
    try:
        seen, _ = await _serve_then(mw, msgs, original, uncached_exc=RuntimeError("x"))
    except Exception as e:
        exc = e
    assert exc is original
    assert seen is None
    assert state["deleted"] == ["c1"]  # only the supersession, no rejection drop


async def test_transient_error_on_a_cached_call_is_not_probed_uncached():
    # An overload is the retry middleware's to handle; resending the whole
    # prompt uncached is the wrong answer to it.
    mw, _ = _mw()
    await mw.awrap_model_call(_Req(_history(0)), _noop)
    msgs = _fat_history(4)
    overload = type("Overloaded", (Exception,), {"code": 503})("503 unavailable")
    seen = {"uncached": 0}

    async def handler(r):
        if r.model == "BASE":
            seen["uncached"] += 1
            return "ok"
        raise overload

    with pytest.raises(Exception) as ei:
        await mw.awrap_model_call(_Req(msgs), handler)
    assert ei.value is overload
    assert seen["uncached"] == 0

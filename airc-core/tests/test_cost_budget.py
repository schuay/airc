# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""BudgetMiddleware: a turn bounded by what it costs, not by how many
calls it makes.

Driven through a real create_agent graph on a scripted model, because the three
things that matter are all wiring: that the accumulator sees one response per
call, that the turn actually stops, and that a read issued inside the closing
window never reaches the tool.
"""

import math

import pytest
from airc_core.agent import (
    _ZERO_CACHE_FLOOR,
    BUDGET_KEY,
    CALLS_LEFT_KEY,
    BudgetMiddleware,
    _Spend,
)
from airc_core.usage import Usage
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# claude-opus-5 is listed at $5/M input, so a 1M-token uncached prompt is $5 a
# call -- a real price off the real table, so the arithmetic below is the
# arithmetic prod does.
_MODEL = "claude-opus-5"
_PER_CALL_USD = 5.0


def _usage(input_tokens: int = 1_000_000, output_tokens: int = 0, cache_read: int = 0):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_token_details": {"cache_read": cache_read},
    }


class _ScriptedModel(BaseChatModel):
    """Calls `read` forever, reporting the same usage on every call, so a turn
    ends only when a governor ends it."""

    calls: int = 0
    usage: dict = {}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        object.__setattr__(self, "calls", self.calls + 1)
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "read", "args": {"x": "f"}, "id": f"c{self.calls}"}],
            usage_metadata=self.usage or _usage(),
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _agent(middleware, usage=None, recursion_limit=200):
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def read(x: str) -> str:
        """Read something."""
        return "contents"

    model = _ScriptedModel(usage=usage or _usage())
    graph = create_agent(
        model, tools=[read], system_prompt="sys", middleware=middleware
    ).with_config({"recursion_limit": recursion_limit})
    return model, graph


# The tripwire is off unless a test asks for it: every other test here scripts
# a model that reports no cache read at all, which is exactly what it watches
# for, and a cost test that ended on the tripwire would assert nothing about
# cost. The tripwire tests pass the real floor.
_TRIPWIRE_OFF = 10**12


def _budget(
    cost_limit=25.0,
    context_target=400_000,
    window=3.0,
    zero_cache_floor=_TRIPWIRE_OFF,
):
    return BudgetMiddleware(
        _MODEL,
        context_target=context_target,
        cost_limit=cost_limit,
        notice="NOTICE: tools are closed, report now",
        refusal="REFUSED: reads are closed",
        window=window,
        zero_cache_floor=zero_cache_floor,
    )


async def test_the_limit_ends_the_turn_one_call_late_at_most():
    """Five calls at $5 reaches a $25 limit; the sixth never happens. The call
    that crosses the limit has already been paid for, which is the accepted
    overshoot -- the bound is 'one call late', not 'never over'."""
    model, graph = _agent([_budget(cost_limit=25.0)])
    state = await graph.ainvoke({"messages": [HumanMessage("go")]})
    assert model.calls == 5
    assert state[BUDGET_KEY].usage.usd == pytest.approx(25.0)
    assert "Cost limit reached" in state["messages"][-1].content


async def test_without_the_limit_the_same_turn_runs_on():
    """The control for the test above: the scripted model never stops on its
    own, so a turn that ends at five calls ended because the limit ended it.
    Without a governor the graph runs to its recursion backstop."""
    from langgraph.errors import GraphRecursionError

    model, graph = _agent([], recursion_limit=25)
    with pytest.raises(GraphRecursionError):
        await graph.ainvoke({"messages": [HumanMessage("go")]})
    assert model.calls > 5


async def test_a_cheaper_model_buys_more_calls_at_the_same_limit():
    """Why the bound is priced, not counted: the same $25
    limit is 5 calls at a 1M-token context and 50 at a 100k one."""
    model, graph = _agent(
        [_budget(cost_limit=25.0)], usage=_usage(100_000), recursion_limit=400
    )
    await graph.ainvoke({"messages": [HumanMessage("go")]})
    assert model.calls == 50


async def test_the_turn_usage_is_the_sum_of_its_priced_calls():
    """One Usage per call, from the response just appended, priced through the
    same Usage.of_call the collector books with -- never recomputed from the
    message list, which summarization strips the usage out of."""
    model, graph = _agent([_budget(cost_limit=25.0)])
    state = await graph.ainvoke({"messages": [HumanMessage("go")]})
    total = state[BUDGET_KEY].usage
    assert total.calls == model.calls == 5
    assert total == sum((Usage.of_call(_usage(), _MODEL) for _ in range(5)), Usage())


async def test_reads_are_refused_once_the_money_runs_low():
    """The closing window, keyed to the limit: at $5 a call a $25 limit closes
    the tools with three calls' worth left, so the last calls have nothing to do
    but report. The refusal is an ordinary tool result at the tail, so the
    cached prefix survives it."""
    _, graph = _agent([_budget(cost_limit=25.0, window=3.0)])
    state = await graph.ainvoke({"messages": [HumanMessage("go")]})
    refused = [
        m
        for m in state["messages"]
        if isinstance(m, ToolMessage) and "REFUSED" in str(m.content)
    ]
    executed = [
        m
        for m in state["messages"]
        if isinstance(m, ToolMessage) and str(m.content) == "contents"
    ]
    # Two calls run before $10 remains; the rest are refused.
    assert len(executed) == 2
    assert len(refused) == 3


async def test_a_read_is_only_refused_by_a_call_that_was_told_so():
    """The alignment FinalAnswerMiddleware gets by subtracting one from its
    counter: the notice is appended to the request, so a call made before the window
    closed must still have its reads executed -- otherwise the model is refused
    for a rule it was never shown."""
    mw = _budget(cost_limit=25.0)
    # The latch has just been set by the call now issuing tool calls; the
    # request that call was built from did not carry the notice.
    just_closed = _Spend(calls_left=2.0, closed=True, closed_prev=False)
    assert mw._refuse(_ToolReq(just_closed)) is None
    already_closed = _Spend(calls_left=1.0, closed=True, closed_prev=True)
    assert mw._refuse(_ToolReq(already_closed)) is not None


class _ToolReq:
    def __init__(self, spend):
        self.state = {BUDGET_KEY: spend}
        self.tool_call = {"name": "read", "args": {}, "id": "c1"}


async def test_an_open_call_gets_nothing_appended_and_a_closed_one_the_notice():
    """No running spend line on the open calls: a figure that climbs on every
    call reads as a clock and had the model wrapping up early."""
    seen = []

    async def handler(req):
        seen.append([str(m.content) for m in req.messages])
        return "ok"

    mw = _budget(cost_limit=25.0)
    await mw.awrap_model_call(_ModelReq(_Spend(usage=Usage(usd=5.0))), handler)
    await mw.awrap_model_call(
        _ModelReq(_Spend(usage=Usage(usd=20.0), closed=True)), handler
    )
    assert seen[0] == ["hi"]
    assert seen[1][0] == "hi" and len(seen[1]) == 2
    assert "NOTICE" in seen[1][1]


class _ModelReq:
    def __init__(self, spend, messages=None):
        self.state = {BUDGET_KEY: spend}
        self.messages = messages or [HumanMessage("hi")]

    def override(self, *, messages=None, model=None):
        return _ModelReq(self.state[BUDGET_KEY], messages or self.messages)


async def test_calls_left_is_remaining_dollars_over_the_last_call():
    """What the two cache brakes read. A count would mean different things at a
    20k context and a 600k one; dollars over the last call's cost is the same
    question ("how many more like this one") in the unit that ends the turn."""
    mw = _budget(cost_limit=25.0)
    out = mw.after_model(
        {
            "messages": [AIMessage(content="", usage_metadata=_usage())],
            BUDGET_KEY: _Spend(usage=Usage(usd=10.0, calls=2)),
        },
        None,
    )
    # $15 left, $5 a call.
    assert out[CALLS_LEFT_KEY] == pytest.approx(2.0)
    assert out[BUDGET_KEY].usage.usd == pytest.approx(15.0)


async def test_a_call_the_provider_priced_at_nothing_leaves_the_brakes_alone():
    """No price to divide by is not "no calls left": returning 0 would slam
    every brake shut on a missing number."""
    mw = _budget(cost_limit=25.0)
    out = mw.after_model(
        {"messages": [AIMessage(content="", usage_metadata=None)]}, None
    )
    assert out[CALLS_LEFT_KEY] == math.inf


async def test_the_brakes_prefer_the_published_calls_left():
    """Both cache brakes fall back to `cap - model_calls` for a stack with no
    budget middleware (the room, the harness), and use the published number
    when there is one."""
    from types import SimpleNamespace

    from airc_core.agent import _AnthropicVertexCaching, _GrowingPrefixCache

    anthropic = _AnthropicVertexCaching(max_calls=150)
    assert anthropic._calls_left(SimpleNamespace(state={"model_calls": 10})) == 140
    assert (
        anthropic._calls_left(
            SimpleNamespace(state={"model_calls": 10, CALLS_LEFT_KEY: 1.5})
        )
        == 1.5
    )
    growing = _GrowingPrefixCache.__new__(_GrowingPrefixCache)
    growing._max_calls = 150
    assert growing._calls_left(SimpleNamespace(state={"model_calls": 10})) == 140
    assert (
        growing._calls_left(
            SimpleNamespace(state={"model_calls": 10, CALLS_LEFT_KEY: 0.5})
        )
        == 0.5
    )


async def _run_calls(mw, shapes):
    """Feed `shapes` (usage dicts) through after_model one at a time, carrying
    the accumulator forward the way the graph does."""
    state = {}
    for shape in shapes:
        state["messages"] = [AIMessage(content="", usage_metadata=shape)]
        state.update(mw.after_model(state, None))
    return state[BUDGET_KEY]


async def test_three_large_uncached_calls_in_a_row_end_the_pass():
    """The symptom: a large prompt served with nothing read back is a
    provider-side eviction, and a run of them means every call is billed at the
    full input rate for work the cache was paying for."""
    from airc_core.agent import CacheLossTrip

    mw = _budget(cost_limit=1000.0, zero_cache_floor=_ZERO_CACHE_FLOOR)
    with pytest.raises(CacheLossTrip) as e:
        await _run_calls(mw, [_usage(400_000)] * 3)
    # The counts, so the log line says what tripped, not just that something did.
    assert "3 calls in a row" in str(e.value) and "400k" in str(e.value)


async def test_the_trip_is_not_the_cost_limit_ending():
    """Two endings, told apart by type. A caller that stops the service on one
    must not stop it on the other: a pass that spent its budget is a pass that
    worked, and reporting a provider fault as "out of budget" hides it."""
    from airc_core.agent import CacheLossTrip

    mw = _budget(cost_limit=3.0, zero_cache_floor=_ZERO_CACHE_FLOOR)
    # Under a dollar a call with healthy cache reads: the limit ends it after
    # four, and nothing raises on the way.
    spend = await _run_calls(mw, [_usage(1_000_000, cache_read=900_000)] * 4)
    assert spend.zero_cache_run == 0
    assert spend.usage.usd > 3.0
    assert mw.before_model({BUDGET_KEY: spend}, None)["jump_to"] == "end"
    assert CacheLossTrip is not None  # the other ending has its own type


async def test_a_single_cold_call_is_not_a_trip():
    """One zero after a retry is a cold replica, and a re-route that warms again
    on the next call must not stop the fleet. The run has to be consecutive."""
    mw = _budget(cost_limit=1000.0, zero_cache_floor=_ZERO_CACHE_FLOOR)
    spend = await _run_calls(
        mw,
        [
            _usage(400_000),
            _usage(400_000),
            _usage(400_000, cache_read=300_000),  # warm again: run resets
            _usage(400_000),
            _usage(400_000),
        ],
    )
    assert spend.zero_cache_run == 2


async def test_small_calls_are_never_judged():
    """Below the floor a zero says nothing: a short prompt has no cache worth
    serving, and every pass starts with one."""
    mw = _budget(cost_limit=1000.0, zero_cache_floor=_ZERO_CACHE_FLOOR)
    spend = await _run_calls(mw, [_usage(10_000)] * 10)
    assert spend.zero_cache_run == 0


async def test_the_trip_reaches_the_caller_through_the_graph():
    """Raised, not ended with a jump: a turn that ends returns no verdict,
    which every caller reads as "the model had nothing to report"."""
    from airc_core.agent import CacheLossTrip

    _, graph = _agent(
        [_budget(cost_limit=1000.0, zero_cache_floor=_ZERO_CACHE_FLOOR)],
        usage=_usage(400_000),
    )
    with pytest.raises(CacheLossTrip):
        await graph.ainvoke({"messages": [HumanMessage("go")]})


async def test_the_trip_is_never_read_as_a_transient():
    """_gather_passes contains a transient to one pass and re-raises everything
    else; _is_transient's text fallback is a substring match, so the message has
    to stay clear of its vocabulary."""
    from airc_core.agent import CacheLossTrip, _is_transient

    mw = _budget(cost_limit=1000.0, zero_cache_floor=_ZERO_CACHE_FLOOR)
    with pytest.raises(CacheLossTrip) as e:
        await _run_calls(mw, [_usage(400_000)] * 3)
    assert not _is_transient(e.value)


# ── the two numbers, and what their degenerate values mean ───────────────────


def test_a_negative_cost_limit_is_a_typo_not_an_unset_one():
    with pytest.raises(ValueError, match="cost_limit must be zero"):
        _budget(cost_limit=-1.0)


async def test_no_cost_limit_never_ends_the_turn_on_spend():
    """0 is no ceiling, for a deployment whose model the price table cannot
    price: a bound there would be a bound on the generic placeholder rate, a
    dollar figure nobody chose. The turn then ends where the caller's own
    backstops put it, not where the money runs out."""
    mw = _budget(cost_limit=0.0)
    spend = await _run_calls(mw, [_usage(400_000)] * 20)
    # Well past the $25 every other test here is bounded by, and still running.
    assert spend.usage.usd > 25.0
    assert mw.before_model({BUDGET_KEY: spend}, None) is None
    # The same spend against a ceiling ends the turn, so it is the limit doing
    # the ending and not the absence of anything to spend.
    assert _budget(cost_limit=25.0).before_model({BUDGET_KEY: spend}, None) is not None


async def test_no_cost_limit_leaves_the_reads_open():
    """The window closes reads once what REMAINS is a few calls' worth. With
    nothing remaining to run down there is no point at which that is true, so a
    pass keeps its tools to the end instead of losing them to a division that
    never had a denominator."""
    mw = _budget(cost_limit=0.0)
    spend = await _run_calls(mw, [_usage(400_000)] * 10)
    assert spend.calls_left == math.inf and not spend.closed


def test_a_negative_context_target_is_a_typo_not_an_unset_one():
    with pytest.raises(ValueError, match="context_target must be zero"):
        _budget(context_target=-1)

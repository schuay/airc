# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""BudgetMiddleware: a turn bounded by what it costs rather than by how many
calls it makes.

Driven through a real create_agent graph on a scripted model, because the three
things that matter are all wiring: that the accumulator sees one response per
call, that the turn actually stops, and that a read issued inside the closing
window never reaches the tool.
"""

import math

import pytest
from airc_core.agent import (
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


def _budget(cost_limit=25.0, context_target=400_000, window=3.0):
    return BudgetMiddleware(
        _MODEL,
        context_target=context_target,
        cost_limit=cost_limit,
        notice="NOTICE: tools are closed, report now",
        refusal="REFUSED: reads are closed",
        window=window,
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
    """The point of pricing the bound rather than counting it: the same $25
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
    counter: the notice rides the request, so a call made before the window
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


async def test_the_pointer_rides_every_call_and_the_notice_only_the_closed_ones():
    seen = []

    async def handler(req):
        seen.append([str(m.content) for m in req.messages])
        return "ok"

    mw = _budget(cost_limit=25.0)
    await mw.awrap_model_call(_ModelReq(_Spend(usage=Usage(usd=5.0))), handler)
    await mw.awrap_model_call(
        _ModelReq(_Spend(usage=Usage(usd=20.0), closed=True)), handler
    )
    assert any("$5.00 spent" in m for m in seen[0])
    assert not any("NOTICE" in m for m in seen[0])
    assert any("$20.00 spent" in m for m in seen[1])
    assert any("NOTICE" in m for m in seen[1])


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

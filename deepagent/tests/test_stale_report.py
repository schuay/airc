# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""A resumed turn must not inherit the prior turn's structured report.

structured_response is a checkpointed channel with no reducer that clears it
between turns. A resume turn that ends without calling the report tool (the model
acknowledges the checkpoint and stops -- no tool call, so the graph takes the
classic 0-tool-calls exit) never overwrites the channel, so run_once would read
the PRIOR turn's report back as this turn's verdict -- a valid CONTINUE the loop
resets its dead-turn cap on, spinning to max_iters doing nothing. These tests
lock the framework behavior the harness fix rests on: the stale read reproduces,
and clearing the channel before the resume invoke removes it.
"""

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from deepagent import REPORT_TOOL_NAME, Report


class _StageReport(Report):
    pass


class _Scripted(GenericFakeChatModel):
    """Turn 1 calls the report tool; every later turn answers with plain text and
    no tool call -- the ghost-turn shape that leaves structured_response stale."""

    def __init__(self):
        super().__init__(messages=iter([]))
        self._calls = 0

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self._calls += 1
        if self._calls == 1:
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": REPORT_TOOL_NAME,
                        "args": {"disposition": "continue", "summary": "turn1"},
                        "id": "call_1",
                    }
                ],
            )
        else:
            msg = AIMessage(content="acknowledged, nothing more to do")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kw):
        return self


def _graph():
    strat = ToolStrategy(schema=_StageReport, handle_errors=True)
    strat.schema_specs[0].name = REPORT_TOOL_NAME
    return create_agent(
        _Scripted(), tools=[], response_format=strat, checkpointer=InMemorySaver()
    ).with_config({"recursion_limit": 100})


async def test_ghost_report_reproduces_without_clear(tmp_path):
    graph = _graph()
    cfg = {"configurable": {"thread_id": "t"}}
    s1 = await graph.ainvoke({"messages": [{"role": "user", "content": "task"}]}, cfg)
    assert isinstance(s1.get("structured_response"), Report)
    # Resume turn makes no tool call; the stale report is still in the channel.
    s2 = await graph.ainvoke({"messages": [{"role": "user", "content": "go"}]}, cfg)
    assert s2.get("structured_response").summary == "turn1"  # the ghost


async def test_clearing_channel_drops_the_ghost(tmp_path):
    graph = _graph()
    cfg = {"configurable": {"thread_id": "t"}}
    await graph.ainvoke({"messages": [{"role": "user", "content": "task"}]}, cfg)
    # The harness fix: blank the channel before the resume invoke. structured_
    # response is OmitFromInput, so it cannot be cleared via ainvoke input -- the
    # state update is the only path.
    await graph.aupdate_state(cfg, {"structured_response": None})
    s2 = await graph.ainvoke({"messages": [{"role": "user", "content": "go"}]}, cfg)
    assert s2.get("structured_response") is None  # no fresh report -> None, not stale

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""A turn that dies mid-tool-call leaves a thread the next turn can continue.

When the per-turn wall clock cancels `ainvoke` while a tool is running, the
checkpoint ends in an AIMessage whose tool_calls have no ToolMessage. Providers
reject that history (a tool_use without a tool_result in the next message is a
400), so a resume that only appends the continue prompt fails on every retry
until the loop's dead-turn cap abandons the job. run_once has to close the
pending calls with a synthetic result before resuming.
"""

import asyncio
from pathlib import Path

from airc_core import CommonConfig
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from deepagent import REPORT_TOOL_NAME, LangGraphHarness, Report


def _assert_tool_calls_answered(messages) -> None:
    """The provider-side rule: every tool_call id in an AIMessage has a
    ToolMessage with that id in the messages that directly follow it."""
    for i, m in enumerate(messages):
        if not isinstance(m, AIMessage) or not m.tool_calls:
            continue
        following = set()
        for n in messages[i + 1 :]:
            if not isinstance(n, ToolMessage):
                break
            following.add(n.tool_call_id)
        missing = {tc["id"] for tc in m.tool_calls} - following
        assert not missing, f"tool_use without tool_result: {missing}"


class _Scripted(GenericFakeChatModel):
    """First call: invoke the slow tool. Every later call: report complete,
    after checking the history is one a provider would accept."""

    def __init__(self):
        super().__init__(messages=iter([]))
        self._calls: list[list] = []

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self._calls.append(list(messages))
        _assert_tool_calls_answered(messages)
        if len(self._calls) == 1:
            msg = AIMessage(
                content="",
                tool_calls=[{"name": "slow", "args": {}, "id": "call_slow"}],
            )
        else:
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": REPORT_TOOL_NAME,
                        "args": {"disposition": "complete", "summary": "done"},
                        "id": "call_report",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kw):
        return self

    @property
    def calls(self) -> list[list]:
        return self._calls


@tool
async def slow() -> str:
    """Takes longer than the turn allows."""
    await asyncio.sleep(30)
    return "never"


def _harness(tmp_path: Path, model) -> LangGraphHarness:
    from langgraph.checkpoint.memory import InMemorySaver

    common = CommonConfig()
    common.models = {"default": "google_genai:gemini-3.1-flash-lite"}
    common.token_db_path = tmp_path / "tokens.db"
    h = LangGraphHarness(common)

    async def _no_init() -> None:
        return None

    h._ensure_init = _no_init
    strat = ToolStrategy(schema=Report, handle_errors=True)
    strat.schema_specs[0].name = REPORT_TOOL_NAME
    graph = create_agent(
        model, tools=[slow], response_format=strat, checkpointer=InMemorySaver()
    ).with_config({"recursion_limit": 100})
    h._graph_for = lambda *a, **k: graph
    return h


async def _turn(h, tmp_path: Path, *, resume: bool, n: int, timeout_s: float):
    prompt = tmp_path / "JOB.md"
    prompt.write_text("the job")
    return await h.run_once(
        prompt_path=prompt,
        workdir=tmp_path,
        result_path=tmp_path / f"result.{n}.json",
        timeout_s=timeout_s,
        agent="draft",
        resume=resume,
    )


async def test_a_resume_after_a_mid_tool_timeout_has_a_valid_history(tmp_path):
    m = _Scripted()
    h = _harness(tmp_path, m)

    run0 = await _turn(h, tmp_path, resume=False, n=0, timeout_s=0.5)
    assert run0.result is None and run0.exit_code == -1  # timed out in `slow`

    run1 = await _turn(h, tmp_path, resume=True, n=1, timeout_s=30)
    assert run1.result is not None and run1.result.summary == "done"

    # The resumed model call saw the interrupted call closed with a result that
    # says so, directly after the AIMessage that made it, and then the continue
    # prompt.
    history = m.calls[-1]
    idx = next(
        i
        for i, msg in enumerate(history)
        if isinstance(msg, AIMessage) and msg.tool_calls
    )
    closed = history[idx + 1]
    assert isinstance(closed, ToolMessage)
    assert closed.tool_call_id == "call_slow"
    assert closed.status == "error"
    assert "interrupted" in str(closed.content)
    assert "Continue from where you left off" in str(history[-1].content)


async def test_a_resume_with_a_complete_history_adds_nothing(tmp_path):
    # The other side: a thread whose last tool call was answered must not get a
    # spurious "interrupted" result on resume.
    m = _Scripted()
    h = _harness(tmp_path, m)
    await _turn(h, tmp_path, resume=False, n=0, timeout_s=0.5)
    await _turn(h, tmp_path, resume=True, n=1, timeout_s=30)
    await _turn(h, tmp_path, resume=True, n=2, timeout_s=30)
    history = m.calls[-1]
    interrupted = [
        msg
        for msg in history
        if isinstance(msg, ToolMessage) and "interrupted" in str(msg.content)
    ]
    assert len(interrupted) == 1  # only the one from turn 0
    assert interrupted[0].tool_call_id == "call_slow"

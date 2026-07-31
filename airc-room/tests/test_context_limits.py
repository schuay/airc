# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Tool-result caps and old-turn pruning (token blowup defenses)."""

import asyncio
import logging
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tools import StructuredTool

from airc_core.mcptools import (
    _MAX_TOOL_RESULT_CHARS,
    _fix_tool,
    _result_chars,
    _truncated,
)
from airc_core.agent import (
    _CallTrace,
    _ELIDED_TOOL_RESULT,
    _MAX_KEPT_RESULT_CHARS,
    _estimate_input_tokens,
    compact_for_budget,
    prune_to_recent_tool_results,
    truncate_oversized_tool_results,
)


def test_tool_results_are_capped():
    async def big_output() -> str:
        return "x" * (_MAX_TOOL_RESULT_CHARS + 10_000)

    tool = StructuredTool.from_function(
        coroutine=big_output, name="big", description="d"
    )
    _fix_tool(tool)
    out = asyncio.run(tool.coroutine())
    assert len(out) < _MAX_TOOL_RESULT_CHARS + 200
    assert "output truncated (10000 more chars)" in out


def test_small_tool_results_untouched():
    async def small_output() -> str:
        return "fine"

    tool = StructuredTool.from_function(
        coroutine=small_output, name="s", description="d"
    )
    _fix_tool(tool)
    assert asyncio.run(tool.coroutine()) == "fine"


def test_tool_result_size_is_logged(caplog):
    async def output() -> str:
        return "y" * 1234

    tool = StructuredTool.from_function(
        coroutine=output, name="big_dump", description="d"
    )
    _fix_tool(tool)
    with caplog.at_level(logging.INFO, logger="airc_core.mcptools"):
        asyncio.run(tool.coroutine())
    assert "tool big_dump result: 1234 chars" in caplog.text


def _content_blocks(text: str) -> list[dict]:
    """The shape the MCP adapter returns: content_and_artifact content is a list
    of {"type": "text", "text": ...} blocks, never a bare string."""
    return [{"type": "text", "text": text, "id": "lc_x"}]


def test_mcp_list_content_result_is_capped():
    # The real MCP path: a (content, artifact) tuple whose content is a list of
    # content blocks. A bare isinstance(str) check let this through uncapped.
    big = "x" * (_MAX_TOOL_RESULT_CHARS + 10_000)

    async def big_output():
        return _content_blocks(big), {"structured": True}

    tool = StructuredTool.from_function(
        coroutine=big_output,
        name="grep",
        description="d",
        response_format="content_and_artifact",
    )
    _fix_tool(tool)
    content, artifact = asyncio.run(tool.coroutine())
    assert _result_chars((content, artifact)) < _MAX_TOOL_RESULT_CHARS + 200
    assert "output truncated (10000 more chars)" in content[0]["text"]
    assert artifact == {"structured": True}  # artifact passes through untouched


def test_mcp_list_content_size_is_logged(caplog):
    async def output():
        return _content_blocks("y" * 1234), None

    tool = StructuredTool.from_function(
        coroutine=output,
        name="grep",
        description="d",
        response_format="content_and_artifact",
    )
    _fix_tool(tool)
    with caplog.at_level(logging.INFO, logger="airc_core.mcptools"):
        asyncio.run(tool.coroutine())
    assert "tool grep result: 1234 chars" in caplog.text


def test_truncated_caps_combined_text_across_blocks():
    blocks = [
        {"type": "text", "text": "a" * (_MAX_TOOL_RESULT_CHARS - 5)},
        {"type": "image", "url": "u"},  # non-text block left intact
        {"type": "text", "text": "b" * 1000},
    ]
    out = _truncated(blocks)
    total = sum(len(b["text"]) for b in out if "text" in b)
    assert total <= _MAX_TOOL_RESULT_CHARS + 200
    assert out[1] == {"type": "image", "url": "u"}


def _llm_result(input_tokens: int) -> LLMResult:
    msg = AIMessage(
        "ok",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": 7,
            "total_tokens": input_tokens + 7,
            "input_token_details": {"cache_read": input_tokens // 2},
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def test_call_trace_tracks_calls_and_max_input(caplog):
    trace = _CallTrace("perf", "turn")
    for n, in_tok in enumerate((8000, 60_000, 240_000)):
        rid = uuid4()
        msgs = [HumanMessage("q")] + [
            ToolMessage("Z" * 100, tool_call_id=f"c{i}") for i in range(n + 1)
        ]
        trace.on_chat_model_start({}, [msgs], run_id=rid)
        with caplog.at_level(logging.INFO, logger="airc_core.agent"):
            trace.on_llm_end(_llm_result(in_tok), run_id=rid)
    assert trace.calls == 3
    assert trace.max_input_tokens == 240_000
    # The per-call line carries the growing input and the request's tool-result
    # count, so the within-turn accumulation is legible in the log.
    assert "call perf/turn #3: 240000 in (120000 cached, 50%)" in caplog.text
    assert "3 tool results (300 chars)" in caplog.text
    # The turn summary derives the hit rate and the uncached (full-price) tokens.
    assert trace.summary()["hit_pct"] == 50
    assert trace.summary()["uncached"] == (8000 + 60_000 + 240_000) // 2


def _long_turn(n_tools, payload_chars=400):
    """One turn (single HumanMessage) with n_tools tool calls -- the within-turn
    blowup that the turn-based pruner cannot age out."""
    msgs = [HumanMessage("q")]
    for i in range(n_tools):
        msgs.append(
            AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": f"c{i}"}])
        )
        msgs.append(ToolMessage("Z" * payload_chars, tool_call_id=f"c{i}"))
    msgs.append(AIMessage("final"))
    return msgs


def test_prune_to_recent_keeps_last_k():
    msgs = _long_turn(5)
    pruned = prune_to_recent_tool_results(msgs, keep=2)
    assert pruned is not None
    tool_msgs = [m for m in pruned if isinstance(m, ToolMessage)]
    assert [m.content for m in tool_msgs[:3]] == [_ELIDED_TOOL_RESULT] * 3
    assert all(m.content == "Z" * 400 for m in tool_msgs[3:])
    # Source list untouched.
    assert isinstance(msgs[2].content, str) and msgs[2].content == "Z" * 400


def test_prune_to_recent_noop_when_within_keep():
    assert prune_to_recent_tool_results(_long_turn(2), keep=3) is None


def test_estimate_uses_usage_floor():
    # Tiny by char count, but the previous call reported a large prompt; the
    # provider-exact count is a floor the char heuristic cannot undercut.
    usage = {"input_tokens": 9000, "output_tokens": 5, "total_tokens": 9005}
    msgs = [HumanMessage("hi"), AIMessage("ok", usage_metadata=usage)]
    assert _estimate_input_tokens(msgs) == 9000


def test_compact_under_budget_is_noop():
    msgs = _long_turn(5)  # ~512 est tokens
    out, drop_tools = compact_for_budget(msgs, window=100_000)
    assert out is msgs and drop_tools is False


def test_compact_below_hard_is_intact_no_soft_shed():
    # ~501 est tokens vs a 600-token window (hard line 540). The old 450 soft
    # threshold would have shed here; with caching we keep everything intact
    # until the hard line, so the prefix cache is never poisoned.
    msgs = _long_turn(5)
    out, drop_tools = compact_for_budget(msgs, window=600)
    assert out is msgs and drop_tools is False


def test_compact_hard_drops_tools_and_keeps_last():
    msgs = _long_turn(5)  # est ~512 tokens; window 550 -> hard 495
    out, drop_tools = compact_for_budget(msgs, window=550)
    assert drop_tools is True
    tool_msgs = [m for m in out if isinstance(m, ToolMessage)]
    assert [m.content for m in tool_msgs[:4]] == [_ELIDED_TOOL_RESULT] * 4
    assert tool_msgs[4].content == "Z" * 400


def test_truncate_oversized_tool_results():
    big = "Z" * (_MAX_KEPT_RESULT_CHARS + 50_000)
    msgs = [HumanMessage("q"), ToolMessage(big, tool_call_id="c1")]
    out = truncate_oversized_tool_results(msgs, _MAX_KEPT_RESULT_CHARS)
    assert out is not None
    assert len(str(out[1].content)) < _MAX_KEPT_RESULT_CHARS + 200
    assert "truncated to fit the context window" in out[1].content
    # Source list untouched (checkpoint must not be mutated).
    assert msgs[1].content == big


def test_truncate_oversized_noop_when_within_cap():
    msgs = [HumanMessage("q"), ToolMessage("Z" * 400, tool_call_id="c1")]
    assert truncate_oversized_tool_results(msgs, _MAX_KEPT_RESULT_CHARS) is None


def test_compact_hard_truncates_unsheddable_single_result():
    # One result larger than the window: keep-one cannot shed it, so without the
    # backstop the request stays over budget and 400s on every retry. The hard
    # branch must truncate it and signal drop_tools.
    payload = "Z" * (_MAX_KEPT_RESULT_CHARS + 200_000)
    msgs = [
        HumanMessage("q"),
        AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": "c0"}]),
        ToolMessage(payload, tool_call_id="c0"),
        AIMessage("final"),
    ]
    out, drop_tools = compact_for_budget(msgs, window=10_000)
    assert drop_tools is True
    kept = [m for m in out if isinstance(m, ToolMessage)]
    assert len(str(kept[0].content)) < _MAX_KEPT_RESULT_CHARS + 200
    assert "truncated to fit the context window" in kept[0].content


def test_build_turn_content_frames_transcript():
    from airc_room.runner import build_turn_content
    from airc_room.store import Message

    m = Message(id=1, thread_id=1, sender="alice", kind="human", text="hi", ts=0.0)
    content = build_turn_content([m])
    assert "[alice] hi" in content
    assert "reply as yourself" in content  # a leading label, not a trailing
    assert "quote" not in content  # meta-instruction the model can echo back
    assert "NOTHING_TO_ADD" not in content
    assert "(You were asked" in build_turn_content([])
    # A direct human address withdraws the NOTHING_TO_ADD escape hatch.
    addressed = build_turn_content([m], addressed=True)
    assert "addressed you directly" in addressed
    assert "NOTHING_TO_ADD" in addressed


def test_build_turn_content_leads_with_current_time():
    # The time is injected in the per-turn tail (uncached), not the system prompt,
    # so it does not bust the prefix cache. It leads the content and reads the
    # wall clock; `now` is injectable for determinism.
    from datetime import datetime

    from airc_room.runner import build_turn_content
    from airc_room.store import Message

    fixed = datetime(2026, 7, 19, 14, 30)
    m = Message(id=1, thread_id=1, sender="alice", kind="human", text="hi", ts=0.0)
    content = build_turn_content([m], now=fixed)
    assert content.startswith("Current time: ")
    assert "2026-07-19 14:30" in content
    assert "Sunday" in content  # 2026-07-19 is a Sunday
    # A naive datetime yields no %Z; the line must not leave a dangling space.
    assert "  ." not in content and " .\n" not in content


def test_strip_self_attribution():
    from airc_room.runner import strip_self_attribution

    # The agent's own label, both shapes, leading only.
    assert strip_self_attribution("[perf] it deopts", "perf") == "it deopts"
    assert strip_self_attribution("perf: it deopts", "perf") == "it deopts"
    assert strip_self_attribution("it deopts", "perf") == "it deopts"
    # Mid-reply, the same string is content, not a label.
    assert strip_self_attribution("see\nperf: it deopts", "perf") == (
        "see\nperf: it deopts"
    )
    # Another participant's name is left alone wherever it appears. A quoted
    # commit subject is the case that made policing these untenable: in a V8
    # room "[compiler] ..." is a CL tag far more often than a fabrication.
    text = (
        "The win is IC-side.\n"
        "[compiler] Fix key check in inlined proxy loads\n"
        "compiler: please verify the lowering"
    )
    assert strip_self_attribution(text, "perf") == text

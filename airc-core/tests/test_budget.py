# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""compact_for_budget only sheds at the hard threshold; below it, intact (so the
prefix cache is never poisoned by a preemptive strip)."""

from airc_core.agent import _ELIDED_TOOL_RESULT, compact_for_budget
from langchain_core.messages import HumanMessage, ToolMessage


def _tool(text: str, i: int) -> ToolMessage:
    return ToolMessage(content=text, tool_call_id=f"c{i}")


def test_intact_below_threshold_is_the_same_object():
    msgs = [HumanMessage("hi"), _tool("x" * 400, 1), _tool("y" * 400, 2)]
    out, drop = compact_for_budget(msgs, window=1_000_000)  # ~200 tok << 900k
    assert out is msgs  # untouched -> cache prefix stays byte-identical
    assert drop is False


def test_sheds_only_at_hard_threshold():
    big = "z" * 4000  # ~1000 tokens each
    msgs = [HumanMessage("hi"), _tool(big, 1), _tool(big, 2), _tool(big, 3)]
    # ~3000 tokens vs a 3000-token window -> over the 90% hard line.
    out, drop = compact_for_budget(msgs, window=3000)
    assert drop is True  # tools dropped to force the turn to wrap up
    stubbed = [
        m
        for m in out
        if isinstance(m, ToolMessage) and str(m.content) == _ELIDED_TOOL_RESULT
    ]
    assert len(stubbed) == 2  # all but the most recent result kept intact

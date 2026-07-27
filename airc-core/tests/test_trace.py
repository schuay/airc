# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""_CallTrace accumulates turn totals and derives the cache signals."""

from types import SimpleNamespace

from airc_core.agent import _CallTrace


def _resp(inp: int, out: int, cached: int):
    """A minimal LLMResult shape for _usage_from_response: generations ->
    message.usage_metadata."""
    msg = SimpleNamespace(
        usage_metadata={
            "input_tokens": inp,
            "output_tokens": out,
            "input_token_details": {"cache_read": cached},
        }
    )
    return SimpleNamespace(generations=[[SimpleNamespace(message=msg)]])


def test_summary_accumulates_and_derives_hit_rate():
    t = _CallTrace("agent", "turn")
    t.on_llm_end(_resp(100, 10, 0), run_id=1)  # cold call: no cache
    t.on_llm_end(_resp(200, 5, 180), run_id=2)  # warm call: mostly cached
    s = t.summary()
    assert s["calls"] == 2
    assert s["input"] == 300
    assert s["output"] == 15
    assert s["cached"] == 180
    assert s["uncached"] == 120  # the full-price tokens
    assert s["hit_pct"] == 60  # 180 / 300
    assert s["max_call_input"] == 200


def test_summary_empty_is_zero_not_div0():
    assert _CallTrace("a", "k").summary()["hit_pct"] == 0


def test_calltrace_skips_summarization_calls():
    # A ceiling-summarization call (tagged lc_source) must not be booked against
    # the persona's turn -- it runs on the cheap filter model, not the turn model.
    t = _CallTrace("perf", "turn")
    t.on_chat_model_start(None, [[]], run_id="r1", metadata={})
    t.on_llm_end(_resp(100, 10, 0), run_id="r1")
    t.on_chat_model_start(
        None, [[]], run_id="r2", metadata={"lc_source": "summarization"}
    )
    t.on_llm_end(_resp(800_000, 50, 0), run_id="r2")
    assert t.calls == 1
    assert t.total_input == 100
    assert t.max_input_tokens == 100  # not the summarizer's 800k


def test_turn_usage_handler_skips_summarization_calls():
    from airc_core.agent import TurnUsageHandler

    h = TurnUsageHandler()
    h.on_chat_model_start(
        None, [[]], run_id="r2", metadata={"lc_source": "summarization"}
    )
    h.on_llm_end(_resp(800_000, 50, 0), run_id="r2")
    assert h.usage_metadata == {}  # not aggregated into the turn total

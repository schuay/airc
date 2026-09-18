# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Usage: the one reader of provider usage_metadata, and the collector."""

from __future__ import annotations

import logging
from uuid import uuid4

import pytest
from airc_core.collector import UsageCollector
from airc_core.usage import MODEL_KEY, SOURCE_KEY, SUMMARIZATION, Usage
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

OPUS = "google_anthropic_vertex:claude-opus-5"
PRO = "google_vertexai:gemini-3.1-pro-preview"


def test_anthropic_call_reads_the_split_cache_write_keys():
    """langchain_anthropic zeroes the generic cache_creation key when the API
    returns the TTL breakdown, so a reader of the generic key alone books
    zero writes for every such response."""
    u = Usage.of_call(
        {
            "input_tokens": 10_000,
            "output_tokens": 500,
            "input_token_details": {
                "cache_read": 8_000,
                "cache_creation": 0,
                "ephemeral_5m_input_tokens": 1_000,
                "ephemeral_1h_input_tokens": 0,
            },
        },
        OPUS,
    )
    assert (u.calls, u.input, u.cache_read, u.cache_write_5m) == (
        1,
        10_000,
        8_000,
        1_000,
    )
    assert u.uncached == 1_000 and u.hit_pct == 80
    assert u.usd_input == pytest.approx((1000 * 5 + 8000 * 0.5 + 1000 * 6.25) / 1e6)
    assert u.usd_output == pytest.approx(500 * 25 / 1e6)
    assert u.usd == pytest.approx(u.usd_input + u.usd_output)
    assert not u.estimated


def test_anthropic_generic_cache_creation_key_is_a_5m_write():
    u = Usage.of_call(
        {
            "input_tokens": 5_000,
            "output_tokens": 1,
            "input_token_details": {"cache_read": 0, "cache_creation": 5_000},
        },
        OPUS,
    )
    assert u.cache_write_5m == 5_000 and u.cache_write_1h == 0


def test_legacy_vertex_gemini_adds_thoughts_to_billable_output(monkeypatch):
    """The langchain-google-vertexai adapter reports candidates alone as
    output_tokens and the thoughts under output_token_details; Google bills
    both as output."""
    monkeypatch.setenv("AIRC_GOOGLE_SDK", "vertexai")
    meta = {
        "input_tokens": 1_000,
        "output_tokens": 200,
        "input_token_details": {"cache_read": 0},
        "output_token_details": {"reasoning": 4_000},
    }
    u = Usage.of_call(meta, PRO)
    assert u.output == 4_200 and u.reasoning == 4_000
    assert u.usd == pytest.approx((1000 * 2 + 4200 * 12) / 1e6)
    # The same dict from a provider whose output_tokens already includes
    # thinking is not double counted.
    assert Usage.of_call(meta, OPUS).output == 200


def test_genai_served_vertex_gemini_does_not_count_thoughts_twice(monkeypatch):
    """The default stack serves google_vertexai: ids through langchain-google-genai,
    whose output_tokens already sums candidates and thoughts and repeats the
    thoughts under output_token_details. Adding them again billed every
    thinking call for its thought twice."""
    monkeypatch.delenv("AIRC_GOOGLE_SDK", raising=False)
    meta = {
        "input_tokens": 1_000,
        "output_tokens": 4_200,
        "input_token_details": {"cache_read": 0},
        "output_token_details": {"reasoning": 4_000},
    }
    u = Usage.of_call(meta, PRO)
    assert u.output == 4_200 and u.reasoning == 4_000
    assert u.usd == pytest.approx((1000 * 2 + 4200 * 12) / 1e6)


def test_gemini_pro_prices_each_call_at_its_own_tier():
    small = Usage.of_call({"input_tokens": 100_000, "output_tokens": 0}, PRO)
    big = Usage.of_call({"input_tokens": 300_000, "output_tokens": 0}, PRO)
    assert small.usd == pytest.approx(0.2)
    assert big.usd == pytest.approx(1.2)
    # A sum is the sum of the priced calls, not a re-pricing of the total.
    assert (small + big).usd == pytest.approx(1.4)


def test_unlisted_model_is_estimated_and_the_flag_survives_a_sum():
    est = Usage.of_call({"input_tokens": 1_000, "output_tokens": 100}, "deepseek:x")
    assert est.estimated and est.usd == pytest.approx((1000 * 1 + 100 * 5) / 1e6)
    real = Usage.of_call({"input_tokens": 1_000, "output_tokens": 100}, OPUS)
    assert not real.estimated
    assert (real + est).estimated and (real + real).estimated is False


def test_sum_keeps_model_only_when_both_agree():
    a = Usage.of_call({"input_tokens": 10, "output_tokens": 1}, OPUS)
    b = Usage.of_call({"input_tokens": 20, "output_tokens": 1}, PRO)
    assert (a + a).model == OPUS
    assert (a + b).model == ""
    assert (a + b).max_call_input == 20 and (a + b).calls == 2


def test_missing_usage_books_an_empty_call():
    u = Usage.of_call(None, OPUS)
    assert u.calls == 1 and u.input == 0 and u.usd == 0.0


def test_cache_creation_is_input_plus_storage_with_no_call():
    u = Usage.of_cache_creation(PRO, tokens=100_000, ttl_hours=0.5)
    assert u.calls == 0 and u.input == 100_000
    assert u.cache_storage_token_hours == 50_000
    assert u.usd == pytest.approx((100_000 * 2 + 50_000 * 4.5) / 1e6)


def test_line_leads_with_cost_and_marks_an_estimate():
    u = Usage.of_call(
        {
            "input_tokens": 412_000,
            "output_tokens": 8_000,
            "input_token_details": {
                "cache_read": 383_000,
                "ephemeral_5m_input_tokens": 12_000,
            },
            "output_token_details": {"reasoning": 5_000},
        },
        OPUS,
    )
    line = u.line()
    assert line.startswith("$") and " over 1 call: " in line
    assert (
        "412k in ($0.35, 93% cached, 12k written), 8k out ($0.20, 5k thinking)" in line
    )
    assert line == f"{u.cost()} over 1 call: {u.shape()}"
    est = Usage.of_call({"input_tokens": 10, "output_tokens": 1}, "deepseek:x")
    assert est.line().startswith("~$")


def test_usage_round_trips_through_its_dict_form():
    """The journal event and the bus payload carry model_dump(); a reader must
    get the same value back, estimate flag included."""
    u = Usage.of_call({"input_tokens": 10, "output_tokens": 1}, "deepseek:x")
    assert Usage.model_validate(u.model_dump()) == u


def _result(usage: dict) -> LLMResult:
    usage = {"total_tokens": usage["input_tokens"] + usage["output_tokens"], **usage}
    return LLMResult(
        generations=[[ChatGeneration(message=AIMessage("", usage_metadata=usage))]]
    )


def test_collector_books_calls_and_routes_summarization_aside(caplog):
    c = UsageCollector("perf", "turn", OPUS)
    r1, r2, r3 = uuid4(), uuid4(), uuid4()
    msgs = [[HumanMessage("q"), ToolMessage("x" * 40, tool_call_id="t")]]
    with caplog.at_level(logging.INFO, logger="airc_core.collector"):
        c.on_chat_model_start({}, msgs, run_id=r1)
        c.on_llm_end(_result({"input_tokens": 1_000, "output_tokens": 10}), run_id=r1)
        # The ceiling summarization runs on the filter model, tagged by the
        # middleware; it is paid for, so it is kept, but off the agent's row.
        c.on_chat_model_start(
            {},
            msgs,
            run_id=r2,
            metadata={
                SOURCE_KEY: SUMMARIZATION,
                MODEL_KEY: "google_vertexai:gemini-3.8-flash",
            },
        )
        c.on_llm_end(
            _result({"input_tokens": 50_000, "output_tokens": 2_000}), run_id=r2
        )
        # A call that errors books nothing; a failed request is not billed.
        c.on_chat_model_start({}, msgs, run_id=r3)
        c.on_llm_error(RuntimeError("503"), run_id=r3)
    assert c.total.calls == 1 and c.total.input == 1_000 and c.total.model == OPUS
    assert c.aside.calls == 1 and c.aside.model == "google_vertexai:gemini-3.8-flash"
    assert c.aside.usd == pytest.approx((50_000 * 0.75 + 2_000 * 3.75) / 1e6)
    assert c._shape == {} and c._aside_model == {}
    # The per-call line carries the call's own cost and the turn's so far.
    assert "call perf/turn #1: $0.01 (turn $0.01): 1k in ($0.01" in caplog.text
    assert "1 tool results (40 chars)" in caplog.text


def test_book_aside_reaches_the_active_collector_only():
    """Spend that is not a model call (an explicit cache the middleware
    creates) has no callback to land in; it is booked against whichever
    collector's invocation is running, and dropped when none is."""
    from airc_core.collector import book_aside

    c = UsageCollector("perf", "turn", OPUS)
    book_aside(Usage.of_cache_creation(PRO, 1000, 0.5))  # nobody active
    assert c.aside.empty
    with c.active():
        book_aside(Usage.of_cache_creation(PRO, 100_000, 0.5))
    assert c.aside.calls == 0 and c.aside.input == 100_000
    assert c.aside.usd == pytest.approx((100_000 * 2 + 50_000 * 4.5) / 1e6)
    assert c.total.empty  # the turn's own row is untouched

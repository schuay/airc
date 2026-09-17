# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""The price table: lookup, tiering, and the generic fallback."""

from __future__ import annotations

import pytest
from airc_core.pricing import GENERIC, Rate, bare_model_name, listed_models, price_for


def test_lookup_strips_provider_prefix_and_vertex_version():
    p = price_for("google_anthropic_vertex:claude-opus-5@20260701")
    assert p.model == "claude-opus-5" and not p.generic
    assert bare_model_name("google_vertexai:gemini-3.8-flash") == "gemini-3.8-flash"


def test_lookup_is_exact_not_prefix():
    """A "-lite" variant is a different price; matching it to its big sibling
    would cost it several times over."""
    assert price_for("google_vertexai:gemini-3.8-flash-lite") is GENERIC


def test_generic_is_the_fallback_and_says_so():
    assert price_for("deepseek:deepseek-chat") is GENERIC
    assert GENERIC.generic and price_for("anthropic:claude-opus-5").generic is False


def test_every_listing_has_a_date_and_the_generic_ratios_hold():
    for name in listed_models():
        p = price_for(name)
        assert p.as_of.year >= 2026
        assert p.rate.cache_read < p.rate.input < p.rate.output
    r = GENERIC.rate
    assert r.cache_read == pytest.approx(0.1 * r.input)
    assert r.cache_write_5m == pytest.approx(1.25 * r.input)
    assert r.cache_write_1h == pytest.approx(2.0 * r.input)


def test_gemini_pro_tier_switches_on_the_calls_own_prompt():
    p = price_for("google_vertexai:gemini-3.1-pro-preview")
    assert p.rate_for(200_000) is p.rate
    assert p.rate_for(200_001) is p.long_context[1]
    # 100k uncached input below the tier costs 100k * $2/M.
    assert p.cost(prompt_tokens=100_000) == pytest.approx(0.2)
    # The same tokens as part of a 300k prompt cost the higher rate.
    assert p.cost(prompt_tokens=300_000) == pytest.approx(300_000 * 4.0 / 1e6)


def test_cost_bills_each_part_at_its_own_rate():
    p = price_for("anthropic:claude-opus-5")
    # 1000 uncached + 8000 read + 1000 written (5m) + 500 out.
    usd = p.cost(
        prompt_tokens=10_000, cache_read=8_000, cache_write_5m=1_000, output=500
    )
    expect = (1000 * 5 + 8000 * 0.5 + 1000 * 6.25 + 500 * 25) / 1e6
    assert usd == pytest.approx(expect)


def test_cost_never_bills_negative_uncached_input():
    """Reads and writes are subsets of the prompt, but a provider that
    reports them inconsistently must not turn into a credit."""
    p = price_for("anthropic:claude-opus-5")
    assert p.cost(prompt_tokens=100, cache_read=200) == pytest.approx(200 * 0.5 / 1e6)


def test_storage_is_billed_by_token_hour():
    r = Rate(input=1, cache_read=0.1, output=5, cache_storage_hour=4.5)
    p = price_for("google_vertexai:gemini-3.1-pro-preview")
    assert p.rate.cache_storage_hour == r.cache_storage_hour
    # 1M tokens held for half an hour at $4.50 per M-token-hour.
    assert p.cost(prompt_tokens=0, cache_storage_token_hours=500_000) == pytest.approx(
        2.25
    )

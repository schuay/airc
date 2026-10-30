# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""The price table: lookup, tiering, and the generic fallback."""

from __future__ import annotations

import pytest
from airc_core.pricing import (
    GENERIC,
    Price,
    Rate,
    bare_model_name,
    listed_models,
    price_for,
    register_alias,
)


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


@pytest.mark.parametrize("model", ["deepseek-flash", "deepseek-v4-flash"])
def test_deepseek_flash_uses_published_peak_rates(model):
    p = price_for(f"deepseek:{model}")
    assert not p.generic
    assert (p.rate.input, p.rate.cache_read, p.rate.output) == (0.3, 0.006, 1.2)
    assert p.cost(
        prompt_tokens=1_000_000, cache_read=800_000, output=100_000
    ).total == pytest.approx(0.1848)
    assert "off-peak" in p.note and "https://api-docs.deepseek.com" in p.note


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
    assert p.cost(prompt_tokens=100_000).total == pytest.approx(0.2)
    # The same tokens as part of a 300k prompt cost the higher rate.
    assert p.cost(prompt_tokens=300_000).total == pytest.approx(300_000 * 4.0 / 1e6)


def test_gemini_4_argon_is_listed_at_the_standard_rate_with_no_cache_fees():
    p = price_for("google_vertexai:gemini-4-argon")
    assert not p.generic and p.long_context is None
    r = p.rate
    assert (r.input, r.cache_read, r.output) == (4.0, 0.2, 20.0)
    # Implicit caching only: nothing to write and nothing to store.
    assert (r.cache_write_5m, r.cache_write_1h, r.cache_storage_hour) == (0, 0, 0)
    # A 100k prompt with 80k served from cache: 20k * $4 + 80k * $0.2.
    cost = p.cost(prompt_tokens=100_000, cache_read=80_000, output=1_000)
    assert cost.input == pytest.approx((20_000 * 4.0 + 80_000 * 0.2) / 1e6)
    assert cost.output == pytest.approx(1_000 * 20.0 / 1e6)


@pytest.mark.parametrize(
    ("model", "rates"),
    [
        ("claude-fable-5-1", (10.0, 0.25, 50.0)),
        ("claude-fable-5", (10.0, 1.0, 50.0)),
        ("claude-opus-5-5", (4.0, 0.2, 20.0)),
        ("claude-opus-5", (5.0, 0.5, 25.0)),
        ("claude-opus-4-8", (5.0, 0.5, 25.0)),
        ("claude-opus-4-7", (5.0, 0.5, 25.0)),
        ("claude-opus-4-6", (5.0, 0.5, 25.0)),
        ("claude-sonnet-5-5", (2.0, 0.2, 10.0)),
        ("claude-sonnet-5", (2.0, 0.2, 10.0)),
        ("claude-sonnet-4-6", (3.0, 0.3, 15.0)),
        ("claude-haiku-5-5", (0.1, 0.01, 0.5)),
        ("claude-haiku-4-5", (1.0, 0.1, 5.0)),
    ],
)
def test_claude_listings_match_the_published_first_party_rates(model, rates):
    """Input, cache read and output per MTok as Anthropic lists them. The read
    rate is the one that varies: 0.1x of input everywhere except claude-opus-5-5
    (0.05x) and claude-fable-5-1 (0.025x)."""
    p = price_for(f"anthropic:{model}")
    assert not p.generic and p.model == model
    assert (p.rate.input, p.rate.cache_read, p.rate.output) == pytest.approx(rates)
    assert price_for(f"google_anthropic_vertex:{model}") is p


def test_every_claude_listing_bills_cache_writes_at_the_messages_api_ratios():
    """A 5-minute write is 1.25x input and a 1-hour write 2x on every Claude
    checkpoint; a read never exceeds 0.1x. No Claude listing stores by the
    token-hour."""
    claude = [n for n in listed_models() if n.startswith("claude-")]
    assert len(claude) >= 12
    for name in claude:
        r = price_for(name).rate
        assert r.cache_write_5m == pytest.approx(1.25 * r.input), name
        assert r.cache_write_1h == pytest.approx(2.0 * r.input), name
        assert r.cache_read <= 0.1 * r.input + 1e-12, name
        assert r.cache_storage_hour == 0, name


def test_haiku_5_5_switches_rate_card_above_100k_prompt_tokens():
    p = price_for("anthropic:claude-haiku-5-5")
    assert p.rate_for(100_000) is p.rate
    assert p.rate_for(100_001) is p.long_context[1]
    high = p.long_context[1]
    assert (high.input, high.cache_read, high.output) == pytest.approx((0.5, 0.05, 2.5))
    assert high.cache_write_5m == pytest.approx(0.625)
    assert high.cache_write_1h == pytest.approx(1.0)
    # 150k prompt, 100k of it read from cache: the whole call is on the second
    # card, reads included.
    cost = p.cost(prompt_tokens=150_000, cache_read=100_000, output=1_000)
    assert cost.input == pytest.approx((50_000 * 0.5 + 100_000 * 0.05) / 1e6)
    assert cost.output == pytest.approx(1_000 * 2.5 / 1e6)


def test_cost_bills_each_part_at_its_own_rate():
    p = price_for("anthropic:claude-opus-5")
    # 1000 uncached + 8000 read + 1000 written (5m) + 500 out.
    cost = p.cost(
        prompt_tokens=10_000, cache_read=8_000, cache_write_5m=1_000, output=500
    )
    assert cost.input == pytest.approx((1000 * 5 + 8000 * 0.5 + 1000 * 6.25) / 1e6)
    assert cost.output == pytest.approx(500 * 25 / 1e6)
    assert cost.total == pytest.approx(cost.input + cost.output)


def test_cost_never_bills_negative_uncached_input():
    """Reads and writes are subsets of the prompt, but a provider that
    reports them inconsistently must not turn into a credit."""
    p = price_for("anthropic:claude-opus-5")
    assert p.cost(prompt_tokens=100, cache_read=200).total == pytest.approx(
        200 * 0.5 / 1e6
    )


def test_storage_is_billed_by_token_hour():
    r = Rate(input=1, cache_read=0.1, output=5, cache_storage_hour=4.5)
    p = price_for("google_vertexai:gemini-3.1-pro-preview")
    assert p.rate.cache_storage_hour == r.cache_storage_hour
    # 1M tokens held for half an hour at $4.50 per M-token-hour.
    cost = p.cost(prompt_tokens=0, cache_storage_token_hours=500_000)
    assert (cost.input, cost.output) == (pytest.approx(2.25), 0.0)


# ── aliases ─────────────────────────────────────────────────────────────────


@pytest.fixture
def listing():
    """A listing of our own plus a clean alias table, both restored afterwards:
    register_alias writes process-global module state, and the alias table
    would otherwise leak between tests in the order they happened to run."""
    from datetime import date

    from airc_core import pricing

    fake = Price(
        model="listed-model",
        rate=Rate(input=2.0, cache_read=0.2, output=10.0),
        as_of=date(2026, 1, 1),
    )
    saved_listed, saved_aliases = dict(pricing._LISTED), dict(pricing._ALIASES)
    pricing._LISTED[fake.model] = fake
    pricing._ALIASES.clear()
    try:
        yield fake
    finally:
        pricing._LISTED.clear()
        pricing._LISTED.update(saved_listed)
        pricing._ALIASES.clear()
        pricing._ALIASES.update(saved_aliases)


def test_alias_prices_an_unlisted_name_at_its_target(listing):
    assert price_for("mybackend:internal-ckpt-7") is GENERIC
    register_alias("mybackend:internal-ckpt-7", "listed-model")
    p = price_for("mybackend:internal-ckpt-7")
    # The target's listing, unchanged: not generic, and not marked in any way.
    assert p is listing and not p.generic


def test_alias_by_bare_name_covers_every_provider_spelling(listing):
    register_alias("internal-ckpt-7", "listed-model")
    assert price_for("mybackend:internal-ckpt-7") is listing
    assert price_for("otherbackend:internal-ckpt-7") is listing
    assert price_for("internal-ckpt-7") is listing


def test_full_id_alias_beats_bare_alias_and_claims_only_its_provider(listing):
    """A provider-qualified alias is a statement about ONE provider's spelling
    of the name; the bare alias, if any, still answers for the others."""
    from datetime import date

    from airc_core import pricing

    other = Price(
        model="other-listed",
        rate=Rate(input=1.0, cache_read=0.1, output=5.0),
        as_of=date(2026, 1, 1),
    )
    pricing._LISTED[other.model] = other
    register_alias("internal-ckpt-7", "listed-model")
    register_alias("mybackend:internal-ckpt-7", "other-listed")
    assert price_for("mybackend:internal-ckpt-7") is other
    assert price_for("otherbackend:internal-ckpt-7") is listing


def test_alias_target_may_be_written_as_a_full_id(listing):
    register_alias("mybackend:internal-ckpt-7", "somewhere:listed-model")
    assert price_for("mybackend:internal-ckpt-7") is listing


def test_alias_to_an_unlisted_name_is_refused(listing):
    """It would resolve to GENERIC, which is the outcome the alias was written
    to prevent -- dead config, refused rather than honoured silently."""
    with pytest.raises(ValueError, match="no listing for 'nowhere-model'"):
        register_alias("mybackend:internal-ckpt-7", "nowhere-model")
    # Chains are not a thing either: an alias is not a listing.
    register_alias("internal-ckpt-7", "listed-model")
    with pytest.raises(ValueError, match="no listing for 'internal-ckpt-7'"):
        register_alias("mybackend:other-ckpt", "internal-ckpt-7")
    assert price_for("mybackend:other-ckpt") is GENERIC


def test_alias_reregistration_same_is_noop_conflicting_raises(listing):
    from datetime import date

    from airc_core import pricing

    pricing._LISTED["other-listed"] = Price(
        model="other-listed",
        rate=Rate(input=1.0, cache_read=0.1, output=5.0),
        as_of=date(2026, 1, 1),
    )
    register_alias("internal-ckpt-7", "listed-model")
    register_alias("internal-ckpt-7", "listed-model")  # same pair: fine
    register_alias("internal-ckpt-7", "somewhere:listed-model")  # same target
    with pytest.raises(ValueError, match="already priced as 'listed-model'"):
        register_alias("internal-ckpt-7", "other-listed")
    assert price_for("internal-ckpt-7") is listing


def test_alias_does_not_shadow_a_real_listing(listing):
    """Aliases are consulted first, so an alias whose KEY is itself a listed
    name would silently re-price a real model. Nothing stops an operator
    writing one; this pins that the lookup then follows the alias, so the
    behaviour is at least the documented one rather than an accident."""
    from datetime import date

    from airc_core import pricing

    other = Price(
        model="other-listed",
        rate=Rate(input=1.0, cache_read=0.1, output=5.0),
        as_of=date(2026, 1, 1),
    )
    pricing._LISTED[other.model] = other
    assert price_for("x:other-listed") is other
    register_alias("other-listed", "listed-model")
    assert price_for("x:other-listed") is listing

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

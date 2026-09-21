# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""What a model call costs, as data.

One table, one lookup. Every dollar figure in the suite comes from here through
`Usage.of_call` (usage.py); nothing else holds a rate or a ratio. Prices are USD
per million tokens as the provider lists them publicly, with the date they were
read, so a stale entry is visible instead of silently wrong.

A model missing from the table gets GENERIC: a placeholder rate at a
mid-market magnitude with the ratios the listed providers share (a cache read
at a tenth of input, output at five times). It exists so an unlisted model --
a private deployment, a new checkpoint -- is still costed instead of dropped
from every total, and so its rows are flagged as estimated instead of passed
off as measured. A deploy that knows better can say which listing such a
name really is (`register_alias`, fed from `[pricing.aliases]` in config) and
have it priced there instead.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict

_MILLION = 1_000_000


class Rate(BaseModel):
    """USD per million tokens for each thing a call is billed for.

    Output covers thinking: every provider bills reasoning at the output rate.
    Cache writes are Anthropic's; Gemini charges no write, only the read and,
    for an explicit cache, storage by token-hour.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: float
    cache_read: float
    output: float
    cache_write_5m: float = 0.0
    cache_write_1h: float = 0.0
    cache_storage_hour: float = 0.0


class Cost(BaseModel):
    """USD for one call, split by what was billed: everything on the prompt
    side (uncached input, cache reads and writes, explicit-cache storage)
    against the output side (output, thinking included)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: float = 0.0
    output: float = 0.0

    @property
    def total(self) -> float:
        return self.input + self.output


class Price(BaseModel):
    """One model's listing: a rate, and optionally a second rate past a prompt
    size (Gemini Pro bills everything at a higher rate once the prompt exceeds
    200k tokens). `rate_for` picks by the prompt size of one call, which is why
    pricing happens per call and never over a turn's sums.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str  # bare name, no provider prefix; "" for GENERIC
    rate: Rate
    long_context: tuple[int, Rate] | None = None  # (prompt tokens, rate above)
    as_of: date
    note: str = ""
    generic: bool = False

    def rate_for(self, prompt_tokens: int) -> Rate:
        if self.long_context is not None and prompt_tokens > self.long_context[0]:
            return self.long_context[1]
        return self.rate

    def cost(
        self,
        *,
        prompt_tokens: int,
        cache_read: int = 0,
        cache_write_5m: int = 0,
        cache_write_1h: int = 0,
        output: int = 0,
        cache_storage_token_hours: float = 0.0,
    ) -> Cost:
        """What one call cost. `prompt_tokens` is the whole prompt as the
        provider counts it, cache reads and writes included; the uncached
        remainder is billed at the input rate."""
        r = self.rate_for(prompt_tokens)
        uncached = max(0, prompt_tokens - cache_read - cache_write_5m - cache_write_1h)
        prompt_side = (
            uncached * r.input
            + cache_read * r.cache_read
            + cache_write_5m * r.cache_write_5m
            + cache_write_1h * r.cache_write_1h
            + cache_storage_token_hours * r.cache_storage_hour
        )
        return Cost(input=prompt_side / _MILLION, output=output * r.output / _MILLION)


_READ = date(2026, 9, 17)

_LISTED: dict[str, Price] = {
    p.model: p
    for p in (
        Price(
            model="gemini-3.1-pro-preview",
            rate=Rate(input=2.0, cache_read=0.2, output=12.0, cache_storage_hour=4.5),
            long_context=(
                200_000,
                Rate(input=4.0, cache_read=0.4, output=18.0, cache_storage_hour=4.5),
            ),
            as_of=_READ,
        ),
        Price(
            model="gemini-3.8-flash",
            rate=Rate(
                input=0.75, cache_read=0.075, output=3.75, cache_storage_hour=0.5
            ),
            as_of=_READ,
            note=(
                "introductory through 2026-12-31; from 2027-01-01 the listed"
                " standard rate is 1.50 / 0.15 / 7.50, storage 1.00"
            ),
        ),
        Price(
            model="claude-opus-5",
            rate=Rate(
                input=5.0,
                cache_read=0.5,
                output=25.0,
                cache_write_5m=6.25,
                cache_write_1h=10.0,
            ),
            as_of=_READ,
        ),
    )
}

# The fallback for anything unlisted. Not a real price: a round mid-market
# input rate with the multipliers the listed providers share, so an estimate
# is the right order of magnitude and moves with cache hits the way a real
# bill would.
GENERIC = Price(
    model="",
    rate=Rate(
        input=1.0,
        cache_read=0.1,
        output=5.0,
        cache_write_5m=1.25,
        cache_write_1h=2.0,
        cache_storage_hour=1.0,
    ),
    as_of=_READ,
    note="placeholder rate for models without a listing",
    generic=True,
)


def bare_model_name(model_id: str) -> str:
    """The provider-side name: "google_anthropic_vertex:claude-opus-5@20260701"
    -> "claude-opus-5". Vertex pins a version with "@"; the price is the
    model's."""
    name = model_id.split(":", 1)[-1]
    return name.split("@", 1)[0]


# Names the table does not list, priced at a listing it does. Keyed by what
# the deploy wrote -- a full "provider:name" id or a bare name -- and valued
# by a LISTED bare name, never another alias, so `price_for` stays one lookup
# with no chain to follow.
#
# This exists for the name that is not a public listing and never will be: a
# private deployment of a released model under an internal label, a checkpoint
# served ahead of its release. Such a model is not free and is not the generic
# placeholder either; the operator knows which listing it is closest to and
# says so. The price that comes back is the target's, unchanged -- there is no
# "approximately" flag, and a budget over an aliased model is a budget at the
# target's rate. That is the deal: an alias is the operator asserting the
# equivalence, and the table takes them at their word.
#
# Filled from config ([pricing.aliases], via config.load_common), which is the
# right home for it: these names are per-deploy and churn faster than a
# release cycle, and a table in code would need a release for each one.
_ALIASES: dict[str, str] = {}


def register_alias(model: str, priced_as: str) -> None:
    """Price `model` (a full id or a bare name) at `priced_as`'s listing.

    `priced_as` must be listed, as a bare name or a full id -- an alias to an
    unlisted name would resolve to GENERIC, which is exactly what the alias was
    written to avoid, so it is dead config and refused. Re-registering the same
    pair is a no-op, for the reason register_provider gives (load_common runs
    more than once per process in some components); a CONFLICTING pair raises,
    because which price applied would otherwise depend on parse order.
    """
    target = bare_model_name(priced_as)
    if target not in _LISTED:
        raise ValueError(
            f"{model!r} cannot be priced as {priced_as!r}: the price table has"
            f" no listing for {target!r} (listed: {', '.join(_LISTED)})"
        )
    if (prior := _ALIASES.get(model)) is not None and prior != target:
        raise ValueError(f"{model!r} is already priced as {prior!r}")
    _ALIASES[model] = target


def price_for(model_id: str) -> Price:
    """The listing for `model_id`, or GENERIC.

    Aliases first, the full id before the bare name so an alias written for
    one provider's spelling does not also claim another's; then the bare name
    itself. Exact matches throughout: a prefix heuristic would price a new
    "-lite" variant as its big sibling.
    """
    bare = bare_model_name(model_id)
    key = _ALIASES.get(model_id) or _ALIASES.get(bare) or bare
    return _LISTED.get(key, GENERIC)


def listed_models() -> tuple[str, ...]:
    return tuple(_LISTED)

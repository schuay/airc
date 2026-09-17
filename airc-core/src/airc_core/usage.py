# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Token usage and its cost, as one value the whole suite books.

`Usage` is the only shape usage takes once it leaves the provider: the ledger
row, the journal event, the job result on the bus and the log line are all
this value or a rendering of it. `Usage.of_call` is the only reader of a
provider's usage_metadata and the only place a price is applied, so the
provider quirks (Anthropic's split cache-write keys, Vertex Gemini's separate
thought count) and the per-call tier are resolved once. Sums never re-price:
Gemini Pro's rate depends on each call's own prompt size, so a total is the
sum of priced calls and nothing else.

This module imports no framework: the wire spec carries a `Usage`, and the
CLI that parses a result must not pay for langchain to do it. The callback
that turns a graph invocation into a `Usage` is UsageCollector in
collector.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from .pricing import price_for
from .providers import traits_for

# Metadata keys a call is tagged with so the collector can book it apart.
# lc_source marks the ceiling summarization the agent middleware runs inside a
# turn; lc_model names the model it ran on, which is not the turn's.
SOURCE_KEY = "lc_source"
MODEL_KEY = "lc_model"
SUMMARIZATION = "summarization"


class Usage(BaseModel):
    """Token counts of one or more model calls and what they cost.

    Counts are the provider's: `input` is the whole prompt, cache reads and
    writes included; `output` is billable output, thinking included;
    `reasoning` is the thinking subset of it. `usd` is summed from per-call
    prices. `estimated` is set when any part was priced at the generic
    fallback rate, so a total that includes an unlisted model says so.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = ""  # configured id; "" once summed across models
    calls: int = 0
    input: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    output: int = 0
    reasoning: int = 0
    max_call_input: int = 0
    # Explicit-cache storage booked by the cache middleware at creation:
    # tokens held times the TTL in hours. Zero on every model call.
    cache_storage_token_hours: float = 0.0
    usd: float = 0.0
    estimated: bool = False

    @classmethod
    def of_call(cls, usage_metadata: Mapping[str, Any] | None, model_id: str) -> Usage:
        """One model call, priced at the tier its own prompt size selects.

        Anthropic reports cache writes under ephemeral_5m/1h keys when the API
        returns the TTL breakdown and zeroes the generic cache_creation key
        when it does; older responses carry only the generic key, which is a
        5m write. Vertex Gemini reports thoughts apart from output_tokens
        (providers.py reasoning_in_output).
        """
        u = usage_metadata or {}
        details = u.get("input_token_details") or {}
        out_details = u.get("output_token_details") or {}
        prompt = int(u.get("input_tokens") or 0)
        cache_read = int(details.get("cache_read") or 0)
        write_5m = int(details.get("ephemeral_5m_input_tokens") or 0)
        write_1h = int(details.get("ephemeral_1h_input_tokens") or 0)
        if not write_5m and not write_1h:
            write_5m = int(details.get("cache_creation") or 0)
        reasoning = int(out_details.get("reasoning") or 0)
        output = int(u.get("output_tokens") or 0)
        if not traits_for(model_id).reasoning_in_output:
            output += reasoning
        price = price_for(model_id)
        return cls(
            model=model_id,
            calls=1,
            input=prompt,
            cache_read=cache_read,
            cache_write_5m=write_5m,
            cache_write_1h=write_1h,
            output=output,
            reasoning=reasoning,
            max_call_input=prompt,
            usd=price.cost(
                prompt_tokens=prompt,
                cache_read=cache_read,
                cache_write_5m=write_5m,
                cache_write_1h=write_1h,
                output=output,
            ),
            estimated=price.generic,
        )

    @classmethod
    def of_cache_creation(cls, model_id: str, tokens: int, ttl_hours: float) -> Usage:
        """An explicit Vertex context cache: its creation is billed as input
        and its life as storage by token-hour. Not a model call, so `calls`
        stays zero and the ledger row carries only the cost."""
        price = price_for(model_id)
        token_hours = tokens * ttl_hours
        return cls(
            model=model_id,
            input=tokens,
            cache_storage_token_hours=token_hours,
            usd=price.cost(prompt_tokens=tokens, cache_storage_token_hours=token_hours),
            estimated=price.generic,
        )

    @property
    def empty(self) -> bool:
        return self.calls == 0 and self.input == 0 and self.usd == 0.0

    def __add__(self, other: Usage) -> Usage:
        # An empty seed takes the other side's model, so an accumulator started
        # as Usage() reports the model it booked; "" is reserved for a sum that
        # mixed two.
        if self.empty:
            model = other.model
        elif other.empty:
            model = self.model
        else:
            model = self.model if self.model == other.model else ""
        return Usage(
            model=model,
            calls=self.calls + other.calls,
            input=self.input + other.input,
            cache_read=self.cache_read + other.cache_read,
            cache_write_5m=self.cache_write_5m + other.cache_write_5m,
            cache_write_1h=self.cache_write_1h + other.cache_write_1h,
            output=self.output + other.output,
            reasoning=self.reasoning + other.reasoning,
            max_call_input=max(self.max_call_input, other.max_call_input),
            cache_storage_token_hours=self.cache_storage_token_hours
            + other.cache_storage_token_hours,
            usd=self.usd + other.usd,
            estimated=self.estimated or other.estimated,
        )

    @property
    def cache_write(self) -> int:
        return self.cache_write_5m + self.cache_write_1h

    @property
    def uncached(self) -> int:
        """Input billed at the full rate."""
        return max(0, self.input - self.cache_read - self.cache_write)

    @property
    def hit_pct(self) -> int:
        return round(100 * self.cache_read / self.input) if self.input else 0

    def line(self) -> str:
        """One log line: the cost first, then the shape that produced it.

        "$1.84 over 37 calls: 412k in (93% cached, 12k written), 8k out
        (5k thinking)". A tilde marks an estimate.
        """
        usd = f"{'~' if self.estimated else ''}${self.usd:.2f}"
        parts = [f"{_k(self.input)} in ({self.hit_pct}% cached"]
        if self.cache_write:
            parts.append(f", {_k(self.cache_write)} written")
        parts.append(f"), {_k(self.output)} out")
        if self.reasoning:
            parts.append(f" ({_k(self.reasoning)} thinking)")
        calls = f"{self.calls} call{'s' if self.calls != 1 else ''}"
        return f"{usd} over {calls}: {''.join(parts)}"


def _k(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.0f}k"
    return f"{n / 1_000_000:.1f}M"

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Per-provider facts, as data rather than as branches.

Three separate tests for "is this Claude?" had accumulated: a model_id prefix in
make_model, an isinstance check in the caching middleware, and a boolean
parameter on a boundary function. Each grew its own notion of what Claude does
differently, so a fact learned at one of them did not reach the others. The
sampling TypeError and the unreadable finish_reason were both that shape.

What lives here is the part that is data: which constructor kwargs a provider
rejects, and what it names the fields in its responses. Behaviour stays in code.
A middleware that has to know one client library's serialization rules is not a
table entry, and pretending otherwise would trade three branches for a
configuration language.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderTraits:
    """What one provider does differently, keyed by the model_id prefix."""

    id: str
    # Constructor kwargs the provider's API refuses. Dropped in make_model
    # before the client sees them.
    unsupported_kwargs: tuple[str, ...] = ()
    # Appended to the drop warning when there is something the caller should
    # know beyond the fact of the drop.
    unsupported_note: str = ""
    # response_metadata keys holding the model's stop reason. Providers pass
    # their own field names straight through, so the reader has to know them.
    stop_reason_keys: tuple[str, ...] = ("finish_reason",)


# temperature, top_p and top_k were removed from the Messages API in Claude
# Opus 4.7: the server returns 400 and the SDK, whose signature is generated
# from the same spec, raises TypeError first. seed was never part of it. The
# same API serves both the direct and the Vertex route, so both entries carry
# it.
_ANTHROPIC = ProviderTraits(
    id="anthropic",
    unsupported_kwargs=("temperature", "top_p", "top_k", "seed"),
    unsupported_note=(
        "Sampling was removed from the Messages API with no replacement --"
        " passes that relied on it for variance get none from this provider."
    ),
    stop_reason_keys=("stop_reason",),
)

_DEFAULT = ProviderTraits(id="")

_TRAITS: dict[str, ProviderTraits] = {
    "anthropic": _ANTHROPIC,
    "google_anthropic_vertex": _ANTHROPIC,
}


def traits_for(model_id: str) -> ProviderTraits:
    """Traits for `model_id`'s provider; a neutral record for anything else.

    An unknown provider gets the default rather than an error: this table is an
    optimization over asking the provider, and a provider missing from it has to
    keep working.
    """
    return _TRAITS.get(model_id.split(":", 1)[0], _DEFAULT)


# Every name a provider might use for the stop reason, in table order. Readers
# that hold a response but not the model it came from iterate this instead of
# dispatching: the metadata dict is already provider-shaped, so the only missing
# piece is the set of names.
STOP_REASON_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(
        key
        for traits in (_DEFAULT, *_TRAITS.values())
        for key in traits.stop_reason_keys
    )
)

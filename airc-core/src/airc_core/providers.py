# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Per-provider facts, as data, not branches.

One table keeps separate "is this Claude?" tests (a model_id prefix in
make_model, an isinstance check in a middleware, a boolean parameter) from each
growing their own notion of what a provider does differently.

What lives here is the part that is data: which constructor kwargs a provider
rejects, and what it names the fields in its responses. Behaviour stays in code:
a middleware that has to know one client library's serialization rules is not a
table entry.
"""

from __future__ import annotations

import os
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
    # Whether the provider takes a reasoning-depth level (EFFORT_LEVELS). False
    # is not "this provider cannot think" -- Gemini thinks too, but it is
    # configured by a token budget (thinking_budget), which is a different knob
    # with a different unit. A level is refused on such a provider instead of
    # converted into a budget: an invented equivalence would change the request
    # into something nobody asked for, on the most expensive parameter there is.
    supports_effort: bool = False
    # Whether the reported output_tokens already includes thinking. Every
    # provider bills thinking at the output rate, but the langchain-google-vertexai
    # adapter reports candidates and thoughts as two numbers and puts only the
    # candidates in output_tokens; the langchain-google-genai adapter to the same
    # models sums them, as Anthropic and OpenAI do. A reader that trusts
    # output_tokens on the first under-counts every thinking call by the whole
    # thought, and one that adds the thought on the second counts it twice.
    reasoning_in_output: bool = True


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
    supports_effort=True,
)

# A google_vertexai: id is served by one of two adapters (google_sdk), and the
# trait that differs between them is the adapter's, not the endpoint's.
_GEMINI_VERTEX_LEGACY = ProviderTraits(id="google_vertexai", reasoning_in_output=False)
_GEMINI_VERTEX_GENAI = ProviderTraits(id="google_vertexai")

_DEFAULT = ProviderTraits(id="")

_TRAITS: dict[str, ProviderTraits] = {
    "anthropic": _ANTHROPIC,
    "google_anthropic_vertex": _ANTHROPIC,
}


def google_sdk() -> str:
    """Which client stack backs google_vertexai: ids: "genai" (the default:
    ChatGoogleGenerativeAI on the google-genai SDK) or "vertexai" (the
    deprecated langchain-google-vertexai path). Same endpoints either way.
    Env-shaped (seeded from [gcp] sdk) so one deployment can revert without a
    code change while the old path still exists. Read here, without the
    framework, because the usage reader has to know which adapter's numbers it
    is looking at."""
    return os.environ.get("AIRC_GOOGLE_SDK", "genai")


def traits_for(model_id: str) -> ProviderTraits:
    """Traits for `model_id`'s provider; a neutral record for anything else.

    An unknown provider gets the default, not an error: this table is an
    optimization over asking the provider, and a provider missing from it has to
    keep working.
    """
    prefix = model_id.split(":", 1)[0]
    if prefix == "google_vertexai":
        return (
            _GEMINI_VERTEX_GENAI if google_sdk() == "genai" else _GEMINI_VERTEX_LEGACY
        )
    return _TRAITS.get(prefix, _DEFAULT)


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

# Anthropic's stop_reason when the classifier refuses a generation (arrives
# with empty content and 0 output tokens). Shared by _EmptyCandidateRetry and
# the reentry loop so neither layer retries a refused turn.
REFUSAL_STOP_REASON = "refusal"


# output_config.effort, in increasing depth. The Messages API's own vocabulary,
# so it is repeated here instead of derived: the SDK exposes it as a Literal
# that cannot be iterated, and config validation needs the set before any
# langchain import has happened.
#
# A provider taking a level is not the same as every model behind it taking one
# -- the older Claudes accept a shorter ladder or none at all, and that is a
# per-model fact no table here tracks. Config checks the vocabulary; the
# provider rejects a model that cannot serve the level asked for.
EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

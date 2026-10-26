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
from enum import Enum

# Every effort level any provider accepts, in increasing depth. Claude's
# output_config.effort vocabulary. The SDK exposes it as a Literal, and config
# validation runs before any langchain import, so the tuple is repeated here.
#
# Which levels a provider accepts is ProviderTraits.effort_levels. Whether a
# given model behind that provider accepts them is not tracked; the provider
# rejects the call.
EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

# Gemini's generation_config.thinking_config.thinking_level. The enum also has
# MINIMAL, which gemini-4-argon rejects, and EXTRA_HIGH/MAX, which the proto
# marks as not served by any Gemini model (gemini-4-argon accepts them and
# thinks no more than at HIGH). None of the three is offered.
_GEMINI_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high")


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
    # The effort levels the provider accepts, a subset of EFFORT_LEVELS; empty
    # when it accepts none. A level outside this set is refused, not mapped to
    # the nearest one: that would change the request without saying so.
    effort_levels: tuple[str, ...] = ()

    @property
    def supports_effort(self) -> bool:
        return bool(self.effort_levels)

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
    effort_levels=EFFORT_LEVELS,
)

# A google_vertexai: id is served by one of two adapters (google_sdk), and the
# traits that differ between them are the adapter's, not the endpoint's. Only
# the genai adapter has a thinking_level field; the deprecated vertexai one has
# thinking_budget alone, so it accepts no effort level.
_GEMINI_VERTEX_LEGACY = ProviderTraits(id="google_vertexai", reasoning_in_output=False)
_GEMINI_VERTEX_GENAI = ProviderTraits(
    id="google_vertexai", effort_levels=_GEMINI_EFFORT_LEVELS
)

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


def effort_problem(model_id: str, effort: str) -> str | None:
    """Why `model_id`'s provider cannot take `effort`, or None when it can.

    Shared by config (SystemExit at load) and make_model (ValueError) so the
    two messages stay the same. `effort` must already be in EFFORT_LEVELS.
    """
    traits = traits_for(model_id)
    if not traits.effort_levels:
        return f"{model_id} takes no effort level: its provider has no such parameter"
    if effort not in traits.effort_levels:
        return (
            f"{model_id} does not accept effort {effort!r}: its provider accepts"
            f" {', '.join(traits.effort_levels)}"
        )
    return None


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

# Structured provider stop reasons that identify a safety or policy classifier.
# Values outside this set retain their existing handling. Numeric enum values are
# intentionally not accepted because their meaning depends on the enum type.
CLASSIFIER_STOP_REASONS = frozenset(
    {
        REFUSAL_STOP_REASON,
        "safety",
        "recitation",
        "blocklist",
        "prohibited_content",
        "spii",
        "model_armor",
    }
)


def normalize_classifier_stop_reason(value: object) -> str:
    """Return the normalized classifier reason, or ``""`` when unrecognized."""
    if isinstance(value, Enum):
        value = value.name
    if not isinstance(value, str):
        return ""
    reason = value.strip().lower()
    return reason if reason in CLASSIFIER_STOP_REASONS else ""


# Where langchain-google-genai puts a prompt-side block: the API returns no
# candidate and a prompt_feedback with block_reason set, and the wrapper hands
# that back as an empty AIMessage with the feedback dumped into
# response_metadata under this key rather than raising. The Vertex wrapper does
# not surface prompt feedback at all, so this path is genai-only.
PROMPT_FEEDBACK_KEY = "prompt_feedback"
PROMPT_BLOCK_PREFIX = "prompt_blocked:"


def normalize_prompt_block_reason(metadata: dict | None) -> str:
    """Return ``prompt_blocked:<reason>`` for a blocked prompt, else ``""``.

    The block reason is an open enum (SAFETY, OTHER, BLOCKLIST,
    PROHIBITED_CONTENT, IMAGE_SAFETY, MODEL_ARMOR, JAILBREAK, ...); any set
    value other than the unspecified sentinel means the input was refused, so
    there is no allowlist here, unlike the candidate stop reasons.
    """
    feedback = (metadata or {}).get(PROMPT_FEEDBACK_KEY)
    if not isinstance(feedback, dict):
        return ""
    value = feedback.get("block_reason")
    if isinstance(value, Enum):
        value = value.name
    if not isinstance(value, str):
        return ""
    reason = value.strip().lower()
    if not reason or reason == "blocked_reason_unspecified":
        return ""
    return PROMPT_BLOCK_PREFIX + reason


def bare_model_name(model_id: str) -> str:
    """The provider-side name: "google_anthropic_vertex:claude-opus-5@20260701"
    -> "claude-opus-5". Vertex pins a version with "@"; the traits and price
    are the model's."""
    name = model_id.split(":", 1)[-1]
    return name.split("@", 1)[0]


@dataclass(frozen=True)
class ModelTraits:
    """What one checkpoint does differently across every provider that serves it,
    keyed by bare_model_name(model_id)."""

    id: str
    # Whether the checkpoint accepts forced tool_choice ("any" or a named tool).
    # False drops a forced tool_choice in bind_tools so the call goes out in
    # the provider's default "auto" mode instead of failing with 400.
    supports_forced_tool_choice: bool = True
    # Whether the checkpoint tends to run long agentic trajectories, many tool
    # calls before it concludes. A consumer that steers a run by call count can
    # read this to converge such a model earlier; a checkpoint that finishes in
    # a handful of calls needs no such pressure.
    slow_to_converge: bool = False


_DEFAULT_MODEL = ModelTraits(id="")

# claude-opus-5-5 rejects tool_choice types "any" and "tool" with 400
# ("tool_choice: type 'tool' and 'any' are not supported for this model"),
# whether or not thinking is configured in the request.
#
# gemini-3.1-pro-preview runs long: it uses several times the tool calls a
# Claude checkpoint spends on the same task before concluding.
_MODEL_TRAITS: dict[str, ModelTraits] = {
    "claude-opus-5-5": ModelTraits(
        id="claude-opus-5-5",
        supports_forced_tool_choice=False,
    ),
    "gemini-3.1-pro-preview": ModelTraits(
        id="gemini-3.1-pro-preview",
        slow_to_converge=True,
    ),
}


# Names that inherit another checkpoint's traits. Keyed by what the deploy
# wrote (a full "provider:name" id or a bare name) and valued by a bare name
# _MODEL_TRAITS lists, never another alias, so model_traits_for stays one lookup.
#
# Same purpose and same home as pricing's aliases ([pricing.aliases]): a model
# that is not its own listing but behaves like one -- a private deploy of a
# released checkpoint under an internal label, a sibling served ahead of its
# release -- can be told to take that checkpoint's traits without a code release.
# Filled from config ([traits.aliases], via config.load_common).
_TRAITS_ALIASES: dict[str, str] = {}


def _checkpoint_name(model_id: str) -> str:
    """The checkpoint name after provider prefix, version and vendor path."""
    return bare_model_name(model_id).rsplit("/", 1)[-1]


def register_traits_alias(model: str, as_model: str) -> None:
    """Give `model` (a full id or a bare name) `as_model`'s traits.

    A full-id alias is an exact key: versions and alternate vendor paths do not
    inherit it. Use a bare alias when every provider spelling should match.

    `as_model` must have a ModelTraits entry -- an alias onto the neutral record
    resolves to the same default `model` already gets, so it is dead config and
    refused, the way pricing refuses an alias to an unlisted name. Re-registering
    the same pair is a no-op (load_common runs more than once per process in some
    components); a CONFLICTING pair raises, since which traits applied would
    otherwise depend on parse order.
    """
    target = _checkpoint_name(as_model)
    if target not in _MODEL_TRAITS:
        raise ValueError(
            f"{model!r} cannot take the traits of {as_model!r}: no ModelTraits"
            f" entry for {target!r} (have: {', '.join(_MODEL_TRAITS)})"
        )
    if (prior := _TRAITS_ALIASES.get(model)) is not None and prior != target:
        raise ValueError(f"{model!r} already takes the traits of {prior!r}")
    _TRAITS_ALIASES[model] = target


def model_traits_for(model_id: str) -> ModelTraits:
    """Traits for `model_id`'s checkpoint; a neutral record for anything else.

    Aliases first, the full id before the bare name so an alias written for one
    provider's spelling does not also claim another's; then the checkpoint name
    itself. An aggregator names the checkpoint behind a vendor path, as in
    "openrouter:anthropic/claude-opus-5-5". The traits are the checkpoint's on
    every route, so the path is dropped here. Pricing keeps the full bare name,
    since the aggregator's rate is its own.
    """
    name = _checkpoint_name(model_id)
    key = _TRAITS_ALIASES.get(model_id) or _TRAITS_ALIASES.get(name) or name
    return _MODEL_TRAITS.get(key, _DEFAULT_MODEL)

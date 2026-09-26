# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""The provider traits table and the readers driven by it."""

from __future__ import annotations

from airc_core import model as model_mod
from airc_core.providers import STOP_REASON_KEYS, model_traits_for, traits_for


def test_both_anthropic_routes_share_one_record():
    """The direct API and the Vertex Model Garden route reach the same Messages
    API, so a fact learned on one holds for the other."""
    assert traits_for("anthropic:claude-opus-5") is traits_for(
        "google_anthropic_vertex:claude-opus-5"
    )


def test_an_unknown_provider_gets_a_neutral_record():
    """The table is an optimization over asking the provider. A provider missing
    from it keeps working, with nothing dropped and the common field name."""
    traits = traits_for("some_new_provider:m")
    assert traits.unsupported_kwargs == ()
    assert traits.stop_reason_keys == ("finish_reason",)


def test_stop_reason_keys_cover_every_provider_without_repeats():
    assert "finish_reason" in STOP_REASON_KEYS
    assert "stop_reason" in STOP_REASON_KEYS
    assert len(STOP_REASON_KEYS) == len(set(STOP_REASON_KEYS))


def test_the_stop_reason_is_read_under_every_providers_name():
    """The Anthropic path reports stop_reason and nothing else, so reading only
    finish_reason made every Claude empty candidate log "unknown" -- in the one
    place where the reason is the entire diagnostic."""
    from airc_core.agent import _finish_reason
    from langchain_core.messages import AIMessage

    def reason(**metadata):
        return _finish_reason(AIMessage("", response_metadata=metadata))

    assert reason(stop_reason="max_tokens") == "max_tokens"
    assert reason(finish_reason="STOP") == "STOP"
    assert reason() == "unknown"
    assert reason(finish_reason="") == "unknown"


def test_the_drop_is_driven_by_the_table(caplog):
    """The sampling kwargs Claude rejects are refused before the client sees
    them, and the caller is told once."""
    model_mod._UNSUPPORTED_WARNED.clear()
    kwargs = {"temperature": 0.7, "seed": 1, "max_tokens": 8}
    with caplog.at_level("WARNING", logger="airc_core.model"):
        model_mod._drop_unsupported_kwargs(kwargs, "google_anthropic_vertex:m")
    assert kwargs == {"max_tokens": 8}
    assert "does not accept" in caplog.text
    # The note from the table is included, since a dropped sampling kwarg means
    # a pass that wanted variance is not getting it.
    assert "no replacement" in caplog.text

    caplog.clear()
    kwargs = {"temperature": 0.7, "seed": 1, "max_tokens": 8}
    with caplog.at_level("WARNING", logger="airc_core.model"):
        model_mod._drop_unsupported_kwargs(kwargs, "google_anthropic_vertex:m")
    assert kwargs == {"max_tokens": 8}
    # Same model, same keys: a model is built per review pass and the condition
    # is static, so the repeat carries nothing new.
    assert caplog.text == ""


def test_a_provider_without_restrictions_keeps_its_kwargs():
    kwargs = {"temperature": 0.7, "seed": 1}
    model_mod._drop_unsupported_kwargs(kwargs, "google_vertexai:gemini")
    assert kwargs == {"temperature": 0.7, "seed": 1}


def test_model_traits_key_by_bare_name_across_providers_and_version_pins():
    """Checkpoint traits follow the model name whether served direct, on Vertex,
    or pinned with @version; unlisted models get the neutral record."""
    for mid in (
        "google_anthropic_vertex:claude-opus-5-5",
        "google_anthropic_vertex:claude-opus-5-5@20260901",
        "anthropic:claude-opus-5-5",
        "openrouter:anthropic/claude-opus-5-5",
    ):
        t = model_traits_for(mid)
        assert t.id == "claude-opus-5-5"
        assert t.supports_forced_tool_choice is False

    assert (
        model_traits_for(
            "google_anthropic_vertex:claude-opus-5"
        ).supports_forced_tool_choice
        is True
    )
    assert model_traits_for("some_new_provider:m").supports_forced_tool_choice is True


def test_slow_to_converge_marks_the_long_running_checkpoint():
    """gemini-3.1-pro-preview is flagged slow_to_converge across its routes and
    version pins; other checkpoints and unlisted models are not."""
    for mid in (
        "google_vertexai:gemini-3.1-pro-preview",
        "google_vertexai:gemini-3.1-pro-preview@20260901",
    ):
        t = model_traits_for(mid)
        assert t.id == "gemini-3.1-pro-preview"
        assert t.slow_to_converge is True

    assert model_traits_for("anthropic:claude-opus-5-5").slow_to_converge is False
    assert model_traits_for("some_new_provider:m").slow_to_converge is False

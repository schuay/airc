# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""A persona's [models] entry reaches the model it is built with.

The knob parses in airc_core.config and is only honoured if the build passes it
on, so what is worth pinning is the seam between the two: resolve_profile ->
make_model. Everything else _build_agent touches is stubbed -- a real graph
needs a live toolset and checkpointer and would test langchain, not this.
"""

import airc_room.runner as runner_mod
import pytest
from airc_core import ModelProfile
from airc_room.config import Config
from airc_room.personas import Persona
from airc_room.runner import AgentRunner


class _Toolset:
    instructions = ""

    def resolve_patterns(self, groups, tools, name):
        return []

    def tools_for(self, patterns):
        return []


@pytest.fixture
def built(tmp_path, monkeypatch):
    """(model_id, kwargs) of the make_model call _build_agent makes."""
    calls: list[tuple] = []
    monkeypatch.setattr(
        runner_mod, "make_model", lambda mid, **kw: calls.append((mid, kw))
    )
    monkeypatch.setattr(runner_mod, "base_middleware", lambda *a, **kw: [])
    monkeypatch.setattr(runner_mod, "growing_cache_middleware", lambda *a, **kw: None)
    monkeypatch.setattr(runner_mod, "create_agent", lambda *a, **kw: _Agent())
    return calls


class _Agent:
    def with_config(self, cfg):
        return self


def _build(tmp_path, cfg, persona_model_id=None):
    cfg.token_db_path = tmp_path / "tokens.db"
    cfg.db_path = tmp_path / "airc.db"
    runner = AgentRunner(cfg, {}, _Toolset(), object())
    persona = Persona(
        name="gc",
        display_name="gc",
        description="d",
        system_prompt="",
        key="gc",
        model_id=persona_model_id,
    )
    runner._build_agent(persona, {}, object())


def test_the_roles_effort_reaches_the_constructed_model(tmp_path, built):
    cfg = Config(default_model="google_anthropic_vertex:claude-opus-5")
    cfg.model_profiles = {
        "default": ModelProfile(
            key="default", id="google_anthropic_vertex:claude-opus-5", effort="low"
        )
    }
    _build(tmp_path, cfg)
    assert built == [("google_anthropic_vertex:claude-opus-5", {"effort": "low"})]


def test_an_entry_without_an_effort_sends_no_knob(tmp_path, built):
    """Unset must stay unset: make_model turns an effort into output_config, and
    sending the API default explicitly is not the same as sending nothing."""
    cfg = Config(default_model="google_vertexai:gemini-3.6-flash")
    _build(tmp_path, cfg)
    assert built == [("google_vertexai:gemini-3.6-flash", {})]

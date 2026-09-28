# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Typed room turns with an explicit tool grant."""

from types import SimpleNamespace

from airc_room.config import Config
from airc_room.personas import Persona
from airc_room.runner import AgentRunner, _AgentEntry
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel


class Digest(BaseModel):
    summary: str


@tool
def _read() -> str:
    """Read source."""
    return "source"


@tool
def _room_action() -> str:
    """Change room state."""
    return "changed"


class _ScriptedModel(BaseChatModel):
    calls: int = 0
    bound_names: list[list[str]] = []

    @property
    def _llm_type(self) -> str:
        return "room-structured-test"

    def bind_tools(self, tools, **kwargs):
        self.bound_names.append([tool.name for tool in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        if self.calls == 0:
            message = AIMessage(
                content="",
                tool_calls=[{"name": "_read", "args": {}, "id": "read"}],
            )
        else:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Digest",
                        "args": {"summary": "checked source"},
                        "id": "result",
                    }
                ],
            )
        object.__setattr__(self, "calls", self.calls + 1)
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_result_tool_is_cached_but_not_registered_as_an_action(tmp_path, monkeypatch):
    from airc_room import runner as runner_module

    seen = {}

    def base(model_id, system_prompt, tools, **kwargs):
        seen["cached"] = [tool.name for tool in tools]
        return []

    class Graph:
        def with_config(self, config):
            return self

    def create(model, *, tools, response_format, **kwargs):
        seen["actions"] = [tool.name for tool in tools]
        seen["schema"] = response_format.schema
        return Graph()

    monkeypatch.setattr(runner_module, "base_middleware", base)
    monkeypatch.setattr(runner_module, "growing_cache_middleware", lambda *a, **k: None)
    monkeypatch.setattr(runner_module, "make_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner_module, "create_agent", create)
    cfg = Config(default_model="test:model", filter_model="")
    cfg.token_db_path = tmp_path / "tokens.db"
    runner = AgentRunner(
        cfg,
        {},
        SimpleNamespace(instructions=""),
        object(),
        structured_tools={"digest": [_read]},
    )
    persona = Persona(
        name="perf",
        display_name="Perf",
        description="d",
        system_prompt="",
        key="perf",
    )

    runner._build_agent(
        persona,
        {"perf": persona},
        None,
        extra_system="return a digest",
        structured_label="digest",
        schema=Digest,
    )

    assert seen == {
        "cached": ["_read", "Digest"],
        "actions": ["_read"],
        "schema": Digest,
    }


async def test_structured_turn_uses_its_grant_and_returns_the_schema(
    tmp_path, monkeypatch
):
    from airc_room import runner as runner_module

    model = _ScriptedModel()
    monkeypatch.setattr(runner_module, "make_model", lambda *args, **kwargs: model)
    cfg = Config(default_model="test:model", filter_model="")
    cfg.token_db_path = tmp_path / "tokens.db"
    cfg.grounding_reminder_tokens = 0

    runner = AgentRunner(
        cfg,
        {},
        SimpleNamespace(instructions=""),
        object(),
        local_tools=[_room_action],
        structured_tools={"digest": [_read]},
    )
    persona = Persona(
        name="perf",
        display_name="Perf",
        description="d",
        system_prompt="",
        key="perf",
    )
    runner._agents = {"perf": _AgentEntry(persona=persona, graph=None)}

    result = await runner.run_structured_turn(
        "perf",
        "commit",
        extra_system="return a digest",
        schema=Digest,
        label="digest",
    )

    assert result == Digest(summary="checked source")
    assert model.calls == 2
    assert model.bound_names
    assert all(
        "_read" in names and "_room_action" not in names for names in model.bound_names
    )

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""The API-v2 built-in allowlist and persona-gated local tool groups."""

import pytest
from airc_room.cli import _call_local_tools
from airc_room.config import Config
from airc_room.personas import Persona
from airc_room.plugin import LocalTools
from airc_room.runner import AgentRunner
from langchain_core.tools import tool


@tool
def _mem() -> str:
    """a stub memory tool"""
    return "ok"


def _persona(name, groups):
    return Persona(
        name=name,
        display_name=name,
        description="d",
        system_prompt="",
        key=name,
        tool_groups=tuple(groups),
    )


def _runner(tmp_path, local_tool_groups, local_tools=None):
    cfg = Config()
    cfg.token_db_path = tmp_path / "tokens.db"
    return AgentRunner(
        cfg,
        {},
        object(),
        object(),
        local_tools=local_tools,
        local_tool_groups=local_tool_groups,
    )


def test_local_group_granted_only_to_persona_listing_it(tmp_path):
    runner = _runner(tmp_path, {"memory": [_mem]})
    granted = _persona("chef", ["read", "memory"])
    withheld = _persona("aide", ["read"])

    def local_for(persona):
        out = []
        for group in persona.tool_groups:
            out.extend(runner._local_tool_groups.get(group, []))
        return out

    assert local_for(granted) == [_mem]
    assert local_for(withheld) == []


def test_local_groups_excluded_from_mcp_resolution(tmp_path):
    runner = _runner(tmp_path, {"memory": [_mem]})
    persona = _persona("chef", ["read", "memory"])
    mcp_groups = [
        group for group in persona.tool_groups if group not in runner._local_tool_groups
    ]
    assert mcp_groups == ["read"]


def test_built_in_tools_are_a_separate_baseline(tmp_path):
    runner = _runner(tmp_path, {"memory": [_mem]}, local_tools=[_mem])
    assert runner._local_tools == [_mem]


def test_no_local_tools_by_default(tmp_path):
    runner = _runner(tmp_path, None)
    assert runner._local_tools == []
    assert runner._local_tool_groups == {}


class _Plugin:
    def build_local_tools(self, cfg, *, room):
        return LocalTools(allowlist=("search_chat",), groups={"icu_tasks": [room]})


class _RaisingPlugin:
    def __init__(self):
        self.calls = 0

    def build_local_tools(self, cfg, *, room):
        self.calls += 1
        raise TypeError("a bug inside the hook body")


def test_room_is_passed_to_the_v2_hook():
    room = object()
    policy = _call_local_tools(_Plugin(), None, room)
    assert policy.allowlist == ("search_chat",)
    assert policy.groups == {"icu_tasks": [room]}


def test_the_v1_dict_return_is_rejected_with_the_migration():
    class _V1Plugin:
        def build_local_tools(self, cfg, *, room):
            return {"memory": [_mem]}

    with pytest.raises(SystemExit, match=r"not airc_room\.plugin\.LocalTools"):
        _call_local_tools(_V1Plugin(), None, object())


def test_a_raising_hook_propagates_and_is_called_once():
    plugin = _RaisingPlugin()
    with pytest.raises(TypeError):
        _call_local_tools(plugin, None, object())
    assert plugin.calls == 1


def test_plugin_tool_instructions_join_the_toolsets(tmp_path):
    from airc_room.runner import AgentRunner

    class _Toolset:
        instructions = "from the server"

    cfg = Config()
    cfg.token_db_path = tmp_path / "tokens.db"
    runner = AgentRunner(cfg, {}, _Toolset(), object(), tool_instructions="from us")
    assert runner._instructions() == "from the server\n\nfrom us"
    runner = AgentRunner(cfg, {}, _Toolset(), object())
    assert runner._instructions() == "from the server"

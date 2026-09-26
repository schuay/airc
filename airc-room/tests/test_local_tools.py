# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Plugin local-tool groups: a persona gets a group's local tools iff it lists
the group -- the same gate MCP tools use.

_build_agent needs a live MCPToolset and checkpointer to build a full graph, so
this tests the gating logic in isolation: the same set operations _build_agent
performs over persona.tool_groups and the runner's local_tool_groups.
"""

import pytest
from airc_room.cli import _call_local_tools
from airc_room.config import Config
from airc_room.personas import Persona
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


def _runner(tmp_path, local_tool_groups):
    cfg = Config()
    cfg.token_db_path = tmp_path / "tokens.db"
    return AgentRunner(cfg, {}, object(), object(), local_tool_groups=local_tool_groups)


def test_local_group_granted_only_to_persona_listing_it(tmp_path):
    runner = _runner(tmp_path, {"memory": [_mem]})
    # The gate _build_agent applies: local tools for each of the persona's groups.
    granted = _persona("chef", ["read", "memory"])
    withheld = _persona("aide", ["read"])

    def local_for(p):
        out = []
        for g in p.tool_groups:
            out.extend(runner._local_tool_groups.get(g, []))
        return out

    assert local_for(granted) == [_mem]
    assert local_for(withheld) == []


def test_local_groups_excluded_from_mcp_resolution(tmp_path):
    # A plugin-local group must not be sent to the MCP resolver (it would log an
    # unknown-group warning); _build_agent filters it out first.
    runner = _runner(tmp_path, {"memory": [_mem]})
    p = _persona("chef", ["read", "memory"])
    mcp_groups = [g for g in p.tool_groups if g not in runner._local_tool_groups]
    assert mcp_groups == ["read"]


def test_no_local_groups_by_default(tmp_path):
    runner = _runner(tmp_path, None)
    assert runner._local_tool_groups == {}


# ── the hook signature: `room` is optional, so both forms must load ──────────


class _NewPlugin:
    def build_local_tools(self, cfg, *, room=None):
        return {"icu_tasks": [room]}


class _OldPlugin:
    def build_local_tools(self, cfg):
        return {"memory": [_mem]}


class _KwargsPlugin:
    def build_local_tools(self, cfg, **kwargs):
        return {"icu_tasks": [kwargs.get("room")]}


class _RaisingPlugin:
    def __init__(self):
        self.calls = 0

    def build_local_tools(self, cfg, *, room=None):
        self.calls += 1
        raise TypeError("a bug inside the hook body")


def test_room_is_passed_to_a_hook_that_takes_it():
    room = object()
    assert _call_local_tools(_NewPlugin(), None, room) == {"icu_tasks": [room]}


def test_room_reaches_a_kwargs_hook():
    # **kwargs is the forward-compat idiom, so a plugin that wrote it
    # to receive a later addition must not be the one form that silently misses
    # it: the tool would build with room=None and quietly fail to post.
    room = object()
    assert _call_local_tools(_KwargsPlugin(), None, room) == {"icu_tasks": [room]}


def test_a_hook_without_room_still_loads():
    # The compatibility property that lets this land without a
    # PLUGIN_API_VERSION bump: an out-of-tree plugin written against
    # build_local_tools(cfg) must keep contributing its tools.
    assert _call_local_tools(_OldPlugin(), None, object()) == {"memory": [_mem]}


def test_a_raising_hook_propagates_and_is_called_exactly_once():
    # Why the dispatch inspects the signature instead of calling with the keyword
    # and catching TypeError. Both halves matter and only together do they rule
    # the fallback out: the error must reach the operator instead of being read
    # as "this plugin ships no local tools", AND the hook body must not run a
    # second time -- a retry would repeat whatever it did before it raised.
    plugin = _RaisingPlugin()
    with pytest.raises(TypeError):
        _call_local_tools(plugin, None, object())
    assert plugin.calls == 1


def test_plugin_tool_instructions_join_the_toolsets(tmp_path):
    """A plugin that builds tools in-process describes them the way an MCP
    server does through its instructions; both reach the prompt together."""
    from airc_room.runner import AgentRunner

    class _Toolset:
        instructions = "from the server"

    cfg = Config()
    cfg.token_db_path = tmp_path / "tokens.db"
    runner = AgentRunner(cfg, {}, _Toolset(), object(), tool_instructions="from us")
    assert runner._instructions() == "from the server\n\nfrom us"
    runner = AgentRunner(cfg, {}, _Toolset(), object())
    assert runner._instructions() == "from the server"

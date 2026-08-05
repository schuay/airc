# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Plugin local-tool groups: a persona gets a group's local tools iff it lists
the group -- the same gate MCP tools use.

_build_agent needs a live MCPToolset and checkpointer to build a full graph, so
this tests the gating logic in isolation: the same set operations _build_agent
performs over persona.tool_groups and the runner's local_tool_groups.
"""

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

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

import asyncio
from dataclasses import dataclass

from airc_core import mcptools
from airc_core.mcptools import MCPToolset, _clean_schema, _fix_tool

# The default read/active groups, mirrored from airc's config so the filtering
# tests exercise the real patterns without depending on the airc package.
TOOL_GROUPS = {
    "read": [
        "v8-utils__repo_git_*",
        "v8-utils__gerrit_comments",
        "v8-utils__gerrit_cq",
        "v8-utils__gerrit_fetch",
        "v8-utils__gerrit_list_cls",
        "v8-utils__godbolt_*",
        "v8-utils__pinpoint_show_*",
        "v8-utils__pinpoint_list_jobs",
        "v8-utils__pd_*",
    ],
    "active": [
        "v8-utils__run_d8",
        "v8-utils__jsb_run_bench",
        "v8-utils__perf_*",
        "v8-utils__llvm_mca",
        "v8-utils__v8log_analyze",
    ],
}


@dataclass
class FakeTool:
    name: str


def make_toolset(tool_names):
    ts = MCPToolset(mcp_servers={}, tool_groups=TOOL_GROUPS)
    ts.tools = [FakeTool(n) for n in tool_names]
    return ts


TOOLS = [
    "v8-utils__repo_git_grep",
    "v8-utils__repo_git_show",
    "v8-utils__gerrit_comments",
    "v8-utils__run_d8",
    "v8-utils__jsb_run_bench",
    "v8-utils__perf_hotspots",
    "v8-utils__llvm_mca",
    "v8-utils__godbolt_compile",
]


def filtered(ts, groups=(), tools=()):
    """Resolve groups+explicit patterns to tools, the way a caller (runner,
    processors) drives the decoupled toolset."""
    patterns = ts.resolve_patterns(groups, tools, label="x")
    return {t.name for t in ts.tools_for(patterns)}


def test_read_group():
    ts = make_toolset(TOOLS)
    names = filtered(ts, groups=["read"])
    assert "v8-utils__repo_git_grep" in names
    assert "v8-utils__gerrit_comments" in names
    assert "v8-utils__godbolt_compile" in names
    assert "v8-utils__run_d8" not in names
    assert "v8-utils__perf_hotspots" not in names


def test_active_group():
    ts = make_toolset(TOOLS)
    names = filtered(ts, groups=["read", "active"])
    assert {
        "v8-utils__run_d8",
        "v8-utils__jsb_run_bench",
        "v8-utils__perf_hotspots",
        "v8-utils__llvm_mca",
    } <= names


def test_explicit_tools():
    ts = make_toolset(TOOLS)
    names = filtered(ts, groups=["read"], tools=["v8-utils__llvm_mca"])
    assert "v8-utils__llvm_mca" in names
    assert "v8-utils__run_d8" not in names


def test_no_groups_no_tools():
    ts = make_toolset(TOOLS)
    assert filtered(ts) == set()


def test_unknown_group_ignored(caplog):
    import logging

    ts = make_toolset(TOOLS)
    with caplog.at_level(logging.WARNING, logger="airc_core.mcptools"):
        assert filtered(ts, groups=["bogus"]) == set()
    # The requester label surfaces in the warning so a typo is traceable.
    assert "x references unknown tool group 'bogus'" in caplog.text


def test_empty_group_warns(caplog):
    import logging

    # A config with no [tool_groups] block leaves the group known but empty; the
    # unknown-group check passes over it, so without this warning every persona
    # boots tool-less silently (the split's D6 trap).
    ts = MCPToolset(mcp_servers={}, tool_groups={"read": [], "active": []})
    ts.tools = [FakeTool(n) for n in TOOLS]
    with caplog.at_level(logging.WARNING, logger="airc_core.mcptools"):
        assert filtered(ts, groups=["read"]) == set()
    assert "x references tool group 'read', but it is empty" in caplog.text


def test_nonempty_group_does_not_warn_empty(caplog):
    import logging

    # A populated group must not trip the empty-group warning.
    ts = make_toolset(TOOLS)
    with caplog.at_level(logging.WARNING, logger="airc_core.mcptools"):
        filtered(ts, groups=["read"])
    assert "it is empty" not in caplog.text


def test_resolve_patterns_label_defaults_to_toolset():
    ts = make_toolset(TOOLS)
    # No label supplied: the warning still identifies the source generically.
    assert ts.resolve_patterns(["bogus"]) == []


async def test_tool_timeout_surfaces_as_error_tool_message(monkeypatch):
    # A hung tool call must come back to the agent as an error ToolMessage, not
    # escape as an exception: BaseTool.arun only routes ToolException through
    # handle_tool_error, so a bare TimeoutError would kill the whole turn (and
    # in the review graph be misread as the review wall-clock). capped()
    # converts the expiry; assert the full arun path yields status="error".
    from langchain_core.tools import StructuredTool

    async def hang() -> str:
        await asyncio.sleep(30)
        return "never"

    tool = _fix_tool(
        StructuredTool.from_function(coroutine=hang, name="hang", description="hangs")
    )
    monkeypatch.setattr(mcptools, "_TOOL_CALL_TIMEOUT_S", 0.01)
    msg = await tool.ainvoke(
        {"type": "tool_call", "name": "hang", "args": {}, "id": "tc1"}
    )
    assert msg.status == "error"
    assert "timed out" in str(msg.content)


def test_clean_schema():
    schema = {
        "type": "object",
        "title": "Args",
        "additionalProperties": False,
        "properties": {
            "xs": {"type": "array", "title": "Xs"},
            "nested": {"$schema": "x", "type": "object", "properties": {}},
        },
    }
    _clean_schema(schema)
    assert "title" not in schema
    assert "additionalProperties" not in schema
    assert schema["properties"]["xs"]["items"] == {}
    assert "$schema" not in schema["properties"]["nested"]


class _FakeSession:
    """A session that either initializes and serves tools, or fails on start."""

    def __init__(self, tools, fail: bool):
        self._tools = tools
        self._fail = fail

    async def __aenter__(self):
        if self._fail:
            raise RuntimeError("spawn failed: no such file")
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        @dataclass
        class _Init:
            instructions: str

        return _Init(instructions="")


async def test_one_dead_server_does_not_sink_the_others(monkeypatch):
    # A stdio MCP server is an arbitrary external binary: absent, crashing, or
    # behind an expired corp credential. Losing every other server's tools (and
    # with them the room, or a job that never wanted the dead server) because one
    # would not start is a fault-isolation failure, so a bad server is dropped
    # and the toolset opens with what did come up.
    from langchain_mcp_adapters import client as adapters_client
    from langchain_mcp_adapters import tools as adapters_tools

    servers = {"good": {"command": "good-mcp"}, "dead": {"command": "missing-mcp"}}
    sessions = {
        "good": _FakeSession([FakeTool("run_d8")], fail=False),
        "dead": _FakeSession([], fail=True),
    }

    class _FakeClient:
        def __init__(self, _servers):
            pass

        def session(self, name, auto_initialize=True):
            return sessions[name]

    async def fake_load(sess, server_name=""):
        return list(sess._tools)

    monkeypatch.setattr(adapters_client, "MultiServerMCPClient", _FakeClient)
    monkeypatch.setattr(adapters_tools, "load_mcp_tools", fake_load)
    monkeypatch.setattr(mcptools, "_fix_tool", lambda t: t)

    async with MCPToolset(servers, TOOL_GROUPS) as ts:
        assert [t.name for t in ts.tools] == ["good__run_d8"]

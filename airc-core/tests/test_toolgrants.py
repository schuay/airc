# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

import logging
from types import SimpleNamespace

import pytest
from airc_core.toolgrants import (
    ToolBuildContext,
    ToolCatalog,
    ToolGrant,
    ToolSource,
    ToolSpec,
    select_tools,
)


def _tools(*names):
    return [SimpleNamespace(name=name) for name in names]


def test_allowlist_selects_exact_names_and_patterns_in_candidate_order():
    tools = _tools("repo_show", "shell", "repo_grep", "write_file")
    got = select_tools(tools, ("repo_*", "write_file"), label="test")
    assert [tool.name for tool in got] == ["repo_show", "repo_grep", "write_file"]


def test_empty_allowlist_selects_nothing():
    assert select_tools(_tools("shell"), (), label="test") == []


def test_duplicate_candidate_name_is_refused_even_when_not_selected():
    with pytest.raises(ValueError, match="duplicate tool name 'shell'"):
        select_tools(_tools("shell", "shell"), (), label="consumer")


def test_unnamed_candidate_is_refused():
    with pytest.raises(ValueError, match="consumer: a tool has no name"):
        select_tools([SimpleNamespace()], ("*",), label="consumer")


def test_grant_selects_before_constructing_and_retains_catalog_order():
    built = []

    def factory(name):
        def build(_context):
            built.append(name)
            return SimpleNamespace(name=name)

        return build

    catalog = ToolCatalog(
        tuple(
            ToolSpec(name, factory(name))
            for name in ("repo_show", "write_file", "repo_grep")
        )
    )
    grant = ToolGrant(catalog, required=("repo_*",))

    resolved = grant.resolve(label="consumer")
    assert built == []
    assert [tool.name for tool in resolved.build(ToolSource.NATIVE)] == [
        "repo_show",
        "repo_grep",
    ]
    assert built == ["repo_show", "repo_grep"]


def test_missing_required_grant_fails_without_running_factories():
    built = []
    catalog = ToolCatalog(
        (
            ToolSpec(
                "repo_show",
                lambda _context: built.append(True),
            ),
        )
    )

    with pytest.raises(ValueError, match=r"required tool grant.*'repo_grep'"):
        ToolGrant(catalog, required=("repo_grep",)).resolve(label="consumer")
    assert built == []


def test_missing_optional_grant_is_valid_and_reported(caplog):
    caplog.set_level(logging.INFO)
    grant = ToolGrant(optional=("run_d8",))
    resolved = grant.resolve(label="consumer")

    assert resolved.build(ToolSource.NATIVE) == []
    assert "optional tool grant(s) unavailable: run_d8" in caplog.text


def test_additional_catalog_completes_a_grant():
    core = ToolCatalog.from_tools(_tools("search_chat"))
    plugin = ToolCatalog.from_tools(_tools("lookup"))
    grant = ToolGrant(plugin, required=("search_chat", "lookup"))

    resolved = grant.resolve(core, label="room")
    assert [tool.name for tool in resolved.build(ToolSource.NATIVE)] == [
        "search_chat",
        "lookup",
    ]


def test_worktree_factory_requires_and_receives_context(tmp_path):
    seen = []
    catalog = ToolCatalog(
        (
            ToolSpec(
                "shell",
                lambda context: seen.append(context) or SimpleNamespace(name="shell"),
                ToolSource.WORKTREE,
            ),
        )
    )
    resolved = ToolGrant(catalog, required=("shell",)).resolve(label="worker")

    with pytest.raises(ValueError, match="worktree tools need a workdir"):
        resolved.build(ToolSource.WORKTREE)
    context = ToolBuildContext(tmp_path, 10.0)
    assert [tool.name for tool in resolved.build(ToolSource.WORKTREE, context)] == [
        "shell"
    ]
    assert seen == [context]


def test_factory_must_return_its_declared_name():
    catalog = ToolCatalog(
        (ToolSpec("declared", lambda _context: SimpleNamespace(name="actual")),)
    )
    resolved = ToolGrant(catalog, required=("declared",)).resolve(label="consumer")

    with pytest.raises(ValueError, match=r"expected \['declared'\]"):
        resolved.build(ToolSource.NATIVE)


def test_wrapper_applies_to_native_worktree_and_mcp_sources(tmp_path):
    seen = []

    def wrapper(tools):
        seen.append([tool.name for tool in tools])
        return tools

    catalog = ToolCatalog(
        (
            ToolSpec("native", lambda _: SimpleNamespace(name="native")),
            ToolSpec(
                "shell",
                lambda _: SimpleNamespace(name="shell"),
                ToolSource.WORKTREE,
            ),
        )
    )
    resolved = ToolGrant(
        catalog, required=("native", "shell"), wrapper=wrapper
    ).resolve(label="consumer")

    resolved.build(ToolSource.NATIVE)
    resolved.build(ToolSource.WORKTREE, ToolBuildContext(tmp_path))
    resolved.wrap(_tools("server__read"), ToolSource.MCP)
    assert seen == [["native"], ["shell"], ["server__read"]]

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest
from airc_core.toolgrants import select_tools


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

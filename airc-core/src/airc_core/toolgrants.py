# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Selection of caller-built tools by explicit name allowlist."""

from __future__ import annotations

from fnmatch import fnmatchcase


def select_tools(candidates, allowlist, *, label: str) -> list:
    """Select candidates whose names match an allowlist entry.

    Candidate order is retained. Duplicate or empty names are rejected before
    filtering: an unselected duplicate is still an ambiguous catalog whose
    behavior would change when the allowlist changes.
    """
    tools = list(candidates)
    patterns = tuple(str(pattern) for pattern in allowlist)
    seen: set[str] = set()
    for tool in tools:
        name = getattr(tool, "name", "")
        if not name:
            raise ValueError(f"{label}: a tool has no name")
        if name in seen:
            raise ValueError(f"{label}: duplicate tool name {name!r}")
        seen.add(name)
    return [
        tool
        for tool in tools
        if any(fnmatchcase(tool.name, pattern) for pattern in patterns)
    ]

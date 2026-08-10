# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Core does not depend on aisan, in either the import graph or the metadata.

The bind model left airc-tools for its own package (`aisan`), and what makes that
a removal rather than a relocation is that the arrow does not come back: aisan
imports nothing of core's (asserted on its side by tests/test_aisan_boundary.py),
and core imports nothing of aisan's. Neither half is implied by the other, and
neither is visible in a suite that only exercises behaviour -- the day a helper in
airc_tools.shell reaches for `aisan.sandbox` to build a wrapper, every other test
here still passes and the graph has quietly grown a cycle.

Stated as a rule about the whole repo rather than about this package, because the
edge could appear in any of the five and this is the one that used to own the code.
"""

from __future__ import annotations

import ast
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[2]

# Every core package's importable source. Derived from the workspace member list
# rather than a glob, so a new member is covered the moment it is declared -- and
# a member that is renamed makes this fail loudly instead of silently skipping.
_MEMBERS = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["uv"][
    "workspace"
]["members"]


def _sources() -> list[Path]:
    return sorted(
        p for m in _MEMBERS for p in (REPO / m / "src").rglob("*.py") if p.is_file()
    )


def test_there_are_sources_to_check():
    # The rest of this file loops over a glob, and an empty glob passes. A layout
    # change that empties it would otherwise leave a green no-op behind.
    assert len(_sources()) >= 10, _MEMBERS


def test_no_core_module_imports_aisan():
    bad: list[str] = []
    for path in _sources():
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # Relative imports stay inside the package that wrote them.
                names = [node.module or ""] if not node.level else []
            else:
                continue
            for name in names:
                if name.split(".")[0] == "aisan":
                    bad.append(f"{path.relative_to(REPO)}:{node.lineno}: {name}")
    assert not bad, (
        "core imports aisan -- the sandbox came back into the layer it left:\n"
        + "\n".join(bad)
    )


def test_no_core_package_depends_on_aisan():
    """And not in the metadata either, which is the other half of the same claim.

    An import is what breaks; a dependency is what permits it. A package that
    declares aisan without importing it yet has already made the next import a
    one-line change that no other test would notice, so both are checked -- across
    dependency groups and extras too, since an optional edge is still an edge.
    """
    bad: list[str] = []
    for member in _MEMBERS:
        meta = tomllib.loads((REPO / member / "pyproject.toml").read_text())
        project = meta.get("project", {})
        groups = [
            ("dependencies", project.get("dependencies", [])),
            *(
                (f"optional-dependencies.{k}", v)
                for k, v in project.get("optional-dependencies", {}).items()
            ),
            *(
                (f"dependency-groups.{k}", v)
                for k, v in meta.get("dependency-groups", {}).items()
            ),
        ]
        for where, reqs in groups:
            for req in reqs:
                # The requirement name is everything up to the first version
                # specifier, extra, or marker; matched exactly so `aisan-extras`
                # would not trip it.
                name = req.split("[")[0].split(";")[0].strip()
                name = name.split("=")[0].split(">")[0].split("<")[0].split("@")[0]
                if name.strip().replace("_", "-").lower() == "aisan":
                    bad.append(f"{member}/pyproject.toml [{where}]: {req}")
    assert not bad, "core declares a dependency on aisan:\n" + "\n".join(bad)

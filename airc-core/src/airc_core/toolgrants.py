# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Deny-by-default grants for tools constructed inside an application."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fnmatch import fnmatchcase
from pathlib import Path

log = logging.getLogger(__name__)


class ToolSource(StrEnum):
    """Where a granted tool is constructed and bound."""

    NATIVE = "native"
    WORKTREE = "worktree"
    MCP = "mcp"


@dataclass(frozen=True)
class ToolBuildContext:
    """Values available when a selected catalog entry is constructed."""

    workdir: Path | None = None
    shell_timeout_s: float = 300.0


@dataclass(frozen=True)
class ToolSpec:
    """The name and deferred constructor for one non-MCP tool."""

    name: str
    factory: Callable[[ToolBuildContext], object]
    source: ToolSource = ToolSource.NATIVE

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool catalog: a tool spec has no name")
        if not callable(self.factory):
            raise TypeError(f"tool catalog: factory for {self.name!r} is not callable")


@dataclass(frozen=True)
class ToolCatalog:
    """Immutable name-to-factory catalog.

    Names are available without calling factories, so grants are resolved before
    an ungranted tool imports dependencies or initializes state.
    """

    specs: tuple[ToolSpec, ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for spec in self.specs:
            if spec.name in seen:
                raise ValueError(f"tool catalog: duplicate tool name {spec.name!r}")
            seen.add(spec.name)

    @classmethod
    def from_factories(
        cls,
        factories: Mapping[str, Callable[[ToolBuildContext], object]],
        *,
        source: ToolSource = ToolSource.NATIVE,
    ) -> ToolCatalog:
        return cls(
            tuple(
                ToolSpec(name=str(name), factory=factory, source=source)
                for name, factory in factories.items()
            )
        )

    @classmethod
    def from_tools(
        cls,
        tools: Sequence[object],
        *,
        source: ToolSource = ToolSource.NATIVE,
    ) -> ToolCatalog:
        """Adapt already-built tools, mainly for tests and compatibility seams."""

        specs = []
        for tool in tools:
            name = getattr(tool, "name", "")
            specs.append(
                ToolSpec(
                    name=name,
                    factory=lambda _context, tool=tool: tool,
                    source=source,
                )
            )
        return cls(tuple(specs))

    def merged(self, *others: ToolCatalog) -> ToolCatalog:
        return ToolCatalog(
            tuple(spec for catalog in (self, *others) for spec in catalog.specs)
        )


@dataclass(frozen=True)
class ToolGrant:
    """A catalog plus the names a consumer requires or accepts when present."""

    catalog: ToolCatalog = ToolCatalog()
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    wrapper: Callable[[list], list] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "required", tuple(map(str, self.required)))
        object.__setattr__(self, "optional", tuple(map(str, self.optional)))
        if self.wrapper is not None and not callable(self.wrapper):
            raise TypeError("tool grant: wrapper is not callable")

    def resolve(
        self, *additional_catalogs: ToolCatalog, label: str
    ) -> ResolvedToolGrant:
        """Validate grants against the complete catalog without building tools."""

        catalog = ToolCatalog().merged(*additional_catalogs, self.catalog)
        names = tuple(spec.name for spec in catalog.specs)
        missing = [
            pattern
            for pattern in self.required
            if not any(fnmatchcase(name, pattern) for name in names)
        ]
        if missing:
            raise ValueError(
                f"{label}: required tool grant(s) matched nothing: "
                + ", ".join(repr(pattern) for pattern in missing)
            )
        unavailable = [
            pattern
            for pattern in self.optional
            if not any(fnmatchcase(name, pattern) for name in names)
        ]
        if unavailable:
            log.info(
                "%s: optional tool grant(s) unavailable: %s",
                label,
                ", ".join(unavailable),
            )
        patterns = (*self.required, *self.optional)
        specs = tuple(
            spec
            for spec in catalog.specs
            if any(fnmatchcase(spec.name, pattern) for pattern in patterns)
        )
        return ResolvedToolGrant(specs, self.wrapper, label)


@dataclass(frozen=True)
class ResolvedToolGrant:
    """A validated selection whose factories have not yet run."""

    specs: tuple[ToolSpec, ...]
    wrapper: Callable[[list], list] | None
    label: str

    def build(
        self,
        source: ToolSource,
        context: ToolBuildContext | None = None,
    ) -> list:
        context = context or ToolBuildContext()
        if source is ToolSource.WORKTREE and context.workdir is None:
            raise ValueError(f"{self.label}: worktree tools need a workdir")
        tools = [spec.factory(context) for spec in self.specs if spec.source is source]
        expected = [spec.name for spec in self.specs if spec.source is source]
        actual = [getattr(tool, "name", "") for tool in tools]
        if actual != expected:
            raise ValueError(
                f"{self.label}: tool factories returned {actual!r}, expected {expected!r}"
            )
        return self.wrap(tools, source)

    def wrap(self, tools: Sequence[object], source: ToolSource) -> list:
        wrapped = list(tools)
        if self.wrapper is not None:
            wrapped = list(self.wrapper(wrapped))
        return select_tools(
            wrapped,
            ("*",),
            label=f"{self.label} {source.value} tools",
        )


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

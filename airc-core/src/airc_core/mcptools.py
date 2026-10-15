# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""MCP tool loading and pattern-based filtering.

Sessions are opened once at daemon startup and stay alive for the daemon's
lifetime (the room is a long-running process; stateful servers like v8-utils
benefit from a persistent subprocess). Nothing reopens one: a server that dies
mid-life, or never started, stays gone until the daemon restarts. The cost is
bounded: every failure mode (server-reported error, timeout, dead transport)
is converted to tool-result text via
handle_tool_error, so an unusable server degrades to per-call errors the agent
can read and work around, never a traceback out of the turn.

Tool names are exposed unprefixed; if two servers ever export the same name
the second one wins and a warning is logged.

This is suite-shared substrate: it takes the mcp-servers dict and tool_groups
mapping directly (not a Config) and filters by tool-name patterns (not a
Persona), so every component -- persona turns in airc, the review graph in
airc-processors -- hosts its toolset the same way without an airc dependency.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
from collections.abc import Iterable
from fnmatch import fnmatch

from langchain_core.tools import BaseTool
from typing_extensions import Self

from airc_core.toolguard import (
    DEFAULT_TOOL_TIMEOUT_S,
    MAX_TOOL_RESULT_CHARS,
    LocalToolError,
    guard_tool,
)

log = logging.getLogger(__name__)

# Keys stripped from tool schemas: redundant ("title" restates the field name)
# or unsupported by Gemini ("additionalProperties").
#
# Not "description". Only the schema's top-level description duplicates the tool
# definition; a description on a property is the documentation for that argument
# and exists nowhere else. Stripping it recursively deleted every per-parameter
# doc a server sent, leaving the model to match prose in the tool description
# against a signature by name -- and silently discarding the docs of any server
# that documents its arguments properly. The top-level one is dropped separately
# below, where the distinction is visible.
_STRIP_KEYS = {"additionalProperties", "$schema", "title"}

# Per-server startup budget: spawn, initialize, and list tools. A stdio server is
# an arbitrary external binary, so it can hang instead of failing -- waiting on a
# credential prompt, or wedged mid-handshake -- and without a bound here every
# caller inherits that hang, including ones that never wanted this server's
# tools. Generous, because a cold uv-tool launcher genuinely takes seconds.
# Expiry is treated like any other startup failure: log, drop the server, carry
# on with the rest.
_SERVER_START_TIMEOUT_S = 60

# Per-call ceiling on one MCP tool execution, read when a tool is loaded. See
# toolguard for the rationale.
_TOOL_CALL_TIMEOUT_S = DEFAULT_TOOL_TIMEOUT_S

# Re-exported: consumers that size their requests against the cap read it here.
__all__ = ["MAX_TOOL_RESULT_CHARS", "MCPToolset"]


def _transport_errors() -> tuple[type[BaseException], ...]:
    """Exception types signalling the server connection is gone, not that the
    tool rejected the call. anyio raises these when the stdio child's streams
    close; McpError covers the CONNECTION_CLOSED the session synthesises for
    requests already in flight when that happens."""
    types: list[type[BaseException]] = []
    try:
        import anyio

        types += [anyio.ClosedResourceError, anyio.BrokenResourceError]
    except ImportError:  # pragma: no cover -- anyio ships with the mcp sdk
        pass
    try:
        from mcp.shared.exceptions import McpError

        types.append(McpError)
    except ImportError:  # pragma: no cover -- no mcp: no transport to lose
        pass
    return tuple(types)


# Resolved once at import: the tuple is fixed for the process, and an except
# clause is hot enough not to want an import behind it.
_TRANSPORT_ERRORS = _transport_errors()


def _clean_schema(schema: dict, top: bool = True) -> None:
    if not isinstance(schema, dict):
        return
    for key in _STRIP_KEYS & schema.keys():
        del schema[key]
    # The root description restates the tool's own description, which is sent
    # alongside; a nested one documents its property and is kept.
    if top:
        schema.pop("description", None)
    if schema.get("type") == "array" and "items" not in schema:
        schema["items"] = {}
    for v in schema.values():
        if isinstance(v, dict):
            _clean_schema(v, top=False)
        elif isinstance(v, list):
            for item in v:
                _clean_schema(item, top=False)


def _convert_transport_errors(tool: BaseTool) -> None:
    """Raise a dead MCP connection as a LocalToolError the guard renders.

    The server process died (stdio child crashed, was OOM-killed, or lost its
    credentials and exited). Sessions are opened once at startup and never
    reopened, so every later call to this server fails the same way until the
    daemon restarts -- but that must degrade to a visible per-call error, not
    take down an unrelated agent turn with a traceback.
    """
    if (orig := getattr(tool, "coroutine", None)) is None:
        return
    name = tool.name

    async def converting(*args, **kwargs):
        try:
            return await orig(*args, **kwargs)
        except _TRANSPORT_ERRORS as e:
            log.warning("tool %s: transport failed: %s", name, e)
            raise LocalToolError(
                f"tool {name} failed: its MCP server is unreachable"
                f" ({type(e).__name__}); it stays unavailable until restart"
            ) from None

    tool.coroutine = converting


def _fix_tool(tool: BaseTool) -> BaseTool:
    # Transport conversion sits inside the guard so its timeout bounds the
    # whole call, including a transport that never answers.
    _convert_transport_errors(tool)
    guard_tool(tool, timeout_s=_TOOL_CALL_TIMEOUT_S)
    schema = getattr(tool, "args_schema", None)
    if isinstance(schema, dict):
        _clean_schema(schema)
    elif schema is not None:
        orig_fn = schema.model_json_schema.__func__

        def patched(cls, **kwargs):
            s = copy.deepcopy(orig_fn(cls, **kwargs))
            _clean_schema(s)
            return s

        schema.model_json_schema = classmethod(patched)
    return tool


class MCPToolset:
    """Holds live MCP sessions and the loaded tools.

    Usage:
        async with MCPToolset(cfg.mcp_servers, cfg.tool_groups) as toolset:
            tools = toolset.tools_for(toolset.resolve_patterns(groups, extra))
    """

    def __init__(
        self,
        mcp_servers: dict[str, dict],
        tool_groups: dict[str, list[str]],
    ) -> None:
        self._mcp_servers = mcp_servers
        self._tool_groups = tool_groups
        self._stack = contextlib.AsyncExitStack()
        self.tools: list[BaseTool] = []
        self.instructions: str = ""

    async def __aenter__(self) -> Self:
        if not self._mcp_servers:
            log.warning("no MCP servers configured; agents run without tools")
            return self

        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        client = MultiServerMCPClient(self._mcp_servers)
        instructions: list[str] = []
        seen: dict[str, str] = {}
        failed: list[str] = []
        for srv in self._mcp_servers:
            # One unreachable server must not cost us every other server's tools.
            # A stdio server is an arbitrary external binary -- not installed,
            # behind an expired corp credential, crashing on startup -- and
            # aborting the whole toolset there takes down the room (or a job that
            # never wanted that server's tools) over a dependency it does not
            # use. Drop it and carry on; a consumer that needs a specific tool
            # already handles its absence (it resolves tools by pattern and
            # self-disables when the pattern matches nothing).
            try:
                async with asyncio.timeout(_SERVER_START_TIMEOUT_S):
                    sess = await self._stack.enter_async_context(
                        client.session(srv, auto_initialize=False)
                    )
                    init_result = await sess.initialize()
                    srv_tools = await load_mcp_tools(sess, server_name=srv)
            except TimeoutError:
                log.error(
                    "mcp: %s: no response within %ds; continuing without it",
                    srv,
                    _SERVER_START_TIMEOUT_S,
                )
                failed.append(srv)
                continue
            except Exception:
                log.exception("mcp: %s: failed to start; continuing without it", srv)
                failed.append(srv)
                continue
            if init_result.instructions:
                instructions.append(f"### {srv}\n{init_result.instructions}")
            for tool in srv_tools:
                # Prefix tool name with server name to avoid collisions across servers.
                # e.g., v8-utils__run_d8, gdb-mcp__backtrace
                tool.name = f"{srv}__{tool.name}"
                if tool.name in seen:
                    log.warning(
                        "tool %s from %s shadows the one from %s",
                        tool.name,
                        srv,
                        seen[tool.name],
                    )
                    self.tools = [t for t in self.tools if t.name != tool.name]
                seen[tool.name] = srv
                self.tools.append(_fix_tool(tool))
        self.instructions = "\n\n".join(instructions)
        loaded = [s for s in self._mcp_servers if s not in failed]
        log.info("loaded %d MCP tools from %s", len(self.tools), ", ".join(loaded))
        if failed:
            # Loud and separate from the success line: every tool from these
            # servers is silently missing for the rest of the process, and that
            # otherwise reads downstream as a tool_groups misconfiguration.
            log.error("mcp: unavailable server(s): %s", ", ".join(failed))
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    def has_group(self, name: str) -> bool:
        """Whether the MCP configuration defines `name`, including empty groups."""
        return name in self._tool_groups

    def resolve_patterns(
        self,
        groups: Iterable[str],
        extra: Iterable[str] = (),
        label: str = "",
    ) -> list[str]:
        """Expand named tool_groups (plus any explicit patterns) to a flat
        pattern list. Unknown groups are skipped with a warning so a typo
        degrades to fewer tools, not a crash. `label` names the requester (e.g.
        a persona handle) in those warnings."""
        patterns: list[str] = list(extra)
        for group in groups:
            group_patterns = self._tool_groups.get(group)
            if group_patterns is None:
                log.warning(
                    "%s references unknown tool group %r", label or "toolset", group
                )
                continue
            # A known-but-empty group grants nothing, yet the unknown-group check
            # above passes over it silently, which is how a config that ships no
            # [tool_groups] block boots every persona and the reviewer tool-less.
            # Warn so a mis-deployed config surfaces at startup instead of as a
            # persona insisting it has no tools.
            if not group_patterns:
                log.warning(
                    "%s references tool group %r, but it is empty"
                    " (no [tool_groups] block, or it grants nothing?)",
                    label or "toolset",
                    group,
                )
            patterns.extend(group_patterns)
        return patterns

    def tools_for(self, patterns: Iterable[str]) -> list[BaseTool]:
        """Loaded tools whose name matches any of the given fnmatch patterns."""
        pats = list(patterns)
        return [t for t in self.tools if any(fnmatch(t.name, p) for p in pats)]

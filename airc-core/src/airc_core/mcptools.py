# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""MCP tool loading and pattern-based filtering.

Sessions are opened once at daemon startup and stay alive for the daemon's
lifetime (the room is a long-running process; stateful servers like v8-utils
benefit from a persistent subprocess). Nothing reopens one: a server that dies
mid-life, or never started, stays gone until the daemon restarts. What that
costs is bounded deliberately -- every failure mode (server-reported error,
timeout, dead transport) is converted to tool-result text via
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

from langchain_core.tools import BaseTool, ToolException

log = logging.getLogger(__name__)

# Keys stripped from tool schemas: redundant ("title", "description" duplicate
# the tool definition) or unsupported by Gemini ("additionalProperties").
_STRIP_KEYS = {"additionalProperties", "$schema", "title", "description"}

# Safety valve on a single tool result, not an efficiency mechanism (the
# context-budget pruning in airc_core.agent handles steady-state size). Its only
# job is to stop one pathological output (a full log dump, a giant diff) from
# single-handedly blowing the context window; the truncation is explicit in the
# returned text so the agent can re-query for the rest. Set high: ~200k chars is
# roughly 50k tokens.
# TODO(jgruber): make this a configurable knob if a workload needs a tighter cap.
_MAX_TOOL_RESULT_CHARS = 200_000

# Per-call ceiling on one tool execution. Legit calls are seconds (a
# repo_git_grep that takes 2 minutes is wedged, not thorough); this reaps a hung
# server/transport so a single dead call cannot stall an agent turn forever. On
# expiry the agent sees a clean error tool result and continues -- see the
# ToolException conversion in capped().
_TOOL_CALL_TIMEOUT_S = 120


class _LocalToolError(ToolException):
    """An error raised by this wrapper (timeout, dead transport), as opposed to
    one the MCP adapter raised for a server-reported isError result. The
    distinction is what lets _install_error_handler answer for ours and delegate
    the adapter's back to it."""


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


def _truncate_text(text: str, limit: int = _MAX_TOOL_RESULT_CHARS) -> str:
    cut = len(text) - limit
    return text[:limit] + (
        f"\n[... output truncated ({cut} more chars);"
        " narrow the query (limit/offset/filter) for the rest]"
    )


def _truncated(value):
    """Cap a tool result's text payload at _MAX_TOOL_RESULT_CHARS.

    Two shapes occur. A plain-function tool returns a string. The MCP adapter
    returns content as a list of content blocks, with text in
    {"type": "text", "text": ...} dicts; a bare isinstance(str) check passes
    that whole list through uncapped, so one multi-megabyte result reaches the
    model and, being un-sheddable past the keep-one floor, wedges every later
    call in the thread. The cap applies to the combined text across the list,
    truncating block by block so no single block slips past; non-text blocks
    (image/file) are left intact.
    """
    if isinstance(value, str):
        return _truncate_text(value) if len(value) > _MAX_TOOL_RESULT_CHARS else value
    if isinstance(value, list):
        budget = _MAX_TOOL_RESULT_CHARS
        for block in value:
            if not (isinstance(block, dict) and isinstance(block.get("text"), str)):
                continue
            text = block["text"]
            if len(text) <= budget:
                budget -= len(text)
                continue
            block["text"] = (
                _truncate_text(text, budget)
                if budget > 0
                else "[tool result truncated to fit the context window]"
            )
            budget = 0
        return value
    return value


def _clean_schema(schema: dict) -> None:
    if not isinstance(schema, dict):
        return
    for key in _STRIP_KEYS & schema.keys():
        del schema[key]
    if schema.get("type") == "array" and "items" not in schema:
        schema["items"] = {}
    for v in schema.values():
        if isinstance(v, dict):
            _clean_schema(v)
        elif isinstance(v, list):
            for item in v:
                _clean_schema(item)


def _result_chars(value) -> int:
    """Total characters of a tool result's text payload.

    Handles both shapes: a plain string, and the content_and_artifact tuple
    whose content is a list of content blocks (the MCP path -- text lives in
    {"type": "text", "text": ...} dicts). Returns 0 for shapes with no
    measurable text, so the size/log signal is simply absent rather than wrong.
    """
    if isinstance(value, tuple) and len(value) == 2:
        value = value[0]
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(
            len(b["text"])
            for b in value
            if isinstance(b, dict) and isinstance(b.get("text"), str)
        )
    return 0


def _install_error_handler(tool: BaseTool) -> None:
    """Route _LocalToolError to text, leaving any existing handler intact.

    The MCP adapter installs its own handle_tool_error callback whose job is to
    preserve non-text content blocks (image/file) in an isError result; it
    documents that as load-bearing and locks it with a test. Overwriting it with
    a bare True -- which this used to do -- collapses those blocks to str().

    But it cannot simply be kept either: that callback RE-RAISES every
    ToolException that is not its own internal type, so the timeout and
    transport errors raised below would escape again, which is the whole thing
    capped() exists to prevent. Hence a composed handler: ours answers for the
    errors we raise, anything else falls through to the adapter's.
    """
    prev = getattr(tool, "handle_tool_error", None)

    def handle(e: ToolException):
        if isinstance(e, _LocalToolError):
            return str(e)
        if callable(prev):
            return prev(e)
        if prev:  # True/str: langchain's own default rendering
            return str(e)
        raise e

    tool.handle_tool_error = handle


def _fix_tool(tool: BaseTool) -> BaseTool:
    _install_error_handler(tool)
    if (orig := getattr(tool, "coroutine", None)) is not None:
        name = tool.name

        async def capped(*args, **kwargs):
            # A bare TimeoutError or transport failure would bypass
            # handle_tool_error (only ToolException routes there), escape the
            # tool boundary, and kill the whole agent turn -- in the review graph
            # a timeout would even be misread as the review-level wall-clock.
            # Convert both to the ToolException channel so the agent sees a clean
            # error message and continues the turn.
            try:
                async with asyncio.timeout(_TOOL_CALL_TIMEOUT_S):
                    out = await orig(*args, **kwargs)
            except TimeoutError:
                log.warning("tool %s: timed out after %ds", name, _TOOL_CALL_TIMEOUT_S)
                raise _LocalToolError(
                    f"tool {name} timed out after {_TOOL_CALL_TIMEOUT_S}s"
                ) from None
            except _TRANSPORT_ERRORS as e:
                # The server process died (stdio child crashed, was OOM-killed,
                # or lost its credentials and exited). Sessions are opened once
                # at startup and never reopened, so every later call to this
                # server fails the same way until the daemon restarts -- but that
                # must degrade to a visible per-call error, not take down an
                # unrelated agent turn with a traceback.
                log.warning("tool %s: transport failed: %s", name, e)
                raise _LocalToolError(
                    f"tool {name} failed: its MCP server is unreachable"
                    f" ({type(e).__name__}); it stays unavailable until restart"
                ) from None
            # Log the raw size and which tool produced it: this is the per-tool
            # evidence for what inflates a turn's context, since every result
            # below the cap is re-sent on subsequent model calls until pruned.
            raw = _result_chars(out)
            if raw:
                log.info(
                    "tool %s result: %d chars%s",
                    name,
                    raw,
                    " (capped)" if raw > _MAX_TOOL_RESULT_CHARS else "",
                )
            # content_and_artifact tools return (content, artifact).
            if isinstance(out, tuple) and len(out) == 2:
                return (_truncated(out[0]), out[1])
            return _truncated(out)

        tool.coroutine = capped
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

    async def __aenter__(self) -> MCPToolset:
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
                sess = await self._stack.enter_async_context(
                    client.session(srv, auto_initialize=False)
                )
                init_result = await sess.initialize()
                srv_tools = await load_mcp_tools(sess, server_name=srv)
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
            # above passes over it silently -- the exact way a config that ships
            # no [tool_groups] block boots every persona and the reviewer
            # tool-less without a peep. Warn so a mis-deployed config surfaces at
            # startup instead of as a persona insisting it has no tools.
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

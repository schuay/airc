# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""MCP tool loading and pattern-based filtering.

HTTP sessions persist across calls. After transport/session loss, the failed call
returns an unknown-outcome error without replay. A subsequent call reconnects;
server session-local state may be lost. Reconnect requires unchanged tool schemas,
metadata, descriptions and server instructions. Changed contracts require restart.

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
from typing import Self

from langchain_core.tools import BaseTool

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
_SERVER_CLOSE_TIMEOUT_S = 5

# Per-call ceiling on one MCP tool execution, read when a tool is loaded. See
# toolguard for the rationale.
_TOOL_CALL_TIMEOUT_S = DEFAULT_TOOL_TIMEOUT_S

# Re-exported: consumers that size their requests against the cap read it here.
__all__ = ["MAX_TOOL_RESULT_CHARS", "MCPToolset"]


def _transport_errors() -> tuple[type[BaseException], ...]:
    """Exceptions rendered at the tool boundary; only _session_lost reconnects."""
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
    import httpx

    types.append(httpx.TransportError)
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
    """Render MCP failures through the guard's tool-error handler."""
    if (orig := getattr(tool, "coroutine", None)) is None:
        return
    name = tool.name

    async def converting(*args, **kwargs):
        try:
            return await orig(*args, **kwargs)
        except _TRANSPORT_ERRORS as e:
            if not _session_lost(e):
                raise LocalToolError(f"MCP request failed: {e}") from None
            log.warning("tool %s: transport failed: %s", name, e)
            raise LocalToolError(
                f"tool {name} failed: its MCP server is unreachable"
                f" ({type(e).__name__}); it stays unavailable until restart"
            ) from None

    tool.coroutine = converting


def _fix_tool(tool: BaseTool, *, persistent_http: bool = False) -> BaseTool:
    # Transport conversion sits inside the guard so its timeout bounds the
    # whole call, including a transport that never answers.
    _convert_transport_errors(tool)
    guard_tool(tool, timeout_s=None if persistent_http else _TOOL_CALL_TIMEOUT_S)
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


_HTTP_TRANSPORTS = {"http", "streamable_http", "streamable-http"}


def _keep_idle_get_open(connection):
    from mcp.shared._httpx_utils import create_mcp_http_client

    if "sse_read_timeout" in connection:
        return
    factory = connection.get("httpx_client_factory") or create_mcp_http_client

    def create_client(**kwargs):
        client = factory(**kwargs)

        async def configure_get(request):
            # SDK 1.28 stops GET retries after two idle read timeouts. An idle
            # notification stream is healthy; retain POST and connection limits.
            if request.method == "GET":
                request.extensions["timeout"] = {
                    **request.extensions.get("timeout", {}),
                    "read": None,
                }

        client.event_hooks["request"].append(configure_get)
        return client

    connection["httpx_client_factory"] = create_client


async def _finish_closing(awaitables):
    closing = asyncio.gather(*awaitables, return_exceptions=True)
    cancelled = None
    while not closing.done():
        try:
            await asyncio.shield(closing)
        except asyncio.CancelledError as e:
            cancelled = e
    if cancelled is not None:
        raise cancelled


class _HTTPGeneration:
    def __init__(self, tools, owner):
        self.tools = tools
        self.owner = owner

    async def call(self, name, args, kwargs):
        import anyio

        call = asyncio.create_task(self.tools[name].coroutine(*args, **kwargs))
        try:
            done, _ = await asyncio.wait(
                (call, self.owner), return_when=asyncio.FIRST_COMPLETED
            )
            if call in done:
                return await call
            raise anyio.ClosedResourceError
        finally:
            call.cancel()
            await asyncio.gather(call, return_exceptions=True)


class _HTTPSession:
    """Own each generation's AnyIO scopes in one task, including shutdown."""

    def __init__(self, client, name):
        self.client = client
        self.name = name
        self.lock = asyncio.Lock()
        self.task = None
        self.current = None
        self.contract = None
        self.instructions = ""
        self.closed = False
        self.retry_at = 0.0
        self.backoff = 1.0

    async def _own(self, ready):
        from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool

        try:
            async with asyncio.timeout(None) as cleanup_limit:
                async with self.client.session(
                    self.name, auto_initialize=False
                ) as session:
                    try:
                        init = await session.initialize()
                        raw_tools = []
                        cursor = None
                        for _ in range(1000):
                            page = await session.list_tools(cursor=cursor)
                            raw_tools.extend(page.tools)
                            if not page.nextCursor:
                                break
                            cursor = page.nextCursor
                        else:
                            raise LocalToolError("MCP tools/list exceeded 1000 pages")
                        contract = {
                            t.name: t.model_dump(
                                mode="json", by_alias=True, exclude_none=True
                            )
                            for t in raw_tools
                        }
                        tools = [
                            convert_mcp_tool_to_langchain_tool(
                                session, t, server_name=self.name
                            )
                            for t in raw_tools
                        ]
                        if self.contract is not None and (
                            contract != self.contract
                            or init.instructions != self.instructions
                        ):
                            ready.set_exception(
                                LocalToolError(
                                    f"MCP server {self.name} changed tools or instructions;"
                                    " restart required to review the new contract"
                                )
                            )
                            return
                        self.contract = contract
                        self.instructions = init.instructions
                        ready.set_result({t.name: t for t in tools})
                        await asyncio.Future()
                    finally:
                        cleanup_limit.reschedule(
                            asyncio.get_running_loop().time() + _SERVER_CLOSE_TIMEOUT_S
                        )
        except BaseException as e:
            if not ready.done():
                ready.set_exception(e)
            elif not isinstance(e, asyncio.CancelledError):
                log.warning("mcp: %s: session ended: %s", self.name, e)

    async def _close_generation(self):
        if self.task is not None:
            # Cancellation is delivered to the owner, never to an exit stack
            # entered by another task.
            self.task.cancel()
            try:
                await _finish_closing([self.task])
            finally:
                self.task = None
                self.current = None
        self.current = None

    async def get(self):
        async with self.lock:
            if self.closed:
                raise LocalToolError(f"MCP server {self.name} is closed")
            if self.current is not None and not self.task.done():
                return self.current
            if asyncio.get_running_loop().time() < self.retry_at:
                raise LocalToolError(
                    f"MCP server {self.name} is unavailable; reconnect backoff active"
                )
            await self._close_generation()
            ready = asyncio.get_running_loop().create_future()
            self.task = asyncio.create_task(self._own(ready), name=f"mcp:{self.name}")
            try:
                async with asyncio.timeout(_SERVER_START_TIMEOUT_S):
                    tools = await asyncio.shield(ready)
                    self.current = _HTTPGeneration(tools, self.task)
            except BaseException:
                await self._close_generation()
                if ready.done() and not ready.cancelled():
                    ready.exception()
                else:
                    ready.cancel()
                self.retry_at = asyncio.get_running_loop().time() + self.backoff
                self.backoff = min(self.backoff * 2, 30)
                raise
            self.backoff = 1.0
            log.info("mcp: %s: session ready", self.name)
            return self.current

    def invalidate(self, generation):
        if self.current is generation:
            self.current = None
            log.warning(
                "mcp: %s: session lost; next call reconnects; session-local state"
                " may be lost",
                self.name,
            )

    def wrap(self, tool):
        from functools import wraps

        name = tool.name
        original = tool.coroutine

        @wraps(original)
        async def dispatch(*args, **kwargs):
            generation = None
            try:
                async with asyncio.timeout(_TOOL_CALL_TIMEOUT_S):
                    try:
                        generation = await self.get()
                    except Exception as e:
                        log.warning("mcp: %s: reconnect failed: %s", self.name, e)
                        raise LocalToolError(
                            f"MCP server {self.name} unavailable; call was not sent ({e})"
                        ) from None
                    return await generation.call(name, args, kwargs)
            except TimeoutError:
                if generation is None:
                    raise LocalToolError(
                        f"MCP server {self.name} reconnect timed out; call was not sent"
                    ) from None
                raise LocalToolError(
                    f"tool {self.name}__{name} timed out; outcome unknown;"
                    " call was not replayed. Inspect state before retrying"
                ) from None
            except _TRANSPORT_ERRORS as e:
                if _request_timed_out(e):
                    raise LocalToolError(
                        f"tool {self.name}__{name} timed out; outcome unknown;"
                        " call was not replayed. Inspect state before retrying"
                    ) from None
                if _session_lost(e):
                    self.invalidate(generation)
                    raise LocalToolError(
                        f"MCP server {self.name} session lost; outcome unknown;"
                        " call was not replayed. A subsequent call reconnects;"
                        " session-local state may be lost"
                    ) from None
                raise LocalToolError(f"MCP request failed: {e}") from None

        tool = tool.model_copy(deep=True, update={"coroutine": dispatch})
        tool.name = f"{self.name}__{name}"
        return tool

    async def close(self):
        async with self.lock:
            self.closed = True
            await self._close_generation()


def _request_timed_out(error):
    from mcp.shared.exceptions import McpError

    return (
        isinstance(error, McpError)
        and error.error.code == 408
        and error.error.message.startswith("Timed out while waiting for response to ")
    )


def _session_lost(error):
    import anyio
    import httpx
    from mcp.shared.exceptions import McpError
    from mcp.types import CONNECTION_CLOSED

    if isinstance(error, McpError):
        return (
            error.error.code == CONNECTION_CLOSED
            and error.error.message == "Connection closed"
        ) or (error.error.code == 32600 and error.error.message == "Session terminated")
    return isinstance(
        error,
        (anyio.ClosedResourceError, anyio.BrokenResourceError, httpx.TransportError),
    )


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
        self._http_sessions: list[_HTTPSession] = []
        self.tools: list[BaseTool] = []
        self.instructions: str = ""

    async def __aenter__(self) -> Self:
        try:
            return await self._start()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def _start(self) -> Self:
        if not self._mcp_servers:
            log.warning("no MCP servers configured; agents run without tools")
            return self

        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        connections = {name: dict(c) for name, c in self._mcp_servers.items()}
        for connection in connections.values():
            if connection.get("transport") in _HTTP_TRANSPORTS:
                _keep_idle_get_open(connection)
        client = MultiServerMCPClient(connections)
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
                connection = self._mcp_servers[srv]
                if connection.get("transport") in _HTTP_TRANSPORTS:
                    owner = _HTTPSession(client, srv)
                    self._http_sessions.append(owner)
                    generation = await owner.get()
                    srv_tools = [owner.wrap(t) for t in generation.tools.values()]
                    server_instructions = owner.instructions
                else:
                    async with asyncio.timeout(_SERVER_START_TIMEOUT_S):
                        sess = await self._stack.enter_async_context(
                            client.session(srv, auto_initialize=False)
                        )
                        init_result = await sess.initialize()
                        srv_tools = await load_mcp_tools(sess, server_name=srv)
                    server_instructions = init_result.instructions
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
            if server_instructions:
                instructions.append(f"### {srv}\n{server_instructions}")
            for tool in srv_tools:
                # Prefix tool name with server name to avoid collisions across servers.
                # e.g., v8-utils__run_d8, gdb-mcp__backtrace
                if connection.get("transport") not in _HTTP_TRANSPORTS:
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
                if connection.get("transport") in _HTTP_TRANSPORTS:
                    self.tools.append(_fix_tool(tool, persistent_http=True))
                else:
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
        try:
            await _finish_closing([owner.close() for owner in self._http_sessions])
        finally:
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

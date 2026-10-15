# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Per-tool protections applied to every tool an agent is handed.

A tool result is pasted into the agent's conversation and re-sent on every
later model call in the thread, so one unbounded result inflates every call
after it and can exceed the model's window outright. ``guard_tool`` installs,
on a built tool:

- a cap on the text of one result (``MAX_TOOL_RESULT_CHARS``), with a note
  that tells the model to narrow its query;
- a per-call timeout, so a hung call ends the tool call and not the turn;
- conversion of the timeout to a ``ToolException`` that the tool renders as
  an error result, composed with any handler the tool already has;
- an INFO log line per result with its size, which is the per-tool record of
  what inflates a turn.

Both tool paths use it. ``mcptools`` applies it when it loads tools from an
MCP server, and ``toolgrants.ResolvedToolGrant.wrap`` applies it to native and
worktree tools, so a tool reaches an agent guarded whichever way it was built.
"""

from __future__ import annotations

import asyncio
import logging
from functools import wraps

from langchain_core.tools import BaseTool, ToolException

log = logging.getLogger(__name__)

# Ceiling on a single tool result, applied at ingest. Stops one pathological
# output (a full log dump, a giant diff) from blowing the context window, and
# bounds what that output costs for the rest of the run: a result is re-sent on
# every later call in the thread, so its real price is size x turns remaining.
# The truncation is explicit in the returned text so the agent can re-query.
#
# This is the only size lever that does not fight prefix caching, because it
# acts before the result enters the cacheable prefix. compact_for_budget in
# airc_core.agent refuses to shed until the request would overflow
# (90% of a 1M window), since stripping a message rebuilds the cache -- so
# nothing downstream bounds the cost of a large result once it is in.
#
# 50k chars is ~12k tokens. Measured over 1687 review and verify tool results:
# p50 2.8k, p90 15k, p95 33k, p99 133k. A normal read, diff or grep passes
# intact; clipping hits ~4% of results and removes ~20% of all re-read volume.
# Lower starts to clip routinely, and a re-query costs a turn, which is the
# more expensive unit.
# TODO(jgruber): make this a configurable knob if a workload needs a tighter cap.
#
# Public because a consumer that is not a model has to size its REQUESTS against
# it: the cap cuts by characters, which turns a structured reply into a parse
# error rather than a shorter one, so airc-watchers' perf detector asks pd for a
# page it knows will fit. Anything reading this name is coupled to the cap on
# purpose -- do not privatise it again without giving those callers a way to ask.
MAX_TOOL_RESULT_CHARS = 50_000

# Per-call ceiling on one tool execution. Legit calls are seconds (a
# repo_git_grep that takes 2 minutes is wedged, not thorough); this reaps a hung
# call so a single dead one cannot stall an agent turn forever. On expiry the
# agent sees a clean error tool result and continues. A tool that bounds its own
# wait (run_d8 waits on a build queue; the worktree shell takes a timeout from
# the model) opts out with ``timeout_s=None``.
DEFAULT_TOOL_TIMEOUT_S = 120

# Key in ``tool.metadata`` that marks a guarded tool. The marker is on metadata
# and not on the coroutine because ``toolpin`` copies a tool with
# ``model_copy(update={"coroutine": ...})``: the shallow copy shares the
# metadata dict and replaces the coroutine, so a metadata marker survives the
# pin and a coroutine marker would not. Without the skip a tool that passes
# through two guard sites (an MCP tool loaded by mcptools and then handed to
# ``wrap``) would log every result twice and nest two timeouts.
GUARD_MARKER = "airc_guarded"


class LocalToolError(ToolException):
    """An error raised by the guard (timeout) or by the MCP transport layer,
    as opposed to one the tool itself raised for a failed call. The handler
    installed by ``guard_tool`` answers for these and delegates the rest to
    whatever handler the tool had."""


def _truncate_text(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    cut = len(text) - limit
    return text[:limit] + (
        f"\n[... output truncated ({cut} more chars);"
        " narrow the query (limit/offset/filter) for the rest]"
    )


def truncated(value):
    """Cap a tool result's text payload at MAX_TOOL_RESULT_CHARS.

    Two forms occur. A plain-function tool returns a string. The MCP adapter
    returns content as a list of content blocks, with text in
    {"type": "text", "text": ...} dicts; a bare isinstance(str) check passes
    that whole list through uncapped, so one multi-megabyte result reaches the
    model and, being un-sheddable past the keep-one floor, wedges every later
    call in the thread. The cap applies to the combined text across the list,
    truncating block by block so no single block slips past; non-text blocks
    (image/file) are left intact.
    """
    if isinstance(value, str):
        return _truncate_text(value) if len(value) > MAX_TOOL_RESULT_CHARS else value
    if isinstance(value, list):
        budget = MAX_TOOL_RESULT_CHARS
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


def result_chars(value) -> int:
    """Total characters of a tool result's text payload.

    Handles both forms: a plain string, and the content_and_artifact tuple
    whose content is a list of content blocks (the MCP path -- text lives in
    {"type": "text", "text": ...} dicts). Returns 0 for forms with no
    measurable text, so the size/log signal is absent, not wrong.
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


def _finish(name: str, out):
    """Log a result's raw size and return it capped."""
    # Log the raw size and which tool produced it: this is the per-tool
    # evidence for what inflates a turn's context, since every result below
    # the cap is re-sent on subsequent model calls until pruned.
    raw = result_chars(out)
    if raw:
        log.info(
            "tool %s result: %d chars%s",
            name,
            raw,
            " (capped)" if raw > MAX_TOOL_RESULT_CHARS else "",
        )
    # content_and_artifact tools return (content, artifact).
    if isinstance(out, tuple) and len(out) == 2:
        return (truncated(out[0]), out[1])
    return truncated(out)


def install_error_handler(tool: BaseTool) -> None:
    """Route LocalToolError to text, leaving any existing handler intact.

    The MCP adapter installs its own handle_tool_error callback whose job is to
    preserve non-text content blocks (image/file) in an isError result; it
    documents that as required and locks it with a test. Overwriting it with
    a bare True -- which this used to do -- collapses those blocks to str().

    It cannot be kept unchanged either: that callback re-raises every
    ToolException that is not its own internal type, so the timeout and
    transport errors would escape again, which the guard exists to prevent.
    So the handler is composed: ours answers for the errors we raise,
    anything else falls through to the adapter's.
    """
    prev = getattr(tool, "handle_tool_error", None)

    def handle(e: ToolException):
        if isinstance(e, LocalToolError):
            return str(e)
        if callable(prev):
            return prev(e)
        if prev:  # True/str: langchain's own default rendering
            return str(e)
        raise e

    tool.handle_tool_error = handle


def is_guarded(tool) -> bool:
    return bool((getattr(tool, "metadata", None) or {}).get(GUARD_MARKER))


def guard_tool(tool, *, timeout_s: float | None = DEFAULT_TOOL_TIMEOUT_S):
    """Install the result cap, size log, timeout and error handler on `tool`.

    Mutates and returns `tool`. Objects that are not a BaseTool (test doubles)
    and tools already carrying the marker pass through unchanged. Both entry
    points are wrapped: the coroutine, which the agent loop awaits, and `func`,
    so a sync invocation does not bypass the cap and a func-only tool (the
    worktree file tools) is still capped. The timeout applies to the coroutine
    only; a func-only tool has no await to bound.
    """
    if not isinstance(tool, BaseTool) or is_guarded(tool):
        return tool
    install_error_handler(tool)
    name = tool.name
    if (orig := getattr(tool, "coroutine", None)) is not None:

        @wraps(orig)
        async def guarded(*args, **kwargs):
            # A bare TimeoutError would bypass handle_tool_error (only
            # ToolException routes there), escape the tool boundary, and kill
            # the whole agent turn -- in the review graph a timeout would even
            # be misread as the review-level wall-clock. Convert it to the
            # ToolException channel so the agent sees a clean error message
            # and continues the turn. A TimeoutError the tool raised itself is
            # not ours to rename; `expired()` tells the two apart.
            limit = asyncio.timeout(timeout_s)
            try:
                async with limit:
                    out = await orig(*args, **kwargs)
            except TimeoutError:
                if not limit.expired():
                    raise
                log.warning("tool %s: timed out after %ss", name, timeout_s)
                raise LocalToolError(
                    f"tool {name} timed out after {timeout_s}s"
                ) from None
            return _finish(name, out)

        tool.coroutine = guarded
    if (func := getattr(tool, "func", None)) is not None:

        @wraps(func)
        def guarded_sync(*args, **kwargs):
            return _finish(name, func(*args, **kwargs))

        tool.func = guarded_sync
    tool.metadata = {**(tool.metadata or {}), GUARD_MARKER: True}
    return tool

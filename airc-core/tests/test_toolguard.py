# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""guard_tool: the protections every tool carries, whichever way it was built."""

import asyncio
import logging

import pytest
from airc_core.toolgrants import (
    ResolvedToolGrant,
    ToolCatalog,
    ToolGrant,
    ToolSource,
    ToolSpec,
)
from airc_core.toolguard import (
    GUARD_MARKER,
    MAX_TOOL_RESULT_CHARS,
    guard_tool,
    is_guarded,
)
from langchain_core.tools import StructuredTool

BIG = "x" * (MAX_TOOL_RESULT_CHARS * 2)


def _native_tool(name="big", payload=BIG):
    """A v8tools-style tool: sync func plus a coroutine that runs it in a
    thread. The coroutine calls the closure, not tool.func."""

    def call() -> str:
        return payload

    async def acall() -> str:
        return await asyncio.to_thread(call)

    return StructuredTool.from_function(
        call, coroutine=acall, name=name, description="d"
    )


def _call(tool, name):
    return {"type": "tool_call", "name": name, "args": {}, "id": "tc1"}


async def test_native_coroutine_result_is_capped_and_logged(caplog):
    tool = guard_tool(_native_tool())
    with caplog.at_level(logging.INFO, logger="airc_core.toolguard"):
        msg = await tool.ainvoke(_call(tool, "big"))
    out = str(msg.content)
    assert len(out) < MAX_TOOL_RESULT_CHARS + 200
    assert f"output truncated ({MAX_TOOL_RESULT_CHARS} more chars)" in out
    assert f"tool big result: {len(BIG)} chars (capped)" in caplog.text


def test_sync_invocation_is_capped():
    tool = guard_tool(_native_tool())
    out = str(tool.invoke(_call(tool, "big")).content)
    assert len(out) < MAX_TOOL_RESULT_CHARS + 200


async def test_func_only_tool_is_capped():
    # The worktree file tools have func and no coroutine; ainvoke runs func in
    # an executor, so the func wrapper is what caps them.
    tool = guard_tool(
        StructuredTool.from_function(func=lambda: BIG, name="read", description="d")
    )
    assert tool.coroutine is None
    out = str((await tool.ainvoke(_call(tool, "read"))).content)
    assert len(out) < MAX_TOOL_RESULT_CHARS + 200


async def test_content_and_artifact_is_capped():
    async def big():
        return [{"type": "text", "text": BIG}], {"artifact": 1}

    tool = guard_tool(
        StructuredTool.from_function(
            coroutine=big,
            name="blocks",
            description="d",
            response_format="content_and_artifact",
        )
    )
    msg = await tool.ainvoke(_call(tool, "blocks"))
    assert len(msg.content[0]["text"]) < MAX_TOOL_RESULT_CHARS + 200
    assert msg.artifact == {"artifact": 1}


async def test_timeout_is_an_error_tool_message():
    async def hang() -> str:
        await asyncio.sleep(30)
        return "never"

    tool = guard_tool(
        StructuredTool.from_function(coroutine=hang, name="hang", description="d"),
        timeout_s=0.01,
    )
    msg = await tool.ainvoke(_call(tool, "hang"))
    assert msg.status == "error"
    assert "timed out after 0.01s" in str(msg.content)


async def test_timeout_none_does_not_cut():
    async def slow() -> str:
        await asyncio.sleep(0.05)
        return "done"

    tool = guard_tool(
        StructuredTool.from_function(coroutine=slow, name="slow", description="d"),
        timeout_s=None,
    )
    assert (await tool.ainvoke(_call(tool, "slow"))).content == "done"


async def test_tools_own_timeout_error_is_not_renamed():
    # A TimeoutError raised inside the tool (a subprocess wait, say) is the
    # tool's own failure and keeps its type; only the guard's expiry becomes a
    # guard error.
    async def inner() -> str:
        raise TimeoutError("subprocess")

    tool = guard_tool(
        StructuredTool.from_function(coroutine=inner, name="inner", description="d"),
        timeout_s=None,
    )
    with pytest.raises(TimeoutError, match="subprocess"):
        await tool.coroutine()


async def test_guard_is_idempotent(caplog):
    tool = guard_tool(guard_tool(_native_tool(payload="short")))
    assert is_guarded(tool)
    with caplog.at_level(logging.INFO, logger="airc_core.toolguard"):
        await tool.ainvoke(_call(tool, "big"))
    assert caplog.text.count("tool big result") == 1


def test_marker_survives_a_model_copy_that_replaces_the_coroutine():
    # toolpin copies a tool with model_copy(update={"coroutine": ...}). The
    # shallow copy shares the metadata dict, so the marker survives and a
    # second guard site does not re-wrap the pinned copy.
    tool = guard_tool(_native_tool())

    async def pinned():
        return "pinned"

    copy = tool.model_copy(update={"coroutine": pinned})
    assert is_guarded(copy)
    assert guard_tool(copy) is copy
    assert copy.coroutine is pinned


def test_non_tools_pass_through():
    from types import SimpleNamespace

    double = SimpleNamespace(name="x")
    assert guard_tool(double) is double


def test_build_and_wrap_guard_every_source():
    grant = ToolGrant(
        ToolCatalog(
            (
                ToolSpec("native", lambda _: _native_tool("native")),
                ToolSpec(
                    "shell",
                    lambda _: _native_tool("shell"),
                    ToolSource.WORKTREE,
                ),
            )
        ),
        required=("native", "shell"),
    )
    resolved = grant.resolve(label="t")
    from pathlib import Path

    from airc_core.toolgrants import ToolBuildContext

    built = resolved.build(ToolSource.NATIVE) + resolved.build(
        ToolSource.WORKTREE, ToolBuildContext(workdir=Path("/tmp"))
    )
    assert [t.name for t in built] == ["native", "shell"]
    assert all(t.metadata[GUARD_MARKER] for t in built)
    prebuilt = ResolvedToolGrant((), None, "t").wrap(
        [_native_tool("mcp")], ToolSource.MCP
    )
    assert is_guarded(prebuilt[0])


async def test_grant_wrapper_is_outside_the_guard():
    # icompleteu passes pin_repo_reads as the grant wrapper. The guard goes on
    # first, so the wrapper's coroutine is the outer layer and the guard still
    # caps what the inner tool returns.
    seen = []

    def wrapper(tools):
        out = []
        for tool in tools:
            seen.append(is_guarded(tool))
            inner = tool.coroutine

            async def outer(_inner=inner):
                return await _inner()

            out.append(tool.model_copy(update={"coroutine": outer}))
        return out

    grant = ToolGrant(
        ToolCatalog((ToolSpec("big", lambda _: _native_tool()),)),
        required=("big",),
        wrapper=wrapper,
    )
    (tool,) = grant.resolve(label="t").build(ToolSource.NATIVE)
    assert seen == [True]
    out = str((await tool.ainvoke(_call(tool, "big"))).content)
    assert len(out) < MAX_TOOL_RESULT_CHARS + 200

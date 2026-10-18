# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

import asyncio
import json
import socket

import pytest
import uvicorn
from airc_core import mcptools
from airc_core.mcptools import MCPToolset
from mcp.types import JSONRPCMessage
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route


class HTTPServer:
    def __init__(self):
        self.sessions = set()
        self.initializations = 0
        self.calls = []
        self.posts = []
        self.gets = 0
        self.disconnect = asyncio.Event()
        self.side_effect = asyncio.Event()
        self.release = asyncio.Event()
        self.offline = False
        self.instructions = "stable instructions"
        self.tool_name = "state"
        self.description = "Read or change session state"
        self.values = {}
        self.delayed = asyncio.Event()
        self.finish_delayed = asyncio.Event()
        self.messages = asyncio.Queue()
        self.responses = []
        self.hang_delete = False
        self.deleting = asyncio.Event()
        self.hang_list = False
        self.output_schema = None
        self.initialize_delay = 0
        self.hang_initialize = False
        self.initializing = asyncio.Event()
        self.port = 0

    async def endpoint(self, request: Request):
        sid = request.headers.get("mcp-session-id")
        if self.offline:
            return Response(status_code=503)
        if request.method == "DELETE":
            self.deleting.set()
            if self.hang_delete:
                await self.release.wait()
            self.sessions.discard(sid)
            return Response(status_code=200)
        if request.method == "GET":
            self.gets += 1
            if sid not in self.sessions:
                return Response(status_code=404)

            async def events():
                yield ": connected\n\n"
                while True:
                    message = asyncio.create_task(self.messages.get())
                    disconnect = asyncio.create_task(self.disconnect.wait())
                    try:
                        done, _ = await asyncio.wait(
                            (message, disconnect), return_when=asyncio.FIRST_COMPLETED
                        )
                        if disconnect in done:
                            return
                        yield (
                            "event: message\ndata: "
                            + json.dumps(message.result())
                            + "\n\n"
                        )
                    finally:
                        message.cancel()
                        disconnect.cancel()
                        await asyncio.gather(
                            message, disconnect, return_exceptions=True
                        )

            return StreamingResponse(events(), media_type="text/event-stream")
        message = JSONRPCMessage.model_validate(await request.json()).root
        if not hasattr(message, "method"):
            self.responses.append(message)
            return Response(status_code=202)
        self.posts.append((sid, message.method))
        if message.method == "initialize":
            self.initializing.set()
            await asyncio.sleep(self.initialize_delay)
            if self.hang_initialize:
                await self.release.wait()
            self.initializations += 1
            sid = str(self.initializations)
            self.sessions.add(sid)
            self.values[sid] = 0
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": message.id,
                    "result": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "test", "version": "1"},
                        "instructions": self.instructions,
                    },
                },
                headers={"mcp-session-id": sid},
            )
        if sid not in self.sessions:
            return Response(status_code=404)
        if message.method.startswith("notifications/"):
            return Response(status_code=202)
        if message.method == "tools/list":
            if self.hang_list:
                await self.release.wait()
            result = {
                "tools": [
                    {
                        "name": self.tool_name,
                        "description": self.description,
                        "inputSchema": {
                            "type": "object",
                            "properties": {"action": {"type": "string"}},
                            "required": ["action"],
                        },
                    }
                ]
            }
            if self.output_schema is not None:
                result["tools"][0]["outputSchema"] = self.output_schema
        elif message.method == "tools/call":
            action = message.params["arguments"]["action"]
            self.calls.append((sid, action))
            if action == "delayed_loss":
                self.delayed.set()
                await self.finish_delayed.wait()
                return Response(status_code=404)
            if action in ("rpc_error", "server_error"):
                code = -32602 if action == "rpc_error" else -32000
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": message.id,
                        "error": {"code": code, "message": "bad argument"},
                    }
                )
            if action == "tool_error":
                result = {
                    "isError": True,
                    "content": [{"type": "text", "text": "rejected"}],
                }
            else:
                if action in ("increment", "slow"):
                    self.values[sid] += 1
                if action == "slow":
                    self.side_effect.set()
                    await self.release.wait()
                result = {"content": [{"type": "text", "text": str(self.values[sid])}]}
        else:
            return Response(status_code=400)
        return JSONResponse({"jsonrpc": "2.0", "id": message.id, "result": result})

    async def start(self):
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", self.port))
        sock.listen()
        self.port = sock.getsockname()[1]
        app = Starlette(
            routes=[Route("/mcp", self.endpoint, methods=["POST", "GET", "DELETE"])]
        )
        self.server = uvicorn.Server(
            uvicorn.Config(app, log_level="error", lifespan="off")
        )
        self.task = asyncio.create_task(self.server.serve(sockets=[sock]))
        await eventually(lambda: self.server.started)

    async def stop(self):
        self.release.set()
        self.finish_delayed.set()
        self.disconnect.set()
        self.server.should_exit = True
        await asyncio.wait_for(self.task, 5)

    def config(self):
        return {"transport": "http", "url": f"http://127.0.0.1:{self.port}/mcp"}


async def eventually(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.fixture
async def server():
    instance = HTTPServer()
    await instance.start()
    try:
        yield instance
    finally:
        await instance.stop()


async def invoke(toolset, action="read", server="test"):
    tool = toolset.tools_for([f"{server}__state"])[0]
    return await tool.ainvoke(
        {
            "type": "tool_call",
            "id": "call",
            "name": tool.name,
            "args": {"action": action},
        }
    )


def value(message):
    assert message.status == "success", message.content
    return message.content[0]["text"]


async def test_idle_get_disconnect_keeps_session_and_state(server):
    async with MCPToolset({"test": server.config()}, {}) as ts:
        assert value(await invoke(ts, "increment")) == "1"
        await eventually(lambda: server.gets == 1)
        server.disconnect.set()
        await eventually(lambda: server.gets >= 2)
        server.disconnect.clear()
        assert value(await invoke(ts)) == "1"
        assert server.initializations == 1


async def test_expired_session_recovers_next_call_without_replay(server):
    async with MCPToolset({"test": server.config()}, {}) as ts:
        assert value(await invoke(ts, "increment")) == "1"
        server.sessions.clear()
        failed = await invoke(ts, "increment")
        assert failed.status == "error"
        assert "not replayed" in str(failed.content)
        assert "session-local state" in str(failed.content)
        assert value(await invoke(ts, "increment")) == "1"
        assert server.initializations == 2
        assert server.calls == [("1", "increment"), ("2", "increment")]


async def test_restart_recovers_and_peer_is_unchanged(server):
    peer = HTTPServer()
    await peer.start()
    try:
        async with MCPToolset(
            {"test": server.config(), "peer": peer.config()}, {}
        ) as ts:
            assert value(await invoke(ts, "increment", "peer")) == "1"
            await server.stop()
            server.sessions.clear()
            server.disconnect = asyncio.Event()
            await server.start()
            assert (await invoke(ts)).status == "error"
            assert value(await invoke(ts)) == "0"
            assert value(await invoke(ts, server="peer")) == "1"
            assert peer.initializations == 1
    finally:
        await peer.stop()


async def test_concurrent_loss_and_single_reconnect(server):
    async with MCPToolset({"test": server.config()}, {}) as ts:
        server.sessions.clear()
        first = await asyncio.gather(*(invoke(ts) for _ in range(8)))
        assert any(m.status == "error" for m in first)
        recovered = await asyncio.gather(*(invoke(ts) for _ in range(8)))
        assert all(value(m) == "0" for m in recovered)
        assert server.initializations == 2


async def test_reconnect_failure_backoff_then_recovery(server, monkeypatch):
    async with MCPToolset({"test": server.config()}, {}) as ts:
        server.sessions.clear()
        assert (await invoke(ts)).status == "error"
        server.offline = True
        monkeypatch.setattr(mcptools, "_SERVER_START_TIMEOUT_S", 0.2)
        assert (await invoke(ts)).status == "error"
        server.offline = False
        assert (await invoke(ts)).status == "error"
        await asyncio.sleep(1.05)
        assert value(await invoke(ts)) == "0"
        assert server.initializations == 2


async def test_timeout_after_side_effect_is_never_replayed(server, monkeypatch):
    monkeypatch.setattr(mcptools, "_TOOL_CALL_TIMEOUT_S", 0.1)
    async with MCPToolset({"test": server.config()}, {}) as ts:
        failed = await invoke(ts, "slow")
        assert server.side_effect.is_set()
        assert failed.status == "error"
        assert "outcome unknown" in str(failed.content)
        assert "not replayed" in str(failed.content)
        server.release.set()
        assert value(await invoke(ts)) == "1"
        assert server.calls == [("1", "slow"), ("1", "read")]
        assert server.initializations == 1


async def test_cancel_call_and_shutdown_clean_up_owners(server):
    async with MCPToolset({"test": server.config()}, {}) as ts:
        call = asyncio.create_task(invoke(ts, "slow"))
        await server.side_effect.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        server.release.set()
        assert value(await invoke(ts)) == "1"
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "mcp:test"]
    assert not server.sessions


@pytest.mark.parametrize("change", ["tool_name", "description", "instructions"])
async def test_changed_contract_is_rejected(server, change):
    async with MCPToolset({"test": server.config()}, {}) as ts:
        server.sessions.clear()
        assert (await invoke(ts)).status == "error"
        setattr(server, change, "changed")
        failed = await invoke(ts)
        assert failed.status == "error"
        assert "changed tools or instructions" in str(failed.content)
        assert not server.calls
        assert [t.name for t in ts.tools] == ["test__state"]


@pytest.mark.parametrize("action", ["rpc_error", "server_error", "tool_error"])
async def test_ordinary_errors_do_not_reconnect(server, action):
    async with MCPToolset({"test": server.config()}, {}) as ts:
        failed = await invoke(ts, action)
        assert failed.status == "error"
        assert "session lost" not in str(failed.content)
        assert value(await invoke(ts)) == "0"
        assert server.initializations == 1


async def test_late_failure_does_not_invalidate_replacement(server):
    async with MCPToolset({"test": server.config()}, {}) as ts:
        delayed = asyncio.create_task(invoke(ts, "delayed_loss"))
        await server.delayed.wait()
        server.sessions.clear()
        assert (await invoke(ts)).status == "error"
        assert value(await invoke(ts, "increment")) == "1"
        server.finish_delayed.set()
        assert (await delayed).status == "error"
        assert value(await invoke(ts)) == "1"
        assert server.initializations == 2


async def test_cancel_during_reconnect_leaves_peer_usable(server):
    peer = HTTPServer()
    await peer.start()
    try:
        async with MCPToolset(
            {"test": server.config(), "peer": peer.config()}, {}
        ) as ts:
            server.sessions.clear()
            assert (await invoke(ts)).status == "error"
            server.hang_initialize = True
            server.initializing.clear()
            reconnect = asyncio.create_task(invoke(ts))
            await server.initializing.wait()
            reconnect.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reconnect
            assert value(await invoke(ts, server="peer")) == "0"
    finally:
        await peer.stop()
    assert not [t for t in asyncio.all_tasks() if t.get_name().startswith("mcp:")]


async def test_cancel_startup_cleans_already_started_peer(server):
    peer = HTTPServer()
    await peer.start()
    try:
        server.hang_initialize = True
        ts = MCPToolset({"peer": peer.config(), "test": server.config()}, {})
        startup = asyncio.create_task(ts.__aenter__())
        await server.initializing.wait()
        assert peer.initializations == 1
        startup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await startup
        assert not peer.sessions
        assert not [t for t in asyncio.all_tasks() if t.get_name().startswith("mcp:")]
    finally:
        await peer.stop()


async def test_idle_read_timeouts_exhaust_sdk_get_retries_but_not_session(
    server, caplog
):
    import logging
    from datetime import timedelta

    config = server.config() | {"sse_read_timeout": timedelta(seconds=0.05)}
    with caplog.at_level(logging.DEBUG, logger="mcp.client.streamable_http"):
        async with MCPToolset({"test": config}, {}) as ts:
            assert value(await invoke(ts, "increment")) == "1"
            await eventually(lambda: "max reconnection attempts" in caplog.text)
            assert server.gets == 2
            assert value(await invoke(ts)) == "1"
            assert server.initializations == 1


async def test_default_idle_get_stays_open_and_answers_server_ping(server):
    import httpx
    from mcp.shared._httpx_utils import create_mcp_http_client

    def short_read_timeout(**kwargs):
        return create_mcp_http_client(**(kwargs | {"timeout": httpx.Timeout(0.05)}))

    config = server.config() | {"httpx_client_factory": short_read_timeout}
    async with MCPToolset({"test": config}, {}) as ts:
        assert value(await invoke(ts, "increment")) == "1"
        await eventually(lambda: server.gets == 1)
        await asyncio.sleep(0.15)
        await server.messages.put(
            {"jsonrpc": "2.0", "id": "heartbeat", "method": "ping"}
        )
        await eventually(lambda: len(server.responses) == 1)
        assert server.responses[0].id == "heartbeat"
        assert server.gets == 1
        assert value(await invoke(ts)) == "1"
        assert server.initializations == 1


async def test_sdk_request_timeout_after_side_effect_is_unknown(server):
    from datetime import timedelta

    config = server.config() | {
        "session_kwargs": {"read_timeout_seconds": timedelta(seconds=0.05)}
    }
    async with MCPToolset({"test": config}, {}) as ts:
        failed = await invoke(ts, "slow")
        assert server.side_effect.is_set()
        assert failed.status == "error"
        assert "outcome unknown" in str(failed.content)
        assert "not replayed" in str(failed.content)
        server.release.set()
        assert value(await invoke(ts)) == "1"
        assert server.calls == [("1", "slow"), ("1", "read")]
        assert server.initializations == 1


async def test_startup_timeout_also_bounds_generation_cleanup(server, monkeypatch):
    from time import monotonic

    server.hang_delete = True
    monkeypatch.setattr(mcptools, "_SERVER_START_TIMEOUT_S", 0.05)
    monkeypatch.setattr(mcptools, "_SERVER_CLOSE_TIMEOUT_S", 0.05)
    server.hang_list = True
    started = monotonic()
    async with MCPToolset({"test": server.config()}, {}) as ts:
        assert ts.tools == []
    assert server.deleting.is_set()
    assert monotonic() - started < 0.5
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "mcp:test"]


async def test_cancelling_shutdown_still_closes_peer(server, monkeypatch):
    peer = HTTPServer()
    await peer.start()
    try:
        ts = await MCPToolset(
            {"test": server.config(), "peer": peer.config()}, {}
        ).__aenter__()
        server.hang_delete = True
        monkeypatch.setattr(mcptools, "_SERVER_CLOSE_TIMEOUT_S", 0.1)
        closing = asyncio.create_task(ts.__aexit__(None, None, None))
        await server.deleting.wait()
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert not peer.sessions
        assert not [t for t in asyncio.all_tasks() if t.get_name().startswith("mcp:")]
    finally:
        await peer.stop()


async def test_changed_output_schema_is_rejected_before_tool_call(server):
    server.output_schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
    }
    async with MCPToolset({"test": server.config()}, {}) as ts:
        server.sessions.clear()
        assert (await invoke(ts)).status == "error"
        server.output_schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
        }
        failed = await invoke(ts)
        assert failed.status == "error"
        assert "changed tools or instructions" in str(failed.content)
        assert server.calls == []


async def test_reconnect_wait_uses_tool_call_budget(server, monkeypatch):
    from time import monotonic

    monkeypatch.setattr(mcptools, "_TOOL_CALL_TIMEOUT_S", 0.05)
    monkeypatch.setattr(mcptools, "_SERVER_START_TIMEOUT_S", 0.5)
    async with MCPToolset({"test": server.config()}, {}) as ts:
        server.sessions.clear()
        assert (await invoke(ts)).status == "error"
        server.hang_initialize = True
        started = monotonic()
        failed = await invoke(ts)
        assert monotonic() - started < 0.25
        assert failed.status == "error"
        assert "call was not sent" in str(failed.content)
        assert server.calls == []


async def test_reconnect_and_execution_share_one_call_budget(server, monkeypatch):
    monkeypatch.setattr(mcptools, "_TOOL_CALL_TIMEOUT_S", 0.2)
    async with MCPToolset({"test": server.config()}, {}) as ts:
        server.sessions.clear()
        assert (await invoke(ts)).status == "error"
        server.initialize_delay = 0.13

        async def release_later():
            await server.side_effect.wait()
            await asyncio.sleep(0.13)
            server.release.set()

        release = asyncio.create_task(release_later())
        try:
            failed = await invoke(ts, "slow")
            assert failed.status == "error"
            assert "outcome unknown" in str(failed.content)
            assert server.side_effect.is_set()
            assert server.calls == [("2", "slow")]
        finally:
            release.cancel()
            await asyncio.gather(release, return_exceptions=True)
            server.release.set()
        assert value(await invoke(ts)) == "1"

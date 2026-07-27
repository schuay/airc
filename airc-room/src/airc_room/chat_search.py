# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""search_chat: grep the room's own chat history (airc's SQLite store).

A local (non-MCP) langchain tool. Unlike the old airc-tools MCP version it reuses
airc's configured db directly and scopes to the CALLER's space automatically
(from its thread), so it needs no separate MCP server, no AIRC_DB_PATH, and no
AIRC_CHAT_SPACE -- a multi-space deployment is correct without configuration.
Every chat persona gets it by default (like the timer tools); it is read-only over the
room's own history, so there is no tool_group to grant and nothing to configure.

Read-only recall over past messages -- "when did we discuss X", a landed CL that
fixes a previously discussed issue. Modeled on repo_git_grep: a regex with
optional context and case-insensitivity, plus chat filters (sender, thread,
since). The db is opened read-only (WAL, so concurrent with the live daemon).
"""

from __future__ import annotations

import asyncio
import os
import re
import sqlite3
import time
from datetime import datetime

import regex  # re with a per-search timeout, to bound a runaway pattern

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

# Bytes of matches returned per call, so a broad grep over a long history cannot
# dominate the turn; the caller narrows the query for the rest.
_MAX_OUTPUT = 100_000
# Wall-clock bound on the whole regex scan. The pattern comes from an LLM tool
# call driven by humans in a shared room; a catastrophic-backtracking pattern
# must not pin a worker on a full-table scan. Enforced per-match and globally.
_SEARCH_TIMEOUT_S = 10.0


def _thread_from_config(config: RunnableConfig | None) -> int | None:
    """The caller's thread id, from configurable.thread_id = "<thread_id>:<agent>"."""
    try:
        raw = (config or {}).get("configurable", {}).get("thread_id", "")
        return int(str(raw).partition(":")[0])
    except (ValueError, AttributeError):
        return None


def _parse_since(s: str) -> float | None:
    """`<N>d`/`<N>h`/`<N>m` relative, or an ISO date/datetime, to an epoch. None
    if unparseable (the caller reports it)."""
    s = s.strip().lower()
    if m := re.fullmatch(r"(\d+)\s*([dhm])", s):
        n, unit = int(m.group(1)), m.group(2)
        return time.time() - n * {"d": 86400, "h": 3600, "m": 60}[unit]
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="minutes")


def _render(conn, row, context: int) -> str:
    msg_id, thread_id, title, sender, text, ts = row
    head = f'-- thread {thread_id} "{title}" --'
    if not context:
        return f"{head}\n{_iso(ts)} {sender}: {text}"
    # The `context` messages before and after, same thread; ids are global-
    # monotonic, so id order is time order.
    before = conn.execute(
        "SELECT sender, text, ts FROM messages WHERE thread_id = ? AND id < ? "
        "ORDER BY id DESC LIMIT ?",
        (thread_id, msg_id, context),
    ).fetchall()[::-1]
    after = conn.execute(
        "SELECT sender, text, ts FROM messages WHERE thread_id = ? AND id > ? "
        "ORDER BY id ASC LIMIT ?",
        (thread_id, msg_id, context),
    ).fetchall()
    lines = [head]
    for s, t, mt in before:
        lines.append(f"  {_iso(mt)} {s}: {t}")
    lines.append(f"> {_iso(ts)} {sender}: {text}")
    for s, t, mt in after:
        lines.append(f"  {_iso(mt)} {s}: {t}")
    return "\n".join(lines)


def _search(
    db_path: str,
    thread_id: int | None,
    pattern: str,
    sender: str,
    thread: str,
    since: str,
    context: int,
    ignore_case: bool,
    limit: int,
) -> str:
    if not os.path.exists(db_path):
        return f"chat store not found at {db_path}"
    try:
        rx = regex.compile(pattern, regex.I if ignore_case else 0)
    except regex.error as e:
        return f"invalid pattern {pattern!r}: {e}"
    since_ts = None
    if since:
        since_ts = _parse_since(since)
        if since_ts is None:
            return f"could not parse since={since!r} (use e.g. 7d, 24h, 2026-06-01)"

    limit = max(1, min(limit, 500))
    context = max(0, min(context, 20))

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # Scope to the caller's own space so one space's history never leaks into
        # another. A thread not in chat_threads (e.g. the console) has no space,
        # so the search spans the db -- correct for a single-space room.
        space = None
        if thread_id is not None:
            row = conn.execute(
                "SELECT space FROM chat_threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            space = row[0] if row else None

        # Bound the scan: each match gets whatever remains of the global budget,
        # and once it is spent every later row returns 0 immediately, so the query
        # finishes fast with partial results rather than pinning a worker.
        deadline = time.monotonic() + _SEARCH_TIMEOUT_S
        timed_out = False

        def chat_match(v):
            nonlocal timed_out
            if v is None or timed_out:
                return 0
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                return 0
            try:
                return 1 if rx.search(v, timeout=remaining) else 0
            except TimeoutError:
                timed_out = True
                return 0

        conn.create_function("chat_match", 1, chat_match)
        where = ["chat_match(m.text)"]
        params: list = []
        if sender:
            where.append("m.sender LIKE ?")
            params.append(f"%{sender}%")
        if thread:
            where.append("(t.title LIKE ? OR m.thread_id = ?)")
            params += [f"%{thread}%", thread if thread.isdigit() else -1]
        if since_ts is not None:
            where.append("m.ts >= ?")
            params.append(since_ts)
        if space:
            where.append(
                "m.thread_id IN (SELECT thread_id FROM chat_threads WHERE space = ?)"
            )
            params.append(space)
        sql = (
            "SELECT m.id, m.thread_id, t.title, m.sender, m.text, m.ts "
            "FROM messages m JOIN threads t ON t.id = m.thread_id "
            f"WHERE {' AND '.join(where)} ORDER BY m.ts DESC LIMIT ?"
        )
        rows = conn.execute(sql, [*params, limit]).fetchall()
        if not rows:
            if timed_out:
                return (
                    f"search timed out after {_SEARCH_TIMEOUT_S:.0f}s with no matches"
                    " yet; simplify the pattern or add filters (sender/thread/since)"
                )
            return "no matches"
        blocks = [_render(conn, r, context) for r in rows]
    finally:
        conn.close()

    body = "\n".join(blocks)
    header = f"{len(rows)} match{'es' if len(rows) != 1 else ''} (newest first)"
    if timed_out:
        header += f"; timed out at {_SEARCH_TIMEOUT_S:.0f}s -- partial results"
    if len(body) > _MAX_OUTPUT:
        header += f"; {len(body) - _MAX_OUTPUT} bytes truncated, narrow the search"
        body = body[:_MAX_OUTPUT]
    return f"{header}\n\n{body}"


def make_search_chat_tool(db_path: str):
    """A local langchain search tool bound to airc's db. Added to every chat
    persona by default (the runner wires it unconditionally)."""

    @tool
    async def search_chat(
        pattern: str,
        sender: str = "",
        thread: str = "",
        since: str = "",
        context: int = 0,
        ignore_case: bool = False,
        limit: int = 50,
        config: RunnableConfig = None,
    ) -> str:
        """Grep the room's own chat history (past messages) with a regex `pattern`.
        Recall across threads -- e.g. when a landed CL fixes a previously discussed
        issue, or a message references another thread. Filters: `sender`
        (substring), `thread` (title substring or id), `since` (e.g. `7d`, `24h`,
        `2026-06-01`). `context` includes surrounding messages in the same thread
        (like grep -C); `ignore_case` and `limit` as usual. Results are newest-
        first and byte-capped. Scoped to this space automatically."""
        return await asyncio.to_thread(
            _search,
            db_path,
            _thread_from_config(config),
            pattern,
            sender,
            thread,
            since,
            context,
            ignore_case,
            limit,
        )

    return search_chat

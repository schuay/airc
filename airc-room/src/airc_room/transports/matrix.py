# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Matrix transport (matrix-nio; in-core).

A single bot user backs the whole room: one fixed identity carries every
persona, so a persona is rendered inside the message (a bold "name" prefix)
rather than as a distinct Matrix user.

Outbound (`deliver`): agent and watcher messages are sent into the Matrix room
their room thread maps to. Flat by default -- every message lands in the room
timeline. With `use_threads`, a room thread is pinned to an `m.thread` relation
(rooted at the first message sent for it) so a thread-aware client groups the
conversation; the relation lives entirely in the sent event's content, so
turning threads on or off never reshapes anything the room core sees.

Inbound (`run`): an `AsyncClient.sync_forever` loop. A first throwaway sync
establishes a token at startup so the backlog is not replayed (a bot that
answers hours-old messages after a restart is worse than one that stays quiet);
from there, only messages that arrive while running are forwarded. Own (bot)
echoes are dropped so agents never react to themselves, and each event is
deduplicated by event id.

Encryption: unencrypted for v1. The seam for E2E is nio's own store plus the
`encryption_enabled` client flag -- a later `[e2e]` extra (python-olm) flips it
on and the send path already routes through `room_send`, which encrypts
transparently for an encrypted room. Nothing in this module's shape assumes
plaintext, so E2E plugs in without an interface change (see design D4).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from html import escape

import mistune
from nio import (
    AsyncClient,
    AsyncClientConfig,
    MatrixRoom,
    RoomMessageText,
)

from ..room import Room
from ..store import Message, MessageKind, Store

log = logging.getLogger(__name__)

# Personas write markdown; Matrix clients render the HTML `formatted_body`, not
# markdown, so the body is converted here. escape=True neutralizes any raw HTML a
# message contains (it becomes text, never live tags) -- the only sanitization we
# need, since the output tag set (strong/em/del, ul/ol/li, pre>code, blockquote,
# table, a, h1-6) is exactly what a Matrix client's allowlist renders and mistune
# emits nothing outside it. Tables and strikethrough are GFM extensions, on
# because FluffyChat (the operator's client) renders both natively.
_md = mistune.create_markdown(escape=True, plugins=["strikethrough", "table"])

# mistune wraps output in block <p> tags. For a one-paragraph message (the common
# agent line) that block wrapper only adds vertical space, so a lone wrapping <p>
# is unwrapped; multi-block content (lists, tables, code) keeps its structure.
_SINGLE_PARAGRAPH = re.compile(r"\A<p>(?P<inner>.*)</p>\Z", re.DOTALL)


def _md_to_html(text: str) -> str:
    """Render a persona's markdown body to the Matrix HTML subset, with a lone
    wrapping <p> stripped so a single-line message is not boxed as a paragraph."""
    html = _md(text).strip()
    if m := _SINGLE_PARAGRAPH.match(html):
        # Only when there is exactly one paragraph (no nested block would leave a
        # </p> in the middle): a multi-block body keeps its <p> wrappers.
        inner = m.group("inner")
        if "<p>" not in inner:
            return inner
    return html


# Backoff bounds for the sync loop when the homeserver is unreachable. nio
# handles per-request retries and 429/Retry-After internally; this is the outer
# guard for a sync that raises out of sync_forever entirely (network down,
# token revoked), so the loop retries instead of the task dying.
_SYNC_RETRY_MIN_S = 5.0
_SYNC_RETRY_MAX_S = 60.0
# How long the server holds a long-poll sync open before returning empty.
_SYNC_TIMEOUT_MS = 30_000

# A Matrix typing notice is a server-side state that AUTO-EXPIRES after the
# timeout sent with it. A single notice therefore cannot cover a turn that runs
# for minutes -- the indicator would vanish mid-thought. So the transport runs a
# refresh loop that re-sends the notice on an interval shorter than its lifetime
# for as long as any agent is thinking in the room. The notice lifetime is kept
# modest so that if the bot dies the indicator clears on its own within it.
_TYPING_NOTICE_MS = 20_000
_TYPING_REFRESH_S = 12.0  # < _TYPING_NOTICE_MS/1000, so the notice never lapses


def _render(msg: Message) -> tuple[str, str | None]:
    """(plain body, html body|None) for a room message.

    Plain text is always set (a client with no HTML rendering still reads it);
    the HTML body carries the bold persona prefix / italic notice. A numeric or
    email ping renders as plain text -- a real Matrix @mention (a pill keyed by
    mxid) is a later refinement, so for now the named user is visible but not
    notified.
    """
    if msg.kind == MessageKind.PING:
        return f"ping {msg.text}", None
    if msg.kind in (MessageKind.NOTICE, MessageKind.SYSTEM):
        # A one-line notice reads as an italic aside; a multi-line body (a
        # rendered result with its own headline, lists, or a table) is converted
        # through markdown so that structure renders instead of arriving as raw
        # asterisks and pipes.
        if "\n" in msg.text:
            return msg.text, _md_to_html(msg.text)
        return msg.text, f"<em>{escape(msg.text)}</em>"
    # Agent / event message: bold persona prefix, no colon (a copied "name:"
    # would read as an address when pasted back), matching the gchat transport.
    # The body is the persona's markdown rendered to HTML; the prefix is prepended
    # as literal <strong> since the sender name is not itself markdown.
    body = f"{msg.sender} {msg.text}"
    html = f"<strong>{escape(msg.sender)}</strong> {_md_to_html(msg.text)}"
    return body, html


class MatrixTransport:
    """The core Matrix Transport. Flat by default; threads map onto m.thread."""

    name = "matrix"

    def __init__(self, cfg, room: Room, store: Store) -> None:
        # cfg is the MatrixConfig dataclass (airc_room.config.MatrixConfig).
        self._cfg = cfg
        self._room = room
        self._store = store
        # store_sync_tokens stays off: v1 deliberately does NOT persist the sync
        # position, so a restart resumes from "now" and never floods the room
        # with a downtime backlog. encryption_enabled is the E2E seam, off for v1.
        client_config = AsyncClientConfig(
            store_sync_tokens=False,
            encryption_enabled=False,
        )
        self._client = AsyncClient(
            cfg.homeserver,
            cfg.user_id,
            device_id=cfg.device_id or "",
            config=client_config,
        )
        # Token login: no password derives, so the bootstrap is one homeserver
        # call. restore_login sets the credentials the send/sync paths need.
        self._client.restore_login(
            user_id=cfg.user_id,
            device_id=cfg.device_id or "",
            access_token=cfg.access_token,
        )
        self._rooms = set(cfg.room_ids)
        # Typing indicator state, keyed by matrix room id. The bot is one Matrix
        # user but several agents can think at once in a room, so a per-room
        # ref-count keeps the notice up while ANY agent is typing and clears it
        # only when the last finishes; one refresh task per room re-sends the
        # notice before it expires.
        self._typing_count: dict[str, int] = {}
        self._typing_tasks: dict[str, asyncio.Task] = {}

    # -- routing --------------------------------------------------------------

    def _room_and_thread(self, thread_id: int) -> tuple[str | None, str | None]:
        """(matrix room id, thread root event id|None) for a room thread.

        The store's generic chat_threads mapping is reused (space = room id,
        chat_thread = thread root event id, "" for a flat thread). An unmapped
        room thread (a proactive announcement with no inbound origin) falls back
        to the first configured room; None room id means nowhere to send.
        """
        linked = self._store.chat_thread_for_thread(thread_id)
        if linked:
            room_id, root = linked
            return room_id, (root or None)
        fallback = self._cfg.room_ids[0] if self._cfg.room_ids else None
        return fallback, None

    def _thread_relation(self, root_event_id: str) -> dict:
        # A reply that "falls back" to a normal reply for non-threaded clients:
        # is_falling_back + m.in_reply_to keep the message visible in the flat
        # timeline of a client that does not render m.thread.
        return {
            "m.relates_to": {
                "rel_type": "m.thread",
                "event_id": root_event_id,
                "is_falling_back": True,
                "m.in_reply_to": {"event_id": root_event_id},
            }
        }

    # -- outbound (room -> Matrix) --------------------------------------------

    async def deliver(self, msg: Message) -> None:
        # Humans already see their own messages in the room.
        if msg.kind == MessageKind.HUMAN:
            return
        room_id, root = self._room_and_thread(msg.thread_id)
        if not room_id:
            log.warning(
                "matrix: no room mapped for thread %d; dropping %s",
                msg.thread_id,
                msg.kind,
            )
            return
        body, html = _render(msg)
        content: dict = {"msgtype": "m.text", "body": body}
        if html is not None:
            content["format"] = "org.matrix.custom.html"
            content["formatted_body"] = html
        # Thread the message only when threads are enabled AND this room thread
        # already has a root; the first message for a thread is sent flat and
        # becomes that root (linked below), so later ones relate back to it.
        if self._cfg.use_threads and root:
            content.update(self._thread_relation(root))
        resp = await self._client.room_send(
            room_id, message_type="m.room.message", content=content
        )
        event_id = getattr(resp, "event_id", None)
        if not event_id:
            log.error("matrix: send to %s failed: %s", room_id, resp)
            return
        log.info("matrix: sent %s to %s", msg.kind, room_id)
        # First send for this room thread: record it as the thread root so
        # subsequent sends (and, with threads on, the relation) converge on it.
        if self._cfg.use_threads and not root:
            self._store.link_chat_thread(room_id, event_id, msg.thread_id)

    async def typing(
        self, thread_id: int, sender: str, active: bool, budget: float | None = None
    ) -> None:
        """Show/hide the room's typing indicator around an agent turn.

        A Matrix typing notice auto-expires, so a turn that runs for minutes needs
        the notice refreshed, not sent once. This ref-counts typers per room (the
        bot is one user; agents can overlap) and runs a refresh task while the
        count is positive, so the indicator persists for as long as any agent is
        thinking and clears promptly when the last one stops. The budget is unused
        -- the refresh loop, not a fixed countdown, bounds how long it shows."""
        room_id, _ = self._room_and_thread(thread_id)
        if not room_id:
            return
        if active:
            first = self._typing_count.get(room_id, 0) == 0
            self._typing_count[room_id] = self._typing_count.get(room_id, 0) + 1
            if first:
                # Send the first notice inline so the indicator appears at once
                # (not after the refresh task is first scheduled), then start the
                # loop that keeps it alive.
                await self._send_typing(room_id)
                self._typing_tasks[room_id] = asyncio.create_task(
                    self._typing_refresh(room_id), name=f"matrix-typing:{room_id}"
                )
        else:
            remaining = self._typing_count.get(room_id, 0) - 1
            if remaining > 0:
                self._typing_count[room_id] = remaining
                return  # another agent is still thinking; keep the notice up
            self._typing_count.pop(room_id, None)
            await self._stop_typing(room_id)

    async def _send_typing(self, room_id: str) -> None:
        """Send one typing notice with the auto-expiry the refresh loop renews. A
        transient failure is swallowed -- the next refresh retries and the notice
        self-expires if the bot is gone, so a drop never pins a stale indicator."""
        try:
            await self._client.room_typing(room_id, True, timeout=_TYPING_NOTICE_MS)
        except Exception:
            log.debug("matrix: typing notice failed for %s", room_id)

    async def _typing_refresh(self, room_id: str) -> None:
        """Re-send the typing notice before it expires, for as long as an agent is
        thinking in the room. The first notice is sent inline by typing(); this
        loop only renews it. One task per active room; cancelled by _stop_typing
        when the last typer finishes."""
        while self._typing_count.get(room_id, 0) > 0:
            await asyncio.sleep(_TYPING_REFRESH_S)
            if self._typing_count.get(room_id, 0) > 0:
                await self._send_typing(room_id)

    async def _stop_typing(self, room_id: str) -> None:
        """Cancel the refresh task and clear the notice for a room."""
        task = self._typing_tasks.pop(room_id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        try:
            await self._client.room_typing(room_id, False)
        except Exception:
            log.debug("matrix: clearing typing failed for %s", room_id, exc_info=True)

    # -- inbound (Matrix -> room) ---------------------------------------------

    async def run(self) -> None:
        # Join the rooms we serve, then take a throwaway first sync so the token
        # advances past the existing backlog before callbacks are attached --
        # otherwise sync_forever would replay the room's whole recent history on
        # startup and the bot would answer stale messages.
        for room_id in self._rooms:
            resp = await self._client.join(room_id)
            if getattr(resp, "room_id", None):
                log.info("matrix: joined %s", room_id)
            else:
                log.warning("matrix: could not join %s: %s", room_id, resp)
        await self._client.sync(timeout=0, full_state=False)
        self._client.add_event_callback(self._on_message, RoomMessageText)
        log.info("matrix: syncing as %s", self._cfg.user_id)
        backoff = _SYNC_RETRY_MIN_S
        while True:
            try:
                await self._client.sync_forever(
                    timeout=_SYNC_TIMEOUT_MS, full_state=False
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("matrix: sync failed; retrying in %.0fs", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _SYNC_RETRY_MAX_S)
            else:
                backoff = _SYNC_RETRY_MIN_S

    async def _on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        # Serve only configured rooms, so a stray invite the bot somehow joined
        # never gets answered.
        if room.room_id not in self._rooms:
            return
        # Drop our own echoes so agents never react to themselves.
        if event.sender == self._cfg.user_id:
            return
        # Dedup by event id: a sync can redeliver an event across reconnects.
        if not self._store.mark_chat_message(event.event_id):
            return
        sender = room.user_name(event.sender) or event.sender
        if sender != event.sender:
            self._store.set_chat_user(event.sender, sender)
        thread_id = self._resolve_thread(room, event)
        log.info("matrix: inbound from %s in %s", sender, room.room_id)
        await self._room.post(thread_id, sender, MessageKind.HUMAN, event.body)

    def _resolve_thread(self, room: MatrixRoom, event: RoomMessageText) -> int:
        """Map an inbound event to a room thread id.

        Flat by default: everything from a room routes to one room thread keyed
        by (room, ""). With threads on, an event carrying an m.thread relation
        routes to the room thread keyed by (room, thread root event id), so a
        reply in a Matrix thread continues the matching airc thread.
        """
        root = ""
        if self._cfg.use_threads:
            relates = (event.source.get("content", {}) or {}).get("m.relates_to", {})
            if relates.get("rel_type") == "m.thread":
                root = relates.get("event_id", "") or ""
        existing = self._store.chat_thread_id(room.room_id, root)
        if existing is not None:
            return existing
        thread = self._room.create_thread(room.display_name or "matrix")
        self._store.link_chat_thread(room.room_id, root, thread.id)
        return thread.id

    async def aclose(self) -> None:
        # Stop every refresh loop and clear its indicator before the client
        # closes, so shutdown never leaves a stale "typing..." in a room.
        self._typing_count.clear()
        for room_id in list(self._typing_tasks):
            await self._stop_typing(room_id)
        try:
            await self._client.close()
        except Exception:
            log.debug("matrix: client close failed", exc_info=True)

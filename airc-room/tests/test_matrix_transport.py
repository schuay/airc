# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""MatrixTransport against a faked matrix-nio AsyncClient.

The transport is exercised with a real in-memory Store (its thread-mapping is
load-bearing) and a stub client that records room_send calls and lets a test
drive the inbound callback. No network, no nio server. These pin the behaviour a
live smoke test cannot cheaply re-check every run: the sent content shape (plain
+ html, thread relation), own-echo/dedup filtering, and flat-vs-threaded routing.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace


from airc_room.config import MatrixConfig
from airc_room.room import Room
from airc_room.store import Message, MessageKind, Store
from airc_room.transports.matrix import MatrixTransport


class _Resp:
    def __init__(self, event_id="$evt:example.org"):
        self.event_id = event_id


class _FakeRoom:
    """Stands in for nio.MatrixRoom in an inbound callback."""

    def __init__(self, room_id, display_name="fam", names=None):
        self.room_id = room_id
        self.display_name = display_name
        self._names = names or {}

    def user_name(self, mxid):
        return self._names.get(mxid)


class _FakeEvent:
    """Stands in for nio.RoomMessageText."""

    def __init__(self, sender, body, event_id, relates_to=None):
        self.sender = sender
        self.body = body
        self.event_id = event_id
        content = {"body": body}
        if relates_to:
            content["m.relates_to"] = relates_to
        self.source = {"content": content}


class _FakeClient:
    """Records room_send/room_typing and captures the event callback."""

    def __init__(self, *a, **k):
        self.sent = []
        self.typing = []
        self.callback = None
        self.joined = []
        self._next_event_id = 0

    def restore_login(self, **k):
        self.login = k

    def add_event_callback(self, cb, filter):
        self.callback = cb

    async def join(self, room_id):
        self.joined.append(room_id)
        return SimpleNamespace(room_id=room_id)

    async def sync(self, **k):
        return SimpleNamespace(next_batch="s1")

    async def room_send(self, room_id, message_type, content):
        self._next_event_id += 1
        eid = f"$sent{self._next_event_id}:example.org"
        self.sent.append((room_id, content))
        return _Resp(eid)

    async def room_typing(self, room_id, state, timeout=None):
        self.typing.append((room_id, state))
        return SimpleNamespace()

    async def close(self):
        self.closed = True


def _cfg(use_threads=False, rooms=("!fam:example.org",)):
    return MatrixConfig(
        homeserver="https://matrix.example.org",
        user_id="@airc:example.org",
        access_token="syt_x",
        room_ids=list(rooms),
        use_threads=use_threads,
    )


def _make(tmp_path, monkeypatch, use_threads=False):
    monkeypatch.setattr("airc_room.transports.matrix.AsyncClient", _FakeClient)
    store = Store(tmp_path / "airc.db")
    room = Room(store)
    t = MatrixTransport(_cfg(use_threads=use_threads), room, store)
    return t, room, store, t._client


def _msg(thread_id, sender, kind, text):
    return Message(
        id=1, thread_id=thread_id, sender=sender, kind=kind, text=text, ts=0.0
    )


# -- outbound -----------------------------------------------------------------


async def test_deliver_renders_plain_and_html(tmp_path, monkeypatch):
    t, room, store, client = _make(tmp_path, monkeypatch)
    tid = room.default_thread().id
    await t.deliver(_msg(tid, "gc", MessageKind.AGENT, "look at <T>"))
    (room_id, content) = client.sent[0]
    assert room_id == "!fam:example.org"
    assert content["msgtype"] == "m.text"
    # Plain body is the raw text (markdown and all); the sender prefixes it.
    assert content["body"] == "gc look at <T>"
    # HTML bolds the persona and escapes the angle brackets (raw HTML inert). A
    # one-paragraph body has its wrapping <p> stripped.
    assert content["formatted_body"] == "<strong>gc</strong> look at &lt;T&gt;"
    assert "m.relates_to" not in content


async def test_deliver_converts_markdown_to_html(tmp_path, monkeypatch):
    # Personas write markdown; the formatted_body must carry the HTML a Matrix
    # client renders, not raw asterisks/pipes. The body stays the raw text.
    t, room, store, client = _make(tmp_path, monkeypatch)
    tid = room.default_thread().id
    md = "the **best** deal on ~~old~~ *fresh* eggs"
    await t.deliver(_msg(tid, "hawk", MessageKind.AGENT, md))
    content = client.sent[0][1]
    assert content["body"] == f"hawk {md}"
    html = content["formatted_body"]
    assert html.startswith("<strong>hawk</strong> ")
    assert "<strong>best</strong>" in html
    assert "<del>old</del>" in html
    assert "<em>fresh</em>" in html


async def test_deliver_renders_list_and_code_and_table(tmp_path, monkeypatch):
    # The block constructs the personas actually use -- a shopping list, a code
    # fence, a price table -- must reach the client as real HTML blocks.
    t, room, store, client = _make(tmp_path, monkeypatch)
    tid = room.default_thread().id
    body = "prices:\n\n| item | eur |\n|---|---|\n| eggs | 1.99 |\n\n- milk\n- bread"
    await t.deliver(_msg(tid, "hawk", MessageKind.AGENT, body))
    html = client.sent[0][1]["formatted_body"]
    assert "<table>" in html and "<th>item</th>" in html and "<td>eggs</td>" in html
    assert "<ul>" in html and "<li>milk</li>" in html
    # A multi-block body keeps its structure (not unwrapped to a single line).
    assert html.startswith("<strong>hawk</strong>")


async def test_deliver_multiline_system_announcement_converts(tmp_path, monkeypatch):
    # A multi-line SYSTEM/NOTICE body (headline + detail) is markdown-rendered so
    # its structure survives; a one-line notice stays a plain italic aside.
    t, room, store, client = _make(tmp_path, monkeypatch)
    tid = room.default_thread().id
    await t.deliver(_msg(tid, "airc", MessageKind.NOTICE, "**heads up**\n\n- a\n- b"))
    html = client.sent[0][1]["formatted_body"]
    assert "<strong>heads up</strong>" in html and "<li>a</li>" in html

    client.sent.clear()
    await t.deliver(_msg(tid, "airc", MessageKind.NOTICE, "one liner"))
    assert client.sent[0][1]["formatted_body"] == "<em>one liner</em>"


async def test_deliver_skips_human(tmp_path, monkeypatch):
    t, room, store, client = _make(tmp_path, monkeypatch)
    tid = room.default_thread().id
    await t.deliver(_msg(tid, "someone", MessageKind.HUMAN, "hi"))
    assert client.sent == []


async def test_deliver_flat_never_threads(tmp_path, monkeypatch):
    # use_threads off: even a mapped root does not add a relation.
    t, room, store, client = _make(tmp_path, monkeypatch, use_threads=False)
    tid = room.default_thread().id
    store.link_chat_thread("!fam:example.org", "$root:example.org", tid)
    await t.deliver(_msg(tid, "gc", MessageKind.AGENT, "x"))
    assert "m.relates_to" not in client.sent[0][1]


async def test_deliver_threaded_relates_to_root(tmp_path, monkeypatch):
    t, room, store, client = _make(tmp_path, monkeypatch, use_threads=True)
    tid = room.default_thread().id
    store.link_chat_thread("!fam:example.org", "$root:example.org", tid)
    await t.deliver(_msg(tid, "gc", MessageKind.AGENT, "x"))
    rel = client.sent[0][1]["m.relates_to"]
    assert rel["rel_type"] == "m.thread"
    assert rel["event_id"] == "$root:example.org"
    assert rel["is_falling_back"] is True


async def test_deliver_threaded_first_send_becomes_root(tmp_path, monkeypatch):
    # With threads on and no root yet, the first send goes flat and is recorded
    # as the thread root, so the next send relates back to it.
    t, room, store, client = _make(tmp_path, monkeypatch, use_threads=True)
    tid = room.default_thread().id
    await t.deliver(_msg(tid, "gc", MessageKind.AGENT, "first"))
    assert "m.relates_to" not in client.sent[0][1]
    assert store.chat_thread_for_thread(tid) == (
        "!fam:example.org",
        "$sent1:example.org",
    )
    await t.deliver(_msg(tid, "gc", MessageKind.AGENT, "second"))
    assert client.sent[1][1]["m.relates_to"]["event_id"] == "$sent1:example.org"


async def test_deliver_unmapped_thread_falls_back_to_first_room(tmp_path, monkeypatch):
    t, room, store, client = _make(tmp_path, monkeypatch)
    # A thread with no chat_threads link (a proactive post) still sends, to the
    # first configured room.
    thread = room.create_thread("proactive")
    await t.deliver(_msg(thread.id, "perf", MessageKind.SYSTEM, "[v8] regressed"))
    assert client.sent[0][0] == "!fam:example.org"


# -- inbound ------------------------------------------------------------------


async def test_inbound_posts_human_message(tmp_path, monkeypatch):
    t, room, store, client = _make(tmp_path, monkeypatch)
    fr = _FakeRoom("!fam:example.org", names={"@bob:example.org": "bob"})
    ev = _FakeEvent("@bob:example.org", "hello airc", "$in1:example.org")
    await t._on_message(fr, ev)
    # One thread created, one human message posted into it.
    msgs = store.thread_messages(store.list_threads()[0].id)
    assert len(msgs) == 1
    assert msgs[0].sender == "bob"
    assert msgs[0].kind == MessageKind.HUMAN
    assert msgs[0].text == "hello airc"


async def test_inbound_ignores_own_echo(tmp_path, monkeypatch):
    t, room, store, client = _make(tmp_path, monkeypatch)
    fr = _FakeRoom("!fam:example.org")
    ev = _FakeEvent("@airc:example.org", "my own reply", "$mine:example.org")
    await t._on_message(fr, ev)
    assert store.list_threads() == [] or all(
        not store.thread_messages(th.id) for th in store.list_threads()
    )


async def test_inbound_ignores_unserved_room(tmp_path, monkeypatch):
    t, room, store, client = _make(tmp_path, monkeypatch)
    fr = _FakeRoom("!stranger:example.org")
    ev = _FakeEvent("@bob:example.org", "hi", "$s:example.org")
    await t._on_message(fr, ev)
    assert store.list_threads() == []


async def test_inbound_dedups_by_event_id(tmp_path, monkeypatch):
    t, room, store, client = _make(tmp_path, monkeypatch)
    fr = _FakeRoom("!fam:example.org")
    ev = _FakeEvent("@bob:example.org", "twice", "$dup:example.org")
    await t._on_message(fr, ev)
    await t._on_message(fr, ev)
    total = sum(len(store.thread_messages(th.id)) for th in store.list_threads())
    assert total == 1


async def test_inbound_flat_routes_all_to_one_thread(tmp_path, monkeypatch):
    t, room, store, client = _make(tmp_path, monkeypatch, use_threads=False)
    fr = _FakeRoom("!fam:example.org")
    await t._on_message(fr, _FakeEvent("@bob:example.org", "one", "$a:example.org"))
    await t._on_message(fr, _FakeEvent("@ann:example.org", "two", "$b:example.org"))
    assert len(store.list_threads()) == 1


async def test_inbound_threaded_routes_by_relation(tmp_path, monkeypatch):
    t, room, store, client = _make(tmp_path, monkeypatch, use_threads=True)
    fr = _FakeRoom("!fam:example.org")
    # A root message (no relation) and a reply in its thread land in the SAME
    # airc thread; an unrelated root opens a second.
    await t._on_message(fr, _FakeEvent("@bob:example.org", "root", "$root:example.org"))
    reply_rel = {"rel_type": "m.thread", "event_id": "$root:example.org"}
    await t._on_message(
        fr, _FakeEvent("@ann:example.org", "reply", "$r:example.org", reply_rel)
    )
    # The reply's thread root ($root) maps to its own airc thread; the bare root
    # message mapped to (room, "") -- so this is 2 threads, and the reply thread
    # is keyed on the relation.
    assert store.chat_thread_id("!fam:example.org", "$root:example.org") is not None


# -- typing / lifecycle -------------------------------------------------------


async def test_typing_sends_notice_and_clears(tmp_path, monkeypatch):
    t, room, store, client = _make(tmp_path, monkeypatch)
    tid = room.default_thread().id
    store.link_chat_thread("!fam:example.org", "", tid)
    await t.typing(tid, "gc", True)
    assert ("!fam:example.org", True) in client.typing
    await t.typing(tid, "gc", False)
    # The last typer stopping clears the notice and tears down the refresh task.
    assert client.typing[-1] == ("!fam:example.org", False)
    assert "!fam:example.org" not in t._typing_tasks


async def test_typing_refreshes_before_expiry(tmp_path, monkeypatch):
    # A single Matrix notice auto-expires, so a long turn needs the notice
    # re-sent. Drive the refresh interval to ~0 and assert it fires repeatedly
    # while the agent is still typing.
    import airc_room.transports.matrix as mx

    monkeypatch.setattr(mx, "_TYPING_REFRESH_S", 0.01)
    t, room, store, client = _make(tmp_path, monkeypatch)
    tid = room.default_thread().id
    store.link_chat_thread("!fam:example.org", "", tid)
    await t.typing(tid, "gc", True)
    await asyncio.sleep(0.05)  # several refresh ticks
    await t.typing(tid, "gc", False)
    trues = [x for x in client.typing if x == ("!fam:example.org", True)]
    assert len(trues) >= 2  # the initial notice plus at least one refresh


async def test_typing_refcounts_concurrent_agents(tmp_path, monkeypatch):
    # The bot is one Matrix user but agents overlap: the notice must stay up while
    # ANY agent thinks and clear only when the last stops.
    t, room, store, client = _make(tmp_path, monkeypatch)
    tid = room.default_thread().id
    store.link_chat_thread("!fam:example.org", "", tid)
    await t.typing(tid, "gc", True)
    await t.typing(tid, "compiler", True)
    await t.typing(tid, "gc", False)  # one stops; the other still thinking
    assert "!fam:example.org" in t._typing_tasks  # notice still up
    assert client.typing[-1] != ("!fam:example.org", False)
    await t.typing(tid, "compiler", False)  # last one stops
    assert client.typing[-1] == ("!fam:example.org", False)
    assert "!fam:example.org" not in t._typing_tasks


async def test_aclose_closes_client(tmp_path, monkeypatch):
    t, room, store, client = _make(tmp_path, monkeypatch)
    await t.aclose()
    assert getattr(client, "closed", False) is True


async def test_aclose_clears_live_typing(tmp_path, monkeypatch):
    # Shutdown mid-turn must not leave a stale "typing..." in the room.
    t, room, store, client = _make(tmp_path, monkeypatch)
    tid = room.default_thread().id
    store.link_chat_thread("!fam:example.org", "", tid)
    await t.typing(tid, "gc", True)
    await t.aclose()
    assert client.typing[-1] == ("!fam:example.org", False)
    assert t._typing_tasks == {}


def test_conforms_to_transport_protocol(tmp_path, monkeypatch):
    # Transport is a plain (non-runtime-checkable) Protocol, so assert the two
    # required members structurally rather than via isinstance.
    t, *_ = _make(tmp_path, monkeypatch)
    assert t.name == "matrix"
    assert callable(t.run) and callable(t.deliver)

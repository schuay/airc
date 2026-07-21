"""IRC-style console transport.

A prompt_toolkit REPL: the prompt stays at the bottom while room traffic
(agent replies, watcher announcements, activity notes) prints above it via
patch_stdout. One console session acts as a single human user named after
the local account.

Commands:
    /help            show commands
    /agents          list agents
    /threads         list threads
    /t <id>          switch to thread <id>
    /new <title>     create a thread and switch to it
    /quit            exit
"""

from __future__ import annotations

import asyncio
import getpass
from datetime import datetime

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import print_formatted_text

from ..room import Room, Transport
from ..store import Message, MessageKind

_SENDER_COLORS = ["ansicyan", "ansigreen", "ansiyellow", "ansimagenta", "ansiblue"]

_HELP = """\
/help            show this help
/agents          list agents
/threads         list threads
/t <id>          switch to thread <id>
/new <title>     create a thread and switch to it
/quit            exit"""


def _color(sender: str) -> str:
    return _SENDER_COLORS[hash(sender) % len(_SENDER_COLORS)]


class ConsoleTransport(Transport):
    """The reference in-tree Transport. Renders threads flat -- it shows the
    thread id inline but has no native thread concept to map `thread_id` onto --
    and needs no E2E, so the encryption seam the protocol reserves is a no-op
    here."""

    name = "console"

    def __init__(self, room: Room, agents: dict) -> None:
        self._room = room
        self.agents = agents
        self.user = getpass.getuser()
        self._thread_id = room.default_thread().id
        self._session: PromptSession = PromptSession()
        self.done = asyncio.Event()

    # ── outbound (room → screen) ─────────────────────────────────────────────

    async def deliver(self, msg: Message) -> None:
        if msg.kind == MessageKind.HUMAN and msg.sender == self.user:
            return  # already on screen as typed input
        ts = datetime.fromtimestamp(msg.ts).strftime("%H:%M")
        style = (
            "italic ansibrightblack"
            if msg.kind in (MessageKind.SYSTEM, MessageKind.NOTICE, MessageKind.PING)
            else ""
        )
        # No real @mention on a console; show who would be pinged. A numeric user
        # id is opaque here, so only the email fallback reads usefully -- both are
        # rendered the same, prefixed, so the intent is visible.
        text = f"ping {msg.text}" if msg.kind == MessageKind.PING else msg.text
        line = [
            ("ansibrightblack", f"{ts} "),
            ("ansibrightblack", f"[t{msg.thread_id}] "),
            (_color(msg.sender), f"[{msg.sender}] "),
            (style, text),
        ]
        print_formatted_text(FormattedText(line))

    async def on_event(self, agent: str, event: str, detail: str) -> None:
        print_formatted_text(
            FormattedText([("ansibrightblack", f"* {agent} {event}: {detail}")])
        )

    # ── inbound (keyboard → room) ────────────────────────────────────────────

    async def run(self) -> None:
        try:
            with patch_stdout():
                await self._loop()
        finally:
            self.done.set()

    async def _loop(self) -> None:
        while True:
            try:
                text = await self._session.prompt_async(f"[t{self._thread_id}] > ")
            except (EOFError, KeyboardInterrupt):
                return
            text = text.strip()
            if not text:
                continue
            if text.startswith("/"):
                if await self._command(text):
                    return
                continue
            await self._room.post(self._thread_id, self.user, MessageKind.HUMAN, text)

    async def _command(self, text: str) -> bool:
        """Handle a slash command; return True to exit."""
        cmd, _, arg = text.partition(" ")
        arg = arg.strip()
        match cmd:
            case "/quit" | "/exit" | "/q":
                return True
            case "/help":
                print(_HELP)
            case "/agents":
                for p in self.agents.values():
                    print(f"  {p.name + ':':<13} {p.description}")
            case "/threads":
                for t in self._room.list_threads():
                    marker = "*" if t.id == self._thread_id else " "
                    print(f" {marker} t{t.id:<4} {t.title}")
            case "/t" | "/thread":
                self._switch(arg)
            case "/new":
                if not arg:
                    print("usage: /new <title>")
                else:
                    t = self._room.create_thread(arg)
                    self._thread_id = t.id
            case _:
                print(f"unknown command {cmd!r}; /help for help")
        return False

    def _switch(self, arg: str) -> None:
        tid = arg.lstrip("t")
        if not tid.isdigit() or not self._room.get_thread(int(tid)):
            print(f"no such thread {arg!r}; /threads to list")
            return
        self._thread_id = int(tid)
        for m in self._room.thread_messages(self._thread_id)[-10:]:
            ts = datetime.fromtimestamp(m.ts).strftime("%H:%M")
            print(f"{ts} [{m.sender}] {m.text}")

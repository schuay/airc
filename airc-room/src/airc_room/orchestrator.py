# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Routing: who speaks next.

The orchestrator consumes every message posted to the room and decides which
agents respond, mimicking a human group chat:

- A plugin may register message handlers, run on each message before any
  routing. One that CONSUMES a message ends it there: no mention parse, no
  coordinator, no turn. This is how an app answers something mechanical (a
  command a human typed) without spending a persona turn on it, and it is the
  only delivery a plugin gets -- everything below routes to personas alone.
- An address (a leading "handle:" prefix) forces a reply from the named agents.
- Otherwise a single COORDINATOR call on the fast model decides whether anyone
  should reply and who: it sees the agent roster, a bounded recent window with
  human/agent tags, and computed signals. Its policy defaults to silence and
  explicitly protects human-to-human conversation; routing fails closed.
- Watcher announcements (kind "system") skip the should-anyone-speak gate:
  they only exist because triage already judged them worth commentary, so a
  dedicated router picks the ONE best-fit commentator. NOTHING_TO_ADD at turn
  time remains the escape hatch for misroutes.
- Agent messages can trigger further agent replies (discussion). There is no
  hard cutoff: as the streak of consecutive agent messages grows past
  soft_turn_budget the coordinator is told the bar has risen, and past
  max_turns that only a decisive contribution may continue -- so exchanges
  peter out naturally, running long only while genuinely substantive.

Concurrency model: a dispatcher routes each message to a per-thread worker.
Each worker *routes* its thread's messages in id order (so streak math and the
coordinator see an ordered transcript) but does not wait for the turns: each
responder runs as a detached task under a per-(thread, agent) lock, so one
agent's turns on a thread are serialized (the invariant run_turn relies on for
its seen-offset/checkpoint) while different agents, and different threads, run
fully in parallel -- a slow agent never blocks the next message. A global
semaphore bounds the total number of simultaneous agent turns. A per-thread
watermark commits a message only once it and every earlier message on the
thread have finished (a contiguous prefix), so out-of-order completion never
loses work; startup recovery replays persisted messages above the watermark, so
queued work survives a process crash.

Failure semantics: an in-turn error is surfaced to the room as a "(name
errored)" notice and the message is then committed (not replayed) -- so a
genuinely poisoned message cannot crash-loop, at the cost of not auto-retrying
a transient failure (the room sees the notice and a human can re-ask). The
replay guarantee covers process crash / shutdown cancellation, NOT a turn that
ran and errored. ModelRetryMiddleware already retries transient errors with
backoff *inside* the turn before it is declared failed.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from airc_core import TokenLog, make_model, retrying, usage_counts

from .config import Config
from .room import Room
from .runner import AgentRunner
from .store import Message, MessageKind, Store

# An app-registered response handler for a SYSTEM announcement, keyed by the
# message's follow_up string. It owns the response (prompt, parse, render, side
# effects) via the TurnContext the room lends it; the room stays domain-blind.
FollowUp = Callable[["TurnContext"], Awaitable[None]]


class Disposition(StrEnum):
    """What a message handler decided about the message it was given.

    CONSUMED suppresses ORCHESTRATION only. The message is already persisted and
    already delivered to every transport by the time a handler sees it, and its
    `kind` is left alone -- so the store's history stays honest about what was
    said, and only the routing (mention parse, coordinator, turns) is skipped.

    PASS carries a noqa: ruff reads any name/value pair spelled like this one as
    a hardcoded credential, and there is no rewording of a routing verdict that
    escapes the heuristic.
    """

    CONSUMED = "consumed"  # stop here: no mention parse, no coordinator, no turn
    PASS = "pass"  # noqa: S105  -- next handler, then normal orchestration


@runtime_checkable
class MessageHandler(Protocol):
    """A plugin's observer on messages, run before routing.

    The room pushes messages end to end -- transport, room.post, per-thread
    worker, turn -- but the only consumer at the end of that pipeline is a
    persona woken by a mention. A plugin had no delivery at all, so every plugin
    feature reacting to what a human typed had to reconstruct arrival by
    re-reading SQLite on a timer. This is the seam that pushes instead: a handler
    IS the delivery, so a near-miss can be answered synchronously, once.

    `name` is for logs (a handler that raises is named there).
    """

    name: str

    async def handle(self, msg: Message) -> Disposition: ...


log = logging.getLogger(__name__)

# Addressing grammar: "handle:" anywhere in the text is an address, full
# stop. This stays unambiguous because attribution uses a different shape
# everywhere agents and humans see it ("[sender] text" in transcripts,
# "*sender* text" in Chat), so the colon belongs exclusively to addressing.
# @ is avoided entirely (Chat reserves it for its own mention UI and personas
# are not real users). Known brittleness, accepted for simplicity: prose like
# "two things happen in the compiler: inlining and dispatch" force-triggers
# that agent; the cost is one possibly-redundant turn.
_ADDRESS_RE = re.compile(r"(?:^|(?<=\s))([a-z][a-z0-9_-]*):(?=\s|$)", re.IGNORECASE)
# Leading "perf, compiler: ..." list form; only meaningful for multi-handle
# addresses (single handles are covered by _ADDRESS_RE wherever they appear).
_ADDRESS_LIST_RE = re.compile(r"^\s*([a-z0-9_,\s-]+?)\s*:(?=\s|$)", re.IGNORECASE)

# Coordinator context: how many trailing thread messages it sees, and the
# per-message truncation. Bounds every routing call regardless of how long
# agent replies get.
_COORDINATOR_CONTEXT = 10
_COORDINATOR_MSG_CHARS = 500

# Split into a stable system prefix (intro + full roster + rules) and a variable
# user suffix (transcript + signals + the sender exclusion) so the prefix is
# byte-identical across calls -- the precondition for provider prompt caching.
# The full roster (every agent, not the per-call candidates) lives in the
# prefix; the candidate set is still hard-enforced by parse_coordinator_reply.
_COORDINATOR_SYSTEM = """\
You coordinate a chat room where human experts and AI agents discuss
{room_topic}. Decide whether any AI agent should reply to the LATEST message,
and if so which.

Agents available:
{roster}

Rules, in priority order:
- Default to NOBODY. Silence is never wrong; an unwanted interjection is.
- When humans are conversing with each other (questions directed at a named
  person, opinions, planning, banter), answer NOBODY unless an agent holds
  materially new evidence: a tool-verifiable fact, not commentary.
- Pick an agent when the latest message has an open question matching its
  expertise, a claim it can verify or refute with its tools, or its
  expertise materially advances the discussion.
- A [event] is an automated world signal (a perf change point, a CI result),
  not a person or agent. Pick an agent when it is worth an expert's take -- a
  regression worth investigating, a signal an agent can explain or act on with
  its tools -- else NOBODY, exactly as for any other message.
- Agents discussing with each other is healthy: allow a reply that disagrees,
  corrects, or adds a new angle to another agent's message, and let an
  exchange continue while it stays substantive. End it (NOBODY) once it turns
  to restatement, summaries, or mutual agreement.
- Prefer exactly ONE agent, the best fit; name a second only for genuinely
  complementary expertise.

Handles are opaque tokens (they may look like names); use them verbatim and
never alter, complete, or embellish one. The only valid handles are:
{handles}

Reply with one line: NOBODY, or one or more of the valid handles above
separated by commas, optionally followed by " -- " and a short reason.\
"""

_COORDINATOR_USER = """\
Recent conversation (oldest first; [human]/[agent]/[system]/[event] tags the
sender; long messages are truncated):

{transcript}
{signals}
The latest message is from {sender}; do not pick {sender}.\
"""

# Routing for watcher announcements (new commits, perf changepoints). These
# already passed triage as worth expert commentary, so the only question is
# WHO comments -- never whether. The picked agent's NOTHING_TO_ADD remains
# the escape hatch for true misroutes.
# Stable system prefix; the announcement text is the variable user message.
_ANNOUNCEMENT_SYSTEM = """\
A new item was posted to a chat room discussing {room_topic}. It has already
been triaged as worth expert commentary; pick WHICH ONE of these agents
should comment on it.

Agents:
{roster}

Reply with exactly one handle from this list, copied verbatim, and nothing
else. A handle is an opaque token (it may look like a name); do not alter,
complete, or embellish it.
{handles}\
"""

# Signal lines as an unbroken agent streak grows. There is no hard cutoff:
# discussions peter out because the bar keeps rising, so an exchange that
# stays genuinely substantive may run long, while polite back-and-forth ends.
_PRESSURE = (
    "This thread has had {streak} agent messages in a row with no human"
    " input, so it now needs to converge: only pick an agent with a genuinely"
    " new technical point, a correction, or new evidence, not a refinement,"
    " restatement, or further agreement."
)
_PRESSURE_FINAL = (
    "This thread has run to {streak} consecutive agent messages with no human"
    " input. Answer NOBODY unless an agent has a decisive correction or major"
    " new evidence; summaries, refinements, and agreement must end now."
)

# Signal line when trailing human messages have not engaged the agents' last
# contribution: the strongest stay-out cue a real participant would read.
_MOVED_ON = (
    "The humans have continued the conversation past the agents' last message"
    " without engaging it; interjecting again needs a much higher bar."
)


def _handle_list(agents) -> str:
    """The valid handles as an explicit closed set for the routing prompts, one
    per line. Naming the exact output vocabulary (not just the roster's bulleted
    descriptions) stops a small filter model from treating a name-like handle as
    a seed to complete -- "jace" -> "jace_of_spades" -- which then fails the
    exact-match parse and routes to nobody."""
    return "\n".join(f"- {n}" for n in agents)


def parse_coordinator_reply(text: str, known: set[str], cap: int) -> list[str]:
    """Handles from the coordinator's reply line, validated and capped.

    Anything unparseable (or NOBODY) yields [] -- routing fails closed, since
    an explicit address always still works.
    """
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    # Cut the optional reason. Normalize the unicode dashes the model often
    # substitutes for the requested " -- " first, or a valid route parses to [].
    line = re.sub(r"[‒-―]", "--", line).split("--", 1)[0].strip()
    if not line or line.upper().startswith("NOBODY"):
        return []
    out: list[str] = []
    for token in line.split(","):
        name = token.strip().strip(".,").lower()
        if name in known and name not in out:
            out.append(name)
    return out[:cap]


def humans_moved_on(messages: list[Message], known: set[str]) -> bool:
    """True when >= 2 human messages followed the agents' last contribution
    without addressing any agent: the room has moved past them."""
    trailing_humans = 0
    for m in reversed(messages):
        if m.kind == MessageKind.AGENT:
            return trailing_humans >= 2
        if m.kind == MessageKind.HUMAN:
            if parse_mentions(m.text, known):
                return False
            trailing_humans += 1
    return False  # no agent contribution yet; nothing to have moved past


def parse_mentions(text: str, known: set[str]) -> list[str]:
    """Agents addressed by "handle:" anywhere in the text, in order,
    deduplicated. A line-leading "perf, compiler: ..." list addresses several
    at once (every token must be a known handle, so "note: ..." and
    "perf and compiler: ..." never list-match -- though the latter still
    forces compiler via the single-handle rule). Unknown words with colons and
    URLs/timestamps never match.
    """
    out: list[str] = []

    def add(name: str) -> None:
        name = name.lower()
        if name in known and name not in out:
            out.append(name)

    for line in text.splitlines():
        if m := _ADDRESS_LIST_RE.match(line):
            tokens = [t.lower() for t in re.split(r"[,\s]+", m.group(1).strip()) if t]
            if len(tokens) > 1 and all(t in known for t in tokens):
                for t in tokens:
                    add(t)
        for m in _ADDRESS_RE.finditer(line):
            add(m.group(1))
    return out


# Idle worker lifetime: a thread quiet this long parks its worker; the
# dispatcher respawns one on the next message.
_IDLE_S = 1800.0


@dataclass
class _Worker:
    queue: asyncio.Queue
    task: asyncio.Task


@dataclass
class _PendingMsg:
    """A routed message awaiting its responder turns.

    remaining starts at 1 (routing in progress); _handle adds one per spawned
    responder and the worker clears the routing 1. The watermark commits a
    message only once it and every earlier message on the thread reach 0, so
    out-of-order turn completion never advances the watermark past incomplete
    work.
    """

    msg_id: int
    remaining: int


class TurnContext:
    """The narrow room handle a follow-up handler drives its turn through.

    The room owns turn EXECUTION (it has already acquired the concurrency slot
    and shown the typing indicator before the handler runs); the handler owns the
    RESPONSE. These are the only primitives it needs -- run a plain or structured
    turn, post as the responder, read the announcement and its persisted meta --
    so a plugin never reaches into orchestrator internals. Both run_* wrappers
    apply the turn timeout and the room's standard error UX, exactly as the plain
    path does.
    """

    def __init__(
        self,
        orch: Orchestrator,
        *,
        responder: str,
        thread_id: int,
        announcement: Message,
    ) -> None:
        self._orch = orch
        self.responder = responder
        self.thread_id = thread_id
        self.announcement = announcement
        self.store = orch._store

    async def run_turn(self, *, task_prompt: str | None = None) -> str | None:
        """A plain conversational turn (the room's default response shape)."""
        return await self._orch._guarded_turn(
            self.responder, self.thread_id, addressed=False, task_prompt=task_prompt
        )

    async def run_structured_turn(
        self, content: str, *, extra_system: str, label: str = "structured"
    ) -> str | None:
        """A structured (result-forcing) turn; returns raw text for the handler
        to parse. The graph/lock/cache/token plumbing is the runner's."""
        return await self._orch._guarded_structured(
            self.responder, content, extra_system=extra_system, label=label
        )

    async def post(self, body: str) -> None:
        await self._orch._room.post(
            self.thread_id, self.responder, MessageKind.AGENT, body
        )


class Orchestrator:
    def __init__(
        self,
        cfg: Config,
        room: Room,
        runner: AgentRunner,
        store: Store,
        follow_ups: dict[str, FollowUp] | None = None,
        message_handlers: list[MessageHandler] | None = None,
    ) -> None:
        self._cfg = cfg
        self._room = room
        self._runner = runner
        self._store = store
        self._tokens = TokenLog(cfg.token_db_path)
        # Plugin observers on arriving messages, run in registration order before
        # any routing. Empty for a bare room and for a plugin without the hook.
        self._message_handlers: list[MessageHandler] = message_handlers or []
        # App-registered response handlers keyed by a message's follow_up string
        # (D2 registry: an explicit dict the app populates, not entry-point magic).
        # A SYSTEM announcement whose follow_up names one dispatches its whole
        # response to that handler; every other message gets the plain turn.
        self._follow_ups: dict[str, FollowUp] = follow_ups or {}
        # Retry transient errors: this bare model is invoked outside any agent
        # graph, so ModelRetryMiddleware never covers it.
        self._filter_model = retrying(make_model(cfg.filter_model))
        self._workers: dict[int, _Worker] = {}
        self._turn_sem = asyncio.Semaphore(cfg.orchestrator.max_concurrent_turns)
        # Per-(thread, agent) lock serializes one agent's turns on a thread (the
        # invariant run_turn relies on) while letting different agents and
        # threads run in parallel.
        self._agent_locks: dict[tuple[int, str], asyncio.Lock] = {}
        # Per-thread FIFO of routed-but-incomplete messages, for the contiguous
        # watermark; and the set of in-flight turn tasks, awaited at shutdown.
        self._pending: dict[int, list[_PendingMsg]] = {}
        self._round_tasks: set[asyncio.Task] = set()

    async def run(self) -> None:
        """Recover unfinished work, then dispatch inbox messages forever."""
        self._recover()
        try:
            while True:
                msg = await self._room.inbox.get()
                self._dispatch(msg)
        finally:
            # Cancel workers AND in-flight turn tasks, then await them: an
            # unawaited task can still be unwinding (and writing the watermark
            # in _finish_one) when cli closes the checkpointer and the store.
            tasks = [w.task for w in self._workers.values()] + list(self._round_tasks)
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            # Dead workers must not linger: anything routed to a cancelled
            # worker's queue would strand. Their unfinished messages are
            # below the watermark and replay on the next run.
            self._workers.clear()
            self._round_tasks.clear()
            self._pending.clear()

    # ── dispatch / recovery ──────────────────────────────────────────────────

    def _dispatch(self, msg: Message) -> None:
        # No await between lookup and put: an idle worker that timed out either
        # sees this message in its non-empty queue, or already removed itself
        # and we spawn a fresh one.
        w = self._workers.get(msg.thread_id)
        if w is None:
            w = self._spawn(msg.thread_id)
        w.queue.put_nowait(msg)

    def _spawn(self, thread_id: int) -> _Worker:
        queue: asyncio.Queue[Message] = asyncio.Queue()
        task = asyncio.create_task(
            self._worker_loop(thread_id, queue), name=f"thread-worker:{thread_id}"
        )
        w = _Worker(queue=queue, task=task)
        self._workers[thread_id] = w
        return w

    def _recover(self) -> None:
        """Re-enqueue persisted messages above each thread's watermark.

        Synchronous on purpose (zero awaits), and the orchestrator must stay
        the first task cli creates: recovery then completes before any
        transport or watcher coroutine runs a step, so a message can never be
        both recovered here and delivered via the inbox.
        """
        for t in self._store.list_threads():
            msgs = self._store.thread_messages(t.id)
            watermark = self._store.get_orchestrated(t.id)
            if watermark is None:
                # Thread predates the watermark table: treat history as
                # orchestrated rather than replaying months of it.
                self._store.set_orchestrated(t.id, msgs[-1].id if msgs else 0)
                continue
            pending = [m for m in msgs if m.id > watermark]
            if not pending:
                continue
            log.info(
                "thread %d: recovering %d unorchestrated message(s)",
                t.id,
                len(pending),
            )
            w = self._workers.get(t.id) or self._spawn(t.id)
            for m in pending:
                w.queue.put_nowait(m)

    def _reap_locks(self, thread_id: int) -> None:
        """Drop this thread's per-agent locks instead of leaking one per
        (thread, agent) for the process lifetime -- but never a held lock. Routed
        turns keep _pending non-empty so the worker stays alive while they hold a
        lock; a timer wake runs outside _pending (deliver_wake), so a held lock
        must survive here. Deleting one would let a later routed turn setdefault a
        fresh lock and run concurrently with the wake, breaking run_turn's
        one-turn-per-(thread, agent) invariant. A held lock is reaped on a later
        idle cycle once the wake releases it."""
        for key in [
            k
            for k in self._agent_locks
            if k[0] == thread_id and not self._agent_locks[k].locked()
        ]:
            del self._agent_locks[key]

    async def _worker_loop(self, thread_id: int, queue: asyncio.Queue) -> None:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=_IDLE_S)
            except TimeoutError:
                # No await between the checks and the removal (see _dispatch for
                # the matching half of the handshake). Stay alive while rounds
                # are in flight so their watermark commit still has an owner.
                if queue.empty() and not self._pending.get(thread_id):
                    del self._workers[thread_id]
                    self._pending.pop(thread_id, None)
                    self._reap_locks(thread_id)
                    return
                continue
            # Route in id order (streak/coordinator context stays ordered) and
            # register the message; the responder turns run as detached tasks,
            # so a slow agent never blocks the next message. The watermark is
            # advanced by _finish_one once a contiguous prefix completes.
            pm = _PendingMsg(msg_id=msg.id, remaining=1)
            self._pending.setdefault(thread_id, []).append(pm)
            try:
                await self._handle(msg, pm)
            except Exception:
                log.exception("error handling message %d", msg.id)
            self._finish_one(pm, thread_id)

    # ── routing ──────────────────────────────────────────────────────────────

    async def _handle(self, msg: Message, pm: _PendingMsg) -> None:
        if msg.kind in (MessageKind.NOTICE, MessageKind.PING):
            return
        if await self._consumed(msg):
            return
        agents = self._runner.agents
        candidates = [n for n in agents if n != msg.sender]
        if not candidates:
            return

        # Streak of consecutive agent messages ending at msg (0 if msg is from
        # a human or the watcher, which always reset the discussion). There is
        # no hard cutoff: the streak feeds escalating pressure signals to the
        # coordinator, so discussions peter out rather than stop mid-sentence.
        streak = (
            self._agent_streak(msg.thread_id) if msg.kind == MessageKind.AGENT else 0
        )

        addressed = False
        if msg.kind == MessageKind.SYSTEM:
            # Announcements never parse addresses: commit messages routinely
            # contain "handle:"-shaped text (the [v8] repo prefix, trailer
            # lines), which must not force agents around the router. The picked
            # agent comments; if the message carries a follow_up handler key, the
            # response is dispatched to it (which may inject a brief, run a
            # structured digest, hand off, ...), otherwise a plain turn -- the
            # orchestrator does not know which, and never learns the domain.
            responders = await self._route_announcement(msg, candidates)
        elif msg.kind == MessageKind.EVENT:
            # A world signal (a perf change point, ...): route it through the same
            # default-silence coordinator a human message uses, so the room reacts
            # only when it is worth a voice. Not an announcement (no forced
            # commentator) and not an address (a signal names no one), so it never
            # parses handles and never forces a reply.
            responders = await self._coordinate(msg, candidates, streak)
        else:
            # Parse the address against ALL handles, then drop the sender: an
            # agent writing "perf, compiler: ..." with its own name must still
            # force compiler, not void the whole address because its own name
            # is not a candidate.
            responders = [
                n for n in parse_mentions(msg.text, set(agents)) if n != msg.sender
            ]
            # A human's direct address forces a substantive reply: the named
            # agent's NOTHING_TO_ADD escape hatch is withdrawn for this turn.
            # An agent addressing another agent does not get that override.
            addressed = bool(responders) and msg.kind == MessageKind.HUMAN
            if not responders:
                responders = await self._coordinate(msg, candidates, streak)
        if not responders:
            return
        # Spawn each responder's turn as a detached task so the worker advances
        # to the next message immediately. The count is bumped before any task
        # is created (and there is no await between here and the spawn loop), so
        # a fast task cannot drive remaining to zero before the full count lands.
        pm.remaining += len(responders)
        for n in responders:
            task = asyncio.create_task(
                self._run_responder(pm, n, msg, addressed=addressed)
            )
            self._round_tasks.add(task)
            task.add_done_callback(self._round_tasks.discard)

    async def _consumed(self, msg: Message) -> bool:
        """Run the plugin handler chain; True once one claims the message.

        Here in the worker loop rather than in room.post, because _recover
        replays persisted messages above the watermark straight into the worker
        queues -- a hook on post would silently skip every replayed message, and
        consumption would not be crash-durable. In the loop the handlers inherit
        what the orchestrator already guarantees: per-thread ordering, the
        durable watermark, and replay after a crash. Replay is also the contract
        they owe back: a handler must be idempotent, exactly like every bus
        subscriber in the suite.

        Handlers run INLINE, so a slow one delays this thread's routing (and only
        this thread's). They are meant for store reads/writes, a bus publish, and
        at most a post; anything heavier belongs on the bus.

        A handler that raises is logged and treated as PASS. A broken plugin must
        not be able to silence the room -- the failure mode to avoid is one
        exception swallowing a message nobody then answers.
        """
        for h in self._message_handlers:
            try:
                if await h.handle(msg) is Disposition.CONSUMED:
                    log.info("handler %s consumed message %d", h.name, msg.id)
                    return True
            except Exception:
                log.exception("handler %s failed on message %d", h.name, msg.id)
        return False

    async def _run_responder(
        self,
        pm: _PendingMsg,
        name: str,
        msg: Message,
        *,
        addressed: bool,
    ) -> None:
        """Run one agent's turn under its per-(thread, agent) lock.

        The lock (acquired outside _respond's _turn_sem -- a consistent order,
        so no deadlock) serializes this agent's turns on this thread, the
        invariant run_turn relies on, while leaving other agents and threads
        parallel. _respond catches its own turn errors but room.post/typing can
        still raise; catch and log those so the task never surfaces an
        unretrieved exception.

        _finish_one runs on normal completion and on a handled error (an
        errored turn will not retry this run, so it counts as done), but NOT on
        CancelledError: a turn cancelled mid-flight at shutdown must leave its
        message below the watermark so recovery replays it (at-least-once).
        """
        thread_id = msg.thread_id
        lock = self._agent_locks.setdefault((thread_id, name), asyncio.Lock())
        try:
            async with lock:
                await self._respond(name, msg, addressed=addressed)
        except asyncio.CancelledError:
            raise  # incomplete; do not commit, let the message replay
        except Exception:
            log.exception("agent %s round failed in thread %d", name, thread_id)
        self._finish_one(pm, thread_id)

    async def deliver_wake(self, thread_id: int, agent: str, note: str) -> None:
        """Fire a timer wake: a fresh turn for `agent` on `thread_id`, driven by
        `note`. `agent` is the persona's STABLE KEY (what timer_create persisted from
        the checkpoint config); translate it to the live addressable name -- which
        differs from the key when use_nicknames is on -- before routing. Runs the
        same execution bracket (semaphore, typing, timeout, post) a routed plain
        turn takes, under the same per-(thread, name) lock, so a wake never races
        a concurrent human-triggered turn for the same agent. addressed=True so
        the agent acts on the note instead of taking the NOTHING_TO_ADD escape.
        The turn resumes with the agent's own context: its checkpoint plus
        whatever the thread accrued since it last spoke, reinjected by run_turn."""
        name = self._runner.name_for_key(agent)
        if name is None:
            log.info(
                "timer: agent %s is gone; dropping wake for thread %d",
                agent,
                thread_id,
            )
            return
        lock = self._agent_locks.setdefault((thread_id, name), asyncio.Lock())
        async with lock, self._turn_sem:
            timeout = self._cfg.orchestrator.turn_timeout
            await self._room.typing(thread_id, name, True, budget=timeout)
            try:
                text = await self._guarded_turn(
                    name, thread_id, addressed=True, task_prompt=note
                )
                if text is None:
                    return
                await self._room.post(thread_id, name, MessageKind.AGENT, text)
            finally:
                await self._room.typing(thread_id, name, False)

    def _finish_one(self, pm: _PendingMsg, thread_id: int) -> None:
        """Mark one outstanding turn (or the routing phase) done; commit prefix.

        Fully synchronous: in single-threaded asyncio the decrement, the
        contiguous-prefix walk, and the watermark write are atomic only with no
        await between them. The watermark advances to the last message of the
        maximal completed contiguous prefix, so a crash never commits past an
        incomplete message (recovery replays id > watermark).
        """
        pm.remaining -= 1
        if pm.remaining > 0:
            return
        pend = self._pending.get(thread_id)
        if not pend:
            return
        last = None
        while pend and pend[0].remaining == 0:
            last = pend.pop(0).msg_id
        if not pend:
            self._pending.pop(thread_id, None)
        if last is not None:
            self._store.set_orchestrated(thread_id, last)

    def _agent_streak(self, thread_id: int) -> int:
        n = 0
        for m in reversed(self._room.thread_messages(thread_id)):
            if m.kind == MessageKind.PING:
                continue  # operational, not a conversational turn: see through it
            if m.kind != MessageKind.AGENT:
                break
            n += 1
        return n

    def _pressure(self, streak: int) -> str:
        o = self._cfg.orchestrator
        if streak < o.soft_turn_budget:
            return ""
        if streak < o.max_turns:
            return _PRESSURE.format(streak=streak)
        return _PRESSURE_FINAL.format(streak=streak)

    async def _respond(
        self,
        name: str,
        msg: Message,
        *,
        addressed: bool = False,
    ) -> None:
        # The semaphore bounds simultaneous turns across all threads; typing-on
        # comes after acquiring so the "thinking..." card appears when work
        # actually starts, not while queued. This is the execution bracket: the
        # room holds the concurrency slot and the typing indicator around the
        # response, whether it is a plain turn or an app follow-up handler. The
        # handler owns the response; the room never learns what it does.
        thread_id = msg.thread_id
        async with self._turn_sem:
            timeout = self._cfg.orchestrator.turn_timeout
            await self._room.typing(thread_id, name, True, budget=timeout)
            try:
                handler = self._follow_ups.get(msg.follow_up)
                if handler is not None:
                    ctx = TurnContext(
                        self, responder=name, thread_id=thread_id, announcement=msg
                    )
                    try:
                        await handler(ctx)
                    except Exception:
                        log.exception(
                            "follow-up %r failed for agent %s in thread %d",
                            msg.follow_up,
                            name,
                            thread_id,
                        )
                    return
                text = await self._guarded_turn(name, thread_id, addressed=addressed)
                if text is None:
                    log.info(
                        "agent %s had nothing to add in thread %d", name, thread_id
                    )
                    return
                await self._room.post(thread_id, name, MessageKind.AGENT, text)
            finally:
                # Clears the placeholder if the reply (deliver) did not consume
                # it first -- e.g. nothing-to-add, an error, a spaceless thread.
                await self._room.typing(thread_id, name, False)

    async def _guarded_turn(
        self,
        name: str,
        thread_id: int,
        *,
        addressed: bool = False,
        task_prompt: str | None = None,
    ) -> str | None:
        """A plain turn under the shared turn timeout + the room's error UX: a
        NOTICE on timeout or error, None on failure. Does not post the reply --
        the caller decides what to do with the text -- but it does post the error
        notices, since those are the room's standard behavior on any turn."""
        timeout = self._cfg.orchestrator.turn_timeout
        try:
            async with asyncio.timeout(timeout):
                return await self._runner.run_turn(
                    name, thread_id, addressed=addressed, task_prompt=task_prompt
                )
        except TimeoutError:
            log.error(
                "agent %s timed out after %.0fs in thread %d", name, timeout, thread_id
            )
            await self._room.post(
                thread_id,
                name,
                MessageKind.NOTICE,
                f"({name} gave up after {timeout:.0f}s)",
            )
            return None
        except Exception:
            log.exception("agent %s failed in thread %d", name, thread_id)
            await self._room.post(
                thread_id, name, MessageKind.NOTICE, f"({name} errored; see logs)"
            )
            return None

    async def _guarded_structured(
        self, name: str, content: str, *, extra_system: str, label: str
    ) -> str | None:
        """A structured turn under the shared timeout; returns raw text or None.
        Unlike _guarded_turn it posts no NOTICE (a structured follow-up's failure
        is the handler's to surface or swallow), only logs -- matching the old
        digest path, which logged and returned on error."""
        timeout = self._cfg.orchestrator.turn_timeout
        try:
            async with asyncio.timeout(timeout):
                return await self._runner.run_structured_turn(
                    name, content, extra_system=extra_system, label=label
                )
        except Exception:
            log.exception("%s turn for agent %s failed", label, name)
            return None

    # ── coordinator ──────────────────────────────────────────────────────────

    async def _coordinate(
        self, msg: Message, candidates: list[str], streak: int
    ) -> list[str]:
        """One routing call: should any agent reply to msg, and who?

        A single room-level judgment that can see all agent descriptions, who
        is human vs agent, and conversational signals (convergence pressure,
        humans having moved on). Fails closed: any error or unparseable reply
        routes to nobody.
        """
        try:
            prompt = self._coordinator_prompt(msg, candidates, streak)
            reply = await self._filter_model.ainvoke(prompt)
        except Exception:
            log.exception("coordinator failed for msg %d", msg.id)
            return []
        if usage := getattr(reply, "usage_metadata", None):
            tin, tout, cached = usage_counts(usage)
            self._tokens.add(
                msg.thread_id,
                "coordinator",
                "coordinator",
                tin,
                tout,
                cached,
                self._cfg.filter_model,
            )
            log.info(
                "coordinator msg %d: %d in (%d cached) / %d out tokens",
                msg.id,
                tin,
                cached,
                tout,
            )
        verdict = str(reply.text).strip()
        log.info(
            "coordinator msg %d: %s",
            msg.id,
            verdict.splitlines()[0] if verdict else "(empty)",
        )
        return parse_coordinator_reply(
            verdict, set(candidates), self._cfg.orchestrator.max_responders
        )

    async def _route_announcement(
        self, msg: Message, candidates: list[str]
    ) -> list[str]:
        """Pick the one agent who comments on a watcher announcement.

        Unlike _coordinate there is no NOBODY option: the announcement only
        exists because triage already judged it worth commentary, so asking
        "should anyone speak" again is what made agents skip commit posts.
        """
        agents = self._runner.agents
        try:
            roster = "\n".join(f"- {n}: {agents[n].description}" for n in agents)
            announcement = msg.text[: _COORDINATOR_MSG_CHARS * 3]
            reply = await self._filter_model.ainvoke(
                [
                    {
                        "role": "system",
                        "content": _ANNOUNCEMENT_SYSTEM.format(
                            room_topic=self._cfg.room_topic,
                            roster=roster,
                            handles=_handle_list(agents),
                        ),
                    },
                    {"role": "user", "content": announcement},
                ]
            )
        except Exception:
            log.exception("announcement routing failed for msg %d", msg.id)
            return []
        if usage := getattr(reply, "usage_metadata", None):
            tin, tout, cached = usage_counts(usage)
            self._tokens.add(
                msg.thread_id,
                "coordinator",
                "coordinator",
                tin,
                tout,
                cached,
                self._cfg.filter_model,
            )
        picked = parse_coordinator_reply(str(reply.text), set(candidates), 1)
        if not picked:
            log.warning(
                "announcement router picked nobody for msg %d (%r)",
                msg.id,
                str(reply.text)[:80],
            )
        else:
            log.info("announcement msg %d routed to %s", msg.id, picked[0])
        return picked

    def _coordinator_prompt(
        self, msg: Message, candidates: list[str], streak: int
    ) -> list[dict]:
        agents = self._runner.agents
        # Full roster in the stable system prefix; the candidate set is enforced
        # downstream by parse_coordinator_reply, and the sender exclusion moves
        # into the variable user message below.
        roster = "\n".join(f"- {n}: {agents[n].description}" for n in agents)
        messages = [
            m
            for m in self._room.thread_messages(msg.thread_id)
            if m.kind not in (MessageKind.NOTICE, MessageKind.PING)
        ]
        lines = []
        for m in messages[-_COORDINATOR_CONTEXT:]:
            text = m.text
            if len(text) > _COORDINATOR_MSG_CHARS:
                text = text[:_COORDINATOR_MSG_CHARS] + " [...]"
            # Every kind left after the notice/ping filter tags as itself; no
            # allowlist to hand-extend when the next kind arrives.
            lines.append(f"[{m.kind}] {m.sender}: {text}")
        signals = []
        if pressure := self._pressure(streak):
            signals.append(pressure)
        if humans_moved_on(messages, set(agents)):
            signals.append(_MOVED_ON)
        signals_text = (
            "\nSignals:\n" + "\n".join(f"- {s}" for s in signals) + "\n"
            if signals
            else ""
        )
        return [
            {
                "role": "system",
                "content": _COORDINATOR_SYSTEM.format(
                    room_topic=self._cfg.room_topic,
                    roster=roster,
                    handles=_handle_list(agents),
                ),
            },
            {
                "role": "user",
                "content": _COORDINATOR_USER.format(
                    transcript="\n".join(lines),
                    signals=signals_text,
                    sender=msg.sender,
                ),
            },
        ]

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""SQLite persistence for threads, messages, and watcher state.

A single database file holds the chat history; LangGraph checkpoints live in
a separate file next to it (managed by the runner). Statements are short and
infrequent, so synchronous sqlite3 from the event loop is acceptable.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

# The findings-badge buckets counted on a chat_headlines row: the severity
# ladder, 'unknown' for an ungraded severity, and 'info' for a finding whose
# repro attempt failed (see bump_headline_finding). The store is bucket-name
# agnostic beyond this set; grading/rendering stays with the plugin.
BADGE_BUCKETS = ("blocker", "high", "medium", "low", "unknown", "info")

# Retention: `airc-prune` (airc_room.prune) ages out old threads -- it redacts
# message text/sender for every kind but SYSTEM, drops the thread's persona
# checkpoints, and vacuums both files. It deliberately keeps the dedup keys
# (commit_threads, chat_threads, chat_seen_messages, handover_jobs,
# delivered_results) so a late
# event cannot re-announce a thread whose discussion was just purged. Run it
# from its systemd timer; the room must be stopped for the vacuum.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created REAL NOT NULL,
    -- Client-assigned chat thread key, set once at create time. Derived
    -- from a uuid, NOT the row id: a wiped/rebuilt DB restarts the id sequence,
    -- and a positional key would collide with a thread still present server-side
    -- on the transport, threading new posts into a stale, unrelated thread.
    chat_key TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id INTEGER NOT NULL REFERENCES threads(id),
    sender TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    ts REAL NOT NULL,
    -- Optional app-defined follow-up handler key for a SYSTEM announcement (e.g.
    -- "commit"): the orchestrator dispatches the response to the registered
    -- handler of this name and is otherwise domain-blind. "" = plain forced turn.
    -- Persisted so a restart's _recover replays the marker with the message.
    follow_up TEXT NOT NULL DEFAULT '',
    -- Stable transport identity, separate from the human-readable sender.
    sender_id TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, id);
CREATE TABLE IF NOT EXISTS agent_seen (
    thread_id INTEGER NOT NULL,
    agent TEXT NOT NULL,
    last_msg_id INTEGER NOT NULL,
    PRIMARY KEY (thread_id, agent)
);
-- Per-thread floor on every persona's seen offset, set by the retention sweep to
-- the thread's tip. A scrubbed thread's messages are redacted to empty text, so
-- injecting them would feed a persona a wall of blank transcript lines; the floor
-- makes get_agent_seen report at least this id and the turn sees only genuinely
-- new messages. A floor rather than an UPDATE of agent_seen because a persona
-- that never took a turn on the thread has NO row -- it would read 0 and replay
-- the whole redacted history, which is exactly the failure this prevents. A
-- missing row is floor 0 (never scrubbed).
CREATE TABLE IF NOT EXISTS thread_seen_floor (
    thread_id INTEGER PRIMARY KEY,
    last_msg_id INTEGER NOT NULL
);
-- Per-thread context generation. The persona checkpoint id folds this in, so
-- bumping it starts every persona on the thread from a fresh (empty) checkpoint
-- without touching langgraph's store -- the truncation half of memory compaction.
-- A missing row is generation 0 (an un-compacted thread).
CREATE TABLE IF NOT EXISTS context_generation (
    thread_id INTEGER PRIMARY KEY,
    generation INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_threads (
    space TEXT NOT NULL,
    chat_thread TEXT NOT NULL,
    thread_id INTEGER NOT NULL,
    PRIMARY KEY (space, chat_thread)
);
CREATE TABLE IF NOT EXISTS chat_headlines (
    thread_id INTEGER PRIMARY KEY,
    message_name TEXT NOT NULL,
    base_text TEXT NOT NULL,
    tag TEXT NOT NULL DEFAULT '',
    badge TEXT NOT NULL DEFAULT '',
    perf_regress INTEGER NOT NULL DEFAULT 0,
    perf_improve INTEGER NOT NULL DEFAULT 0,
    -- Findings-badge counters, one per bucket: the severity ladder, 'unknown'
    -- for an ungraded severity, and 'info' for a finding whose repro attempt
    -- failed. The badge accumulates as findings surface (a review-time post,
    -- then one repro result at a time), so it is counters, not a one-shot
    -- string like `badge` (the legacy pre-counter form, kept as a fallback).
    badge_blocker INTEGER NOT NULL DEFAULT 0,
    badge_high INTEGER NOT NULL DEFAULT 0,
    badge_medium INTEGER NOT NULL DEFAULT 0,
    badge_low INTEGER NOT NULL DEFAULT 0,
    badge_unknown INTEGER NOT NULL DEFAULT 0,
    badge_info INTEGER NOT NULL DEFAULT 0,
    -- The last composed+edited headline text, so a recompose that changes
    -- nothing issues no Chat edit. Seeded with base_text at record time.
    composed TEXT NOT NULL DEFAULT ''
);
-- The findings already counted toward a thread's badge, keyed by the finding's
-- stable id, so at-least-once delivery (claim retry, topic replay) can never
-- double-count. The bucket is kept so a failed repro ('info') can later be
-- upgraded to its severity when a re-run verifies.
CREATE TABLE IF NOT EXISTS chat_headline_findings (
    thread_id INTEGER NOT NULL,
    finding_id TEXT NOT NULL,
    bucket TEXT NOT NULL,
    PRIMARY KEY (thread_id, finding_id)
);
CREATE TABLE IF NOT EXISTS source_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    ts REAL NOT NULL,
    note TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_notes ON source_notes(source, id);
CREATE TABLE IF NOT EXISTS chat_seen_messages (
    name TEXT PRIMARY KEY,
    ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS delivered_results (
    job_id TEXT PRIMARY KEY,
    ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS handover_jobs (
    job_id TEXT PRIMARY KEY,
    ts REAL NOT NULL
);
-- Durable queue of bugs to file for verified repros, so a tracker outage (the
-- operator's credentials lapse ~daily, and over a weekend) neither blocks the
-- chat post nor loses the bug. Keyed by the finding's stable id (dedup: a
-- re-published result cannot double-file). Stores only result-derived content;
-- component/assignee/priority are re-derived from current config on each retry,
-- so a config fix takes effect without rewriting rows. Deleted on a successful
-- file.
CREATE TABLE IF NOT EXISTS pending_bugs (
    finding_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    repo TEXT NOT NULL,
    commit_hash TEXT NOT NULL,
    author_email TEXT NOT NULL,
    isolates INTEGER,            -- 1 regression / 0 pre-existing / NULL unknown
    thread_id INTEGER,           -- where to post the "filed" follow-up
    attempts INTEGER NOT NULL,
    created_ts REAL NOT NULL,
    last_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS announcement_meta (
    thread_id INTEGER PRIMARY KEY,
    repo_name TEXT NOT NULL,
    repo_path TEXT NOT NULL,
    hash TEXT NOT NULL
);
-- Reverse index: the thread a commit hash owns, so commentary and findings for
-- one commit converge on a single thread regardless of which arrives first.
CREATE TABLE IF NOT EXISTS commit_threads (
    hash TEXT PRIMARY KEY,
    thread_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_pending (
    thread_id INTEGER NOT NULL,
    agent TEXT NOT NULL,
    space TEXT NOT NULL,
    message TEXT NOT NULL,
    ts REAL NOT NULL,
    PRIMARY KEY (thread_id, agent)
);
CREATE TABLE IF NOT EXISTS orchestrated (
    thread_id INTEGER PRIMARY KEY,
    last_msg_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_users (
    user TEXT PRIMARY KEY,
    display TEXT NOT NULL
);
-- Agent-set timers ("wake me in X min"). seq is the scheduler's stable id, not
-- autoincremented here: the scheduler owns id assignment (it seeds its counter
-- above the max on restart), so persisting the id it already handed out keeps
-- timer_cancel's id stable across a restart. Rows live only while pending -- the
-- scheduler deletes on cancel and on fire -- so this table repopulates the heap
-- at startup and is otherwise empty.
CREATE TABLE IF NOT EXISTS timers (
    seq INTEGER PRIMARY KEY,
    thread_id INTEGER NOT NULL,
    agent TEXT NOT NULL,
    fire_at REAL NOT NULL,
    note TEXT NOT NULL
);
-- Thread-scoped durable state a PLUGIN owns, with core blind to its shape. The
-- namespace is the plugin's own (e.g. "icu_task_proposals"), key its own id, and
-- json its own payload -- so a plugin gets mutable, thread-queryable state in
-- the same connection and ordering domain as `messages` without a domain table
-- in core's schema. (pending_bugs/handover_jobs above predate the core/plugin
-- split and are the shape this exists to stop adding to.) One _DROP_TABLES entry
-- in airc-prune then covers every plugin's state, present and future, so a
-- plugin adding state needs no core change at all. Not an unconditional
-- retention guarantee: prune only selects threads still holding unredacted
-- content, so a row written into an already-swept thread outlives later sweeps.
-- Tiny and bounded, but do not treat this as self-cleaning for a record a
-- plugin keeps writing long after its thread went quiet.
CREATE TABLE IF NOT EXISTS plugin_state (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    thread_id INTEGER NOT NULL,
    json TEXT NOT NULL,
    ts REAL NOT NULL,
    PRIMARY KEY (namespace, key)
);
CREATE INDEX IF NOT EXISTS idx_plugin_state_thread
    ON plugin_state(namespace, thread_id);
"""


@dataclass(frozen=True)
class Thread:
    id: int
    title: str
    created: float


class MessageKind(StrEnum):
    """How a room message is treated. StrEnum so a member is its own wire form:
    it stores as that TEXT in SQLite and compares equal to the bare string, so
    old rows and any un-migrated comparison keep working.

    - HUMAN  -- a person spoke; routed through the default-silence coordinator.
    - AGENT  -- a persona spoke; feeds the converge-pressure streak.
    - SYSTEM -- a watcher announcement; forces exactly one commentator.
    - EVENT  -- an automated world signal (a perf change point, ...): routed like
      HUMAN through the coordinator, but neither a person nor a persona.
    - NOTICE -- operational aside; never routed to a persona.
    - PING   -- a mention-only nudge; never routed, seen through by the streak.
    """

    HUMAN = "human"
    AGENT = "agent"
    SYSTEM = "system"
    EVENT = "event"
    NOTICE = "notice"
    PING = "ping"


@dataclass(frozen=True)
class Message:
    id: int
    thread_id: int
    sender: str
    kind: MessageKind
    text: str
    ts: float
    # App-defined follow-up handler key (see the messages.follow_up schema note).
    follow_up: str = ""
    # Stable transport identity; sender remains the human-readable display name.
    # Optional/defaulted so callers constructing Message directly remain valid.
    sender_id: str = ""

    def __post_init__(self) -> None:
        # One coercion point for every construction path (add_message, Message(*row)
        # from SQLite, tests passing a bare string): normalize kind to the typed
        # member. Idempotent -- MessageKind(MessageKind.X) is MessageKind.X.
        if not isinstance(self.kind, MessageKind):
            object.__setattr__(self, "kind", MessageKind(self.kind))


class Store:
    def __init__(self, path: Path | str) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        # check_same_thread=False allows an accidental off-loop access (the
        # codebase uses asyncio.to_thread elsewhere); WAL + a busy timeout make
        # that survivable (a brief wait, not an instant "database is locked").
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.executescript(_SCHEMA)
        self._migrate()
        self._db.commit()

    def _migrate(self) -> None:
        # CREATE TABLE IF NOT EXISTS leaves pre-existing tables alone, so new
        # columns need an explicit additive migration. Keep these idempotent.
        tcols = {r[1] for r in self._db.execute("PRAGMA table_info(threads)")}
        if "chat_key" not in tcols:
            self._db.execute(
                "ALTER TABLE threads ADD COLUMN chat_key TEXT NOT NULL DEFAULT ''"
            )
            # Backfill the legacy positional key so existing threads keep the
            # identity the Chat space already knows them by (only new threads get
            # a uuid key). ADD COLUMN cannot carry a per-row default, hence the
            # separate UPDATE.
            self._db.execute(
                "UPDATE threads SET chat_key = 'airc-' || id WHERE chat_key = ''"
            )
        # chat_headlines gained composable tag/badge parts (was a single one-shot
        # annotation); add the columns to pre-existing tables.
        hcols = {r[1] for r in self._db.execute("PRAGMA table_info(chat_headlines)")}
        for col in ("tag", "badge"):
            if col not in hcols:
                self._db.execute(
                    f"ALTER TABLE chat_headlines ADD COLUMN {col} TEXT NOT NULL"
                    " DEFAULT ''"
                )
        for col in ("perf_regress", "perf_improve"):
            if col not in hcols:
                self._db.execute(
                    f"ALTER TABLE chat_headlines ADD COLUMN {col} INTEGER NOT NULL"
                    " DEFAULT 0"
                )
        # chat_headlines gained accumulating findings-badge counters (was a
        # one-shot lifted string in `badge`, kept as the legacy fallback) and a
        # `composed` change-guard for the Chat edit; add the columns to
        # pre-existing tables.
        for col in BADGE_BUCKETS:
            if f"badge_{col}" not in hcols:
                self._db.execute(
                    f"ALTER TABLE chat_headlines ADD COLUMN badge_{col} INTEGER"
                    " NOT NULL DEFAULT 0"
                )
        if "composed" not in hcols:
            self._db.execute(
                "ALTER TABLE chat_headlines ADD COLUMN composed TEXT NOT NULL"
                " DEFAULT ''"
            )
            # Seed the guard with what an un-annotated row displays (its base
            # text) so the first post-upgrade recompose does not issue a
            # redundant edit. Annotated legacy rows recompose to the parts they
            # already show, so their one extra edit carries identical text.
            self._db.execute(
                "UPDATE chat_headlines SET composed = base_text"
                " WHERE tag = '' AND badge = '' AND perf_regress = 0"
                " AND perf_improve = 0"
            )
        # messages gained follow_up (the announcement's app-handler key); add it
        # to pre-existing tables. Default '' means every persisted message keeps
        # the plain-turn behavior it had before, so no backfill is needed.
        mcols = {r[1] for r in self._db.execute("PRAGMA table_info(messages)")}
        if "follow_up" not in mcols:
            self._db.execute(
                "ALTER TABLE messages ADD COLUMN follow_up TEXT NOT NULL DEFAULT ''"
            )
        if "sender_id" not in mcols:
            self._db.execute(
                "ALTER TABLE messages ADD COLUMN sender_id TEXT NOT NULL DEFAULT ''"
            )

    def close(self) -> None:
        self._db.close()

    # ── threads ──────────────────────────────────────────────────────────────

    def create_thread(self, title: str) -> Thread:
        now = time.time()
        # uuid key, not airc-<id>: see the threads.chat_key schema note.
        cur = self._db.execute(
            "INSERT INTO threads (title, created, chat_key) VALUES (?, ?, ?)",
            (title, now, f"airc-{uuid4().hex}"),
        )
        # Seed the orchestration watermark in the same transaction, so every
        # post-upgrade thread has a row; threads without one are pre-upgrade
        # history that recovery initializes without replaying.
        self._db.execute(
            "INSERT INTO orchestrated (thread_id, last_msg_id) VALUES (?, 0)",
            (cur.lastrowid,),
        )
        self._db.commit()
        return Thread(id=cur.lastrowid, title=title, created=now)

    def get_thread(self, thread_id: int) -> Thread | None:
        row = self._db.execute(
            "SELECT id, title, created FROM threads WHERE id = ?", (thread_id,)
        ).fetchone()
        return Thread(*row) if row else None

    def chat_key(self, thread_id: int) -> str:
        """The thread's client-assigned chat thread key.

        Set once at create time (uuid-based) and immutable, so an outbound post
        that must start or pin a thread before its server name is known uses a
        key that is unique across DB rebuilds. Raises if the thread is unknown:
        every thread has a key, so a miss is a bug, not a fallback case.
        """
        row = self._db.execute(
            "SELECT chat_key FROM threads WHERE id = ?", (thread_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no thread {thread_id}")
        return row[0]

    def list_threads(self) -> list[Thread]:
        rows = self._db.execute(
            "SELECT id, title, created FROM threads ORDER BY id"
        ).fetchall()
        return [Thread(*r) for r in rows]

    # ── messages ─────────────────────────────────────────────────────────────

    def add_message(
        self,
        thread_id: int,
        sender: str,
        kind: MessageKind,
        text: str,
        follow_up: str = "",
        sender_id: str = "",
    ) -> Message:
        now = time.time()
        cur = self._db.execute(
            "INSERT INTO messages"
            " (thread_id, sender, kind, text, ts, follow_up, sender_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (thread_id, sender, kind, text, now, follow_up, sender_id),
        )
        self._db.commit()
        return Message(
            id=cur.lastrowid,
            thread_id=thread_id,
            sender=sender,
            kind=kind,
            text=text,
            ts=now,
            follow_up=follow_up,
            sender_id=sender_id,
        )

    def thread_messages(self, thread_id: int) -> list[Message]:
        rows = self._db.execute(
            "SELECT id, thread_id, sender, kind, text, ts, follow_up, sender_id"
            " FROM messages"
            " WHERE thread_id = ? ORDER BY id",
            (thread_id,),
        ).fetchall()
        return [Message(*r) for r in rows]

    # ── per-agent history offsets ────────────────────────────────────────────
    #
    # Each agent's own LangGraph thread already contains everything up to and
    # including last_msg_id; only later room messages are injected next turn.

    def get_agent_seen(self, thread_id: int, agent: str) -> int:
        """This agent's seen offset, never below the thread's scrub floor.

        The floor is what makes a retention sweep safe: the sweep redacts old
        messages to empty text, and a persona whose offset predates that (or which
        has no row at all, reading 0) would otherwise be handed the whole scrubbed
        history as blank transcript lines. MAX of the two covers both cases in one
        read, so no caller has to know the floor exists.
        """
        row = self._db.execute(
            "SELECT MAX(COALESCE(a.last_msg_id, 0), COALESCE(f.last_msg_id, 0))"
            " FROM (SELECT 1) AS one"
            " LEFT JOIN agent_seen a ON a.thread_id = ? AND a.agent = ?"
            " LEFT JOIN thread_seen_floor f ON f.thread_id = ?",
            (thread_id, agent, thread_id),
        ).fetchone()
        return row[0] if row else 0

    def set_thread_seen_floor(self, thread_id: int, last_msg_id: int) -> None:
        """Raise the thread's floor on every persona's seen offset, pinning it to
        the thread tip. Monotonic: a later write can only move it forward, so a
        stale value can never re-expose scrubbed history.

        The retention sweep is the real writer and does this same upsert inline,
        because it must land in the same transaction as the redaction -- a
        committed redaction with no floor is precisely the failure the floor
        exists to prevent. This method is the store-level entry point for any
        other caller (and what the tests drive the semantics through).
        """
        self._db.execute(
            "INSERT INTO thread_seen_floor (thread_id, last_msg_id) VALUES (?, ?)"
            " ON CONFLICT(thread_id) DO UPDATE SET"
            " last_msg_id = MAX(last_msg_id, excluded.last_msg_id)",
            (thread_id, last_msg_id),
        )
        self._db.commit()

    def set_agent_seen(self, thread_id: int, agent: str, last_msg_id: int) -> None:
        self._db.execute(
            "INSERT INTO agent_seen (thread_id, agent, last_msg_id)"
            " VALUES (?, ?, ?) ON CONFLICT(thread_id, agent)"
            " DO UPDATE SET last_msg_id = excluded.last_msg_id",
            (thread_id, agent, last_msg_id),
        )
        self._db.commit()

    # ── orchestration watermarks ─────────────────────────────────────────────
    #
    # Highest message id whose round the orchestrator completed (or attempted),
    # per thread. Startup recovery re-enqueues persisted messages above the
    # watermark, so queued routing work survives a crash (at-least-once).

    def get_orchestrated(self, thread_id: int) -> int | None:
        """Watermark for the thread, or None if no row (pre-upgrade history)."""
        row = self._db.execute(
            "SELECT last_msg_id FROM orchestrated WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        return row[0] if row else None

    def set_orchestrated(self, thread_id: int, last_msg_id: int) -> None:
        # Monotonic: concurrent posters can enqueue close together; never let a
        # smaller id overwrite a larger one.
        self._db.execute(
            "INSERT INTO orchestrated (thread_id, last_msg_id) VALUES (?, ?)"
            " ON CONFLICT(thread_id) DO UPDATE SET"
            " last_msg_id = MAX(last_msg_id, excluded.last_msg_id)",
            (thread_id, last_msg_id),
        )
        self._db.commit()

    # ── context generation (memory compaction / truncation) ──────────────────
    #
    # The per-persona chat checkpoint id includes this generation, so bumping it
    # is a clean, race-free truncation: an in-flight turn finishes into the old
    # generation, the next turn reads a fresh checkpoint. A memory-compaction
    # service bumps it after summarizing the thread into durable memory.

    def context_generation(self, thread_id: int) -> int:
        """The thread's current context generation (0 if never compacted)."""
        row = self._db.execute(
            "SELECT generation FROM context_generation WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        return row[0] if row else 0

    def bump_context_generation(self, thread_id: int) -> int:
        """Advance the thread's context generation and return the new value, so
        the next turn for each persona starts from an empty checkpoint."""
        self._db.execute(
            "INSERT INTO context_generation (thread_id, generation) VALUES (?, 1)"
            " ON CONFLICT(thread_id) DO UPDATE SET generation = generation + 1",
            (thread_id,),
        )
        self._db.commit()
        return self.context_generation(thread_id)

    # ── transport thread mapping ─────────────────────────────────────────────
    #
    # Maps a transport (container, server thread) pair to an airc thread id, both
    # directions: filled when we post (from the create response) and when an
    # inbound human message arrives in an as-yet-unseen server thread. Generic
    # across transports -- the gchat plugin uses (space, Chat thread), the Matrix
    # transport uses (room id, thread root event id).

    def chat_thread_id(self, space: str, chat_thread: str) -> int | None:
        row = self._db.execute(
            "SELECT thread_id FROM chat_threads WHERE space = ? AND chat_thread = ?",
            (space, chat_thread),
        ).fetchone()
        return row[0] if row else None

    def link_chat_thread(self, space: str, chat_thread: str, thread_id: int) -> None:
        self._db.execute(
            "INSERT INTO chat_threads (space, chat_thread, thread_id)"
            " VALUES (?, ?, ?) ON CONFLICT(space, chat_thread)"
            " DO UPDATE SET thread_id = excluded.thread_id",
            (space, chat_thread, thread_id),
        )
        self._db.commit()

    # ── commit-announcement headlines ────────────────────────────────────────
    #
    # The Chat message that starts a commit thread (the only part the main window
    # shows). Recorded when posted so the commentator's tag, the review findings
    # badge, and the perf direction counts can later be composed into it. base_text
    # is the message as first rendered; tag/badge are first-writer-wins parts, the
    # perf_* counters accumulate.

    _HEADLINE_COLS = "message_name, base_text, tag, badge, perf_regress, perf_improve"

    def record_headline(
        self, thread_id: int, message_name: str, base_text: str
    ) -> None:
        # composed is seeded with base_text: a fresh root displays exactly
        # that, so a recompose with no parts yet is a no-op. A re-announce
        # resets every annotation, including the findings-badge counters and
        # their dedup rows (the badge rebuilds from scratch on the new root).
        self._db.execute(
            "INSERT INTO chat_headlines (thread_id, message_name, base_text,"
            " tag, badge, composed) VALUES (?, ?, ?, '', '', ?)"
            " ON CONFLICT(thread_id) DO UPDATE"
            " SET message_name = excluded.message_name,"
            " base_text = excluded.base_text, tag = '', badge = '',"
            " perf_regress = 0, perf_improve = 0,"
            " badge_blocker = 0, badge_high = 0, badge_medium = 0,"
            " badge_low = 0, badge_unknown = 0, badge_info = 0,"
            " composed = excluded.base_text",
            (thread_id, message_name, base_text, base_text),
        )
        self._db.execute(
            "DELETE FROM chat_headline_findings WHERE thread_id = ?", (thread_id,)
        )
        self._db.commit()

    def set_headline_part(
        self, thread_id: int, part: str, value: str
    ) -> tuple[str, str, str, str, int, int] | None:
        """Set one composable headline part (tag or badge) the first time, returning
        the headline row (message_name, base_text, tag, badge, perf_regress,
        perf_improve) so the caller re-renders and edits the Chat message, or None
        when there is no headline or the part is already set.

        First-writer-wins per part: the `= ''` guard keeps the original "only the
        first qualifying reply edits it" contract for the tag, sets the findings
        badge exactly once, and makes a double-delivery (at-least-once replay) a
        no-op. record_headline clears both parts, so a re-announce starts fresh.
        """
        if part not in ("tag", "badge"):
            raise ValueError(f"unknown headline part: {part}")
        row = self._db.execute(
            f"UPDATE chat_headlines SET {part} = ?"
            f" WHERE thread_id = ? AND {part} = ''"
            f" RETURNING {self._HEADLINE_COLS}",
            (value, thread_id),
        ).fetchone()
        self._db.commit()
        return tuple(row) if row else None

    def bump_perf(
        self, thread_id: int, direction: str
    ) -> tuple[str, str, str, str, int, int] | None:
        """Increment the thread's perf regression/improvement counter and return the
        headline row, or None when there is no headline. Unlike tag/badge this
        accumulates -- a CL moves many line items -- so the marker recomputes from
        the running counts rather than locking on the first."""
        col = "perf_improve" if direction == "improvement" else "perf_regress"
        row = self._db.execute(
            f"UPDATE chat_headlines SET {col} = {col} + 1"
            f" WHERE thread_id = ? RETURNING {self._HEADLINE_COLS}",
            (thread_id,),
        ).fetchone()
        self._db.commit()
        return tuple(row) if row else None

    # ── findings badge (accumulating) ────────────────────────────────────────

    def bump_headline_finding(
        self, thread_id: int, finding_id: str, bucket: str
    ) -> bool:
        """Count one finding toward the thread's badge, exactly once.

        Dedup is by (thread_id, finding_id) with INSERT OR IGNORE, so an
        at-least-once redelivery of the same finding (claim retry, topic
        replay) re-attempts the bump but cannot double-count. `bucket` is one
        of BADGE_BUCKETS: a severity grade, or 'info' for a finding whose repro
        attempt did not produce a verified repro. One bucket move is allowed:
        'info' upgrades to a severity when a failed repro is later re-run and
        verifies (the finding was counted on the failure, so the count moves
        rather than double-counting); a severity never moves, so a later
        failure cannot downgrade a verified repro.

        Returns True when a counter actually changed. Advisory only: the chat
        transport recomposes the headline from the counters and suppresses the
        edit by comparing against the stored `composed` text, so it does not
        need this. Kept for a caller that wants the signal without a reread.
        """
        if bucket not in BADGE_BUCKETS:
            raise ValueError(f"unknown badge bucket: {bucket}")
        if not self._db.execute(
            "SELECT 1 FROM chat_headlines WHERE thread_id = ?", (thread_id,)
        ).fetchone():
            # No headline recorded (the thread has no root announcement): there
            # is nothing to annotate, so the finding is not counted -- and no
            # dedup row is left behind that would suppress a later count.
            return False
        cur = self._db.execute(
            "INSERT OR IGNORE INTO chat_headline_findings (thread_id, finding_id,"
            " bucket) VALUES (?, ?, ?)",
            (thread_id, finding_id, bucket),
        )
        if cur.rowcount:
            self._db.execute(
                f"UPDATE chat_headlines SET badge_{bucket} = badge_{bucket} + 1"
                " WHERE thread_id = ?",
                (thread_id,),
            )
            self._db.commit()
            return True
        row = self._db.execute(
            "SELECT bucket FROM chat_headline_findings"
            " WHERE thread_id = ? AND finding_id = ?",
            (thread_id, finding_id),
        ).fetchone()
        if row and row[0] == "info" and bucket != "info":
            self._db.execute(
                "UPDATE chat_headline_findings SET bucket = ?"
                " WHERE thread_id = ? AND finding_id = ?",
                (bucket, thread_id, finding_id),
            )
            self._db.execute(
                f"UPDATE chat_headlines SET badge_info = badge_info - 1,"
                f" badge_{bucket} = badge_{bucket} + 1 WHERE thread_id = ?",
                (thread_id,),
            )
            self._db.commit()
            return True
        return False

    def headline_for_compose(
        self, thread_id: int
    ) -> tuple[str, str, str, str, int, int, dict, str] | None:
        """The wide headline row for a recompose: (message_name, base_text, tag,
        legacy_badge, perf_regress, perf_improve, badge_counts, composed), or
        None when the thread has no recorded headline. badge_counts maps every
        BADGE_BUCKETS bucket to its count; legacy_badge is the pre-counters
        one-shot string, kept so rows annotated before the counters existed
        still render a badge until their first counter bump."""
        row = self._db.execute(
            "SELECT message_name, base_text, tag, badge, perf_regress,"
            " perf_improve, badge_blocker, badge_high, badge_medium, badge_low,"
            " badge_unknown, badge_info, composed FROM chat_headlines"
            " WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        name, base, tag, badge, regress, improve, *counts, composed = row
        return (
            name,
            base,
            tag,
            badge,
            regress,
            improve,
            dict(zip(BADGE_BUCKETS, counts, strict=False)),
            composed,
        )

    def mark_headline_composed(self, thread_id: int, text: str) -> None:
        """Record the composed text just edited into the Chat message, so the
        next recompose can tell a real change from a no-op (a re-delivered
        message, an unrelated post matching the sender filter)."""
        self._db.execute(
            "UPDATE chat_headlines SET composed = ? WHERE thread_id = ?",
            (text, thread_id),
        )
        self._db.commit()

    # ── chat user display names ──────────────────────────────────────────────
    #
    # Space-subscription events carry only the user resource id; DM/bridge
    # events carry the display name. Learn the mapping whenever both are seen
    # so transcripts show "Ada Lovelace" instead of "users/1093...".

    def set_chat_user(self, user: str, display: str) -> None:
        self._db.execute(
            "INSERT INTO chat_users (user, display) VALUES (?, ?)"
            " ON CONFLICT(user) DO UPDATE SET display = excluded.display",
            (user, display),
        )
        self._db.commit()

    def chat_user(self, user: str) -> str | None:
        row = self._db.execute(
            "SELECT display FROM chat_users WHERE user = ?", (user,)
        ).fetchone()
        return row[0] if row else None

    def chat_message_seen(self, name: str) -> bool:
        """Whether this Chat message was already processed (read-only check)."""
        row = self._db.execute(
            "SELECT 1 FROM chat_seen_messages WHERE name = ?", (name,)
        ).fetchone()
        return row is not None

    def mark_chat_message(self, name: str) -> bool:
        """Record a Chat message by resource name; return True if newly seen.

        Dedups a message delivered by more than one path: an @mention in a
        watched space arrives both via the add-on bridge and via the user-auth
        space subscription. Returns False on a duplicate so the caller skips it.
        """
        cur = self._db.execute(
            "INSERT OR IGNORE INTO chat_seen_messages (name, ts) VALUES (?, ?)",
            (name, time.time()),
        )
        self._db.commit()
        return cur.rowcount > 0

    def set_announcement_meta(
        self, thread_id: int, repo_name: str, repo_path: str, hash: str
    ) -> None:
        """Persist a commit announcement's source identity, keyed by thread.

        Announcement.meta is not otherwise stored (room.post keeps only text), so
        the orchestrator's digest path -- which sees Messages, not Announcements
        -- recovers repo/hash for the handover from here."""
        self._db.execute(
            "INSERT OR REPLACE INTO announcement_meta"
            " (thread_id, repo_name, repo_path, hash) VALUES (?, ?, ?, ?)",
            (thread_id, repo_name, repo_path, hash),
        )
        self._db.commit()

    def announcement_meta(self, thread_id: int) -> dict | None:
        row = self._db.execute(
            "SELECT repo_name, repo_path, hash FROM announcement_meta"
            " WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        return {"repo_name": row[0], "repo_path": row[1], "hash": row[2]}

    def commit_thread(self, hash: str) -> int | None:
        """The thread a commit hash already owns, or None."""
        row = self._db.execute(
            "SELECT thread_id FROM commit_threads WHERE hash = ?", (hash,)
        ).fetchone()
        return row[0] if row else None

    def set_commit_thread(self, hash: str, thread_id: int) -> None:
        """Bind a commit hash to its thread. Idempotent: the first binding wins,
        so a racing second creator does not steal the mapping."""
        self._db.execute(
            "INSERT OR IGNORE INTO commit_threads (hash, thread_id) VALUES (?, ?)",
            (hash, thread_id),
        )
        self._db.commit()

    def result_delivered(self, job_id: str) -> bool:
        """Whether this result's outcome has already been posted to the room.

        Result delivery is at-least-once by design (a crash between the state
        save and the publish re-finalizes on restart), and the two things a
        result does on arrival -- post it to the room, file its bug -- are the
        two that are not idempotent. The finding badge and the fix enqueue key
        off ids and already are. Durable rather than in-memory because the
        redelivery this guards against is the one a RESTART causes.
        """
        row = self._db.execute(
            "SELECT 1 FROM delivered_results WHERE job_id = ?", (job_id,)
        ).fetchone()
        return row is not None

    def mark_result_delivered(self, job_id: str) -> None:
        """Record that a result reached the room, after the post succeeded.

        Recorded after rather than claimed before: a post that raises is retried
        by the caller, and a claim taken up front would make that retry skip the
        very post it is retrying. The residual window (crash between post and
        this write) duplicates one message, which is the lesser failure and far
        narrower than the restart redelivery result_delivered exists to stop."""
        self._db.execute(
            "INSERT OR IGNORE INTO delivered_results (job_id, ts) VALUES (?, ?)",
            (job_id, time.time()),
        )
        self._db.commit()

    def mark_handover(self, job_id: str) -> bool:
        """Claim a job_id for handover to icompleteu; return True if newly claimed.

        The deterministic job_id makes this idempotent: a re-detected commit
        (re-poll, rebase) or a persona re-run produces the same id, so the second
        attempt returns False and the caller skips the duplicate enqueue.
        """
        cur = self._db.execute(
            "INSERT OR IGNORE INTO handover_jobs (job_id, ts) VALUES (?, ?)",
            (job_id, time.time()),
        )
        self._db.commit()
        return cur.rowcount > 0

    def handover_claimed(self, job_id: str) -> bool:
        """Whether a job_id was already handed over, without claiming it. Lets
        the caller order publish-then-claim: the claim is durable with no
        unclaim, so claiming first would turn a failed publish into a
        permanently skipped job."""
        cur = self._db.execute(
            "SELECT 1 FROM handover_jobs WHERE job_id = ?", (job_id,)
        )
        return cur.fetchone() is not None

    # ── pending bug filing (durable across tracker-credential expiry) ─────────
    #
    # A verified repro's bug is filed best-effort; on any failure the intent is
    # queued here and retried on a timer until it lands, so a lapsed credential
    # (daily, or over a weekend) never loses the bug. Keyed by finding_id, so a
    # re-published result is idempotent.

    def enqueue_pending_bug(
        self,
        *,
        finding_id: str,
        title: str,
        body: str,
        repo: str,
        commit_hash: str,
        author_email: str,
        isolates: bool | None,
        thread_id: int | None,
    ) -> None:
        """Queue (or refresh) a bug to file. Idempotent on finding_id: a re-run
        overwrites the content but preserves the original created_ts and does not
        reset attempts, so a genuinely stuck row still ages toward its park cap."""
        now = time.time()
        self._db.execute(
            """INSERT INTO pending_bugs
                 (finding_id, title, body, repo, commit_hash, author_email,
                  isolates, thread_id, attempts, created_ts, last_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
               ON CONFLICT(finding_id) DO UPDATE SET
                 title=excluded.title, body=excluded.body,
                 author_email=excluded.author_email, isolates=excluded.isolates,
                 thread_id=excluded.thread_id, last_ts=excluded.last_ts""",
            (
                finding_id,
                title,
                body,
                repo,
                commit_hash,
                author_email,
                None if isolates is None else int(isolates),
                thread_id,
                now,
                now,
            ),
        )
        self._db.commit()

    def pending_bug_exists(self, finding_id: str) -> bool:
        cur = self._db.execute(
            "SELECT 1 FROM pending_bugs WHERE finding_id = ?", (finding_id,)
        )
        return cur.fetchone() is not None

    def list_pending_bugs(self) -> list[dict]:
        """Every queued bug, oldest first, as plain dicts (isolates back to
        bool|None). The drainer walks these each sweep."""
        cur = self._db.execute(
            """SELECT finding_id, title, body, repo, commit_hash, author_email,
                      isolates, thread_id, attempts, created_ts, last_ts
                 FROM pending_bugs ORDER BY created_ts"""
        )
        cols = [c[0] for c in cur.description]
        rows = []
        for r in cur.fetchall():
            d = dict(zip(cols, r, strict=False))
            d["isolates"] = None if d["isolates"] is None else bool(d["isolates"])
            rows.append(d)
        return rows

    def bump_pending_bug(self, finding_id: str) -> None:
        """Record a failed attempt (increment count, stamp last_ts)."""
        self._db.execute(
            "UPDATE pending_bugs SET attempts = attempts + 1, last_ts = ?"
            " WHERE finding_id = ?",
            (time.time(), finding_id),
        )
        self._db.commit()

    def delete_pending_bug(self, finding_id: str) -> None:
        """Drop a row after a successful file (or a permanent park)."""
        self._db.execute("DELETE FROM pending_bugs WHERE finding_id = ?", (finding_id,))
        self._db.commit()

    # ── pending "thinking" placeholders (Chat typing indicator) ──────────────
    #
    # A placeholder card posted while an agent thinks, keyed by (thread, agent)
    # so it is robust to concurrent turns. Persisted so a crash can't strand a
    # "thinking..." card: a clean shutdown and the next startup both sweep these.

    def add_pending_card(
        self, thread_id: int, agent: str, space: str, message: str
    ) -> None:
        self._db.execute(
            "INSERT INTO chat_pending (thread_id, agent, space, message, ts)"
            " VALUES (?, ?, ?, ?, ?) ON CONFLICT(thread_id, agent) DO UPDATE SET"
            " space = excluded.space, message = excluded.message, ts = excluded.ts",
            (thread_id, agent, space, message, time.time()),
        )
        self._db.commit()

    def get_pending_card(self, thread_id: int, agent: str) -> str | None:
        """Resource name of this agent's placeholder, without consuming it."""
        row = self._db.execute(
            "SELECT message FROM chat_pending WHERE thread_id = ? AND agent = ?",
            (thread_id, agent),
        ).fetchone()
        return row[0] if row else None

    def remove_pending_card(self, message: str) -> None:
        """Remove one resolved placeholder row, by Chat message name.

        Keyed by the (globally unique) message name, not (thread, agent), and
        called only AFTER the Chat-side resolve: the row is the durable promise
        that the card gets cleaned up, so deleting it first would strand the
        card on a crash in between, and deleting by agent could take out a
        newer placeholder the same agent posted meanwhile."""
        self._db.execute("DELETE FROM chat_pending WHERE message = ?", (message,))
        self._db.commit()

    def all_pending_cards(self) -> list[tuple[str, str, str]]:
        """(agent, space, message) for every outstanding placeholder."""
        return [
            (r[0], r[1], r[2])
            for r in self._db.execute(
                "SELECT agent, space, message FROM chat_pending"
            ).fetchall()
        ]

    # ── agent-set timers (durable across restart) ────────────────────────────

    def add_timer(
        self, seq: int, thread_id: int, agent: str, fire_at: float, note: str
    ) -> None:
        """Persist a pending timer under the scheduler-assigned id. INSERT OR
        REPLACE so a reused-file edge (a seq colliding with a stale row) resolves
        to the live timer rather than raising."""
        self._db.execute(
            "INSERT OR REPLACE INTO timers (seq, thread_id, agent, fire_at, note)"
            " VALUES (?, ?, ?, ?, ?)",
            (seq, thread_id, agent, fire_at, note),
        )
        self._db.commit()

    def remove_timer(self, seq: int) -> None:
        """Drop a timer once it is cancelled or has fired. Idempotent."""
        self._db.execute("DELETE FROM timers WHERE seq = ?", (seq,))
        self._db.commit()

    def all_timers(self) -> list[tuple[int, int, str, float, str]]:
        """(seq, thread_id, agent, fire_at, note) for every persisted timer,
        oldest id first, so startup can rebuild the heap and seed the id
        counter."""
        return [
            (r[0], r[1], r[2], r[3], r[4])
            for r in self._db.execute(
                "SELECT seq, thread_id, agent, fire_at, note FROM timers ORDER BY seq"
            ).fetchall()
        ]

    def chat_thread_for_thread(self, thread_id: int) -> tuple[str, str] | None:
        """(space, Chat thread name) a room thread is linked to, or None.

        Prefers the earliest link: that is the originating Chat thread (e.g.
        the human's), so outbound posts land in it by server name. Later links
        for the same room thread can be bot-created threads from the era when
        outbound posted by client key only.
        """
        row = self._db.execute(
            "SELECT space, chat_thread FROM chat_threads WHERE thread_id = ?"
            " ORDER BY rowid LIMIT 1",
            (thread_id,),
        ).fetchone()
        return (row[0], row[1]) if row else None

    # ── source announcement notes ────────────────────────────────────────────
    #
    # A short log of what each proactive source has already announced, fed back
    # to the source's filter so it can judge whether new data is genuinely novel.

    def add_source_note(self, source: str, note: str) -> None:
        self._db.execute(
            "INSERT INTO source_notes (source, ts, note) VALUES (?, ?, ?)",
            (source, time.time(), note),
        )
        self._db.commit()

    def recent_source_notes(self, source: str, limit: int = 30) -> list[str]:
        rows = self._db.execute(
            "SELECT note FROM source_notes WHERE source = ? ORDER BY id DESC LIMIT ?",
            (source, limit),
        ).fetchall()
        return [r[0] for r in reversed(rows)]

    # ── plugin-owned thread state ────────────────────────────────────────────
    #
    # Core stores and returns the payload as opaque JSON text: it never parses
    # it, so the schema inside stays entirely with the plugin that wrote it.
    # Encoding is the caller's too -- a plugin serializes its own model (pydantic
    # or otherwise) rather than having core guess at a dict.

    def put_plugin_state(
        self, namespace: str, key: str, thread_id: int, json: str
    ) -> None:
        """Insert or replace one row. Upsert rather than insert-only because the
        natural use is a small mutable record (a proposal marked submitted), and
        read-modify-write through the owning plugin is the only writer."""
        self._db.execute(
            "INSERT INTO plugin_state (namespace, key, thread_id, json, ts)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(namespace, key) DO UPDATE SET"
            " thread_id = excluded.thread_id, json = excluded.json, ts = excluded.ts",
            (namespace, key, thread_id, json, time.time()),
        )
        self._db.commit()

    def get_plugin_state(self, namespace: str, key: str) -> tuple[int, str] | None:
        """(thread_id, json) for one row, or None. thread_id comes back because
        the caller's usual next question is "was this recorded in THIS thread",
        which it must not have to take on trust from the payload."""
        row = self._db.execute(
            "SELECT thread_id, json FROM plugin_state WHERE namespace = ? AND key = ?",
            (namespace, key),
        ).fetchone()
        return (row[0], row[1]) if row else None

    def list_plugin_state(
        self, namespace: str, thread_id: int
    ) -> list[tuple[str, str]]:
        """(key, json) for a thread's rows, oldest first. Ordered by rowid, not
        ts: the upsert above rewrites ts on every mutation, so a record the
        plugin edits would otherwise jump to the end of its own listing."""
        rows = self._db.execute(
            "SELECT key, json FROM plugin_state"
            " WHERE namespace = ? AND thread_id = ? ORDER BY rowid",
            (namespace, thread_id),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def sweep_plugin_state(
        self, namespace: str, older_than: float
    ) -> list[tuple[str, int, str]]:
        """(key, thread_id, json) for rows written before `older_than`, oldest
        first. The thread-scoped listing above answers "what is in this thread";
        this answers "what is overdue", which is the shape a plugin polling for
        something it is still waiting on needs -- it holds no thread to ask about
        until it finds the row.

        `older_than` is an absolute epoch cutoff rather than an age, so the
        caller's clock arithmetic is visible at its own call site.
        """
        rows = self._db.execute(
            "SELECT key, thread_id, json FROM plugin_state"
            " WHERE namespace = ? AND ts < ? ORDER BY ts",
            (namespace, older_than),
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def drop_plugin_state(self, namespace: str, key: str) -> None:
        """Forget one row. The plugin's own lifecycle, not prune's: a record that
        has served its purpose (a job that finally reported) should go when it
        does, rather than waiting for the thread it belongs to to be swept."""
        self._db.execute(
            "DELETE FROM plugin_state WHERE namespace = ? AND key = ?",
            (namespace, key),
        )
        self._db.commit()

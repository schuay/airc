# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Shared token-usage ledger for the daemon suite.

One SQLite file every component writes to (persona turns, triage, commit
review, ...), so the suite's spend is a single query rather than a sum across
per-component stores. WAL plus a busy timeout make concurrent writers from
separate processes survivable: a brief wait, not an instant "database is
locked".

The ledger is deliberately ignorant of threads: it stores an opaque
`thread_id` integer and never joins a titles table (that lives in airc's own
store). `top_threads` returns ids and sums; a caller that wants titles resolves
them against whatever store owns the threads. This keeps the ledger free of any
airc dependency so airc-processors can log to it directly.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import time
from pathlib import Path

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    thread_id INTEGER NOT NULL,
    agent TEXT NOT NULL,
    kind TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    -- Subset of input_tokens served from the provider prompt cache. Lets the
    -- summaries report a cache hit rate; the cost lever for tool-heavy turns.
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    -- Configured model id that served the call (e.g. google_vertexai:gemini-...).
    -- Separates the cheap filter model (coordinator, triage) from agent/review
    -- models. Empty for rows written before this column existed.
    model TEXT NOT NULL DEFAULT '',
    -- Number of model calls aggregated into this row (one turn/review can make
    -- many). input_tokens / model_calls is the average per-call prompt; a high
    -- ratio with few calls means one large prompt, a high product means a long
    -- tool-calling loop re-sending an accumulating context.
    model_calls INTEGER NOT NULL DEFAULT 0,
    -- Largest single-call input_tokens within this row. Distinguishes a turn
    -- that grew quadratically (max near the per-row sum) from many even calls.
    max_call_input_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_token_usage_thread ON token_usage(thread_id);
"""


class TokenLog:
    """Append-only ledger of per-turn token usage, with aggregate report queries.

    One row is one turn/review/triage run (which may aggregate several model
    calls). Open one instance per component; multiple instances against the same
    file are fine (WAL serializes writers).
    """

    def __init__(self, path: Path | str) -> None:
        # The ledger is non-critical bookkeeping: a component (notably the
        # sandboxed icompleteu worker, whose $HOME is a throwaway tmpfs and
        # whose real ledger is credited out-of-band from the journal) must
        # never fail to start or abandon a turn because the db path is
        # read-only or unwritable. So a connect/init failure disables the log
        # (self._db = None) with one warning instead of raising; add() and the
        # queries then no-op.
        self._db: sqlite3.Connection | None = None
        try:
            if isinstance(path, Path):
                path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(str(path), check_same_thread=False)
            # check_same_thread=False allows an accidental off-loop access (the
            # codebase uses asyncio.to_thread elsewhere); WAL + a busy timeout
            # make that survivable, and let separate suite processes write
            # concurrently.
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA busy_timeout=5000")
            db.executescript(_SCHEMA)
            self._db = db
            self._migrate()
            db.commit()
        except (sqlite3.Error, OSError) as e:
            log.warning("token ledger disabled (%s not writable): %s", path, e)
            if self._db is not None:
                self._db.close()
                self._db = None

    def _migrate(self) -> None:
        # CREATE TABLE IF NOT EXISTS leaves a pre-existing table alone, so a file
        # written before a column existed needs an additive, idempotent migration
        # (e.g. a ledger carried over from airc's old combined store).
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(token_usage)")}
        if "cached_input_tokens" not in cols:
            self._db.execute(
                "ALTER TABLE token_usage ADD COLUMN cached_input_tokens"
                " INTEGER NOT NULL DEFAULT 0"
            )
        if "model" not in cols:
            self._db.execute(
                "ALTER TABLE token_usage ADD COLUMN model TEXT NOT NULL DEFAULT ''"
            )
        if "model_calls" not in cols:
            self._db.execute(
                "ALTER TABLE token_usage ADD COLUMN model_calls INTEGER NOT NULL"
                " DEFAULT 0"
            )
        if "max_call_input_tokens" not in cols:
            self._db.execute(
                "ALTER TABLE token_usage ADD COLUMN max_call_input_tokens INTEGER"
                " NOT NULL DEFAULT 0"
            )

    def close(self) -> None:
        if self._db is not None:
            self._db.close()

    def add(
        self,
        thread_id: int,
        agent: str,
        kind: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        model: str = "",
        model_calls: int = 0,
        max_call_input_tokens: int = 0,
    ) -> None:
        if self._db is None:
            return
        try:
            self._db.execute(
                "INSERT INTO token_usage (ts, thread_id, agent, kind, input_tokens,"
                " output_tokens, cached_input_tokens, model, model_calls,"
                " max_call_input_tokens)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    thread_id,
                    agent,
                    kind,
                    input_tokens,
                    output_tokens,
                    cached_input_tokens,
                    model,
                    model_calls,
                    max_call_input_tokens,
                ),
            )
            self._db.commit()
        except (sqlite3.Error, OSError) as e:
            # Accounting is never part of the caller's result. In particular,
            # this method commonly runs in a finally block, where raising would
            # replace the model error or cancellation the caller is handling.
            #
            # BUSY/LOCKED is the transient case, and in a suite where several
            # components share one store it is the LIKELY case: another writer
            # held the file for a moment. Disabling the ledger on it turned one
            # collision into silent zero accounting until restart. Drop the one
            # row (roll back so no transaction dangles) and keep the ledger;
            # everything else -- a full disk, a corrupt or vanished file -- is
            # structural, and retrying per call would log a warning per model
            # call forever, so disable as before.
            transient = isinstance(e, sqlite3.Error) and getattr(
                e, "sqlite_errorcode", None
            ) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
            if transient:
                log.warning("token ledger: dropped one entry (%s)", e)
                with contextlib.suppress(sqlite3.Error):
                    self._db.rollback()
                return
            log.warning("token ledger disabled after write failure: %s", e)
            with contextlib.suppress(sqlite3.Error):
                self._db.close()
            self._db = None

    def totals(self, since: float = 0.0) -> tuple[int, int]:
        if self._db is None:
            return (0, 0)
        row = self._db.execute(
            "SELECT COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0)"
            " FROM token_usage WHERE ts >= ?",
            (since,),
        ).fetchone()
        return (row[0], row[1])

    def recent_max_input(self, thread_id: int, since: float = 0.0) -> int:
        """The largest single-turn input on a thread since `since` -- the memory
        compaction size signal. Uses max_call_input (the biggest single model call
        in a turn) so it reflects the actual per-request context size, and drops
        right after a context-generation bump because the next turn's checkpoint
        is small. 0 if no turns (or the ledger is disabled)."""
        if self._db is None:
            return 0
        row = self._db.execute(
            "SELECT COALESCE(MAX(max_call_input_tokens), 0) FROM token_usage"
            " WHERE thread_id = ? AND ts >= ?",
            (thread_id, since),
        ).fetchone()
        return row[0]

    def cached_input_total(self, since: float = 0.0) -> int:
        if self._db is None:
            return 0
        row = self._db.execute(
            "SELECT COALESCE(SUM(cached_input_tokens), 0) FROM token_usage"
            " WHERE ts >= ?",
            (since,),
        ).fetchone()
        return row[0]

    def totals_by_kind(self, since: float = 0.0) -> list[tuple[str, int, int]]:
        if self._db is None:
            return []
        rows = self._db.execute(
            "SELECT kind, SUM(input_tokens), SUM(output_tokens) FROM token_usage"
            " WHERE ts >= ? GROUP BY kind ORDER BY SUM(input_tokens) DESC",
            (since,),
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def totals_by_model(self, since: float = 0.0) -> list[tuple[str, int, int, int]]:
        """(model, input, output, cached), heaviest first. Empty model is '?'."""
        if self._db is None:
            return []
        rows = self._db.execute(
            "SELECT COALESCE(NULLIF(model, ''), '?'), SUM(input_tokens),"
            " SUM(output_tokens), SUM(cached_input_tokens) FROM token_usage"
            " WHERE ts >= ? GROUP BY 1 ORDER BY SUM(input_tokens) DESC",
            (since,),
        ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def totals_by_agent(self, since: float = 0.0) -> list[tuple[str, int, int]]:
        if self._db is None:
            return []
        rows = self._db.execute(
            "SELECT agent, SUM(input_tokens), SUM(output_tokens) FROM token_usage"
            " WHERE ts >= ? GROUP BY agent"
            " ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC",
            (since,),
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def top_threads(self, n: int = 5, since: float = 0.0) -> list[tuple[int, int, int]]:
        """(thread_id, input, output) for the heaviest threads, heaviest first.

        No title: the ledger does not own threads. A caller resolves the id to a
        title against its own store.
        """
        if self._db is None:
            return []
        rows = self._db.execute(
            "SELECT thread_id, SUM(input_tokens), SUM(output_tokens)"
            " FROM token_usage WHERE ts >= ? GROUP BY thread_id"
            " ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC LIMIT ?",
            (since, n),
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def heaviest_turns(
        self, n: int = 10, since: float = 0.0
    ) -> list[tuple[int, str, str, int, int, int, int]]:
        """Single token_usage rows with the most input, heaviest first.

        (thread_id, agent, kind, input, model_calls, max_call_input, output).
        One row is one turn/review run; surfacing the heaviest individually --
        with the call count and the largest single call -- is what distinguishes
        a quadratic tool-calling loop (input >> max_call_input, many calls) from
        one large prompt (input ~= max_call_input, one call). model_calls is 0
        for rows written before that column existed.
        """
        if self._db is None:
            return []
        rows = self._db.execute(
            "SELECT thread_id, agent, kind, input_tokens, model_calls,"
            " max_call_input_tokens, output_tokens FROM token_usage"
            " WHERE ts >= ? ORDER BY input_tokens DESC LIMIT ?",
            (since, n),
        ).fetchall()
        return [(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows]

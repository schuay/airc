# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Shared token-usage ledger for the daemon suite.

One SQLite file every component writes to (persona turns, triage, commit
review, ...), so the suite's spend is a single query rather than a sum across
per-component stores. WAL plus a busy timeout make concurrent writers from
separate processes survivable: a brief wait, not an instant "database is
locked".

A row is one `Usage` (usage.py): the counts as the provider reported them and
the dollars they cost at the price in force when the row was written. Cost is
stored, not derived at query time, because prices change and a ledger should
say what was paid.

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

from .usage import Usage

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
    -- Subset of input_tokens written to the provider prompt cache this call,
    -- both TTLs. A write costs more than plain input and only pays off when a
    -- later call reads it. 0 for Gemini, whose implicit cache reports no write.
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
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
    max_call_input_tokens INTEGER NOT NULL DEFAULT 0,
    -- Thinking subset of output_tokens.
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    -- Explicit-cache storage booked at creation: tokens held times TTL hours.
    cache_storage_token_hours REAL NOT NULL DEFAULT 0,
    -- What the row cost at the price in force when it was written. estimated
    -- marks a model priced at the generic fallback rate.
    usd REAL NOT NULL DEFAULT 0,
    estimated INTEGER NOT NULL DEFAULT 0,
    -- The same dollars split by side: prompt (input, cache reads and writes,
    -- storage) against output (thinking included).
    usd_input REAL NOT NULL DEFAULT 0,
    usd_output REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_token_usage_thread ON token_usage(thread_id);
-- The window caps query "what did the fleet spend since <ts>" at every
-- admission, and the reports scan by time too; the table was indexed on
-- thread_id alone, so both were full scans over a growing ledger.
CREATE INDEX IF NOT EXISTS idx_token_usage_ts ON token_usage(ts);
"""

# Columns added after the table first shipped, with the DDL that adds each.
# CREATE TABLE IF NOT EXISTS leaves a pre-existing table alone, so a file
# written before a column existed needs an additive, idempotent migration.
_ADDED_COLUMNS = (
    ("cached_input_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("model", "TEXT NOT NULL DEFAULT ''"),
    ("model_calls", "INTEGER NOT NULL DEFAULT 0"),
    ("max_call_input_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("cache_write_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("reasoning_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("cache_storage_token_hours", "REAL NOT NULL DEFAULT 0"),
    ("usd", "REAL NOT NULL DEFAULT 0"),
    ("estimated", "INTEGER NOT NULL DEFAULT 0"),
    ("usd_input", "REAL NOT NULL DEFAULT 0"),
    ("usd_output", "REAL NOT NULL DEFAULT 0"),
)


class TokenLog:
    """Append-only ledger of per-turn usage, with aggregate report queries.

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
            # A ledger this handle cannot write is disabled now, not at its
            # first add(): the schema statements above are no-ops on a file
            # that already carries it, so a read-only file opens clean and a
            # reader (the window cap's own handle) would report a healthy
            # window that no writer is filling. It has to touch a page: under
            # WAL, BEGIN IMMEDIATE takes its lock on the -shm file and passes,
            # and an insert rolled back never reaches the file.
            (version,) = db.execute("PRAGMA user_version").fetchone()
            db.execute(f"PRAGMA user_version = {int(version)}")
        except (sqlite3.Error, OSError) as e:
            log.warning("token ledger disabled (%s not writable): %s", path, e)
            if self._db is not None:
                self._db.close()
                self._db = None

    def _migrate(self) -> None:
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(token_usage)")}
        for name, ddl in _ADDED_COLUMNS:
            if name not in cols:
                self._db.execute(f"ALTER TABLE token_usage ADD COLUMN {name} {ddl}")

    @property
    def enabled(self) -> bool:
        """Whether the ledger is actually being kept.

        The rest of TokenLog fails soft exactly as its docstring promises: a
        disabled log's queries all return 0, which is right for a report (say
        nothing rather than crash) and wrong for exactly one reader, the window
        cap. A cap that reads that zero concludes nothing was spent and stops
        existing at the moment nobody can see the spending.
        """
        return self._db is not None

    def close(self) -> None:
        if self._db is not None:
            self._db.close()

    def add(self, usage: Usage, *, thread_id: int, agent: str, kind: str) -> None:
        if self._db is None:
            return
        try:
            self._db.execute(
                "INSERT INTO token_usage (ts, thread_id, agent, kind, input_tokens,"
                " output_tokens, cached_input_tokens, model, model_calls,"
                " max_call_input_tokens, cache_write_tokens, reasoning_tokens,"
                " cache_storage_token_hours, usd, estimated, usd_input, usd_output)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    thread_id,
                    agent,
                    kind,
                    usage.input,
                    usage.output,
                    usage.cache_read,
                    usage.model,
                    usage.calls,
                    usage.max_call_input,
                    usage.cache_write,
                    usage.reasoning,
                    usage.cache_storage_token_hours,
                    usage.usd,
                    int(usage.estimated),
                    usage.usd_input,
                    usage.usd_output,
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

    def usd_total(self, since: float = 0.0) -> tuple[float, bool]:
        """(dollars, any_estimated): the spend, and whether any of it was
        priced at the generic fallback rate."""
        if self._db is None:
            return (0.0, False)
        row = self._db.execute(
            "SELECT COALESCE(SUM(usd), 0), COALESCE(MAX(estimated), 0)"
            " FROM token_usage WHERE ts >= ?",
            (since,),
        ).fetchone()
        return (float(row[0]), bool(row[1]))

    def usd_in(self, since: float, until: float) -> float:
        """Dollars booked in [since, until). What the window-cap report calls
        "next free": the rows about to roll out of a rolling window."""
        if self._db is None:
            return 0.0
        row = self._db.execute(
            "SELECT COALESCE(SUM(usd), 0) FROM token_usage WHERE ts >= ? AND ts < ?",
            (since, until),
        ).fetchone()
        return float(row[0])

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

    def cache_write_total(self, since: float = 0.0) -> int:
        """Tokens written to the provider prompt cache. Writes far exceeding
        cached_input_total means paying the write premium without the reads."""
        if self._db is None:
            return 0
        row = self._db.execute(
            "SELECT COALESCE(SUM(cache_write_tokens), 0) FROM token_usage"
            " WHERE ts >= ?",
            (since,),
        ).fetchone()
        return row[0]

    def totals_by_kind(self, since: float = 0.0) -> list[tuple[str, int, int, float]]:
        """(kind, input, output, usd), heaviest input first."""
        if self._db is None:
            return []
        rows = self._db.execute(
            "SELECT kind, SUM(input_tokens), SUM(output_tokens), SUM(usd)"
            " FROM token_usage WHERE ts >= ? GROUP BY kind"
            " ORDER BY SUM(input_tokens) DESC",
            (since,),
        ).fetchall()
        return [(r[0], r[1], r[2], float(r[3])) for r in rows]

    def totals_by_model(
        self, since: float = 0.0
    ) -> list[tuple[str, int, int, int, int, float]]:
        """(model, input, output, cached, cache_written, usd), heaviest input
        first. Empty model is '?'."""
        if self._db is None:
            return []
        rows = self._db.execute(
            "SELECT COALESCE(NULLIF(model, ''), '?'), SUM(input_tokens),"
            " SUM(output_tokens), SUM(cached_input_tokens),"
            " SUM(cache_write_tokens), SUM(usd) FROM token_usage"
            " WHERE ts >= ? GROUP BY 1 ORDER BY SUM(input_tokens) DESC",
            (since,),
        ).fetchall()
        return [(r[0], r[1], r[2], r[3], r[4], float(r[5])) for r in rows]

    def totals_by_agent(self, since: float = 0.0) -> list[tuple[str, int, int, float]]:
        """(agent, input, output, usd), heaviest first."""
        if self._db is None:
            return []
        rows = self._db.execute(
            "SELECT agent, SUM(input_tokens), SUM(output_tokens), SUM(usd)"
            " FROM token_usage WHERE ts >= ? GROUP BY agent"
            " ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC",
            (since,),
        ).fetchall()
        return [(r[0], r[1], r[2], float(r[3])) for r in rows]

    def top_threads(
        self, n: int = 5, since: float = 0.0
    ) -> list[tuple[int, int, int, float]]:
        """(thread_id, input, output, usd) for the heaviest threads, heaviest
        first.

        No title: the ledger does not own threads. A caller resolves the id to a
        title against its own store.
        """
        if self._db is None:
            return []
        rows = self._db.execute(
            "SELECT thread_id, SUM(input_tokens), SUM(output_tokens), SUM(usd)"
            " FROM token_usage WHERE ts >= ? GROUP BY thread_id"
            " ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC LIMIT ?",
            (since, n),
        ).fetchall()
        return [(r[0], r[1], r[2], float(r[3])) for r in rows]

    def heaviest_turns(
        self, n: int = 10, since: float = 0.0
    ) -> list[tuple[int, str, str, int, int, int, int, float]]:
        """Single token_usage rows with the most input, heaviest first.

        (thread_id, agent, kind, input, model_calls, max_call_input, output,
        usd). One row is one turn/review run; surfacing the heaviest
        individually -- with the call count and the largest single call -- is
        what distinguishes a quadratic tool-calling loop (input >>
        max_call_input, many calls) from one large prompt (input ~=
        max_call_input, one call). model_calls is 0 for rows written before
        that column existed.
        """
        if self._db is None:
            return []
        rows = self._db.execute(
            "SELECT thread_id, agent, kind, input_tokens, model_calls,"
            " max_call_input_tokens, output_tokens, usd FROM token_usage"
            " WHERE ts >= ? ORDER BY input_tokens DESC LIMIT ?",
            (since, n),
        ).fetchall()
        return [(r[0], r[1], r[2], r[3], r[4], r[5], r[6], float(r[7])) for r in rows]


DAY_S = 24 * 3600.0
WEEK_S = 7 * DAY_S


class SpendWindows:
    """The rolling daily and weekly fleet caps, as one admission decision.

    Steps 4 and 5 bound one pass and one job; neither bounds how many of them a
    day runs, so on an expensive roster the fleet's only protection against a
    bad night is that somebody is watching it. This is that bound: one query per
    window at each intake boundary, `usd_total(now - window) < cap`.

    Deliberately not a reservation. It does not know what the work it admits
    will cost, and exact observance is not what the cap is for: spend converges
    to the cap plus one generation of work already running, and that generation
    is bounded by the per-pass and per-job budgets. Nothing already admitted is
    re-checked or killed -- cutting a review off mid-fan-out throws away what it
    spent and reports clean, which is indistinguishable from a good commit.

    The windows roll rather than aligning to a calendar: a calendar week permits
    the whole cap on Sunday night and the whole cap again on Monday morning,
    where a rolling window is a leaky bucket that actually bounds, with no
    timezone or DST question. What rolling costs is that it never visibly
    resets, so a bound fleet reads as a stuck one; `token_report.py` answers
    that by naming when the next dollars free up.

    One log line on the transition into bound and one out of it, never one per
    refused admission.
    """

    def __init__(
        self,
        log: TokenLog,
        daily_usd_cap: float | None = None,
        weekly_usd_cap: float | None = None,
    ) -> None:
        self._log = log
        self._caps = (
            ("daily", DAY_S, daily_usd_cap),
            ("weekly", WEEK_S, weekly_usd_cap),
        )
        # Which bound is announced: a cap name, "ledger", or None for open. The
        # latch is keyed on this rather than on the reason text, which carries
        # the spent amount and so changes on nearly every poll while bound --
        # in-flight work keeps booking rows and old rows keep rolling out.
        self._bound: str | None = None
        self._announced = False

    @property
    def configured(self) -> bool:
        return any(cap is not None for _, _, cap in self._caps)

    def bound(self, now: float | None = None) -> str | None:
        """Why the fleet may not start new work, or None when it may.

        A disabled ledger is no headroom, not an empty window: it is the one
        state where "nothing was spent" and "nobody can see what was spent" look
        identical from a query, and the safe reading of the second is to stop.
        """
        if not self.configured:
            return None
        now = time.time() if now is None else now
        key = reason = None
        if not self._log.enabled:
            key = "ledger"
            reason = (
                "the token ledger is disabled, so the spend window cannot be"
                " read; treating it as no headroom"
            )
        else:
            for name, window, cap in self._caps:
                if cap is None:
                    continue
                spent, estimated = self._log.usd_total(now - window)
                if spent >= cap:
                    key = name
                    reason = (
                        f"{name} spend cap reached:"
                        f" {'~' if estimated else ''}${spent:.2f} of ${cap:g}"
                        f" in the last {window / DAY_S:g}d"
                    )
                    break
        self._announce(key, reason)
        return reason

    def _announce(self, key: str | None, reason: str | None) -> None:
        if key == self._bound and self._announced:
            return
        self._announced = True
        if reason is not None:
            log.warning(
                "spend cap: %s; deferring new work until the window frees", reason
            )
        elif self._bound is not None:
            log.info("spend cap: window freed; taking new work again")
        self._bound = key

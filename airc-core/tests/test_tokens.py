# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""TokenLog: the shared token-usage ledger (aggregates, windowing, migration)."""

import sqlite3
import time

import pytest
from airc_core import TokenLog, Usage


def _u(
    input: int,
    output: int,
    cached: int = 0,
    model: str = "",
    *,
    calls: int = 0,
    max_call_input: int = 0,
    written: int = 0,
    usd: float = 0.0,
    estimated: bool = False,
) -> Usage:
    """A row's worth of usage with the counts spelled out; the ledger stores
    what it is handed and never prices, so the dollars are given too."""
    return Usage(
        model=model,
        calls=calls,
        input=input,
        cache_read=cached,
        cache_write_5m=written,
        output=output,
        max_call_input=max_call_input,
        usd=usd,
        estimated=estimated,
    )


def test_token_accounting(tmp_path):
    t = TokenLog(tmp_path / "tokens.db")
    t.add(_u(1000, 50), thread_id=1, agent="perf", kind="turn")
    t.add(_u(100, 1), thread_id=1, agent="perf", kind="coordinator")
    t.add(_u(300, 20), thread_id=2, agent="compiler", kind="turn")
    assert t.totals() == (1400, 71)
    kinds = {k: (i, o) for k, i, o, _usd in t.totals_by_kind()}
    assert kinds == {"turn": (1300, 70), "coordinator": (100, 1)}
    agents = {a: (i, o) for a, i, o, _usd in t.totals_by_agent()}
    assert agents["perf"] == (1100, 51)
    top = t.top_threads(n=1)
    assert top[0] == (1, 1100, 51, 0.0)  # (thread_id, input, output, usd)
    assert t.totals(since=1e12) == (0, 0)  # window filter


def test_cost_is_stored_per_row_and_summed_with_the_estimate_flag(tmp_path):
    """The ledger records the dollars it is handed, never re-prices, and a
    total that includes one generic-rate row says so."""
    t = TokenLog(tmp_path / "tokens.db")
    t.add(_u(1000, 50, model="a", usd=0.5), thread_id=1, agent="perf", kind="turn")
    t.add(
        _u(1000, 50, model="b", usd=0.25, estimated=True),
        thread_id=2,
        agent="perf",
        kind="turn",
    )
    assert t.usd_total() == (pytest.approx(0.75), True)
    assert t.usd_total(since=1e12) == (0.0, False)
    by_model = {m: usd for m, _i, _o, _c, _w, usd in t.totals_by_model()}
    assert by_model == {"a": pytest.approx(0.5), "b": pytest.approx(0.25)}
    rows = t.heaviest_turns(n=2)
    assert {r[7] for r in rows} == {0.5, 0.25}
    row = t._db.execute(
        "SELECT usd, estimated, reasoning_tokens FROM token_usage WHERE model='b'"
    ).fetchone()
    assert row == (0.25, 1, 0)


def test_heaviest_turns_surfaces_call_shape(tmp_path):
    t = TokenLog(tmp_path / "tokens.db")
    # A quadratic review (many calls, sum >> the largest single call) and a
    # single large prompt (sum ~= the one call) with the same input total.
    t.add(
        _u(6_000_000, 5000, model="pro", calls=50, max_call_input=240_000),
        thread_id=1,
        agent="review",
        kind="review",
    )
    t.add(
        _u(6_000_000, 4000, model="pro", calls=1, max_call_input=6_000_000),
        thread_id=1,
        agent="perf",
        kind="turn",
    )
    rows = t.heaviest_turns(n=2)
    shape = {(r[1], r[4]): r[5] for r in rows}  # (agent, calls) -> max_call_input
    assert shape[("review", 50)] == 240_000
    assert shape[("perf", 1)] == 6_000_000


def test_totals_by_model(tmp_path):
    t = TokenLog(tmp_path / "tokens.db")
    t.add(_u(1000, 50, 400, "vertexai:pro"), thread_id=1, agent="c", kind="turn")
    t.add(_u(200, 1, 0, "vertexai:flash"), thread_id=0, agent="triage", kind="triage")
    t.add(_u(300, 2, 150, "vertexai:flash"), thread_id=1, agent="co", kind="co")
    t.add(_u(10, 1), thread_id=1, agent="old", kind="turn")  # no model -> '?'
    by_model = {m: (i, o, c) for m, i, o, c, _w, _usd in t.totals_by_model()}
    assert by_model["vertexai:flash"] == (500, 3, 150)
    assert by_model["vertexai:pro"] == (1000, 50, 400)
    assert by_model["?"] == (10, 1, 0)
    assert t.cached_input_total() == 550


def test_cache_writes_are_tracked_separately_from_reads(tmp_path):
    # A write is billed ABOVE base input and only pays off via a later read, so
    # writes-without-reads must be visible rather than folded into input.
    t = TokenLog(tmp_path / "tokens.db")
    t.add(_u(1000, 10, 0, "claude", written=900), thread_id=1, agent="a", kind="t")
    t.add(_u(1000, 10, 900, "claude"), thread_id=1, agent="a", kind="t")
    t.add(_u(500, 5, 100, "gemini"), thread_id=1, agent="b", kind="t")
    assert t.cache_write_total() == 900
    assert t.cached_input_total() == 1000
    by_model = {m: (c, w) for m, _i, _o, c, w, _usd in t.totals_by_model()}
    assert by_model["claude"] == (900, 900)
    assert by_model["gemini"] == (100, 0)


def test_both_write_ttls_land_in_the_one_write_column(tmp_path):
    t = TokenLog(tmp_path / "tokens.db")
    u = Usage(model="claude", input=1000, cache_write_5m=300, cache_write_1h=200)
    t.add(u, thread_id=1, agent="a", kind="t")
    assert t.cache_write_total() == 500


def test_new_columns_are_added_to_an_existing_ledger(tmp_path):
    # The migration must be additive: a ledger written before a column
    # existed keeps its rows and reports 0 for them. The legacy row is
    # inserted with raw SQL because add() targets the post-migration shape.
    path = tmp_path / "tokens.db"
    old = TokenLog(path)
    for col in ("cache_write_tokens", "usd", "estimated", "reasoning_tokens"):
        old._db.execute(f"ALTER TABLE token_usage DROP COLUMN {col}")
    old._db.execute(
        "INSERT INTO token_usage (ts, thread_id, agent, kind, input_tokens,"
        " output_tokens, cached_input_tokens, model)"
        " VALUES (0, 1, 'a', 'turn', 100, 5, 20, 'm')"
    )
    old._db.commit()
    old.close()

    t = TokenLog(path)
    assert t.cache_write_total() == 0
    assert t.totals_by_model() == [("m", 100, 5, 20, 0, 0.0)]
    assert t.usd_total() == (0.0, False)
    t.add(_u(100, 5, 0, "m", written=42, usd=0.1), thread_id=1, agent="a", kind="t")
    assert t.cache_write_total() == 42
    assert t.usd_total() == (pytest.approx(0.1), False)


def test_shared_ledger_across_instances(tmp_path):
    # Separate suite processes each open their own TokenLog on the same file;
    # the totals are the union (WAL serializes the concurrent writers).
    path = tmp_path / "tokens.db"
    a = TokenLog(path)
    b = TokenLog(path)
    a.add(_u(100, 5), thread_id=1, agent="perf", kind="turn")
    b.add(_u(200, 9), thread_id=2, agent="review", kind="review")
    assert TokenLog(path).totals() == (300, 14)


def test_legacy_file_migration_adds_columns(tmp_path):
    """A token_usage table predating the model/cached/calls columns migrates
    cleanly (e.g. a ledger carried over from airc's old combined store)."""
    path = tmp_path / "legacy.db"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE token_usage (id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " ts REAL NOT NULL, thread_id INTEGER NOT NULL, agent TEXT NOT NULL,"
        " kind TEXT NOT NULL, input_tokens INTEGER NOT NULL,"
        " output_tokens INTEGER NOT NULL)"
    )
    db.execute(
        "INSERT INTO token_usage (ts, thread_id, agent, kind, input_tokens,"
        " output_tokens) VALUES (1.0, 1, 'perf', 'turn', 100, 5)"
    )
    db.commit()
    db.close()

    t = TokenLog(path)  # runs _migrate
    assert t.totals() == (100, 5)
    assert t.totals_by_model() == [("?", 100, 5, 0, 0, 0.0)]
    t.add(_u(50, 2, 10, "vertexai:flash"), thread_id=1, agent="perf", kind="turn")
    by_model = {m: (i, o, c) for m, i, o, c, _w, _usd in t.totals_by_model()}
    assert by_model["vertexai:flash"] == (50, 2, 10)


def test_readonly_db_disables_ledger_without_raising(tmp_path, caplog):
    # The sandboxed icompleteu worker's $HOME is a throwaway tmpfs and its real
    # ledger is credited from the journal; a read-only or unwritable db path
    # must disable the ledger (no-op add + empty reads), never raise and abandon
    # the turn ("attempt to write a readonly database").
    import logging
    import os
    import stat

    db = tmp_path / "ro" / "tokens.db"
    db.parent.mkdir()
    TokenLog(db).close()  # create it once, writable
    # Make the file and its directory read-only, so both connect-time writes
    # (WAL) and INSERTs fail.
    db.chmod(stat.S_IREAD)
    os.chmod(db.parent, stat.S_IREAD | stat.S_IEXEC)
    try:
        with caplog.at_level(logging.WARNING, logger="airc_core.tokens"):
            t = TokenLog(db)
            t.add(_u(100, 10), thread_id=1, agent="perf", kind="turn")  # no raise
        assert t.totals() == (0, 0)
        assert t.totals_by_kind() == []
        assert "token ledger disabled" in caplog.text
    finally:
        os.chmod(db.parent, stat.S_IRWXU)  # let tmp_path cleanup remove it


def test_runtime_write_failure_is_noncritical(tmp_path, caplog):
    import logging

    class BrokenConnection:
        def execute(self, *args):
            raise sqlite3.OperationalError("database is locked")

        def close(self):
            pass

    tokens = TokenLog(tmp_path / "tokens.db")
    tokens._db.close()
    tokens._db = BrokenConnection()

    with caplog.at_level(logging.WARNING, logger="airc_core.tokens"):
        tokens.add(_u(100, 10), thread_id=1, agent="triage", kind="structured-task")

    assert tokens._db is None
    assert "disabled after write failure" in caplog.text


def test_transient_busy_drops_the_row_and_keeps_the_ledger(tmp_path, caplog):
    import logging

    class BusyConnection:
        rolled_back = False

        def execute(self, *args):
            e = sqlite3.OperationalError("database is locked")
            e.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise e

        def rollback(self):
            self.rolled_back = True

    tokens = TokenLog(tmp_path / "tokens.db")
    real = tokens._db
    busy = BusyConnection()
    tokens._db = busy

    with caplog.at_level(logging.WARNING, logger="airc_core.tokens"):
        tokens.add(_u(100, 10), thread_id=1, agent="triage", kind="structured-task")

    # One collision costs one row, never the ledger: the connection stays,
    # the failed write is rolled back, and the next write records normally.
    assert tokens._db is busy
    assert busy.rolled_back
    assert "dropped one entry" in caplog.text
    tokens._db = real
    tokens.add(_u(5, 7), thread_id=1, agent="triage", kind="structured-task")
    assert tokens.totals() == (5, 7)


# ── rolling spend windows ────────────────────────────────────────────────────


def _spent(log, usd, ts):
    """One booked row at a chosen time. add() stamps time.time(), so the ts is
    rewritten -- the windows are entirely about how old a row is."""
    from airc_core.usage import Usage

    log.add(
        Usage(model="claude-opus-5", calls=1, usd=usd), thread_id=0, agent="a", kind="k"
    )
    log._db.execute(
        "UPDATE token_usage SET ts = ? WHERE id = last_insert_rowid()", (ts,)
    )
    log._db.commit()


def test_no_cap_configured_never_binds(tmp_path):
    from airc_core.tokens import SpendWindows, TokenLog

    log = TokenLog(tmp_path / "t.db")
    _spent(log, 10_000.0, time.time())
    windows = SpendWindows(log)
    assert not windows.configured
    assert windows.bound() is None


def test_the_daily_cap_binds_on_the_last_24h_alone(tmp_path):
    from airc_core.tokens import SpendWindows, TokenLog

    log = TokenLog(tmp_path / "t.db")
    now = time.time()
    _spent(log, 90.0, now - 25 * 3600)  # yesterday: out of the window
    _spent(log, 40.0, now - 3600)
    windows = SpendWindows(log, daily_usd_cap=50.0)
    assert windows.bound(now) is None  # 40 of 50, the old row does not count
    _spent(log, 15.0, now - 60)
    reason = windows.bound(now)
    assert reason is not None and "daily" in reason and "$55.00" in reason


def test_either_window_binds(tmp_path):
    """Both are checked; a week that is over does not need the day to be."""
    from airc_core.tokens import SpendWindows, TokenLog

    log = TokenLog(tmp_path / "t.db")
    now = time.time()
    for day_ago in range(1, 7):
        _spent(log, 30.0, now - day_ago * 86400 + 60)
    windows = SpendWindows(log, daily_usd_cap=50.0, weekly_usd_cap=100.0)
    assert windows.bound(now) is not None  # 180 over the week, 0 today


def test_the_window_rolls_rather_than_resetting(tmp_path):
    """The leaky bucket: spend that ages out of the window frees up, with no
    calendar boundary that would permit the whole cap twice in two hours."""
    from airc_core.tokens import SpendWindows, TokenLog

    log = TokenLog(tmp_path / "t.db")
    now = time.time()
    _spent(log, 60.0, now - 23 * 3600)
    windows = SpendWindows(log, daily_usd_cap=50.0)
    assert windows.bound(now) is not None
    assert windows.bound(now + 2 * 3600) is None  # the row has rolled out


def test_a_disabled_ledger_is_no_headroom_not_an_empty_window(tmp_path):
    """The hazard the whole check is written around. TokenLog returns 0 from
    every query when it cannot be opened or written, and a cap that reads that
    zero concludes nothing was spent -- so it stops existing at exactly the
    moment nobody can see the spending."""
    from airc_core.tokens import SpendWindows, TokenLog

    log = TokenLog(tmp_path / "t.db")
    assert log.enabled
    log._db = None  # what a full disk or a vanished file leaves behind
    assert not log.enabled
    assert log.usd_total(0.0) == (0.0, False)  # reads as an empty window
    windows = SpendWindows(log, daily_usd_cap=50.0)
    reason = windows.bound()
    assert reason is not None and "ledger is disabled" in reason


def test_an_unwritable_ledger_is_disabled_at_open_not_at_first_write(tmp_path):
    """The schema statements are no-ops on a file that already carries it, so a
    read-only file used to open clean and only a write disabled the handle. A
    handle that never writes -- a window cap's own -- then reported a healthy
    window that no writer was filling."""
    import os

    from airc_core.tokens import TokenLog

    path = tmp_path / "t.db"
    TokenLog(path).close()
    os.chmod(path, 0o444)
    if os.access(path, os.W_OK):
        pytest.skip("chmod does not bind here (root or a permissive fs)")
    assert not TokenLog(path).enabled


def test_a_disabled_ledger_without_a_cap_is_nobody_business(tmp_path):
    from airc_core.tokens import SpendWindows, TokenLog

    log = TokenLog(tmp_path / "t.db")
    log._db = None
    assert SpendWindows(log).bound() is None


def test_the_transition_is_logged_once_in_each_direction(tmp_path, caplog):
    """One line on the way into bound and one on the way out, never one per
    refused admission -- the loop asks on every poll."""
    import logging

    from airc_core.tokens import SpendWindows, TokenLog

    log = TokenLog(tmp_path / "t.db")
    now = time.time()
    _spent(log, 60.0, now - 3600)
    windows = SpendWindows(log, daily_usd_cap=50.0)
    with caplog.at_level(logging.INFO, logger="airc_core.tokens"):
        for _ in range(5):
            windows.bound(now)
        for _ in range(5):
            windows.bound(now + 25 * 3600)
    lines = [r.message for r in caplog.records if "spend cap" in r.message]
    assert len(lines) == 2, lines
    assert "deferring new work" in lines[0]
    assert "window freed" in lines[1]


def test_a_moving_ledger_does_not_re_announce_the_same_bound(tmp_path, caplog):
    """The bound amount is never still: in-flight work keeps booking rows and
    old rows keep rolling out, so a latch on the reason text (which carries the
    amount) re-fires on nearly every poll. The latch is on WHICH cap binds."""
    import logging

    from airc_core.tokens import SpendWindows, TokenLog

    log = TokenLog(tmp_path / "t.db")
    now = time.time()
    windows = SpendWindows(log, daily_usd_cap=50.0)
    with caplog.at_level(logging.INFO, logger="airc_core.tokens"):
        for _ in range(5):
            _spent(log, 20.0, now - 60)  # 20, 40, 60, 80, 100
            windows.bound(now)
    lines = [r.message for r in caplog.records if "spend cap" in r.message]
    assert len(lines) == 1, lines


def test_the_ledger_is_indexed_on_time(tmp_path):
    """The window query runs at every admission; the table used to be indexed on
    thread_id alone, so it was a full scan over a growing ledger."""
    from airc_core.tokens import TokenLog

    log = TokenLog(tmp_path / "t.db")
    names = {r[1] for r in log._db.execute("PRAGMA index_list(token_usage)")}
    assert "idx_token_usage_ts" in names
    plan = log._db.execute(
        "EXPLAIN QUERY PLAN SELECT SUM(usd) FROM token_usage WHERE ts >= 0"
    ).fetchall()
    assert any("idx_token_usage_ts" in str(r) for r in plan), plan

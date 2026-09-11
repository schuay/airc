# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""TokenLog: the shared token-usage ledger (aggregates, windowing, migration)."""

import sqlite3

from airc_core import TokenLog


def test_token_accounting(tmp_path):
    t = TokenLog(tmp_path / "tokens.db")
    t.add(1, "perf", "turn", 1000, 50)
    t.add(1, "perf", "coordinator", 100, 1)
    t.add(2, "compiler", "turn", 300, 20)
    assert t.totals() == (1400, 71)
    kinds = {k: (i, o) for k, i, o in t.totals_by_kind()}
    assert kinds == {"turn": (1300, 70), "coordinator": (100, 1)}
    agents = {a: (i, o) for a, i, o in t.totals_by_agent()}
    assert agents["perf"] == (1100, 51)
    top = t.top_threads(n=1)
    assert top[0] == (1, 1100, 51)  # (thread_id, input, output); no title
    assert t.totals(since=1e12) == (0, 0)  # window filter


def test_heaviest_turns_surfaces_call_shape(tmp_path):
    t = TokenLog(tmp_path / "tokens.db")
    # A quadratic review (many calls, sum >> the largest single call) and a
    # single large prompt (sum ~= the one call) with the same input total.
    t.add(
        1,
        "review",
        "review",
        6_000_000,
        5000,
        0,
        "pro",
        model_calls=50,
        max_call_input_tokens=240_000,
    )
    t.add(
        1,
        "perf",
        "turn",
        6_000_000,
        4000,
        0,
        "pro",
        model_calls=1,
        max_call_input_tokens=6_000_000,
    )
    t.add(1, "perf", "turn", 100, 2)  # small; not surfaced first
    rows = t.heaviest_turns(n=2)
    assert [r[3] for r in rows] == [6_000_000, 6_000_000]  # by input desc
    shape = {(r[1], r[4]): r[5] for r in rows}  # (agent, calls) -> max_call_input
    assert shape[("review", 50)] == 240_000
    assert shape[("perf", 1)] == 6_000_000


def test_totals_by_model(tmp_path):
    t = TokenLog(tmp_path / "tokens.db")
    t.add(1, "compiler", "turn", 1000, 50, 400, "vertexai:pro")
    t.add(0, "triage", "triage", 200, 1, 0, "vertexai:flash")
    t.add(1, "coordinator", "coordinator", 300, 2, 150, "vertexai:flash")
    t.add(1, "old", "turn", 10, 1)  # no model -> '?'
    by_model = {m: (i, o, c) for m, i, o, c, _w in t.totals_by_model()}
    assert by_model["vertexai:flash"] == (500, 3, 150)
    assert by_model["vertexai:pro"] == (1000, 50, 400)
    assert by_model["?"] == (10, 1, 0)
    assert t.cached_input_total() == 550


def test_cache_writes_are_tracked_separately_from_reads(tmp_path):
    # A write is billed ABOVE base input and only pays off via a later read, so
    # writes-without-reads must be visible rather than folded into input.
    t = TokenLog(tmp_path / "tokens.db")
    t.add(1, "a", "turn", 1000, 10, 0, "claude", cache_write_tokens=900)
    t.add(1, "a", "turn", 1000, 10, 900, "claude")  # the read that pays it back
    t.add(1, "b", "turn", 500, 5, 100, "gemini")  # implicit cache: no write
    assert t.cache_write_total() == 900
    assert t.cached_input_total() == 1000
    by_model = {m: (c, w) for m, _i, _o, c, w in t.totals_by_model()}
    assert by_model["claude"] == (900, 900)
    assert by_model["gemini"] == (100, 0)


def test_add_keeps_model_positional_after_the_token_counts(tmp_path):
    # Call sites splat a 3-tuple: add(..., *usage_counts(usage), model). If
    # cache_write_tokens were ever inserted before `model`, the model id would
    # land in a token column and this would fail.
    t = TokenLog(tmp_path / "tokens.db")
    t.add(1, "a", "turn", *(100, 5, 20), "some-model")
    assert t.totals_by_model() == [("some-model", 100, 5, 20, 0)]


def test_cache_write_column_is_added_to_an_existing_ledger(tmp_path):
    # The migration must be additive: a ledger written before the column
    # existed keeps its rows and reports 0 writes for them. The legacy row is
    # inserted with raw SQL because add() targets the post-migration shape.
    path = tmp_path / "tokens.db"
    old = TokenLog(path)
    old._db.execute("ALTER TABLE token_usage DROP COLUMN cache_write_tokens")
    old._db.execute(
        "INSERT INTO token_usage (ts, thread_id, agent, kind, input_tokens,"
        " output_tokens, cached_input_tokens, model)"
        " VALUES (0, 1, 'a', 'turn', 100, 5, 20, 'm')"
    )
    old._db.commit()
    old.close()

    t = TokenLog(path)
    assert t.cache_write_total() == 0
    assert t.totals_by_model() == [("m", 100, 5, 20, 0)]
    t.add(1, "a", "turn", 100, 5, 0, "m", cache_write_tokens=42)
    assert t.cache_write_total() == 42


def test_shared_ledger_across_instances(tmp_path):
    # Separate suite processes each open their own TokenLog on the same file;
    # the totals are the union (WAL serializes the concurrent writers).
    path = tmp_path / "tokens.db"
    a = TokenLog(path)
    b = TokenLog(path)
    a.add(1, "perf", "turn", 100, 5)
    b.add(2, "review", "review", 200, 9)
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
    assert t.totals_by_model() == [("?", 100, 5, 0, 0)]
    t.add(1, "perf", "turn", 50, 2, 10, "vertexai:flash")
    by_model = {m: (i, o, c) for m, i, o, c, _w in t.totals_by_model()}
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
            t.add(1, "perf", "turn", 100, 10)  # no raise
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
        tokens.add(1, "triage", "structured-task", 100, 10)

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
        tokens.add(1, "triage", "structured-task", 100, 10)

    # One collision costs one row, never the ledger: the connection stays,
    # the failed write is rolled back, and the next write records normally.
    assert tokens._db is busy
    assert busy.rolled_back
    assert "dropped one entry" in caplog.text
    tokens._db = real
    tokens.add(1, "triage", "structured-task", 5, 7)
    assert tokens.totals() == (5, 7)

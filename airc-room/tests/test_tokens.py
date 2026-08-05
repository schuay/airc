# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Token report formatting (cli helpers)."""

from airc_core import TokenLog
from airc_room.cli import _fmt_tokens, _pct, _token_summary_line
from airc_room.store import Store


def test_fmt_tokens():
    assert _fmt_tokens(999) == "999"
    assert _fmt_tokens(1000) == "1k"
    assert _fmt_tokens(1_234_000) == "1.2M"


def test_pct():
    assert _pct(0, 0) == "n/a"
    assert _pct(25, 100) == "25%"


def test_token_summary_line(tmp_path):
    s = Store(tmp_path / "t.db")
    tokens = TokenLog(tmp_path / "tokens.db")
    t = s.create_thread("perf chat")
    tokens.add(t.id, "perf", "turn", 50_000, 2_000, 40_000)
    line = _token_summary_line(s, tokens)
    assert "all-time 50k in (80% cached) / 2k out" in line
    # The thread title is resolved from the store, not the ledger.
    assert "perf chat" in line


def test_recent_max_input_is_the_compaction_signal(tmp_path):
    tokens = TokenLog(tmp_path / "tokens.db")
    # The signal is the largest single-call input on the thread, not the sum.
    tokens.add(1, "chef", "turn", 30_000, 500, 0, max_call_input_tokens=30_000)
    tokens.add(1, "hawk", "turn", 90_000, 900, 0, max_call_input_tokens=90_000)
    tokens.add(2, "aide", "turn", 10_000, 100, 0, max_call_input_tokens=10_000)
    assert tokens.recent_max_input(1) == 90_000  # per-thread max, not summed
    assert tokens.recent_max_input(2) == 10_000
    assert tokens.recent_max_input(999) == 0  # no turns
    # A `since` cutoff excludes older turns, so the signal drops after a
    # compaction bump makes later turns small.
    import time

    cutoff = time.time() + 1
    assert tokens.recent_max_input(1, since=cutoff) == 0

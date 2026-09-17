# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Token report formatting (cli helpers)."""

from airc_core import TokenLog, Usage
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
    tokens.add(
        Usage(input=50_000, output=2_000, cache_read=40_000, usd=0.25),
        thread_id=t.id,
        agent="perf",
        kind="turn",
    )
    line = _token_summary_line(s, tokens)
    assert "all-time $0.25, 50k in (80% cached) / 2k out" in line
    # The thread title is resolved from the store, not the ledger.
    assert "perf chat" in line


def test_recent_max_input_is_the_compaction_signal(tmp_path):
    tokens = TokenLog(tmp_path / "tokens.db")
    # The signal is the largest single-call input on the thread, not the sum.
    for tid, agent, n in (
        (1, "chef", 30_000),
        (1, "hawk", 90_000),
        (2, "aide", 10_000),
    ):
        u = Usage(input=n, output=n // 100, max_call_input=n)
        tokens.add(u, thread_id=tid, agent=agent, kind="turn")
    assert tokens.recent_max_input(1) == 90_000  # per-thread max, not summed
    assert tokens.recent_max_input(2) == 10_000
    assert tokens.recent_max_input(999) == 0  # no turns
    # A `since` cutoff excludes older turns, so the signal drops after a
    # compaction bump makes later turns small.
    import time

    cutoff = time.time() + 1
    assert tokens.recent_max_input(1, since=cutoff) == 0

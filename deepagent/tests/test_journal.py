# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""The event journal appends, follows, and survives partial writes."""

import threading

from deepagent import Event, EventKind, Journal


def test_append_and_read_roundtrip(tmp_path):
    j = Journal(tmp_path / "events.jsonl")
    j.emit(EventKind.STEP, name="draft", text="review")
    j.emit(EventKind.TOOL_START, agent="draft", turn=0, name="shell", text="ls")
    assert j.count == 2

    events = Journal.read(tmp_path / "events.jsonl")
    assert [e.kind for e in events] == [EventKind.STEP, EventKind.TOOL_START]
    assert events[1].name == "shell" and events[1].turn == 0
    assert events[0].ts  # stamped


def test_read_skips_partial_trailing_line(tmp_path):
    p = tmp_path / "events.jsonl"
    good = Event(kind=EventKind.MESSAGE, text="hi").model_dump_json()
    p.write_text(good + "\n{ truncated half-writ")
    events = Journal.read(p)
    assert len(events) == 1 and events[0].text == "hi"


def test_read_missing_is_empty(tmp_path):
    assert Journal.read(tmp_path / "nope.jsonl") == []


def test_append_failure_does_not_raise(tmp_path):
    # A journal whose parent is a file, not a dir, cannot be written -- appending
    # must degrade quietly, never crash the run it is only observing.
    (tmp_path / "blocked").write_text("i am a file")
    j = Journal(tmp_path / "blocked" / "events.jsonl")
    j.emit(EventKind.STEP, name="setup")  # no raise
    assert j.count == 0  # nothing recorded


def test_progress_counts_only_work_events(tmp_path):
    # The reentry loop's liveness cursor must ignore the harness's own TURN/USAGE
    # bookkeeping (emitted unconditionally every turn) and orchestration events,
    # counting only agent output/tool calls -- else every turn looks alive.
    j = Journal(tmp_path / "events.jsonl")
    for kind in (EventKind.TURN, EventKind.USAGE, EventKind.STEP, EventKind.NOTIFY):
        j.emit(kind, agent="draft")
    assert j.count == 4 and j.progress == 0  # bookkeeping/orchestration: not work
    # Friction rides alongside the REPORT that already counts as progress, so it
    # must not add a second progress tick for the same turn.
    j.emit(EventKind.FRICTION, agent="draft", text="build was broken")
    assert j.count == 5 and j.progress == 0
    j.emit(EventKind.TOOL_START, name="shell")
    j.emit(EventKind.THINKING, text="hmm")
    j.emit(EventKind.MESSAGE, text="done")
    j.emit(EventKind.REPORT, text="ok")
    assert j.count == 9 and j.progress == 4  # the four agent-work events


def test_concurrent_appends_do_not_tear(tmp_path):
    # The langgraph harness drives append() from callback hooks on a shared
    # executor thread pool, so appends can race. The lock must keep every record
    # a whole line -- Journal.read tolerates only a torn TRAILING line, so a
    # mid-file tear would silently drop events from the sole crash trace.
    p = tmp_path / "events.jsonl"
    j = Journal(p)
    n_threads, per_thread = 16, 40
    barrier = threading.Barrier(n_threads)

    def writer(tid: int) -> None:
        barrier.wait()  # maximize overlap
        for k in range(per_thread):
            # A payload well past pipe-atomic size, so an unlocked write would
            # interleave visibly.
            j.emit(EventKind.MESSAGE, agent=f"t{tid}", text="x" * 3000)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = n_threads * per_thread
    assert j.count == total
    # Every line parses (no interleave), and none was dropped.
    events = Journal.read(p)
    assert len(events) == total
    assert all(e.kind is EventKind.MESSAGE and len(e.text) == 3000 for e in events)
    # Raw byte check: exactly `total` newline-terminated lines, no partial.
    raw = p.read_text()
    assert raw.count("\n") == total and raw.endswith("\n")

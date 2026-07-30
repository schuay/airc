# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Retention sweep tests.

The fixture is built through the code that actually writes the store (a real
Store, real Room.post), not by hand-crafted INSERTs: a hand-built row cannot
catch a schema or kind-value drift, which is exactly the failure mode a sweep
keyed on `kind` has.

Two of these are load-bearing and were each shown to fail with their guard
removed (see the comments on test_scrubbed_thread_injects_no_blank_lines and
test_system_messages_survive_scrub).
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from airc_room.prune import (
    Counts,
    aged_threads,
    delete_checkpoints,
    live_threads,
    parse_duration,
    redact_threads,
    vacuum,
)
from airc_room.room import Room
from airc_room.runner import build_turn_content
from airc_room.store import MessageKind, Store

OLD = 40 * 86400  # comfortably past the 30d default window


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "airc.db")
    yield s
    s.close()


def _age(store: Store, thread_id: int, seconds: float) -> None:
    """Backdate a thread's creation so the age predicate selects it. Threads are
    stamped with time.time() at create, so a test cannot otherwise produce one
    old enough without waiting."""
    store._db.execute(
        "UPDATE threads SET created = ? WHERE id = ?",
        (time.time() - seconds, thread_id),
    )
    store._db.commit()


async def _populate(store: Store, thread_id: int) -> None:
    """One message of every kind, through Room.post -- the real write path."""
    room = Room(store)
    await room.post(thread_id, "watcher", MessageKind.SYSTEM, "[v8] commit subject")
    await room.post(thread_id, "alice", MessageKind.HUMAN, "what broke here?")
    await room.post(thread_id, "compiler", MessageKind.AGENT, "the map got deprecated")
    await room.post(thread_id, "review", MessageKind.AGENT, "finding: OOB read")
    await room.post(thread_id, "perf", MessageKind.EVENT, "regression on js3")
    await room.post(thread_id, "icompleteu", MessageKind.NOTICE, "filed as b/123")
    await room.post(thread_id, "review", MessageKind.PING, "1234567890")


# ── the window predicate ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,seconds",
    [("48h", 172800), ("30d", 2592000), ("6w", 3628800), ("1d", 86400)],
)
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("bad", ["30", "30m", "d30", "", "-1d", "30 d"])
def test_parse_duration_rejects_ambiguous(bad):
    # A bare number is the dangerous one: seconds-vs-days silently purges
    # everything. Refuse rather than guess.
    with pytest.raises(Exception):
        parse_duration(bad)


async def test_aged_threads_respects_window(store):
    old = store.create_thread("old")
    new = store.create_thread("new")
    await _populate(store, old.id)
    await _populate(store, new.id)
    _age(store, old.id, OLD)
    assert aged_threads(store._db, time.time() - 30 * 86400) == [old.id]


async def test_aged_threads_skips_already_scrubbed(store):
    """A second sweep must not re-report a thread it already emptied, or the
    counts (and the audit note) become meaningless on a weekly timer."""
    t = store.create_thread("old")
    await _populate(store, t.id)
    _age(store, t.id, OLD)
    cutoff = time.time() - 30 * 86400
    assert aged_threads(store._db, cutoff) == [t.id]
    redact_threads(store._db, [t.id])
    assert aged_threads(store._db, cutoff) == []


async def test_aged_thread_with_only_system_messages_is_not_swept(store):
    # Nothing to redact: a commit thread nobody replied to.
    t = store.create_thread("bare")
    room = Room(store)
    await room.post(t.id, "watcher", MessageKind.SYSTEM, "[v8] subject")
    _age(store, t.id, OLD)
    assert aged_threads(store._db, time.time() - 30 * 86400) == []


# ── what survives ─────────────────────────────────────────────────────────────


async def test_system_messages_survive_scrub(store):
    """LOAD-BEARING. Verified to fail when the `kind != 'system'` guard is
    dropped from redact_threads' UPDATE (the digest is blanked too, leaving a
    thread with no intelligible record at all)."""
    t = store.create_thread("commit thread")
    await _populate(store, t.id)
    redact_threads(store._db, [t.id])

    msgs = store.thread_messages(t.id)
    kept = [m for m in msgs if m.kind == MessageKind.SYSTEM]
    assert len(kept) == 1
    assert kept[0].text == "[v8] commit subject"
    assert kept[0].sender == "watcher"
    # Everything else, findings (kind AGENT, sender "review") included.
    for m in msgs:
        if m.kind != MessageKind.SYSTEM:
            assert m.text == "" and m.sender == "", m.kind


async def test_findings_are_scrubbed_despite_being_automated(store):
    """Review findings post under kind AGENT, so a "keep the automated posts"
    rule phrased on sender would retain them. The one-kind rule must not."""
    t = store.create_thread("t")
    await _populate(store, t.id)
    redact_threads(store._db, [t.id])
    assert all(
        m.text == "" for m in store.thread_messages(t.id) if m.sender == "review"
    )


async def test_ping_gaia_id_is_scrubbed(store):
    """A PING carries a bare numeric Gaia id as its TEXT (it is the @mention key,
    so unlike an inbound sender id it cannot be hashed -- a hash notifies nobody).
    The sweep is therefore the only thing that removes it, which works because
    PING is not a retained kind. Guards that: were PING ever added to the retained
    set, corporate ids would start outliving the retention window."""
    t = store.create_thread("t")
    await _populate(store, t.id)
    redact_threads(store._db, [t.id])
    pings = [m for m in store.thread_messages(t.id) if m.kind == MessageKind.PING]
    assert pings and all(m.text == "" for m in pings)


async def test_hard_mode_deletes_rows_but_keeps_system(store):
    t = store.create_thread("t")
    await _populate(store, t.id)
    c = redact_threads(store._db, [t.id], hard=True)
    assert c.messages_deleted == 6 and c.messages_redacted == 0
    kinds = [m.kind for m in store.thread_messages(t.id)]
    assert kinds == [MessageKind.SYSTEM]


# ── the dedup keys, i.e. why this redacts instead of deleting ──────────────────


async def test_dedup_keys_survive_so_a_late_result_still_routes(store):
    """The reason the shape is redaction. An icompleteu result can arrive days
    after the thread aged out; with commit_threads intact it routes into the
    existing thread instead of announcing the commit a second time."""
    t = store.create_thread("commit")
    await _populate(store, t.id)
    store.set_commit_thread("deadbeef", t.id)
    store.link_chat_thread("spaces/A", "spaces/A/threads/T", t.id)
    store.mark_chat_message("spaces/A/messages/M1")
    store.mark_handover("job-1")
    store.set_announcement_meta(t.id, "v8", "/src/v8", "deadbeef")

    redact_threads(store._db, [t.id])

    assert store.commit_thread("deadbeef") == t.id
    assert store.chat_thread_id("spaces/A", "spaces/A/threads/T") == t.id
    assert store.chat_message_seen("spaces/A/messages/M1")
    assert store.handover_claimed("job-1")
    assert store.announcement_meta(t.id) is not None
    assert store.get_thread(t.id) is not None


async def test_commit_thread_title_kept_human_thread_blanked(store):
    """A commit thread's title is the commit subject (public git data, and what
    makes a scrubbed row auditable); a human-started thread's may be authored."""
    commit = store.create_thread("[v8] Fix a thing")
    human = store.create_thread("my private topic")
    await _populate(store, commit.id)
    await _populate(store, human.id)
    store.set_commit_thread("cafe", commit.id)

    redact_threads(store._db, [commit.id, human.id])

    assert store.get_thread(commit.id).title == "[v8] Fix a thing"
    assert store.get_thread(human.id).title == ""


async def test_pending_bug_is_unlinked_not_dropped(store):
    """The bug still wants filing; it only loses its follow-up destination."""
    t = store.create_thread("t")
    await _populate(store, t.id)
    store.enqueue_pending_bug(
        finding_id="f1",
        title="crash",
        body="...",
        repo="v8",
        commit_hash="cafe",
        author_email="dev@chromium.org",
        isolates=True,
        thread_id=t.id,
    )
    redact_threads(store._db, [t.id])
    rows = store.list_pending_bugs()
    assert len(rows) == 1 and rows[0]["thread_id"] is None


async def test_side_tables_are_dropped(store):
    t = store.create_thread("t")
    await _populate(store, t.id)
    store.record_headline(t.id, "spaces/A/messages/M1", "[v8] subject")
    store.bump_headline_finding(t.id, "f1", "high")
    store.add_pending_card(t.id, "compiler", "spaces/A", "spaces/A/messages/P1")
    store.add_timer(1, t.id, "compiler", time.time() + 60, "wake")
    store.bump_context_generation(t.id)

    redact_threads(store._db, [t.id])

    assert store.headline_for_compose(t.id) is None
    assert store.get_pending_card(t.id, "compiler") is None
    assert store.all_timers() == []
    assert store.context_generation(t.id) == 0


# ── the offsets: the silent failure this design exists to avoid ────────────────


async def test_scrubbed_thread_injects_no_blank_lines(store):
    """LOAD-BEARING. Verified to fail when the thread_seen_floor write is removed
    from redact_threads: the persona's offset stays 0, every redacted message is
    "unseen", and build_turn_content hands the model a wall of "[] " lines.

    The persona here has NO agent_seen row -- the case an UPDATE of agent_seen
    cannot fix, and the reason the floor is a per-thread row.
    """
    t = store.create_thread("t")
    await _populate(store, t.id)
    redact_threads(store._db, [t.id])

    seen = store.get_agent_seen(t.id, "compiler")
    unseen = [m for m in store.thread_messages(t.id) if m.id > seen]
    assert unseen == []

    content = build_turn_content(unseen)
    assert "[] " not in content

    # And a genuinely new message still gets through -- the floor must not
    # silence the thread permanently.
    room = Room(store)
    fresh = await room.post(t.id, "alice", MessageKind.HUMAN, "still here?")
    seen = store.get_agent_seen(t.id, "compiler")
    assert [m.id for m in store.thread_messages(t.id) if m.id > seen] == [fresh.id]


async def test_floor_does_not_lower_an_advanced_offset(store):
    """A persona that read past the tip keeps its own higher offset."""
    t = store.create_thread("t")
    await _populate(store, t.id)
    store.set_agent_seen(t.id, "compiler", 9999)
    redact_threads(store._db, [t.id])
    assert store.get_agent_seen(t.id, "compiler") == 9999


async def test_orchestrated_watermark_pinned_to_thread_tip(store):
    """Dropping it would make a running daemon re-route scrubbed history; only a
    restart's _recover would repair it, and a live process never runs that."""
    t = store.create_thread("t")
    await _populate(store, t.id)
    tip = store.thread_messages(t.id)[-1].id
    redact_threads(store._db, [t.id])
    assert store.get_orchestrated(t.id) == tip


async def test_offsets_pinned_per_thread_not_globally(store):
    """A global MAX(id) would pin an untouched thread's offset past its own
    messages, silencing a thread the sweep never selected."""
    old = store.create_thread("old")
    await _populate(store, old.id)
    new = store.create_thread("new")
    await _populate(store, new.id)
    _age(store, old.id, OLD)

    redact_threads(store._db, [old.id])

    old_tip = store.thread_messages(old.id)[-1].id
    assert store.get_agent_seen(old.id, "compiler") == old_tip
    # The untouched thread keeps a zero floor: its history is still readable.
    assert store.get_agent_seen(new.id, "compiler") == 0
    assert all(m.text for m in store.thread_messages(new.id))


# ── live-thread exclusion ─────────────────────────────────────────────────────


async def test_live_threads_from_pending_bug_and_timer(store):
    a, b, c = (store.create_thread(n) for n in ("a", "b", "c"))
    store.enqueue_pending_bug(
        finding_id="f1",
        title="t",
        body="b",
        repo="v8",
        commit_hash="cafe",
        author_email="d@e.com",
        isolates=None,
        thread_id=a.id,
    )
    store.add_timer(1, b.id, "compiler", time.time() + 60, "wake")
    assert live_threads(store._db, None) == {a.id, b.id}
    assert c.id not in live_threads(store._db, None)


async def test_live_threads_from_unterminated_icompleteu_job(store, tmp_path):
    import json

    root = tmp_path / "control"
    (root / "job-live").mkdir(parents=True)
    (root / "job-done").mkdir(parents=True)
    (root / "job-live" / "state.json").write_text(
        json.dumps({"step": "await-review", "job": {"thread_id": 7}})
    )
    (root / "job-done" / "state.json").write_text(
        json.dumps({"step": "done", "job": {"thread_id": 8}})
    )
    assert live_threads(store._db, root) == {7}


async def test_corrupt_job_state_does_not_fail_the_sweep(store, tmp_path):
    """Best-effort by design: this check reaches into another component's
    on-disk layout, and redaction (not the exclusion) is what makes a late
    result safe."""
    root = tmp_path / "control"
    (root / "job-bad").mkdir(parents=True)
    (root / "job-bad" / "state.json").write_text("{not json")
    assert live_threads(store._db, root) == set()
    assert live_threads(store._db, tmp_path / "missing") == set()


# ── checkpoints: 98% of the volume, and the same text a second time ────────────


def _ckpt_db(path):
    db = sqlite3.connect(str(path))
    # The real schema's shape for the two columns this touches; the saver creates
    # the full table itself at runtime.
    db.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint BLOB)")
    db.execute("CREATE TABLE writes (thread_id TEXT, value BLOB)")
    return db


def test_checkpoint_delete_covers_every_persona_and_generation(tmp_path):
    """The LangGraph key is "<thread>:<persona>:g<gen>", so the prefix match must
    take every persona and every context generation of the thread -- and nothing
    from a thread whose id merely starts with the same digits."""
    db = _ckpt_db(tmp_path / "c.db")
    for tid in ("1:compiler:g0", "1:compiler:g1", "1:runtime:g0", "12:compiler:g0"):
        db.execute("INSERT INTO checkpoints VALUES (?, ?)", (tid, b""))
        db.execute("INSERT INTO writes VALUES (?, ?)", (tid, b""))

    n_ckpt, n_writes = delete_checkpoints(db, [1])

    assert (n_ckpt, n_writes) == (3, 3)
    left = [r[0] for r in db.execute("SELECT thread_id FROM checkpoints")]
    assert left == ["12:compiler:g0"]
    db.close()


def test_vacuum_reclaims_freed_pages(tmp_path):
    """A skipped VACUUM is the failure mode that *looks* done: the rows are blank
    but the old text is still in the file's freelist."""
    path = tmp_path / "v.db"
    db = sqlite3.connect(str(path))
    db.execute("CREATE TABLE t (x TEXT)")
    db.executemany("INSERT INTO t VALUES (?)", [("x" * 4000,) for _ in range(200)])
    db.commit()
    before = path.stat().st_size
    db.execute("DELETE FROM t")
    db.commit()
    assert path.stat().st_size >= before  # freelist, not reclaimed
    vacuum(db)
    assert path.stat().st_size < before
    db.close()


def test_counts_summary_reports_the_right_verb():
    assert "redacted" in Counts(messages_redacted=3).summary()
    assert "deleted" in Counts(messages_deleted=3).summary()

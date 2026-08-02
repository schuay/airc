# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

import sqlite3

import pytest

from airc_room.store import MessageKind, Store


def make_store(tmp_path):
    return Store(tmp_path / "test.db")


def test_message_kind_coerces_and_round_trips(tmp_path):
    # add_message stores a bare string (SQLite TEXT) and reads it back as the
    # typed member; StrEnum keeps equality against the string, so old rows and
    # any string comparison still hold. This is the migration's safety net.
    s = make_store(tmp_path)
    t = s.create_thread("x")
    m = s.add_message(t.id, "perf:skiz", "event", "down 6%")
    assert m.kind is MessageKind.EVENT and m.kind == "event"
    (loaded,) = s.thread_messages(t.id)
    assert loaded.kind is MessageKind.EVENT  # coerced on read from the DB string


def test_chat_key_unique_stable_and_uuid_based(tmp_path):
    s = make_store(tmp_path)
    t1, t2 = s.create_thread("a"), s.create_thread("b")
    k1 = s.chat_key(t1.id)
    assert k1.startswith("airc-")
    assert k1 != f"airc-{t1.id}"  # uuid, not the row id
    assert s.chat_key(t1.id) != s.chat_key(t2.id)  # unique per thread
    assert s.chat_key(t1.id) == k1  # stable across reads
    with pytest.raises(KeyError):
        s.chat_key(9999)  # unknown thread is a bug, not a fallback


def test_chat_key_backfills_legacy_keys_on_migration(tmp_path):
    # A pre-migration DB: threads table without the chat_key column.
    path = tmp_path / "old.db"
    raw = sqlite3.connect(path)
    raw.execute(
        "CREATE TABLE threads (id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " title TEXT NOT NULL, created REAL NOT NULL)"
    )
    raw.execute("INSERT INTO threads (title, created) VALUES ('old', 0)")
    raw.execute("INSERT INTO threads (title, created) VALUES ('old2', 0)")
    raw.commit()
    raw.close()

    s = Store(path)
    # Existing threads keep the positional identity the Chat space already knows;
    # only new threads get a uuid key.
    assert s.chat_key(1) == "airc-1"
    assert s.chat_key(2) == "airc-2"
    new = s.create_thread("new")
    assert s.chat_key(new.id) != f"airc-{new.id}"  # uuid, no collision risk


def test_threads_and_messages(tmp_path):
    s = make_store(tmp_path)
    t = s.create_thread("main")
    assert s.get_thread(t.id) == t
    assert s.list_threads() == [t]

    m1 = s.add_message(t.id, "alice", "human", "hello")
    m2 = s.add_message(t.id, "perf", "agent", "hi")
    msgs = s.thread_messages(t.id)
    assert [m.id for m in msgs] == [m1.id, m2.id]
    assert msgs[0].sender == "alice"
    assert msgs[1].kind == "agent"


def test_agent_seen_offsets(tmp_path):
    s = make_store(tmp_path)
    t = s.create_thread("main")
    assert s.get_agent_seen(t.id, "perf") == 0
    s.set_agent_seen(t.id, "perf", 5)
    s.set_agent_seen(t.id, "perf", 9)
    assert s.get_agent_seen(t.id, "perf") == 9
    assert s.get_agent_seen(t.id, "compiler") == 0


def test_context_generation_bumps(tmp_path):
    s = make_store(tmp_path)
    t = s.create_thread("main")
    assert s.context_generation(t.id) == 0  # never compacted
    assert s.bump_context_generation(t.id) == 1
    assert s.context_generation(t.id) == 1
    assert s.bump_context_generation(t.id) == 2
    # Per-thread and independent.
    t2 = s.create_thread("other")
    assert s.context_generation(t2.id) == 0
    assert s.context_generation(t.id) == 2


def test_commit_thread_binding_is_stable(tmp_path):
    s = make_store(tmp_path)
    assert s.commit_thread("h1") is None
    s.set_commit_thread("h1", 7)
    assert s.commit_thread("h1") == 7
    s.set_commit_thread("h1", 9)  # first binding wins (INSERT OR IGNORE)
    assert s.commit_thread("h1") == 7


def test_source_notes(tmp_path):
    s = make_store(tmp_path)
    assert s.recent_source_notes("perf:skiz") == []
    for i in range(5):
        s.add_source_note("perf:skiz", f"note{i}")
    s.add_source_note("repo:v8", "other")
    # Oldest-first, scoped per source, honoring the limit.
    assert s.recent_source_notes("perf:skiz") == [f"note{i}" for i in range(5)]
    assert s.recent_source_notes("perf:skiz", 2) == ["note3", "note4"]
    assert s.recent_source_notes("repo:v8") == ["other"]


def test_chat_thread_mapping(tmp_path):
    s = make_store(tmp_path)
    space, ct = "spaces/AAA", "spaces/AAA/threads/T1"
    assert s.chat_thread_id(space, ct) is None
    s.link_chat_thread(space, ct, 7)
    assert s.chat_thread_id(space, ct) == 7
    # Idempotent upsert; other (space, thread) keys are independent.
    s.link_chat_thread(space, ct, 7)
    assert s.chat_thread_id(space, ct) == 7
    assert s.chat_thread_id(space, "spaces/AAA/threads/T2") is None
    assert s.chat_thread_id("spaces/BBB", ct) is None
    # Reverse lookup routes an outbound reply back to its originating Chat
    # thread; the earliest link wins over later (bot-created) ones.
    assert s.chat_thread_for_thread(7) == (space, ct)
    s.link_chat_thread(space, "spaces/AAA/threads/bot", 7)
    assert s.chat_thread_for_thread(7) == (space, ct)
    assert s.chat_thread_for_thread(999) is None


def test_pending_cards(tmp_path):
    s = make_store(tmp_path)
    assert s.get_pending_card(1, "perf") is None
    s.add_pending_card(1, "perf", "spaces/A", "spaces/A/messages/M1")
    s.add_pending_card(1, "compiler", "spaces/A", "spaces/A/messages/M2")
    assert s.get_pending_card(1, "perf") == "spaces/A/messages/M1"
    # Removal is by message name (the Chat-side resolve happens first, so a
    # newer card the same agent posted meanwhile must not be taken out).
    s.remove_pending_card("spaces/A/messages/M1")
    assert s.get_pending_card(1, "perf") is None
    s.remove_pending_card("spaces/A/messages/M1")  # idempotent
    # keyed by (thread, agent): same agent in another thread is independent.
    s.add_pending_card(2, "perf", "spaces/B", "spaces/B/messages/M3")
    assert {a for a, _, _ in s.all_pending_cards()} == {"compiler", "perf"}


def test_headline_compose_parts(tmp_path):
    s = make_store(tmp_path)
    assert s.set_headline_part(1, "tag", "opt") is None  # nothing recorded yet
    s.record_headline(1, "spaces/A/messages/M1", "*v8* ▸ thing")
    # First write of a part returns the row to re-render; tag and badge compose
    # independently.
    # Rows are (message_name, base_text, tag, badge, perf_regress, perf_improve).
    assert s.set_headline_part(1, "tag", "opt") == (
        "spaces/A/messages/M1",
        "*v8* ▸ thing",
        "opt",
        "",
        0,
        0,
    )
    # First-writer-wins: a second tag (or an at-least-once replay) is a no-op.
    assert s.set_headline_part(1, "tag", "other") is None
    assert s.set_headline_part(1, "badge", "🔴1") == (
        "spaces/A/messages/M1",
        "*v8* ▸ thing",
        "opt",
        "🔴1",
        0,
        0,
    )
    # A re-announce clears every part so they can be set again.
    s.record_headline(1, "spaces/A/messages/M2", "*v8* ▸ thing")
    assert s.set_headline_part(1, "badge", "🟠2") == (
        "spaces/A/messages/M2",
        "*v8* ▸ thing",
        "",
        "🟠2",
        0,
        0,
    )
    # Independent per thread.
    assert s.set_headline_part(2, "tag", "fix") is None


def test_bump_perf_accumulates(tmp_path):
    s = make_store(tmp_path)
    assert s.bump_perf(1, "regression") is None  # nothing recorded yet
    s.record_headline(1, "m", "*v8* ▸ thing")
    assert s.bump_perf(1, "regression")[4:] == (1, 0)  # perf_regress, perf_improve
    assert s.bump_perf(1, "regression")[4:] == (2, 0)  # accumulates, not first-wins
    assert s.bump_perf(1, "improvement")[4:] == (2, 1)
    # A re-announce resets the counters.
    s.record_headline(1, "m2", "*v8* ▸ thing")
    assert s.bump_perf(1, "improvement")[4:] == (0, 1)


def _counts(s, thread_id):
    return s.headline_for_compose(thread_id)[6]


def test_bump_headline_finding_dedups_and_accumulates(tmp_path):
    s = make_store(tmp_path)
    # No headline recorded: a clean no-op (no counter, no dedup row), so a
    # thread with no root announcement simply has no badge.
    assert s.bump_headline_finding(1, "f1", "high") is False
    s.record_headline(1, "m", "*v8* ▸ thing")
    assert s.bump_headline_finding(1, "f1", "high") is True
    assert _counts(s, 1) == {
        "blocker": 0,
        "high": 1,
        "medium": 0,
        "low": 0,
        "unknown": 0,
        "info": 0,
    }
    # A redelivery of the same finding (claim retry, topic replay) re-attempts
    # the bump but cannot double-count; another finding accumulates.
    assert s.bump_headline_finding(1, "f1", "high") is False
    assert s.bump_headline_finding(1, "f2", "info") is True
    assert _counts(s, 1)["high"] == 1
    assert _counts(s, 1)["info"] == 1
    # A re-announce resets the counters AND the dedup rows: the badge rebuilds.
    s.record_headline(1, "m2", "*v8* ▸ thing")
    assert s.bump_headline_finding(1, "f1", "high") is True
    assert _counts(s, 1) == {
        "blocker": 0,
        "high": 1,
        "medium": 0,
        "low": 0,
        "unknown": 0,
        "info": 0,
    }


def test_bump_headline_finding_info_upgrades_to_severity(tmp_path):
    s = make_store(tmp_path)
    s.record_headline(1, "m", "*v8* ▸ thing")
    # A failed repro counts as 'info'; a manual re-run that then verifies moves
    # the count to the severity bucket rather than double-counting.
    assert s.bump_headline_finding(1, "f1", "info") is True
    assert s.bump_headline_finding(1, "f1", "blocker") is True
    assert _counts(s, 1)["info"] == 0
    assert _counts(s, 1)["blocker"] == 1
    # Never the other way: a verified repro cannot be downgraded by a later
    # failure, and a severity-vs-severity bump keeps the first grade.
    assert s.bump_headline_finding(1, "f1", "info") is False
    assert s.bump_headline_finding(1, "f1", "low") is False
    assert _counts(s, 1)["blocker"] == 1
    import pytest

    with pytest.raises(ValueError):
        s.bump_headline_finding(1, "f2", "bogus")


def test_headline_compose_guard(tmp_path):
    s = make_store(tmp_path)
    s.record_headline(1, "spaces/A/messages/M1", "*v8* ▸ thing")
    row = s.headline_for_compose(1)
    assert row is not None
    name, base, tag, badge, regress, improve, counts, composed = row
    # composed is seeded with base_text at record time: a recompose with no
    # parts yet is a no-op, so no redundant Chat edit is issued.
    assert (name, base, tag, badge, regress, improve) == (
        "spaces/A/messages/M1",
        "*v8* ▸ thing",
        "",
        "",
        0,
        0,
    )
    assert composed == "*v8* ▸ thing"
    s.mark_headline_composed(1, "*v8* ▸ thing  *[opt]*")
    assert s.headline_for_compose(1)[7] == "*v8* ▸ thing  *[opt]*"
    assert s.headline_for_compose(2) is None


def test_chat_message_dedup(tmp_path):
    s = make_store(tmp_path)
    n = "spaces/AAA/messages/M1"
    assert s.chat_message_seen(n) is False  # read-only check, does not mark
    assert s.mark_chat_message(n) is True  # first delivery: process
    assert s.chat_message_seen(n) is True
    assert s.mark_chat_message(n) is False  # duplicate (bridge + space sub): skip
    assert s.mark_chat_message("spaces/AAA/messages/M2") is True


def test_get_pending_card_is_non_destructive(tmp_path):
    s = make_store(tmp_path)
    assert s.get_pending_card(1, "perf") is None
    s.add_pending_card(1, "perf", "spaces/A", "spaces/A/messages/M1")
    assert s.get_pending_card(1, "perf") == "spaces/A/messages/M1"
    assert s.get_pending_card(1, "perf") == "spaces/A/messages/M1"  # still there
    s.remove_pending_card("spaces/A/messages/M1")
    assert s.get_pending_card(1, "perf") is None


def test_chat_user_display_names(tmp_path):
    s = make_store(tmp_path)
    assert s.chat_user("users/42") is None
    s.set_chat_user("users/42", "Ada Lovelace")
    assert s.chat_user("users/42") == "Ada Lovelace"
    s.set_chat_user("users/42", "Ada L.")  # upsert keeps the latest
    assert s.chat_user("users/42") == "Ada L."


def test_pending_bug_queue_lifecycle(tmp_path):
    s = make_store(tmp_path)
    assert s.list_pending_bugs() == []
    assert s.pending_bug_exists("v8-abc-f1") is False
    s.enqueue_pending_bug(
        finding_id="v8-abc-f1",
        title="[v8] oob",
        body="repro...",
        repo="v8",
        commit_hash="abc123",
        author_email="a@example.com",
        isolates=True,
        thread_id=7,
    )
    assert s.pending_bug_exists("v8-abc-f1") is True
    (row,) = s.list_pending_bugs()
    assert row["finding_id"] == "v8-abc-f1"
    assert row["isolates"] is True  # 1 -> bool
    assert row["thread_id"] == 7
    assert row["attempts"] == 0
    s.bump_pending_bug("v8-abc-f1")
    assert s.list_pending_bugs()[0]["attempts"] == 1
    s.delete_pending_bug("v8-abc-f1")
    assert s.list_pending_bugs() == []


def test_pending_bug_enqueue_is_idempotent_and_preserves_attempts(tmp_path):
    s = make_store(tmp_path)
    s.enqueue_pending_bug(
        finding_id="v8-abc-f1",
        title="t",
        body="b",
        repo="v8",
        commit_hash="abc",
        author_email="",
        isolates=None,
        thread_id=None,
    )
    s.bump_pending_bug("v8-abc-f1")
    # A re-published result refreshes content but must not reset the attempt count
    # (a genuinely stuck row keeps aging toward its park cap) or duplicate the row.
    s.enqueue_pending_bug(
        finding_id="v8-abc-f1",
        title="t2",
        body="b2",
        repo="v8",
        commit_hash="abc",
        author_email="",
        isolates=None,
        thread_id=None,
    )
    (row,) = s.list_pending_bugs()
    assert row["title"] == "t2" and row["attempts"] == 1
    assert row["isolates"] is None


def test_result_delivery_is_recorded_after_the_fact_and_survives_reopen(tmp_path):
    # Marked after the post rather than claimed before it: a post that raises is
    # retried by the caller, and a claim taken up front would make that retry
    # skip the very post it is retrying.
    s = make_store(tmp_path)
    assert not s.result_delivered("v8-abc-repro-1")
    s.mark_result_delivered("v8-abc-repro-1")
    assert s.result_delivered("v8-abc-repro-1")
    s.mark_result_delivered("v8-abc-repro-1")  # replay: no error, no second row
    assert not s.result_delivered("v8-abc-repro-2")
    # Durable, because the redelivery it guards against is the one a restart
    # causes -- an in-memory set would be empty exactly when it is needed.
    assert Store(tmp_path / "test.db").result_delivered("v8-abc-repro-1")

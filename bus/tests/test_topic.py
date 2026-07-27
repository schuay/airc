# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Pub/sub topics: append-only log, contiguous seq, independent cursors."""

import pytest

from bus import Envelope, Subscription, Topic


def _env(n: int) -> Envelope:
    return Envelope(type="commit", payload={"n": n})


def test_publish_assigns_contiguous_seq(tmp_path):
    t = Topic(tmp_path, "repo", "v8")
    assert t.latest() == 0
    assert t.publish(_env(1)) == 1
    assert t.publish(_env(2)) == 2
    assert t.publish(_env(3)) == 3
    assert t.latest() == 3
    # Files are zero-padded so they sort lexicographically by seq.
    names = sorted(p.name for p in t.dir.iterdir())
    assert names == ["00000001.json", "00000002.json", "00000003.json"]


def test_read_round_trips(tmp_path):
    t = Topic(tmp_path, "repo", "v8")
    t.publish(_env(7))
    assert t.read(1).payload == {"n": 7}


def test_subscription_polls_only_new_and_acks(tmp_path):
    t = Topic(tmp_path, "perf", "pd_changepoints")
    for n in range(1, 4):
        t.publish(_env(n))
    sub = Subscription(tmp_path, "airc", "perf", "pd_changepoints")
    assert sub.cursor() == 0
    got = sub.poll()
    assert [seq for seq, _ in got] == [1, 2, 3]
    assert [e.payload["n"] for _, e in got] == [1, 2, 3]
    # Ack the first two; only the third remains.
    sub.ack(2)
    assert sub.cursor() == 2
    assert [seq for seq, _ in sub.poll()] == [3]
    # New message after the cursor shows up; already-acked ones do not.
    t.publish(_env(4))
    assert [seq for seq, _ in sub.poll()] == [3, 4]


def test_cursors_are_independent_per_subscriber(tmp_path):
    t = Topic(tmp_path, "repo", "v8")
    for n in range(1, 4):
        t.publish(_env(n))
    a = Subscription(tmp_path, "chat", "repo", "v8")
    b = Subscription(tmp_path, "review", "repo", "v8")
    a.ack(3)
    assert a.poll() == []
    # b has its own cursor and still sees everything.
    assert [seq for seq, _ in b.poll()] == [1, 2, 3]


def test_ack_never_moves_backwards_reset_does(tmp_path):
    t = Topic(tmp_path, "repo", "v8")
    for n in range(1, 4):
        t.publish(_env(n))
    sub = Subscription(tmp_path, "s", "repo", "v8")
    sub.ack(3)
    sub.ack(1)  # ignored
    assert sub.cursor() == 3
    sub.reset(0)  # replay from the start
    assert [seq for seq, _ in sub.poll()] == [1, 2, 3]


def test_tmp_files_are_ignored(tmp_path):
    t = Topic(tmp_path, "repo", "v8")
    t.publish(_env(1))
    (t.dir / ".tmp-stray.json").write_text("{}")  # a crashed publish's leftover
    assert t.latest() == 1
    sub = Subscription(tmp_path, "s", "repo", "v8")
    assert [seq for seq, _ in sub.poll()] == [1]


def test_invalid_name_segment_rejected(tmp_path):
    with pytest.raises(ValueError):
        Topic(tmp_path, "repo", "../escape")


def test_poll_quarantines_corrupt_file(tmp_path):
    # A corrupt/half-written file must not abort the whole poll and wedge every
    # later message; it is quarantined out of the seq namespace and the rest flow.
    t = Topic(tmp_path, "repo", "v8")
    t.publish(_env(1))
    t.publish(_env(2))
    t.publish(_env(3))
    (t.dir / "00000002.json").write_bytes(b"{ not valid json")
    sub = Subscription(tmp_path, "s", "repo", "v8")
    got = sub.poll()
    assert [seq for seq, _ in got] == [1, 3]  # 2 skipped, 1 and 3 delivered
    assert (t.dir / "00000002.json.bad").exists()
    assert not (t.dir / "00000002.json").exists()
    # And acking past it advances the cursor cleanly.
    for seq, _ in got:
        sub.ack(seq)
    assert sub.cursor() == 3

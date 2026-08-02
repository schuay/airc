# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Channel publish/claim semantics and the blob store.

Transport-only: these exercise claim/complete/fail/release and the blob store
with bare Envelopes, no typed payload. The domain payloads (JobSpec/...) moved to
airc_coding, and bus must not depend on a plugin -- a correlation_id on a plain
Envelope is all the channel semantics need.
"""

from bus import BlobStore, Channel, Envelope
from bus.channel import Claim


def _job(i: int) -> Envelope:
    # A minimal transport envelope standing in for any job message: the channel
    # only reads message_id/correlation_id, never the payload's shape.
    return Envelope(
        type="patch.job", payload={"job_id": f"j{i}"}, correlation_id=f"j{i}"
    )


def test_publish_then_claim_each_once(tmp_path):
    ch = Channel(tmp_path / "patch.jobs")
    id1 = ch.publish(_job(1))
    id2 = ch.publish(_job(2))

    c1 = ch.claim()
    c2 = ch.claim()
    assert c1 and c2
    # Each message is claimed exactly once (no duplicate delivery).
    assert {c1.env.message_id, c2.env.message_id} == {id1, id2}
    assert ch.claim() is None  # drained

    c1.complete()
    c2.fail()
    assert (ch.root / "done" / f"{c1.env.message_id}.json").exists()
    assert (ch.root / "failed" / f"{c2.env.message_id}.json").exists()


def test_double_resolution_is_tolerated_not_a_crash(tmp_path, caplog):
    # Two consumers (or two daemons) racing one bus: one resolves the claim,
    # the other's later complete()/fail() finds the message already gone. It
    # must warn and no-op, never raise a FileNotFoundError that escapes as an
    # unretrieved task exception in the consumer's event loop.
    import logging

    ch = Channel(tmp_path / "c")
    ch.publish(_job(1))
    # Two Claim handles for the same in-progress message (what two daemons see
    # after each adopts it via in_progress()).
    a = ch.claim()
    b = Claim(ch, a.path, a.env)
    a.complete()  # a wins
    with caplog.at_level(logging.WARNING, logger="bus.channel"):
        b.complete()  # b's file is gone -- tolerated
        b.fail()
    assert (ch.root / "done" / f"{a.env.message_id}.json").exists()
    assert "already gone" in caplog.text


def test_in_progress_readopts_unfinished(tmp_path):
    ch = Channel(tmp_path / "patch.jobs")
    ch.publish(_job(1))
    ch.publish(_job(2))
    a = ch.claim()  # moves one to in-progress, leaves it there
    assert a is not None
    readopted = ch.in_progress()
    assert [c.env.correlation_id for c in readopted] == [a.env.correlation_id]
    a.complete()
    assert ch.in_progress() == []


def test_release_requeues(tmp_path):
    ch = Channel(tmp_path / "c")
    ch.publish(_job(1))
    claim = ch.claim()
    assert claim is not None
    claim.release()
    again = ch.claim()  # back in incoming, claimable
    assert again is not None and again.env.correlation_id == "j1"


def test_partial_writes_are_invisible(tmp_path):
    # A file mid-write lives in tmp/, never incoming/, so pending() never sees it.
    ch = Channel(tmp_path / "c")
    ch.publish(_job(1))
    assert len(ch.pending()) == 1
    assert not any((ch.root / "tmp").iterdir())  # tmp drained after publish


def test_blob_roundtrip_is_content_addressed(tmp_path):
    bs = BlobStore(tmp_path / "blobs")
    ref = bs.put_text("a big diff")
    assert bs.get_text(ref) == "a big diff"
    assert ref == bs.put_text("a big diff")  # stable: same content, same ref
    assert ref != bs.put_text("other")


def test_an_unparseable_claim_is_quarantined_not_raised(tmp_path):
    # A consumer re-lists in-progress/ on every tick, so raising past one bad
    # file wedges the whole daemon: it crash-loops under systemd forever, no job
    # runs, and the first diagnostic (which reads the same directory) is down
    # too. Both shapes must be survivable -- a torn write, and a well-formed
    # JSON envelope from a producer whose schema has drifted.
    ch = Channel(tmp_path / "c")
    ch.publish(_job(1))
    ch.claim()  # a real claim, sitting in in-progress/
    (ch.root / "in-progress" / "torn.json").write_bytes(b'{"type": "patch.job"')
    (ch.root / "in-progress" / "drifted.json").write_text('{"not": "an envelope"}')

    claims = ch.in_progress()

    # The good claim still comes back...
    assert [c.env.correlation_id for c in claims] == ["j1"]
    # ...and the bad ones are moved aside for inspection, not deleted.
    assert not (ch.root / "in-progress" / "torn.json").exists()
    assert (ch.root / "in-progress" / "torn.json.bad").exists()
    assert (ch.root / "in-progress" / "drifted.json.bad").exists()


def test_quarantining_does_not_repeat_on_the_next_tick(tmp_path):
    # The bad file leaves the .json namespace, so a re-list neither re-reports
    # it nor re-renames it -- otherwise every tick would log the same error.
    ch = Channel(tmp_path / "c")
    (ch.root / "in-progress" / "torn.json").write_bytes(b"{")
    assert ch.in_progress() == []
    assert ch.in_progress() == []
    assert len(list((ch.root / "in-progress").glob("*.bad"))) == 1

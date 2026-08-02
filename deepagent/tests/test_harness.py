# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT


def _common(tmp_path):
    from airc_core.config import CommonConfig

    c = CommonConfig()
    c.models = {"default": "google_genai:test"}
    c.token_db_path = tmp_path / "tokens.db"
    return c


async def test_checkpoint_db_gives_a_durable_saver(tmp_path):
    # The feature only engages inside a running loop -- aiosqlite.connect needs
    # one -- and _saver's except-and-degrade would otherwise hide that: a
    # harness built in the wrong context silently keeps conversations in memory
    # and phase 5 buys nothing. Assert the type, not just the absence of a
    # crash. This test IS the async context; _graph_for is too.
    from deepagent import LangGraphHarness

    h = LangGraphHarness(_common(tmp_path), checkpoint_db=tmp_path / "cp.db")
    assert type(h._saver()).__name__ == "AsyncSqliteSaver"
    # One saver across threads: they are keyed internally, and a per-graph
    # connection would leak a file handle per LRU slot.
    assert h._saver() is h._saver()


async def test_no_checkpoint_db_stays_in_memory(tmp_path):
    from deepagent import LangGraphHarness

    h = LangGraphHarness(_common(tmp_path))
    assert type(h._saver()).__name__ == "InMemorySaver"


async def test_an_unusable_checkpoint_db_degrades(tmp_path):
    # Checkpointing is a cache: run_once re-sends the full prompt when a thread
    # is missing, so an unopenable DB must cost turns, never the job.
    from deepagent import LangGraphHarness

    h = LangGraphHarness(_common(tmp_path), checkpoint_db="/proc/nope/cp.db")
    assert type(h._saver()).__name__ == "InMemorySaver"


async def test_a_corrupt_checkpoint_db_degrades_rather_than_killing_the_goal(tmp_path):
    # aiosqlite.connect() is LAZY -- it returns before touching the file, and
    # the real open happens in the worker thread on the first await. So building
    # the saver proved nothing: a corrupt DB produced a working-looking
    # AsyncSqliteSaver that raised DatabaseError inside ainvoke, where run_once's
    # generic handler scores it a dead turn. Three of those and the goal abandons
    # as "no valid result after 3 dead attempts" -- checkpointing costing the
    # job, which is the opposite of the documented guarantee.
    from deepagent import LangGraphHarness

    bad = tmp_path / "cp.db"
    bad.write_bytes(b"this is not a sqlite database" * 200)
    h = LangGraphHarness(_common(tmp_path), checkpoint_db=bad)
    assert type(h._saver()).__name__ == "InMemorySaver"


async def test_a_read_only_checkpoint_dir_degrades(tmp_path):
    # The other half: a read-only mount OPENS fine and only fails when the saver
    # first writes, so the probe has to write too.
    from deepagent import LangGraphHarness

    d = tmp_path / "ro"
    d.mkdir()
    d.chmod(0o500)
    try:
        h = LangGraphHarness(_common(tmp_path), checkpoint_db=d / "cp.db")
        assert type(h._saver()).__name__ == "InMemorySaver"
    finally:
        d.chmod(0o700)  # so pytest can clean the tmpdir up


async def test_aclose_closes_the_saver_connection_and_reclaims_its_pages(tmp_path):
    # Two failures in one. aiosqlite's connection thread is NOT a daemon, so an
    # unclosed connection hangs interpreter exit -- the unit then dies to
    # TimeoutStopSec and SIGKILL leaves the WAL uncheckpointed. And `forget`
    # deletes rows without reclaiming pages: measured, a 240KB conversation left
    # a 3.7MB db + 4.2MB WAL that deleting every row did not shrink at all.
    import threading

    from deepagent import LangGraphHarness

    db = tmp_path / "cp.db"
    h = LangGraphHarness(_common(tmp_path), checkpoint_db=db)
    saver = h._saver()
    assert type(saver).__name__ == "AsyncSqliteSaver"
    # Force the lazy connect (awaiting the Connection starts its worker thread),
    # then leave a big dead table behind for the vacuum to reclaim.
    conn = await h._saver_conn
    await conn.execute("CREATE TABLE big (x BLOB)")
    await conn.executemany(
        "INSERT INTO big VALUES (?)", [(b"x" * 4000,) for _ in range(500)]
    )
    await conn.execute("DELETE FROM big")
    await conn.commit()
    before = db.stat().st_size
    assert before > 1_000_000, before

    await h.aclose()

    assert db.stat().st_size < before / 4
    assert h._saver_conn is None
    assert not [t for t in threading.enumerate() if "_connection_worker" in t.name]

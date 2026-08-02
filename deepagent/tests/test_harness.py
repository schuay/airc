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

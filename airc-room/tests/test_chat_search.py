from airc_room.chat_search import make_search_chat_tool
from airc_room.store import Store


def _db(tmp_path):
    """A store with two spaces, so space-scoping is exercised."""
    store = Store(tmp_path / "airc.db")
    a = store.create_thread("regexp perf")
    store.link_chat_thread("spaces/A", "chatA", a.id)
    store.add_message(a.id, "perf", "agent", "the regexp benchmark regressed 3% here")
    store.add_message(a.id, "sonic", "agent", "confirmed, bisecting the range now")
    b = store.create_thread("other room")
    store.link_chat_thread("spaces/B", "chatB", b.id)
    store.add_message(b.id, "max", "agent", "a regexp note in another space entirely")
    store.close()
    return str(tmp_path / "airc.db"), a.id, b.id


def _cfg(thread_id):
    return {"configurable": {"thread_id": f"{thread_id}:perf"}}


async def test_basic_match(tmp_path):
    path, a, _ = _db(tmp_path)
    tool = make_search_chat_tool(path)
    out = await tool.ainvoke({"pattern": "regress"}, config=_cfg(a))
    assert "regressed 3%" in out
    assert "newest first" in out


async def test_scopes_to_callers_space(tmp_path):
    path, a, b = _db(tmp_path)
    tool = make_search_chat_tool(path)
    # From space A, the space-B "regexp" message must not leak in.
    out = await tool.ainvoke({"pattern": "regexp"}, config=_cfg(a))
    assert "benchmark regressed" in out
    assert "another space" not in out
    # From space B, only B's message.
    out_b = await tool.ainvoke({"pattern": "regexp"}, config=_cfg(b))
    assert "another space" in out_b
    assert "benchmark regressed" not in out_b


async def test_no_space_context_spans_db(tmp_path):
    # No turn context (e.g. the console): search spans the whole db.
    path, _, _ = _db(tmp_path)
    tool = make_search_chat_tool(path)
    out = await tool.ainvoke({"pattern": "regexp"}, config={})
    assert "benchmark regressed" in out and "another space" in out


async def test_sender_filter_and_no_matches(tmp_path):
    path, a, _ = _db(tmp_path)
    tool = make_search_chat_tool(path)
    out = await tool.ainvoke({"pattern": ".", "sender": "sonic"}, config=_cfg(a))
    assert "bisecting" in out and "regressed" not in out
    assert "no matches" in await tool.ainvoke(
        {"pattern": "nonexistent-zzz"}, config=_cfg(a)
    )


async def test_invalid_pattern_reports(tmp_path):
    path, a, _ = _db(tmp_path)
    tool = make_search_chat_tool(path)
    out = await tool.ainvoke({"pattern": "("}, config=_cfg(a))
    assert "invalid pattern" in out


async def test_search_bounds_runaway_pattern(tmp_path, monkeypatch):
    # The regex scan is wall-clock bounded; once the deadline passes it returns
    # partial/no results with a note instead of pinning a worker. A negative
    # budget forces the deadline to be already spent on the first row.
    import airc_room.chat_search as cs

    monkeypatch.setattr(cs, "_SEARCH_TIMEOUT_S", -1.0)
    path, a, _ = _db(tmp_path)
    tool = make_search_chat_tool(path)
    out = await tool.ainvoke({"pattern": "regexp"}, config=_cfg(a))
    assert "timed out" in out

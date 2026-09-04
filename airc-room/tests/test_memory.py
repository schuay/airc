# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Core long-term memory: jail, auto-committing tools, derived index, injection."""

import os
import subprocess
from pathlib import Path

import pytest
from airc_room.memory import (
    MEMORY_GROUP,
    MEMORY_RULES,
    MemoryIndexMiddleware,
    make_memory_tools,
    memory_index,
)
from airc_room.memory.jail import Jailbreak, jail
from airc_room.memory.middleware import _REMINDER_TOKENS, _SRC
from airc_room.runner import build_turn_content
from langchain_core.messages import AIMessage, HumanMessage

_VALID = (
    "---\ntitle: T\ntype: user\ndate: 2026-07-23\n"
    "summary: prefers explicit type annotations\n---\n\nthe body\n"
)


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout


def _commit_count(root) -> int:
    """HEAD commit count, 0 before the first commit (unborn HEAD, where rev-list
    errors)."""
    r = subprocess.run(
        ["git", "-C", str(root), "rev-list", "--count", "HEAD"],
        capture_output=True,
        text=True,
        check=False,  # unborn HEAD errors; the docstring says that returns 0
    )
    return int(r.stdout.strip()) if r.returncode == 0 else 0


def _make_store(root: Path, *, schema_hook: bool = True) -> None:
    """A git repo standing in for a memory store; optionally with a pre-commit hook
    that requires a `summary:` line (the store schema gate, in miniature)."""
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    if schema_hook:
        hook = root / ".git" / "hooks" / "pre-commit"
        hook.write_text(
            "#!/bin/sh\n"
            "for f in $(git diff --cached --name-only --diff-filter=AM -- '*.md'); do\n"
            '  grep -q "^summary: " "$f" || { echo "schema: $f missing summary"; exit 1; }\n'
            "done\n"
        )
        os.chmod(hook, 0o755)


def _tools(root: Path) -> dict:
    return {t.name: t for t in make_memory_tools(root)}


# --- jail ------------------------------------------------------------------


def test_jail_confines_relative_and_blocks_escapes(tmp_path):
    assert jail(tmp_path, "a.md") == tmp_path / "a.md"
    # A not-yet-existing nested target still resolves (create must work).
    assert jail(tmp_path, "sub/new.md") == tmp_path / "sub" / "new.md"
    for escape in ("../out.md", "/etc/passwd", "sub/../../out.md"):
        with pytest.raises(Jailbreak):
            jail(tmp_path, escape)


def test_jail_blocks_symlink_out_of_tree(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (store / "link").symlink_to(outside)
    # A path through a symlink that leaves the tree is refused even though the
    # symlink's parent is inside the store.
    with pytest.raises(Jailbreak):
        jail(store, "link/secret.md")


# --- write / auto-commit / schema rejection --------------------------------


async def test_write_autocommits_a_valid_note(tmp_path):
    _make_store(tmp_path)
    tools = _tools(tmp_path)
    out = await tools["memory_write"].ainvoke(
        {"path": "note.md", "content": _VALID, "message": "add note"}
    )
    assert out == "saved note.md"
    assert _commit_count(tmp_path) == 1
    # nothing left staged or dirty
    assert _git(tmp_path, "status", "--porcelain").strip() == ""


async def test_schema_rejection_keeps_file_and_leaves_index_clean(tmp_path):
    _make_store(tmp_path)
    tools = _tools(tmp_path)
    out = await tools["memory_write"].ainvoke(
        {"path": "bad.md", "content": "no frontmatter\n", "message": "add bad"}
    )
    assert "rejected" in out
    assert "missing summary" in out  # the hook's own error reaches the agent
    # The file stays on disk so the agent can read+fix it...
    assert (tmp_path / "bad.md").exists()
    # ...but is NOT left staged, or it would ride the next unrelated commit.
    assert _git(tmp_path, "diff", "--cached", "--name-only").strip() == ""
    assert _commit_count(tmp_path) == 0


async def test_edit_autocommits_and_requires_exact_match(tmp_path):
    _make_store(tmp_path)
    tools = _tools(tmp_path)
    await tools["memory_write"].ainvoke(
        {"path": "note.md", "content": _VALID, "message": "add"}
    )
    # A fragment that is not a whole line does not match (line-oriented engine).
    miss = await tools["memory_edit"].ainvoke(
        {"path": "note.md", "search": "explicit type", "replace": "x", "message": "m"}
    )
    assert "failed to match" in miss
    assert _commit_count(tmp_path) == 1  # no commit
    # A full-line search matches and commits.
    hit = await tools["memory_edit"].ainvoke(
        {
            "path": "note.md",
            "search": "summary: prefers explicit type annotations",
            "replace": "summary: prefers concise names",
            "message": "refine",
        }
    )
    assert hit == "updated note.md"
    assert _commit_count(tmp_path) == 2


async def test_unrelated_staged_file_cannot_wedge_a_write(tmp_path):
    # The regression that motivated pathspec-scoped commits: a memory store is a
    # shared checkout, so an operator (or a persona with no delete tool) can leave
    # a malformed file staged. A plain `git commit` takes the whole index, so that
    # one file fails the hook for EVERY subsequent write -- which wedged a live
    # store for a week, blocking all persona writes and compaction. The agent's
    # own valid entry must commit regardless of what else sits in the index.
    _make_store(tmp_path)
    (tmp_path / "junk.md").write_text("no frontmatter, malformed\n")
    _git(tmp_path, "add", "--", "junk.md")

    tools = _tools(tmp_path)
    out = await tools["memory_write"].ainvoke(
        {"path": "note.md", "content": _VALID, "message": "add note"}
    )
    assert out == "saved note.md"
    assert _commit_count(tmp_path) == 1
    # Only the agent's file is in the commit, and the operator's staged work is
    # neither committed nor discarded -- just left as they left it.
    committed = _git(tmp_path, "show", "--name-only", "--format=", "HEAD").split()
    assert committed == ["note.md"]
    assert _git(tmp_path, "diff", "--cached", "--name-only").split() == ["junk.md"]


async def test_rejection_about_another_file_tells_the_agent_not_to_retry(tmp_path):
    # A hook is free to validate more than the committed path. When it fails on a
    # file the agent did not write, "fix the entry and write again" is actively
    # wrong -- the entry is fine and rewriting it can never clear the error. The
    # result must say so, or the agent loops on a file that was never the problem.
    _make_store(tmp_path, schema_hook=False)
    hook = tmp_path / ".git" / "hooks" / "pre-commit"
    # Validates the whole tree, not just what is staged.
    hook.write_text(
        "#!/bin/sh\n"
        "for f in $(ls *.md 2>/dev/null); do\n"
        '  grep -q "^summary: " "$f" || { echo "schema: $f missing summary"; exit 1; }\n'
        "done\n"
    )
    os.chmod(hook, 0o755)
    (tmp_path / "junk.md").write_text("malformed, not written by the agent\n")

    tools = _tools(tmp_path)
    out = await tools["memory_write"].ainvoke(
        {"path": "note.md", "content": _VALID, "message": "add note"}
    )
    assert "NOT about note.md" in out
    assert "Do not rewrite this entry" in out
    assert "junk.md" in out  # the hook's own text still comes through verbatim
    # The agent's entry survives on disk, ready to commit once the store is fixed.
    assert (tmp_path / "note.md").exists()


async def test_write_outside_the_jail_is_refused(tmp_path):
    _make_store(tmp_path)
    tools = _tools(tmp_path)
    out = await tools["memory_write"].ainvoke(
        {"path": "../escape.md", "content": _VALID, "message": "m"}
    )
    assert out.startswith("error")
    assert not (tmp_path.parent / "escape.md").exists()


# --- delete ----------------------------------------------------------------


async def test_delete_removes_the_entry_and_commits(tmp_path):
    _make_store(tmp_path)
    tools = _tools(tmp_path)
    await tools["memory_write"].ainvoke(
        {"path": "note.md", "content": _VALID, "message": "add"}
    )
    out = await tools["memory_delete"].ainvoke(
        {"path": "note.md", "message": "drop superseded note"}
    )
    assert out == "deleted note.md"
    assert not (tmp_path / "note.md").exists()
    assert _commit_count(tmp_path) == 2
    # The removal is committed, not merely staged, and leaves the tree clean.
    assert _git(tmp_path, "status", "--porcelain").strip() == ""
    # Recoverable: the content is still in history.
    assert "the body" in _git(tmp_path, "show", "HEAD~1:note.md")


async def test_delete_refuses_a_directory(tmp_path):
    # The guard that matters most on a tool an LLM holds: entries are files, and
    # `git rm -r` on the store root would be one call away from erasing the whole
    # memory. Nothing legitimate needs it.
    _make_store(tmp_path)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "keep.md").write_text(_VALID)
    tools = _tools(tmp_path)
    out = await tools["memory_delete"].ainvoke({"path": "sub", "message": "m"})
    assert out.startswith("error")
    assert (tmp_path / "sub" / "keep.md").exists()


async def test_delete_outside_the_jail_is_refused(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    _make_store(store)
    victim = tmp_path / "victim.md"
    victim.write_text("not the agent's to delete\n")
    tools = _tools(store)
    out = await tools["memory_delete"].ainvoke({"path": "../victim.md", "message": "m"})
    assert out.startswith("error")
    assert victim.exists()


async def test_delete_rejected_by_the_hook_says_how_to_restore(tmp_path):
    # A store may refuse removals on policy, and such a hook usually never names
    # the path -- which the write-side heuristic would report as "the errors are
    # NOT about your entry, it is saved on disk", about a file just unlinked. A
    # rejected delete gets its own message: still tracked, here is how to get it
    # back. The agent cannot rewrite content it no longer holds.
    _make_store(tmp_path, schema_hook=False)
    hook = tmp_path / ".git" / "hooks" / "pre-commit"
    hook.write_text(
        "#!/bin/sh\n"
        "git diff --cached --name-only --diff-filter=D | grep -q . && {\n"
        '  echo "policy: deletions are not allowed"; exit 1; }\n'
        "exit 0\n"
    )
    os.chmod(hook, 0o755)
    tools = _tools(tmp_path)
    await tools["memory_write"].ainvoke(
        {"path": "note.md", "content": _VALID, "message": "add"}
    )

    out = await tools["memory_delete"].ainvoke({"path": "note.md", "message": "drop"})
    assert "delete of note.md rejected" in out
    assert "git checkout -- note.md" in out
    assert "saved on disk" not in out  # the write-side wording must not leak in
    # Nothing left staged, so the removal is recoverable exactly as advertised.
    assert _git(tmp_path, "diff", "--cached", "--name-only").strip() == ""
    _git(tmp_path, "checkout", "--", "note.md")
    assert (tmp_path / "note.md").exists()


async def test_delete_of_a_missing_entry_says_so(tmp_path):
    # A model working from a stale index line should be told the entry is already
    # gone, not handed a raw git error to interpret.
    _make_store(tmp_path)
    tools = _tools(tmp_path)
    out = await tools["memory_delete"].ainvoke({"path": "ghost.md", "message": "m"})
    assert out.startswith("error")
    assert "does not exist" in out


# --- search ----------------------------------------------------------------


async def test_search_finds_committed_notes(tmp_path):
    _make_store(tmp_path)
    tools = _tools(tmp_path)
    await tools["memory_write"].ainvoke(
        {"path": "note.md", "content": _VALID, "message": "add"}
    )
    hit = await tools["memory_search"].ainvoke({"query": "explicit type"})
    assert "note.md" in hit
    miss = await tools["memory_search"].ainvoke({"query": "nonexistent-xyz"})
    assert "no matches" in miss


# --- derived index ---------------------------------------------------------


async def test_index_lists_only_committed_summaries(tmp_path):
    _make_store(tmp_path)
    tools = _tools(tmp_path)
    assert await memory_index(tmp_path) == ""  # empty store injects nothing
    await tools["memory_write"].ainvoke(
        {"path": "note.md", "content": _VALID, "message": "add"}
    )
    idx = await memory_index(tmp_path)
    assert idx == "- note.md -- prefers explicit type annotations"


async def test_index_empty_for_non_git_dir(tmp_path):
    # No repo: git grep fails; the caller must get "" (inject nothing), not raise.
    assert await memory_index(tmp_path) == ""


async def test_index_excludes_templates_and_underscore_files(tmp_path):
    # A template's placeholder summary is not a real entry; it must not leak into
    # the injected index (it would otherwise show on every turn).
    _make_store(tmp_path)
    tools = _tools(tmp_path)
    (tmp_path / "_templates").mkdir()
    (tmp_path / "_templates" / "user.md").write_text(
        "---\ntitle: T\ntype: user\ndate: 2026-07-23\nsummary: placeholder hook\n---\n"
    )
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "templates", "--no-verify")
    await tools["memory_write"].ainvoke(
        {"path": "note.md", "content": _VALID, "message": "add"}
    )
    idx = await memory_index(tmp_path)
    assert idx == "- note.md -- prefers explicit type annotations"


# --- injection into the per-turn tail --------------------------------------


def test_build_turn_content_never_carries_the_index():
    # The composed turn text must stay free of the block: folded in here it
    # would be indistinguishable from the turn around it, which is what made it
    # repeat every turn.
    body = build_turn_content([], now=None)
    assert "Current time:" in body
    assert "Memory (" not in body


def test_memory_group_and_rules_are_exported():
    assert MEMORY_GROUP == "memory"
    assert "memory_read" in MEMORY_RULES


# --- config ----------------------------------------------------------------


def _cfg_from_toml(tmp_path, body: str):
    from airc_room.config import load_config

    p = tmp_path / "airc.toml"
    p.write_text(body)
    return load_config(p)


def test_memory_config_defaults_off(tmp_path):
    cfg = _cfg_from_toml(tmp_path, "[airc]\n")
    assert cfg.memory.enabled is False
    assert cfg.memory.path is None


def test_memory_config_parses_enabled_path(tmp_path):
    cfg = _cfg_from_toml(
        tmp_path, f'[airc.memory]\nenabled = true\npath = "{tmp_path}/store"\n'
    )
    assert cfg.memory.enabled is True
    assert cfg.memory.path == tmp_path / "store"


def test_memory_enabled_without_path_is_a_config_error(tmp_path):
    with pytest.raises(SystemExit):
        _cfg_from_toml(tmp_path, "[airc.memory]\nenabled = true\n")


# --- placement: middleware decides from the conversation ---------------------

_IDX = "- note.md -- prefers explicit type annotations"


def _block(index: str = _IDX) -> HumanMessage:
    return HumanMessage(
        MemoryIndexMiddleware()._block(index), additional_kwargs={"lc_source": _SRC}
    )


def _decide(messages: list, index: str = _IDX):
    """What the middleware would add to state, given a history and an index."""
    return MemoryIndexMiddleware().before_model(
        {"messages": messages, "memory_index": index}, None
    )


def _injected(result) -> str:
    assert result is not None, "expected the index to be injected"
    return str(result["messages"][0].content)


def test_a_conversation_that_has_never_seen_the_index_gets_it():
    assert _IDX in _injected(_decide([HumanMessage("first turn")]))


def test_a_conversation_that_has_it_is_not_given_it_again():
    assert _decide([HumanMessage("first turn"), _block(), AIMessage("ok")]) is None


def test_a_changed_index_is_injected_again():
    grown = _IDX + "\n- other.md -- runs the suite first"
    assert "other.md" in _injected(_decide([_block(), AIMessage("ok")], grown))


def test_a_summarized_away_block_comes_back():
    # What the retired side-dict got wrong: a compaction replaces old history
    # with a summary, so the block is gone while the store is unchanged. Absence
    # is the trigger, so it returns on the next call with no state to reset.
    summarized = [
        HumanMessage("Summary of earlier conversation: they discussed X."),
        AIMessage("ok"),
    ]
    assert _IDX in _injected(_decide(summarized))


def test_a_buried_block_is_refreshed():
    # Still present, but far enough back that the persona has stopped seeing it.
    filler = AIMessage("x" * (_REMINDER_TOKENS * 4 + 100))
    assert _IDX in _injected(_decide([_block(), filler]))


def test_a_recent_block_is_not_refreshed():
    filler = AIMessage("x" * 400)
    assert _decide([_block(), filler]) is None


def test_no_index_no_block():
    # An empty store (or a persona without memory) contributes nothing.
    assert _decide([HumanMessage("first turn")], "") is None


async def test_dedup_holds_across_real_turns(tmp_path):
    # The unit tests above assert the decision; this asserts it survives a real
    # graph, checkpointer and state reducer -- that the block persists into the
    # checkpoint and is found again on the next turn.
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langgraph.checkpoint.memory import InMemorySaver

    model = GenericFakeChatModel(messages=iter([AIMessage("a"), AIMessage("b")]))
    graph = create_agent(
        model,
        tools=[],
        middleware=[MemoryIndexMiddleware()],
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "t1"}}

    await graph.ainvoke({"messages": [HumanMessage("one")], "memory_index": _IDX}, cfg)
    out = await graph.ainvoke(
        {"messages": [HumanMessage("two")], "memory_index": _IDX}, cfg
    )
    blocks = [
        m for m in out["messages"] if m.additional_kwargs.get("lc_source") == _SRC
    ]
    assert len(blocks) == 1, "the second turn must not repeat an unchanged index"


async def test_runner_hands_the_index_to_the_graph(tmp_path, monkeypatch):
    # The runner's half of the split: read the store once per turn and pass it
    # in. The key rides only for a memory-enabled persona -- a graph without the
    # middleware has no such state key to accept it.
    from airc_room.config import Config
    from airc_room.personas import Persona
    from airc_room.runner import AgentRunner, _AgentEntry, _TurnUsage
    from airc_room.store import Store

    root = tmp_path / "store"
    root.mkdir()
    _make_store(root)
    tools = _tools(root)
    await tools["memory_write"].ainvoke(
        {"path": "note.md", "content": _VALID, "message": "add"}
    )

    store = Store(tmp_path / "airc.db")
    thread = store.create_thread("t")
    store.add_message(thread.id, "human", "human", "hello")
    cfg = Config()
    cfg.token_db_path = tmp_path / "tokens.db"
    cfg.memory.enabled = True
    cfg.memory.path = root
    runner = AgentRunner(cfg, {}, object(), store)
    persona = Persona(
        name="Sonic",
        display_name="Sonic",
        description="d",
        system_prompt="",
        key="perf",
        tool_groups=(MEMORY_GROUP,),
    )
    plain = Persona(
        name="Tails",
        display_name="Tails",
        description="d",
        system_prompt="",
        key="docs",
    )
    runner._agents = {
        "Sonic": _AgentEntry(persona=persona, graph=object()),
        "Tails": _AgentEntry(persona=plain, graph=object()),
    }

    payloads: list[dict] = []

    async def _fake_stream(graph, agent_name, payload, config):
        payloads.append(payload)
        return "ok", _TurnUsage()

    monkeypatch.setattr(runner, "_stream", _fake_stream)
    await runner.run_turn("Sonic", thread.id, addressed=True)
    await runner.run_turn("Tails", thread.id, addressed=True)
    store.close()

    assert _IDX in payloads[0]["memory_index"]
    assert "memory_index" not in payloads[1]  # no middleware, no state key


def test_reminder_interval_stays_inside_the_compaction_keep_window():
    # The refresh interval is what keeps a copy of the block inside the tail a
    # summarization keeps, so absence stays a backstop rather than the usual
    # path. Reaching across packages for a private constant is the point: the
    # relationship is invisible from either side alone, and lowering
    # _SUMMARY_KEEP_TOKENS without looking here would degrade recall silently
    # (every long thread losing the index at each compaction, recovering only on
    # the next call) rather than fail anything.
    from airc_core.agent import _SUMMARY_KEEP_TOKENS

    assert _REMINDER_TOKENS < _SUMMARY_KEEP_TOKENS

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
    make_memory_tools,
    memory_index,
)
from airc_room.memory.jail import Jailbreak, jail
from airc_room.runner import build_turn_content

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


def test_build_turn_content_injects_index_after_time_line():
    body = build_turn_content([], memory_index="- note.md -- prefers X", now=None)
    assert "Current time:" in body
    assert "Memory (read a note" in body
    assert "- note.md -- prefers X" in body


def test_build_turn_content_omits_memory_when_absent():
    body = build_turn_content([], now=None)
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

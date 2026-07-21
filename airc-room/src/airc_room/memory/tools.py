"""Memory tools: the personas' hands on the store.

Local (non-MCP) langchain tools, granted to a persona via the reserved "memory"
tool_group like the room's timer/chat_search tools. They give an agent
read/search/write/edit over a git repo of markdown memory entries, HARD-JAILED to
the store root (see jail.py -- an LLM with write access must not reach outside it).

The file bodies (verbatim read, SEARCH/REPLACE edit, size limits) are REUSED from
airc-tools, not copied: airc_tools.resolve_path returns an absolute path as-is, so
we pre-resolve a jailed absolute path and hand it straight to the airc-tools
primitives. The memory-specific parts are the jail, git-grep recall, and
auto-commit through the store's schema hook.

Writes AUTO-COMMIT (there is no separate commit tool, and no shared dirty-set --
which in a toolset shared across personas would be cross-persona mutable state).
Each write/edit stages exactly its own path and commits it under a per-store lock,
so concurrent persona turns cannot race on the git index. A commit rejected by the
store's pre-commit schema hook is unstaged and its file left on disk, so the agent
can read, fix, and rewrite it -- the validator errors come back as the tool result.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from airc_tools.edit import apply_edits, write_file as _write_file
from airc_tools.read import read_file as _read_file
from langchain_core.tools import tool

from .jail import Jailbreak, jail

# Bound git-grep/commit output so a broad recall or a noisy commit cannot dominate
# a turn; the agent narrows the query for the rest.
_MAX_OUTPUT = 60_000
_GIT_TIMEOUT_S = 20.0


def _clip(text: str, limit: int = _MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... {len(text) - limit} bytes truncated ...]"


def _git_sync(root: Path, *args: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_S,
    )
    return proc.returncode, (proc.stdout + proc.stderr)


async def _git(root: Path, *args: str) -> tuple[int, str]:
    """Run a git command in the store root off the event loop, so a slow git call
    does not block the room."""
    return await asyncio.to_thread(_git_sync, root, *args)


def make_memory_tools(store_root: Path) -> list:
    """The memory tool set bound to a store checkout. Returned as a list the runner
    grants to a persona that lists the "memory" tool_group.

    Closes over the resolved root and a per-store commit lock; every path routes
    through jail(), confined to the root. A path escape comes back as a plain error
    string, never an exception into the turn."""
    root = store_root.expanduser().resolve()
    # Serializes write+stage+commit across concurrent persona turns sharing this
    # toolset, so two writers cannot collide on the git index (index.lock) or fold
    # one another's staged path into the wrong commit.
    commit_lock = asyncio.Lock()

    def _rel(abs_path: Path) -> str:
        """A store-relative path for messages and staging (the jailed absolute
        path is what the airc-tools primitives see; the agent thinks in relatives)."""
        return str(abs_path.relative_to(root))

    async def _commit_path(abs_path: Path, message: str) -> str | None:
        """Stage exactly `abs_path` and commit it with `message`, running the
        store's pre-commit schema hook. Returns None on success, or an error
        string (hook rejection or git failure) with the file left on disk and
        unstaged so the agent can fix and retry."""
        rel = _rel(abs_path)
        async with commit_lock:
            # -- ends the option list so a path that looks like a flag is still a
            # pathspec; `git add` stages a deletion too (a write that emptied the
            # file is covered).
            code, out = await _git(root, "add", "--", rel)
            if code != 0:
                return f"could not stage {rel}:\n{_clip(out)}"
            code, out = await _git(root, "commit", "-m", message)
            if code == 0:
                return None
            # The pre-commit schema hook (or another git error) rejected it. Unstage
            # so the bad entry does not ride along on the next unrelated commit; the
            # hook's per-file errors are in `out` for the agent to act on.
            await _git(root, "reset", "--quiet", "--", rel)
            return f"commit rejected (fix the entry and write again):\n{_clip(out)}"

    @tool
    async def memory_search(query: str, context: int = 0) -> str:
        """Search long-term memory for `query`, a regular expression, across every
        entry. Returns matching path:line results, with `context` lines around each
        match. Use this to recall what memory exists before answering, and to find
        an existing entry to update rather than creating a near-duplicate. Read a
        hit in full with memory_read."""
        args = ["grep", "-rInE", f"-C{max(0, int(context))}", "--", query]
        code, out = await _git(root, *args)
        if code not in (0, 1):  # 1 = no matches (not an error)
            return f"memory_search failed: {out.strip()}"
        return _clip(out) if out.strip() else f"no matches for {query!r} in memory."

    @tool
    async def memory_read(path: str, offset: int = 1, limit: int = 400) -> str:
        """Read a memory entry verbatim (path relative to the memory root, e.g.
        "prefers-explicit-types.md"). Output is exact file content, safe to paste
        into a memory_edit search. Advance a long read by raising `offset`."""
        try:
            target = jail(root, path)
        except Jailbreak as e:
            return f"error: {e}"
        return await asyncio.to_thread(_read_file, str(target), offset, limit)

    @tool
    async def memory_write(path: str, content: str, message: str) -> str:
        """Create or fully overwrite a memory entry with `content` (path relative to
        the memory root), then commit it with `message` (short, imperative, e.g.
        "record explicit-types preference"). Use for a NEW entry (copy a
        _templates/<type>.md shape so the frontmatter validates) or a full rewrite;
        for a partial change use memory_edit. The commit runs the store's schema
        hook: if the entry is malformed the commit is REJECTED and the validator
        errors come back here -- fix the entry and write again."""
        try:
            target = jail(root, path)
        except Jailbreak as e:
            return f"error: {e}"
        out = await asyncio.to_thread(_write_file, str(target), content)
        if out.startswith("error"):
            return out
        if err := await _commit_path(target, message):
            return err
        return f"saved {_rel(target)}"

    @tool
    async def memory_edit(path: str, search: str, replace: str, message: str) -> str:
        """Apply one exact SEARCH/REPLACE edit to a memory entry, then commit it
        with `message`. `search` must match the file byte-for-byte (whitespace
        included); keep it small with a few surrounding lines for a unique match.
        Nothing is written or committed on a mismatch -- the result names the
        failure so you can fix and resend. For a new file use memory_write. A
        schema-hook rejection comes back here to fix and retry."""
        try:
            target = jail(root, path)
        except Jailbreak as e:
            return f"error: {e}"
        out = await asyncio.to_thread(apply_edits, str(target), [(search, replace)])
        # apply_edits returns "applied N edit(s) to <path>" on success, a failure
        # report otherwise; only commit a real change.
        if not out.startswith("applied"):
            return out
        if err := await _commit_path(target, message):
            return err
        return f"updated {_rel(target)}"

    return [memory_search, memory_read, memory_write, memory_edit]

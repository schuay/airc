# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Memory tools: the personas' hands on the store.

Local (non-MCP) langchain tools, granted to a persona via the reserved "memory"
tool_group like the room's timer/chat_search tools. They give an agent
read/search/write/edit/delete over a git repo of markdown memory entries,
HARD-JAILED to the store root (see jail.py -- an LLM with write access must not
reach outside it).

The file bodies (verbatim read, SEARCH/REPLACE edit, size limits) are REUSED from
airc-tools, not copied: airc_tools.resolve_path returns an absolute path as-is, so
we pre-resolve a jailed absolute path and hand it straight to the airc-tools
primitives. The memory-specific parts are the jail, git-grep recall, and
auto-commit through the store's schema hook.

Writes AUTO-COMMIT (there is no separate commit tool, and no shared dirty-set --
which in a toolset shared across personas would be cross-persona mutable state).
Each write/edit stages exactly its own path and commits THAT PATH under a
per-store lock, so concurrent persona turns cannot race on the git index and an
unrelated file staged in the shared checkout cannot fail the hook for everyone
(see _commit_path). A commit rejected by the store's pre-commit schema hook is
unstaged and its file left on disk, so the agent can read, fix, and rewrite it --
the validator errors come back as the tool result, led by whether they concern
the agent's own entry or something else in the store.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from airc_tools.edit import apply_edits
from airc_tools.edit import write_file as _write_file
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


def _rejection(rel: str, hook_output: str, *, deleting: bool = False) -> str:
    """The tool result for a rejected commit, told from the agent's point of view.

    A hook reports per-file errors, and the agent's job is to know whether the
    complaint is about ITS entry (fix and rewrite) or about some other file (it
    cannot fix that, and retrying forever is the failure mode we actually saw).
    Scoping the commit means the hook is normally handed only this path, so the
    mixed case is rare -- but a hook free to validate the whole repo can still
    produce it, and that is exactly when the agent most needs to be told to stop.

    The distinction is drawn on the file's own path appearing in the output, which
    is a heuristic, not a parse: hook formats are store-specific and unknowable
    here. So the hook's full text is always passed through unchanged, and this
    only prepends a lead line -- a wrong guess costs a misleading sentence above
    the real errors, never a swallowed one.

    `deleting` exists because the heuristic reads backwards for a removal: a hook
    refusing a deletion on policy ("deletions are not allowed") often never names
    the path, which would otherwise be reported as "the errors are NOT about your
    entry -- it is saved on disk", about a file just unlinked. A rejected delete
    has one accurate thing to say regardless of whose fault it was, so say that.
    """
    if deleting:
        # _commit_path has already unstaged, so the removal is uncommitted; git
        # restores it. Naming the command matters -- the agent cannot rewrite
        # content it no longer holds.
        return (
            f"delete of {rel} rejected by the store; it is still tracked. Restore the"
            f" file with `git checkout -- {rel}` if you need it back, and report this"
            f" rather than retrying:\n{_clip(hook_output)}"
        )
    if rel in hook_output:
        lead = "commit rejected (fix the entry and write again)"
    else:
        # Nothing about this path: the store is failing for a reason outside this
        # write, so rewriting the entry cannot clear it. Say so, or the agent
        # burns its turn re-editing a file that was never the problem.
        lead = (
            f"commit rejected, but the errors are NOT about {rel} -- something else"
            " in the store is failing validation. Do not rewrite this entry: it is"
            " saved on disk and will commit once the store is repaired. Report this"
            " instead of retrying"
        )
    return f"{lead}:\n{_clip(hook_output)}"


def _git_sync(root: Path, *args: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_S,
        check=False,  # the caller inspects returncode; a raise would hide it
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

    async def _commit_path(
        abs_path: Path, message: str, *, deleting: bool = False
    ) -> str | None:
        """Commit exactly `abs_path` with `message`, running the store's pre-commit
        schema hook. Returns None on success, or an error string (hook rejection or
        git failure) with the file left on disk and unstaged so the agent can fix
        and retry.

        The commit is PATHSPEC-SCOPED (`git commit -- <path>`) rather than a plain
        commit of the index, because a store is a shared checkout an operator also
        touches by hand. A plain commit takes everything staged, so one unrelated
        malformed file sitting in the index fails the hook for EVERY write -- the
        agent is handed a rejection naming a file it never wrote, and no amount of
        fixing its own entry can clear it. That is not hypothetical: a stray
        frontmatter-less file wedged a live store for a week, blocking every
        persona write and every compaction. Scoping the commit makes a polluted
        index structurally unable to wedge the tools; the operator's staged work is
        left exactly as they left it.

        `git add` still runs first: the pathspec form commits the worktree state,
        and staging keeps a delete (a write that emptied the file) covered.
        """
        rel = _rel(abs_path)
        async with commit_lock:
            # -- ends the option list so a path that looks like a flag is still a
            # pathspec; `git add` stages a deletion too (a write that emptied the
            # file is covered).
            code, out = await _git(root, "add", "--", rel)
            if code != 0:
                return f"could not stage {rel}:\n{_clip(out)}"
            code, out = await _git(root, "commit", "-m", message, "--", rel)
            if code == 0:
                return None
            # The pre-commit schema hook (or another git error) rejected it. Unstage
            # so the bad entry does not ride along on the next unrelated commit; the
            # hook's per-file errors are in `out` for the agent to act on.
            await _git(root, "reset", "--quiet", "--", rel)
            return _rejection(rel, out, deleting=deleting)

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

    @tool
    async def memory_delete(path: str, message: str) -> str:
        """Delete a memory entry that is wrong or obsolete, committing the removal
        with `message` (e.g. "drop superseded bagel note"). Use when a fact stopped
        being true and no rewrite makes sense -- if it merely CHANGED, prefer
        memory_edit or memory_write so the note keeps its history in one place.
        The entry stays recoverable from git history after deletion."""
        try:
            target = jail(root, path)
        except Jailbreak as e:
            return f"error: {e}"
        rel = _rel(target)
        # Refuse a directory outright. `git rm -r` on the store root would be one
        # tool call away from erasing the whole memory, and no legitimate use needs
        # it: entries are files. Checked before existence so the message is about
        # the real problem.
        if target.is_dir():
            return f"error: {rel} is a directory; memory_delete removes one entry"
        if not target.exists():
            # A model that misremembers a path from the index should be told the
            # entry is already gone, not handed a git error to interpret.
            return f"error: {rel} does not exist (nothing to delete)"

        def _unlink() -> None:
            target.unlink()

        await asyncio.to_thread(_unlink)
        # Commit through the same scoped path as a write: `git add` stages the
        # deletion, and _commit_path's pathspec commit records it. A hook that
        # rejects the removal leaves the file deleted on disk but uncommitted,
        # which _commit_path unstages -- so the entry is restorable with git
        # checkout and the store is never left half-changed in the index.
        if err := await _commit_path(target, message, deleting=True):
            return err
        return f"deleted {rel}"

    return [memory_search, memory_read, memory_write, memory_edit, memory_delete]

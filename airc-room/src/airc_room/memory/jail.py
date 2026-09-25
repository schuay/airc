# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Path confinement for the memory tools -- the security boundary.

The memory tools give an LLM read/write over a git repo of markdown entries (the
room's autonomous long-term memory). They run in-process with no sandbox, so a
write must be provably unable to touch anything outside the store: this module is
that proof.

`jail(root, path)` resolves any path -- relative or absolute, existing or not --
against `root` and asserts the result stays inside it after full symlink
resolution. A `..` segment, an absolute path elsewhere, or a symlink pointing out
of the tree all raise Jailbreak, which the tool wrappers turn into a plain error
string (never an exception into the turn).

The root is a parameter, not a module global, so multiple stores can coexist in
one process (a coding room and a grocery room, or a future per-space store) with
no shared mutable state.

Containment in the root is enough for reads but not for writes: the root also
holds .git, the schema hook, and the validator the hook runs, so a write there
executes on the next commit. `jail_entry` narrows a write to an entry file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# A lowercase .md file directly in the store root. The leading character keeps
# out dotfiles, _templates, and the uppercase meta docs (AGENTS.md, README.md);
# no separators keeps out every subdirectory, including a nested .git.
_ENTRY_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*\.md")


class Jailbreak(Exception):
    """A path resolved outside its memory-store root."""


def jail(root: Path, path: str) -> Path:
    """Resolve `path` under `root`, or raise Jailbreak if it escapes.

    A relative path is taken under `root`; an absolute path must already be
    inside it. The result is fully resolved (symlinks included) before the
    containment check, so neither `..` segments nor a symlink out of the tree can
    escape. A not-yet-existing target (a new entry) resolves via its nearest
    existing parent, so create still works while staying jailed.
    """
    root = root.resolve()
    p = Path(path)
    candidate = p if p.is_absolute() else root / p
    resolved = _resolve_allowing_missing(candidate)
    if resolved != root and root not in resolved.parents:
        raise Jailbreak(f"path {path!r} resolves outside the memory store ({root})")
    return resolved


def jail_entry(root: Path, path: str) -> Path:
    """Resolve `path` like jail(), and also require an entry file: a name
    matching _ENTRY_NAME directly in `root`. Checked on the resolved path, so an
    entry-named symlink into the store's machinery is refused too."""
    resolved = jail(root, path)
    if resolved.parent != root.resolve() or not _ENTRY_NAME.fullmatch(resolved.name):
        raise Jailbreak(
            f"path {path!r} is not a memory entry: use a lowercase name like"
            " prefers-explicit-types.md, directly in the store root"
        )
    # Writes go through the file in place. A symlink left after resolution is a
    # loop, and a second hard link shares its inode with a file elsewhere, so a
    # write to either would land somewhere other than this entry.
    if resolved.is_symlink():
        raise Jailbreak(f"path {path!r} is a symlink loop")
    if resolved.is_file() and resolved.stat().st_nlink > 1:
        raise Jailbreak(f"path {path!r} is hard-linked to another file")
    return resolved


def _resolve_allowing_missing(p: Path) -> Path:
    """Fully resolve `p` even if it does not exist yet: resolve the longest
    existing prefix (following symlinks), then re-attach the missing tail. This
    keeps a create path jailed -- a symlinked existing parent cannot smuggle the
    target out of the tree -- without requiring the file to exist."""
    existing = p
    tail: list[str] = []
    # lexists, not exists: a dangling symlink must be resolved through to its
    # target, or it passes as a plain name and a write creates the target.
    while not os.path.lexists(existing):
        tail.append(existing.name)
        parent = existing.parent
        if parent == existing:  # reached the filesystem root
            break
        existing = parent
    base = existing.resolve()
    for name in reversed(tail):
        base = base / name
    # Collapse any `..` in the non-existent tail lexically. The existing prefix is
    # already symlink-resolved and the tail has no symlinks (its components do not
    # exist), so lexical normalization equals the real path the OS would create --
    # and without it a `..` in the tail (e.g. "sub/../../x.md") would survive into
    # the containment check and read as inside the root while resolving outside it.
    return Path(os.path.normpath(base))

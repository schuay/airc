# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Path confinement for the memory tools -- the security boundary.

The memory tools give an LLM read/write over a git repo of markdown entries (the
room's autonomous long-term memory). They run IN-PROCESS with no sandbox, so a
write must be provably unable to touch anything outside the store: this module is
that proof.

`jail(root, path)` resolves any path -- relative or absolute, existing or not --
against `root` and asserts the result stays inside it after full symlink
resolution. A `..` segment, an absolute path elsewhere, or a symlink pointing out
of the tree all raise Jailbreak, which the tool wrappers turn into a plain error
string (never an exception into the turn).

Unlike airc-home's original (a `set_root` module global), the root is a PARAMETER
-- so multiple stores can coexist in one process (a coding room and a grocery
room, or a future per-space store) with no shared mutable state.
"""

from __future__ import annotations

import os
from pathlib import Path


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


def _resolve_allowing_missing(p: Path) -> Path:
    """Fully resolve `p` even if it does not exist yet: resolve the longest
    existing prefix (following symlinks), then re-attach the missing tail. This
    keeps a create path jailed -- a symlinked existing parent cannot smuggle the
    target out of the tree -- without requiring the file to exist."""
    existing = p
    tail: list[str] = []
    while not existing.exists():
        tail.append(existing.name)
        parent = existing.parent
        if parent == existing:  # reached the filesystem root
            break
        existing = parent
    base = existing.resolve()
    for name in reversed(tail):
        base = base / name
    # Collapse any `..` in the non-existent tail LEXICALLY. The existing prefix is
    # already symlink-resolved and the tail has no symlinks (its components do not
    # exist), so lexical normalization equals the real path the OS would create --
    # and without it a `..` in the tail (e.g. "sub/../../x.md") would survive into
    # the containment check and read as inside the root while resolving outside it.
    return Path(os.path.normpath(base))

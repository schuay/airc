# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Derive the memory index for per-turn injection.

There is deliberately NO stored index file (no MEMORY.md to drift). The index is
produced on demand by one `git grep` over the entries' `summary:` frontmatter --
so it is complete and fresh by construction: the store's schema hook REQUIRES a
`summary`, auto-commit means every entry is committed, and git grep scans tracked
files, so every entry's hook is always present the moment it lands.

The injected block is a table of contents (path + one-line hook), not the
content: a persona reads a full entry with memory_read when a line looks
relevant. Cheap enough (one subprocess, threaded) to run every memory-enabled
persona turn against a small store; when it stops being cheap the block clips,
which is the documented signal to graduate to query-keyed retrieval.

The grep runs every turn; whether its result is put in front of the persona is
MemoryIndexMiddleware's decision, made from the conversation's own history.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

# Bound the injected block so a store that has grown large cannot dominate every
# turn's tail. Hitting this cap is the tripwire to move off "inject the whole
# index" onto query-keyed retrieval.
_MAX_INDEX_BYTES = 20_000
_GIT_TIMEOUT_S = 10.0


async def memory_index(root: Path) -> str:
    """The memory table of contents: one `- <path> -- <hook>` line per entry that
    carries a `summary:` frontmatter field. Empty string when the store has no
    such entries (a fresh store, or one git cannot read) -- the caller then
    injects nothing."""
    return await asyncio.to_thread(_build, root)


def _build(root: Path) -> str:
    try:
        proc = subprocess.run(
            # `path:lineno:summary: <hook>` per match; the path is the entry, the
            # trailing text is its one-line index hook. Templates and any `_`/`.`
            # prefixed file are excluded via pathspec -- their placeholder summaries
            # are not real entries and must not leak into the injected index.
            [
                "git",
                "-C",
                str(root),
                "grep",
                "--no-color",
                "-nI",
                "-e",
                "^summary: ",
                "--",
                "*.md",
                ":(exclude)_*",
                ":(exclude).*",
                ":(exclude)**/_*",
            ],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,  # a grep miss is an empty index, not an error
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    # rc 1 = no matches (empty store), anything else = a real git error; either
    # way there is nothing to inject.
    if proc.returncode not in (0, 1) or not proc.stdout.strip():
        return ""

    lines: list[str] = []
    for raw in proc.stdout.splitlines():
        # `path:lineno:summary: <hook>` -- split off path and lineno, then the
        # `summary: ` label, leaving the hook. A malformed line is skipped rather
        # than guessed at.
        parts = raw.split(":", 2)
        if len(parts) != 3:
            continue
        path, _lineno, field = parts
        hook = field.partition("summary: ")[2].strip().strip("\"'")
        if hook:
            lines.append(f"- {path} -- {hook}")

    if not lines:
        return ""
    lines.sort()
    body = "\n".join(lines)
    if len(body) > _MAX_INDEX_BYTES:
        body = body[:_MAX_INDEX_BYTES] + "\n[... memory index truncated ...]"
    return body

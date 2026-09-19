# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Run-artifact logging: write human-readable markdown/text records to disk.

A shared seam for any component -- a processor, a watcher, anything -- that wants
its reasoning to survive outside chat (an eval trail, a debugging record). The
caller owns the content and a stable key; this owns the folder layout, the dated
slugged filename, and the best-effort write. Logging must never sink the work it
traces, so a write failure is logged and swallowed.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path

log = logging.getLogger(__name__)


def slug(text: str, maxlen: int = 60) -> str:
    """Filesystem-safe slug: lowercased, non-alphanumeric runs collapsed to a
    single hyphen, trimmed. Empty input yields a stable placeholder so a file is
    still written."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:maxlen].strip("-") or "untitled"


class ArtifactLog:
    """Best-effort writer for run artifacts under a root dir, namespaced by
    category (a subfolder, e.g. "reviews"). Stateless beyond the root, so a
    component may hold its own instance; disabled (a no-op) when root is None."""

    def __init__(self, root: Path | str | None) -> None:
        self._root = Path(root).expanduser() if root else None

    @property
    def enabled(self) -> bool:
        return self._root is not None

    def exists(self, category: str, key: str, ext: str = "md") -> bool:
        """Whether an artifact for this key was already written (any date).

        The filename embeds the write date, so this globs across dates -- a
        consumer uses it as an idempotency guard (skip work already recorded)
        independent of when the first run happened. Always False when disabled,
        so a disabled log never reports phantom artifacts."""
        if self._root is None:
            return False
        return any((self._root / category).glob(f"*-{slug(key)}.{ext}"))

    def write_now(
        self, category: str, key: str, text: str, ext: str = "md"
    ) -> Path | None:
        """`write`, synchronously. For a caller that cannot await: the one real
        case is a cancellation handler preserving its artifact during shutdown,
        where an await invites a second cancel.

        The write is a single small file, so blocking a caller that could await
        costs it nothing measurable; `write` exists for tidiness.
        """
        if self._root is None:
            return None
        try:
            folder = self._root / category
            folder.mkdir(parents=True, exist_ok=True)
            name = f"{time.strftime('%Y-%m-%d')}-{slug(key)}.{ext}"
            path = folder / name
            # Always UTF-8: model output and code carry non-ASCII, and a service
            # without LANG set defaults write_text to the locale (often ASCII),
            # which would raise UnicodeEncodeError on the first such char.
            path.write_text(text, encoding="utf-8")
        except Exception as e:
            # Best-effort trace: any failure (disk full, bad path, an encode
            # error) is logged and swallowed -- it must never sink, or re-loop,
            # the work it traces. Broad because a non-OSError here (e.g.
            # UnicodeEncodeError, a ValueError) used to escape and fail the turn.
            log.warning("artifacts: %s/%s: not written: %s", category, key, e)
            return None
        log.info("artifacts: wrote %s", path)
        return path

    async def write(
        self, category: str, key: str, text: str, ext: str = "md"
    ) -> Path | None:
        """Write `text` to <root>/<category>/<date>-<slug(key)>.<ext>. A same-day
        re-run with the same key overwrites (idempotent, latest wins).

        Returns the path written, or None when disabled or the write failed. A
        caller that only wants the trace can ignore it; one whose artifact is the
        recovery path for work it is about to abandon has to be able to name the
        file in the message telling a human to go read it -- and to say so
        loudly when there is no file to name.
        """
        return await asyncio.to_thread(self.write_now, category, key, text, ext)

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""read_file: a verbatim, range-limited, byte-capped read.

Verbatim and gutter-free by design. edit_file matches SEARCH text against the
file byte-for-byte, so read output must be safe to paste straight into a search
field. A line-number gutter would have to be stripped perfectly every time or
matches fail -- and search/replace is content-addressed anyway; it finds the
text, line numbers are irrelevant to the apply. So the line range goes in a
header, never in the body. `rg -n` via shell is there when a number is wanted.
"""

from .limits import MAX_READ_BYTES, MAX_READ_LINES, MAX_READ_SCAN, cap_head
from .paths import io_guarded, resolve_path


@io_guarded
def read_file(path: str, offset: int = 1, limit: int = MAX_READ_LINES) -> str:
    p = resolve_path(path)
    if not p.exists():
        return f"error: {path} does not exist"
    if p.is_dir():
        return f"error: {path} is a directory; use `ls`/`rg --files` via shell"
    if not p.is_file():
        # A FIFO/socket/device: open+read would block a worker thread forever
        # (uncancellably -- it runs via to_thread), and a few of those wedge
        # the whole tool server.
        return f"error: {path} is not a regular file"

    offset = max(1, offset)
    limit = max(1, min(limit, MAX_READ_LINES))

    with p.open("rb") as f:
        raw = f.read(MAX_READ_SCAN)
    scanned_all = len(raw) < MAX_READ_SCAN
    lines = raw.decode("utf-8", errors="replace").splitlines(keepends=True)
    total = len(lines)

    start = offset - 1
    if start >= total:
        tail = "" if scanned_all else " (file truncated at scan limit)"
        return f"{path} has {total} lines; offset {offset} is past the end{tail}"

    end = min(total, start + limit)
    body, dropped = cap_head("".join(lines[start:end]), MAX_READ_BYTES)
    if dropped:
        # Byte cap hit mid-window: report the last whole line actually kept, so
        # the model knows where to resume with a higher offset.
        end = start + body.count("\n")
        note = f" (byte cap: {dropped} bytes past line {end} dropped; re-read from {end + 1})"
    else:
        note = "" if scanned_all else " (file truncated at scan limit)"

    return f"{path} lines {offset}-{end} of {total}{note}\n{body}"

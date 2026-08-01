# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Byte-limit helpers shared by all three tools.

Every tool result crosses back into an LLM context window, so each is capped
before return -- one runaway `cat` of a megabyte file, an infinite `yes`, or a
giant edit-failure echo must not blow the turn (or, for shell, OOM the server
while buffering). The caps and the one truncation-marker style live here so the
three tools share a single policy.
"""

# Displayed shell output (head+tail). Well under airc_core's ~200k result cap so
# one command can't dominate a turn; the model is told to pipe rg/tail itself.
MAX_SHELL_OUTPUT = 2_000
# Hard ceiling on bytes buffered from a child before we kill it. Bounds memory
# against unbounded producers (`yes`, `cat /dev/zero`); we keep the head and drop
# the rest rather than read-and-discard forever.
MAX_SHELL_CAPTURE = 400_000

# read_file: bytes returned per call, plus the default line window. The byte cap
# is the real guard (one 5MB minified line would beat a line count); the line
# window is the ergonomic default.
MAX_READ_BYTES = 100_000
MAX_READ_LINES = 2000

# Never pull more than this off disk for one read: a stray multi-GB file must not
# OOM the server. We read a prefix and say the file was truncated at the scan.
MAX_READ_SCAN = 20_000_000

# edit_file: refuse files above this. Matching against a giant buffer is
# pathological, and legitimate source files sit far below it.
MAX_EDIT_FILE_BYTES = 5_000_000
# Cap on a whole edit-failure message and on each echoed block within it, so a
# failed edit against a large file returns actionable text, not the file back.
MAX_FAILURE_TEXT = 20_000
MAX_ECHO_BLOCK = 4_000


def head_tail(text: str, limit: int = MAX_SHELL_OUTPUT) -> str:
    """Keep the head and tail of `text`, eliding the middle with a byte/line
    count. Both ends matter for command output: the head has the start (the
    command, first errors), the tail has the exit summary a truncation-from-the-
    front would drop."""
    if len(text) <= limit:
        return text
    head_len = limit * 3 // 4
    tail_len = limit - head_len
    elided = len(text) - head_len - tail_len
    elided_lines = text.count("\n", head_len, len(text) - tail_len)
    return (
        f"{text[:head_len]}\n"
        f"[... {elided} bytes / {elided_lines} lines elided;"
        f" re-run piping through rg/tail to narrow ...]\n"
        f"{text[-tail_len:]}"
    )


def cap_head(text: str, limit: int) -> tuple[str, int]:
    """Truncate to the first `limit` bytes. Returns (kept, dropped). Head-only:
    a read starts at a chosen offset, so the meaningful part is the front and the
    model advances by re-reading at a higher offset."""
    if len(text) <= limit:
        return text, 0
    return text[:limit], len(text) - limit


def clip(text: str, limit: int = MAX_ECHO_BLOCK) -> str:
    """Shorten one echoed SEARCH/REPLACE block inside a failure message."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... {len(text) - limit} more bytes ...]\n"

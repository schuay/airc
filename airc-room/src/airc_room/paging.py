# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Lossless paging for text delivered by chat transports."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256

DEFAULT_PAGE_LIMIT = 3800
_FENCE_OVERHEAD = len("```\n") + len("\n```")


def part_marker(part: int, total: int) -> str:
    return f"_(part {part}/{total})_"


def page_token(scope: str, message_id: int, part: int) -> str:
    """A deterministic transport id for an idempotently retried page."""
    material = f"{scope}\0{message_id}\0{part}".encode()
    return sha256(material).hexdigest()


def _cut(text: str, limit: int) -> int:
    """Choose a nonempty prefix no longer than limit, preferring clean breaks."""
    if len(text) <= limit:
        return len(text)
    # The break must END within the budget, not start there: a needle whose last
    # character sits at index `limit` yields a prefix of limit + 1. One over is
    # enough to breach, because a fenced page already spends its whole
    # _FENCE_OVERHEAD allowance and has no slack left to absorb it.
    window = text[:limit]
    for needle in ("\n\n", "\n", ". ", " "):
        at = window.rfind(needle)
        if at > 0:
            return at + len(needle)
    return limit


def _split(text: str, limit: int) -> list[str]:
    if limit < 1:
        raise ValueError("page body limit must be positive")
    pages: list[str] = []
    rest = text
    while rest:
        cut = _cut(rest, limit)
        pages.append(rest[:cut])
        rest = rest[cut:]
    return pages or [""]


def _fence_open_after(text: str, opened: bool) -> bool:
    """Whether a fence is open at the end of text, given its state at the start.

    A walk rather than a parity count, because the two directions do not obey
    the same rule: a fence line carrying an info string (```js) can only OPEN a
    block. Markdown closes only on backticks and whitespace, so a ```js inside
    an inlined diff is content, and counting it as a toggle would close a block
    the renderer leaves open -- then every later page is decorated inside out.
    Indentation is stripped because an indented fence still is one.
    """
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("```"):
            continue
        if not opened:
            opened = True
        elif not stripped.removeprefix("```").strip():
            opened = False
    return opened


def _balance_fences(bodies: list[str]) -> list[str]:
    """Close and reopen a Markdown fence split across adjacent pages."""
    balanced: list[str] = []
    opened = False
    for body in bodies:
        starts_open = opened
        opened = _fence_open_after(body, starts_open)
        page = body
        if starts_open:
            page = "```\n" + page
        if opened:
            page += "\n```"
        balanced.append(page)
    return balanced


def _page_pairs(
    text: str,
    limit: int,
    render: Callable[[str, int, int], str],
) -> list[tuple[str, str]]:
    total = 1
    for _ in range(32):
        overhead = max(len(render("", part, total)) for part in range(1, total + 1))
        bodies = _balance_fences(_split(text, limit - overhead - _FENCE_OVERHEAD))
        new_total = len(bodies)
        if new_total != total:
            total = new_total
            continue
        pages = [render(body, part, total) for part, body in enumerate(bodies, 1)]
        if any(len(page) > limit for page in pages):
            raise ValueError("page decorator is not body-linear")
        return list(zip(bodies, pages, strict=True))
    raise RuntimeError("pagination did not converge")


def paginate(
    text: str,
    *,
    limit: int = DEFAULT_PAGE_LIMIT,
    decorate: Callable[[str, int, int], str] | None = None,
) -> list[str]:
    """Return bounded pages containing all of text in order.

    ``decorate`` applies transport rendering and part markers. It is included in
    the limit, so callers cannot accidentally budget only the undecorated body.
    The function converges on the final page count because marker widths can
    change when a split crosses a decimal boundary.
    """
    render = decorate or (
        lambda body, part, total: (
            body if total == 1 else f"{body}\n\n{part_marker(part, total)}"
        )
    )
    return [page for _, page in _page_pairs(text, limit, render)]


def paginate_message(
    text: str,
    *,
    limit: int = DEFAULT_PAGE_LIMIT,
    render: Callable[[str], str] = lambda body: body,
) -> list[tuple[str, str]]:
    """Return ``(page body, rendered page)`` pairs within a wire limit."""

    def decorate(body: str, part: int, total: int) -> str:
        page = body if total == 1 else f"{body}\n\n{part_marker(part, total)}"
        return render(page)

    pairs = _page_pairs(text, limit, decorate)
    total = len(pairs)
    return [
        (
            body if total == 1 else f"{body}\n\n{part_marker(part, total)}",
            rendered,
        )
        for part, (body, rendered) in enumerate(pairs, 1)
    ]

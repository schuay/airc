# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Lossless paging for text delivered by chat transports."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256

DEFAULT_PAGE_LIMIT = 3800

# The shortest run of backticks that can open a fence. Longer openings are
# ordinary: a sender that needs a block to survive content mentioning a fence
# opens with more than the longest run inside it, so the length is data here and
# not a constant (see _fence_overhead).
_MIN_FENCE = 3


def part_marker(part: int, total: int) -> str:
    return f"_(part {part}/{total})_"


def page_token(scope: str, message_id: int, part: int) -> str:
    """A deterministic transport id for an idempotently retried page."""
    material = f"{scope}\0{message_id}\0{part}".encode()
    return sha256(material).hexdigest()


def _budget_for_pages(
    pages: int, *, limit: int = DEFAULT_PAGE_LIMIT, fence_overhead: int
) -> int:
    """The most text that could fit in `pages` pages, if every page filled.

    The overheads are the ones _page_pairs charges: the part marker on every
    page (widest count, so the answer does not depend on which page it lands
    on) and a fence that may reopen and close.

    A ceiling, not a promise, so it is private: _cut stops at a clean
    boundary, so a page closes early whenever the text has no break near its
    end, and the spilled remainder can still need one more page. A caller that
    needs the bound to hold has to measure -- see truncate_to_pages.
    """
    if pages < 1:
        raise ValueError("page count must be positive")
    overhead = len(f"\n\n{part_marker(pages, pages)}") + fence_overhead
    return pages * (limit - overhead)


def truncate_to_pages(
    text: str,
    pages: int,
    *,
    limit: int = DEFAULT_PAGE_LIMIT,
    note: str = "",
) -> str:
    """`text`, cut until it pages into at most `pages` pages. Appends `note`.

    Measured by paging instead of computed. The arithmetic ceiling assumes every page
    fills, and a page that ends on a clean break does not: a repro whose diff
    is one long minified line can leave most of a page unused, so a character
    cap picked to look right is a cap that holds for the text it was tuned on.
    This asks the pager instead, so the bound is the one the caller stated.

    Shrinks by a fraction of a page at a time. The loop is entered only by text
    already over the bound, and each step re-pages a string that is only
    getting shorter, so it costs a handful of passes at the point where a post
    was going to be cut anyway.
    """
    if len(paginate(text, limit=limit)) <= pages:
        return text
    room = _budget_for_pages(pages, limit=limit, fence_overhead=_fence_overhead(text))
    step = max(1, limit // 8)
    while room > len(note):
        candidate = text[: room - len(note)] + note
        if len(paginate(candidate, limit=limit)) <= pages:
            return candidate
        room -= step
    return note


def _cut(text: str, limit: int) -> int:
    """Choose a nonempty prefix no longer than limit, preferring clean breaks."""
    if len(text) <= limit:
        return len(text)
    # The break must end within the budget, not start there: a needle whose last
    # character sits at index `limit` yields a prefix of limit + 1. One over is
    # enough to breach, because a fenced page already spends its whole
    # _FENCE_OVERHEAD allowance and has no slack left to absorb it.
    window = text[:limit]
    # The latest break, not the strongest one anywhere in the window. Ranking the
    # needles instead costs a whole page whenever a paragraph break sits early:
    # a repro post opens "headline\n\n" and then runs thousands of unbroken diff
    # lines, so preferring "\n\n" put the headline alone on page one and pushed
    # everything after it one message further along. Every candidate here is
    # already a clean boundary, so the tie between them is worth less than the
    # room lost, and a tie on the same end position keeps the widest needle.
    best = 0
    for needle in ("\n\n", "\n", ". ", " "):
        at = window.rfind(needle)
        if at > 0:
            best = max(best, at + len(needle))
    return best or limit


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


def _fence_run(line: str) -> int:
    """The length of the backtick run opening `line`, or 0 if it is not a fence.

    Indentation is stripped because an indented fence still is one.
    """
    stripped = line.lstrip()
    run = len(stripped) - len(stripped.lstrip("`"))
    return run if run >= _MIN_FENCE else 0


def _fence_overhead(text: str) -> int:
    """What _balance_fences may add to a page that splits `text` mid-block.

    The longest fence in the text bounds the one that can be open at a page
    boundary, and a page pays for it twice -- a reopen at the top and a close at
    the bottom, each on its own line. Derived instead of fixed at three
    backticks: a block quoting a fence is opened with more (see the sender-side
    fencing helper), and budgeting three for a four-backtick reopen overruns the
    wire limit by exactly the difference.
    """
    longest = max((_fence_run(line) for line in text.splitlines()), default=0)
    return 2 * (max(longest, _MIN_FENCE) + len("\n"))


def _fence_open_after(text: str, opened: int) -> int:
    """The length of the fence open at the end of text, or 0 if none is.

    Takes the state at the start in the same terms. A walk instead of a parity
    count, because the two directions do not obey the same rule: a fence line
    carrying an info string (```js) can only open a block. Markdown closes only
    on backticks and whitespace, so a ```js inside an inlined diff is content,
    and counting it as a toggle would close a block the renderer leaves open --
    then every later page is decorated inside out.

    The length, not a flag, because markdown closes a fence only on a run
    at least as long as the one that opened it: a block opened with four
    backticks to survive quoting a fence is not closed by the three-backtick line
    inside it. Read as a flag, that line ended the block, the closing
    four-backtick line opened a new one, and the pages in between were emitted
    with no fence at all -- spilling the block's content into live chat markup,
    which is the leak the long opening fence existed to prevent.
    """
    for line in text.splitlines():
        run = _fence_run(line)
        if not run:
            continue
        if not opened:
            opened = run
        elif run >= opened and not line.lstrip()[run:].strip():
            opened = 0
    return opened


def _balance_fences(bodies: list[str]) -> list[str]:
    """Close and reopen a Markdown fence split across adjacent pages.

    Reopened at the length it was opened with, so a continuation page protects
    the remaining content as well as the first page did.
    """
    balanced: list[str] = []
    opened = 0
    for body in bodies:
        starts_open = opened
        opened = _fence_open_after(body, starts_open)
        page = body
        if starts_open:
            page = "`" * starts_open + "\n" + page
        if opened:
            page += "\n" + "`" * opened
        balanced.append(page)
    return balanced


def _page_pairs(
    text: str,
    limit: int,
    render: Callable[[str, int, int], str],
) -> list[tuple[str, str]]:
    fence_overhead = _fence_overhead(text)
    total = 1
    for _ in range(32):
        overhead = max(len(render("", part, total)) for part in range(1, total + 1))
        bodies = _balance_fences(_split(text, limit - overhead - fence_overhead))
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

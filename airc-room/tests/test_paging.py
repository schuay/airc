# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

from airc_room.paging import page_token, paginate, part_marker


def _payload(page: str) -> str:
    if "\n\n_(part " not in page:
        return page
    return page.rsplit("\n\n_(part ", 1)[0]


def test_one_page_uses_the_same_path_without_a_marker():
    assert paginate("hello", limit=32) == ["hello"]


def test_pages_are_bounded_and_lossless():
    text = "paragraph one\n\n" + "word " * 80 + "tail"
    pages = paginate(text, limit=64)
    assert len(pages) > 1
    assert all(len(page) <= 64 for page in pages)
    assert "".join(_payload(page) for page in pages) == text
    assert all(part_marker(i, len(pages)) in page for i, page in enumerate(pages, 1))


def test_one_indivisible_token_is_hard_split_without_loss():
    text = "x" * 1000
    pages = paginate(text, limit=80)
    assert "".join(_payload(page) for page in pages) == text
    assert all(len(page) <= 80 for page in pages)


def test_transport_decoration_is_inside_the_budget():
    def decorate(body: str, part: int, total: int) -> str:
        marker = "" if total == 1 else f" [{part}/{total}]"
        return f"sender: {body}{marker}"

    pages = paginate("x" * 100, limit=32, decorate=decorate)
    assert all(page.startswith("sender: ") for page in pages)
    assert all(len(page) <= 32 for page in pages)


def test_a_fence_is_closed_and_reopened_on_every_page():
    pages = paginate("before\n```js\n" + "x\n" * 100 + "```\nafter", limit=64)
    assert len(pages) > 1
    assert all(page.count("```") % 2 == 0 for page in pages)
    assert all(len(page) <= 64 for page in pages)


def test_page_tokens_are_stable_and_page_specific():
    assert page_token("chat", 42, 1) == page_token("chat", 42, 1)
    assert page_token("chat", 42, 1) != page_token("chat", 42, 2)


def test_a_clean_break_at_the_budget_edge_stays_inside_it():
    # A break candidate ending one past the budget must not be taken: the page
    # body it yields is over by one, and a fenced page spends its whole fence
    # allowance, so the overflow lands on the rendered page.
    text = "b.  a\n```\n\n\n\na. a. \n\n\n "
    pages = paginate(text, limit=30)
    assert all(len(page) <= 30 for page in pages)

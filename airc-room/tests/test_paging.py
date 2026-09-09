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


def test_an_info_string_fence_inside_a_block_does_not_close_it():
    # A diff inlined in a ```diff block can carry a ```js context line. Markdown
    # closes only on bare backticks, so that line is content -- counting it as a
    # toggle would leave every later page's fence decoration inverted.
    body = "\n".join(f"+ line {i}" for i in range(60))
    text = "```diff\n ```js\n" + body + "\n```\nafter"
    pages = paginate(text, limit=120)
    assert len(pages) > 1
    assert all(len(page) <= 120 for page in pages)
    # The trailing prose left the block, so the last page must not be fenced open.
    assert "after" in pages[-1]
    assert not pages[-1].rstrip().endswith("```")


def test_truncate_to_pages_bounds_text_that_a_character_budget_would_not():
    from airc_room.paging import truncate_to_pages

    # One unbroken line: _cut can only hard-split it, and a diff whose lines are
    # long leaves each page short of its budget -- the case a tuned constant
    # misses. 40k characters is well past three pages at any of our limits.
    text = "*headline*\n\n```diff\n" + "x" * 40000 + "\n```\n"
    cut = truncate_to_pages(text, 3, note="\n\n_(truncated)_")
    assert len(paginate(cut)) <= 3
    assert cut.endswith("_(truncated)_")
    assert cut.startswith("*headline*")


def test_truncate_to_pages_leaves_text_that_already_fits_alone():
    from airc_room.paging import truncate_to_pages

    text = "short enough\n\nto need nothing"
    assert truncate_to_pages(text, 3, note="_(truncated)_") == text


def test_a_paragraph_break_near_the_start_does_not_cost_a_whole_page():
    # A repro post opens "headline\n\n" and then runs unbroken diff lines. The
    # break after the headline must not be the one chosen for page one.
    text = "headline\n\n" + "\n".join("+ diff line here" for _ in range(400))
    pages = paginate(text, limit=600)
    assert len(pages[0]) > 400


def test_a_long_fence_is_not_closed_by_a_shorter_run_inside_it():
    # A sender that needs a block to survive content mentioning a fence opens it
    # with four backticks, which markdown closes only on four. Read as a flag,
    # the three-backtick line ended the block and the real closing line opened a
    # new one -- so a page that was whole came out carrying a stray fence.
    text = "````diff\nquoting a fence:\n```\nstill inside\n````"
    assert paginate(text, limit=200) == [text]


def test_a_long_fence_survives_the_pages_it_is_split_across():
    # The leak the long opening fence exists to prevent, reintroduced by the
    # pager: the shorter run inside was read as a close, so the middle pages
    # were emitted with no fence at all and their content rendered as live chat
    # markup. Every page must reopen at the length the block was opened with.
    body = "A" * 200 + "\n```\n" + "B" * 400 + "\n"
    text = "headline\n\n````diff\n" + body + "````"
    pages = paginate(text, limit=200)
    assert len(pages) > 3
    assert all(len(page) <= 200 for page in pages)
    for page in pages[1:]:
        assert page.lstrip().startswith("````")
    for page in pages[:-1]:
        assert _payload(page).rstrip().endswith("````")


def test_a_reopened_long_fence_is_paid_for_in_the_budget():
    # The fence allowance was fixed at three backticks, so a four-backtick
    # reopen overran the wire limit by exactly the difference -- and _page_pairs
    # turns that into "page decorator is not body-linear" rather than a short
    # page.
    text = "`````\n" + "x\n" * 200 + "`````"
    pages = paginate(text, limit=64)
    assert len(pages) > 1
    assert all(len(page) <= 64 for page in pages)

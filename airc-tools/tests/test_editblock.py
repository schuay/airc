# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

from airc_tools.editblock import (
    find_similar_lines,
    replace_most_similar_chunk,
)

WHOLE = """def greet(name):
    msg = "hi " + name
    return msg
"""


def test_perfect_match():
    out = replace_most_similar_chunk(
        WHOLE, '    msg = "hi " + name\n', '    msg = f"hi {name}"\n'
    )
    assert '    msg = f"hi {name}"' in out
    assert "return msg" in out


def test_leading_whitespace_flex():
    # The model outdents the block (drops the 4-space indent); the whitespace
    # tier must still match and re-derive the file's real indentation.
    out = replace_most_similar_chunk(
        WHOLE, 'msg = "hi " + name\n', 'msg = "hello " + name\n'
    )
    assert out is not None
    assert '    msg = "hello " + name' in out  # indentation restored


def test_no_match_returns_none():
    assert replace_most_similar_chunk(WHOLE, "nonexistent line\n", "x\n") is None


def test_dotdotdots():
    whole = "a\nb\nc\nd\ne\n"
    part = "a\n...\ne\n"
    replace = "A\n...\nE\n"
    out = replace_most_similar_chunk(whole, part, replace)
    assert out == "A\nb\nc\nd\nE\n"


def test_find_similar_lines_hint():
    # Matching is by whole-line equality, so a single-line near-miss yields
    # nothing; the hint fires when the outer lines of a multi-line block match.
    search = 'def greet(name):\n    msg = "hi" + name\n    return msg\n'
    hint = find_similar_lines(search, WHOLE)
    assert '    msg = "hi " + name' in hint  # the real middle line is surfaced


def test_find_similar_lines_below_threshold():
    assert find_similar_lines("total garbage nowhere near\n", WHOLE) == ""


def test_over_indented_search_is_no_match_not_corruption():
    # Over-indented SEARCH + unindented REPLACE: the shared outdent is capped
    # by the replace block's zero indent, so the search line stays LONGER than
    # the file line. The prefix slice then goes negative and used to glue real
    # content chars ("x = ") onto the replacement -- silent corruption reported
    # as a successful apply ('  x = x = 3'). It must be a clean no-match.
    whole = "  x = 1\ny = 2\n"
    assert replace_most_similar_chunk(whole, "    x = 1\n", "x = 3\n") is None

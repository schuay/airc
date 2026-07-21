"""The SEARCH/REPLACE apply engine, lifted from aider's editblock_coder.py.

Only the apply cascade and the failure-hint helper are ported -- the reliability
core, and pure/self-contained. The transport (parsing blocks out of prose) and
aider's git/multi-file-fallback opinionation are left behind: edits arrive here
as structured (search, replace) strings via edit_file, one file at a time.

The cascade, in decreasing precision:
  1. exact line-tuple match (perfect_replace)
  2. a leading-whitespace flexible tier that re-derives the file's real
     indentation when the model outdents a block -- the tier that kills the
     largest real failure class (replace_part_with_missing_leading_whitespace)
  3. a dotdotdots tier for elided `...` sections (try_dotdotdots)

Fuzzy edit-distance matching is deliberately absent. In aider it sits below a
bare `return` in replace_most_similar_chunk and never runs on the apply path; a
fuzzy apply that silently mismatches is worse than a clean failure the model can
retry. SequenceMatcher survives only in find_similar_lines, for the diagnostic
did-you-mean hint.
"""

import re
from difflib import SequenceMatcher


def prep(content):
    if content and not content.endswith("\n"):
        content += "\n"
    lines = content.splitlines(keepends=True)
    return content, lines


def perfect_replace(whole_lines, part_lines, replace_lines):
    part_tup = tuple(part_lines)
    part_len = len(part_lines)
    for i in range(len(whole_lines) - part_len + 1):
        if tuple(whole_lines[i : i + part_len]) == part_tup:
            res = whole_lines[:i] + replace_lines + whole_lines[i + part_len :]
            return "".join(res)


def match_but_for_leading_whitespace(whole_lines, part_lines):
    num = len(whole_lines)
    # The non-whitespace of every line must agree...
    if not all(whole_lines[i].lstrip() == part_lines[i].lstrip() for i in range(num)):
        return
    # ...and every non-blank line must be offset by the same leading prefix.
    # A search line LONGER than the file line has no prefix to lift -- the
    # length delta goes negative and the slice below would cut real content
    # chars off the END of the file line, gluing them onto every replacement
    # line: silent corruption reported as a successful apply. No match instead.
    if any(
        len(whole_lines[i]) < len(part_lines[i])
        for i in range(num)
        if whole_lines[i].strip()
    ):
        return
    add = set(
        whole_lines[i][: len(whole_lines[i]) - len(part_lines[i])]
        for i in range(num)
        if whole_lines[i].strip()
    )
    if len(add) != 1:
        return
    return add.pop()


def replace_part_with_missing_leading_whitespace(
    whole_lines, part_lines, replace_lines
):
    # The model usually botches leading whitespace uniformly -- omitting all of
    # it, or keeping only some. Outdent both blocks by the largest shared amount,
    # then look for a match that differs only by a constant leading prefix, and
    # re-apply that prefix to the replacement.
    leading = [len(p) - len(p.lstrip()) for p in part_lines if p.strip()] + [
        len(p) - len(p.lstrip()) for p in replace_lines if p.strip()
    ]
    if leading and min(leading):
        num_leading = min(leading)
        part_lines = [p[num_leading:] if p.strip() else p for p in part_lines]
        replace_lines = [p[num_leading:] if p.strip() else p for p in replace_lines]

    num_part_lines = len(part_lines)
    for i in range(len(whole_lines) - num_part_lines + 1):
        add_leading = match_but_for_leading_whitespace(
            whole_lines[i : i + num_part_lines], part_lines
        )
        if add_leading is None:
            continue
        replace_lines = [
            add_leading + rline if rline.strip() else rline for rline in replace_lines
        ]
        whole_lines = (
            whole_lines[:i] + replace_lines + whole_lines[i + num_part_lines :]
        )
        return "".join(whole_lines)
    return None


def perfect_or_whitespace(whole_lines, part_lines, replace_lines):
    res = perfect_replace(whole_lines, part_lines, replace_lines)
    if res:
        return res
    return replace_part_with_missing_leading_whitespace(
        whole_lines, part_lines, replace_lines
    )


def try_dotdotdots(whole, part, replace):
    """Handle a search/replace that elides unchanged spans with `...` lines.

    Returns None if the block has no `...`. Raises ValueError on a malformed or
    non-unique elision so the caller falls through to a clean failure.
    """
    dots_re = re.compile(r"(^\s*\.\.\.\n)", re.MULTILINE | re.DOTALL)
    part_pieces = re.split(dots_re, part)
    replace_pieces = re.split(dots_re, replace)

    if len(part_pieces) != len(replace_pieces):
        raise ValueError("Unpaired ... in SEARCH/REPLACE block")
    if len(part_pieces) == 1:
        return

    # The `...` separators themselves (odd indices) must match on both sides.
    if not all(
        part_pieces[i] == replace_pieces[i] for i in range(1, len(part_pieces), 2)
    ):
        raise ValueError("Unmatched ... in SEARCH/REPLACE block")

    part_pieces = [part_pieces[i] for i in range(0, len(part_pieces), 2)]
    replace_pieces = [replace_pieces[i] for i in range(0, len(replace_pieces), 2)]

    for part, replace in zip(part_pieces, replace_pieces):
        if not part and not replace:
            continue
        if not part and replace:
            if not whole.endswith("\n"):
                whole += "\n"
            whole += replace
            continue
        if whole.count(part) == 0:
            raise ValueError("elided section not found")
        if whole.count(part) > 1:
            raise ValueError("elided section not unique")
        whole = whole.replace(part, replace, 1)

    return whole


def replace_most_similar_chunk(whole, part, replace):
    """Find `part` in `whole` and replace it with `replace`.

    Returns the updated whole, or None if no tier matched.
    """
    whole, whole_lines = prep(whole)
    part, part_lines = prep(part)
    replace, replace_lines = prep(replace)

    res = perfect_or_whitespace(whole_lines, part_lines, replace_lines)
    if res:
        return res

    # The model sometimes prepends a spurious blank line; retry without it.
    if len(part_lines) > 2 and not part_lines[0].strip():
        res = perfect_or_whitespace(whole_lines, part_lines[1:], replace_lines)
        if res:
            return res

    try:
        res = try_dotdotdots(whole, part, replace)
        if res:
            return res
    except ValueError:
        pass

    return None


def find_similar_lines(search, content, threshold=0.6):
    """The did-you-mean hint: the run of file lines most similar to a failed
    search, so the failure message can show the model what it likely meant.
    Diagnostic only -- never used to apply an edit."""
    search_lines = search.splitlines()
    content_lines = content.splitlines()
    if not search_lines or len(content_lines) < len(search_lines):
        return ""

    best_ratio = 0.0
    best_match = None
    best_match_i = 0
    for i in range(len(content_lines) - len(search_lines) + 1):
        chunk = content_lines[i : i + len(search_lines)]
        ratio = SequenceMatcher(None, search_lines, chunk).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = chunk
            best_match_i = i

    if best_ratio < threshold or best_match is None:
        return ""

    if best_match[0] == search_lines[0] and best_match[-1] == search_lines[-1]:
        return "\n".join(best_match)

    # Fuzzy ends: widen the window a little so the model sees the neighbourhood.
    n = 5
    start = max(0, best_match_i - n)
    end = min(len(content_lines), best_match_i + len(search_lines) + n)
    return "\n".join(content_lines[start:end])

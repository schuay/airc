# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

from airc_tools.limits import MAX_READ_BYTES
from airc_tools.read import read_file

SRC = "".join(f"line {i}\n" for i in range(1, 21))  # 20 lines


def test_verbatim_no_gutter(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text(SRC)
    out = read_file(str(f))
    header, _, body = out.partition("\n")
    assert "lines 1-20 of 20" in header
    # Body is exact content: no "1\t" / "1|" / "1->" gutter prefix.
    assert body == SRC


def test_range(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text(SRC)
    out = read_file(str(f), offset=5, limit=3)
    header, _, body = out.partition("\n")
    assert "lines 5-7 of 20" in header  # inclusive: offset 5, three lines
    assert body == "line 5\nline 6\nline 7\n"


def test_byte_cap(tmp_path):
    # ~100 bytes/line so the default 2000-line window exceeds the byte cap and
    # the byte guard (not the line count) is what truncates.
    f = tmp_path / "big.txt"
    f.write_text("".join(f"{i:06d} " + "x" * 92 + "\n" for i in range(4000)))
    out = read_file(str(f), offset=1, limit=100000)
    assert "byte cap" in out
    assert len(out) < MAX_READ_BYTES + 500  # header + note overhead only


def test_offset_past_end(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text(SRC)
    out = read_file(str(f), offset=999)
    assert "past the end" in out


def test_missing(tmp_path):
    out = read_file(str(tmp_path / "nope.txt"))
    assert "does not exist" in out


def test_directory(tmp_path):
    out = read_file(str(tmp_path))
    assert "is a directory" in out


def test_read_fifo_refused(tmp_path, monkeypatch):
    # Opening a FIFO for read blocks until a writer appears -- in a to_thread
    # worker that is an uncancellable, permanent wedge. Refuse it up front.
    import os

    monkeypatch.setenv("AIRC_TOOLS_ROOT", str(tmp_path))
    os.mkfifo(tmp_path / "pipe")
    assert "not a regular file" in read_file(str(tmp_path / "pipe"))


def test_read_unreadable_returns_error_not_raises(tmp_path):
    # A regular file the process cannot read (permission bit) passes the shape
    # guards but open() raises. It must surface as the tool's error string.
    f = tmp_path / "secret.txt"
    f.write_text(SRC)
    f.chmod(0o000)
    try:
        out = read_file(str(f))
        assert out.startswith("error: cannot access")
    finally:
        f.chmod(0o644)


def test_line_numbers_opt_in(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text(SRC)
    out = read_file(str(f), offset=5, limit=3, line_numbers=True)
    header, _, body = out.partition("\n")
    assert "lines 5-7 of 20" in header
    assert "numbered" in header  # the not-edit-safe warning
    # Gutter carries ABSOLUTE file line numbers, not window-relative ones --
    # a number that is only correct at offset=1 is worse than none at all.
    assert body.splitlines()[0].split("\t") == ["     5", "line 5"]
    assert [ln.split("\t")[0].strip() for ln in body.splitlines()] == ["5", "6", "7"]


def test_line_numbers_default_off_stays_paste_safe(tmp_path):
    # The invariant the gutter must not break: default output is byte-identical
    # to the file, so it pastes into an edit_file SEARCH.
    f = tmp_path / "a.txt"
    f.write_text(SRC)
    _, _, body = read_file(str(f)).partition("\n")
    assert body == SRC


def test_line_numbers_preserve_content_bytes(tmp_path):
    # Stripping the gutter must recover the exact source text: the numbered read
    # is the same bytes, only labelled.
    f = tmp_path / "a.txt"
    f.write_text(SRC)
    _, _, body = read_file(str(f), line_numbers=True).partition("\n")
    stripped = "".join(ln.split("\t", 1)[1] + "\n" for ln in body.splitlines())
    assert stripped == SRC


def test_line_numbers_no_trailing_newline(tmp_path):
    # A file whose last line has no newline must not gain one from numbering.
    f = tmp_path / "a.txt"
    f.write_text("one\ntwo")
    _, _, body = read_file(str(f), line_numbers=True).partition("\n")
    assert body == "     1\tone\n     2\ttwo"


def test_line_numbers_under_byte_cap(tmp_path):
    # Numbering happens after the byte cap, so the gutter never displaces
    # content: a capped numbered read covers the same lines as a plain one.
    f = tmp_path / "big.txt"
    f.write_text("".join(f"{i:06d} " + "x" * 92 + "\n" for i in range(4000)))
    plain = read_file(str(f), offset=1, limit=100000)
    numbered = read_file(str(f), offset=1, limit=100000, line_numbers=True)
    assert "byte cap" in numbered
    # Same window in both: compare the "lines A-B of N" span in the header.
    assert (
        plain.partition("\n")[0].split("(")[0]
        == (numbered.partition("\n")[0].split("(")[0])
    )

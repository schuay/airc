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

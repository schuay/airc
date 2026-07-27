# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

from airc_tools.edit import apply_edits, write_file
from airc_tools.limits import MAX_EDIT_FILE_BYTES

SRC = "line one\nline two\nline three\n"


def test_single_edit(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text(SRC)
    out = apply_edits(str(f), [("line two\n", "LINE TWO\n")])
    assert "applied 1 edit" in out
    assert f.read_text() == "line one\nLINE TWO\nline three\n"


def test_multiple_edits_one_call(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text(SRC)
    apply_edits(str(f), [("line one\n", "1\n"), ("line three\n", "3\n")])
    assert f.read_text() == "1\nline two\n3\n"


def test_all_or_nothing_rollback(tmp_path):
    # Second edit cannot match: nothing is written, including the first edit.
    f = tmp_path / "a.txt"
    f.write_text(SRC)
    out = apply_edits(str(f), [("line one\n", "1\n"), ("nope\n", "x\n")])
    assert "failed to match" in out
    assert f.read_text() == SRC  # untouched


def test_create_new_file(tmp_path):
    f = tmp_path / "sub" / "new.txt"
    out = apply_edits(str(f), [("", "fresh contents\n")])
    assert "applied" in out
    assert f.read_text() == "fresh contents\n"


def test_create_rejects_nonempty_search(tmp_path):
    f = tmp_path / "missing.txt"
    out = apply_edits(str(f), [("something\n", "x\n")])
    assert "does not exist" in out
    assert not f.exists()


def test_append_empty_search(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text(SRC)
    apply_edits(str(f), [("", "line four\n")])
    assert f.read_text() == SRC + "line four\n"


def test_failure_text_has_hint(tmp_path):
    # A multi-line search with one line wrong: it fails, and since the outer
    # lines match exactly the did-you-mean hint surfaces the real middle line.
    f = tmp_path / "a.txt"
    f.write_text(SRC)
    out = apply_edits(str(f), [("line one\nline twoo\nline three\n", "x\n")])
    assert "failed to match" in out
    assert "Closest lines" in out
    assert "line two" in out


def test_trailing_newline_preserved(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("no newline at end")  # no trailing \n
    apply_edits(str(f), [("no newline at end", "still none")])
    assert f.read_text() == "still none"  # not "still none\n"


def test_directory_guard(tmp_path):
    out = apply_edits(str(tmp_path), [("x\n", "y\n")])
    assert "is a directory" in out


def test_file_too_big_guard(tmp_path):
    f = tmp_path / "big.txt"
    f.write_bytes(b"x" * (MAX_EDIT_FILE_BYTES + 1))
    out = apply_edits(str(f), [("x", "y")])
    assert "over the" in out and "edit limit" in out


def test_write_file_creates_verbatim(tmp_path):
    f = tmp_path / "sub" / "new.js"  # parent does not exist yet
    out = write_file(str(f), SRC)
    assert "created" in out
    assert f.read_text() == SRC  # verbatim, parent dir created


def test_write_file_overwrites_not_appends(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("old contents\n")
    out = write_file(str(f), "new\n")
    assert "overwrote" in out
    assert f.read_text() == "new\n"  # replaced, unlike edit_file's empty search


def test_write_file_directory_guard(tmp_path):
    out = write_file(str(tmp_path), "x")
    assert "is a directory" in out


def test_write_file_too_big_guard(tmp_path):
    out = write_file(str(tmp_path / "big.txt"), "x" * (MAX_EDIT_FILE_BYTES + 1))
    assert "over the" in out and "byte limit" in out


def test_edit_and_write_fifo_refused(tmp_path, monkeypatch):
    # Same wedge as read_file: never open a non-regular file.
    import os

    monkeypatch.setenv("AIRC_TOOLS_ROOT", str(tmp_path))
    os.mkfifo(tmp_path / "pipe")
    assert "not a regular file" in apply_edits(str(tmp_path / "pipe"), [("a", "b")])
    assert "not a regular file" in write_file(str(tmp_path / "pipe"), "x")


def test_write_readonly_returns_error_not_raises(tmp_path):
    # A worktree's third_party deps can symlink into a mirror mounted read-only;
    # the path is a regular file so the shape guards pass, but write_text raises
    # OSError. It must come back as the tool's error string, not escape and kill
    # the turn.
    f = tmp_path / "ro.txt"
    f.write_text(SRC)
    (tmp_path).chmod(0o555)  # remove write on the dir so replace fails
    f.chmod(0o444)
    try:
        out = apply_edits(str(f), [("line two\n", "LINE TWO\n")])
        assert out.startswith("error: cannot access")
        out = write_file(str(f), "x")
        assert out.startswith("error: cannot access")
    finally:
        tmp_path.chmod(0o755)
        f.chmod(0o644)


def test_create_in_readonly_dir_returns_error(tmp_path):
    # mkdir/create under a read-only parent raises PermissionError (an OSError);
    # guarded, it is a message the model can act on.
    sub = tmp_path / "ro"
    sub.mkdir()
    sub.chmod(0o555)
    try:
        out = write_file(str(sub / "new.txt"), "x\n")
        assert out.startswith("error: cannot access")
    finally:
        sub.chmod(0o755)

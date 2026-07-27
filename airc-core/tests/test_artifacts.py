# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""The shared run-artifact log primitive."""

from airc_core import ArtifactLog, slug


def test_slug():
    assert slug("[wasm] Fix OOB in foo!") == "wasm-fix-oob-in-foo"
    assert slug("a/b:c  d") == "a-b-c-d"
    assert slug("") == "untitled"
    assert len(slug("x" * 200)) == 60


async def test_write_creates_dated_namespaced_file(tmp_path):
    log = ArtifactLog(tmp_path)
    assert log.enabled
    await log.write("reviews", "abc123-some subject", "# hi\n")
    files = list((tmp_path / "reviews").glob("*.md"))
    assert len(files) == 1
    # <date>-<slug>.md, category as subfolder.
    assert files[0].name.endswith("-abc123-some-subject.md")
    assert files[0].read_text() == "# hi\n"


async def test_write_custom_extension(tmp_path):
    log = ArtifactLog(tmp_path)
    await log.write("perf", "run-7", "raw text", ext="txt")
    assert list((tmp_path / "perf").glob("*.txt"))


async def test_write_is_utf8_regardless_of_locale(tmp_path):
    # Non-ASCII (an em dash, an accented char) must round-trip; write_text used to
    # fall back to the locale encoding, raising UnicodeEncodeError under a service
    # with no LANG set -- the artifact then vanished with no OSError warning.
    log = ArtifactLog(tmp_path)
    body = "review — café 你好\n"
    await log.write("reviews", "k", body)
    f = next((tmp_path / "reviews").glob("*.md"))
    assert f.read_text(encoding="utf-8") == body


async def test_write_swallows_errors(tmp_path):
    # A trace must never sink (or loop) the work it traces: a write failure is
    # logged and swallowed, not raised. Root is a file, so mkdir fails.
    bad = tmp_path / "not-a-dir"
    bad.write_text("x")
    log = ArtifactLog(bad)
    await log.write("reviews", "k", "body")  # must not raise
    assert log.enabled


async def test_disabled_is_noop(tmp_path):
    log = ArtifactLog(None)
    assert not log.enabled
    await log.write("reviews", "k", "body")
    assert not any(tmp_path.glob("**/*"))


async def test_exists_is_date_independent(tmp_path):
    log = ArtifactLog(tmp_path)
    key = "abc123-some subject"
    assert not log.exists("reviews", key)
    await log.write("reviews", key, "# hi\n")
    assert log.exists("reviews", key)
    # Slug-matched, so the same commit+subject is recognized regardless of when
    # it was first written, while a different key is not.
    assert not log.exists("reviews", "other-key")
    # Honors the extension and never reports across categories.
    assert not log.exists("reviews", key, ext="txt")
    assert not log.exists("perf", key)


def test_exists_false_when_disabled():
    assert not ArtifactLog(None).exists("reviews", "k")

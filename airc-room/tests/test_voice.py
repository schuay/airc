# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

from airc_room.personas import Persona
from airc_room.runner import build_system_prompt, voice_body

# A synthetic guide, deliberately not modeled on anyone: the identity line is a
# placeholder with an era predating this project, so the fixture cannot be read
# as a fingerprint of a real contributor. It carries one of every element
# voice_body() strips or redacts (frontmatter, H1 title, email, hash token,
# sources tail) plus a year, which must survive.
GUIDE = """\
---
title: someone voice guide
type: workdoc
date: 2026-07-01
---

# someone voice guide

Governs TONE only.

A. Placeholder (someone@example.com), active ~1994. Calm and terse.

## Voice tells

- Trailing `..` for a musing beat.
- Landed in deadbeef1 as a fixup.

## Sources mined

- Synthetic fixture; no real corpus. CL 1234567.
"""


def _persona(name="perf"):
    return Persona(
        name=name,
        display_name=name.capitalize(),
        description="d",
        system_prompt="# Role",
        key=name,
    )


def test_voice_body_strips_frontmatter_and_sources():
    body = voice_body(GUIDE)
    assert not body.startswith("#")  # leading H1 title (names the source) gone
    assert "voice guide" not in body  # the handle-bearing title is stripped
    assert "title: someone" not in body  # frontmatter gone
    assert "Trailing `..`" in body  # tone content kept
    assert "Calm and terse." in body  # tone descriptor after the identity kept
    assert "Sources mined" not in body  # provenance tail dropped


def test_voice_body_redacts_email_and_hash():
    body = voice_body(GUIDE)
    assert "@chromium.org" not in body  # email redacted
    assert "someone@" not in body
    assert "deadbeef1" not in body  # commit-hash token redacted
    # A plain year is not a hash and must survive.
    assert "1994" in body


def test_voice_body_robust_without_markers():
    plain = "Just some voice text, no frontmatter, no sources."
    assert voice_body(plain) == plain


def test_build_system_prompt_appends_voice_section():
    p = _persona()
    out = build_system_prompt(p, {p.name: p}, "", voice="Sound terse.")
    assert "## Voice" in out
    assert "TONE ONLY" in out
    assert "Sound terse." in out
    # Voice is the trailing section so it carries recency weight.
    assert out.rstrip().endswith("Sound terse.")


def test_build_system_prompt_no_voice_by_default():
    p = _persona()
    assert "## Voice" not in build_system_prompt(p, {p.name: p}, "")

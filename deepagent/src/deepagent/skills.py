# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Skill index rendering for progressive disclosure.

A skill is a markdown file with `name`/`description` frontmatter. Only the index
(name + one line + filename) goes in the cached system prompt; the body is read
on demand by the agent (for a repo-backed store, via the read tools it already has).
This keeps situational playbooks out of every turn's prefix while still telling
the agent they exist.

Generic over the store: this reads frontmatter from a local directory of files
(the application scans wherever its skills live). How the agent reads a body --
a local read_file, or repo_git_show against an MCP repo -- is the application's
choice, conveyed via `read_hint`.
"""

from __future__ import annotations

from pathlib import Path


def _frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(errors="replace")
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    out: dict[str, str] = {}
    for line in text[3:end].splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def render_skill_index(skill_dir: Path, read_hint: str = "") -> str:
    """Render `name (file) -- description` lines for the *.md skills in a dir.

    Returns "" when the dir is absent or empty, so an application can splice the
    result into its system prompt unconditionally. `read_hint` is a sentence
    telling the agent how to read a body (e.g. a repo_git_show call), placed
    under the heading.
    """
    skill_dir = Path(skill_dir)
    if not skill_dir.is_dir():
        return ""
    entries = []
    for f in sorted(skill_dir.glob("*.md")):
        fm = _frontmatter(f)
        name = fm.get("name") or f.stem
        desc = fm.get("description", "").strip()
        entries.append(f"- {name} ({f.name})" + (f" -- {desc}" if desc else ""))
    if not entries:
        return ""
    head = "## Skills (read the body before acting; the index only names them)"
    hint = f"\n{read_hint}" if read_hint else ""
    return f"{head}{hint}\n" + "\n".join(entries) + "\n"

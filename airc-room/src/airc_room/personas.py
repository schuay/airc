# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Persona discovery: one agent per folder.

    agents/
      perf/
        agent.toml    -- metadata, model, tool access
        system.md     -- the system prompt

agent.toml fields (all optional except description):

    display_name = "Perf"                 # defaults to capitalized folder name
    description = "..."                   # one line; shown to other agents and
                                          # used by the coordinator for routing
    model = "google_genai:gemini-..."     # defaults to [models].default
    tool_groups = ["read", "active"]      # named groups from config
    tools = ["repo_git_grep"]             # extra explicit tool name patterns

The folder name is the agent's handle: lowercase, used to address it (a leading
"handle:" prefix).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import tomllib

_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


class PersonaError(Exception):
    pass


@dataclass(frozen=True)
class Persona:
    name: str
    display_name: str
    description: str
    system_prompt: str
    model_id: str | None = None
    tool_groups: tuple[str, ...] = ()
    tools: tuple[str, ...] = field(default_factory=tuple)
    path: Path | None = None
    # Optional human nickname (e.g. "Sonic" for the perf agent). Only takes
    # effect when [airc].use_nicknames is on, at which point it replaces both the
    # handle (name) and display_name; the folder name stays the on-disk identity.
    nickname: str = ""
    # Stable identity for persisted per-thread state (seen offsets, checkpoints):
    # the folder name, unchanged by the nickname swap. Keeping state keyed on this
    # rather than the addressable handle lets use_nicknames toggle on/off without
    # orphaning a persona's thread memory. Defaults to name for direct construction.
    key: str = ""

    @property
    def state_key(self) -> str:
        return self.key or self.name


def load_persona(folder: Path) -> Persona:
    name = folder.name
    if not _NAME_RE.match(name):
        raise PersonaError(
            f"{folder}: agent folder name must be a lowercase handle"
            " (letters, digits, '-', '_')"
        )
    toml_path = folder / "agent.toml"
    prompt_path = folder / "system.md"
    if not prompt_path.is_file():
        raise PersonaError(f"{folder}: missing system.md")
    meta = tomllib.loads(toml_path.read_text()) if toml_path.is_file() else {}
    description = meta.get("description", "").strip()
    if not description:
        raise PersonaError(f"{folder}: agent.toml must set a one-line 'description'")
    return Persona(
        name=name,
        display_name=meta.get("display_name", name.capitalize()),
        description=description,
        system_prompt=prompt_path.read_text().strip(),
        model_id=meta.get("model"),
        tool_groups=tuple(meta.get("tool_groups", [])),
        tools=tuple(meta.get("tools", [])),
        path=folder,
        nickname=meta.get("nickname", "").strip(),
        key=name,
    )


def _load_doc(agents_dir: Path, name: str) -> str:
    f = agents_dir / name
    return f.read_text().strip() if f.is_file() else ""


def load_room_prompt(agents_dir: Path) -> str:
    """Optional shared prompt (etiquette, house rules) from agents_dir/room.md,
    appended to every agent's system prompt. Empty if the file is absent."""
    return _load_doc(agents_dir, "room.md")


def load_commit_brief(agents_dir: Path) -> str:
    """Per-turn task brief from agents_dir/commit_commentary.md, injected into a
    turn only when an agent was picked to comment on a commit. Empty if absent."""
    return _load_doc(agents_dir, "commit_commentary.md")


def _apply_nickname(p: Persona) -> Persona:
    """Swap in the persona's nickname as both handle and display name. The
    lowercased nickname becomes the addressable handle, so it must be a valid
    handle; routing, display, and prompts then use the nickname uniformly.
    Persisted state stays keyed on p.state_key (the folder name, untouched by the
    replace below), so the toggle does not orphan a persona's thread memory."""
    if not p.nickname:
        return p
    handle = p.nickname.lower()
    if not _NAME_RE.match(handle):
        raise PersonaError(
            f"{p.path}: nickname {p.nickname!r} is not a valid handle"
            " (letters, digits, '-', '_'; must start with a letter)"
        )
    return replace(p, name=handle, display_name=p.nickname)


def discover_personas(
    agents_dir: Path, *, use_nicknames: bool = False
) -> dict[str, Persona]:
    """Load all agent folders under agents_dir, sorted by name. With
    use_nicknames, each persona that declares a nickname is addressed and
    displayed by it (see _apply_nickname)."""
    if not agents_dir.is_dir():
        raise PersonaError(f"agents directory not found: {agents_dir}")
    personas: dict[str, Persona] = {}
    for folder in sorted(agents_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith((".", "_")):
            continue
        p = load_persona(folder)
        if use_nicknames:
            p = _apply_nickname(p)
        if p.name in personas:
            raise PersonaError(f"duplicate persona handle {p.name!r}")
        personas[p.name] = p
    if not personas:
        raise PersonaError(f"no agent folders found in {agents_dir}")
    return personas

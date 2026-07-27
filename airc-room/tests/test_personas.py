# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

import pytest

from airc_room.personas import PersonaError, discover_personas, load_persona


def write_agent(root, name, toml='description = "a test agent"', prompt="# Role"):
    folder = root / name
    folder.mkdir(parents=True)
    if toml is not None:
        (folder / "agent.toml").write_text(toml)
    (folder / "system.md").write_text(prompt)
    return folder


def test_load_persona(tmp_path):
    folder = write_agent(
        tmp_path,
        "perf",
        toml=(
            'display_name = "Perf"\n'
            'description = "perf things"\n'
            'model = "anthropic:claude-sonnet-4-6"\n'
            'tool_groups = ["read", "active"]\n'
            'tools = ["llvm_mca"]\n'
        ),
    )
    p = load_persona(folder)
    assert p.name == "perf"
    assert p.display_name == "Perf"
    assert p.model_id == "anthropic:claude-sonnet-4-6"
    assert p.tool_groups == ("read", "active")
    assert p.tools == ("llvm_mca",)
    assert p.system_prompt == "# Role"


def test_defaults(tmp_path):
    p = load_persona(write_agent(tmp_path, "gc"))
    assert p.display_name == "Gc"
    assert p.model_id is None
    assert p.tool_groups == ()


def test_missing_description(tmp_path):
    folder = write_agent(tmp_path, "x", toml='display_name = "X"')
    with pytest.raises(PersonaError, match="description"):
        load_persona(folder)


def test_missing_system_md(tmp_path):
    folder = tmp_path / "y"
    folder.mkdir()
    (folder / "agent.toml").write_text('description = "y"')
    with pytest.raises(PersonaError, match="system.md"):
        load_persona(folder)


def test_bad_name(tmp_path):
    with pytest.raises(PersonaError, match="lowercase"):
        load_persona(write_agent(tmp_path, "BadName"))


def test_discover(tmp_path):
    write_agent(tmp_path, "perf")
    write_agent(tmp_path, "compiler")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "notes.txt").write_text("ignored")
    personas = discover_personas(tmp_path)
    assert list(personas) == ["compiler", "perf"]


def test_discover_empty(tmp_path):
    with pytest.raises(PersonaError, match="no agent folders"):
        discover_personas(tmp_path)


def test_nickname_parsed_but_not_applied_by_default(tmp_path):
    # load_persona records the nickname but never swaps it in; the toggle lives
    # in discover_personas so the functional handle is the default everywhere.
    folder = write_agent(tmp_path, "perf", toml='description = "d"\nnickname = "Sonic"')
    p = load_persona(folder)
    assert p.nickname == "Sonic"
    assert p.name == "perf"
    assert p.display_name == "Perf"


def test_discover_use_nicknames_swaps_handle_and_display(tmp_path):
    write_agent(tmp_path, "perf", toml='description = "d"\nnickname = "Sonic"')
    write_agent(tmp_path, "gc", toml='description = "d"')  # no nickname
    personas = discover_personas(tmp_path, use_nicknames=True)
    # perf is now addressed and displayed as its nickname; gc, lacking one,
    # stays functional. The folder identity is untouched (path still perf/).
    assert set(personas) == {"sonic", "gc"}
    sonic = personas["sonic"]
    assert sonic.display_name == "Sonic"
    assert sonic.path.name == "perf"
    # State stays keyed on the folder identity, so toggling nicknames does not
    # orphan a persona's persisted thread state.
    assert sonic.state_key == "perf"
    assert personas["gc"].display_name == "Gc"


def test_state_key_matches_functional_handle_across_toggle(tmp_path):
    write_agent(tmp_path, "perf", toml='description = "d"\nnickname = "Sonic"')
    off = discover_personas(tmp_path)["perf"]
    on = discover_personas(tmp_path, use_nicknames=True)["sonic"]
    # Same stable key both ways: flipping the flag on an existing store reuses
    # the state written under the functional handle.
    assert off.state_key == on.state_key == "perf"


def test_discover_without_nicknames_keeps_functional(tmp_path):
    write_agent(tmp_path, "perf", toml='description = "d"\nnickname = "Sonic"')
    personas = discover_personas(tmp_path)
    assert set(personas) == {"perf"}
    assert personas["perf"].display_name == "Perf"


def test_nickname_must_be_valid_handle(tmp_path):
    write_agent(tmp_path, "perf", toml='description = "d"\nnickname = "Sonic!"')
    with pytest.raises(PersonaError, match="not a valid handle"):
        discover_personas(tmp_path, use_nicknames=True)


def test_load_room_prompt(tmp_path):
    from airc_room.personas import load_room_prompt

    assert load_room_prompt(tmp_path) == ""
    (tmp_path / "room.md").write_text("Be concise. Cite sources.\n")
    assert load_room_prompt(tmp_path) == "Be concise. Cite sources."


def test_room_md_is_not_an_agent(tmp_path):
    # room.md sits beside agent folders; it must not be loaded as a persona.
    write_agent(tmp_path, "perf")
    (tmp_path / "room.md").write_text("house rules")
    assert set(discover_personas(tmp_path)) == {"perf"}

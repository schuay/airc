# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Where personas come from, and how the plugin's hook reaches the config.

Persona resolution runs before any plugin config is parsed, because personas
have to exist before anything else is built. A plugin whose personas come from a
configured source rather than from its own package -- the coding app resolves
them out of a pinned project pack -- therefore has no other window onto the
config than the one this hook opens.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from airc_room.cli import _resolve_agents_dir
from airc_room.config import CONFIG_DIR, Config


class _OldPlugin:
    """Written against the original no-argument contract."""

    def __init__(self, path):
        self.path = path

    def personas_dir(self):
        return self.path


class _CfgPlugin:
    def __init__(self):
        self.seen = None

    def personas_dir(self, cfg=None):
        self.seen = cfg
        return Path(cfg.raw["prompt_pack"]["agents"])


class _KwargsPlugin:
    def personas_dir(self, **kw):
        return Path(kw["cfg"].raw["prompt_pack"]["agents"])


def _args(agents_dir=None):
    return argparse.Namespace(agents_dir=agents_dir)


def test_the_hook_receives_the_config_when_it_takes_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no ./agents, so the plugin gets asked
    cfg = Config()
    cfg.raw = {"prompt_pack": {"agents": str(tmp_path / "packagents")}}
    plugin = _CfgPlugin()

    assert _resolve_agents_dir(_args(), plugin, cfg) == tmp_path / "packagents"
    assert plugin.seen is cfg


def test_a_kwargs_hook_receives_it_too(tmp_path, monkeypatch):
    # **kwargs is the forward-compat idiom, so a plugin that wrote it precisely
    # to receive a later addition must not be the one shape that misses it.
    monkeypatch.chdir(tmp_path)
    cfg = Config()
    cfg.raw = {"prompt_pack": {"agents": str(tmp_path / "packagents")}}

    assert _resolve_agents_dir(_args(), _KwargsPlugin(), cfg) == tmp_path / "packagents"


def test_a_hook_without_cfg_still_supplies_its_personas(tmp_path, monkeypatch):
    """The compatibility property that lets this land without a
    PLUGIN_API_VERSION bump."""
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_agents_dir(_args(), _OldPlugin(tmp_path / "packaged"), Config())

    assert resolved == tmp_path / "packaged"


def test_explicit_and_local_still_outrank_the_plugin(tmp_path, monkeypatch):
    plugin = _OldPlugin(tmp_path / "packaged")
    assert _resolve_agents_dir(_args(tmp_path / "cli"), plugin, Config()) == (
        tmp_path / "cli"
    )

    monkeypatch.chdir(tmp_path)
    (tmp_path / "agents").mkdir()
    assert _resolve_agents_dir(_args(), plugin, Config()) == tmp_path / "agents"


def test_no_plugin_falls_back_to_the_config_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert _resolve_agents_dir(_args(), None, Config()) == CONFIG_DIR / "agents"


def test_load_config_carries_the_whole_file(tmp_path):
    """A plugin owning TOP-LEVEL sections core does not model has nowhere else to
    read them from: plugin_config carries only what is left inside [airc]."""
    from airc_room.config import load_config

    path = tmp_path / "airc.toml"
    path.write_text('[prompt_pack]\nproject = "v8"\n\n[airc]\nplugin_module = "x"\n')

    cfg = load_config(path)

    assert cfg.raw["prompt_pack"] == {"project": "v8"}
    assert "prompt_pack" not in cfg.plugin_config

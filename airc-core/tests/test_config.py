"""Tests for the shared config substrate."""

from __future__ import annotations

from pathlib import Path

from airc_core import (
    DEFAULT_BUS_ROOT,
    DEFAULT_TOOL_GROUPS,
    CommonConfig,
    apply_gcp_env_defaults,
    load_common,
)


def test_empty_raw_uses_defaults():
    cfg = load_common({})
    assert cfg.models == {}
    assert cfg.mcp_servers == {}
    assert cfg.tool_groups == DEFAULT_TOOL_GROUPS
    assert cfg.bus_root == DEFAULT_BUS_ROOT
    assert cfg.caching_explicit is True
    assert cfg.cache_ttl_minutes == 30


def test_models_table_kept_raw_for_per_component_selection():
    # Components select their own key from one shared [models] section.
    cfg = load_common({"models": {"default": "a", "filter": "b", "review": "c"}})
    assert cfg.models == {"default": "a", "filter": "b", "review": "c"}


def test_mcp_servers_and_paths_parsed():
    raw = {
        "bus_root": "~/somewhere/bus",
        "token_db_path": "~/somewhere/tokens.db",
        "mcp": {
            "servers": {
                "v8-utils": {
                    "command": "v8-mcp",
                    "args": ["--enable-pd"],
                    "enable_in_sandbox": True,
                },
                "buganizer": {"command": "bug-mcp"},
            }
        },
        "repos": {"v8": "/path/to/v8/v8"},
    }
    cfg = load_common(raw)
    assert cfg.mcp_servers["v8-utils"]["args"] == ["--enable-pd"]
    assert "enable_in_sandbox" not in cfg.mcp_servers["v8-utils"]
    assert cfg.mcp_enable_in_sandbox == {
        "v8-utils": True,
        "buganizer": False,
    }
    assert cfg.bus_root == Path("~/somewhere/bus").expanduser()
    assert cfg.token_db_path == Path("~/somewhere/tokens.db").expanduser()
    assert cfg.repos == {"v8": str(Path("/path/to/v8/v8").expanduser())}


def test_tool_groups_override_replaces_default():
    cfg = load_common({"tool_groups": {"read": ["repo_git_*"]}})
    assert cfg.tool_groups == {"read": ["repo_git_*"]}


def test_gcp_values_coerced_to_str():
    # A bare project number in TOML is an int; os.environ would reject it.
    cfg = load_common({"gcp": {"project": 12345}})
    assert cfg.gcp == {"project": "12345"}


def test_apply_gcp_env_defaults_only_fills_unset(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_QUOTA_PROJECT", raising=False)
    apply_gcp_env_defaults({"project": "p", "location": "us-central1"})
    import os

    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "p"
    assert os.environ["GOOGLE_CLOUD_LOCATION"] == "us-central1"
    # quota project falls back to project when unset
    assert os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] == "p"


def test_apply_gcp_env_defaults_external_override_wins(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "already-set")
    apply_gcp_env_defaults({"project": "from-config"})
    import os

    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "already-set"


def test_common_config_is_constructible_directly():
    # Components may build one in tests without going through TOML.
    cfg = CommonConfig(bus_root=Path("/tmp/bus"))
    assert cfg.bus_root == Path("/tmp/bus")

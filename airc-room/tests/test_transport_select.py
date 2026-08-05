# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Transport-kind resolution: flags, config, and the TTY/plugin default.

_resolve_transport_kind is the load-bearing prod-path branch that decides which
frontend the room binds; these pin its precedence so a config or flag change
cannot silently swing prod onto the wrong transport. Core names no transport
itself -- the headless default comes from the plugin's default_transport_kind,
so these fake a plugin object rather than a plugin-module string.
"""

import argparse
from types import SimpleNamespace

import pytest
from airc_room.cli import _resolve_transport_kind


def _args(chat=False, headless=False):
    return argparse.Namespace(chat=chat, headless=headless)


def _cfg(transport_kind="", plugin_module="airc_coding.app"):
    return SimpleNamespace(transport_kind=transport_kind, plugin_module=plugin_module)


def _plugin(default="gchat"):
    # A plugin that declares gchat as its headless default (the coding app). None
    # models a bare room / a plugin that declares no default.
    if default is None:
        return None
    return SimpleNamespace(default_transport_kind=lambda: default)


def test_chat_flag_wins_over_everything(monkeypatch):
    # Legacy --chat maps to gchat even if config names something else.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert (
        _resolve_transport_kind(_args(chat=True), _cfg("console"), _plugin()) == "gchat"
    )


def test_headless_flag(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert (
        _resolve_transport_kind(_args(headless=True), _cfg(), _plugin()) == "headless"
    )


def test_config_kind_wins_over_tty_default(monkeypatch):
    # An explicit [transport] kind overrides the non-TTY plugin default, so a
    # headless coding deploy can still pin console.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert _resolve_transport_kind(_args(), _cfg("console"), _plugin()) == "console"


def test_interactive_coding_defaults_to_console(monkeypatch):
    # A TTY run with no flags/config is the dev console, NOT the plugin default --
    # the isatty check must precede the plugin default.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert _resolve_transport_kind(_args(), _cfg(), _plugin()) == "console"


def test_headless_uses_plugin_default_with_nudge(monkeypatch, caplog):
    # A non-TTY deploy with no [transport] binds whatever the plugin declares as
    # its default (gchat for the coding app), and logs a nudge to set it
    # explicitly. Core does not name the transport -- the plugin does.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    import logging

    with caplog.at_level(logging.WARNING):
        assert _resolve_transport_kind(_args(), _cfg(), _plugin()) == "gchat"
    assert "no [transport] kind set" in caplog.text


def test_headless_no_plugin_default_falls_back_to_console(monkeypatch):
    # A bare room (no plugin) or a plugin that declares no default has no network
    # transport to fall back to.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert _resolve_transport_kind(_args(), _cfg(plugin_module=""), None) == "console"
    assert _resolve_transport_kind(_args(), _cfg(), _plugin(default=None)) == "console"


@pytest.mark.parametrize("kind", ["gchat", "matrix", "weird"])
def test_config_kind_passed_through_verbatim(monkeypatch, kind):
    # Resolution does not judge the kind (cli.py binds/errors on it); it just
    # returns what config named, over the default.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert _resolve_transport_kind(_args(), _cfg(kind), _plugin()) == kind

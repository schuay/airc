# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

import logging

from airc_core.logs import _NOISY, quiet_noisy_loggers


def _levels() -> dict[str, int]:
    return {name: logging.getLogger(name).level for name in _NOISY}


def _restore(saved: dict[str, int]) -> None:
    for name, level in saved.items():
        logging.getLogger(name).setLevel(level)


def test_the_noisy_loggers_are_raised_to_warning():
    saved = _levels()
    try:
        for name in _NOISY:
            logging.getLogger(name).setLevel(logging.NOTSET)
        quiet_noisy_loggers()
        for name in _NOISY:
            assert logging.getLogger(name).level == logging.WARNING, name
    finally:
        _restore(saved)


def test_httpx2_is_covered():
    """httpx 2.x ships as its own package and logs under its own name, so the
    "httpx" entry does not cover it. It is the client the Anthropic, Google and
    OpenAI SDKs use, i.e. one INFO line per model call."""
    assert "httpx" in _NOISY and "httpx2" in _NOISY


def test_a_debug_root_is_left_alone(monkeypatch):
    """Debug output is asked for when transport, auth or sandbox egress is
    being chased, which is what these loggers carry."""
    saved = _levels()
    root = logging.getLogger()
    root_level = root.level
    try:
        for name in _NOISY:
            logging.getLogger(name).setLevel(logging.NOTSET)
        root.setLevel(logging.DEBUG)
        quiet_noisy_loggers()
        for name in _NOISY:
            assert logging.getLogger(name).level == logging.NOTSET, name
    finally:
        root.setLevel(root_level)
        _restore(saved)


def test_exported_from_the_package():
    """The entry points import it from airc_core, through the lazy map."""
    import airc_core

    assert airc_core.quiet_noisy_loggers is quiet_noisy_loggers

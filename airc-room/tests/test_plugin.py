# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""The plugin contract validation: a loaded plugin must supply the three
factories and (if it declares one) a compatible API version, else load fails
loudly at startup rather than mysteriously at first use."""

from types import SimpleNamespace

import pytest
from airc_room.plugin import PLUGIN_API_VERSION, validate_plugin


def _ok_module(**extra):
    # A module with the three required factories present and callable.
    base = dict(  # noqa: C408 -- kwargs form mirrors the module attrs it fakes
        build_subscribers=lambda *a, **k: [],
        build_follow_ups=lambda *a, **k: {},
        build_transport=lambda *a, **k: None,
    )
    base.update(extra)
    return SimpleNamespace(**base)


def test_valid_plugin_passes():
    validate_plugin(_ok_module(), "some.plugin")


def test_missing_factory_is_rejected_by_name():
    mod = _ok_module()
    del mod.build_transport
    with pytest.raises(SystemExit, match="build_transport"):
        validate_plugin(mod, "some.plugin")


def test_non_callable_factory_is_rejected():
    mod = _ok_module(build_subscribers="not a function")
    with pytest.raises(SystemExit, match="build_subscribers"):
        validate_plugin(mod, "some.plugin")


def test_matching_declared_version_passes():
    validate_plugin(_ok_module(PLUGIN_API_VERSION=PLUGIN_API_VERSION), "some.plugin")


def test_mismatched_version_is_rejected():
    with pytest.raises(SystemExit, match="plugin API version"):
        validate_plugin(_ok_module(PLUGIN_API_VERSION=PLUGIN_API_VERSION + 1), "p")


def test_absent_version_is_tolerated():
    # A plugin predating the version field is allowed (in-tree, lockstep-versioned);
    # only a declared, mismatched version is a hard error.
    validate_plugin(_ok_module(), "some.plugin")

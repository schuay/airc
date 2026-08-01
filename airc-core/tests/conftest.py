# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

import pytest


@pytest.fixture(autouse=True)
def _restore_vertex_global_config():
    """Undo writes to the Vertex SDK's process-global initializer.

    _seed_vertex_cache_globals sets credentials/location/project on a module
    singleton that no test owns and monkeypatch does not cover. Left set, a live
    credential outlives the test that made it and any later test of the
    ADC-fallback path silently passes for the wrong reason, depending on run
    order. Restores on failure too, which the ad-hoc resets inside the tests
    cannot.
    """
    from google.cloud.aiplatform import initializer

    cfg = initializer.global_config
    saved = (cfg._credentials, cfg._location, cfg._project)
    try:
        yield
    finally:
        cfg._credentials, cfg._location, cfg._project = saved

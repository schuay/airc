# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Shared substrate for the airc daemon suite."""

from .agent import (
    CallBudgetMiddleware,
    EmptyCandidateError,
    RequireStructuredResultMiddleware,
    TimeBudgetMiddleware,
    base_middleware,
    growing_cache_middleware,
    retrying,
)
from .artifacts import ArtifactLog, slug
from .config import (
    DATA_DIR,
    DEFAULT_BUS_ROOT,
    DEFAULT_TOKEN_DB,
    DEFAULT_TOOL_GROUPS,
    CommonConfig,
    apply_gcp_env_defaults,
    load_common,
)
from .mcptools import MCPToolset
from .model import (
    SUPPORTED_PROVIDERS,
    check_model_id,
    list_models,
    make_model,
    missing_key,
    supported_models_hint,
    usage_counts,
)
from .tokens import TokenLog

__all__ = [
    "CallBudgetMiddleware",
    "EmptyCandidateError",
    "RequireStructuredResultMiddleware",
    "TimeBudgetMiddleware",
    "base_middleware",
    "growing_cache_middleware",
    "retrying",
    "ArtifactLog",
    "slug",
    "DATA_DIR",
    "DEFAULT_BUS_ROOT",
    "DEFAULT_TOKEN_DB",
    "DEFAULT_TOOL_GROUPS",
    "CommonConfig",
    "apply_gcp_env_defaults",
    "load_common",
    "MCPToolset",
    "TokenLog",
    "SUPPORTED_PROVIDERS",
    "check_model_id",
    "list_models",
    "make_model",
    "missing_key",
    "supported_models_hint",
    "usage_counts",
]

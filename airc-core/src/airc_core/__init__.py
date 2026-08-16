# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Shared substrate for the airc daemon suite."""

from .agent import (
    CallBudgetMiddleware,
    EmptyCandidateError,
    GroundingReminderMiddleware,
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
    HandoverFields,
    apply_gcp_env_defaults,
    load_common,
    parse_handover_fields,
)
from .mcptools import MCPToolset
from .model import (
    SUPPORTED_PROVIDERS,
    check_model_id,
    list_models,
    make_model,
    missing_key,
    register_provider,
    supported_models_hint,
    usage_counts,
)
from .tokens import TokenLog

__all__ = [
    "DATA_DIR",
    "DEFAULT_BUS_ROOT",
    "DEFAULT_TOKEN_DB",
    "DEFAULT_TOOL_GROUPS",
    "SUPPORTED_PROVIDERS",
    "ArtifactLog",
    "CallBudgetMiddleware",
    "CommonConfig",
    "EmptyCandidateError",
    "GroundingReminderMiddleware",
    "HandoverFields",
    "MCPToolset",
    "RequireStructuredResultMiddleware",
    "TimeBudgetMiddleware",
    "TokenLog",
    "apply_gcp_env_defaults",
    "base_middleware",
    "check_model_id",
    "growing_cache_middleware",
    "list_models",
    "load_common",
    "make_model",
    "missing_key",
    "parse_handover_fields",
    "register_provider",
    "retrying",
    "slug",
    "supported_models_hint",
    "usage_counts",
]

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Shared substrate for the airc daemon suite."""

# Resolved per name on first access rather than eagerly, because the substrate is
# shared by components that use very different parts of it. Naming load_common --
# which every component does, to read the suite file -- used to import the agent
# middleware, the MCP toolset and the structured-task runner too, roughly 700ms of
# langchain for a config parse. A CLI listing job states pays none of it now.
_LAZY = {
    "ArtifactLog": ".artifacts",
    "CallBudgetMiddleware": ".agent",
    "CommonConfig": ".config",
    "DATA_DIR": ".config",
    "DEFAULT_BUS_ROOT": ".config",
    "DEFAULT_TOKEN_DB": ".config",
    "DEFAULT_TOOL_GROUPS": ".config",
    "EFFORT_LEVELS": ".providers",
    "EmptyCandidateError": ".agent",
    "FinalAnswerMiddleware": ".agent",
    "GroundingReminderMiddleware": ".agent",
    "HandoverFields": ".config",
    "MCPToolset": ".mcptools",
    "ModelProfile": ".config",
    "RequireStructuredResultMiddleware": ".agent",
    "SUPPORTED_PROVIDERS": ".model",
    "StructuredTaskError": ".structured",
    "StructuredTaskRunner": ".structured",
    "TimeBudgetMiddleware": ".agent",
    "TokenLog": ".tokens",
    "Usage": ".usage",
    "UsageCollector": ".collector",
    "apply_gcp_env_defaults": ".config",
    "base_middleware": ".agent",
    "check_model_id": ".model",
    "growing_cache_middleware": ".agent",
    "list_models": ".model",
    "load_common": ".config",
    "make_model": ".model",
    "price_for": ".pricing",
    "missing_key": ".model",
    "parse_handover_fields": ".config",
    "quiet_noisy_loggers": ".logs",
    "register_provider": ".model",
    "retrying": ".agent",
    "slug": ".artifacts",
    "supported_models_hint": ".model",
}


def __getattr__(name: str):
    where = _LAZY.get(name)
    if where is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(where, __name__), name)
    # __getattr__ runs only on a miss, so binding it here retires the lookup.
    globals()[name] = value
    return value


__all__ = [
    "DATA_DIR",
    "DEFAULT_BUS_ROOT",
    "DEFAULT_TOKEN_DB",
    "DEFAULT_TOOL_GROUPS",
    "EFFORT_LEVELS",
    "SUPPORTED_PROVIDERS",
    "ArtifactLog",
    "CallBudgetMiddleware",
    "CommonConfig",
    "EmptyCandidateError",
    "FinalAnswerMiddleware",
    "GroundingReminderMiddleware",
    "HandoverFields",
    "MCPToolset",
    "ModelProfile",
    "RequireStructuredResultMiddleware",
    "StructuredTaskError",
    "StructuredTaskRunner",
    "TimeBudgetMiddleware",
    "TokenLog",
    "Usage",
    "UsageCollector",
    "apply_gcp_env_defaults",
    "base_middleware",
    "check_model_id",
    "growing_cache_middleware",
    "list_models",
    "load_common",
    "make_model",
    "missing_key",
    "parse_handover_fields",
    "price_for",
    "quiet_noisy_loggers",
    "register_provider",
    "retrying",
    "slug",
    "supported_models_hint",
]

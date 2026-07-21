"""Shared configuration substrate for the daemon suite.

The suite runs as several processes (airc, airc-watchers, airc-processors) that
must agree on a handful of facts: where the bus lives, which token ledger they
all write, the GCP project, the MCP servers and tool groups, the repo->path map.
A drifted `bus_root` or `[gcp]` between two component configs is a silent footgun
(producers and consumers miss each other on disk; Vertex auth fails in one
daemon but not another). So the parsing of those shared sections lives here, in
the one package every component already imports, and each component overlays its
own keys on top.

This module must not import any component package (airc, airc-processors,
airc-watchers) -- the dependency only ever points inward, into core.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_data_path

DATA_DIR = user_data_path("airc")
DEFAULT_BUS_ROOT = DATA_DIR / "bus"
DEFAULT_TOKEN_DB = DATA_DIR / "tokens.db"

# Maps a short group name (referenced from agent configs) to fnmatch patterns
# over MCP tool names. Tools are named as <server_name>__<tool_name> to avoid
# collisions across servers (e.g. v8-utils__run_d8, gdb-mcp__backtrace).
# "read" tools only inspect state (pd queries pre-recorded perf history, so it
# is read); "active" tools execute code/benchmarks on this machine -- expensive
# and measurement-sensitive, granted to no agent by default. The single source
# of truth for the suite; components select the groups they grant. Strictly
# read-only: gerrit_create_comments (posts drafts) and pinpoint_create_job/
# cancel_job (mutate try jobs) are deliberately NOT matched, hence enumerated
# patterns rather than gerrit_*/pinpoint_*. d8_trace_index is excluded: it
# reads an arbitrary caller-supplied path.
# Core ships no tool groups: the substrate is domain-neutral, so the default is
# empty and every app supplies its own [tool_groups] via config (the coding app's
# v8-utils/gdb groups live in airc_coding.tool_groups, written into airc.toml).
# The keys stay present so a config that sets only one group still has the other.
DEFAULT_TOOL_GROUPS: dict[str, list[str]] = {"read": [], "active": []}


@dataclass
class CommonConfig:
    """The sections shared across every component config.

    `models` is the raw `[models]` table rather than resolved ids: components
    legitimately want different defaults from one shared section (airc a cheap
    conversational model, the processor a capable review model), so each selects
    the key it needs (`models["default"]`, `models["filter"]`, ...).
    """

    models: dict[str, str] = field(default_factory=dict)
    mcp_servers: dict[str, dict] = field(default_factory=dict)
    mcp_enable_in_sandbox: dict[str, bool] = field(default_factory=dict)
    tool_groups: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_TOOL_GROUPS.items()}
    )
    gcp: dict[str, str] = field(default_factory=dict)
    bus_root: Path = field(default_factory=lambda: DEFAULT_BUS_ROOT)
    token_db_path: Path = field(default_factory=lambda: DEFAULT_TOKEN_DB)
    repos: dict[str, str] = field(default_factory=dict)  # logical name -> checkout
    caching_explicit: bool = True
    cache_ttl_minutes: int = 30


def load_common(raw: Mapping) -> CommonConfig:
    """Parse the shared sections out of an already-parsed TOML mapping.

    Takes the parsed dict (not a path) so a component reads its file once and
    hands the same mapping here and to its own overlay parser.
    """
    cfg = CommonConfig()
    cfg.models = {k: str(v) for k, v in raw.get("models", {}).items()}
    for name, server in raw.get("mcp", {}).get("servers", {}).items():
        spec = dict(server)
        enabled = spec.pop("enable_in_sandbox", False)
        if not isinstance(enabled, bool):
            raise ValueError(f"mcp.servers.{name}.enable_in_sandbox must be a boolean")
        cfg.mcp_servers[name] = spec
        cfg.mcp_enable_in_sandbox[name] = enabled
    if groups := raw.get("tool_groups"):
        cfg.tool_groups = {k: list(v) for k, v in groups.items()}
    cfg.gcp = {k: str(v) for k, v in raw.get("gcp", {}).items()}
    if v := raw.get("bus_root"):
        cfg.bus_root = Path(v).expanduser()
    if v := raw.get("token_db_path"):
        cfg.token_db_path = Path(v).expanduser()
    cfg.repos = {k: str(Path(v).expanduser()) for k, v in raw.get("repos", {}).items()}
    if caching := raw.get("caching"):
        cfg.caching_explicit = bool(caching.get("explicit", True))
        cfg.cache_ttl_minutes = int(caching.get("ttl_minutes", 30))
    return cfg


def apply_gcp_env_defaults(gcp: Mapping[str, str]) -> None:
    """Apply [gcp] config as GOOGLE_CLOUD_* env defaults for Vertex AI.

    Only fills unset variables so an external override wins. Vertex model
    classes read these for project/location; auth itself comes from ADC
    (gcloud auth application-default login). A google_vertexai:* default model
    fails with "Unable to find your project" without this or the env set some
    other way.
    """
    for key, env in (
        ("project", "GOOGLE_CLOUD_PROJECT"),
        ("location", "GOOGLE_CLOUD_LOCATION"),
        ("quota_project", "GOOGLE_CLOUD_QUOTA_PROJECT"),
    ):
        if env not in os.environ and (val := gcp.get(key)):
            os.environ[env] = str(val)  # a TOML int/float would crash os.environ
    if "GOOGLE_CLOUD_QUOTA_PROJECT" not in os.environ:
        if proj := os.environ.get("GOOGLE_CLOUD_PROJECT"):
            os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = proj

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path

from platformdirs import user_data_path

from .model import register_provider

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


@dataclass(frozen=True)
class HandoverFields:
    """Shared [handover] fields before component-specific path handling."""

    enabled: bool
    autonomy: str
    bus_root: str | None
    kinds: list[str]


def reject_unknown_fields(
    table: Mapping, spec: type, where: str, *, aliases: Iterable[str] = ()
) -> None:
    """Strict-check `table` against the dataclass that models it.

    The allowed keys ARE `spec`'s field names -- the dataclass is the single
    source of truth, so adding a setting means adding a field and nothing else.
    Hand-written key sets were the obvious alternative and the wrong one: they
    restate the fields a few lines below their definition, and the copy drifts
    silently in the direction that matters (a field added without its key gets
    rejected in config that legitimately sets it).

    `aliases` covers keys a section accepts that are not fields -- deliberate
    back-compat spellings, e.g. [airc.orchestrator] still honouring the old
    `turn_budget` for `soft_turn_budget`. Naming them here keeps each one visible
    as a decision rather than a leftover.

    Sections with no dataclass of their own (a flattened one like [caching], whose
    keys land on differently-named fields of a larger config) call reject_unknown
    with an explicit set instead. There is no source of truth to derive from
    there, and inventing one would be a worse lie than writing the keys down.
    """
    known = {f.name for f in fields(spec)} | set(aliases)
    reject_unknown(table, known, where)


def reject_unknown(
    table: Mapping, known: set[str] | frozenset[str], where: str
) -> None:
    """Raise on any key in `table` that is not in `known`. `where` names the
    section, e.g. "[airc.perf]".

    Config is the one input a running daemon cannot argue with, and a key it
    silently ignores is indistinguishable from one it honoured -- the operator
    reads their own file back as proof of a setting that was never applied. Most
    of the time that costs a default nobody wanted; at least once it cost a
    security boundary (a misspelled `spaces` left a publish-to-gerrit allowlist
    empty, which means unrestricted). Since the two are indistinguishable at the
    point of the typo, every section is strict.

    SystemExit rather than an exception: this runs during startup config parsing,
    where a traceback buries the one line the operator needs.

    Sections that are open by design -- user-named maps like [repos],
    [tool_groups], [mcp.servers], and role maps like [models] -- do not call this,
    and each says why at its parse site.
    """
    if unknown := set(table) - set(known):
        raise SystemExit(
            f"unknown {where} key(s): {', '.join(sorted(unknown))}"
            f" (known: {', '.join(sorted(known))})"
        )


def parse_handover_fields(
    h: Mapping, *, error: type[BaseException] = SystemExit
) -> HandoverFields:
    """Parse the shared [handover] shape without owning its kind vocabulary."""
    if "repro_only" in h:
        raise error(
            "[handover] repro_only is gone: allowlist the kinds instead -- "
            'kinds = ["repro"] is the old repro_only = true'
        )
    if "repro" in h:
        raise error(
            "[handover] repro is gone: allowlisting the kind replaces it -- add"
            ' "repro" to the kinds array (a repro-suitable finding then takes'
            " the verified-repro detour instead of a direct fix)"
        )
    reject_unknown_fields(h, HandoverFields, "[handover]")
    kinds_raw = h.get("kinds")
    if kinds_raw is not None and (
        isinstance(kinds_raw, str) or not isinstance(kinds_raw, list)
    ):
        raise error('[handover] kinds must be a list, e.g. kinds = ["bugfix", "repro"]')
    enabled = bool(h.get("enabled", False))
    if kinds_raw is None and enabled:
        raise error(
            "[handover] enabled = true without kinds: state the allowlist. The"
            ' default is just ["repro"]; every kind that can produce a CL'
            " (bugfix, perf, task) is opt-in now. E.g."
            ' kinds = ["bugfix", "repro", "perf", "task"] restores the'
            ' pre-kinds behaviour, kinds = ["repro"] is the old'
            " repro_only = true."
        )
    return HandoverFields(
        enabled=enabled,
        autonomy=h.get("autonomy", "draft-only"),
        bus_root=str(h["bus_root"]) if h.get("bus_root") else None,
        kinds=[str(k) for k in kinds_raw] if kinds_raw is not None else ["repro"],
    )


@dataclass
class CommonConfig:
    """The sections shared across every component config.

    `models` is the raw `[models]` table rather than resolved ids: components
    legitimately want different defaults from one shared section (airc a cheap
    conversational model, the processor a capable review model), so each selects
    the key it needs (`models["default"]`, `models["filter"]`, ...).
    """

    models: dict[str, str] = field(default_factory=dict)
    #: [model_providers] verbatim, prefix -> spec. Kept on the config as well as
    #: registered in airc_core.model, so a component can SEE what was declared
    #: (an inspector, a test) without reading module state it does not own.
    model_providers: dict[str, dict] = field(default_factory=dict)
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


def _load_model_providers(raw: Mapping, cfg: CommonConfig) -> None:
    """Parse [model_providers] and register each one with airc_core.model.

    Registering as a SIDE EFFECT of parsing, which is the one thing here worth
    knowing about. The alternative -- return the specs and have each component
    register -- needs the call added at four entry points (the room, the
    processor, the watchers, icompleteu), and a missed one fails asymmetrically:
    the room starts on a config the processor rejects, for the same file. Since
    make_model is a free function with no cfg in scope, module state is where
    the registration has to land either way, and load_common is the single point
    every component already passes through.

    The table is user-named ([model_providers.<prefix>]), so the section itself
    is open; each SPEC is strict, for the reason reject_unknown states -- a
    misspelled requires_env silently means "no credential check" and reads back
    as if it were honoured.
    """
    for prefix, spec in raw.get("model_providers", {}).items():
        where = f"[model_providers.{prefix}]"
        if not isinstance(spec, Mapping):
            raise SystemExit(f"{where} must be a table (factory = 'module:attr')")
        reject_unknown(spec, {"factory", "requires_env"}, where)
        if not (factory := spec.get("factory")):
            raise SystemExit(f"{where} needs factory = 'module:attr'")
        cfg.model_providers[prefix] = dict(spec)
        requires_env = spec.get("requires_env")
        try:
            register_provider(
                prefix,
                str(factory),
                requires_env=str(requires_env) if requires_env else None,
            )
        except ValueError as e:
            # SystemExit, like every other config error here: this runs during
            # startup parsing, where a traceback buries the one line naming the
            # section the operator has to fix.
            raise SystemExit(f"{where}: {e}") from e


def load_common(raw: Mapping) -> CommonConfig:
    """Parse the shared sections out of an already-parsed TOML mapping.

    Takes the parsed dict (not a path) so a component reads its file once and
    hands the same mapping here and to its own overlay parser.
    """
    cfg = CommonConfig()
    # [models] is deliberately OPEN: it is a role map, and a persona's `model =`
    # may name any role in it (resolve_model). Only default/filter are read here,
    # but constraining the table would reject a role a persona legitimately uses.
    cfg.models = {k: str(v) for k, v in raw.get("models", {}).items()}
    _load_model_providers(raw, cfg)
    if mcp := raw.get("mcp"):
        reject_unknown(mcp, {"servers"}, "[mcp]")
    # A server spec is passed verbatim to MultiServerMCPClient, whose key set is
    # that library's, not ours -- so only our own added key is checked, by the
    # explicit pop and type check below.
    for name, server in raw.get("mcp", {}).get("servers", {}).items():
        spec = dict(server)
        enabled = spec.pop("enable_in_sandbox", False)
        if not isinstance(enabled, bool):
            raise TypeError(f"mcp.servers.{name}.enable_in_sandbox must be a boolean")
        cfg.mcp_servers[name] = spec
        cfg.mcp_enable_in_sandbox[name] = enabled
    if groups := raw.get("tool_groups"):
        cfg.tool_groups = {k: list(v) for k, v in groups.items()}
    if gcp := raw.get("gcp"):
        reject_unknown(gcp, {"project", "location", "quota_project"}, "[gcp]")
    cfg.gcp = {k: str(v) for k, v in raw.get("gcp", {}).items()}
    if v := raw.get("bus_root"):
        cfg.bus_root = Path(v).expanduser()
    if v := raw.get("token_db_path"):
        cfg.token_db_path = Path(v).expanduser()
    cfg.repos = {k: str(Path(v).expanduser()) for k, v in raw.get("repos", {}).items()}
    if caching := raw.get("caching"):
        reject_unknown(caching, {"explicit", "ttl_minutes"}, "[caching]")
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
    if "GOOGLE_CLOUD_QUOTA_PROJECT" not in os.environ and (
        proj := os.environ.get("GOOGLE_CLOUD_PROJECT")
    ):
        os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = proj

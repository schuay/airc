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

import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path

from platformdirs import user_data_path

from .pricing import price_for
from .providers import EFFORT_LEVELS, traits_for

log = logging.getLogger(__name__)

DATA_DIR = user_data_path("airc")
DEFAULT_BUS_ROOT = DATA_DIR / "bus"
DEFAULT_TOKEN_DB = DATA_DIR / "tokens.db"
DEFAULT_ARTIFACTS_DIR = DATA_DIR / "artifacts"

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


@dataclass(frozen=True)
class ModelProfile:
    """One `[models]` entry: which model, and how hard it is told to think.

    A profile rather than a bare id because "the verify model" is a thing a
    deployment names, and its identity is not just the checkpoint -- the same
    Opus 5 at `low` and at `xhigh` differ by more in cost and in behaviour than
    two sibling checkpoints do. With ids alone the only way to say that was to
    repeat the literal at each call site and set the depth nowhere, which is how
    the review roster and the verify stage ended up unable to differ.

    Two entries SHARING an id is therefore normal and intended, not duplication
    to factor out.

    `call_kwargs` is make_model's keyword surface, not the provider's: the
    translation from a level to whatever the provider calls it lives there,
    behind one name, so config never learns that one Claude route has an
    `effort` field and the other reaches the API through model_kwargs.
    """

    key: str
    id: str
    effort: str | None = None

    @property
    def call_kwargs(self) -> dict:
        return {"effort": self.effort} if self.effort else {}


# Knobs a [models] entry may carry besides the id. One name per provider-native
# parameter; Gemini's thinking_budget is the obvious next one and is deliberately
# absent until something runs it, since an accepted-but-unused knob reads exactly
# like an honoured one.
_PROFILE_KEYS = frozenset({"id", "effort"})


def _parse_model_profile(key: str, value: object) -> ModelProfile:
    """One [models] entry, as either a bare id or a table.

    The string form stays first-class rather than deprecated: most entries have
    nothing to say beyond which model, and making them all grow a table would
    add a line of noise per entry to buy nothing.
    """
    where = f"[models] {key}"
    if isinstance(value, str):
        return ModelProfile(key=key, id=value)
    if not isinstance(value, Mapping):
        raise SystemExit(
            f'{where} must be a model id or a table with id = "provider:model"'
        )
    reject_unknown(value, _PROFILE_KEYS, f"[models.{key}]")
    if not (model_id := value.get("id")):
        raise SystemExit(f'[models.{key}] needs id = "provider:model"')
    effort = value.get("effort")
    if effort is not None:
        effort = str(effort)
        if effort not in EFFORT_LEVELS:
            raise SystemExit(
                f"[models.{key}] effort = {effort!r} is not a level:"
                f" expected one of {', '.join(EFFORT_LEVELS)}"
            )
        # Checked here as well as in make_model because this is the failure the
        # operator can act on: it names the section to edit, at startup, instead
        # of at the first call the daemon happens to make.
        if not traits_for(str(model_id)).supports_effort:
            raise SystemExit(
                f"[models.{key}] effort is a Claude parameter and {model_id}"
                " is not a Claude model. Gemini's thinking depth is a token"
                " budget, not a level, and the two are not interconvertible."
            )
    return ModelProfile(key=key, id=str(model_id), effort=effort)


@dataclass
class CommonConfig:
    """The sections shared across every component config.

    `models` is the raw `[models]` table rather than resolved ids: components
    legitimately want different defaults from one shared section (airc a cheap
    conversational model, the processor a capable review model), so each selects
    the key it needs (`models["default"]`, `models["filter"]`, ...).

    `models` holds ids alone and `model_profiles` the full entries, the same set
    keyed the same way. Both, because readers genuinely want different things:
    the sandbox's egress allowlists and the Claude-needs-a-proxy check want every
    model this deploy can reach and nothing else, while a component building a
    graph wants the knobs too. Neither is derivable from the other cheaply enough
    to be worth one of them being a function.
    """

    models: dict[str, str] = field(default_factory=dict)
    model_profiles: dict[str, ModelProfile] = field(default_factory=dict)
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
    #: Root for ArtifactLog renderings (a review trail, a report for work that
    #: had to be abandoned). Suite-wide like bus_root, and for the same reason:
    #: more than one component writes there, and a per-component key is how two
    #: of them end up writing to different directories after an operator moves
    #: one. None disables the trail entirely ("" in the config).
    artifacts_dir: Path | None = field(default_factory=lambda: DEFAULT_ARTIFACTS_DIR)
    repos: dict[str, str] = field(default_factory=dict)  # logical name -> checkout
    caching_explicit: bool = True
    cache_ttl_minutes: int = 30
    #: Rolling fleet-wide spend caps in USD over the last 24h and 7d. None is no
    #: cap; both are checked and either binds. Suite-wide for the reason
    #: bus_root and token_db_path are: more than one component reads them, and a
    #: per-component key is how two of them end up with different numbers --
    #: which also sum to a fleet total nobody chose.
    #:
    #: Read live at each admission, deliberately unlike Limits, which is stamped
    #: at enqueue so a job's behaviour is fixed at creation. A window cap is a
    #: property of the fleet at the moment work starts, not of the job, so
    #: raising a binding cap takes a config edit and a restart.
    daily_usd_cap: float | None = None
    weekly_usd_cap: float | None = None


def _warn_unpriced(models: Mapping[str, str]) -> None:
    """Name, at load, every [models] entry the price table has no listing for.

    Such a model is costed at the generic placeholder rate and every total it
    touches is marked estimated. That is the intended fallback, but it should
    be a known state rather than one discovered in a report, and startup is
    where the operator is looking.
    """
    for key, model_id in models.items():
        if price_for(model_id).generic:
            log.warning(
                "[models] %s = %s has no price listing; its cost is estimated at"
                " the generic rate",
                key,
                model_id,
            )


def refuse_unpriced(models: Mapping[str, str], what: str) -> None:
    """Refuse, at load, a dollar budget over a model the price table does not
    list. `what` names the budget in the error, e.g. "[processors.review]
    review_cost_limit".

    A warning is the right answer for an unlisted model in general (see
    _warn_unpriced): the generic rate exists so an unlisted model is still
    costed rather than dropped from every total, and an estimated row is
    labelled as one. It is the wrong answer for a BOUND. A budget on the
    placeholder rate reads as a dollar figure the operator chose and behaves as
    an arbitrary one: the placeholder is a round mid-market number, so the same
    "$25 a pass" is a different amount of work on every unlisted model and
    changes meaning the moment a listing is added. Refused at startup, where the
    operator is looking, rather than discovered in a report.
    """
    unpriced = sorted(f"{k} = {v}" for k, v in models.items() if price_for(v).generic)
    if not unpriced:
        return
    raise SystemExit(
        f"{what} is a dollar budget, and the price table has no listing for"
        f" {', '.join(unpriced)}: the bound would be on the generic placeholder"
        " rate rather than on what the model costs. Add the model to"
        " airc_core.pricing, or point the budgeted stage at a listed model."
    )


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
    if not raw.get("model_providers"):
        return
    # Deferred, and behind the early return: this is the only thing in config
    # that reaches the model layer, and reaching it costs a langchain import.
    # Every component calls load_common to read the suite file; almost none
    # declare a custom provider, and those should not pay for the ones that do.
    from .model import register_provider

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
    # Each ENTRY is strict, for the usual reason -- a misspelled knob reads back
    # as an honoured one.
    cfg.model_profiles = {
        k: _parse_model_profile(k, v) for k, v in raw.get("models", {}).items()
    }
    cfg.models = {k: p.id for k, p in cfg.model_profiles.items()}
    _warn_unpriced(cfg.models)
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
    # Presence-checked rather than truth-checked: "" is the documented spelling
    # for "no on-disk trail", and `if v :=` would silently leave the default.
    if "artifacts_dir" in raw:
        v = raw["artifacts_dir"]
        cfg.artifacts_dir = Path(v).expanduser() if v else None
    cfg.repos = {k: str(Path(v).expanduser()) for k, v in raw.get("repos", {}).items()}
    if caching := raw.get("caching"):
        reject_unknown(caching, {"explicit", "ttl_minutes"}, "[caching]")
        cfg.caching_explicit = bool(caching.get("explicit", True))
        cfg.cache_ttl_minutes = int(caching.get("ttl_minutes", 30))
    # Presence-checked, not truth-checked: 0 is a cap of zero (admit nothing),
    # a coherent thing to write while an incident is being looked at, and
    # `if v :=` would silently read it as unset.
    for key in ("daily_usd_cap", "weekly_usd_cap"):
        if key in raw:
            v = raw[key]
            setattr(cfg, key, None if v is None else float(v))
    if cfg.daily_usd_cap is not None or cfg.weekly_usd_cap is not None:
        # Worse here than at a pass: a window cap sums EVERY model's rows, so one
        # mis-priced model puts a placeholder number into the total that stalls
        # every capped stream in the suite.
        refuse_unpriced(cfg.models, "daily_usd_cap/weekly_usd_cap")
    _warn_explicit_vertex_cache(cfg)
    return cfg


def _warn_explicit_vertex_cache(cfg: CommonConfig) -> None:
    """Say loudly, at load, when a config would run the explicit Vertex
    context cache: a google_vertexai model with [caching] explicit on (the
    default). That path has been inactive since 2026-09 and its cost rule
    still carries a read-price ratio tuned by hand rather than read from the
    price table (agent.py _CACHE_READ_RATIO); enabling it again means
    checking that rule first. A warning rather than a refusal because a
    google_vertexai entry may serve a role that never builds an agent graph.
    """
    if not cfg.caching_explicit:
        return
    for key, model_id in cfg.models.items():
        if model_id.startswith("google_vertexai:"):
            log.warning(
                "[models] %s = %s with [caching] explicit on would run the"
                " explicit Vertex context cache, inactive since 2026-09; its cost"
                " rule predates the price table and must be re-checked before"
                " this deploy relies on it (agent.py _CACHE_READ_RATIO)",
                key,
                model_id,
            )


def apply_gcp_env_defaults(gcp: Mapping[str, str]) -> None:
    """Apply [gcp] config as GOOGLE_CLOUD_* env defaults for Vertex AI.

    Only fills unset variables so an external override wins. Vertex model
    classes read these for project/location; auth itself comes from ADC
    (gcloud auth application-default login). A google_vertexai:* or
    google_anthropic_vertex:* default model fails with "Unable to find your
    project" without this or the env set some other way.
    """
    for key, env in (
        ("project", "GOOGLE_CLOUD_PROJECT"),
        ("location", "GOOGLE_CLOUD_LOCATION"),
        ("quota_project", "GOOGLE_CLOUD_QUOTA_PROJECT"),
        # Client-stack selector for google_vertexai: ids ("vertexai"/"genai");
        # see model._google_sdk.
        ("sdk", "AIRC_GOOGLE_SDK"),
    ):
        if env not in os.environ and (val := gcp.get(key)):
            os.environ[env] = str(val)  # a TOML int/float would crash os.environ
    if "GOOGLE_CLOUD_QUOTA_PROJECT" not in os.environ and (
        proj := os.environ.get("GOOGLE_CLOUD_PROJECT")
    ):
        os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = proj

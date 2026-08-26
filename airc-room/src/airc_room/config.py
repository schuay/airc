# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Global configuration.

Loaded from ~/.config/airc/airc.toml (override with --config), the single config
file the whole suite shares. airc reads its own sections plus the shared ones
(parsed in airc_core.load_common). All sections are optional; defaults below keep
a bare `airc` invocation functional.

Example:

    [models]
    default = "google_genai:gemini-3-flash-preview"
    filter = "google_genai:gemini-3-flash-preview"

    [gcp]
    project = "my-vertex-project"
    location = "us-central1"

    [mcp.servers.v8-utils]
    transport = "stdio"
    command = "v8-mcp"
    args = ["--enable-pd", "--no-gerrit-drafts", "--no-default-user"]

    # Core ships no tool groups; each app sets its own. The coding app's full
    # v8-utils/gdb list is airc_coding.tool_groups.CODING_TOOL_GROUPS.
    [tool_groups]
    read = ["v8-utils__repo_git_*", "v8-utils__gerrit_comments", "v8-utils__pd_*", ...]
    active = ["v8-utils__run_d8", "v8-utils__jsb_run_bench", "v8-utils__perf_*"]

    [repos]
    v8 = "/path/to/v8/v8"

    # airc's own sections are namespaced under [airc.*]; the watcher's under
    # [watchers.*] and the processor's under [processors.*] (same file).
    [airc.orchestrator]
    soft_turn_budget = 8
    max_turns = 24
    max_responders = 2

    [[airc.commentary]]
    repo = "v8"

    [airc.chat]
    project = "my-project"
    space = "spaces/AAAAAAAAAAA"
    subscription = "projects/my-project/subscriptions/airc-sub"
    credentials = "~/.config/airc/chat-sa.json"   # omit to use ADC
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import tomllib
from airc_core import DEFAULT_TOOL_GROUPS, load_common, parse_handover_fields
from airc_core import apply_gcp_env_defaults as _apply_gcp_env
from airc_core.config import reject_unknown, reject_unknown_fields
from platformdirs import user_config_path, user_data_path

log = logging.getLogger(__name__)

CONFIG_DIR = user_config_path("airc")
DATA_DIR = user_data_path("airc")

# The full set of top-level sections the shared suite file may carry. A key
# outside this set is a typo (`[watchers]`, `[air]`) that would otherwise be
# silently ignored, so load_config errors on it. The set spans the whole suite,
# not just the room: models/model_providers/mcp/gcp/tool_groups/caching/
# bus_root/token_db_path/repos are the shared sections
# airc_core.load_common parses; handover is suite
# policy read by airc and the processor; the sibling-daemon namespaces
# ([watchers.*]/[processors.*]/[icompleteu.*]) live in this same file and are
# known-not-ours, so they are permitted here rather than flagged. [airc] and
# [transport] are the room's own. The keys INSIDE [airc] are validated
# separately: core consumes its own and the app plugin validates the remainder
# (a domain-neutral core cannot know a plugin's key set).
_KNOWN_TOPLEVEL = frozenset(
    {
        "models",
        "model_providers",
        "mcp",
        "gcp",
        "tool_groups",
        "caching",
        "bus_root",
        "token_db_path",
        "repos",
        "handover",
        "transport",
        "matrix",
        "airc",
        "watchers",
        "processors",
        "icompleteu",
        # Project-pack policy is interpreted by the coding plugin and sibling
        # coding daemons. Core tolerates it like their other suite sections but
        # remains ignorant of its schema.
        "prompt_pack",
        "projects",
        # Sibling daemons' own sections. The suite shares ONE file, so every
        # component must tolerate the sections it does not own -- a component
        # added without its name here makes the room refuse to start on a
        # perfectly valid suite config.
        "discovery",
    }
)

# The keys core itself consumes out of the [airc] table. Everything else in
# [airc] belongs to the app plugin and is carried through as cfg.plugin_config.
_CORE_AIRC_KEYS = frozenset(
    {
        "orchestrator",
        "transport",
        "matrix",
        "room_topic",
        "plugin_module",
        "use_nicknames",
        "grounding_reminder_tokens",
        "voices",
        "catchup_max_age_s",
        "db_path",
        "memory",
    }
)

# Built-in fallback when [models] is absent. Vertex (ADC + [gcp] project)
# rather than an API key: keys default to the AI Studio free tier, whose
# quota cannot sustain the room.
DEFAULT_MODEL = "google_vertexai:gemini-2.5-flash"

# Starter config written by `airc --init-config`. Core sections only: an app
# plugin appends its own via the config_template() hook (see airc_room.plugin),
# so this string never needs to know what any app configures. Kept in sync with
# the loader below; every section here is optional at load time.
TEMPLATE_CONFIG = """\
# airc suite configuration -- ONE file read by every component. Top-level
# sections are SHARED (every component reads them and they MUST agree -- stating
# them once is the point); per-component sections are namespaced ([airc.*] for
# the room, and whatever namespaces the app's own daemons use) so it is obvious
# who owns each, and each ignores the others'. All sections optional; see the
# README for details.

# == shared (every component) ================================================

[models]
# init_chat_model ids ("provider:model"). google_vertexai needs the [gcp]
# section below (auth via Application Default Credentials); google_genai,
# anthropic, openai, deepseek and openrouter each read their own *_API_KEY env
# var (openrouter:<model> is served OpenAI-compatibly -- e.g. GLM).
# One table, but each component picks its key: the room uses default + filter,
# and an app's own components read the keys they document.
default = "google_vertexai:gemini-2.5-flash"
filter  = "google_vertexai:gemini-2.5-flash"   # coordinator (routing) + triage

[gcp]
# Only for google_vertexai:* models.
project  = "my-project"
location = "global"        # current Gemini models are served on the global endpoint

[caching]
# Explicit Vertex context caching of each conversation's growing [system +
# history] prefix (~10% input cost on the cached tokens, plus token-hour
# storage). A no-op for non google_vertexai:* models; degrades to uncached if a
# cache cannot be made. In an environment that enforces client-certificate mTLS
# the service env must set GOOGLE_API_USE_CLIENT_CERTIFICATE=false so the cache
# REST client skips it. Worth the storage only where implicit caching is
# uncredited (gemini-3.1-pro-preview); turn off for models with working implicit
# caching.
explicit    = true
ttl_minutes = 30           # cache lifetime; clamped up if below the turn deadline

# MCP servers the personas reach tools through, one block per server. Omit the
# whole [mcp] section to run tool-less. Which tools a persona may actually call
# is gated separately by [tool_groups] plus its agent.toml; an app documents the
# servers and groups it expects.
# enable_in_sandbox defaults false. Set it only for servers a sandboxed worker
# may launch.
# [mcp.servers.example]
# transport = "stdio"
# command   = "example-mcp"
# args      = []
# enable_in_sandbox = false

# Bus root every component meets on (directory-backed). Must be identical across
# the suite; every topic lives under it.
# bus_root = "~/.local/share/airc/bus"

# Logical repo name -> local checkout. Shared, so components that resolve a repo
# by name agree on where it is.
# [repos]
# myrepo = "/path/to/myrepo"

# Handover of work to an external worker over the bus. Off by default. bus_root
# defaults to the suite bus_root above and MUST match the worker's; set it only
# when the worker polls a different bus. autonomy names how far the worker may
# go on its own; the values are the worker's vocabulary, and the most
# conservative one is a good setting while bringing up.
# [handover]
# enabled  = true
# autonomy = "draft-only"
# bus_root = "/path/to/worker/bus"
# kinds = ["repro"]           # allowlist of job kinds this suite may hand over.
#                              # REQUIRED once enabled (startup refuses to guess
#                              # for a live deployment). Values are the app's
#                              # vocabulary (the coding suite's: bugfix, repro,
#                              # perf, task), validated at startup. The default
#                              # is repro alone -- the one kind that cannot
#                              # produce a CL; every uploading kind is opt-in.

# == [airc.*]  (the chat room; run `airc` / `airc --headless`) =================

[airc]
# The app plugin the room loads (its build_subscribers/build_follow_ups/
# build_transport factories; see the Plugin protocol). Core names no plugin by
# default -- an unset value is a bare room -- so a real deploy sets this. The
# plugin also ships its own personas and parses its own [airc.*] sections.
# plugin_module = "myapp.app"
# db_path = "~/.local/share/airc/airc.db"   # threads/messages store
# The room's subject, filled into the coordinator/announcement routing prompts so
# they read naturally. The app plugin owns the domain, so set it to match.
# room_topic = "what this room is about"
# Catch-up window (seconds) for event subscribers after downtime: events older
# than this are acked unread instead of replayed, so a long outage does not flood
# the room. Default 3600 (1h); a subscriber may exempt itself.
# catchup_max_age_s = 3600
# Address and display each persona by its agent.toml `nickname` instead of the
# functional folder handle. Persisted per-thread state keys on the stable folder
# identity, so toggling this preserves each persona's thread memory.
# use_nicknames = true

# Per-persona voice guides (TONE only), keyed by the persona's functional handle
# (its agents/ folder name). Each points at a distilled tone guide; applied to
# chat turns only. A guide distilled from a real person's writing needs that
# person's consent before you deploy it.
# [airc.voices]
# somepersona = "/path/to/voices/somepersona.md"

[airc.orchestrator]
soft_turn_budget = 8       # agent streak past which the coordinator converges the thread
max_turns        = 24      # streak where only decisive contributions continue (no hard stop)
max_responders   = 2       # max coordinator-selected repliers per message
max_concurrent_turns = 4   # global cap on simultaneously-running agent turns
turn_timeout     = 900     # hard per-turn deadline in seconds

# An app plugin's own [airc.*] sections (event subscribers, its transport's
# config, ...) are appended below by its config_template(); core carries them
# through to the plugin verbatim and models none of them.

# Which chat frontend the room binds. "console" (the default for an interactive
# run) and "matrix" are in-core; a plugin may register others, and names the one
# it wants when config sets none.
# [transport]
# kind = "console"

# Matrix transport (used when [transport] kind = "matrix"). An in-core frontend
# built on matrix-nio: token login (no password), the bot pre-joins the
# rooms listed here. homeserver/user_id/access_token are required; the loader
# errors at startup if any is missing. access_token may instead come from
# $MATRIX_ACCESS_TOKEN (the file wins when set), so a deployment can keep the
# secret out of the config file. Threads are off by default (a flat room is the
# widest-supported shape); use_threads maps airc's thread_id onto an m.thread
# relation for clients that render them.
# [matrix]
# homeserver   = "https://matrix.example.org"
# user_id      = "@airc:example.org"
# access_token = "syt_..."          # long-lived bot token; or $MATRIX_ACCESS_TOKEN
# room_ids     = ["!abc123:example.org"]
# device_id    = "AIRCBOT"          # optional; a stable id keeps one device row
# use_threads  = false              # true: reply inside an m.thread per airc thread
"""

# Tool groups (DEFAULT_TOOL_GROUPS) live in airc_core now -- the whole suite
# shares one definition. Re-exported above for the agents/tests that import it
# from here.


@dataclass
class MatrixConfig:
    """The Matrix transport (in-core; built on matrix-nio).

    Token login only: access_token is a long-lived bot token, so nothing derives
    a password and the bootstrap is one homeserver call. The bot pre-joins the
    rooms it should serve (room_ids); it never auto-accepts invites, so it cannot
    be pulled into a stranger's room. homeserver/user_id/access_token are
    required -- an incomplete [matrix] section is an operator error the loader
    rejects at startup rather than booting a transport that cannot authenticate.

    use_threads is off by default: a flat room is the shape every Matrix client
    renders, and threads are unevenly supported. When on, the transport maps a
    room thread_id onto an m.thread relation so a thread-aware client groups the
    conversation; the fields stay generic so the mapping is the transport's alone.

    device_id is optional but recommended: a stable id keeps the homeserver from
    minting a fresh device row per restart, and it is where a future E2E device
    key would attach without reshaping this config.
    """

    homeserver: str
    user_id: str
    access_token: str
    room_ids: list[str] = field(default_factory=list)
    device_id: str = ""
    use_threads: bool = False


@dataclass
class OrchestratorConfig:
    # Streak length (consecutive agent messages since the last human/watcher
    # message) past which the coordinator is told to converge the discussion.
    # Below this, normal routing.
    soft_turn_budget: int = 8
    # Streak past which the coordinator is told only a decisive contribution
    # may continue. Not a hard cutoff: discussions peter out via escalating
    # pressure rather than stopping mid-point. Always >= soft_turn_budget.
    max_turns: int = 24
    # Max agents allowed to respond to a single message via the coordinator.
    max_responders: int = 2
    # Global cap on simultaneously-running agent turns across all threads:
    # bounds LLM cost/rate-limit exposure when threads run in parallel.
    max_concurrent_turns: int = 4
    # Hard deadline on a single agent turn (model calls + tools). A turn past
    # this is cut off and reported as errored, so it cannot pin its
    # "thinking..." card and a turn-semaphore slot indefinitely.
    turn_timeout: float = 900.0


@dataclass
class MemoryConfig:
    """Autonomous long-term memory (airc_room.memory): a git repo of markdown
    notes the personas maintain themselves.

    Off by default. `enabled` gates BOTH the memory tools (granted to a persona
    that lists the "memory" tool_group) and the per-turn index injection -- one
    switch, no tools-on/index-off split until something proves it needs one.
    `path` is the store checkout the tools are jailed to; the store owns its own
    schema (validator + pre-commit hook). Core ships no default path (it is
    deployment-specific); enabling memory without a path is a config error."""

    enabled: bool = False
    path: Path | None = None


@dataclass
class HandoverConfig:
    """Hand confirmed findings off to icompleteu by enqueuing JobSpecs.

    Off by default: enabling it lets airc emit autonomous patch jobs. autonomy
    is the level icompleteu runs them at (draft-only does no upload; the higher
    levels upload a CL / run CQ+Pinpoint). bus_root must match icompleteu's.
    """

    enabled: bool = False
    autonomy: str = "draft-only"  # draft-only | upload-wip | upload-cq-pinpoint
    bus_root: Path = field(default_factory=lambda: DATA_DIR / "bus")
    # Which job kinds this suite may hand over. The per-kind gate: every
    # producer's enqueue funnels through one choke per component (airc's
    # Handover.submit, the processor's _submit), so one list here is
    # authoritative for all of them -- replacing repro_only, which had to be
    # taught to each new producer as it appeared and even then covered only
    # "fix" jobs (a perf or task job sailed through a config that promised
    # repro-gathering). Defaults to repro alone, the one kind that cannot
    # produce a CL; every kind that uploads, runs CQ, or spends Pinpoint is an
    # explicit opt-in. Deliberately no wildcard: a future kind must be a
    # decision in this list, not a default that appears when it is added. The
    # vocabulary is the apps' (the coding suite: bugfix, repro, perf, task),
    # validated by them -- core must not depend on a component package, so the
    # strings stay opaque here. [] is the drain state; the components warn
    # that enabled = false is the clearer switch for that.
    kinds: list[str] = field(default_factory=lambda: ["repro"])


@dataclass
class Config:
    default_model: str = DEFAULT_MODEL
    # Fast/cheap model for coordinator routing and source triage filters.
    filter_model: str = DEFAULT_MODEL
    mcp_servers: dict[str, dict] = field(default_factory=dict)
    tool_groups: dict[str, list[str]] = field(
        default_factory=lambda: dict(DEFAULT_TOOL_GROUPS)
    )
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    # Bus root airc subscribes from -- must match the watchers'/processors' bus
    # (and handover.bus_root). The commit and findings topics live under it.
    bus_root: Path = field(default_factory=lambda: DATA_DIR / "bus")
    # Logical repo name -> local checkout, for provenance on a commit
    # announcement (so a handover off that thread can recover the path).
    repos: dict[str, str] = field(default_factory=dict)
    gcp: dict[str, str] = field(default_factory=dict)
    # Which transport the room binds. "console" and "" (unset) are core; every
    # other kind is supplied by the app plugin's build_transport (the coding app
    # ships "gchat", a future deploy "matrix"). Core never names a plugin
    # transport, so it stays domain- and corp-neutral; an unknown kind errors at
    # startup in cli.py.
    transport_kind: str = ""
    # The raw, uninterpreted [airc] TOML table minus the keys core consumes
    # itself (orchestrator, room_topic, transport, voices, ...): whatever is left
    # is the plugin's to parse. Core stays domain-neutral -- it models none of the
    # plugin's config (commit commentary, findings, perf, gchat, ...); the plugin
    # overlay (e.g. airc_coding.config.parse_config) reads its sections out of
    # this dict and owns their validation. Empty when [airc] has no plugin keys.
    plugin_config: dict = field(default_factory=dict)
    # Parsed [matrix] config when [transport] kind = "matrix", else None. Unlike
    # gchat (a plugin transport whose config core passes through as opaque
    # plugin_config), Matrix is a core transport, so core models its config as a
    # typed dataclass.
    matrix: MatrixConfig | None = None
    handover: HandoverConfig = field(default_factory=HandoverConfig)
    # Autonomous long-term memory. Off by default; enabling it grants the memory
    # tools to personas that list the "memory" group and injects the memory index
    # into each of their turns.
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    # Insert the grounding rule into the conversation once per this many tokens of
    # context growth, so a long thread keeps a recent copy of the rule near the
    # tail (the system prompt loses weight far from it). 0 disables it. See
    # agent.GroundingReminderMiddleware.
    grounding_reminder_tokens: int = 200_000
    # Address and display personas by their human nickname (agent.toml `nickname`)
    # instead of the functional folder handle. Off keeps the functional names.
    # Persisted per-thread state keys on the stable folder identity (not the
    # handle), so toggling this on or off preserves each persona's thread memory.
    use_nicknames: bool = False
    # Per-persona voice guides, keyed by functional handle (gc, compiler, ...).
    # Each value is a path to a distilled TONE guide.
    # Applied to that persona's CHAT turns only, appended to its system prompt as
    # a tone reference; never to digest/review/verify. Empty = neutral voice.
    # Independent of use_nicknames -- keyed on the stable handle, so it survives a
    # nickname toggle.
    #
    # A guide distilled from a real person's writing needs that person's consent
    # before it is deployed, and the persona must not be presented as them. The
    # load-time scrub (voice_body) drops provenance, but it cannot make an
    # unconsented imitation acceptable -- that is a judgement call, not a filter.
    # Prefer a synthetic register ("terse, cites the spec") over a mined one.
    voices: dict[str, Path] = field(default_factory=dict)
    # Catch-up window (seconds) for the ephemeral chat subscribers (commentary,
    # perf). On restart, events whose own timestamp is older than this are acked
    # unread instead of replayed, so a short outage catches up but a long one
    # does not flood the room. Findings ignore it (a confirmed defect is always
    # worth posting). Default 1h.
    catchup_max_age_s: float = 3600.0
    db_path: Path = field(default_factory=lambda: DATA_DIR / "airc.db")
    # Shared token-usage ledger for the whole suite (its own file so separate
    # component processes can write to it concurrently under WAL). Kept apart
    # from db_path, which holds airc's threads/messages.
    token_db_path: Path = field(default_factory=lambda: DATA_DIR / "tokens.db")
    # Explicit Vertex context caching of each conversation's growing
    # [system + history] prefix. A no-op for non google_vertexai:* models, and
    # degrades to uncached when a cache cannot be created (e.g. an mTLS-enforcing
    # env without GOOGLE_API_USE_CLIENT_CERTIFICATE=false). Explicit caching pays
    # token-hour storage, so it is a net win only where IMPLICIT caching is
    # uncredited -- gemini-3.1-pro-preview. On models
    # whose implicit caching already works (gemini-2.5/3.5-flash) leave this off:
    # implicit gives the same read discount for free with no storage charge.
    caching_explicit: bool = True
    # Cache lifetime. Must exceed the longest single turn (orchestrator
    # turn_timeout) so a cache cannot expire mid-turn; a conversation gap longer
    # than this expires the cache and the next turn's reactive recovery rebuilds.
    cache_ttl_minutes: int = 30
    # The app plugin the room loads: a module exposing the plugin factories
    # (build_subscribers/build_follow_ups/build_transport; see the Plugin
    # protocol and airc_coding.app). The room is domain-neutral and resolves this
    # dynamically, so core never imports a plugin by name and names no specific
    # plugin as a default -- an unset value is a bare room (console, no
    # subscribers). Every real deploy sets it explicitly (the coding app to
    # "airc_coding.app").
    plugin_module: str = ""
    # The room's subject, filled into the coordinator/announcement routing prompts
    # so they read naturally without a domain literal baked into core (the app
    # sets it, e.g. "JS engine work" for coding, "weekly groceries and cooking"
    # for a grocery room). A generic default keeps a bare room sensible.
    room_topic: str = "the room's topics"

    def resolve_model(self, persona_model_id: str | None) -> str:
        """The concrete model id for a persona's declared `model`.

        A persona may name a real id (`provider:model`, always with a colon) or a
        ROLE alias -- "default" or "filter" -- so a cheap generalist can ride the
        same fast/cheap model the coordinator uses without hardcoding its id in
        two places (change [models] filter and the persona follows). An empty
        value (no `model` in agent.toml) is the default model. A real id passes
        through untouched; the colon makes the alias set unambiguous."""
        if not persona_model_id or persona_model_id == "default":
            return self.default_model
        if persona_model_id == "filter":
            return self.filter_model
        return persona_model_id


def load_config(path: Path | None = None) -> Config:
    explicit = path is not None
    path = path or CONFIG_DIR / "airc.toml"
    try:
        raw = tomllib.loads(path.read_text())
    except FileNotFoundError:
        # Only the implicit default location may be absent (fresh install runs
        # on defaults). An explicit --config that does not exist is an operator
        # error -- a typo'd systemd unit would otherwise start "successfully"
        # as a quietly dead deployment on all defaults.
        if explicit:
            raise
        raw = {}

    # Reject an unknown top-level section up front: a mis-namespaced or typo'd
    # section (a bare [chat], [watcher], [air]) is otherwise silently ignored and
    # the deployment runs quietly wrong. The [airc] sub-keys are checked later --
    # core consumes its own and the plugin validates the rest.
    if unknown := set(raw) - _KNOWN_TOPLEVEL:
        raise SystemExit(
            f"unknown config section(s): {', '.join(sorted(unknown))}"
            f" (known: {', '.join(sorted(_KNOWN_TOPLEVEL))})"
        )

    # Shared suite sections (models, mcp, gcp, tool_groups, caching, bus_root,
    # token_db_path, repos) are parsed once in airc_core so every component reads
    # them identically; airc overlays its own keys below.
    common = load_common(raw)
    default_model = common.models.get("default", DEFAULT_MODEL)
    cfg = Config(
        default_model=default_model,
        filter_model=common.models.get("filter", default_model),
        mcp_servers=common.mcp_servers,
        tool_groups=common.tool_groups,
        gcp=common.gcp,
    )
    cfg.bus_root = common.bus_root
    cfg.token_db_path = common.token_db_path
    cfg.repos = common.repos
    cfg.caching_explicit = common.caching_explicit
    cfg.cache_ttl_minutes = common.cache_ttl_minutes
    # airc's own sections live under [airc] in the shared suite file, so it is
    # obvious which daemon owns them next to [watchers.*] / [processors.*].
    # [handover] stays top-level: it is suite policy, read by airc and processors.
    own = raw.get("airc", {})
    if orch := own.get("orchestrator"):
        # turn_budget is the legacy spelling honoured just below, so it is an
        # accepted alias rather than a field.
        reject_unknown_fields(
            orch, OrchestratorConfig, "[airc.orchestrator]", aliases=["turn_budget"]
        )
        # Accept legacy `turn_budget` as an alias for the soft threshold.
        soft = int(orch.get("soft_turn_budget", orch.get("turn_budget", 8)))
        max_turns = int(orch.get("max_turns", max(24, soft)))
        cfg.orchestrator = OrchestratorConfig(
            soft_turn_budget=soft,
            max_turns=max(max_turns, soft),
            max_responders=int(orch.get("max_responders", 2)),
            max_concurrent_turns=int(orch.get("max_concurrent_turns", 4)),
            turn_timeout=float(orch.get("turn_timeout", 900)),
        )
    if mem := own.get("memory"):
        reject_unknown_fields(mem, MemoryConfig, "[airc.memory]")
        enabled = bool(mem.get("enabled", MemoryConfig.enabled))
        raw_path = mem.get("path")
        path = Path(str(raw_path)).expanduser() if raw_path else None
        if enabled and path is None:
            raise SystemExit("[airc.memory] enabled = true requires a path")
        cfg.memory = MemoryConfig(enabled=enabled, path=path)
    # Whatever [airc] keys core does not consume itself belong to the app plugin;
    # carry them through verbatim for the plugin overlay to parse and validate.
    # Core models none of them -- an app's own sections, and any config a plugin
    # transport needs, are opaque here. That is what keeps this file
    # domain-neutral.
    cfg.plugin_config = {k: v for k, v in own.items() if k not in _CORE_AIRC_KEYS}
    # Transport selection: [transport] kind = "console" | "gchat" | "matrix".
    # Absent means unset (""), and cli.py applies the app's default (gchat for the
    # coding app, so an un-edited prod config keeps working). "console" and "" are
    # bound by core; any other kind is resolved through the plugin. Canonical
    # placement is top-level [transport] (D4), but the [airc.transport] namespace
    # (matching every other room section) is accepted too so it is not silently
    # ignored; the namespaced form wins when both are present.
    if transport := (own.get("transport") or raw.get("transport")):
        reject_unknown(transport, {"kind"}, "[transport]")
        cfg.transport_kind = str(transport.get("kind", ""))
    # Matrix is a core transport, so its config is parsed here into a typed
    # dataclass (top-level [matrix], or the namespaced [airc.matrix]; the
    # namespaced form wins). Only when the section is present -- an absent one
    # leaves matrix=None, and cli.py errors clearly if kind == "matrix" without
    # it. The three connection fields are required: a partial section is an
    # operator error, caught here rather than as an opaque nio auth failure.
    if mx := (own.get("matrix") or raw.get("matrix")):
        reject_unknown_fields(mx, MatrixConfig, "[matrix]")
        # access_token is a secret, so it may come from $MATRIX_ACCESS_TOKEN
        # instead of the file: a deployment (e.g. a container) can then ship a
        # config with no token baked into an image layer or mounted file. The
        # file wins when set; the env var is only a fallback for an empty/absent
        # value, so it never silently overrides a real token in the config.
        access_token = mx.get("access_token") or os.environ.get(
            "MATRIX_ACCESS_TOKEN", ""
        )
        resolved = {**mx, "access_token": access_token}
        missing = [
            k for k in ("homeserver", "user_id", "access_token") if not resolved.get(k)
        ]
        if missing:
            raise ValueError(
                f"[matrix] is missing required field(s): {', '.join(missing)}"
                " (access_token may also be supplied via $MATRIX_ACCESS_TOKEN)"
            )
        cfg.matrix = MatrixConfig(
            homeserver=str(resolved["homeserver"]),
            user_id=str(resolved["user_id"]),
            access_token=str(resolved["access_token"]),
            room_ids=[str(r) for r in mx.get("room_ids", [])],
            device_id=str(mx.get("device_id", "")),
            use_threads=bool(mx.get("use_threads", False)),
        )
    # Parsed even when the section is absent so the default bus_root is the
    # suite one (cfg.bus_root, already set from [bus_root] above), not a
    # hardcoded DATA_DIR fallback: a suite configured onto a non-default bus
    # must not have its handover publish to a bus nothing polls.
    h = raw.get("handover", {})
    hf = parse_handover_fields(h)
    cfg.handover = HandoverConfig(
        enabled=hf.enabled,
        autonomy=hf.autonomy,
        bus_root=Path(hf.bus_root).expanduser() if hf.bus_root else cfg.bus_root,
        kinds=hf.kinds,
    )
    if "plugin_module" in own:
        cfg.plugin_module = str(own["plugin_module"])
    if "room_topic" in own:
        cfg.room_topic = str(own["room_topic"])
    if "use_nicknames" in own:
        cfg.use_nicknames = bool(own["use_nicknames"])
    if "grounding_reminder_tokens" in own:
        cfg.grounding_reminder_tokens = int(own["grounding_reminder_tokens"])
    for handle, vpath in own.get("voices", {}).items():
        cfg.voices[str(handle)] = Path(vpath).expanduser()
    if "catchup_max_age_s" in own:
        cfg.catchup_max_age_s = float(own["catchup_max_age_s"])
    if db := own.get("db_path"):
        cfg.db_path = Path(db).expanduser()
    # A context cache must outlive the longest single turn or it can expire
    # mid-turn (recoverable, but wasteful). Enforce the documented invariant
    # rather than trusting two independently-set knobs.
    min_ttl = math.ceil(cfg.orchestrator.turn_timeout / 60) + 5
    if cfg.cache_ttl_minutes < min_ttl:
        log.warning(
            "cache_ttl_minutes=%d is below the %dm a %ds turn_timeout needs; using %dm",
            cfg.cache_ttl_minutes,
            min_ttl,
            int(cfg.orchestrator.turn_timeout),
            min_ttl,
        )
        cfg.cache_ttl_minutes = min_ttl
    # A bare room (no plugin) has nobody to parse the plugin's [airc] sections, so
    # they would be silently dropped -- the exact quiet-misconfig the strict check
    # exists to prevent, except here no plugin ever runs the check. Warn loudly:
    # an operator who commented out plugin_module while debugging, or forgot to set
    # it, would otherwise boot a room that ignores [[airc.commentary]]/[airc.chat]
    # with no signal. (Not fatal: a genuinely bare room with a leftover section is
    # recoverable, and cli.py's startup line still shows the empty subscriber set.)
    if not cfg.plugin_module and cfg.plugin_config:
        log.warning(
            "no plugin_module set but [airc] carries plugin sections %s;"
            " a bare room ignores them. set [airc] plugin_module to parse them",
            ", ".join(sorted(cfg.plugin_config)),
        )
    return cfg


def write_template_config(
    path: Path,
    force: bool = False,
    plugin_template: str | None = None,
    plugin_module: str | None = None,
) -> None:
    """Write the starter config to path; refuse to clobber unless force.

    Core's template covers only the sections core itself loads. An app's
    sections come from its plugin's config_template() and are appended verbatim,
    so `--init-config --plugin <module>` still produces one complete file
    without core knowing what any app configures. The plugin's text lands after
    core's, so it must not reopen a table core already declared (see
    airc_room.plugin for the rule)."""
    if path.exists() and not force:
        raise SystemExit(f"{path} already exists; pass --force to overwrite")
    body = TEMPLATE_CONFIG
    if plugin_module:
        # plugin_module is a CORE key, so core writes it -- the plugin's own text
        # cannot, having landed after core already declared [airc]. Uncommenting
        # here is what makes the emitted file runnable rather than a form to fill
        # in.
        body = body.replace(
            '# plugin_module = "myapp.app"', f'plugin_module = "{plugin_module}"'
        )
    if plugin_template:
        # A blank line between the two halves regardless of how the plugin's
        # text is terminated, so the seam reads the same for every plugin.
        body = f"{body.rstrip()}\n\n{plugin_template.lstrip()}"
        if not body.endswith("\n"):
            body += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    print(f"wrote {path}")


def apply_gcp_env_defaults(cfg: Config) -> None:
    """Set GOOGLE_CLOUD_* env defaults from [gcp] for Vertex AI.

    Thin adapter over the airc_core helper, kept so airc's call sites (cli +
    scripts) keep passing a Config; the env-var logic lives once in airc_core.
    """
    _apply_gcp_env(cfg.gcp)

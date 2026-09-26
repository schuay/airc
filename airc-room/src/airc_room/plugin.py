# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""The plugin contract: what a module must expose to be loaded as an airc app.

The room (airc-room) is domain-neutral. It becomes a concrete app -- the V8
coding room, a grocery room -- by loading one plugin module named in config
(`[airc] plugin_module`) and calling a small, fixed set of factories on it. Core
never imports a plugin by name; it resolves the module dynamically and validates
it against this contract, so a non-coding deploy pulls in none of another app's
code.

A plugin is exactly four things (see the core/plugin split design): the bus
payload schemas it owns, the subscribers/producers it registers, its `agents/`
personas, and its config block. This module publishes the *interface* half of
that -- the factory signatures the room calls -- as a runtime-checkable Protocol
plus a compatibility version, so an external plugin codes against a typed
contract instead of matching an undocumented convention by reading source, and a
stale plugin fails loudly at load instead of mysteriously at first use.

Required:
- build_subscribers(cfg, room, store, toolset) -> list[Subscriber]
- build_follow_ups(cfg, store, *, agents_dir) -> dict[str, FollowUp]
- build_transport(cfg, room, store, kind) -> Transport | None
- build_local_tools(cfg, *, room) -> LocalTools

Optional (duck-typed; absent means the room's default behavior):
- default_transport_kind() -> str | None  -- the transport a headless deploy
  binds when config names none (the coding app returns "gchat"). Core names no
  transport itself; this is how a plugin, not core, owns that default.
- personas_dir(cfg=...) -> Path | None  -- the `agents/` directory this plugin
  supplies, so its personas travel with it instead of relying on the service
  cwd. Takes the loaded Config when it declares the parameter, for a
  plugin whose personas come from a configured source instead of its package.
- parse_config(cfg) -> object  -- validate and type the plugin's own [airc]
  sub-table (carried on cfg.plugin_config), returning the config object its
  subscribers read. This lets core drop the domain config fields.
- config_template() -> str | None  -- the plugin's own commented TOML sections,
  appended to core's starter config by `airc --init-config --plugin <module>`.
  The counterpart to parse_config: a plugin that parses its own sections also
  documents them, so core's template covers only what core itself loads and
  setup stays one command. Returned text is written verbatim, so it must be
  valid TOML *after* core's half: core already opens [airc], and TOML forbids
  declaring a table twice, so contribute [airc.<name>] sub-tables (plus any
  top-level sections in _KNOWN_TOPLEVEL) instead of reopening [airc].
- tool_instructions(cfg) -> str  -- prose about the plugin's local tools for every
  persona's system prompt, joined with the MCP servers' instructions under the
  same heading. Absent or empty means nothing is added.
- build_message_handlers(cfg, room, store) -> list[MessageHandler]  -- observers
  on arriving messages, run by the orchestrator before it routes anything (see
  orchestrator.MessageHandler / Disposition). A handler that returns CONSUMED
  ends the message there: no mention parse, no coordinator, no persona turn. The
  room pushes messages to personas and nowhere else, so without this a plugin
  feature reacting to what a human typed can only reconstruct arrival by polling
  the store; this is the push. Handlers run inline in the per-thread worker, so
  they inherit its ordering, watermark and crash replay -- and owe idempotency
  back, since replay re-delivers. Absent means no handlers.
- build_services(cfg, room, store) -> list  -- long-running background services
  the room supervises as tasks (each a `.name` and an async `.run()`, like a
  Subscriber but not bus-driven). For periodic/clock work with no bus topic: e.g.
  grocery's memory-compaction loop (summarize a grown thread into durable memory,
  then bump its context generation to truncate). Absent means no services.

Note `aux_services()` is not a plugin-module hook: it is an optional method on
the Transport a plugin returns from build_transport, read off that transport
instance (see cli.py), so a transport can own side loops (gchat's
space-subscription renewal). It is documented with the Transport surface, not
here.

PLUGIN_API_VERSION is bumped when a required signature changes incompatibly. An
external plugin declares the version it was written against as a literal
(PLUGIN_API_VERSION = 2), so a core that has moved on rejects it loudly rather
than calling a changed signature; importing and re-exporting core's constant
defeats the check because it always matches itself. A plugin that declares no
version is rejected.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from airc_core import ToolGrant

# Bumped on an incompatible change to a required factory signature. Integer, so
# the check is a simple equality (there is one contract at a time); a plugin
# built against a different number is refused with a clear message instead of
# blowing up on a renamed/removed argument deep in startup.
PLUGIN_API_VERSION = 2

# The names a plugin module must define to be loadable. Kept as data (not just
# the Protocol) so the loader can name the exact missing attribute.
_REQUIRED = (
    "build_subscribers",
    "build_follow_ups",
    "build_transport",
    "build_local_tools",
)


@dataclass(frozen=True)
class LocalTools:
    """A plugin's baseline grant and persona-gated groups.

    The grant resolves against the union of core and plugin catalog entries for
    every conversational persona. `groups` stays opt-in through agent.toml.
    Configured MCP tools remain on their read/active group path.
    """

    tool_grant: ToolGrant = field(default_factory=ToolGrant)
    groups: Mapping[str, Sequence[object]] = field(default_factory=dict)


@runtime_checkable
class Plugin(Protocol):
    """Structural type for a loaded plugin module."""

    def build_subscribers(self, cfg, room, store, toolset) -> list: ...

    def build_follow_ups(
        self, cfg, store, *, agents_dir: Path
    ) -> dict[str, Callable[..., Awaitable[None]]]: ...

    def build_transport(self, cfg, room, store, kind: str): ...

    def build_local_tools(self, cfg, *, room) -> LocalTools: ...

    # Optional hooks. aux_services is absent: it lives on the Transport.
    def default_transport_kind(self) -> str | None: ...

    def personas_dir(self, cfg=None) -> Path | None: ...

    def parse_config(self, cfg): ...

    def config_template(self) -> str | None: ...

    def build_message_handlers(self, cfg, room, store) -> list: ...

    def build_services(self, cfg, room, store) -> list: ...


def validate_plugin(module, module_name: str) -> None:
    """Reject a plugin that cannot fulfil the contract, with a message that says
    which part is wrong. Called right after import so a misconfigured
    plugin_module fails at startup, not at the first subscriber build.

    Required factories must be callable, and the plugin must declare the current
    API version as a literal."""
    missing = [name for name in _REQUIRED if not callable(getattr(module, name, None))]
    if missing:
        raise SystemExit(
            f"plugin {module_name!r} is missing required factor(y/ies):"
            f" {', '.join(missing)}. A plugin module must define"
            f" {', '.join(_REQUIRED)} (see airc_room.plugin.Plugin)."
        )
    declared = getattr(
        module, "PLUGIN_API_VERSION", getattr(module, "plugin_api_version", None)
    )
    if declared is None:
        raise SystemExit(
            f"plugin {module_name!r} declares no PLUGIN_API_VERSION; this"
            f" airc-room requires version {PLUGIN_API_VERSION}. Follow the"
            " migration in airc-room/PLUGINS.md."
        )
    if declared != PLUGIN_API_VERSION:
        raise SystemExit(
            f"plugin {module_name!r} targets plugin API version {declared},"
            f" but this airc-room speaks {PLUGIN_API_VERSION}. Install a matching"
            f" plugin/core pair."
        )

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""The plugin contract: what a module must expose to be loaded as an airc app.

The room (airc-room) is domain-neutral. It becomes a concrete app -- the V8
coding room, a grocery room -- by loading ONE plugin module named in config
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
stale plugin fails loudly at load rather than mysteriously at first use.

Required (a module without all three is rejected at load):
- build_subscribers(cfg, room, store, toolset) -> list[Subscriber]
- build_follow_ups(cfg, store, *, agents_dir) -> dict[str, FollowUp]
- build_transport(cfg, room, store, kind) -> Transport | None

Optional (duck-typed; absent means the room's default behavior):
- default_transport_kind() -> str | None  -- the transport a headless deploy
  binds when config names none (the coding app returns "gchat"). Core names no
  transport itself; this is how a plugin, not core, owns that default.
- personas_dir() -> Path | None  -- the packaged `agents/` directory this plugin
  ships, so its personas travel with the package instead of relying on the
  service cwd.
- parse_config(cfg) -> object  -- validate and type the plugin's own [airc]
  sub-table (carried on cfg.plugin_config), returning the config object its
  subscribers read. Owning this is what lets core delete the domain config fields.
- config_template() -> str | None  -- the plugin's own commented TOML sections,
  appended to core's starter config by `airc --init-config --plugin <module>`.
  The counterpart to parse_config: a plugin that parses its own sections also
  documents them, so core's template covers only what core itself loads and
  setup stays one command. Returned text is written verbatim, so it must be
  valid TOML *after* core's half: core already opens [airc], and TOML forbids
  declaring a table twice, so contribute [airc.<name>] sub-tables (plus any
  top-level sections in _KNOWN_TOPLEVEL) rather than reopening [airc].
- build_local_tools(cfg, *, room=None) -> dict[str, list[BaseTool]]  -- local (non-MCP)
  langchain tools the plugin contributes, keyed by tool_group name. The room
  grants a persona the tools under a group iff that group is in the persona's
  tool_groups -- the SAME gate MCP tools use, so a persona's grants stay in its
  agent.toml whether the tool is MCP or local. These groups live only in this
  dict (not in the [tool_groups] config), so a persona may name one without it
  being a configured MCP group. Used for tools that need in-process wiring an MCP
  server cannot get (e.g. the grocery memory tools jailed to an akbase path).
  `room` is passed by keyword for a tool that must POST rather than only compute
  -- e.g. one whose integrity property is that the ROOM, not the model's prose,
  puts the text in the thread. Keyword-OPTIONAL deliberately: an older plugin
  declaring `build_local_tools(cfg)` keeps working, so this stays a compatible
  addition and needs no PLUGIN_API_VERSION bump. The room inspects the hook and
  passes `room` only to one that accepts it, by name or through `**kwargs`.
- build_services(cfg, room, store) -> list  -- long-running background services
  the room supervises as tasks (each a `.name` and an async `.run()`, like a
  Subscriber but not bus-driven). For periodic/clock work with no bus topic: e.g.
  grocery's memory-compaction loop (summarize a grown thread into durable memory,
  then bump its context generation to truncate). Absent means no services.

Note `aux_services()` is NOT a plugin-module hook: it is an optional method on
the Transport a plugin returns from build_transport, read off that transport
instance (see cli.py), so a transport can own side loops (gchat's
space-subscription renewal). It is documented with the Transport surface, not
here.

PLUGIN_API_VERSION is bumped when a required signature changes incompatibly. An
external plugin declares the version it was written against as a LITERAL
(PLUGIN_API_VERSION = 1), so a core that has moved on rejects it loudly rather
than calling a changed signature; importing and re-exporting core's constant
defeats the check (it always matches itself) and is only correct for the in-tree
plugin, versioned in lockstep with core. A plugin that declares none is tolerated
but forgoes the check.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

# Bumped on an incompatible change to a required factory signature. Integer, so
# the check is a simple equality (there is one contract at a time); a plugin
# built against a different number is refused with a clear message rather than
# blowing up on a renamed/removed argument deep in startup.
PLUGIN_API_VERSION = 1

# The names a plugin module must define to be loadable. Kept as data (not just
# the Protocol) so the loader can name the exact missing attribute.
_REQUIRED = ("build_subscribers", "build_follow_ups", "build_transport")


@runtime_checkable
class Plugin(Protocol):
    """Structural type for a loaded plugin module. The three factories are
    required; the rest are optional and duck-typed at their call sites."""

    def build_subscribers(self, cfg, room, store, toolset) -> list: ...

    def build_follow_ups(
        self, cfg, store, *, agents_dir: Path
    ) -> dict[str, Callable[..., Awaitable[None]]]: ...

    def build_transport(self, cfg, room, store, kind: str): ...

    # Optional -- see the module docstring. Declared so the Protocol documents the
    # full surface, but a plugin need not implement them (callers use getattr).
    # aux_services is deliberately absent: it lives on the Transport, not here.
    def default_transport_kind(self) -> str | None: ...

    def personas_dir(self) -> Path | None: ...

    def parse_config(self, cfg): ...

    def config_template(self) -> str | None: ...

    def build_local_tools(self, cfg, *, room=None) -> dict: ...

    def build_services(self, cfg, room, store) -> list: ...


def validate_plugin(module, module_name: str) -> None:
    """Reject a plugin that cannot fulfil the contract, with a message that says
    which part is wrong. Called right after import so a misconfigured
    plugin_module fails at startup, not at the first subscriber build.

    Two checks: the three required factories must be present and callable, and the
    plugin's declared API version (if it declares one) must match core's. A plugin
    that declares no version is allowed for now -- an in-tree plugin predating the
    field -- but a declared, mismatched one is a hard error."""
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
    if declared is not None and declared != PLUGIN_API_VERSION:
        raise SystemExit(
            f"plugin {module_name!r} targets plugin API version {declared},"
            f" but this airc-room speaks {PLUGIN_API_VERSION}. Install a matching"
            f" plugin/core pair."
        )

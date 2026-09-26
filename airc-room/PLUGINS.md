# The airc plugin contract

`airc-room` is a domain-neutral chat room: room, orchestrator, turn runner,
personas, console transport, and the subscriber/transport interfaces. It becomes
a concrete app -- the V8 coding room, a grocery room -- by loading **one plugin
module** named in config and calling a small, fixed set of factories on it. Core
imports no plugin by name; it resolves the module dynamically
(`[airc] plugin_module`) and validates it against this contract.

A plugin is exactly four things: the bus payload schemas it owns, the
subscribers/producers it registers, its `agents/` personas, and its config block.
It is **not** a framework -- the room, orchestrator, and turn loop stay concrete
code. This doc specifies the interface half: the factories core calls.

The reference implementation is `airc_coding.app` (the V8 app). A new plugin
mirrors its structure.

## Loading and validation

The room imports `cfg.plugin_module` and calls `airc_room.plugin.validate_plugin`
on it before use. A module missing a required factory, or declaring an
incompatible API version, fails at startup with a clear message instead of deep
in the wiring.

Declare the API version you build against as a literal:

```python
PLUGIN_API_VERSION = 2  # the core contract version this plugin targets
```

`airc_room.plugin.PLUGIN_API_VERSION` is an integer core bumps on an
incompatible change to a required signature. `validate_plugin` compares the
number your plugin declares against core's and rejects a mismatch. Pin a literal:
do not `from airc_room.plugin import PLUGIN_API_VERSION` and re-export it -- that
resolves to whatever the *installed* core is at import time, so the number always
matches itself and the check can never fire. The literal
freezes the contract version you coded against, so a core that has moved
on rejects your stale plugin loudly instead of calling a changed signature.

A plugin that declares no version is rejected.

## Required factories

A plugin module MUST define all four (each callable):

```python
def build_subscribers(cfg, room, store, toolset) -> list:
    """The bus subscribers that read this app's topics and post into the room.
    Each is an airc_room Subscriber (a `run()` loop over a bus Subscription).
    Return [] for a chat-only app with no bus feeds."""


def build_follow_ups(cfg, store, *, agents_dir) -> dict[str, FollowUp]:
    """The announcement-response handlers the orchestrator dispatches by the
    `follow_up` key a subscriber stamps on a SYSTEM announcement. The room stays
    domain-blind: it injects whatever follow-up prompt the message carries. Return
    {} for an app with no announcement follow-ups."""


def build_transport(cfg, room, store, kind: str):
    """Build a plugin-owned transport by kind, or None if this app supplies none
    for it. Core binds console/matrix itself and delegates any other kind here
    (the coding app owns 'gchat'). Returning None lets the caller raise one
    uniform 'unknown transport kind' error."""


def build_local_tools(cfg, *, room) -> LocalTools:
    """Declare the plugin's built-in candidates and grants.

    `allowlist` selects the baseline tools available to every conversational
    persona from the room's and plugin's combined candidates. `groups` contains
    persona-gated feature tools. Configured MCP groups remain separate.
    """
```

## Optional hooks (duck-typed)

Absent means the room's default behavior:

```python
def default_transport_kind() -> str | None:
    """The transport a headless deploy binds when config names none. Core names
    no transport itself -- this is how a plugin owns that default (the coding app
    returns 'gchat'). Absent/None falls back to the console."""


def personas_dir() -> Path | None:
    """The packaged agents/ directory this plugin ships, so its personas travel
    with the package instead of relying on the service cwd. Resolution order in
    the room: --agents-dir, then ./agents in the cwd, then this hook, then
    ~/.config/airc/agents."""


def build_message_handlers(cfg, room, store) -> list[MessageHandler]:
    """Observers on arriving messages, run before the orchestrator routes. A
    handler returning CONSUMED ends the message there -- no mention parse, no
    coordinator, no persona turn. Absent means no handlers."""


def parse_config(cfg) -> object:
    """Validate and type this app's own [airc] sub-table. Core parses only the
    generic [airc] keys and carries the rest through as cfg.plugin_config; this
    hook reads the plugin's sections out of that dict, returning the config object
    the subscribers read. Owning this is what lets core model none of the app's
    config. Reject unknown [airc] keys here -- core cannot know the plugin's key
    set, so the plugin owns that half of the strict-config check."""
```

## Migrating a plugin from API v1 to v2

API v2 makes every built-in grant explicit. Update a v1 plugin as follows:

1. Set the literal `PLUGIN_API_VERSION = 2`.
2. Define `build_local_tools(cfg, *, room)` even if the plugin has no tools.
3. Return `airc_room.plugin.LocalTools`, not a group dictionary.
4. Put tools every conversational persona may use in `candidates`, and list
   their exact names or patterns in `allowlist`.
5. Keep persona-gated feature tools in `groups`. Personas continue to opt into
   these groups through `agent.toml`.

For example:

```python
from airc_room.plugin import LocalTools

PLUGIN_API_VERSION = 2


def build_local_tools(cfg, *, room):
    return LocalTools(
        allowlist=("search_chat", "timer_*", "lookup"),
        candidates=(build_lookup(cfg),),
        groups={"memory": build_memory_tools(cfg)},
    )
```

`search_chat` and `timer_*` are room-owned candidates. A plugin must grant them
if its personas should retain them. Configured MCP `read` and `active` groups do
not change in API v2. The room rejects duplicate tool names instead of choosing
one by order.

### Message handlers

The room delivers messages to exactly one kind of consumer: a persona woken by a
mention. A plugin gets no delivery at all -- its only surface is the store, so
any plugin feature reacting to what a human typed has to reconstruct arrival by
re-reading SQLite on a timer. `build_message_handlers` is the push instead:

```python
class MessageHandler(Protocol):
    name: str

    async def handle(self, msg: Message) -> Disposition: ...

    # Disposition.CONSUMED -- stop: no mention parse, no coordinator, no turn
    # Disposition.PASS     -- next handler, then normal orchestration
```

Both types are importable from `airc_room.orchestrator`. Handlers run in
registration order and the first CONSUMED wins; NOTICE and PING messages never
reach them (they are never routed either). A handler that raises is logged and
treated as PASS, so a broken plugin cannot silence the room.

Three properties are worth knowing before writing one:

- **CONSUMED suppresses orchestration, not the message.** By the time a handler
  runs the message is already persisted and already delivered to every
  transport, and its `kind` is left untouched. The store's history stays honest;
  only the routing is skipped.
- **Handlers must be idempotent.** They run in the per-thread worker loop, which
  is what gives them ordering, the durable watermark, and crash replay -- and
  replay means a handler will see the same message again after a crash. This is
  the same contract every bus subscriber already honors. (This is also why the
  hook is not in `room.post`: startup recovery replays store messages straight
  into the workers, bypassing `post` entirely, so a hook there would silently
  skip exactly the messages that most need re-processing.)
- **Handlers run inline.** A slow handler delays its own thread's routing. Do
  store reads/writes, a bus publish, at most a post; put anything heavier on the
  bus.

### Transport-level hook: `aux_services`

Not a plugin-module hook -- a method a `Transport` (the object `build_transport`
returns) may define. If present, the room runs each returned coroutine-factory as
its own background task alongside the transport's `run()` loop. Use it for side
loops a transport owns; the coding app's gchat transport uses it for the
space-subscription renewal. Define it on the transport class, not the plugin
module (core reads it off the transport instance):

```python
class MyTransport:
    def aux_services(self) -> Iterable[Callable[[], Awaitable[None]]]:
        return [self._renewal_loop]
```

## Config ownership

One suite file (`airc.toml`), split ownership. Core parses the generic sections
and the generic `[airc]` keys (`orchestrator`, `room_topic`, `transport`,
`voices`, ...); everything else under `[airc]` is carried through verbatim as
`cfg.plugin_config` for the plugin's `parse_config` to read. Both halves validate
strictly: an unknown top-level section is rejected by core, an unknown `[airc]`
sub-key by the plugin.

A common pattern is a `Config` subclass carrying the generic core fields forward
plus the plugin's typed fields, so a subscriber reads both `cfg.bus_root` (core)
and its own typed section off the same object.

Documentation follows the same split. `airc --init-config` scaffolds core's
sections; `--plugin <module>` appends whatever the plugin's `config_template()`
returns and sets `plugin_module`, so a plugin that parses a section also
documents it and setup stays one command. The returned text is written verbatim
after core's, so it must not reopen a table core already declared -- core opens
`[airc]`, and TOML forbids declaring a table twice. Contribute `[airc.<name>]`
sub-tables and top-level sections instead.

## Bus payloads

A plugin owns its wire schemas: typed models wrapped in the shared
`bus.Envelope` under its own `type` strings. `bus` stays pure transport and never
learns what a payload is. Two plugins coexist on the same bus without sharing a
field. See `airc_coding.protocol` / `airc_coding.events` for the coding payloads.

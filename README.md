# airc (core)

The domain-neutral core of airc: a multi-agent chat room where expert agent
personas share a room with humans, answer with live tools, discuss among
themselves, and act on events posted onto a message bus. This repo is the
substrate only. Apps are plugins in their own repos that consume this one (as a
git submodule) and register their subscribers, personas, payloads, and config.

An app pulls this repo in at `vendor/core`. See `airc-room/PLUGINS.md` for the
plugin contract.

## Quick start

```sh
uv sync --all-packages
uv run airc            # a bare room: console transport, no plugin, no subscribers
```

A bare core room binds the console transport and has no domain behavior -- no
watchers, no personas beyond what you point it at, no tool groups. It is meant to
be driven by a plugin (`[airc] plugin_module = "..."` in the config) that supplies
those.

`airc --init-config` scaffolds a starter config with the sections core itself
loads. Add `--plugin <module>` to append that app's own sections and set
`plugin_module`, so an app is configured in one command rather than a core file
plus a manual paste -- see `config_template()` in `airc-room/PLUGINS.md`.

Talk in the console like IRC. Write `perf:` anywhere in a message (or a leading
`perf, compiler: ...` list) to force a reply from that agent; unaddressed messages
go to whichever agents the coordinator picks as adding value. `/help` lists
commands (`/agents`, `/threads`, `/t <id>`, `/new <title>`, `/quit`).

## The five core packages

A uv workspace, one `uv.lock`, one shared `.venv`. The root is a virtual
workspace (not itself a package). Every member is identically shaped
(`<member>/pyproject.toml`, `<member>/src/<name>/`, `<member>/tests/`).

- `bus/` -- directory-backed message bus: the domain-neutral transport primitives
  (`Envelope` with a typed routing string + opaque JSON payload, append-only
  `Topic`s with per-subscriber cursors, a claim `Channel`, a `BlobStore`, ulids).
  It knows nothing about what any payload means; typed domain payloads live in the
  consuming app.
- `airc-core/` -- shared substrate: the model stack + middleware, the MCP toolset
  and `tool_groups` gate, the token ledger, and common config (`load_common`).
- `airc-tools/` -- shell/read/edit coding tools exposed as an MCP server, plus the
  bwrap+cgroup sandbox mechanism. A standalone stdio server; not wired to the room
  except through the same tool-group gate any MCP server passes.
- `deepagent/` -- reusable agent-turn runtime: the harness, the bounded resumable
  reentry loop, the journal, the skill index. Extracted to be application-neutral;
  apps pass their own system prompt and tools. See its `DESIGN.md`.
- `airc-room/` -- the chat room core: `Room`, orchestrator, runner, personas,
  `subscribers/base`, `transports/` (console + Matrix), the store, timers, the
  structural prompts with domain holes. The `airc` console command launches it.
  The plugin contract is `airc-room/PLUGINS.md`.

## Architecture

```
console / matrix transport ──┐                    ┌── agent "a"  (LangGraph)
                             ├──> Room ──> Orchestr┼── agent "b"
plugin subscriber (bus) ─────┘    (SQLite)         └── agent "c"
                                                        │
                                                  MCP tools (stdio, tool_groups-gated)
```

- **Room** (`room.py`): message bus + persistence. Threads and messages; every
  message fans out to transports and queues for the orchestrator.
- **Orchestrator** (`orchestrator.py`): `handle:` address prefixes force repliers;
  otherwise a single fast-model **coordinator** call decides whether anyone should
  reply and who. It defaults to silence, protects human-to-human conversation, and
  converges a thread as it runs long. Threads run concurrently (a worker per
  thread); per-thread progress is persisted, so routing queued at crash time is
  replayed on restart. It is domain-blind: a `SYSTEM` announcement injects whatever
  follow-up prompt the announcing subscriber attached, and nothing when there is
  none.
- **Runner** (`runner.py`): one `create_agent` graph per persona. Per (thread,
  agent) LangGraph state in SQLite keeps each agent's own tool-call history; other
  participants' messages are injected as transcript lines on the agent's next turn.
- **Subscribers** (`subscribers/base.py`): the read side of the bus. Core defines
  the `Subscriber` protocol and a focus-aware fast-model triage base; concrete
  subscribers (commit commentary, findings, perf) are supplied by the app plugin
  through `build_subscribers`, not imported by name.
- **Transports** (`transports/`): display sinks + human input sources behind a
  `Transport` protocol (`run` + `deliver`, generic `thread_id` routing). Console
  (IRC-style, prompt_toolkit) and Matrix (matrix-nio; token-login, flat by
  default with optional `m.thread` mapping) are in-core. Which one binds is
  config: `[transport] kind = "console" | "matrix"` (a plugin may register more).
  Matrix reads a typed `[matrix]`
  section for its homeserver/token/rooms.

## The plugin seam

An app is four things, no framework required: bus payload schemas it owns,
subscribers/producers it registers, `agents/` personas it ships, and a config
block. Core resolves `[airc] plugin_module` via `importlib` and calls the plugin
factories (`build_subscribers`, `build_follow_ups`, `build_transport`, optional
`aux_services`, `personas_dir`); core carries zero imports of any specific plugin.
The contract, its version, and validation are in `airc-room/PLUGINS.md`.

## Development

```sh
uv sync --all-packages
scripts/run-suite-tests.sh          # pytest + ruff across the five members
```

Python 3.12+, `uv` for everything, `ruff` (line length 88). Tests use
`pytest-asyncio` in `asyncio_mode = "auto"`. Comment style is rationale-first
(explain the why/tradeoff, not the what); ascii ` -- `, never unicode
dashes/arrows/ellipsis in code or commit messages.

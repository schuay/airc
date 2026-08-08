# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""airc entry point: wire config, agents, room, transports, watchers.

Logs go to ~/.local/state/airc/airc.log so the console stays clean; -v
mirrors them to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import logging
import signal
import sys
from pathlib import Path

from airc_core import MCPToolset, TokenLog
from platformdirs import user_state_path

from . import __version__
from .config import CONFIG_DIR, apply_gcp_env_defaults, load_config
from .orchestrator import Orchestrator
from .personas import discover_personas, load_room_prompt
from .room import Room
from .runner import AgentRunner
from .store import Store
from .transports.console import ConsoleTransport

log = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> Path:
    log_dir = user_state_path("airc") / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "airc.log"
    # The file owns its timestamps (read directly with no journald around it);
    # the stderr stream is for journald under systemd, which stamps every line
    # itself, so drop asctime there to avoid a double timestamp in the journal.
    file_h = logging.FileHandler(log_file)
    file_h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname).1s %(name)s: %(message)s")
    )
    handlers: list[logging.Handler] = [file_h]
    if verbose:
        stream_h = logging.StreamHandler(sys.stderr)
        stream_h.setFormatter(
            logging.Formatter("%(levelname).1s %(name)s: %(message)s")
        )
        handlers.append(stream_h)
    # basicConfig leaves a handler's own formatter intact, so the per-handler
    # formats above win; it only wires the handlers + root level here.
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    for noisy in ("httpx", "httpcore", "google.auth", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_file


def _configured_model_ids(cfg, personas) -> dict[str, str]:
    ids = {"models.default": cfg.default_model, "models.filter": cfg.filter_model}
    for p in personas.values():
        # Resolve the "default"/"filter" role aliases to the concrete id so the
        # format check below sees a real provider:model, not the alias word (and
        # an aliased persona validates via the [models] entry it points at).
        if p.model_id and p.model_id not in ("default", "filter"):
            ids[f"agent {p.name} model"] = p.model_id
    return ids


def _validate_models(cfg, personas) -> None:
    """Reject malformed model ids up front (offline; provider prefix only).

    Model names are not checked here; that needs a provider API call, which is
    available on demand via --list-models.
    """
    from airc_core import check_model_id, supported_models_hint

    ids = _configured_model_ids(cfg, personas)
    bad = {where: mid for where, mid in ids.items() if check_model_id(mid)}
    if bad:
        listing = "\n".join(f"  {where}: {mid}" for where, mid in bad.items())
        raise SystemExit(
            f"invalid model id(s):\n{listing}\n{supported_models_hint()}"
            "\n(run `airc --list-models` to see available model names)"
        )


def _print_models(args: argparse.Namespace) -> None:
    """Handle --list-models: print available models per configured provider."""
    from airc_core import list_models

    from .config import apply_gcp_env_defaults, load_config

    cfg = load_config(args.config)
    apply_gcp_env_defaults(cfg)
    personas = discover_personas(
        _resolve_agents_dir(args, _load_plugin(cfg)), use_nicknames=cfg.use_nicknames
    )
    seen: set[str] = set()
    for mid in _configured_model_ids(cfg, personas).values():
        provider = mid.split(":", 1)[0]
        if provider in seen:
            continue
        seen.add(provider)
        names = list_models(mid)
        if names is None:
            print(f"{provider}: could not list (check provider/credentials)")
            continue
        print(f"{provider}: {len(names)} models")
        for n in names:
            print(f"  {provider}:{n}")


async def _list_tools(args: argparse.Namespace) -> None:
    """Handle --list-tools: print each MCP tool and the groups it matches."""
    from fnmatch import fnmatch

    cfg = load_config(args.config)
    apply_gcp_env_defaults(cfg)
    async with MCPToolset(cfg.mcp_servers, cfg.tool_groups) as toolset:
        for t in sorted(toolset.tools, key=lambda t: t.name):
            groups = [
                g
                for g, pats in cfg.tool_groups.items()
                if any(fnmatch(t.name, p) for p in pats)
            ]
            tag = ",".join(groups) if groups else "-- ungrouped --"
            desc = (t.description or "").strip().splitlines()
            print(f"  {t.name:<26} [{tag}]")
            if desc:
                print(f"      {desc[0][:88]}")
        print(f"\n{len(toolset.tools)} tools. Groups: {', '.join(cfg.tool_groups)}")


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "n/a"


def _thread_title(store, tid: int) -> str:
    """Resolve a ledger thread_id to its title via airc's store; the token
    ledger itself is thread-agnostic. '?' for thread 0 (triage) and any id with
    no thread row."""
    t = store.get_thread(tid) if tid else None
    return t.title if t else "?"


def _token_summary_line(store, tokens) -> str:
    """One-line aggregate for the periodic log: totals, 24h, top threads."""
    import time as _time

    tin, tout = tokens.totals()
    tcached = tokens.cached_input_total()
    since = _time.time() - 86400
    d_in, d_out = tokens.totals(since=since)
    d_cached = tokens.cached_input_total(since=since)
    top = ", ".join(
        f"#{tid} {_thread_title(store, tid)[:24]!r} {_fmt_tokens(i + o)}"
        for tid, i, o in tokens.top_threads(n=3)
    )
    by_model = "; ".join(
        f"{model} {_fmt_tokens(i)} ({_pct(c, i)} cached)/{_fmt_tokens(o)}"
        for model, i, o, c in tokens.totals_by_model(since=since)
    )
    return (
        f"tokens: all-time {_fmt_tokens(tin)} in ({_pct(tcached, tin)} cached)"
        f" / {_fmt_tokens(tout)} out;"
        f" 24h {_fmt_tokens(d_in)} ({_pct(d_cached, d_in)} cached)/{_fmt_tokens(d_out)};"
        f" 24h by model: {by_model or '(none)'};"
        f" top threads: {top or '(none)'}"
    )


async def _token_log_loop(store, tokens) -> None:
    """Log the token summary at startup and then hourly."""
    while True:
        log.info("%s", _token_summary_line(store, tokens))
        await asyncio.sleep(3600)


def _print_token_report(args: argparse.Namespace) -> None:
    """Handle --token-report: aggregated token costs from the shared ledger."""
    from .store import Store

    cfg = load_config(args.config)
    if args.db:
        cfg.db_path = args.db
    # The ledger is its own file; the store is read only to resolve thread titles.
    store = Store(cfg.db_path)
    tokens = TokenLog(cfg.token_db_path)
    tin, tout = tokens.totals()
    tcached = tokens.cached_input_total()
    print(
        f"all-time: {_fmt_tokens(tin)} in ({_pct(tcached, tin)} cached)"
        f" / {_fmt_tokens(tout)} out"
    )
    print("\nby kind:")
    for kind, i, o in tokens.totals_by_kind():
        print(f"  {kind:<12} {_fmt_tokens(i):>8} in  {_fmt_tokens(o):>8} out")
    print("\nby model:")
    for model, i, o, c in tokens.totals_by_model():
        print(
            f"  {model:<32} {_fmt_tokens(i):>8} in ({_pct(c, i)} cached)"
            f"  {_fmt_tokens(o):>8} out"
        )
    print("\nby agent:")
    for agent, i, o in tokens.totals_by_agent():
        print(f"  {agent:<12} {_fmt_tokens(i):>8} in  {_fmt_tokens(o):>8} out")
    print("\ntop threads:")
    for tid, i, o in tokens.top_threads(n=10):
        title = _thread_title(store, tid)
        print(
            f"  #{tid:<5} {_fmt_tokens(i):>8} in  {_fmt_tokens(o):>8} out  {title[:48]}"
        )
    print("\nheaviest turns (calls / max-per-call flags context accumulation):")
    for tid, agent, kind, i, calls, mx, _o in tokens.heaviest_turns(n=10):
        per = f"{calls} calls" if calls else "? calls"
        print(
            f"  #{tid:<5} {_fmt_tokens(i):>8} in over {per:>9}"
            f"  (max {_fmt_tokens(mx)}/call)  {agent}/{kind}"
        )
    tokens.close()
    store.close()


def _load_plugin(cfg):
    """Import the configured app plugin (cfg.plugin_module), or None for a bare
    room. Resolving it here -- not importing a plugin package by name -- is what
    keeps core domain-neutral: a non-coding deploy points plugin_module elsewhere
    (or leaves it empty) and never pulls the coding subscribers/handover in.

    The imported module is validated against the plugin contract (the three
    required factories + a compatible API version), so a misconfigured or stale
    plugin fails at startup with a clear message rather than deep in the wiring."""
    if not cfg.plugin_module:
        return None
    import importlib

    from .plugin import validate_plugin

    module = importlib.import_module(cfg.plugin_module)
    validate_plugin(module, cfg.plugin_module)
    return module


def _plugin_config_template(module_name: str | None) -> str | None:
    """The plugin's own starter-config sections, for --init-config --plugin.

    Imported directly rather than through _load_plugin: --init-config runs
    before a config exists, so there is nothing to read plugin_module from, and
    scaffolding a file is not a reason to enforce the full plugin contract. A
    plugin that ships no config_template() simply contributes nothing."""
    if not module_name:
        return None
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"--plugin {module_name!r} is not importable: {exc}."
            " install the app package, or drop --plugin for core sections only."
        ) from exc
    template = getattr(module, "config_template", None)
    if template is None:
        log.warning(
            "plugin %s ships no config_template(); writing core sections only",
            module_name,
        )
        return None
    return template()


def _resolve_agents_dir(args: argparse.Namespace, plugin) -> Path:
    """Where personas load from. Precedence: an explicit --agents-dir; then a
    ./agents in the service cwd (the dev/console path); then the plugin's own
    packaged personas (personas_dir(), so an app ships its agents/ with the
    package); then ~/.config/airc/agents. Letting the plugin contribute its
    directory is what frees an out-of-tree app from staging personas into the
    process cwd."""
    if args.agents_dir:
        return args.agents_dir
    local = Path.cwd() / "agents"
    if local.is_dir():
        return local
    personas_dir = getattr(plugin, "personas_dir", None) if plugin else None
    if personas_dir and (d := personas_dir()) is not None:
        return d
    return CONFIG_DIR / "agents"


def _call_local_tools(plugin, cfg, room) -> dict:
    """Call the plugin's build_local_tools, passing `room` only if it takes one.

    `room` is a compatible addition to the hook (see plugin.py), so a plugin
    written against build_local_tools(cfg) must keep contributing its tools.
    Dispatch by inspecting the signature rather than calling with the keyword and
    catching TypeError: the fallback would also swallow a TypeError raised inside
    the hook's own body, silently reporting "this plugin ships no local tools"
    for what is really a crash in it -- and would run the body twice.
    """
    hook = plugin.build_local_tools
    try:
        params = inspect.signature(hook).parameters.values()
    except (TypeError, ValueError):
        # A builtin or C-implemented callable has no introspectable signature.
        # Not expected for a plugin module function; fall back to the old form
        # rather than refusing to load any local tools at all.
        return hook(cfg)
    # **kwargs counts: it is the forward-compat idiom, so a plugin that wrote it
    # to receive exactly this kind of later addition must actually receive it.
    takes_room = any(
        p.name == "room" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params
    )
    return hook(cfg, room=room) if takes_room else hook(cfg)


def _resolve_transport_kind(args: argparse.Namespace, cfg, plugin) -> str:
    """The effective transport kind: config-selected ([transport] kind), with the
    legacy flags still honored and a plugin-declared default for a headless deploy
    that names no transport.

    Precedence: explicit legacy flags first (--chat, --headless), so an existing
    systemd unit keeps working unchanged; then [transport] kind from config; then
    a default. When neither a flag nor config names a transport, an interactive
    run (a TTY) is the console, and a non-interactive one asks the plugin for its
    default transport (the coding app names "gchat", the prod path) with a nudge to
    set [transport] explicitly. Core names no plugin here -- the default comes from
    the plugin's own default_transport_kind, so a bare room (or a plugin that
    declares none) falls back to the console.
    """
    if args.chat:
        return "gchat"
    if args.headless:
        return "headless"
    if cfg.transport_kind:
        return cfg.transport_kind
    # No explicit selection. An interactive run (a TTY) is the console -- this is
    # the `uv run airc` dev path and must not try to bind a network transport.
    # For a non-interactive deploy (systemd, no TTY) ask the plugin for the
    # transport it wants by default: the coding plugin names gchat, so an
    # un-edited prod config that has dropped --chat still binds it, with a nudge to
    # set [transport] explicitly. A plugin that declares no default (or no plugin
    # at all) falls back to the console. Core never names a transport itself.
    default = getattr(plugin, "default_transport_kind", None)
    if not sys.stdin.isatty() and default and (kind := default()):
        log.warning(
            "no [transport] kind set; defaulting to %r from the %s plugin."
            " set [transport] kind explicitly to silence this",
            kind,
            cfg.plugin_module,
        )
        return kind
    return "console"


async def _log_event(agent: str, event: str, detail: str) -> None:
    """on_event sink for headless runs: agent tool calls go to the log."""
    log.info("%s %s: %s", agent, event, detail)


async def _serve_headless() -> None:
    """Block until SIGTERM/SIGINT, keeping orchestrator and watchers alive.

    Replaces the console loop when running as a background service: there is
    no TTY to drive, so the foreground task just waits for a stop signal and
    lets the finally-block in amain() cancel the background tasks cleanly.
    """
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)
    log.info("running headless; send SIGTERM/SIGINT to stop")
    await stop.wait()
    log.info("stop signal received; shutting down")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="airc", description="Multi-agent chat room for V8 development"
    )
    ap.add_argument(
        "--config", type=Path, default=None, help="suite config path (airc.toml)"
    )
    ap.add_argument(
        "--agents-dir",
        type=Path,
        default=None,
        help="agent folders (default: ./agents, else the plugin's packaged"
        " personas, else ~/.config/airc/agents)",
    )
    ap.add_argument("--db", type=Path, default=None, help="database file override")
    ap.add_argument(
        "--no-watch",
        action="store_true",
        help="disable the plugin's bus subscribers",
    )
    ap.add_argument(
        "--init-config",
        action="store_true",
        help="write a template config.toml to the config path and exit",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="with --init-config, overwrite an existing file",
    )
    ap.add_argument(
        "--plugin",
        default=None,
        metavar="MODULE",
        help="with --init-config, also write this app plugin's own config"
        " sections (e.g. myapp.app)",
    )
    ap.add_argument(
        "--list-models",
        action="store_true",
        help="list models available from the configured providers and exit",
    )
    ap.add_argument(
        "--list-tools",
        action="store_true",
        help="list MCP tools and the tool_groups each matches, then exit",
    )
    ap.add_argument(
        "--token-report",
        action="store_true",
        help="print aggregated token usage (overall, by kind/agent, top threads)",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="run without the interactive console (for systemd/background use)",
    )
    ap.add_argument(
        "--chat",
        action="store_true",
        help="run the Google Chat transport (implies headless; equivalent to"
        ' [transport] kind = "gchat"); needs an [airc.chat] section',
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="log to stderr too")
    ap.add_argument("--version", action="version", version=f"airc {__version__}")
    return ap.parse_args(argv)


async def amain(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.db:
        cfg.db_path = args.db
    apply_gcp_env_defaults(cfg)

    # Load the plugin first: it may contribute the personas directory, so agents
    # resolution has to know it before discovering personas.
    plugin = _load_plugin(cfg)
    agents_dir = _resolve_agents_dir(args, plugin)
    personas = discover_personas(agents_dir, use_nicknames=cfg.use_nicknames)
    room_prompt = load_room_prompt(agents_dir)
    log.info("personas from %s: %s", agents_dir, ", ".join(personas))

    _validate_models(cfg, personas)

    store = Store(cfg.db_path)
    tokens = TokenLog(cfg.token_db_path)
    room = Room(store)

    # Which chat frontend to bind. "console" and "headless" are core; every other
    # kind (gchat now, matrix later) is resolved through the app plugin, so core
    # stays domain- and corp-neutral. aux_tasks are extra background coroutines a
    # transport owns (gchat's full-space renewal loop).
    kind = _resolve_transport_kind(args, cfg, plugin)
    console = None
    transport = None
    aux_tasks: list = []
    if kind == "console":
        console = ConsoleTransport(room, personas)
        room.add_transport(console)
    elif kind == "matrix":
        if cfg.matrix is None:
            raise SystemExit(
                'transport kind "matrix" needs a [matrix] config section'
                " (homeserver, user_id, access_token)"
            )
        from .transports.matrix import MatrixTransport

        transport = MatrixTransport(cfg.matrix, room, store)
        room.add_transport(transport)
    elif kind != "headless":
        if plugin is None or not hasattr(plugin, "build_transport"):
            raise SystemExit(
                f"transport kind {kind!r} needs an app plugin that supplies it;"
                f" none is configured (plugin_module={cfg.plugin_module!r})"
            )
        transport = plugin.build_transport(cfg, room, store, kind)
        if transport is None:
            raise SystemExit(
                f"unknown transport kind {kind!r};"
                f" the {cfg.plugin_module!r} plugin supplies no such transport"
            )
        room.add_transport(transport)
        # A transport may own side loops (e.g. gchat's space-subscription
        # renewal); the room runs each as its own background task, below.
        aux = getattr(transport, "aux_services", None)
        aux_tasks = list(aux()) if aux else []

    on_event = console.on_event if console else _log_event
    # In-memory timer scheduler: every chat persona gets the timer tools
    # (create/list/cancel), which schedule through it; scheduler.deliver (wired
    # below) drives the wake turn. Always present -- it idles until a timer is set.
    from .timers import TimerScheduler

    scheduler = TimerScheduler(store)
    # Local (non-MCP) tools the plugin contributes, keyed by tool_group name (e.g.
    # the grocery memory tools under "memory"). The runner grants each group to a
    # persona that lists it, exactly as for MCP groups. A plugin without the hook
    # (or a bare room) contributes none.
    local_tool_groups: dict = {}
    if plugin and hasattr(plugin, "build_local_tools"):
        # `room` reaches only a hook that declares it -- see _call_local_tools.
        local_tool_groups = _call_local_tools(plugin, cfg, room) or {}
    # Long-term memory is a CORE feature (config + per-turn injection live in
    # core), so core -- not a plugin -- provides its tool_group. On when
    # [airc.memory].enabled; a persona opts in by listing "memory" in tool_groups.
    if cfg.memory.enabled:
        from .memory import MEMORY_GROUP, make_memory_tools

        local_tool_groups[MEMORY_GROUP] = make_memory_tools(cfg.memory.path)
    async with (
        MCPToolset(cfg.mcp_servers, cfg.tool_groups) as toolset,
        AgentRunner(
            cfg,
            personas,
            toolset,
            store,
            on_event=on_event,
            room_prompt=room_prompt,
            timer_scheduler=scheduler,
            local_tool_groups=local_tool_groups,
        ) as runner,
    ):
        if console:
            # Console listed all personas above; narrow to the usable ones.
            console.agents = runner.agents
        # The app plugin (resolved from config) supplies the announcement
        # response handlers; the room dispatches to them by follow_up key and
        # stays domain-blind. A bare room (no plugin) registers none.
        follow_ups = (
            plugin.build_follow_ups(cfg, store, agents_dir=agents_dir) if plugin else {}
        )
        orchestrator = Orchestrator(cfg, room, runner, store, follow_ups=follow_ups)
        scheduler.deliver = orchestrator.deliver_wake
        # Rebuild pending timers from the store before run() starts, so a
        # timer set before the restart still fires (one immediately if it came
        # due while the daemon was down). deliver is wired just above, so a
        # due timer has a live path the moment run() ticks.
        scheduler.restore()
        # The orchestrator must be the FIRST task created: its synchronous
        # recovery pass then runs before any transport/watcher coroutine,
        # which is what excludes double-processing of recovered messages.
        tasks = [asyncio.create_task(orchestrator.run(), name="orchestrator")]
        tasks.append(asyncio.create_task(scheduler.run(), name="timers"))
        tasks.append(asyncio.create_task(_token_log_loop(store, tokens), name="tokens"))
        # A non-console transport runs its inbound loop as a background task
        # (the console owns the foreground, below); its aux services (gchat's
        # space-subscription renewal) join as peers.
        if transport is not None:
            tasks.append(asyncio.create_task(transport.run(), name=transport.name))
            for i, svc in enumerate(aux_tasks):
                tasks.append(
                    asyncio.create_task(svc(), name=f"{transport.name}-aux{i}")
                )
        subscribers = (
            plugin.build_subscribers(cfg, room, store, toolset)
            if plugin and not args.no_watch
            else []
        )
        # Background services the plugin supervises: periodic/clock loops with
        # no bus topic (e.g. grocery's memory compaction). Gated by --no-watch
        # alongside subscribers, since both are the plugin's autonomous work.
        services = (
            plugin.build_services(cfg, room, store)
            if plugin and not args.no_watch and hasattr(plugin, "build_services")
            else []
        )
        # One startup line stating what was actually wired: the transport and
        # the bus subscribers. A silently-ignored config section (e.g. a flat
        # [chat]/[[commentary]] after the [airc.*] namespacing) shows up here
        # as transport=console or an empty subscriber list, instead of just
        # going quiet.
        log.info(
            "airc: transport=%s bus_root=%s subscribers=[%s] services=[%s]",
            kind,
            cfg.bus_root,
            ", ".join(s.name for s in subscribers) or "none",
            ", ".join(s.name for s in services) or "none",
        )
        for src in subscribers:
            tasks.append(asyncio.create_task(src.run(), name=f"source:{src.name}"))
        for svc in services:
            tasks.append(asyncio.create_task(svc.run(), name=f"service:{svc.name}"))
        try:
            await (console.run() if console else _serve_headless())
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # Give a transport a chance to resolve outstanding state (gchat
            # sweeps its "thinking..." cards); duck-typed, so console/matrix
            # without one are skipped.
            aclose = getattr(transport, "aclose", None) if transport else None
            if aclose:
                await aclose()
    store.close()


def main() -> None:
    args = parse_args()
    if args.init_config:
        from .config import CONFIG_DIR, write_template_config

        write_template_config(
            args.config or CONFIG_DIR / "airc.toml",
            args.force,
            _plugin_config_template(args.plugin),
            args.plugin,
        )
        return
    if args.list_models:
        _print_models(args)
        return
    if args.list_tools:
        asyncio.run(_list_tools(args))
        return
    if args.token_report:
        _print_token_report(args)
        return
    log_file = _setup_logging(args.verbose)
    print(f"airc {__version__} (logs: {log_file})")
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(amain(args))


if __name__ == "__main__":
    main()

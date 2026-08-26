# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

import pytest
from airc_room.config import Config, load_config, write_template_config


def _write(tmp_path, body: str):
    p = tmp_path / "config.toml"
    p.write_text(body)
    return p


def test_resolve_model_aliases_and_passthrough():
    cfg = Config(default_model="prov:big", filter_model="prov:cheap")
    # Role aliases map to the configured ids; a change to [models] follows.
    assert cfg.resolve_model("filter") == "prov:cheap"
    assert cfg.resolve_model("default") == "prov:big"
    # Empty/None (no `model` in agent.toml) is the default model.
    assert cfg.resolve_model(None) == "prov:big"
    assert cfg.resolve_model("") == "prov:big"
    # A real provider:model id passes through untouched.
    assert cfg.resolve_model("openrouter:x/y") == "openrouter:x/y"


def test_orchestrator_defaults(tmp_path):
    cfg = load_config(_write(tmp_path, ""))
    assert cfg.orchestrator.soft_turn_budget == 8
    assert cfg.orchestrator.max_turns == 24
    assert cfg.orchestrator.max_responders == 2


def test_room_tolerates_coding_project_pack_sections(tmp_path):
    cfg = load_config(
        _write(
            tmp_path,
            """
            [prompt_pack]
            repo = "/packs"
            revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            cache_root = "/cache"
            project = "chromium-blink"
            workflows = ["bugfix"]

            [projects.chromium-blink]
            repo = "chromium"
            checkout = "/chromium"
            """,
        )
    )
    assert cfg.plugin_config == {}


def test_legacy_turn_budget_maps_to_soft(tmp_path):
    cfg = load_config(_write(tmp_path, "[airc.orchestrator]\nturn_budget = 5\n"))
    assert cfg.orchestrator.soft_turn_budget == 5
    assert cfg.orchestrator.max_turns == 24


def test_explicit_knobs(tmp_path):
    body = "[airc.orchestrator]\nsoft_turn_budget = 10\nmax_turns = 40\n"
    cfg = load_config(_write(tmp_path, body))
    assert cfg.orchestrator.soft_turn_budget == 10
    assert cfg.orchestrator.max_turns == 40


def test_max_turns_clamped_to_at_least_soft(tmp_path):
    body = "[airc.orchestrator]\nsoft_turn_budget = 30\nmax_turns = 10\n"
    cfg = load_config(_write(tmp_path, body))
    assert cfg.orchestrator.max_turns == 30


def test_plugin_config_empty_by_default(tmp_path):
    # Core models no plugin section; an [airc] with only core keys leaves
    # plugin_config empty.
    assert load_config(_write(tmp_path, "")).plugin_config == {}
    cfg = load_config(_write(tmp_path, '[airc]\nroom_topic = "x"\n'))
    assert cfg.plugin_config == {}


def test_bare_room_with_plugin_sections_warns(tmp_path, caplog):
    # No plugin_module but plugin sections present: nobody parses them, so they
    # would be silently dropped. Core warns rather than booting a quietly-wrong
    # room. (plugin_module defaults to "" -- a bare room -- so just omit it.)
    import logging

    body = '[[airc.commentary]]\nrepo = "v8"\n'
    with caplog.at_level(logging.WARNING):
        cfg = load_config(_write(tmp_path, body))
    assert cfg.plugin_module == "" and "commentary" in cfg.plugin_config
    assert "no plugin_module set" in caplog.text
    # With a plugin_module set, the plugin owns them -- no warning.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        load_config(_write(tmp_path, body + '[airc]\nplugin_module = "x.y"\n'))
    assert "no plugin_module set" not in caplog.text


def test_voices_default_empty(tmp_path):
    assert load_config(_write(tmp_path, "")).voices == {}


def test_voices_parsed_and_expanded(tmp_path):
    body = '[airc.voices]\ncompiler = "~/guides/compiler.md"\ngc = "/abs/gc.md"\n'
    cfg = load_config(_write(tmp_path, body))
    assert set(cfg.voices) == {"compiler", "gc"}
    assert not str(cfg.voices["compiler"]).startswith("~")  # expanduser applied
    assert str(cfg.voices["gc"]) == "/abs/gc.md"


def test_caching_on_by_default(tmp_path):
    cfg = load_config(_write(tmp_path, ""))
    assert cfg.caching_explicit is True
    assert cfg.cache_ttl_minutes == 30


def test_caching_overrides(tmp_path):
    body = "[caching]\nexplicit = false\nttl_minutes = 30\n"
    cfg = load_config(_write(tmp_path, body))
    assert cfg.caching_explicit is False
    assert cfg.cache_ttl_minutes == 30


def test_caching_section_without_explicit_stays_on(tmp_path):
    cfg = load_config(_write(tmp_path, "[caching]\nttl_minutes = 90\n"))
    assert cfg.caching_explicit is True
    assert cfg.cache_ttl_minutes == 90


def test_cache_ttl_clamped_above_turn_timeout(tmp_path):
    # A TTL that would let a cache expire mid-turn is clamped up so it cannot.
    body = "[caching]\nttl_minutes = 10\n[airc.orchestrator]\nturn_timeout = 1800\n"
    cfg = load_config(_write(tmp_path, body))
    assert cfg.cache_ttl_minutes * 60 > cfg.orchestrator.turn_timeout
    assert cfg.cache_ttl_minutes == 35  # ceil(1800/60) + 5


def test_plugin_sections_passed_through_raw(tmp_path):
    # Core does not model any plugin section (commit commentary, findings, the
    # gchat [airc.chat], ...); it carries every non-core [airc] key through
    # verbatim as plugin_config for the app overlay to parse.
    body = (
        "[airc]\n"
        "commit_digest = true\n"
        "[airc.chat]\n"
        'project = "p"\n'
        'subscription = "s"\n'
        "[[airc.commentary]]\n"
        'repo = "v8"\n'
    )
    pc = load_config(_write(tmp_path, body)).plugin_config
    assert pc["commit_digest"] is True
    assert pc["chat"] == {"project": "p", "subscription": "s"}
    assert pc["commentary"] == [{"repo": "v8"}]


@pytest.mark.parametrize(
    "body,where",
    [
        ("[airc.orchestrator]\nmax_responder = 2\n", r"\[airc\.orchestrator\]"),
        (
            '[airc.memory]\nenabled = true\npath = "/m"\nenable = true\n',
            r"\[airc\.memory\]",
        ),
        ('[transport]\nkind = "console"\nknd = "x"\n', r"\[transport\]"),
        ('[handover]\nenabled = true\nautonmy = "draft-only"\n', r"\[handover\]"),
        ("[caching]\nttl_minute = 30\n", r"\[caching\]"),
        ('[gcp]\nprojekt = "p"\n', r"\[gcp\]"),
        ("[mcp]\nserver = {}\n", r"\[mcp\]"),
    ],
)
def test_a_typo_in_any_section_errors(tmp_path, body, where):
    """Every section is strict, not just the top level.

    A key the loader silently ignores reads back to the operator as a setting
    that was applied. Usually that costs an unwanted default; once it cost a
    security boundary (a misspelled allowlist key left it empty, i.e.
    unrestricted). The two are indistinguishable at the point of the typo.
    """
    with pytest.raises(SystemExit, match=f"unknown {where} key"):
        load_config(_write(tmp_path, body))


def test_allowed_keys_track_the_dataclass_not_a_hand_written_list():
    """The strict check derives its key set from the dataclass that models the
    section, so a new setting is one field and nothing else.

    Pinned because the failure of a hand-written copy is silent and backwards: a
    field added without its key would reject config that legitimately sets it.
    Adding a field HERE (not touching any parser) must make the key acceptable.
    """
    import dataclasses

    from airc_core.config import reject_unknown_fields

    @dataclasses.dataclass
    class _Spec:
        alpha: int = 1

    reject_unknown_fields({"alpha": 1}, _Spec, "[x]")  # a field is accepted
    with pytest.raises(SystemExit, match="unknown \\[x\\] key"):
        reject_unknown_fields({"beta": 1}, _Spec, "[x]")

    _WithBeta = dataclasses.make_dataclass("_WithBeta", [("alpha", int), ("beta", int)])
    reject_unknown_fields({"beta": 1}, _WithBeta, "[x]")  # now it is


def test_a_legacy_alias_stays_accepted(tmp_path):
    """turn_budget is the old spelling of soft_turn_budget and is still honoured,
    so strictness must not break a config that predates the rename."""
    cfg = load_config(_write(tmp_path, "[airc.orchestrator]\nturn_budget = 3\n"))
    assert cfg.orchestrator.soft_turn_budget == 3


@pytest.mark.parametrize(
    "body",
    [
        '[repos]\nanything_goes = "/src/x"\n',
        "[tool_groups]\nmy_own_group = []\n",
        '[models]\nsome_role = "provider:m"\n',
        '[mcp.servers.whatever]\ncommand = "x"\nargs = []\n',
    ],
)
def test_open_sections_stay_open(tmp_path, body):
    """User-named maps and role maps have no fixed key set: [repos] names
    checkouts, [tool_groups] names groups, [models] names roles a persona may
    select, and an [mcp.servers.*] spec is passed verbatim to a third-party
    client whose schema is not ours. Strictness there would be a category error."""
    load_config(_write(tmp_path, body))


def test_unknown_toplevel_section_errors(tmp_path):
    # A mis-namespaced or typo'd top-level section is a silent-misconfig footgun,
    # so the loader rejects it rather than ignoring it.
    with pytest.raises(SystemExit, match="unknown config section"):
        load_config(_write(tmp_path, "[watcher]\nx = 1\n"))  # should be [watchers]
    with pytest.raises(SystemExit, match="chat"):
        load_config(
            _write(tmp_path, '[chat]\nproject = "p"\n')
        )  # bare, not [airc.chat]


def test_known_sibling_sections_allowed(tmp_path):
    # The shared suite file carries the other daemons' sections; core must not
    # choke on [watchers.*]/[processors.*] just because they are not its own.
    body = (
        '[[watchers.repo]]\nname = "v8"\n'
        "[processors.review]\npasses = 2\n"
        '[icompleteu]\ncontrol_root = "/srv/ctl"\n'
        '[handover]\nenabled = true\nkinds = ["repro"]\n'
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.handover.enabled is True


def test_transport_kind_absent_and_parsed(tmp_path):
    assert load_config(_write(tmp_path, "")).transport_kind == ""
    cfg = load_config(_write(tmp_path, '[transport]\nkind = "gchat"\n'))
    assert cfg.transport_kind == "gchat"
    # The [airc.transport] namespace (matching every other room section) is
    # accepted too, so it is not silently ignored; it wins over top-level.
    cfg = load_config(_write(tmp_path, '[airc.transport]\nkind = "console"\n'))
    assert cfg.transport_kind == "console"
    both = '[transport]\nkind = "gchat"\n[airc.transport]\nkind = "console"\n'
    assert load_config(_write(tmp_path, both)).transport_kind == "console"


def test_repos_parsed(tmp_path):
    # repos is a shared suite section core still owns (commit provenance); the
    # commentary/findings that used to live beside it are the plugin's now.
    body = 'bus_root = "/tmp/airc-bus"\n[repos]\nv8 = "~/v8/v8"\njsc = "/src/jsc"\n'
    cfg = load_config(_write(tmp_path, body))
    assert str(cfg.bus_root) == "/tmp/airc-bus"
    assert cfg.repos["v8"].endswith("/v8/v8")  # ~ expanded
    assert not cfg.repos["v8"].startswith("~")
    assert cfg.repos["jsc"] == "/src/jsc"


def test_explicit_missing_config_raises(tmp_path):
    # A typo'd --config must fail the start, not run "successfully" on all
    # defaults; only the implicit default location may be absent.
    import pytest

    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "no-such.toml")


def test_handover_bus_root_defaults_to_suite_bus_root(tmp_path):
    # Without an explicit [handover].bus_root the handover publishes to the
    # suite bus_root, where icompleteu polls -- not to a hardcoded DATA_DIR
    # fallback nothing reads.
    body = 'bus_root = "/tmp/airc-bus"\n[handover]\nenabled = true\nkinds = ["repro"]\n'
    cfg = load_config(_write(tmp_path, body))
    assert str(cfg.handover.bus_root) == "/tmp/airc-bus"
    # And an explicit one still wins.
    body += 'bus_root = "/tmp/icu-bus"\n'
    cfg = load_config(_write(tmp_path, body))
    assert str(cfg.handover.bus_root) == "/tmp/icu-bus"


def test_handover_kinds_round_trip(tmp_path):
    # Unset stays at the repro-only default when the section is disabled --
    # the one kind that cannot produce a CL, so enabling handover later
    # without pinning a list is conservative by construction, and the parser
    # makes that moment a decision (see the next test).
    h = load_config(_write(tmp_path, "[handover]\n")).handover
    assert h.enabled is False and h.kinds == ["repro"]
    body = '[handover]\nenabled = true\nkinds = ["repro", "perf"]\n'
    assert load_config(_write(tmp_path, body)).handover.kinds == ["repro", "perf"]
    # The drain state parses; the components that read it do the warning.
    assert (
        load_config(_write(tmp_path, "[handover]\nkinds = []\n")).handover.kinds == []
    )


def test_handover_enabled_without_kinds_refuses_to_guess(tmp_path):
    # Before kinds existed, enabled = true with no per-kind policy meant EVERY
    # kind; the default is now repro only, so honouring the omission would
    # silently stop a live deployment's bugfix/perf/task jobs on upgrade.
    # Refuse at startup and name both resolutions -- the operator states the
    # list once and the policy is explicit from then on.
    with pytest.raises(SystemExit, match="without kinds"):
        load_config(_write(tmp_path, "[handover]\nenabled = true\n"))


def test_handover_repro_only_names_its_replacement(tmp_path):
    # The strict check would reject the old key generically, with a message
    # that reads like a rename; this says what the config becomes, so a prod
    # file still carrying it is a one-line fix at the deploy moment it fires.
    with pytest.raises(SystemExit, match="repro_only is gone"):
        load_config(_write(tmp_path, "[handover]\nrepro_only = true\n"))


def test_handover_repro_names_its_replacement(tmp_path):
    # Same treatment for the route switch the allowlist subsumed: repro = true
    # is kinds += "repro" (which also routes repro-suitable findings through
    # the verified-repro detour -- the one producer of repro jobs derives its
    # route from the permission now).
    with pytest.raises(SystemExit, match="repro is gone"):
        load_config(_write(tmp_path, "[handover]\nrepro = true\n"))


def test_handover_kinds_must_be_a_list(tmp_path):
    # A bare string iterates per character into an allowlist matching no kind;
    # the same guard [airc.cl_review] spaces carries, caught here for both
    # components that read this one table.
    with pytest.raises(SystemExit, match="kinds must be a list"):
        load_config(_write(tmp_path, '[handover]\nkinds = "repro"\n'))


def test_core_default_tool_groups_are_empty():
    # The core substrate ships no tool groups: every app supplies its own. The
    # coding v8-utils/gdb groups (and their content tests) live in airc-coding.
    from airc_room.config import DEFAULT_TOOL_GROUPS

    assert DEFAULT_TOOL_GROUPS == {"read": [], "active": []}


def test_template_config_writes_and_loads(tmp_path):
    path = tmp_path / "sub" / "config.toml"
    write_template_config(path)
    assert path.exists()
    # The emitted template must parse and yield the documented defaults.
    cfg = load_config(path)
    assert cfg.default_model == "google_vertexai:gemini-2.5-flash"
    assert cfg.orchestrator.max_turns == 24
    # Core scaffolds only what core loads: no plugin is named, and no app
    # section rides along. An app's sections come from its config_template().
    assert cfg.plugin_module == ""
    assert cfg.plugin_config == {}


def test_template_config_appends_plugin_sections(tmp_path):
    # The two halves must concatenate into ONE parseable file -- the whole point
    # of the hook is that setup stays a single command.
    path = tmp_path / "config.toml"
    write_template_config(
        path,
        plugin_template="[airc.myfeature]\nenabled = true\n[airc.other]\nk = 1\n",
        plugin_module="myapp.app",
    )
    cfg = load_config(path)
    assert cfg.default_model == "google_vertexai:gemini-2.5-flash"
    assert cfg.plugin_config["myfeature"] == {"enabled": True}
    assert cfg.plugin_config["other"] == {"k": 1}
    # Naming the plugin must also ACTIVATE it: an emitted file that carries app
    # sections but no plugin_module is a bare room that silently ignores them.
    assert cfg.plugin_module == "myapp.app"


def test_template_config_plugin_sections_survive_odd_spacing(tmp_path):
    # Plugins terminate their template however they like; the seam must not
    # depend on it, so a missing or doubled newline still parses.
    path = tmp_path / "config.toml"
    write_template_config(path, plugin_template="\n\n[airc.myfeature]\nenabled = true")
    cfg = load_config(path)
    assert cfg.plugin_config["myfeature"] == {"enabled": True}


def test_template_config_refuses_clobber(tmp_path):
    path = tmp_path / "config.toml"
    write_template_config(path)
    with pytest.raises(SystemExit):
        write_template_config(path)
    write_template_config(path, force=True)  # force overwrites


def test_orchestrator_turn_settings(tmp_path):
    body = "[airc.orchestrator]\nmax_concurrent_turns = 9\nturn_timeout = 60\n"
    orch = load_config(_write(tmp_path, body)).orchestrator
    assert orch.max_concurrent_turns == 9
    assert orch.turn_timeout == 60.0


def test_grounding_reminder_default_and_override(tmp_path):
    assert load_config(_write(tmp_path, "")).grounding_reminder_tokens == 200_000
    cfg = load_config(_write(tmp_path, "[airc]\ngrounding_reminder_tokens = 0\n"))
    assert cfg.grounding_reminder_tokens == 0

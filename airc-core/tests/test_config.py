# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Tests for the shared config substrate."""

from __future__ import annotations

from pathlib import Path

import pytest
from airc_core import (
    DEFAULT_BUS_ROOT,
    DEFAULT_TOOL_GROUPS,
    CommonConfig,
    ModelProfile,
    apply_gcp_env_defaults,
    load_common,
    parse_handover_fields,
)


def test_empty_raw_uses_defaults():
    cfg = load_common({})
    assert cfg.models == {}
    assert cfg.mcp_servers == {}
    assert cfg.tool_groups == DEFAULT_TOOL_GROUPS
    assert cfg.bus_root == DEFAULT_BUS_ROOT
    assert cfg.caching_explicit is True
    assert cfg.cache_ttl_minutes == 30


def test_models_table_kept_raw_for_per_component_selection():
    # Components select their own key from one shared [models] section.
    cfg = load_common({"models": {"default": "a", "filter": "b", "review": "c"}})
    assert cfg.models == {"default": "a", "filter": "b", "review": "c"}


def test_mcp_servers_and_paths_parsed():
    raw = {
        "bus_root": "~/somewhere/bus",
        "token_db_path": "~/somewhere/tokens.db",
        "mcp": {
            "servers": {
                "v8-utils": {
                    "command": "v8-mcp",
                    "args": ["--enable-pd"],
                    "enable_in_sandbox": True,
                },
                "buganizer": {"command": "bug-mcp"},
            }
        },
        "repos": {"v8": "/path/to/v8/v8"},
    }
    cfg = load_common(raw)
    assert cfg.mcp_servers["v8-utils"]["args"] == ["--enable-pd"]
    assert "enable_in_sandbox" not in cfg.mcp_servers["v8-utils"]
    assert cfg.mcp_enable_in_sandbox == {
        "v8-utils": True,
        "buganizer": False,
    }
    assert cfg.bus_root == Path("~/somewhere/bus").expanduser()
    assert cfg.token_db_path == Path("~/somewhere/tokens.db").expanduser()
    assert cfg.repos == {"v8": str(Path("/path/to/v8/v8").expanduser())}


def test_tool_groups_override_replaces_default():
    cfg = load_common({"tool_groups": {"read": ["repo_git_*"]}})
    assert cfg.tool_groups == {"read": ["repo_git_*"]}


def test_gcp_values_coerced_to_str():
    # A bare project number in TOML is an int; os.environ would reject it.
    cfg = load_common({"gcp": {"project": 12345}})
    assert cfg.gcp == {"project": "12345"}


def test_apply_gcp_env_defaults_only_fills_unset(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_QUOTA_PROJECT", raising=False)
    apply_gcp_env_defaults({"project": "p", "location": "us-central1"})
    import os

    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "p"
    assert os.environ["GOOGLE_CLOUD_LOCATION"] == "us-central1"
    # quota project falls back to project when unset
    assert os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] == "p"


def test_apply_gcp_env_defaults_external_override_wins(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "already-set")
    apply_gcp_env_defaults({"project": "from-config"})
    import os

    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "already-set"


def test_common_config_is_constructible_directly():
    # Components may build one in tests without going through TOML.
    cfg = CommonConfig(bus_root=Path("/tmp/bus"))
    assert cfg.bus_root == Path("/tmp/bus")


def test_handover_fields_parse_shared_shape():
    fields = parse_handover_fields(
        {
            "enabled": True,
            "autonomy": "upload-wip",
            "bus_root": "~/bus",
            "kinds": ["bugfix", "repro"],
        }
    )
    assert fields.enabled is True
    assert fields.autonomy == "upload-wip"
    assert fields.bus_root == "~/bus"
    assert fields.kinds == ["bugfix", "repro"]


@pytest.mark.parametrize("old", ["repro", "repro_only"])
def test_handover_fields_reject_old_keys_with_migration_help(old):
    with pytest.raises(ValueError, match=f"{old} is gone"):
        parse_handover_fields({old: True}, error=ValueError)


def test_handover_fields_reject_ambiguous_or_missing_allowlists():
    with pytest.raises(ValueError, match="must be a list"):
        parse_handover_fields({"kinds": "repro"}, error=ValueError)
    with pytest.raises(ValueError, match="without kinds"):
        parse_handover_fields({"enabled": True}, error=ValueError)


# ── [model_providers] ───────────────────────────────────────────────────────


@pytest.fixture
def registry():
    """A clean provider registry, restored afterwards -- load_common registers
    into process-global module state, so two configs in one session would
    otherwise see each other."""
    from airc_core import model as m

    saved = dict(m._CUSTOM_PROVIDERS)
    m._CUSTOM_PROVIDERS.clear()
    try:
        yield m
    finally:
        m._CUSTOM_PROVIDERS.clear()
        m._CUSTOM_PROVIDERS.update(saved)


def test_model_providers_parsed_and_registered(registry):
    cfg = load_common(
        {
            "model_providers": {
                "mybackend": {
                    "factory": "mypkg.provider:make_model",
                    "requires_env": "MYBACKEND_TOKEN",
                }
            },
            "models": {"default": "mybackend:v1"},
        }
    )
    assert cfg.model_providers["mybackend"]["factory"] == "mypkg.provider:make_model"
    # Registered as a side effect, which is what makes the id valid suite-wide:
    # every component reaches make_model through load_common.
    assert registry.check_model_id("mybackend:v1") is None


def test_model_providers_absent_leaves_registry_empty(registry):
    # The dormant case: a config with no [model_providers] must not make any
    # previously-invalid id start validating.
    cfg = load_common({"models": {"default": "google_vertexai:gemini-3.6-flash"}})
    assert cfg.model_providers == {}
    assert registry.check_model_id("mybackend:v1") is not None


def test_model_providers_rejects_typo_and_missing_factory(registry):
    # A misspelled requires_env would silently mean "no credential check" and
    # read back to the operator as if it had been honoured.
    with pytest.raises(SystemExit, match="requires_envv"):
        load_common(
            {"model_providers": {"mine": {"factory": "m:f", "requires_envv": "X"}}}
        )
    with pytest.raises(SystemExit, match="needs factory"):
        load_common({"model_providers": {"mine": {"requires_env": "X"}}})
    with pytest.raises(SystemExit, match="must be a table"):
        load_common({"model_providers": {"mine": "m:f"}})


def test_model_providers_builtin_prefix_names_the_section(registry):
    # register_provider's ValueError becomes a SystemExit naming the section, so
    # the operator gets the line to edit instead of a traceback.
    with pytest.raises(SystemExit, match=r"\[model_providers.anthropic\]"):
        load_common({"model_providers": {"anthropic": {"factory": "m:f"}}})


def test_model_providers_reparse_of_same_config_is_idempotent(registry):
    # icompleteu calls load_common several times per process over the same file.
    raw = {"model_providers": {"mybackend": {"factory": "mypkg:make"}}}
    load_common(raw)
    load_common(raw)
    assert registry.check_model_id("mybackend:v1") is None


def test_model_providers_bad_factory_shape_fails_at_parse(registry):
    # Shape is checked at registration, so a malformed path fails at STARTUP with
    # the section named -- not inside the first turn that needs the model.
    with pytest.raises(SystemExit, match="must be 'module:attr'"):
        load_common({"model_providers": {"mine": {"factory": "not_dotted"}}})
    assert "mine" not in registry._CUSTOM_PROVIDERS


def test_a_string_entry_stays_a_bare_id():
    # Most entries have nothing to say beyond which model; the string form is
    # not a legacy form being tolerated.
    cfg = load_common({"models": {"default": "prov:m"}})
    assert cfg.model_profiles["default"] == ModelProfile(key="default", id="prov:m")
    assert cfg.model_profiles["default"].call_kwargs == {}


def test_a_table_entry_carries_the_effort_level():
    cfg = load_common(
        {
            "models": {
                "review": {
                    "id": "google_anthropic_vertex:claude-opus-5",
                    "effort": "xhigh",
                },
                "verify": {
                    "id": "google_anthropic_vertex:claude-opus-5",
                    "effort": "low",
                },
            }
        }
    )
    assert cfg.model_profiles["review"].effort == "xhigh"
    assert cfg.model_profiles["verify"].call_kwargs == {"effort": "low"}
    # Two keys on one id is normal for the type, not duplication.
    assert cfg.model_profiles["review"].id == cfg.model_profiles["verify"].id


def test_models_still_maps_key_to_id_for_the_egress_allowlists():
    # The sandbox builds its Vertex/Anthropic route allowlists by splitting
    # every value here on ":", and a profile that never reaches that set means
    # a box with no route for it -- every call in-box fails.
    cfg = load_common(
        {
            "models": {
                "default": "google_vertexai:gemini-3.6-flash",
                "review": {
                    "id": "google_anthropic_vertex:claude-opus-5",
                    "effort": "max",
                },
            }
        }
    )
    assert cfg.models == {
        "default": "google_vertexai:gemini-3.6-flash",
        "review": "google_anthropic_vertex:claude-opus-5",
    }


def test_effort_on_a_non_claude_model_is_refused_by_name():
    # Gemini's depth is a token budget; converting a level into one would be
    # inventing an equivalence on the most expensive parameter there is.
    with pytest.raises(SystemExit) as e:
        load_common(
            {
                "models": {
                    "filter": {
                        "id": "google_vertexai:gemini-3.6-flash",
                        "effort": "low",
                    }
                }
            }
        )
    assert "not a Claude model" in str(e.value)


def test_an_effort_outside_the_ladder_is_refused():
    with pytest.raises(SystemExit) as e:
        load_common(
            {
                "models": {
                    "review": {"id": "anthropic:claude-opus-5", "effort": "extreme"}
                }
            }
        )
    assert "expected one of" in str(e.value)
    assert "xhigh" in str(e.value)


def test_a_misspelled_knob_is_refused_rather_than_ignored():
    # The whole reason every section is strict: silently ignored reads back as
    # honoured, and here that means paying for high while the file says low.
    with pytest.raises(SystemExit) as e:
        load_common(
            {"models": {"review": {"id": "anthropic:claude-opus-5", "efort": "low"}}}
        )
    assert "efort" in str(e.value)


def test_a_table_entry_without_an_id_is_refused():
    with pytest.raises(SystemExit) as e:
        load_common({"models": {"review": {"effort": "low"}}})
    assert "needs id" in str(e.value)


def test_unpriced_model_is_named_at_load(caplog):
    """A [models] entry the price table has no listing for is costed at the
    generic rate; the operator hears about it at startup, once per entry."""
    import logging

    with caplog.at_level(logging.WARNING, logger="airc_core.config"):
        load_common(
            {
                "models": {
                    "default": "google_anthropic_vertex:claude-opus-5",
                    "filter": "deepseek:deepseek-chat",
                }
            }
        )
    assert caplog.text.count("has no price listing") == 1
    assert "filter = deepseek:deepseek-chat" in caplog.text


# ── [pricing] ───────────────────────────────────────────────────────────────


@pytest.fixture
def priced():
    """A listing of our own and a clean alias table, restored afterwards --
    load_common registers aliases into process-global module state."""
    from datetime import date

    from airc_core import pricing

    fake = pricing.Price(
        model="listed-model",
        rate=pricing.Rate(input=2.0, cache_read=0.2, output=10.0),
        as_of=date(2026, 1, 1),
    )
    saved_listed, saved_aliases = dict(pricing._LISTED), dict(pricing._ALIASES)
    pricing._LISTED[fake.model] = fake
    pricing._ALIASES.clear()
    try:
        yield pricing
    finally:
        pricing._LISTED.clear()
        pricing._LISTED.update(saved_listed)
        pricing._ALIASES.clear()
        pricing._ALIASES.update(saved_aliases)


def test_pricing_alias_is_registered_before_the_unpriced_checks(priced, caplog):
    """The section exists to satisfy _warn_unpriced and refuse_unpriced, so it
    has to be parsed before they run: an alias registered afterwards is a
    warning already printed, or a refusal that already exited."""
    import logging

    raw = {
        "pricing": {"aliases": {"mybackend:internal-ckpt-7": "listed-model"}},
        "models": {"default": "mybackend:internal-ckpt-7"},
        "daily_usd_cap": 10,
    }
    with caplog.at_level(logging.WARNING, logger="airc_core.config"):
        cfg = load_common(raw)  # would SystemExit on the cap if unpriced
    assert "has no price listing" not in caplog.text
    assert cfg.pricing_aliases == {"mybackend:internal-ckpt-7": "listed-model"}
    assert priced.price_for("mybackend:internal-ckpt-7").model == "listed-model"
    # Reparse of the same file is what icompleteu does; must not conflict.
    load_common(raw)


def test_pricing_absent_registers_nothing(priced):
    cfg = load_common({"models": {"default": "mybackend:internal-ckpt-7"}})
    assert cfg.pricing_aliases == {}
    assert priced.price_for("mybackend:internal-ckpt-7").generic


def test_pricing_alias_errors_name_the_entry(priced):
    # register_alias's ValueError becomes a SystemExit naming the entry, so the
    # operator gets the line to edit rather than a traceback.
    with pytest.raises(
        SystemExit, match=r"\[pricing.aliases\] mybackend:x.*no listing"
    ):
        load_common({"pricing": {"aliases": {"mybackend:x": "nowhere-model"}}})
    with pytest.raises(SystemExit, match=r"\[pricing.aliases\] mybackend:x must name"):
        load_common({"pricing": {"aliases": {"mybackend:x": 3}}})
    with pytest.raises(SystemExit, match=r"\[pricing.aliases\] must be a table"):
        load_common({"pricing": {"aliases": "listed-model"}})
    with pytest.raises(SystemExit, match="aliasses"):
        load_common({"pricing": {"aliasses": {}}})
    # Two configs in one process disagreeing about a name is a conflict, not a
    # last-writer-wins.
    priced._LISTED["other-listed"] = priced._LISTED["listed-model"].model_copy(
        update={"model": "other-listed"}
    )
    load_common({"pricing": {"aliases": {"mybackend:x": "listed-model"}}})
    with pytest.raises(SystemExit, match="already priced as 'listed-model'"):
        load_common({"pricing": {"aliases": {"mybackend:x": "other-listed"}}})


# ── [traits] ──────────────────────────────────────────────────────────────────


@pytest.fixture
def traits_clean():
    """A clean traits-alias table, restored afterwards -- load_common registers
    aliases into process-global module state."""
    from airc_core import providers

    saved = dict(providers._TRAITS_ALIASES)
    providers._TRAITS_ALIASES.clear()
    try:
        yield providers
    finally:
        providers._TRAITS_ALIASES.clear()
        providers._TRAITS_ALIASES.update(saved)


def test_traits_alias_is_registered_and_on_the_config(traits_clean):
    raw = {
        "traits": {"aliases": {"mybackend:internal-pro": "gemini-3.1-pro-preview"}},
        "models": {"default": "mybackend:internal-pro"},
    }
    cfg = load_common(raw)
    assert cfg.traits_aliases == {"mybackend:internal-pro": "gemini-3.1-pro-preview"}
    assert traits_clean.model_traits_for("mybackend:internal-pro").slow_to_converge
    # Reparse of the same file is what icompleteu does; must not conflict.
    load_common(raw)


def test_traits_absent_registers_nothing(traits_clean):
    cfg = load_common({"models": {"default": "mybackend:internal-pro"}})
    assert cfg.traits_aliases == {}
    assert not traits_clean.model_traits_for("mybackend:internal-pro").slow_to_converge


def test_traits_alias_errors_name_the_entry(traits_clean):
    with pytest.raises(
        SystemExit, match=r"\[traits.aliases\] mybackend:x.*no ModelTraits entry"
    ):
        load_common({"traits": {"aliases": {"mybackend:x": "nowhere-model"}}})
    with pytest.raises(SystemExit, match=r"\[traits.aliases\] mybackend:x must name"):
        load_common({"traits": {"aliases": {"mybackend:x": 3}}})
    with pytest.raises(SystemExit, match=r"\[traits.aliases\] must be a table"):
        load_common({"traits": {"aliases": "gemini-3.1-pro-preview"}})
    with pytest.raises(SystemExit, match="aliasses"):
        load_common({"traits": {"aliasses": {}}})
    # Two configs in one process disagreeing about a name is a conflict.
    load_common({"traits": {"aliases": {"mybackend:x": "gemini-3.1-pro-preview"}}})
    with pytest.raises(SystemExit, match="already takes the traits of"):
        load_common({"traits": {"aliases": {"mybackend:x": "claude-opus-5-5"}}})


def test_enabling_the_explicit_vertex_cache_is_said_at_load(caplog):
    """The explicit Vertex cache path has been inactive since 2026-09 and its
    cost rule still carries a hand-tuned ratio; a config that would run it is
    named once per model at startup, and only when caching is explicit."""
    import logging

    raw = {"models": {"default": "google_vertexai:gemini-3.1-pro-preview"}}
    with caplog.at_level(logging.WARNING, logger="airc_core.config"):
        load_common(raw)
    assert caplog.text.count("explicit Vertex context cache") == 1
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="airc_core.config"):
        load_common(raw | {"caching": {"explicit": False}})
        load_common({"models": {"default": "google_anthropic_vertex:claude-opus-5"}})
    assert "explicit Vertex context cache" not in caplog.text


def test_refuse_unpriced_names_every_unlisted_model_under_a_budget():
    """A dollar bound on the placeholder rate is an arbitrary number presented as
    dollars. Every offender at once, so an operator fixing a roster is not
    told about them one restart at a time."""
    import pytest
    from airc_core.config import refuse_unpriced

    with pytest.raises(SystemExit) as e:
        refuse_unpriced(
            {
                "review": "google_anthropic_vertex:claude-opus-5",
                "verify": "someprovider:x",
                "wide": "someprovider:y",
            },
            "[processors.review] cost_limit",
        )
    msg = str(e.value)
    assert "someprovider:x" in msg and "someprovider:y" in msg
    assert "claude-opus-5" not in msg
    assert "[processors.review] cost_limit" in msg


def test_refuse_unpriced_passes_a_fully_listed_roster():
    from airc_core.config import refuse_unpriced

    refuse_unpriced(
        {
            "review": "google_vertexai:gemini-3.1-pro-preview",
            "verify": "google_anthropic_vertex:claude-opus-5@20260701",
        },
        "[processors.review] cost_limit",
    )


def test_profile_for_carries_the_knobs_the_id_lookup_dropped():
    """The bug this exists to stop: `common.models` is ids alone, so a component
    resolving there gets the checkpoint and silently drops the entry's effort --
    a config that said xhigh parsed, validated and started clean while every
    call ran at the provider default."""
    from airc_core.config import CommonConfig, ModelProfile, profile_for

    common = CommonConfig(
        models={"coding": "anthropic:claude-opus-5"},
        model_profiles={
            "coding": ModelProfile(
                key="coding", id="anthropic:claude-opus-5", effort="xhigh"
            )
        },
    )
    profile = profile_for(common, "coding", "")
    assert profile.id == "anthropic:claude-opus-5"
    assert profile.call_kwargs == {"effort": "xhigh"}


def test_profile_for_falls_back_to_default_then_refuses():
    from airc_core.config import CommonConfig, ModelProfile, profile_for

    common = CommonConfig(
        model_profiles={"default": ModelProfile(key="default", id="p:d")}
    )
    assert profile_for(common, "coding", "").id == "p:d"
    with pytest.raises(ValueError, match=r"\[models\].coding or .default"):
        profile_for(CommonConfig(), "coding", "")


def test_profile_for_accepts_an_ids_only_config():
    """Only load_common fills both tables. A CommonConfig built by hand -- a
    test, a one-shot driver -- carries ids alone and cannot have meant "no
    model" by it; it carries no knobs either, so a bare profile is exactly what
    it expressed."""
    from airc_core.config import CommonConfig, profile_for

    common = CommonConfig(models={"judge": "test:model"})
    profile = profile_for(common, "judge", "")
    assert profile.id == "test:model" and profile.call_kwargs == {}

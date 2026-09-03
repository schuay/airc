# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

import logging

import pytest
from airc_core import SUPPORTED_PROVIDERS, check_model_id, list_models, make_model
from airc_core.model import _silence_vertex_noise


def test_vertex_noise_filter_drops_benign_warnings(caplog):
    _silence_vertex_noise()
    cm = logging.getLogger("langchain_google_vertexai.chat_models")
    fu = logging.getLogger("langchain_google_vertexai.functions_utils")
    # Filters run only when the logger actually handles a record; force it.
    cm.setLevel(logging.WARNING)
    fu.setLevel(logging.WARNING)
    with caplog.at_level(logging.WARNING):
        cm.warning("Using cached content, parameter system_instruction will be ignored")
        fu.warning("Key 'additionalProperties' is not supported in schema, ignoring")
        fu.warning("a real warning worth seeing")
    assert "Using cached content" not in caplog.text
    assert "is not supported in schema" not in caplog.text
    assert "a real warning worth seeing" in caplog.text


def test_silence_vertex_noise_is_idempotent():
    cm = logging.getLogger("langchain_google_vertexai.chat_models")
    before = len(cm.filters)
    _silence_vertex_noise()
    _silence_vertex_noise()
    # The shared filter is added at most once (addFilter dedups by identity).
    assert len(cm.filters) <= before + 1


def test_retry_noise_filter_condenses_prefill_dump():
    from airc_core.model import _RetryNoiseFilter

    f = _RetryNoiseFilter()
    # A realistic tenacity before_sleep_log line with the multi-kilobyte dump.
    huge = "x" * 26000
    rec = logging.LogRecord(
        "langchain_google_vertexai._retry",
        logging.WARNING,
        __file__,
        1,
        "Retrying ... in 8 seconds as it raised ResourceExhausted: 429 "
        "PREFILL_QUEUE_OVERLOADED " + huge,
        (),
        None,
    )
    assert f.filter(rec) is True  # surfaced, not dropped
    out = rec.getMessage()
    assert out == "vertex retry in 8s (PREFILL_QUEUE_OVERLOADED)"
    assert len(out) < 60  # the dump is gone


def test_tool_first_guard_converts_a_tool_opening_history():
    # The growing cache's tail opens on the tool response(s) answering a cached
    # function call; without the guard the converter raises IndexError on that
    # shape (langchain-google issue #392).
    from airc_core.model import _install_vertex_tool_first_guard
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_google_vertexai import chat_models as cm
    from langchain_google_vertexai._image_utils import ImageBytesLoader

    _install_vertex_tool_first_guard()
    guarded = cm._parse_chat_history_gemini
    _install_vertex_tool_first_guard()
    assert cm._parse_chat_history_gemini is guarded  # no double wrap

    tail = [
        ToolMessage("int main()", tool_call_id="c1", name="look"),
        ToolMessage("42", tool_call_id="c2", name="count"),
        AIMessage("both read"),
    ]
    system, contents = cm._parse_chat_history_gemini(tail, ImageBytesLoader())
    assert system is None
    assert [c.role for c in contents] == ["user", "model"]
    # Parallel responses merge into one user content; the sentinel is stripped.
    assert [p.function_response.name for p in contents[0].parts] == ["look", "count"]
    assert all(not p.text for p in contents[0].parts)


def test_tool_first_guard_leaves_normal_histories_alone():
    from airc_core.model import _install_vertex_tool_first_guard
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_google_vertexai import chat_models as cm
    from langchain_google_vertexai._image_utils import ImageBytesLoader

    _install_vertex_tool_first_guard()
    _, contents = cm._parse_chat_history_gemini(
        [HumanMessage("hi"), AIMessage("yo")], ImageBytesLoader()
    )
    assert [c.role for c in contents] == ["user", "model"]
    assert contents[0].parts[0].text == "hi"


def test_genai_tool_first_guard_converts_a_tool_opening_history():
    # Same tail shape as the vertexai guard test; the genai converter's failure
    # mode is silent DROPPING of orphan tool responses rather than a crash.
    from airc_core.model import _install_genai_tool_first_guard
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_google_genai import chat_models as cm

    _install_genai_tool_first_guard()
    guarded = cm._parse_chat_history
    _install_genai_tool_first_guard()
    assert cm._parse_chat_history is guarded  # no double wrap

    tail = [
        ToolMessage("int main()", tool_call_id="c1", name="look"),
        ToolMessage("42", tool_call_id="c2", name="count"),
        AIMessage("both read"),
    ]
    system, contents = cm._parse_chat_history(tail, model="gemini-3.8-flash")
    assert system is None
    assert [c.role for c in contents] == ["user", "model"]
    # Parallel responses share one user content; the sentinel model turn that
    # claimed their ids is gone.
    assert [p.function_response.name for p in contents[0].parts] == ["look", "count"]
    assert all(p.function_call is None for p in contents[0].parts)


def test_genai_tool_first_guard_leaves_normal_histories_alone():
    from airc_core.model import _install_genai_tool_first_guard
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_google_genai import chat_models as cm

    _install_genai_tool_first_guard()
    _, contents = cm._parse_chat_history(
        [HumanMessage("hi"), AIMessage("yo")], model="gemini-3.8-flash"
    )
    assert [c.role for c in contents] == ["user", "model"]
    assert contents[0].parts[0].text == "hi"


def test_google_sdk_flag_routes_vertex_ids_to_genai(monkeypatch):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_google_genai import chat_models as cm

    monkeypatch.setenv("AIRC_GOOGLE_SDK", "genai")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    m = make_model("google_vertexai:gemini-3.8-flash")
    assert isinstance(m, ChatGoogleGenerativeAI)
    # Vertex backend via the project kwarg, same call-time defaults as the
    # vertexai path, and the orphan-tool guard installed at construction.
    assert (m.project, m.location) == ("test-project", "global")
    assert (m.include_thoughts, m.max_retries, m.timeout) == (False, 1, 600.0)
    assert getattr(cm._parse_chat_history, "_airc_tool_first_guard", False)

    # Without a project the class's own detection lands on the Developer API
    # (GOOGLE_API_KEY) -- the local-verification path.
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert make_model("google_vertexai:gemini-3.8-flash").project is None


def test_google_sdk_defaults_to_vertexai(monkeypatch):
    from airc_core.model import _google_sdk

    monkeypatch.delenv("AIRC_GOOGLE_SDK", raising=False)
    assert _google_sdk() == "vertexai"


def test_retry_noise_filter_passes_unrelated_records():
    from airc_core.model import _RetryNoiseFilter

    f = _RetryNoiseFilter()
    rec = logging.LogRecord(
        "langchain_google_vertexai._retry",
        logging.WARNING,
        __file__,
        1,
        "some other warning",
        (),
        None,
    )
    assert f.filter(rec) is True
    assert rec.getMessage() == "some other warning"  # untouched


def test_valid_ids_pass():
    assert check_model_id("google_vertexai:gemini-2.5-flash") is None
    for provider in SUPPORTED_PROVIDERS:
        assert check_model_id(f"{provider}:some-model") is None


def test_missing_prefix_flagged():
    msg = check_model_id("gemini-2.5-flash")
    assert msg and "provider" in msg


def test_unknown_provider_flagged():
    msg = check_model_id("google_vertex:gemini-2.5-flash")  # typo: missing 'ai'
    assert msg and "unknown provider" in msg


def test_make_model_rejects_bad_id_with_hint():
    # Raises before any network call, and the message lists valid providers.
    with pytest.raises(ValueError) as e:
        make_model("gemini-2.5-flash")
    assert "google_vertexai" in str(e.value)


def test_list_models_unknown_provider_is_none():
    # Unsupported provider returns None rather than raising (offline).
    assert list_models("nope:whatever") is None


def test_openrouter_is_supported_provider():
    assert "openrouter" in SUPPORTED_PROVIDERS
    assert check_model_id("openrouter:z-ai/glm-4.6") is None


def test_openrouter_missing_key_reports_env_var(monkeypatch):
    from airc_core.model import missing_key

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert missing_key("openrouter:z-ai/glm-4.6") == "OPENROUTER_API_KEY"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-xxx")
    assert missing_key("openrouter:z-ai/glm-4.6") is None


def test_make_model_openrouter_rewrites_to_openai_with_base_url(monkeypatch):
    # openrouter has no langchain package: make_model serves it through the openai
    # provider pointed at OpenRouter's endpoint, with OPENROUTER_API_KEY. No
    # google-cloud/ADC is touched (the no-Cloud-setup persona path).
    from airc_core import model as m

    captured: dict = {}

    def fake_init(model_id, **kw):
        captured.clear()
        captured["model_id"] = model_id
        captured.update(kw)
        return "MODEL"

    monkeypatch.setattr(m, "init_chat_model", fake_init)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-xxx")

    m.make_model("openrouter:z-ai/glm-4.6")
    # The provider prefix is rewritten to openai; the model name (which itself
    # contains a slash) is preserved verbatim after the first colon.
    assert captured["model_id"] == "openai:z-ai/glm-4.6"
    assert captured["base_url"] == m._OPENROUTER_BASE_URL
    assert captured["api_key"] == "sk-or-xxx"

    # An explicit caller base_url/api_key wins over the OpenRouter defaults.
    m.make_model("openrouter:z-ai/glm-4.6", base_url="http://proxy", api_key="k")
    assert captured["base_url"] == "http://proxy"
    assert captured["api_key"] == "k"


def test_make_model_openrouter_unset_key_does_not_leak_openai_key(monkeypatch):
    # With OPENROUTER_API_KEY unset and OPENAI_API_KEY set, make_model must NOT
    # produce a client that carries the OpenAI key. init_chat_model is left
    # UNMOCKED on purpose: the leak lives in ChatOpenAI's api_key default_factory
    # (it reads OPENAI_API_KEY when no key is passed), so a test that mocks
    # init_chat_model cannot see it. make_model refuses rather than build a
    # openrouter.ai client bound to the OpenAI key.
    from airc_core import model as m

    monkeypatch.setenv("OPENAI_API_KEY", "sk-SENTINEL-OPENAI-KEY")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        m.make_model("openrouter:z-ai/glm-4.6")

    # An explicit caller key is honored and is the only key the client carries --
    # never the OpenAI env key.
    model = m.make_model("openrouter:z-ai/glm-4.6", api_key="sk-or-explicit")
    assert model.root_client.api_key == "sk-or-explicit"
    # The client is aimed at openrouter (the SDK normalizes with a trailing slash).
    assert str(model.root_client.base_url).startswith(m._OPENROUTER_BASE_URL)

    # With OPENROUTER_API_KEY set, that key (not the OpenAI one) reaches the client.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    model = m.make_model("openrouter:z-ai/glm-4.6")
    assert model.root_client.api_key == "sk-or-env"


def test_list_models_openrouter_unset_key_does_not_leak_openai_key(monkeypatch):
    # list_models points the openai client at openrouter.ai; with OPENROUTER_API_KEY
    # unset it must not fall back to OPENAI_API_KEY (which .models.list() would then
    # send to the third-party host). "" blocks the env fallback, the client raises
    # its own missing-credential error, and list_models swallows it to None.
    from airc_core import model as m

    monkeypatch.setenv("OPENAI_API_KEY", "sk-SENTINEL-OPENAI-KEY")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    sent_keys: list = []

    class _Boom(Exception):
        pass

    class _FakeOpenAI:
        def __init__(self, **kw):
            sent_keys.append(kw.get("api_key"))
            # Mimic the real client refusing an empty credential before any call.
            if not kw.get("api_key"):
                raise _Boom("missing credentials")

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    assert m.list_models("openrouter:z-ai/glm-4.6") is None
    # The OpenAI sentinel key was never handed to the openrouter-bound client.
    assert "sk-SENTINEL-OPENAI-KEY" not in sent_keys


def test_make_model_suppresses_vertex_thoughts(monkeypatch):
    # Vertex models default to include_thoughts=False so a thinking model's
    # reasoning can never reach the room as answer text; other providers are
    # untouched, and an explicit caller value wins.
    from airc_core import model as m

    captured: dict = {}

    def fake_init(model_id, **kw):
        captured.clear()
        captured.update(kw)
        return "MODEL"

    monkeypatch.setattr(m, "init_chat_model", fake_init)

    m.make_model("google_vertexai:gemini-2.5-flash")
    assert captured["include_thoughts"] is False

    m.make_model("google_vertexai:gemini-2.5-flash", include_thoughts=True)
    assert captured["include_thoughts"] is True  # explicit caller value wins

    m.make_model("anthropic:claude-fable-5")
    assert "include_thoughts" not in captured  # non-vertex untouched


def test_make_model_shallows_vertex_sdk_retries(monkeypatch):
    # The SDK retry loop is kept shallow so it does not amplify a prefill overload
    # under airc's own retry middleware; other providers and explicit callers are
    # untouched.
    from airc_core import model as m

    captured: dict = {}

    def fake_init(model_id, **kw):
        captured.clear()
        captured.update(kw)
        return "MODEL"

    monkeypatch.setattr(m, "init_chat_model", fake_init)

    m.make_model("google_vertexai:gemini-2.5-flash")
    assert captured["max_retries"] == 1

    m.make_model("google_vertexai:gemini-2.5-flash", max_retries=6)
    assert captured["max_retries"] == 6  # explicit caller value wins

    m.make_model("anthropic:claude-fable-5")
    assert "max_retries" not in captured  # non-vertex untouched


def test_vertex_proxy_env_configures_the_client_for_the_loopback_seam(monkeypatch):
    """The sandbox proxy hook: no credential, async REST transport, our endpoint.

    Both settings are forced rather than chosen. With a custom endpoint langchain
    skips its "rest" -> "grpc_asyncio" upgrade and plain "rest" resolves to a
    SYNC transport whose methods return non-awaitables -- and airc drives models
    via astream, so that path fails at runtime, not at construction. The aio
    credential is required by type and carries no authority; the proxy supplies
    the real one.
    """
    from airc_core.model import _VERTEX_PROXY_ENV, _proxy_kwargs

    monkeypatch.setenv(_VERTEX_PROXY_ENV, "http://127.0.0.1:9999")
    kw = _proxy_kwargs("http://127.0.0.1:9999")
    assert kw["api_endpoint"] == "http://127.0.0.1:9999"
    assert kw["api_transport"] == "rest_asyncio"
    # An aio credential, not a sync one: rest_asyncio validates the TYPE.
    from google.auth.aio.credentials import Credentials as AioCredentials

    assert isinstance(kw["credentials"], AioCredentials)


def test_the_proxy_is_the_only_credential_path_into_a_box(monkeypatch):
    """A sandboxed client authenticates as nobody, whatever else is in the env.

    This once guarded a precedence rule between the proxy and a brokered token
    file; the broker is gone and the proxy is the only path left. The claim it
    protects is unchanged and is the reason the mode exists: if construction
    could pick up a real credential from the ambient environment, the box would
    authenticate itself and 'no credential in the sandbox' would quietly stop
    being true. Asserted against the credential that reaches the constructor,
    not against the absence of the deleted branch -- ADC is read lazily from
    env/metadata, so anonymity has to be positive.
    """
    from airc_core.model import _VERTEX_PROXY_ENV
    from google.auth.aio.credentials import AnonymousCredentials

    monkeypatch.setenv(_VERTEX_PROXY_ENV, "http://127.0.0.1:9999")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/adc.json")

    captured = {}

    def fake_init(model_id, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("airc_core.model.init_chat_model", fake_init)
    from airc_core.model import make_model

    make_model("google_vertexai:gemini-3.1-pro-preview")
    assert captured["api_endpoint"] == "http://127.0.0.1:9999"
    assert captured["api_transport"] == "rest_asyncio"
    assert isinstance(captured["credentials"], AnonymousCredentials)


# ── external model providers ────────────────────────────────────────────────


@pytest.fixture
def registry():
    """A clean _CUSTOM_PROVIDERS, restored afterwards.

    The registry is process-global module state (make_model is a free function),
    so a test that registers a prefix would otherwise leak it into every later
    test -- and the one that matters here asserts an UNREGISTERED prefix is
    still rejected, which a leak turns into a false pass.
    """
    from airc_core import model as m

    saved = dict(m._CUSTOM_PROVIDERS)
    m._CUSTOM_PROVIDERS.clear()
    try:
        yield m
    finally:
        m._CUSTOM_PROVIDERS.clear()
        m._CUSTOM_PROVIDERS.update(saved)


def _stub_factory(model_id, **kwargs):
    return ("STUB", model_id, kwargs)


_STUB = f"{__name__}:_stub_factory"


def test_registered_provider_passes_check_and_unregistered_still_fails(registry):
    # The pair is the point: without the second half, a check_model_id that
    # accepted everything would pass the first half just as well.
    registry.register_provider("mybackend", _STUB)
    assert check_model_id("mybackend:v1") is None
    assert check_model_id("otherbackend:v1") is not None
    assert "unknown provider" in check_model_id("otherbackend:v1")
    # The shape check still applies to a registered provider.
    assert check_model_id("mybackend") is not None


def test_make_model_uses_factory_and_never_touches_init_chat_model(
    registry, monkeypatch
):
    def boom(*a, **kw):  # pragma: no cover -- the assertion is that it is unused
        raise AssertionError("init_chat_model must not run for a custom provider")

    monkeypatch.setattr(registry, "init_chat_model", boom)
    registry.register_provider("mybackend", _STUB)

    # The factory gets the FULL id (one provider can serve several models) and
    # the caller kwargs verbatim.
    kind, model_id, kwargs = registry.make_model("mybackend:v1", temperature=0.4)
    assert (kind, model_id) == ("STUB", "mybackend:v1")
    assert kwargs == {"temperature": 0.4}


def test_make_model_unregistered_provider_raises(registry):
    # The guard that has to stay live: an id that merely LOOKS like a custom
    # provider is rejected, so a typo'd prefix fails at startup rather than
    # reaching a factory that does not exist.
    with pytest.raises(ValueError, match="unknown provider"):
        registry.make_model("mybackend:v1")


def test_custom_provider_does_not_disturb_builtin_construction(registry, monkeypatch):
    # A registered external provider must be inert for every other id: the
    # branch is an early return keyed on the prefix, so a built-in still flows
    # through init_chat_model untouched.
    captured = {}

    def fake_init(model_id, **kw):
        captured["model_id"] = model_id
        return "MODEL"

    monkeypatch.setattr(registry, "init_chat_model", fake_init)
    registry.register_provider("mybackend", _STUB)
    assert registry.make_model("anthropic:claude-fable-5") == "MODEL"
    assert captured["model_id"] == "anthropic:claude-fable-5"


def test_missing_key_reports_requires_env_only_when_declared(registry, monkeypatch):
    registry.register_provider("needy", _STUB, requires_env="MYBACKEND_TOKEN")
    registry.register_provider("keyless", _STUB)

    monkeypatch.delenv("MYBACKEND_TOKEN", raising=False)
    assert registry.missing_key("needy:v1") == "MYBACKEND_TOKEN"
    monkeypatch.setenv("MYBACKEND_TOKEN", "t")
    assert registry.missing_key("needy:v1") is None
    # A backend authenticating some other way declares no env var and is
    # correctly not checked -- it is not "missing" anything.
    assert registry.missing_key("keyless:v1") is None


def test_register_provider_rejects_builtin_and_conflicting_prefixes(registry):
    with pytest.raises(ValueError, match="built-in provider"):
        registry.register_provider("anthropic", _STUB)
    with pytest.raises(ValueError, match="colon-free"):
        registry.register_provider("bad:prefix", _STUB)

    registry.register_provider("mybackend", _STUB)
    # Idempotent for the identical spec (icompleteu parses its config several
    # times per process), but a CONFLICTING one raises rather than silently
    # deciding by parse order.
    registry.register_provider("mybackend", _STUB)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_provider("mybackend", "other.module:make")


def test_hint_lists_registered_providers_and_stays_clean_when_empty(registry):
    from airc_core import supported_models_hint

    hint = supported_models_hint()
    assert "mybackend" not in hint
    # No trailing separator when nothing is registered -- the hint is printed
    # verbatim on an --check failure.
    assert "anthropic, openrouter " not in hint
    assert hint.count(", ,") == 0

    registry.register_provider("mybackend", _STUB)
    assert "mybackend" in supported_models_hint()


def test_factory_shape_is_rejected_at_registration_not_first_use(registry):
    # Shape is checkable without importing, and a path that is not "module:attr"
    # at all can never become one -- so it fails at startup rather than inside
    # the first turn that happens to need this model.
    for bad in ("not_a_dotted_path", "mod:", ":attr"):
        with pytest.raises(ValueError, match="must be 'module:attr'"):
            registry.register_provider("mybackend", bad)
    assert "mybackend" not in registry._CUSTOM_PROVIDERS


def test_unloadable_factory_names_the_config_section(registry):
    # The import stays deferred, so this failure lands mid-turn. A bare
    # ModuleNotFoundError there names a module the operator never typed under
    # that name; the message has to tie it back to the config line.
    registry.register_provider("mybackend", "no_such_module_xyz:make")
    with pytest.raises(RuntimeError, match=r"\[model_providers.mybackend\]"):
        registry.make_model("mybackend:v1")

    registry.register_provider("attrless", f"{__name__}:nope")
    with pytest.raises(RuntimeError, match=r"\[model_providers.attrless\]"):
        registry.make_model("attrless:v1")


def test_list_models_returns_none_for_custom_provider(registry):
    # No listing hook on the spec: --list-models prints "could not list" rather
    # than failing, which is the documented degradation.
    registry.register_provider("mybackend", _STUB)
    assert list_models("mybackend:v1") is None


def test_multi_colon_model_name_reaches_factory_whole(registry):
    # A provider may serve vendor-qualified names (openrouter's ids look like
    # this). Only the FIRST colon separates the prefix; the rest is the model's.
    registry.register_provider("mybackend", _STUB)
    _, model_id, _ = registry.make_model("mybackend:vendor:model:v2")
    assert model_id == "mybackend:vendor:model:v2"


def test_custom_prefix_does_not_shadow_a_builtin_key_check(registry, monkeypatch):
    # missing_key matches built-ins by "<name>:" WITH the colon, so a custom
    # prefix that merely starts with one ("openai_custom") must not inherit its
    # env-var requirement.
    registry.register_provider("openai_custom", _STUB)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert registry.missing_key("openai_custom:v1") is None

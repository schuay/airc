# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT


import warnings

import pytest
from airc_core import SUPPORTED_PROVIDERS, check_model_id, list_models, make_model


def test_tool_first_index_matches_only_a_leading_tool_response():
    # The guards' decision, tested apart from either converter: the end-to-end
    # cases cannot pin it, because splicing a sentinel into a history that never
    # needed one happens to be a no-op in both converters.
    from airc_core.model import _tool_first_index
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    def at(history):
        return _tool_first_index(history, SystemMessage, ToolMessage)

    tool = ToolMessage("int main()", tool_call_id="c1", name="look")
    sys, human = SystemMessage("sys"), HumanMessage("hi")
    # The cache's tail, bare and behind the system turn langchain prepends.
    assert at([tool]) == 0
    assert at([sys, tool]) == 1
    assert at([sys, sys, tool]) == 2
    # Everything else is an ordinary history the converters handle themselves.
    assert at([]) is None
    assert at([sys]) is None
    assert at([human, tool]) is None
    assert at([sys, human, AIMessage("yo")]) is None


def test_genai_tool_first_guard_converts_a_tool_opening_history():
    # The growing cache's tail opens on the tool response(s) answering a
    # cached function call; without the guard the converter silently drops
    # orphan tool responses.
    from airc_core.model import _install_genai_tool_first_guard
    from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
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
    # Bare, and with the leading system turn a real request carries.
    for history in (tail, [SystemMessage("you are a reviewer"), *tail]):
        system, contents = cm._parse_chat_history(
            list(history), model="gemini-3.8-flash"
        )
        assert (system is not None) == isinstance(history[0], SystemMessage)
        assert [c.role for c in contents] == ["user", "model"]
        # Parallel responses share one user content; the sentinel model turn that
        # claimed their ids is gone.
        assert [p.function_response.name for p in contents[0].parts] == [
            "look",
            "count",
        ]
        assert all(p.function_call is None for p in contents[0].parts)


def test_genai_tool_first_guard_leaves_normal_histories_alone():
    from airc_core.model import _install_genai_tool_first_guard
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_google_genai import chat_models as cm

    _install_genai_tool_first_guard()
    turns = [HumanMessage("hi"), AIMessage("yo")]
    for history in (turns, [SystemMessage("sys"), *turns]):
        _, contents = cm._parse_chat_history(list(history), model="gemini-3.8-flash")
        assert [c.role for c in contents] == ["user", "model"]
        assert contents[0].parts[0].text == "hi"


def test_vertex_ids_are_served_by_the_genai_adapter(monkeypatch):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_google_genai import chat_models as cm

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    m = make_model("google_vertexai:gemini-3.8-flash")
    assert isinstance(m, ChatGoogleGenerativeAI)
    # Vertex backend via the project kwarg, the call-time defaults, and the
    # orphan-tool guard installed at construction.
    assert (m.project, m.location) == ("test-project", "global")
    assert (m.include_thoughts, m.max_retries, m.timeout) == (False, 1, 600.0)
    assert getattr(cm._parse_chat_history, "_airc_tool_first_guard", False)

    # Without a project the class's own detection lands on the Developer API
    # (GOOGLE_API_KEY) -- the local-verification path.
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert make_model("google_vertexai:gemini-3.8-flash").project is None


def test_proxy_placeholder_credentials_never_refresh():
    # google-genai refreshes any credential without a token, and google-auth's
    # AnonymousCredentials RAISE on refresh -- so the placeholder must present
    # as permanently valid, or the first in-box call dies before the proxy.
    from airc_core.model import proxy_placeholder_credentials

    creds = proxy_placeholder_credentials()
    assert creds.valid and not creds.expired
    creds.refresh(None)  # a no-op, not an error
    assert creds.token


def test_genai_proxy_env_routes_base_url_and_credentials(monkeypatch):
    from airc_core.model import _VERTEX_PROXY_ENV

    monkeypatch.setenv(_VERTEX_PROXY_ENV, "http://127.0.0.1:1")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    m = make_model("google_vertexai:gemini-3.8-flash")
    assert m.base_url == "http://127.0.0.1:1"
    # The placeholder also flips the class's backend detection to Vertex.
    assert m.credentials is not None and m.credentials.valid


def test_anthropic_vertex_in_a_box_gets_a_placeholder_token(monkeypatch):
    """With neither access_token nor credentials, ChatAnthropicVertex calls
    google.auth.default(), which raises in a sandbox before the proxy is
    reached. The token grants nothing; the proxy attaches the real bearer."""
    pytest.importorskip("langchain_google_vertexai")
    from airc_core.model import _PROXY_PLACEHOLDER_TOKEN, _VERTEX_PROXY_ENV

    monkeypatch.setenv(_VERTEX_PROXY_ENV, "http://127.0.0.1:1")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    m = make_model("google_anthropic_vertex:claude-opus-5")
    assert m.access_token == _PROXY_PLACEHOLDER_TOKEN


def test_anthropic_vertex_outside_a_box_keeps_normal_adc(monkeypatch):
    """On the host the client must do its own ADC lookup."""
    pytest.importorskip("langchain_google_vertexai")
    from airc_core.model import _VERTEX_PROXY_ENV

    monkeypatch.delenv(_VERTEX_PROXY_ENV, raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    m = make_model("google_anthropic_vertex:claude-opus-5")
    assert not m.access_token


def test_anthropic_vertex_gets_the_same_call_time_guards_as_gemini(monkeypatch):
    """Each replaces a client default measured wrong on the real class:
    max_output_tokens 4096 (silent truncation), max_retries 3 (three attempts
    under a middleware that already retries; 1 is no SDK retry), timeout None
    (no deadline at all), location us-central1 (no quota, and refused by the
    sandbox allowlist)."""
    pytest.importorskip("langchain_google_vertexai")
    from airc_core.model import _ANTHROPIC_MAX_OUTPUT_TOKENS, _VERTEX_CALL_TIMEOUT_S

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    m = make_model("google_anthropic_vertex:claude-opus-5", access_token="x")
    assert m.max_output_tokens == _ANTHROPIC_MAX_OUTPUT_TOKENS
    assert m.max_retries == 1
    assert m.location == "global"
    # On the constructed client, which is where the deadline is set.
    assert m.client.timeout == _VERTEX_CALL_TIMEOUT_S
    assert m.async_client.timeout == _VERTEX_CALL_TIMEOUT_S


def test_an_explicit_caller_value_still_wins_over_the_anthropic_defaults(monkeypatch):
    pytest.importorskip("langchain_google_vertexai")

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-east5")
    m = make_model(
        "google_anthropic_vertex:claude-opus-5",
        access_token="x",
        max_tokens=64,
        timeout=5,
    )
    assert m.max_output_tokens == 64
    assert m.client.timeout == 5
    # A configured location is still the configured one.
    assert m.location == "us-east5"


def test_direct_anthropic_gets_the_same_call_time_guards_as_the_vertex_route(
    monkeypatch,
):
    """ChatAnthropic's own defaults fail the same way ChatAnthropicVertex's do:
    max_tokens from the model profile (128k, past what the deadline can
    deliver), two SDK retries under a middleware that already retries, and a
    timeout the branch pins so both Claude routes share one deadline."""
    pytest.importorskip("langchain_anthropic")
    from airc_core.model import _ANTHROPIC_MAX_OUTPUT_TOKENS, _VERTEX_CALL_TIMEOUT_S
    from langchain_core.messages import HumanMessage

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_URL", raising=False)
    m = make_model("anthropic:claude-haiku-5-5")
    assert m.max_tokens == _ANTHROPIC_MAX_OUTPUT_TOKENS
    assert m.max_retries == 1
    assert m.default_request_timeout == _VERTEX_CALL_TIMEOUT_S
    assert m._client.timeout == _VERTEX_CALL_TIMEOUT_S
    assert m._async_client.timeout == _VERTEX_CALL_TIMEOUT_S
    # Adaptive thinking with the text channel suppressed, on the wire.
    body = m._get_request_payload([HumanMessage("hi")])
    assert body["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert body["max_tokens"] == _ANTHROPIC_MAX_OUTPUT_TOKENS
    assert "temperature" not in body


def test_an_explicit_caller_value_wins_over_the_direct_anthropic_defaults(
    monkeypatch,
):
    pytest.importorskip("langchain_anthropic")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    m = make_model(
        "anthropic:claude-haiku-5-5",
        max_tokens=64,
        timeout=5,
        max_retries=3,
        thinking={"type": "adaptive", "display": "summarized"},
    )
    assert m.max_tokens == 64
    assert m._client.timeout == 5
    assert m.max_retries == 3
    assert m.thinking == {"type": "adaptive", "display": "summarized"}


def test_direct_anthropic_follows_the_base_url_an_egress_relay_sets(monkeypatch):
    """What a sandbox relies on: with no endpoint wiring in make_model, the
    client picks up ANTHROPIC_BASE_URL and the placeholder key from the
    environment, so a box reaches its loopback relay and nothing else."""
    pytest.importorskip("langchain_anthropic")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "aisan-placeholder-not-a-credential")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8713")
    monkeypatch.delenv("ANTHROPIC_API_URL", raising=False)
    m = make_model("anthropic:claude-haiku-5-5")
    assert str(m._client.base_url).rstrip("/") == "http://127.0.0.1:8713"
    assert m._client.api_key == "aisan-placeholder-not-a-credential"


def _anthropic_wire_body(model, **call_kwargs) -> dict:
    """The body ChatAnthropicVertex would splat into messages.create()."""
    from langchain_core.messages import HumanMessage

    return model._format_params(messages=[HumanMessage("hi")], **call_kwargs)


def test_anthropic_vertex_drops_the_sampling_the_messages_api_removed(monkeypatch):
    """The Messages API has no temperature/top_p/top_k and never had seed, but
    ChatAnthropicVertex re-emits its own temperature field and puts seed in
    model_kwargs, and both are splatted into create(). One configured
    temperature therefore raises TypeError on every call to the provider rather
    than degrading. Review sets temperature and seed.

    Asserted against the SDK's signature instead of a hardcoded key list, so
    the test tracks the API if a parameter returns.
    """
    pytest.importorskip("langchain_google_vertexai")
    import inspect

    from anthropic.resources.messages import AsyncMessages

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    m = make_model(
        "google_anthropic_vertex:claude-opus-5",
        access_token="x",
        temperature=0.7,
        seed=3,
        top_p=0.9,
        top_k=40,
    )
    body = _anthropic_wire_body(m)
    accepted = set(inspect.signature(AsyncMessages.create).parameters)
    assert not set(body) - accepted, "would raise TypeError in the SDK"
    assert "temperature" not in body and "seed" not in body


def test_anthropic_vertex_redacts_thinking_but_keeps_the_signature(monkeypatch):
    """The Claude branch's counterpart to include_thoughts=False: we consume
    text blocks only, and a degraded turn can emit reasoning as ordinary answer
    text. display "omitted" drops the reasoning text and keeps the signature
    that multi-turn tool continuity needs, so thinking itself stays on.
    adaptive because the SDK deprecates thinking.type "enabled"; depth belongs
    to effort, whose default (high) is what sending no output_config produces.
    """
    pytest.importorskip("langchain_google_vertexai")

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    m = make_model("google_anthropic_vertex:claude-opus-5", access_token="x")
    body = _anthropic_wire_body(m)
    assert body["thinking"] == {"type": "adaptive", "display": "omitted"}
    # Depth is effort's job and its default is the one we want, so send nothing.
    assert "output_config" not in body


def test_an_explicit_thinking_choice_still_wins(monkeypatch):
    pytest.importorskip("langchain_google_vertexai")

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    m = make_model(
        "google_anthropic_vertex:claude-opus-5",
        access_token="x",
        model_kwargs={
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "xhigh"},
        },
    )
    body = _anthropic_wire_body(m)
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "xhigh"}


def test_other_providers_keep_their_sampling(monkeypatch):
    """The strip applies to Claude on Vertex only. Gemini accepts both, so
    review's ensemble still varies its passes there."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    kwargs = {"temperature": 0.7, "seed": 3}
    m = make_model("google_vertexai:gemini-3.1-pro-preview", **kwargs)
    assert m.temperature == 0.7
    # The caller's dict is not modified either.
    assert kwargs == {"temperature": 0.7, "seed": 3}


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
    # Unsupported provider returns None instead of raising (offline).
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
    # unmocked: the leak lives in ChatOpenAI's api_key default_factory
    # (it reads OPENAI_API_KEY when no key is passed), so a test that mocks
    # init_chat_model cannot see it. make_model refuses instead of building a
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

    m.make_model("openai:gpt-5")
    assert "max_retries" not in captured  # providers without a branch untouched


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

    monkeypatch.setenv(_VERTEX_PROXY_ENV, "http://127.0.0.1:9999")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/adc.json")

    captured = {}

    def fake_init(model_id, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("airc_core.model.init_chat_model", fake_init)
    from airc_core.model import make_model

    # Loopback base_url and a placeholder bearer that authenticates nothing.
    make_model("google_vertexai:gemini-3.1-pro-preview")
    assert captured["base_url"] == "http://127.0.0.1:9999"
    assert captured["credentials"].token == "sandbox-proxy-placeholder"


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
    # Both assertions are needed: without the second, a check_model_id that
    # accepted everything would pass the first half just as well.
    registry.register_provider("mybackend", _STUB)
    assert check_model_id("mybackend:v1") is None
    assert check_model_id("otherbackend:v1") is not None
    assert "unknown provider" in check_model_id("otherbackend:v1")
    # The format check still applies to a registered provider.
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
    # provider is rejected, so a typo'd prefix fails at startup instead of
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
    # times per process), but a CONFLICTING one raises instead of silently
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
    # at all can never become one -- so it fails at startup instead of inside
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


def test_effort_on_anthropic_vertex_rides_model_kwargs_to_the_api(monkeypatch):
    """ChatAnthropicVertex has no effort field: _default_params merges
    model_kwargs and _format_params splats the result into messages.create(),
    so output_config is how a level reaches the wire on this route."""
    pytest.importorskip("langchain_google_vertexai")

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    m = make_model(
        "google_anthropic_vertex:claude-opus-5", access_token="x", effort="low"
    )
    assert m.model_kwargs["output_config"] == {"effort": "low"}
    # The thinking default the branch sets must survive alongside it -- adaptive
    # thinking with the summary channel suppressed is what makes a level mean
    # anything at all.
    assert m.model_kwargs["thinking"]["type"] == "adaptive"
    # And it has to be in the dict the client actually sends.
    assert m._default_params["output_config"] == {"effort": "low"}


def test_effort_on_anthropic_vertex_keeps_a_caller_output_config(monkeypatch):
    pytest.importorskip("langchain_google_vertexai")

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    m = make_model(
        "google_anthropic_vertex:claude-opus-5",
        access_token="x",
        effort="max",
        model_kwargs={"output_config": {"format": {"type": "json_schema"}}},
    )
    assert m.model_kwargs["output_config"] == {
        "format": {"type": "json_schema"},
        "effort": "max",
    }


def test_effort_on_direct_anthropic_uses_the_first_class_field():
    """The other Claude route names it differently: langchain-anthropic has an
    `effort` field that it folds into output_config itself."""
    pytest.importorskip("langchain_anthropic")

    m = make_model("anthropic:claude-opus-5", api_key="k", effort="medium")
    assert m.effort == "medium"


def test_effort_on_gemini_genai_becomes_thinking_level(monkeypatch):
    """ChatGoogleGenerativeAI calls the parameter thinking_level; the level
    names match, so the value passes through unchanged."""
    pytest.importorskip("langchain_google_genai")

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    m = make_model("google_vertexai:gemini-4-argon", effort="medium")
    assert m.thinking_level == "medium"
    # And it has to be in the config the client actually sends.
    assert m._build_thinking_config().thinking_level == "MEDIUM"
    # The branch's include_thoughts default is kept.
    assert m.include_thoughts is False


def test_no_effort_on_gemini_genai_sends_no_thinking_level(monkeypatch):
    """Omitting the level leaves the model at its default, as on the Anthropic
    path."""
    pytest.importorskip("langchain_google_genai")

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    m = make_model("google_vertexai:gemini-4-argon")
    assert m.thinking_level is None


def test_an_effort_gemini_does_not_accept_raises():
    """Gemini accepts low/medium/high. The API also accepts EXTRA_HIGH and MAX
    without serving them, so xhigh is refused rather than sent."""
    with pytest.raises(ValueError) as e:
        make_model("google_vertexai:gemini-4-argon", effort="xhigh")
    assert "does not accept effort 'xhigh'" in str(e.value)
    assert "low, medium, high" in str(e.value)


def test_effort_on_a_provider_without_levels_raises():
    with pytest.raises(ValueError) as e:
        make_model("openai:gpt-5", api_key="k", effort="low")
    assert "no effort level" in str(e.value)


def test_an_effort_outside_the_ladder_raises():
    with pytest.raises(ValueError) as e:
        make_model("anthropic:claude-opus-5", effort="enormous")
    assert "unknown effort" in str(e.value)


def test_no_effort_sends_no_output_config(monkeypatch):
    """Omitting the level must leave the request untouched, not pin it to the
    default: `high` is the API's own default, and sending it explicitly would
    make every existing deployment's requests differ byte-for-byte from what
    they send today -- a cache miss on the whole prefix."""
    pytest.importorskip("langchain_google_vertexai")

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    m = make_model("google_anthropic_vertex:claude-opus-5", access_token="x")
    assert "output_config" not in m.model_kwargs


def test_direct_anthropic_drops_forced_tool_choice_while_thinking():
    """The direct API refuses forced tool use with thinking on, for every
    checkpoint. make_model enables adaptive thinking by default, so the guard
    strips the choice before langchain-anthropic would, which also removes the
    per-call UserWarning that adapter emits. With thinking off the trait alone
    decides, and Haiku keeps a forced choice."""
    pytest.importorskip("langchain_anthropic")

    tool = {
        "name": "ReviewResult",
        "description": "report",
        "input_schema": {"type": "object", "properties": {}},
    }
    m = make_model("anthropic:claude-haiku-5-5", api_key="x")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for forced in ("any", "ReviewResult", {"type": "any"}):
            assert "tool_choice" not in m.bind_tools([tool], tool_choice=forced).kwargs
        assert m.bind_tools([tool], tool_choice="auto").kwargs["tool_choice"] == {
            "type": "auto"
        }

    plain = make_model("anthropic:claude-haiku-5-5", api_key="x", thinking=None)
    assert plain.bind_tools([tool], tool_choice="any").kwargs["tool_choice"] == {
        "type": "any"
    }


def test_models_without_forced_tool_choice_drop_any_and_named_tool(monkeypatch):
    """ToolStrategy binds tool_choice='any' and salvage binds the schema name.
    claude-opus-5-5 rejects both with 400 ('tool_choice: type "tool" and "any"
    are not supported for this model'); bind_tools drops them to None while
    keeping 'auto', and leaves checkpoints that accept them untouched."""
    pytest.importorskip("langchain_google_vertexai")

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    tool = {
        "name": "ReviewResult",
        "description": "report",
        "input_schema": {"type": "object", "properties": {}},
    }

    m55 = make_model("google_anthropic_vertex:claude-opus-5-5", access_token="x")
    for forced in (
        "any",
        "ReviewResult",
        {"type": "any"},
        {"type": "tool", "name": "ReviewResult"},
    ):
        bound = m55.bind_tools([tool], tool_choice=forced)
        assert "tool_choice" not in bound.kwargs

    # Non-forced modes and parallel_tool_calls flags survive.
    auto_bound = m55.bind_tools([tool], tool_choice="auto")
    assert auto_bound.kwargs["tool_choice"] == {"type": "auto"}
    no_parallel = m55.bind_tools(
        [tool], tool_choice={"type": "any", "disable_parallel_tool_use": True}
    )
    assert no_parallel.kwargs["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }

    # claude-opus-5 still accepts forced tool_choice and keeps it.
    m5 = make_model("google_anthropic_vertex:claude-opus-5", access_token="x")
    assert m5.bind_tools([tool], tool_choice="any").kwargs["tool_choice"] == {
        "type": "any"
    }
    assert m5.bind_tools([tool], tool_choice="ReviewResult").kwargs["tool_choice"] == {
        "type": "tool",
        "name": "ReviewResult",
    }

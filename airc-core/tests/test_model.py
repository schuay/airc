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


def test_vertex_file_credentials_reads_and_refreshes(tmp_path):
    # The in-box Vertex credential reads a reclient token file and re-reads it on
    # refresh (the host rewrites it), never touching the network.
    import datetime
    import json

    from airc_core.model import _vertex_file_credentials

    f = tmp_path / "token.json"
    f.write_text(
        json.dumps({"token": "tok-1", "expiry": "Fri Jul 10 12:07:06 UTC 2026"})
    )
    cred = _vertex_file_credentials(str(f))
    assert cred.token == "tok-1"
    # Go UnixDate parsed to a NAIVE datetime (what google.auth compares against).
    assert cred.expiry == datetime.datetime(2026, 7, 10, 12, 7, 6)
    assert cred.expiry.tzinfo is None

    # A host refresh: refresh() picks up the new token without a new object.
    f.write_text(
        json.dumps({"token": "tok-2", "expiry": "Sat Jul 11 00:00:00 UTC 2026"})
    )
    cred.refresh(None)
    assert cred.token == "tok-2"


def test_vertex_file_credentials_parses_iso_expiry(tmp_path):
    # The vertextoken broker writes ISO 8601 with explicit UTC; parsed to the
    # same naive-UTC shape google.auth compares against.
    import datetime
    import json

    from airc_core.model import _vertex_file_credentials

    f = tmp_path / "vertex.json"
    f.write_text(json.dumps({"token": "vt", "expiry": "2026-07-11T12:00:00+00:00"}))
    cred = _vertex_file_credentials(str(f))
    assert cred.token == "vt"
    assert cred.expiry == datetime.datetime(2026, 7, 11, 12, 0, 0)
    assert cred.expiry.tzinfo is None
    # A non-UTC offset converts, not truncates.
    f.write_text(json.dumps({"token": "vt", "expiry": "2026-07-11T14:00:00+02:00"}))
    assert _vertex_file_credentials(str(f)).expiry == datetime.datetime(
        2026, 7, 11, 12, 0, 0
    )


def test_vertex_file_credentials_tolerates_bad_expiry(tmp_path):
    # A space-padded single-digit day (Go's `_2`) still parses; a garbage expiry
    # falls back to "expire soon" so google.auth re-reads rather than trusting it.
    import datetime
    import json

    from airc_core.model import _vertex_file_credentials

    f = tmp_path / "t.json"
    f.write_text(json.dumps({"token": "x", "expiry": "Fri Jul  9 12:07:06 UTC 2026"}))
    assert _vertex_file_credentials(str(f)).expiry == datetime.datetime(
        2026, 7, 9, 12, 7, 6
    )

    f.write_text(json.dumps({"token": "x", "expiry": "not-a-date"}))
    soon = _vertex_file_credentials(str(f)).expiry
    now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    assert now <= soon <= now + datetime.timedelta(minutes=2)


def test_make_model_injects_vertex_credentials_only_when_file_present(
    monkeypatch, tmp_path
):
    # make_model injects the file-token credential when the env points at an
    # existing token, and falls back to ADC (no credentials kwarg) when the file
    # is absent -- so a not-yet-minted token never crashes construction.
    import json

    from airc_core import model as m

    captured: dict = {}

    def fake_init(model_id, **kw):
        captured.clear()
        captured.update(kw)
        return "MODEL"

    monkeypatch.setattr(m, "init_chat_model", fake_init)

    tok = tmp_path / "token.json"
    tok.write_text(json.dumps({"token": "t", "expiry": "Fri Jul 10 12:07:06 UTC 2026"}))
    monkeypatch.setenv(m._VERTEX_TOKEN_ENV, str(tok))
    m.make_model("google_vertexai:gemini-2.5-flash")
    assert "credentials" in captured

    monkeypatch.setenv(m._VERTEX_TOKEN_ENV, str(tmp_path / "gone.json"))
    m.make_model("google_vertexai:gemini-2.5-flash")
    assert "credentials" not in captured

    # Non-vertex providers never get the credential, even with the env set.
    monkeypatch.setenv(m._VERTEX_TOKEN_ENV, str(tok))
    m.make_model("anthropic:claude-fable-5")
    assert "credentials" not in captured


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

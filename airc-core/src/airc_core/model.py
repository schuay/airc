# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Chat model construction.

Models are referenced by init_chat_model id strings ("provider:model"), so
any installed langchain provider package works. Two auth styles:

  - API key providers (google_genai, anthropic, openai, deepseek, ...):
    the provider package reads its env var; missing keys disable the persona
    at startup rather than failing mid-conversation.
  - google_vertexai: no key; uses Application Default Credentials. Project
    and location come from GOOGLE_CLOUD_* env vars, seeded from [gcp] config
    (see config.apply_gcp_env_defaults).

openrouter is an OpenAI-compatible aggregator with no langchain package of its
own, so it is served through the openai provider with OpenRouter's base_url and
OPENROUTER_API_KEY (same shape as deepseek). A persona with an "openrouter:..."
model_id constructs with no google-cloud/ADC dependency, so a deploy can run its
personas without any Google Cloud setup (e.g. GLM via OpenRouter).
"""

from __future__ import annotations

import logging
import os
import re

from langchain.chat_models import init_chat_model

# Env var naming the reclient token file the in-box Vertex client authenticates
# from, in place of GCE metadata ADC (which the sandbox netfilter blocks). Set by
# the sandbox profile in token+vertex mode; unset everywhere else, so the daemon
# and any non-sandboxed caller keep normal ADC. See _vertex_file_credentials.
_VERTEX_TOKEN_ENV = "AIRC_VERTEX_TOKEN_FILE"

# OpenRouter: an OpenAI-compatible endpoint. Served through the openai provider
# (no langchain-openrouter package exists), so make_model rewrites an
# "openrouter:<model>" id to the openai provider pointed at this base_url with
# OPENROUTER_API_KEY. Same pattern deepseek uses for its own base_url.
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Per-attempt gRPC deadline (seconds) on a Vertex generate_content call. The only
# job of a wall-clock at this layer is reaping a stream that never returns -- the
# review-level wall-clock is off by default precisely because it drained on
# backoff sleep, not hangs. Legit calls finish in single-digit minutes and never
# touch it; a deadline error is a transient like any 5xx and flows into the
# existing retry/redelivery path. With max_retries=1 this is one deadline per
# logical call.
_VERTEX_CALL_TIMEOUT_S = 600

# Providers airc ships with (their langchain packages are dependencies). The
# model name after the colon is provider-side and can't be validated offline.
SUPPORTED_PROVIDERS = (
    "google_vertexai",
    "google_genai",
    "anthropic",
    "openai",
    "deepseek",
    "openrouter",
)

_PROVIDER_KEYS = {
    "openai:": "OPENAI_API_KEY",
    "anthropic:": "ANTHROPIC_API_KEY",
    "google_genai:": "GOOGLE_API_KEY",
    "deepseek:": "DEEPSEEK_API_KEY",
    "openrouter:": "OPENROUTER_API_KEY",
}


def missing_key(model_id: str) -> str | None:
    """Return the name of a required-but-unset env var, or None if usable."""
    for prefix, key in _PROVIDER_KEYS.items():
        if model_id.startswith(prefix) and not os.environ.get(key):
            return key
    return None


def check_model_id(model_id: str) -> str | None:
    """Return a problem description for a model id, or None if it looks valid.

    Only the "<provider>:<model>" shape and the provider are checked; the model
    name is validated by the provider at call time.
    """
    provider, sep, name = model_id.partition(":")
    if not sep or not name:
        return f"{model_id!r} is not in '<provider>:<model>' form"
    if provider not in SUPPORTED_PROVIDERS:
        return f"{model_id!r} has unknown provider {provider!r}"
    return None


def supported_models_hint() -> str:
    """One-line reminder of the valid provider prefixes and id format."""
    return (
        "model ids are '<provider>:<model>'; supported providers: "
        + ", ".join(SUPPORTED_PROVIDERS)
        + " (e.g. google_vertexai:gemini-2.5-flash)"
    )


def list_models(model_id: str) -> list[str] | None:
    """Live list of model names available for model_id's provider, or None.

    Calls the provider's models.list API; returns the short names (the part
    after the colon) of generate-capable models, sorted. Returns None when the
    provider can't be enumerated here (unsupported, missing credentials, or the
    API call failed) so callers can fall back gracefully.
    """
    provider = model_id.split(":", 1)[0]
    try:
        if provider == "google_vertexai":
            return _google_models(vertex=True)
        if provider == "google_genai":
            return _google_models(vertex=False)
        if provider == "anthropic":
            from anthropic import Anthropic

            return sorted(m.id for m in Anthropic().models.list().data)
        if provider in ("openai", "deepseek", "openrouter"):
            from openai import OpenAI

            kw = {}
            # For the OpenAI-compatible providers we point the openai client at a
            # third-party host. os.environ.get(...) returns None when the provider
            # key is unset, and OpenAI(api_key=None) falls back to OPENAI_API_KEY
            # from the environment -- which would then be transmitted to
            # api.deepseek.com / openrouter.ai by models.list(). Default to "" so
            # the client raises its own missing-credential error instead of
            # leaking the OpenAI key cross-provider.
            if provider == "deepseek":
                kw = {
                    "api_key": os.environ.get("DEEPSEEK_API_KEY") or "",
                    "base_url": "https://api.deepseek.com",
                }
            elif provider == "openrouter":
                kw = {
                    "api_key": os.environ.get("OPENROUTER_API_KEY") or "",
                    "base_url": _OPENROUTER_BASE_URL,
                }
            return sorted(m.id for m in OpenAI(**kw).models.list().data)
    except Exception:
        return None
    return None


def _google_models(vertex: bool) -> list[str]:
    from google import genai

    if vertex:
        client = genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION") or "global",
        )
        # query_base lists the publisher (Gemini) catalog, not tuned models.
        pager = client.models.list(config={"query_base": True})
    else:
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        pager = client.models.list()
    names = []
    for m in pager:
        if "generateContent" not in (m.supported_actions or []):
            continue
        if name := (m.name or "").rsplit("/", 1)[-1]:
            names.append(name)
    return sorted(set(names))


def _provider_kwargs(model_id: str) -> dict:
    if model_id.startswith("google_vertexai:"):
        kw: dict = {}
        if proj := os.environ.get("GOOGLE_CLOUD_PROJECT"):
            kw["project"] = proj
        if loc := os.environ.get("GOOGLE_CLOUD_LOCATION"):
            kw["location"] = loc
        return kw
    return {}


def usage_counts(usage) -> tuple[int, int, int]:
    """(input, output, cached_input) from a langchain usage_metadata dict.

    cached_input is the prompt-cache-served subset of input (cache_read), 0 when
    the provider does not report it. Mirrors the aggregation the streaming runner
    does over UsageMetadataCallbackHandler, for the direct-ainvoke call sites.
    """
    usage = usage or {}
    details = usage.get("input_token_details") or {}
    return (
        int(usage.get("input_tokens", 0)),
        int(usage.get("output_tokens", 0)),
        int(details.get("cache_read", 0)),
    )


class _VertexNoiseFilter(logging.Filter):
    """Drop ChatVertexAI's benign, constant per-call log warnings.

    Neither is API misuse, both fire on every call, and both would otherwise
    flood the journal:
    - "Using cached content, parameter ... will be ignored" -- expected with a
      context cache attached (system/tools live in the cache; re-sending them is
      the point).
    - "Key '...' is not supported in schema, ignoring" -- Vertex drops schema
      keys it does not accept (e.g. additionalProperties from a pydantic
      extra=forbid model used for structured output); harmless.
    """

    _NOISE = ("Using cached content", "is not supported in schema")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(s in msg for s in self._NOISE)


class _RetryNoiseFilter(logging.Filter):
    """Condense langchain_google_vertexai._retry's before-sleep WARNING.

    Its tenacity before_sleep_log dumps the whole exception -- a multi-kilobyte
    RPC error with nested original error, stack, and source-location trace -- on
    every retry, so a prefill-overload storm fills the journal with repeated 26k
    blocks. The retry itself is worth surfacing (an ongoing overload), the dump
    is not: rewrite the record to the status code and the retry delay, keeping
    one readable line. Mutates and passes the record rather than dropping it."""

    # "... Retrying ... in N seconds as it raised <Error>: <huge text>".
    _RETRY_RE = re.compile(r"in ([\d.]+) seconds as it raised (\w+)")
    _STATUS_RE = re.compile(
        r"RESOURCE_EXHAUSTED|PREFILL_QUEUE_OVERLOADED|PREFILL_QUEUE_PREEMPTED"
        r"|UNAVAILABLE|DEADLINE_EXCEEDED|INTERNAL"
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "as it raised" not in msg:
            return True
        m = self._RETRY_RE.search(msg)
        status = self._STATUS_RE.search(msg)
        if m:
            reason = status.group(0) if status else m.group(2)
            record.msg = "vertex retry in %ss (%s)" % (m.group(1), reason)
            record.args = ()
        return True


# Shared instances: addFilter dedups by identity, so installing them from every
# make_model call is idempotent -- each lands on its logger once.
_VERTEX_NOISE = _VertexNoiseFilter()
_RETRY_NOISE = _RetryNoiseFilter()

# The two langchain_google_vertexai loggers that emit the benign per-call noise.
_VERTEX_NOISY_LOGGERS = (
    "langchain_google_vertexai.chat_models",
    "langchain_google_vertexai.functions_utils",
)


def _silence_vertex_noise() -> None:
    for name in _VERTEX_NOISY_LOGGERS:
        logging.getLogger(name).addFilter(_VERTEX_NOISE)
    logging.getLogger("langchain_google_vertexai._retry").addFilter(_RETRY_NOISE)


def _vertex_file_credentials(path: str):
    """google.auth Credentials that read a host-refreshed token file.

    In the sandbox the box holds no durable credential and (once the metadata
    netfilter lands) cannot reach the metadata-server SA, so Vertex ADC has no
    source. A host-side broker instead refreshes a {"token", "expiry"} file at
    `path` (icompleteu's vertextoken broker mints a short-lived identity); this
    credential reads the current token and, when google.auth finds it stale,
    `refresh()` re-reads the file rather than contacting the network.
    google.auth is imported lazily so non-Vertex callers never pull it in.
    """
    import datetime
    import json
    from pathlib import Path

    from google.auth.credentials import Credentials

    def _parse_expiry(s: str | None) -> datetime.datetime:
        # Two producer formats: the vertextoken broker writes ISO 8601, and
        # luci-auth's reclient JSON emits Go's UnixDate ("Mon Jan _2 15:04:05
        # MST 2006", space-padded day). google.auth compares expiry as a NAIVE
        # UTC datetime, so both parse to that; on any format surprise expire in
        # a minute so google.auth re-reads soon rather than trusting a token
        # forever or treating it as always-expired.
        now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        try:
            t = datetime.datetime.fromisoformat(s)
            if t.tzinfo is not None:
                t = t.astimezone(datetime.UTC).replace(tzinfo=None)
            return t
        except (ValueError, TypeError):
            pass
        try:
            return datetime.datetime.strptime(
                " ".join(s.split()), "%a %b %d %H:%M:%S UTC %Y"
            )
        except (ValueError, TypeError, AttributeError):
            return now + datetime.timedelta(minutes=1)

    class _FileToken(Credentials):
        def __init__(self, p: str) -> None:
            super().__init__()
            self._p = Path(p)
            self._load()

        def _load(self) -> None:
            data = json.loads(self._p.read_text())
            self.token = data.get("token")
            self.expiry = _parse_expiry(data.get("expiry"))

        def refresh(self, request) -> None:  # noqa: ARG002 -- no network; re-read
            self._load()

    return _FileToken(path)


def make_model(model_id: str, **kwargs):
    if problem := check_model_id(model_id):
        raise ValueError(f"{problem}; {supported_models_hint()}")
    if model_id.startswith("openrouter:"):
        # No langchain-openrouter package: serve the model through the openai
        # provider pointed at OpenRouter's OpenAI-compatible endpoint. An explicit
        # caller base_url/api_key wins (so a proxy or a pre-read key still works).
        name = model_id.split(":", 1)[1]
        kwargs.setdefault("base_url", _OPENROUTER_BASE_URL)
        # We serve openrouter through langchain's ChatOpenAI, whose api_key field
        # has a default_factory that reads OPENAI_API_KEY from the environment when
        # no key is passed. So merely omitting api_key does NOT fail cleanly -- it
        # silently binds the OpenAI key to a client aimed at openrouter.ai, sending
        # that credential to a third-party host on the first call. The key must be
        # set explicitly: an explicit caller key wins (a proxy or pre-read key),
        # else OPENROUTER_API_KEY, else refuse -- never fall through to another
        # provider's key. missing_key() disables keyless personas earlier at the
        # gated call sites; this raise guards the ungated ones.
        if "api_key" not in kwargs:
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise ValueError(
                    "openrouter model needs OPENROUTER_API_KEY (or an explicit"
                    " api_key); refusing to fall back to OPENAI_API_KEY, which"
                    " would leak the OpenAI key to openrouter.ai"
                )
            kwargs["api_key"] = key
        return init_chat_model(f"openai:{name}", **kwargs)
    if model_id.startswith("google_vertexai:"):
        # Construction is the universal Vertex chokepoint (every agent/review
        # graph builds its model here), so silence the benign call-time warnings
        # once, for every component, cached or not.
        _silence_vertex_noise()
        # In the sandbox, authenticate from the bound reclient token instead of
        # metadata ADC. Gated on the env var being set (only the sandbox sets it)
        # and the file existing, so a missing/not-yet-minted token falls back to
        # ADC rather than crashing model construction. An explicit caller-passed
        # credentials wins.
        tok = os.environ.get(_VERTEX_TOKEN_ENV)
        if tok and "credentials" not in kwargs and os.path.exists(tok):
            kwargs = {**kwargs, "credentials": _vertex_file_credentials(tok)}
        # Never surface the model's thought summaries. We consume only text
        # blocks, so thoughts are pure liability: on a degraded turn (prefill
        # overload, thrashed context) a thinking model can emit its reasoning as
        # ordinary answer text, which then posts verbatim. This suppresses the
        # summary channel only; thought signatures (which Gemini 3 needs for
        # multi-turn tool continuity) are independent and still flow. It does not
        # disable thinking. An explicit caller value wins.
        kwargs.setdefault("include_thoughts", False)
        # Keep the SDK's own retry loop shallow: under a prefill overload its
        # default (6) retries each call hard and fast, and stacked under airc's
        # ModelRetryMiddleware (which re-attempts the whole call) one logical
        # call can fire dozens of prefills -- the server rejects with
        # TOO_MANY_RETRIES_PER_REQUEST and the overload feeds itself. Let the
        # middleware be the retry authority: it backs off far wider (5/15/45/60s,
        # sized to outlast a per-minute quota window). One shallow SDK retry
        # still absorbs a lone blip without a middleware round-trip.
        kwargs.setdefault("max_retries", 1)
        # A per-call deadline reaps a hung stream (see _VERTEX_CALL_TIMEOUT_S);
        # an explicit caller value wins.
        kwargs.setdefault("timeout", _VERTEX_CALL_TIMEOUT_S)
    return init_chat_model(model_id, **(_provider_kwargs(model_id) | kwargs))

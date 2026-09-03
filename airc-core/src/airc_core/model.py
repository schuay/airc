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

A third style, for a backend that is not an init_chat_model provider at all: an
external BaseChatModel subclass, named in config under [model_providers] and
built by a dotted-path factory (register_provider). make_model is the suite's
only chat-model constructor, so that one branch serves the room, the review
graph and the harness alike.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from langchain.chat_models import init_chat_model

# Env var naming a loopback endpoint that fronts Vertex for a sandboxed caller.
# The box holds NO credential at all; a host-side proxy attaches the real one.
# Set only by the sandbox profile, so every other caller is untouched.
#
# This has to live here rather than at the call sites: make_model's callers pass
# no kwargs, and the endpoint must reach the ChatVertexAI constructor.
_VERTEX_PROXY_ENV = "AISAN_VERTEX_PROXY_ENDPOINT"


def _proxy_kwargs(endpoint: str) -> dict:
    """Client settings for talking to the sandbox's Vertex proxy.

    Two of these are not free choices:

    - `rest_asyncio`, not the default. With a custom endpoint langchain skips its
      "rest" -> "grpc_asyncio" upgrade (_client_utils.py checks the HOSTNAME), and
      plain "rest" resolves to a SYNCHRONOUS transport class whose methods return
      non-awaitables. airc drives models via astream, so that path fails with
      "object ResponseIterator can't be used in 'await' expression".
    - an aio credential. rest_asyncio validates the credential TYPE, so a
      google.auth.credentials instance is rejected outright. It carries no
      authority: the proxy replaces the header. The box holding a credential that
      authenticates nothing is the entire point.
    """
    from google.auth.aio.credentials import AnonymousCredentials

    return {
        "api_endpoint": endpoint,
        "api_transport": "rest_asyncio",
        "credentials": AnonymousCredentials(),
    }


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


@dataclass(frozen=True)
class _ProviderSpec:
    factory: str  # "module:attr", imported on first use
    requires_env: str | None = None  # unset var disables the persona at startup


# External model providers, registered from config ([model_providers], parsed by
# load_common). A custom backend is a BaseChatModel subclass wrapping an in-house
# mechanism -- it has no init_chat_model id, so make_model needs a branch that
# returns a caller-built object rather than a provider string.
#
# Module state rather than something hung off a config object because make_model
# is a free function with no cfg in scope, and every one of its call sites
# reaches it that way. The registry has to live HERE and not behind the room's
# plugin contract: airc-processors and the coding subscribers import make_model
# without ever loading the room plugin, so a provider registered there would
# serve chat and leave commit review on the built-in providers.
_CUSTOM_PROVIDERS: dict[str, _ProviderSpec] = {}


def register_provider(
    prefix: str, factory: str, *, requires_env: str | None = None
) -> None:
    """Register an external model provider under `prefix` ("mybackend:...").

    `factory` is a "module:attr" path to a callable taking (model_id, **kwargs)
    and returning a BaseChatModel; it is imported on FIRST USE, not here. Every
    component calls load_common, but only the ones that build this model need
    its package importable -- a watcher polling gerrit should not die because a
    chat-only backend is missing.

    Re-registering the same spec is a no-op: load_common runs once per process
    in most components but several times in icompleteu, always over the same
    file. A CONFLICTING re-registration raises instead of quietly winning, since
    which model got built would then depend on parse order.
    """
    if ":" in prefix or not prefix:
        # split(":", 1)[0] is how a provider is recovered from a model id, so a
        # prefix holding a colon could never match one -- dead config, not a
        # subtle bug to leave for later.
        raise ValueError(f"provider prefix {prefix!r} must be non-empty and colon-free")
    if prefix in SUPPORTED_PROVIDERS:
        raise ValueError(f"{prefix!r} is a built-in provider; pick another prefix")
    # Shape now, import later: a factory that is not "module:attr" at all cannot
    # become one, so there is nothing to gain by discovering it mid-turn. This is
    # as far as startup validation can go without importing the package, which
    # would make a chat-only backend a hard dependency of every component.
    split_factory(factory)
    spec = _ProviderSpec(factory, requires_env)
    if (prior := _CUSTOM_PROVIDERS.get(prefix)) and prior != spec:
        raise ValueError(
            f"provider {prefix!r} is already registered as {prior.factory!r}"
        )
    _CUSTOM_PROVIDERS[prefix] = spec


def _custom_spec(model_id: str) -> _ProviderSpec | None:
    return _CUSTOM_PROVIDERS.get(model_id.split(":", 1)[0])


def split_factory(factory: str) -> tuple[str, str]:
    """("module", "attr") from a "module:attr" path, or raise.

    Shape-only, so config parsing can reject a malformed path at STARTUP without
    importing the package -- the import itself stays deferred to first use.
    """
    module, sep, attr = factory.partition(":")
    if not (sep and module and attr):
        raise ValueError(f"factory {factory!r} must be 'module:attr'")
    return module, attr


def _custom_factory(prefix: str, spec: _ProviderSpec):
    module, attr = split_factory(spec.factory)
    from importlib import import_module

    try:
        return getattr(import_module(module), attr)
    except (ImportError, AttributeError) as e:
        # The failure this re-raises is the deferred one: the path passed startup
        # and only now turns out to be unimportable, which surfaces INSIDE a turn
        # or a bus-driven review. A bare ModuleNotFoundError there names a module
        # the operator never typed under that name, with nothing tying it to the
        # config line that chose it -- so name the section and the model.
        raise RuntimeError(
            f"[model_providers.{prefix}] factory {spec.factory!r} could not be"
            f" loaded: {e}"
        ) from e


def missing_key(model_id: str) -> str | None:
    """Return the name of a required-but-unset env var, or None if usable."""
    for prefix, key in _PROVIDER_KEYS.items():
        if model_id.startswith(prefix) and not os.environ.get(key):
            return key
    # Without this an external provider is never disabled at startup (the loop
    # above only knows built-in prefixes), so a persona with no credential fails
    # mid-conversation instead of being skipped. Only as good as requires_env:
    # a backend authenticating by token file or socket declares none and is
    # correctly not checked here.
    spec = _custom_spec(model_id)
    if spec and spec.requires_env and not os.environ.get(spec.requires_env):
        return spec.requires_env
    return None


def check_model_id(model_id: str) -> str | None:
    """Return a problem description for a model id, or None if it looks valid.

    Only the "<provider>:<model>" shape and the provider are checked; the model
    name is validated by the provider at call time.
    """
    provider, sep, name = model_id.partition(":")
    if not sep or not name:
        return f"{model_id!r} is not in '<provider>:<model>' form"
    if provider not in SUPPORTED_PROVIDERS and provider not in _CUSTOM_PROVIDERS:
        return f"{model_id!r} has unknown provider {provider!r}"
    return None


def supported_models_hint() -> str:
    """One-line reminder of the valid provider prefixes and id format.

    Registered external providers are listed alongside the built-ins so an
    `airc --check` failure names a configured prefix rather than implying it is
    invalid -- the hint is printed by the same code that rejected the id.
    """
    return (
        "model ids are '<provider>:<model>'; supported providers: "
        + ", ".join((*SUPPORTED_PROVIDERS, *sorted(_CUSTOM_PROVIDERS)))
        + " (e.g. google_vertexai:gemini-2.5-flash)"
    )


def list_models(model_id: str) -> list[str] | None:
    """Live list of model names available for model_id's provider, or None.

    Calls the provider's models.list API; returns the short names (the part
    after the colon) of generate-capable models, sorted. Returns None when the
    provider can't be enumerated here (unsupported, an external provider whose
    spec carries no listing hook, missing credentials, or the API call failed)
    so callers can fall back gracefully.
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
            record.msg = f"vertex retry in {m.group(1)}s ({reason})"
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


def _install_vertex_tool_first_guard() -> None:
    """Make a request that opens on a tool response convertible.

    The growing prefix cache cuts the cached prefix after the model's function
    call (gemini-3.8-flash rejects a prefix ending on the function response),
    so the uncached tail begins with a ToolMessage. langchain's history
    converter crashes on exactly that shape -- an unguarded vertex_messages[-1]
    in its ToolMessage branch -- although the API accepts it. Known upstream
    since 2024 with the one-line fix spelled out, never landed:
    https://github.com/langchain-ai/langchain-google/issues/392

    Prepending a sentinel user turn routes the tool response through the
    converter's existing merge-into-previous-user-content path; stripping the
    sentinel part afterwards leaves exactly what the upstream fix would
    produce.

    TODO: retire by migrating off langchain-google-vertexai (ChatVertexAI is
    deprecated upstream) to google-genai -- after verifying its converter
    against this shape: today it silently DROPS a ToolMessage whose calling
    AIMessage is absent from the history.
    """
    from langchain_core.messages import HumanMessage, ToolMessage
    from langchain_google_vertexai import chat_models as cm

    orig = cm._parse_chat_history_gemini
    if getattr(orig, "_airc_tool_first_guard", False):
        return

    def guarded(history, *args, **kwargs):
        if not (history and isinstance(history[0], ToolMessage)):
            return orig(history, *args, **kwargs)
        system, contents = orig([HumanMessage("-"), *history], *args, **kwargs)
        first = contents[0]
        rebuilt = cm.Content(role=first.role, parts=list(first.parts)[1:])
        return system, [rebuilt, *contents[1:]]

    guarded._airc_tool_first_guard = True
    cm._parse_chat_history_gemini = guarded


def make_model(model_id: str, **kwargs):
    if problem := check_model_id(model_id):
        raise ValueError(f"{problem}; {supported_models_hint()}")
    # A registered external provider owns construction entirely: it gets the full
    # id (so one provider can serve several models) and the caller kwargs, and
    # returns a BaseChatModel. Nothing below is reachable for it -- including the
    # Vertex proxy endpoint, which is provider-specific, so a custom backend that
    # must work from inside the sandbox needs its own egress arrangement.
    if spec := _custom_spec(model_id):
        return _custom_factory(model_id.split(":", 1)[0], spec)(model_id, **kwargs)
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
        # and install the tool-first converter guard once, for every component.
        _silence_vertex_noise()
        _install_vertex_tool_first_guard()
        # In the sandbox the box holds no credential at all: point the client at
        # the loopback proxy, which attaches the real one host-side. Gated on the
        # env var (only the sandbox profile sets it), so every other caller keeps
        # normal ADC. An explicit caller-passed credentials still wins.
        if (proxy := os.environ.get(_VERTEX_PROXY_ENV)) and "credentials" not in kwargs:
            kwargs = {**_proxy_kwargs(proxy), **kwargs}
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

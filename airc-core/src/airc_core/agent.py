# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Shared agent-execution substrate: middleware stack, context budgeting, caching.

The model-call middleware every agent graph in the suite composes -- context-window
sizing, empty-response stripping, transient-error retry, prompt caching, and the
growing-prefix context cache -- plus the per-turn call-budget governor and the
per-call tracer. Both the persona-turn runner (airc) and the commit-review graph
(airc-processors) build on this so cost and robustness behavior cannot drift
between them. Nothing here knows about Config, Persona, Store, or the Room; the
two former Config couplings are now explicit parameters.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import logging
import math
import os
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    hook_config,
)
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import get_buffer_string
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.constants import TAG_NOSTREAM

from .collector import book_aside
from .model import _VERTEX_PROXY_ENV, _google_sdk, make_model
from .providers import STOP_REASON_KEYS
from .usage import MODEL_KEY, SOURCE_KEY, SUMMARIZATION, Usage, _k

log = logging.getLogger(__name__)

# The context-window size assumed for every model. Every budget and cache
# threshold below is a fraction of this. We deliberately assume 1M (Gemini, and
# Claude's 1M tier) rather than deriving it per model: the shedders are
# safe-biased and the growing cache measures the real prefix exactly after the
# first cached call, so the slack absorbs the difference for the models we run.
CONTEXT_WINDOW = 1_000_000


# How long to stop attempting cache creation after a failure. A permanent
# problem (IAM, disabled API, size floor) degrades to uncached for a long window
# instead of retrying inline on every turn. A transient failure (prefill
# overload, 429/503) is the storm the cache exists to relieve, so it backs off
# only briefly: the next turn rebuilds the cache as soon as the overload clears,
# rather than running uncached -- and feeding the overload with full re-prefills
# -- for many minutes. See _recache.
_CACHE_FAIL_COOLDOWN_S = 15 * 60
_CACHE_TRANSIENT_COOLDOWN_S = 30


def _is_cache_gone(e: Exception, name: str | None = None) -> bool:
    """A cached_content reference the server will not honor: deleted
    ("CachedContent not found", 404) or TTL-lapsed ("Cache content <id> is
    expired.", 400 INVALID_ARGUMENT). Either way the fix is recreate-and-retry,
    not fail the turn.

    When the cache id is known it is the most reliable signal -- the server
    quotes it in the error, so a match is wording-independent. Otherwise fall
    back to matching the deleted/expired wording (all the 404 form provides).
    A bare status code is not enough: 400 INVALID_ARGUMENT also covers unrelated
    request errors (e.g. a missing thought_signature), which must NOT be treated
    as a cache miss and silently retried.
    """
    s = str(e)
    if name and name in s:
        return True
    s = s.lower()
    return "cache" in s and ("not found" in s or "expired" in s)


def _response_cache_stats(response) -> tuple[int, int]:
    """(cache_read, cache_creation) as the model reported them; (0, 0) if
    unavailable.

    Their SUM is the measured size of the cached prefix: a call that reads the
    prefix reports it under cache_read, a call that writes it reports it under
    cache_creation, and a call that reads a shorter prefix while extending it
    splits the same span across both. So read+create names the span either way,
    which is what lets the Anthropic breakpoint be placed from measurement
    rather than from a chars-per-token estimate.
    """
    mr = getattr(response, "model_response", response)  # unwrap ExtendedModelResponse
    msgs = getattr(mr, "result", None)
    if msgs is None:
        msgs = [response] if isinstance(response, AIMessage) else []
    for m in reversed(msgs):
        usage = getattr(m, "usage_metadata", None)
        if usage:
            details = usage.get("input_token_details", {})
            return (
                int(details.get("cache_read", 0)),
                int(details.get("cache_creation", 0)),
            )
    return 0, 0


def _response_cache_read(response) -> int:
    """The cache_read tokens the model reported for a ModelResponse -- i.e. the
    true size of the cached prefix it just served. 0 if unavailable. Lets the
    serve-time window guard replace its char estimate of the cached prefix with
    the provider's exact count after the first cached call."""
    return _response_cache_stats(response)[0]


# Bound on the cause appended after the status. Wide enough for a real Vertex
# reason ("Requests ending with a model turn are not supported"), narrow enough
# that a per-call transient still costs one log line.
_SHORT_ERROR_CAUSE_CHARS = 160


def _clip(text: str, limit: int) -> str:
    # Marked, so a cut clause cannot read as the complete reason.
    return text if len(text) <= limit else text[:limit] + "..."


def _short_error(exc: Exception) -> str:
    """A one-line reason for a model/cache error, for logs.

    Vertex wraps failures in multi-kilobyte RPC dumps (nested original error,
    stack, source-location trace). The signal is the status and a short cause;
    this pulls those out so a transient overload does not print a 3k-char block
    on every occurrence. Falls back to a truncated first line.

    The cause is not decoration. A bare status triages a transient fine -- every
    RESOURCE_EXHAUSTED is the same blip -- but on a 400 the reason IS the bug,
    and every path that reports one (cache create, a lost review pass, a
    re-dispatch) logs through here. Dropping it left a permanent INVALID_ARGUMENT
    recorded as the word "INVALID_ARGUMENT" and nothing else, which no amount of
    log reading can diagnose. The dump is part of the server's status string
    itself, so no field selection avoids it: the cap bounds the line, and the
    reason survives because the server puts it first. .message is preferred
    over str() only because google-api-core's str() prepends the numeric code,
    which the status name already conveys; the cause chain is consulted for
    the same reason _is_transient consults it (the genai stack wraps)."""
    s = " ".join(str(exc).split())
    message = getattr(exc, "message", "") or getattr(exc.__cause__, "message", "")
    cause = _clip(" ".join(str(message or s).split()), _SHORT_ERROR_CAUSE_CHARS)
    for status in (
        "RESOURCE_EXHAUSTED",
        "PREFILL_QUEUE_OVERLOADED",
        "PREFILL_QUEUE_PREEMPTED",
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
        "PERMISSION_DENIED",
        "NOT_FOUND",
        "INVALID_ARGUMENT",
        "INTERNAL",
    ):
        if status in s:
            return f"{type(exc).__name__}: {status}: {cause}"
    return f"{type(exc).__name__}: {_clip(s, 120)}"


def _is_transient(exc: Exception) -> bool:
    """Is this a retry-worthy provider blip, as opposed to a permanent error?

    A structured status wins when the exception carries one: google-api-core
    sets .code and google.genai .code/.status_code to the HTTP status, and a
    number decides cleanly. The old substring pass over the full error text
    matched "429"/"503" inside request ids and token counts, laundering a
    permanent 400 into a transient -- prod burned _MAX_REVIEW_ATTEMPTS
    redeliveries re-spending a whole review on exactly that. 504 is in the
    set deliberately: a deadline reaping a hung stream is a transient like
    any 5xx (see _VERTEX_CALL_TIMEOUT_S). 529 is Anthropic's overloaded_error,
    which the anthropic SDK raises with that status, so the structured branch
    has to know it; the "overloaded" word below is never reached for it.

    Word matching remains only as the fallback for text-only exceptions
    (provider wording varies), with the bare digit strings gone. The anthropic
    SDK's timeout and connection errors carry no status and land here. The
    cause chain is consulted because langchain-google-genai re-raises API
    errors as a plain ChatGoogleGenerativeAIError with the structured original
    chained underneath -- on that stack the wrapper is the common case."""
    for e in (exc, exc.__cause__):
        code = getattr(e, "code", None) or getattr(e, "status_code", None)
        if isinstance(code, int):
            return code in (429, 500, 502, 503, 504, 529)
    msg = str(exc).lower()
    return any(
        k in msg
        for k in (
            "quota",
            "rate limit",
            "resource_exhausted",
            "exhausted",
            "unavailable",
            "overloaded",
            "internal server error",
            "timed out",
            "connection error",
        )
    )


# The transient-retry policy, shared by the agent-turn middleware and the bare
# retry helper below so airc has one definition of "how to retry a transient
# model error". Backoff 5+15+45+60+60+60 = ~4min worst case per model call. Early
# retries stay fast (5s, 15s) for the common per-minute 429 blip; the extra
# attempts at the 60s cap add runway for a sustained Gemini overload episode
# (commonly 5-15min). Cheap against icompleteu's 7200s turn timeout; a real but
# accepted bite out of the room's 900s one, where a fully exhausted retry costs
# ~28% of the turn and the turn then ends in the orchestrator's timeout notice
# rather than a reply. Sized for the harness because that is where a lost turn
# costs a whole job; the room degrades to a missed reply either way.
# Empty candidates do not use this ladder at all: _EmptyCandidateRetry retries
# them itself, once, and raises a non-retryable error (see _is_retryable).
_RETRY_MAX = 6
_RETRY_INITIAL_DELAY = 5.0
_RETRY_BACKOFF_FACTOR = 3.0
_RETRY_MAX_DELAY = 60.0


# Set by _EmptyCandidateRetry for the duration of its one mutated retry, read by
# the cache middlewares (nested inside it) and by CallBudgetMiddleware /
# FinalAnswerMiddleware to suppress a second nudge or notice. Two non-zero values:
# _RETRY_EMPTY (1) steps aside from the Vertex cached content resource because a
# zero-part STOP candidate is deterministic on cached prefixes; _RETRY_UNPARSABLE (2)
# keeps serving from the cache because a truncated tool-call stream is a transient
# output flake, while still suppressing cache-mark advancement and duplicate
# ephemeral notices on the mutated request.
_RETRY_EMPTY = 1
_RETRY_UNPARSABLE = 2
_empty_retry: contextvars.ContextVar[int] = contextvars.ContextVar(
    "airc_empty_candidate_retries", default=0
)

# In-place attempts on an unparsable tool call, and the wait before a repeat one
# (the first is immediate: the common shape is a one-off wire truncation). Two,
# so an agent with no graph-level re-ask -- the room, which does not use
# RequireStructuredResultMiddleware -- gets a second chance before its turn dies
# silently. Cheap: _RETRY_UNPARSABLE keeps the cached prefix, so an attempt buys
# only output tokens. Kept small deliberately: these calls never re-enter
# before_model, so they are invisible to the call and time budgets, and the
# graph-level re-ask (not more of the same request) is the real escape hatch.
_UNPARSABLE_RETRIES = 2
_UNPARSABLE_RETRY_DELAY = 2.0

# Ceiling on graph-level re-asks for an unparsable response, independent of the
# caller's max_reasks (set in the hundreds for prose, where the model is still
# answering and the call cap is the real bound). Each round here also spends
# _UNPARSABLE_RETRIES in-place attempts, so 6 is already up to 18 calls against
# a provider fault. Observed clusters have never needed more than 4.
_MAX_UNPARSABLE_REASKS = 6


class EmptyCandidateError(Exception):
    """A model call returned a zero-part candidate: no text and no tool calls.

    Gemini's known bug: returns finish_reason=STOP with zero parts -- reads as a
    benign end-of-turn but carries nothing (the silent-dead-turn shape). Raised
    by _EmptyCandidateRetry AFTER its own single mutated retry (drop the cached
    prefix + append a nudge) also comes back empty, so the empty is deterministic
    rather than a one-off flake. Not retryable by ModelRetryMiddleware (see
    _is_retryable): an identical resend would reproduce it. on_failure="error"
    re-raises; the harness catches it by type and surfaces a named diagnostic
    instead of a generic traceback.
    """


def _is_retryable(exc: Exception) -> bool:
    """Retry policy for ModelRetryMiddleware: transient provider errors (429/503/
    overloaded) only. A zero-part empty candidate is NOT retried here --
    _EmptyCandidateRetry owns that path and mutates the request on its single
    retry (drop the cached prefix + append a nudge), because an identical resend
    reproduces the deterministic empty (observed: 6 identical calls, 0 output
    tokens). _is_transient falls back to string matching (provider-dependent
    wording) for the bare retrying() helper which never sees empty candidates
    (single-shot calls, no ToolStrategy).

    The type check is what enforces that, and it has to come first: the raise
    embeds the provider's finish_reason in the message, so a reason reading
    MODEL_OVERLOADED or naming an unavailable region would match _is_transient
    and hand the empty back to the retry layer -- 14 model calls and the full
    backoff ladder, worse than the wedge this path exists to prevent."""
    if isinstance(exc, EmptyCandidateError):
        return False
    return _is_transient(exc)


def retrying(model):
    """Wrap a bare chat model so transient errors retry with the same policy the
    agent-turn middleware uses.

    For the single-shot classifier calls (coordinator routing, commit triage)
    that invoke a model directly, outside any create_agent graph -- so
    ModelRetryMiddleware never sees them. Keyed on _is_transient, not an
    exception type, because the transient signal is provider-dependent wording
    ("429", "overloaded", ...), not a class; Runnable.with_retry only filters by
    type, so it cannot express this. Retries only _is_transient errors; anything
    else (and a final exhausted retry) propagates to the caller's own guard."""
    from langchain_core.runnables import RunnableLambda

    async def _call(prompt):
        delay = _RETRY_INITIAL_DELAY
        for attempt in range(_RETRY_MAX + 1):
            try:
                return await model.ainvoke(prompt)
            except Exception as e:
                if attempt == _RETRY_MAX or not _is_transient(e):
                    raise
                log.info(
                    "model retry %d/%d after transient (%s)",
                    attempt + 1,
                    _RETRY_MAX,
                    _short_error(e),
                )
                await asyncio.sleep(delay)
                delay = min(delay * _RETRY_BACKOFF_FACTOR, _RETRY_MAX_DELAY)
        return None

    return RunnableLambda(_call)


_ELIDED_TOOL_RESULT = "[tool result from an earlier turn elided to save context]"
# Backstop ceiling on a single intact tool result's characters in a request.
# The eliding pruners always keep the most recent result intact, so one result
# larger than the window overflows the request no matter how much else is shed,
# and every retry 400s. This caps that survivor. Set above the source cap in
# mcptools (_MAX_TOOL_RESULT_CHARS, 50k) so a normally-capped result passes
# untouched; only a result that bypassed the source cap (an unexpected content
# shape, a tool from another server) is truncated here. ~240k chars is ~60-120k
# tokens depending on density.
_MAX_KEPT_RESULT_CHARS = 240_000


def prune_to_recent_tool_results(messages: list, keep: int) -> list | None:
    """Elide every tool result except the most recent `keep`, by recency.

    The hard-threshold backstop's shedder: when a long tool-using turn would
    overflow the window, keep only the most recent result and stub the rest, so
    the turn can still complete with a final reply. Returns the pruned list, or
    None if nothing changed. Only the model request is pruned; the checkpoint is
    left intact.
    """
    tool_idx = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    victims = set(tool_idx[:-keep] if keep > 0 else tool_idx)
    out = list(messages)
    changed = False
    for i in victims:
        m = out[i]
        if len(str(m.content)) > len(_ELIDED_TOOL_RESULT):
            out[i] = m.model_copy(
                update={"content": _ELIDED_TOOL_RESULT, "artifact": None}
            )
            changed = True
    return out if changed else None


def truncate_oversized_tool_results(messages: list, max_chars: int) -> list | None:
    """Hard-cap each intact tool result's characters in the request.

    Counterpart to the eliding pruners: those drop whole results by age or
    recency but always keep the most recent one intact, so a single result
    larger than the window overflows the request no matter how much is shed and
    every retry 400s. This bounds any result still over max_chars to a truncated
    prefix, so no single result can wedge a turn regardless of which tool
    produced it. Request-only; the checkpoint is left intact.
    """
    out = list(messages)
    changed = False
    for i, m in enumerate(out):
        if not isinstance(m, ToolMessage):
            continue
        content = str(m.content)
        if content == _ELIDED_TOOL_RESULT or len(content) <= max_chars:
            continue
        cut = len(content) - max_chars
        out[i] = m.model_copy(
            update={
                "content": content[:max_chars]
                + f"\n[... {cut} more chars truncated to fit the context window]",
                "artifact": None,
            }
        )
        changed = True
    return out if changed else None


# The fraction of the window at which we shed to avoid a 400. The gap to 100%
# absorbs the estimate's one-call lag (the usage signal is the previous call's
# input_tokens) plus one capped tool result. Below it, requests go intact so the
# prefix cache is never poisoned by a preemptive strip.
_HARD_FRACTION = 0.90
# Rough chars-per-token. Overestimates slightly for English prose, so the
# token estimate runs a touch high -- pruning a little early is the safe bias.
_CHARS_PER_TOKEN = 4

_TERMINATE_NUDGE = (
    "Context budget for this turn is nearly exhausted. Do not call any more"
    " tools; write your final reply now from what you already have."
)


def _prev_input_tokens(messages: list) -> int:
    """Input tokens the previous model call reported, from the latest AIMessage.

    The provider's own count of the prompt it last saw, persisted on the
    message and free to read. 0 before any model call has happened in this
    thread (a cold first turn, which is always small).
    """
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.usage_metadata:
            return int(m.usage_metadata.get("input_tokens", 0))
    return 0


def _estimate_input_tokens(messages: list) -> int:
    """Estimate the token size of a request's messages.

    The local char estimate always measures the full current message list, so
    the decision stays consistent across calls even though pruning only touches
    the request (never the checkpoint). The provider-exact previous count is a
    floor: it can only push the estimate up, catching token-dense content (tool
    schemas, code, non-English) that the char heuristic underweights -- it can
    never hide growth.
    """
    local = sum(len(str(m.content)) for m in messages) // _CHARS_PER_TOKEN
    return max(local, _prev_input_tokens(messages))


def _intact_tool_results(messages: list) -> int:
    """Tool results still carrying their real payload (not the elided stub)."""
    return sum(
        1
        for m in messages
        if isinstance(m, ToolMessage) and str(m.content) != _ELIDED_TOOL_RESULT
    )


def compact_for_budget(messages: list, window: int) -> tuple[list, bool]:
    """Shed context only when a request would otherwise overflow `window`.

    Returns (messages, drop_tools). With prefix caching on, a re-sent tool result
    behind the cache boundary is a cheap cache_read, and stripping mutates the
    cacheable prefix (forcing a cache rebuild and converting cache_reads into a
    miss) -- so we do NOT shed preemptively. Only at the hard threshold, where the
    request would 400, do we shed down to the last tool result, truncate an
    oversized survivor, and signal the caller to drop tools so the turn wraps up
    with a final reply. By this point the growing cache has stepped aside per its
    own window guard, so the shed is not fighting a live cache. Below the
    threshold the request is returned intact; the return value is a pure function
    of the inputs.
    """
    est = _estimate_input_tokens(messages)
    if est < window * _HARD_FRACTION:
        return messages, False
    before = _intact_tool_results(messages)
    shed = prune_to_recent_tool_results(messages, keep=1) or messages
    # If the single kept result alone exceeds the window the request still 400s;
    # truncate any oversized survivor so the turn can complete with a final reply.
    capped = truncate_oversized_tool_results(shed, _MAX_KEPT_RESULT_CHARS)
    shed = capped or shed
    log.info(
        "context budget: est %d tok >= hard %.0f%% of %d; shed %d of %d tool"
        " results to the last one%s and dropping tools to force a reply",
        est,
        _HARD_FRACTION * 100,
        window,
        before - _intact_tool_results(shed),
        before,
        ", truncating an oversized result" if capped else "",
    )
    return shed, True


class _ContextBudget(AgentMiddleware):
    """Keep a turn's request under the model's context window.

    Without this, a single long turn (recursion_limit lets one turn make many
    tool calls) accumulates every result until the request exceeds the window
    and the API rejects it, wasting the whole turn. compact_for_budget only acts
    at the hard threshold -- with prefix caching, keeping results is cheap and
    stripping poisons the cache, so below that the request goes intact; at it, the
    last-resort shed sheds to one result and forces the turn to wrap up. The
    checkpoint is never mutated -- only the request.
    """

    def __init__(self, window: int = CONTEXT_WINDOW) -> None:
        self._window = window

    async def awrap_model_call(self, request, handler):
        messages, drop_tools = compact_for_budget(request.messages, self._window)
        overrides: dict = {}
        if messages is not request.messages:
            overrides["messages"] = messages
        if drop_tools:
            overrides["tools"] = []
            sys = request.system_message
            base = f"{sys.content}\n\n" if sys and sys.content else ""
            overrides["system_message"] = SystemMessage(base + _TERMINATE_NUDGE)
        if overrides:
            request = request.override(**overrides)
        return await handler(request)


def _is_empty_ai(m) -> bool:
    """An AIMessage carrying neither text nor tool calls.

    Gemini serializes it to a Content with zero parts and rejects any
    request that replays it. Intermediate tool-call steps have empty text
    but carry tool_calls, so they are not empty by this definition.
    """
    return isinstance(m, AIMessage) and not m.tool_calls and not str(m.content).strip()


class _DropEmptyResponses(AgentMiddleware):
    """Keep empty model responses out of both the checkpoint and the request.

    A turn that yields no text and no tool calls leaves an empty AIMessage in
    the thread; Gemini then rejects the next turn that replays it ("must
    include at least one parts field"). aafter_model removes it before the
    checkpoint write so new turns never persist it. awrap_model_call also
    strips any empty AIMessage from the outgoing request, which heals
    checkpoints poisoned before this middleware existed (the stale message
    sits mid-history, where aafter_model -- which only inspects the last
    message -- cannot reach it).

    Overlaps _EmptyCandidateRetry, deliberately, and the two halves differ in
    how live they are. The retry raises on an empty candidate before it can
    reach state, so aafter_model now only fires on a path that bypasses the
    wrap -- a cheap belt, kept. awrap_model_call is NOT redundant: the room
    checkpoints to durable SQLite, so threads poisoned before either middleware
    existed still carry mid-history empties, and this is the only thing that
    keeps them off the wire.
    """

    async def awrap_model_call(self, request, handler):
        if any(_is_empty_ai(m) for m in request.messages):
            kept = [m for m in request.messages if not _is_empty_ai(m)]
            request = request.override(messages=kept)
        return await handler(request)

    async def aafter_model(self, state, runtime):
        messages = state.get("messages") or []
        if messages and _is_empty_ai(messages[-1]) and messages[-1].id is not None:
            return {"messages": [RemoveMessage(id=messages[-1].id)]}
        return None


class _EmptyCandidateRetry(AgentMiddleware):
    """Retry a provider-side empty candidate once, mutated, then surface it.

    Gemini's known bug: returns a zero-part response (finish_reason=STOP or
    SAFETY/RECITATION, but no content and no tool calls) that reads as a benign
    end-of-turn but carries nothing -- the silent-dead-turn shape. Without this,
    the empty AIMessage sails through (or is scrubbed by _DropEmptyResponses)
    and the turn ends with no report, scored as a dead turn by the reentry loop.
    The retry happens HERE rather than through ModelRetryMiddleware's backoff,
    because that layer resends the identical request and a deterministic
    zero-part candidate reproduces on identical input (observed: 6 identical
    calls, 0 output tokens). When the mutated retry is also empty the turn
    errors visibly (the harness names the empty candidate) instead of going
    silent.

    Detection scope: empty candidate (0 parts: no text AND no tool calls) and
    STOP-with-no-content only -- the flake family. A SAFETY/RECITATION block
    WITH content is a genuine refusal, not a flake; not retried here. Nor is a
    tool call the provider truncated, which reads as zero-part to a check on
    content and tool_calls alone: that has its own retry and its own wording
    (see _unparsable_tool_call and _retry_unparsable).

    Note there is no legitimately-empty reply to churn on: an agent with nothing
    to say answers with a sentinel (the room's NOTHING_TO_ADD) or calls its
    report tool, both non-empty. A zero-part candidate is always the bug.

    Sits in front of _DropEmptyResponses, which stays for the request-scrubbing
    half its docstring describes (durable checkpoints poisoned before either
    middleware existed).

    Placement: listed AFTER ModelRetryMiddleware in base_middleware, so a raise
    propagates up to it. EmptyCandidateError is NOT retryable (see _is_retryable):
    this middleware owns the empty retry, because ModelRetryMiddleware would
    resend the identical request and a deterministic zero-part candidate
    reproduces on identical input.
    """

    async def awrap_model_call(self, request, handler):
        # Attempt 1 is the turn's own call, unchanged and cached as usual: this
        # middleware costs nothing until something comes back empty.
        resp = await handler(request)
        if not _empty_retry.get() and _first_unparsable_tool_call(resp) is not None:
            return await self._retry_unparsable(request, handler, resp)
        empty = _empty_ai_message(resp)
        if empty is None:
            # A non-empty response ends the episode: clear the counter so a later
            # empty in the same turn starts fresh against a warm cache.
            if _empty_retry.get():
                _empty_retry.set(0)
            return resp
        # Still empty. Do NOT raise for an identical retry -- the cache (nested
        # inside) would re-read the same prefix and the model would reproduce the
        # zero-part candidate. Mutate, once: set _empty_retry so the growing
        # cache steps aside for this one call (a poisoned prefix can only
        # reproduce the empty) and append a nudge that forces output, in case
        # the poison is in the message history. Bounded: one mutated retry, then
        # surface. Raises EmptyCandidateError (not retryable) so it propagates
        # to the harness as a named dead turn.
        _empty_retry.set(_RETRY_EMPTY)
        log.warning(
            "empty candidate (finish_reason=%s); retrying once uncached with a"
            " nudge to force output",
            _finish_reason(empty),
        )
        resp2 = await handler(
            request.override(messages=[*request.messages, _EMPTY_NUDGE])
        )
        empty2 = _empty_ai_message(resp2)
        if empty2 is None:
            _empty_retry.set(0)
            return resp2
        raise EmptyCandidateError(
            f"empty candidate (finish_reason={_finish_reason(empty2)})"
        )

    async def _retry_unparsable(self, request, handler, resp):
        """Re-call the model on an unparsable response, in place and bounded,
        then hand the response back rather than raising.

        What the attempt sends depends on which shape came back.

        _DROPPED_CALL gets the nudge. The model emitted a call, the arguments
        did not arrive whole, and "send the same call with less in it" is both
        true and actionable against the common cause, a large structured result
        truncated mid-string.

        _EMPTY_TOOL_STOP gets the request unchanged. Nothing arrived, so there
        is nothing to correct, and the nudge would assert a truncation that did
        not happen and ask a reviewer to shrink reads it never made. A bare
        resend is not a weaker correction than a nudged one here: the Messages
        API is stateless and the empty response never goes back on the wire, so
        either way the next attempt is a fresh sample of the same prefix, not a
        continuation. Nor does the mutation buy determinism-breaking the way it
        does for an empty candidate -- Anthropic removed temperature, top_p,
        top_k and seed, and _drop_anthropic_sampling strips them, so an
        identical request already resamples.

        Not EmptyCandidateError: that names a different failure, _is_retryable
        rejects it by type, and its consumer logs a dead turn. An unparsable
        response is not a dead turn -- if every attempt fails,
        RequireStructuredResultMiddleware re-asks without charging the prose
        re-ask budget.

        _empty_retry is set to _RETRY_UNPARSABLE so inner middlewares do not
        append their ephemeral nudges or advance cache marks into the nudge
        tail, while VertexContextCacheMiddleware continues serving from the
        existing cached prefix instead of paying an uncached resend.
        """
        bad = _first_unparsable_tool_call(resp)
        kind = _unparsable_kind(bad)
        # One ephemeral nudged request, built once and reused: a tail append, so
        # a _DROPPED_CALL attempt still reads the cached prefix. An
        # _EMPTY_TOOL_STOP attempt sends `request` itself, an exact repeat.
        nudged = request.override(
            messages=[*request.messages, _UNPARSABLE_TOOL_CALL_NUDGE]
        )
        _empty_retry.set(_RETRY_UNPARSABLE)
        try:
            for attempt in range(1, _UNPARSABLE_RETRIES + 1):
                log.warning(
                    # Every shape is informative: a retry that comes back
                    # identical says the request, not the wire, decides this.
                    "unparsable[%s] attempt %d/%d (%s): %s",
                    kind,
                    attempt,
                    _UNPARSABLE_RETRIES,
                    "nudged" if kind == _DROPPED_CALL else "bare re-call",
                    _response_shape(resp, bad),
                )
                # An immediate resend already failed, so wait out a connection
                # hiccup before repeating it.
                if attempt > 1:
                    await asyncio.sleep(_UNPARSABLE_RETRY_DELAY)
                resp = await handler(nudged if kind == _DROPPED_CALL else request)
                if (bad := _first_unparsable_tool_call(resp)) is None:
                    # Logged so the recovery rate can be counted from the log
                    # rather than reconstructed by pairing warnings; it is the
                    # measurement that says whether re-calling works at all.
                    log.info(
                        "unparsable[%s] recovered on attempt %d/%d: %s",
                        kind,
                        attempt,
                        _UNPARSABLE_RETRIES,
                        _response_shape(resp, None),
                    )
                    return resp
                # Recomputed: a resend can come back the other shape, and the
                # next attempt should send what that shape calls for.
                kind = _unparsable_kind(bad)
        finally:
            _empty_retry.set(0)
        log.warning(
            "unparsable[%s] not recovered after %d attempts; handing the turn"
            " back to its own recovery: %s",
            kind,
            _UNPARSABLE_RETRIES,
            _response_shape(resp, bad),
        )
        return resp


# The one-shot nudge appended on a _DROPPED_CALL, and on that shape only. Names
# the actual failure: the zero-part nudge below tells the model its response was
# empty, which for this failure is false and unactionable -- it emitted a call,
# the arguments just did not arrive whole. Asking for a smaller one matters
# because the common shape is a large structured result truncated mid-string,
# which an identical retry can reproduce for the same reason.
#
# _EMPTY_TOOL_STOP does not get it. There the premise is false the other way --
# nothing arrived, so nothing was truncated -- and the instruction is worse than
# useless: a reviewer told to send less reads less, which costs review depth
# silently, with no error and no sign in the verdict.
_UNPARSABLE_TOOL_CALL_NUDGE = HumanMessage(
    "Your previous tool call could not be parsed: the arguments arrived"
    " truncated or malformed, so the call was dropped and nothing ran. Emit it"
    " again, complete and valid. If it was long, send the same call with less in"
    " it rather than a partial one."
)

# The one-shot nudge appended on a repeated empty candidate. Ephemeral --
# _EmptyCandidateRetry overrides the request with it, never mutates graph state,
# so it cannot reach the checkpoint and poison the next turn. Names the empty
# explicitly because Gemini's zero-part STOP reads as a benign end-of-turn.
_EMPTY_NUDGE = HumanMessage(
    "Your previous response was empty: zero parts, no text and no tool calls"
    " (finish_reason STOP). That is a known provider bug, not a valid end-of-turn."
    " Regenerate now -- call a tool or write your reply. An empty response is"
    " never correct here."
)


# Stop reasons that say the model WAS emitting a tool call. A message carrying
# one but no call at all -- neither parsed nor recorded as invalid -- did not
# come back zero-part: it came back with a call that could not be parsed, which
# is a different failure with a different cure. Anthropic writes "tool_use",
# Gemini "MALFORMED_FUNCTION_CALL"; compared case-insensitively because each
# provider cases its own vocabulary.
_TOOL_CALL_STOP_REASONS = frozenset({"tool_use", "malformed_function_call"})


# The two shapes _unparsable_kind separates. They share a detection site and
# nothing else: one is a call that arrived in pieces, the other is a response
# that arrived with no pieces at all, and the cure differs (see
# _EmptyCandidateRetry._retry_unparsable).
_DROPPED_CALL = "dropped-call"
_EMPTY_TOOL_STOP = "empty-tool-stop"


def _unparsable_kind(msg: AIMessage) -> str | None:
    """Which unparsable shape msg is, or None if it is a healthy message.

    _DROPPED_CALL: langchain recorded a call in invalid_tool_calls, which is
    where it puts one whose arguments would not parse. A non-empty list is
    proof there were parts and that they were cut short.

    _EMPTY_TOOL_STOP: the provider's stop reason says it was emitting a tool
    call, and neither a parsed nor a recorded-invalid call is behind it. The
    observed shape on Claude is content=[], tool_calls=[], invalid=[],
    stop_reason="tool_use", with output tokens billed -- the provider says it
    sent a call and sent nothing.

    The stop reason counts only when nothing parsed. Anthropic ends every
    successful tool-calling turn with "tool_use", so without that guard the
    predicate matched every Claude tool call: each one spent a second model
    call, dropped the growing cache, and was told its call had been dropped.
    """
    if msg.invalid_tool_calls:
        return _DROPPED_CALL
    if msg.tool_calls:
        return None
    if _finish_reason(msg).strip().lower() in _TOOL_CALL_STOP_REASONS:
        return _EMPTY_TOOL_STOP
    return None


def _unparsable_tool_call(msg: AIMessage) -> bool:
    """Whether msg is a tool call the provider truncated, or a response that
    claims a tool call and carries none. See _unparsable_kind for the two."""
    return _unparsable_kind(msg) is not None


def _usage_shape(msg) -> str:
    """What the provider billed for this message, read the way the ledger reads
    it.

    Usage.of_call holds the per-adapter quirks -- Anthropic's two cache-write
    keys and its missing output details, Vertex reporting thoughts apart from
    output_tokens -- and tolerates the None values adapters leave in the detail
    dicts. Reusing it keeps this diagnostic from becoming a second, thinner
    parse that drifts from the one the cost ledger trusts. Its cost fields go
    unused here; pricing is total (an unlisted model falls back to a generic
    rate), so paying for them on a failure path costs nothing worth avoiding.

    reasoning is the count this exists for. On an empty response it separates
    tokens billed for thinking -- the reading in which the model produced only
    reasoning and the provider sent none of it -- from tokens that were
    something else and went missing. Note it reads 0 on an adapter that
    attaches no output details, which is not the same as a call that did not
    think.

    The model id comes from the message because the response is all this has;
    a bare name misses the provider traits table and takes its defaults, which
    is the right reading for a provider that counts thinking inside output.
    """
    meta = getattr(msg, "response_metadata", None) or {}
    usage = Usage.of_call(
        getattr(msg, "usage_metadata", None),
        str(meta.get("model_name") or meta.get("model") or ""),
    )
    return (
        f"usage=(in={usage.input} cached={usage.cache_read}"
        f" out={usage.output} reasoning={usage.reasoning})"
    )


def _message_shape(msg) -> str:
    """One message's parts: class, content block types, call counts, stop reason.

    Types, names and counts only, never content -- these are review transcripts
    and this runs on every hit.

    The class name matters because an unaggregated AIMessageChunk carries its
    call in tool_call_chunks and leaves tool_calls empty, which is
    indistinguishable from a dropped call unless both are reported.
    """
    if not isinstance(msg, AIMessage):
        return type(msg).__name__
    content = msg.content
    if isinstance(content, str):
        blocks = ["text"] if content.strip() else []
    elif isinstance(content, list):
        blocks = [
            b.get("type", "?") if isinstance(b, dict) else type(b).__name__
            for b in content
        ]
    else:
        blocks = [type(content).__name__]
    chunks = len(getattr(msg, "tool_call_chunks", None) or ())
    return (
        f"{type(msg).__name__} blocks={blocks}"
        f" calls={len(msg.tool_calls or ())}"
        f" invalid={[c.get('name') for c in msg.invalid_tool_calls or ()]}"
        f" chunks={chunks}"
        f" stop={_finish_reason(msg)}"
        f" {_usage_shape(msg)}"
    )


def _response_shape(resp, bad) -> str:
    """Every message the call returned, with the matched one starred.

    _first_unparsable_tool_call takes the FIRST matching AIMessage, so a
    response split into a thinking message and a tool-calling message matches on
    the thinking one while the call sits intact beside it. That reading cannot
    be told from a genuinely dropped call by looking at the matched message
    alone, and it would reproduce on every retry -- see the TODO above.

    Also reports the keys of additional_kwargs and response_metadata for the
    matched message, where a provider puts what it could not model otherwise,
    the model that produced it (the shape has only been seen on one provider,
    and a roster runs several), and the refusal details if any.

    stop_details is Anthropic's structured refusal field. It is None on
    everything but a refusal, and its categories include reasoning_extraction
    -- a request to reproduce internal reasoning -- which a prompt that asks a
    model to explain itself could plausibly trip. Only type and category are
    reported; the explanation is free text and stays out of the log.

    Every expression is total. A diagnostic that raised here would turn the
    recoverable call it is describing into a dead turn.
    """
    msgs = list(getattr(resp, "result", ()) or ())
    parts = [
        f"{i}{'*' if m is bad else ''}: {_message_shape(m)}" for i, m in enumerate(msgs)
    ]
    keys = ""
    if isinstance(bad, AIMessage):
        meta = bad.response_metadata or {}
        details = meta.get("stop_details")
        if isinstance(details, dict):
            details = f"({details.get('type')},{details.get('category')})"
        keys = (
            f" model={meta.get('model_name') or meta.get('model')}"
            f" stop_details={details}"
            f" kwargs={list(bad.additional_kwargs or ())}"
            f" meta={list(meta)}"
        )
    return f"result[{len(msgs)}]=({'; '.join(parts)}){keys}"


def _empty_ai_message(resp):
    """The first zero-part AIMessage in resp (no text AND no tool calls), or None.

    A tool-calling step with empty text carries parts (the tool_calls), so it is
    not empty by this definition. resp.result is the message list the model
    returned; a non-AI response (e.g. a structured-output object) has none.

    A truncated tool call reads as zero-part to a check that looks only at
    content and tool_calls -- empty content, and nothing in tool_calls because
    the arguments did not parse -- so it is excluded explicitly. Treating one as
    the zero-part flake told the model its response had been empty (it had not),
    dropped the cached prefix to defeat a determinism that was never the cause,
    and ended in EmptyCandidateError, which _is_retryable rejects BY TYPE. A
    truncation is exactly the transient an ordinary retry fixes, so that
    misdiagnosis converted a recoverable call into a dead turn."""
    for msg in getattr(resp, "result", ()) or ():
        if not isinstance(msg, AIMessage):
            continue
        if _unparsable_tool_call(msg):
            continue
        if not (msg.tool_calls or []) and not str(msg.content or "").strip():
            return msg
    return None


def _first_unparsable_tool_call(resp):
    """The first AIMessage in resp whose tool call did not parse, or None."""
    for msg in getattr(resp, "result", ()) or ():
        if isinstance(msg, AIMessage) and _unparsable_tool_call(msg):
            return msg
    return None


def _finish_reason(msg: AIMessage) -> str:
    """Why the model stopped, whatever its provider named the field.

    response_metadata is passed through with the provider's own key names.
    Google writes finish_reason; the Anthropic path writes stop_reason and
    nothing else, so reading only finish_reason made every Claude empty
    candidate report "unknown" -- in the one situation where the reason is the
    whole diagnostic.
    """
    metadata = msg.response_metadata or {}
    for key in STOP_REASON_KEYS:
        if value := metadata.get(key):
            return str(value)
    return "unknown"


# Marks a persisted call-budget nudge in the message history, so a threshold
# re-entered without the count advancing does not append a second copy. Mirrors
# GroundingReminderMiddleware's lc_source tagging; the companion lc_stage carries
# the threshold, because one nudge text fires at several counts by design.
_CALL_BUDGET_SRC = "call-budget"


class _CallBudgetState(AgentState):
    # UntrackedValue: per-turn (per graph invocation), NEVER checkpointed -- so on
    # a checkpointed persona graph the count resets each turn instead of
    # accumulating across the conversation's lifetime.
    model_calls: NotRequired[Annotated[int, UntrackedValue]]


class CallBudgetMiddleware(AgentMiddleware):
    """Steer a long tool-using turn toward converging, via escalating wrap-up
    nudges at given model-call counts, and optionally a per-call position
    pointer ("call 95 of 150").

    The hard cap itself is ModelCallLimitMiddleware; these nudges only get the
    model to wrap up before it. The schedule is caller-supplied -- a persona
    reply uses two thresholds, a commit review a longer escalation -- so both
    share the mechanism. List the final nudge a few calls below the cap and make
    it insistent (e.g. "produce your result now"), so the run rarely ends with
    nothing to show.

    Where a nudge lives is the caller's choice, and it is the difference between
    a one-call impulse and a standing instruction:

    - Ephemeral (default): appended to that one model request and never to graph
      state, so on a CHECKPOINTED persona graph it cannot bake "stop using tools"
      into every future turn. The cost is that the model sees it on exactly one
      call -- the next call's messages are rebuilt from state, which never held
      it. Fine for an instruction whose whole compliance is a single call ("write
      your reply now"); useless for one that means to govern a span ("stop
      widening the search"), which is present for 1 of the ~45 calls it addresses.
    - persist=True: written into state via the messages reducer as a TAIL append
      (never a mid-history insert, which would rewrite and poison the cached
      prefix), so it settles into the growing prefix and costs a cache read
      thereafter. Only for a graph with NO checkpointer, where one turn is the
      whole run and there is no later turn to poison.

    `progress` is the complement to a persisted schedule: a caller-supplied
    formatter over the completed-call count, appended ephemerally on EVERY call,
    so the tail always carries the current position and nothing stale. Deliberately
    not persisted -- only the latest value means anything, and persisting would
    accumulate one wrong counter per call.
    """

    state_schema = _CallBudgetState

    def __init__(
        self,
        stages: list[tuple[int, str]],
        *,
        persist: bool = False,
        progress: Callable[[int], str] | None = None,
    ) -> None:
        super().__init__()
        # threshold -> nudge; each fires once (model_calls steps by 1 per call,
        # so an exact-equality lookup hits each threshold exactly once).
        self._stages = dict(stages)
        self._persist = persist
        self._progress = progress

    def after_model(self, state, runtime) -> dict[str, Any]:
        return {"model_calls": state.get("model_calls", 0) + 1}

    async def aafter_model(self, state, runtime) -> dict[str, Any]:
        return self.after_model(state, runtime)

    def before_model(self, state, runtime) -> dict[str, Any] | None:
        # The persisting half: awrap_model_call cannot write state (it returns a
        # ModelResponse), so a nudge that has to survive its own call is appended
        # here instead, as GroundingReminderMiddleware does.
        if not self._persist:
            return None
        n = state.get("model_calls", 0)
        nudge = self._stages.get(n)
        # Keyed on the THRESHOLD, not the text: a schedule fires the same prose
        # at several counts on purpose (a re-ask 20 calls later), so deduping by
        # content would silently drop every repeat after the first.
        if nudge is None or any(
            m.additional_kwargs.get("lc_stage") == n
            for m in state["messages"]
            if isinstance(m, HumanMessage)
            and m.additional_kwargs.get("lc_source") == _CALL_BUDGET_SRC
        ):
            return None
        log.info("call budget: wrap-up nudge at %d model calls (persisted)", n)
        return {
            "messages": [
                HumanMessage(
                    nudge,
                    additional_kwargs={
                        "lc_source": _CALL_BUDGET_SRC,
                        "lc_stage": n,
                    },
                )
            ]
        }

    async def abefore_model(self, state, runtime) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    async def awrap_model_call(self, request, handler):
        # state.model_calls is the count of calls already completed this turn.
        n = request.state.get("model_calls", 0)
        # after_model increments model_calls, but an EmptyCandidateError raised
        # innermore skips it -- the count freezes and this threshold re-fires on
        # every retry of the same call (observed: the 45-call nudge appended to
        # each of 7 empty-candidate retries, none of which the model answered).
        # A retry is the same call, so append nothing: _EmptyCandidateRetry
        # re-enters this handler with the request that already carries both the
        # nudge and the pointer, and re-appending stacks copies of them.
        if _empty_retry.get():
            return await handler(request)
        extra = []
        if not self._persist and (nudge := self._stages.get(n)) is not None:
            log.info("call budget: wrap-up nudge at %d model calls", n)
            extra.append(HumanMessage(nudge))
        if self._progress is not None:
            extra.append(HumanMessage(self._progress(n)))
        if extra:
            request = request.override(messages=[*request.messages, *extra])
        return await handler(request)


class _FinalAnswerState(AgentState):
    # Per turn, never checkpointed, for the same reason as model_calls above.
    answer_calls: NotRequired[Annotated[int, UntrackedValue]]


class FinalAnswerMiddleware(AgentMiddleware):
    """Close the tools for the last `window` calls of a capped turn, so the
    turn ends with a result instead of at the cap.

    ModelCallLimitMiddleware ends a turn at its cap by jumping to the end with
    an artificial message. For a ToolStrategy agent that is the worst outcome
    available: the structured result was never produced, the caller sees None,
    and every read the turn made is thrown away -- a review pass that hits the
    cap contributes nothing to its ensemble. The nudges before the cap ask the
    model to stop; this leaves it nothing else to do.

    Mechanism: inside the window every tool call is answered with `refusal`
    instead of being executed, and `notice` rides on each model request
    (request-only, like a nudge) saying the tools are closed and to report what
    it has. With ToolStrategy the model is bound with tool_choice="any", so a
    refused read leaves exactly one productive call: the result tool, which the
    model node parses itself and never routes through the tool hook, so it is
    never refused.

    close_after_reasks arms the same mechanism on the other no-verdict path,
    where the budget is not the problem: see _closed.

    Refusing at execution rather than withdrawing the tools from the request is
    deliberate: the tool list is the first thing in the cached prefix, and
    changing it re-bills the whole context on every call of the window. A
    refusal is an ordinary tool result appended at the tail, so the window
    costs a cached read per call, which is why it can be three calls rather
    than one -- a model that spends its first closed call on a read it is then
    refused still has two to hand in.

    The count is this middleware's own, per turn like CallBudgetMiddleware's,
    so the window does not depend on which other governors are in the stack.
    `max_calls` must equal ModelCallLimitMiddleware's run cap, or the window
    ends somewhere other than where the cap does -- and it is None on a stack
    whose turn ends on a dollar limit instead, where BudgetMiddleware owns the
    window and keys it to the money. close_after_reasks is the other arm and is
    about a different failure, so it survives that move.
    """

    state_schema = _FinalAnswerState

    def __init__(
        self,
        max_calls: int | None,
        notice: str,
        refusal: str,
        window: int = 3,
        *,
        close_after_reasks: int | None = None,
    ) -> None:
        super().__init__()
        self._max = max_calls
        self._notice = notice
        self._refusal = refusal
        self._window = window
        self._close_after_reasks = close_after_reasks

    def _closed(self, completed: int, state=None) -> bool:
        """Whether the model call made after `completed` calls has its tools shut.

        Two ways in. The window at the end of the budget is the original: the
        turn is about to be cut off, so leave nothing to do but report.

        close_after_reasks is the other, and it is about a different failure.
        RequireStructuredResultMiddleware re-asks a turn that ended in prose, but
        a re-ask lands while every read tool is still open -- and the model is
        bound tool_choice="any" over ALL of them, so it can satisfy the constraint
        with another read and never produce the verdict. That is not a terminal
        plain-text turn, so the re-ask does not fire again and the pass wanders on
        to the cap. Because REASKS_KEY increments when a re-ask is issued,
        close_after_reasks=1 closes read tools on the re-ask following the
        model's first prose ending, leaving the result tool as the only call
        that does anything.
        """
        if self._max is not None and completed >= self._max - self._window:
            return True
        if self._close_after_reasks is None or state is None:
            return False
        return state.get(REASKS_KEY, 0) >= self._close_after_reasks

    def after_model(self, state, runtime) -> dict[str, Any]:
        return {"answer_calls": state.get("answer_calls", 0) + 1}

    async def aafter_model(self, state, runtime) -> dict[str, Any]:
        return self.after_model(state, runtime)

    async def awrap_model_call(self, request, handler):
        n = request.state.get("answer_calls", 0)
        if not self._closed(n, request.state):
            return await handler(request)
        log.info("final answer: tools closed at call %d/%s", n + 1, self._max or "-")
        # The empty-candidate retry re-enters this wrap with its own nudge on
        # the request; the notice is already there from the first pass (same
        # reasoning as CallBudgetMiddleware).
        if _empty_retry.get():
            return await handler(request)
        return await handler(
            request.override(messages=[*request.messages, HumanMessage(self._notice)])
        )

    def _refuse(self, request):
        # The call that issued this tool call has already been counted, so the
        # count it was made after is one less.
        if not self._closed(request.state.get("answer_calls", 0) - 1, request.state):
            return None
        call = request.tool_call
        return ToolMessage(
            content=self._refusal,
            tool_call_id=call["id"],
            name=call.get("name"),
            status="error",
        )

    def wrap_tool_call(self, request, handler):
        return self._refuse(request) or handler(request)

    async def awrap_tool_call(self, request, handler):
        return self._refuse(request) or await handler(request)


# Per-turn spend, accumulated by BudgetMiddleware and read by its own governors.
# UntrackedValue for the same reason model_calls is: a budget is per graph
# invocation, never carried across the turns of a checkpointed conversation.
BUDGET_KEY = "budget"
# Calls left in the turn as the two cache brakes read it: remaining dollars over
# what the last call cost. Its own key rather than a field of _Spend, because a
# brake reads a scalar and knows nothing about this middleware -- it falls back
# to `cap - model_calls` when no budget middleware is in the stack (the room and
# the harness, until they migrate).
CALLS_LEFT_KEY = "calls_left"

# The zero-cache tripwire. A large prompt served with NOTHING read from cache
# is a provider-side eviction, not a client decision: at these sizes nothing on
# our side rewrites the prefix (the grounding reminder is a tail append, the
# ceiling actions fire far higher), so a run of them means the cache is simply
# not being served and every call is being billed at the full input rate.
#
# Observed 2026-09: Gemini 3.1 Pro passes at ~400k flipping between normal hit
# rates and runs of "0 cached" within one pass, both directions, at up to $1.60
# per uncached call. The floor keeps small calls out of it, and three in a row
# is what separates a run from a single cold replica after a retry or a
# re-route that warms again on the next call. Both are first guesses; read the
# per-call lines of the first trips before trusting them.
_ZERO_CACHE_FLOOR = 150_000
_ZERO_CACHE_TRIPS = 3


class CacheLossTrip(Exception):
    """A turn abandoned because the provider stopped serving its prompt cache.

    Its own type, and deliberately not folded into the cost limit: under the
    symptom a 25 USD limit is about fifteen calls, so the limit alone lets every
    affected pass burn most of its budget and then report "out of budget" for
    what is a provider fault. A caller that can stop doing expensive work --
    airc-processors exits the service on it -- needs to tell the two endings
    apart by type, not by reading a message.
    """


@dataclass(frozen=True)
class _Spend:
    """What a turn has spent, and the shape of the call that last spent it.

    `closed` is the read-closing window's latch: once what remains is under a
    few calls' worth, it stays shut for the rest of the turn. `closed_prev` is
    the latch as the model saw it when it made the call now issuing tool calls,
    so a read is refused only by a call that was TOLD the tools were closed --
    the same alignment FinalAnswerMiddleware gets by subtracting one from its
    counter.
    """

    usage: Usage = field(default_factory=Usage)
    last_usd: float = 0.0
    last_input: int = 0
    calls_left: float = math.inf
    closed: bool = False
    closed_prev: bool = False
    #: Consecutive large calls served with no cache read. Reset by any call
    #: that read something back, or that was too small to judge.
    zero_cache_run: int = 0


class _BudgetState(AgentState):
    budget: NotRequired[Annotated[_Spend, UntrackedValue]]
    calls_left: NotRequired[Annotated[float, UntrackedValue]]


def _last_ai(messages: list):
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m
    return None


class BudgetMiddleware(AgentMiddleware):
    """Bound one turn by what it costs rather than by how many calls it makes.

    A model call is a poor proxy for spend: what a call costs is the size of its
    context, which varies by an order of magnitude between turns and grows
    within one, so a call cap buys a cheap short turn or an expensive long one
    and the operator cannot say which in advance.

    Two numbers, answering different questions. `cost_limit` (USD) is the most
    a turn may ever cost: it ends the turn, keys the read-closing window, and is
    what "calls left" is measured against. `context_target` (prompt tokens) is
    where a typical turn should be wrapping up; it appears in the pointer and
    drives nothing else yet -- the nudge cadence still runs off call counts in
    CallBudgetMiddleware until the schedule is re-sized against the ledger.

    What it does per call:

    - Accumulates one `Usage` from the response just appended, priced through
      `Usage.of_call`. Never recomputed from the message list: summarization
      drops the AIMessages that carry usage, and a turn would then read as
      free. The collector prices the same response again to book it; both go
      through the one function, so they cannot disagree, and the graph must not
      depend on which callbacks a caller happened to attach.
    - Ends the turn once the limit is reached, which is one call late at most
      (the call that crosses it has already been paid for).
    - Closes the read tools once what remains is under `window` times the last
      call's cost -- three calls at a 20k context and three at a 600k one,
      rather than a count that means different things at each. Inside the
      window every read is refused with `refusal` and `notice` rides each
      request, so with ToolStrategy's tool_choice="any" the result tool is the
      only call left that does anything.
    - Publishes calls-left under CALLS_LEFT_KEY for the two cache brakes.
    - Raises CacheLossTrip when the provider stops serving the prompt cache on
      a large context, which is a fault to stop on rather than to pay out.

    The per-turn usage here is the agent's own calls only. Spend an invocation
    causes beside them -- a ceiling summarization on the filter model, an
    explicit cache creation -- reaches the ledger through the collector and is
    deliberately not charged against the turn's limit: the limit governs what
    the model does, and neither of those is the model's decision.
    """

    state_schema = _BudgetState

    def __init__(
        self,
        model_id: str,
        *,
        context_target: int,
        cost_limit: float,
        notice: str,
        refusal: str,
        window: float = 3.0,
        zero_cache_floor: int = _ZERO_CACHE_FLOOR,
        zero_cache_trips: int = _ZERO_CACHE_TRIPS,
    ) -> None:
        super().__init__()
        # A non-positive limit is not "no limit": before_model would end the turn
        # before its first call, so the graph would run nothing and report a
        # verdict-shaped nothing. Nobody means that -- the way to stop doing the
        # work is to stop asking for it -- and the failure is silent, so it is
        # refused where it is written rather than discovered in a log.
        if cost_limit <= 0:
            raise ValueError(f"cost_limit must be positive, got {cost_limit!r}")
        # 0 is legitimate and means "no target": it drives nothing yet, and the
        # pointer simply does not mention a target the operator has not chosen.
        # Negative is a typo.
        if context_target < 0:
            raise ValueError(
                f"context_target must be zero (no target) or positive, got"
                f" {context_target!r}"
            )
        self._model_id = model_id
        self._context_target = context_target
        self._cost_limit = cost_limit
        self._notice = notice
        self._refusal = refusal
        self._window = window
        self._zero_cache_floor = zero_cache_floor
        self._zero_cache_trips = zero_cache_trips

    def _spend(self, state) -> _Spend:
        return state.get(BUDGET_KEY) or _Spend()

    def _advance(self, prev: _Spend, response) -> _Spend:
        call = Usage.of_call(getattr(response, "usage_metadata", None), self._model_id)
        total = prev.usage + call
        left = self._cost_limit - total.usd
        # A call the provider reported no usage for (or a fake model in a test)
        # gives no unit to divide by; inf is the reading that leaves every brake
        # and the window where they were rather than slamming them shut on a
        # missing number.
        calls_left = max(0.0, left / call.usd) if call.usd > 0 else math.inf
        lost = call.input >= self._zero_cache_floor and call.cache_read == 0
        return _Spend(
            usage=total,
            last_usd=call.usd,
            last_input=call.input,
            calls_left=calls_left,
            closed=prev.closed or calls_left <= self._window,
            closed_prev=prev.closed,
            zero_cache_run=prev.zero_cache_run + 1 if lost else 0,
        )

    def after_model(self, state, runtime) -> dict[str, Any]:
        spend = self._spend(state)
        if (msg := _last_ai(state.get("messages") or [])) is not None:
            spend = self._advance(spend, msg)
        if spend.zero_cache_run >= self._zero_cache_trips:
            # Raised rather than ended with a jump: a turn that ends returns a
            # verdict-shaped nothing, which every caller reads as "the model had
            # nothing to report". This is not a verdict about the work, and the
            # caller has to be able to stop rather than take the next unit.
            raise CacheLossTrip(
                f"prompt cache not served: {spend.zero_cache_run} calls in a row"
                f" at or above {_k(self._zero_cache_floor)} input with 0 read"
                f" back (last {_k(spend.last_input)} for"
                f" {spend.usage.cost(spend.last_usd)}); ending the turn after"
                f" {spend.usage.calls} calls and {spend.usage.cost()}"
            )
        return {BUDGET_KEY: spend, CALLS_LEFT_KEY: spend.calls_left}

    async def aafter_model(self, state, runtime) -> dict[str, Any]:
        return self.after_model(state, runtime)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime) -> dict[str, Any] | None:
        spend = self._spend(state)
        if spend.usage.usd < self._cost_limit:
            return None
        reason = (
            f"Cost limit reached: {spend.usage.cost()} of"
            f" ${self._cost_limit:g} over {spend.usage.calls} calls."
        )
        log.info("budget: %s", reason)
        return {"jump_to": "end", "messages": [AIMessage(content=reason)]}

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    def pointer(self, spend: _Spend) -> str:
        """The position line riding every request: what has been spent against
        the ceiling, and how much has been read against the target. Data, not
        pressure -- the nudges do the steering, and this is on the tail of every
        call of the turn. "Of cap", because the cap is a ceiling, not a target.

        With no context target set the clause is dropped rather than rendered
        against a zero. This is prose the model reads on every call of the turn,
        so "context 410k of 0 target" is not a cosmetic problem: it is a
        sentence that means nothing being asserted to the model a hundred times.
        """
        u = spend.usage
        pct = round(100 * u.usd / self._cost_limit)
        line = f"{u.cost()} spent, {pct}% of the ${self._cost_limit:g} cap"
        if self._context_target > 0:
            line += (
                f"; context {_k(spend.last_input)} of {_k(self._context_target)} target"
            )
        return line

    async def awrap_model_call(self, request, handler):
        # The empty-candidate retry re-enters with the request that already
        # carries the pointer and the notice; re-appending stacks copies of
        # both (CallBudgetMiddleware's reasoning).
        if _empty_retry.get():
            return await handler(request)
        spend = self._spend(getattr(request, "state", None) or {})
        extra = [HumanMessage(self.pointer(spend))]
        if spend.closed:
            log.info(
                "budget: reads closed with %s of the $%g cap spent",
                spend.usage.cost(),
                self._cost_limit,
            )
            extra.append(HumanMessage(self._notice))
        return await handler(request.override(messages=[*request.messages, *extra]))

    def _refuse(self, request):
        if not self._spend(request.state).closed_prev:
            return None
        call = request.tool_call
        return ToolMessage(
            content=self._refusal,
            tool_call_id=call["id"],
            name=call.get("name"),
            status="error",
        )

    def wrap_tool_call(self, request, handler):
        return self._refuse(request) or handler(request)

    async def awrap_tool_call(self, request, handler):
        return self._refuse(request) or await handler(request)


# Tag on the re-ask reminders this middleware injects. No longer load-bearing
# for the bound (the re-ask count is an UntrackedValue counter, below) -- kept
# so a reminder is identifiable in logs/checkpoints and a test can assert it was
# injected, the same self-tagging scheme GroundingReminderMiddleware uses.
_REQUIRE_RESULT_SRC = "require_result_reask"


# The state key RequireStructuredResultMiddleware counts re-asks in, named once
# because FinalAnswerMiddleware reads it too (close_after_reasks). Asserted
# against the schema below at import: renaming the field without this constant
# would not fail anything, it would quietly stop the reads ever closing.
REASKS_KEY = "reasks"


class _RequireResultState(AgentState):
    # UntrackedValue: per-turn (per graph invocation), NEVER checkpointed -- so the
    # re-ask count resets each turn instead of accumulating across a stage-loop's
    # resumed turns on a shared thread. Verified to persist across a jump_to's
    # supersteps WITHIN one ainvoke (so the bound holds within the turn it guards),
    # which is the property the re-ask cap needs. Replacing the earlier scheme
    # (counting marker HumanMessages in the checkpointed messages channel): that
    # accumulated across turns on a shared thread, exhausting the budget for the
    # job's lifetime, and a per-turn reset via RemoveMessage risked a silent
    # prefix/tail mismatch in the growing cache (a deleted marker landing in the
    # cached prefix with no length change to trip the shrink guard). The counter
    # never touches the messages channel, so it cannot interact with caching.
    reasks: NotRequired[Annotated[int, UntrackedValue]]
    unparsable_reasks: NotRequired[Annotated[int, UntrackedValue]]


assert REASKS_KEY in _RequireResultState.__annotations__, (
    "REASKS_KEY must name a field of _RequireResultState: FinalAnswerMiddleware"
    " reads it to decide when to close the read tools"
)


class RequireStructuredResultMiddleware(AgentMiddleware):
    """Recover a ToolStrategy turn that ends in plain text with no structured
    result, instead of silently accepting the empty verdict.

    The failure this guards against: with ToolStrategy the loop exits the instant
    a model turn has no tool calls (the classic agent stop condition), so a model
    that writes its conclusion as prose -- rather than CALLING the result tool --
    ends the run with structured_response unset. The caller then sees None and,
    for a review verifier, that reads as "no verdict" and the finding it was
    checking is dropped. The observed shape is a short turn (a handful of calls,
    well under the call cap) that simply answers in prose; the call-budget nudges,
    which only fire near the cap, never see it.

    On such a terminal, this re-asks: it appends a corrective message (a TAIL
    append via the messages reducer, cache-friendly like the grounding reminder)
    and jumps back to the model via the framework's jump_to mechanism. The bound
    is a per-turn UntrackedValue counter (reasks) -- resets each turn (ainvoke),
    survives the jump's supersteps within the turn, and never touches the messages
    channel, so it cannot interact with prefix caching. It composes with the hard
    call cap without competing: ModelCallLimitMiddleware enforces in before_model,
    so a jump back to the model re-enters that check and an exhausted turn still
    ends there -- this only recovers turns that stopped early with budget to spare.
    Domain-neutral: the reminder wording is the caller's (a review verifier names
    its result tool and verdict), so airc-core stays free of any persona or stage
    vocabulary.
    """

    state_schema = _RequireResultState

    def __init__(self, reminder: str, max_reasks: int = 3) -> None:
        super().__init__()
        self._reminder = reminder
        self._max = max_reasks

    @staticmethod
    def _is_reask(m) -> bool:
        # Not the bound (the UntrackedValue counter is); kept for observability so
        # a reminder is identifiable in the message list and a test can assert it.
        return (
            isinstance(m, HumanMessage)
            and m.additional_kwargs.get("lc_source") == _REQUIRE_RESULT_SRC
        )

    def _reask(self, state) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not messages:
            return None
        last = messages[-1]
        # Only a terminal plain-text answer or a dropped tool call qualifies. An
        # AIMessage carrying tool calls is an intermediate step (read tools, or a
        # pending structured call) the loop handles itself; a delivered verdict
        # sets structured_response. A failed structured call leaves a ToolMessage
        # last (handle_errors re-prompts) -- also not ours. So: last is an
        # AIMessage, no tool calls, and no structured_response yet.
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        if state.get("structured_response") is not None:
            return None
        if kind := _unparsable_kind(last):
            # An unparsable response that survived _EmptyCandidateRetry's
            # in-place attempts. Jump back to the model so the turn does not
            # exit early, but do not increment REASKS_KEY: FinalAnswerMiddleware
            # reads that key to close read tools when a model writes a prose
            # conclusion instead of calling the result tool, and a provider
            # failure is not a prose conclusion. Bounded separately by
            # unparsable_reasks so a broken provider cannot spin.
            #
            # The bound is its own, and tighter than max_reasks. That budget is
            # sized for nudging a model that is still answering (callers set it
            # in the hundreds, against the call cap); this one rides a provider
            # fault where each round also spends _UNPARSABLE_RETRIES in-place
            # attempts, so the same number would be that many times the calls
            # on a failure no amount of asking has ever fixed.
            u = state.get("unparsable_reasks", 0)
            cap = min(self._max, _MAX_UNPARSABLE_REASKS)
            if u >= cap:
                log.warning(
                    "require-result: unparsable[%s] after %d re-asks; giving up",
                    kind,
                    u,
                )
                return None
            log.info(
                "require-result: turn ended unparsable[%s]; re-asking (%d/%d) %s",
                kind,
                u + 1,
                cap,
                "with the nudge" if kind == _DROPPED_CALL else "bare",
            )
            out: dict[str, Any] = {"jump_to": "model", "unparsable_reasks": u + 1}
            # Same split as _retry_unparsable: a call that arrived truncated can
            # be told to send less, and a response that arrived with nothing in
            # it has nothing to be told. _DropEmptyResponses strips the empty
            # AIMessage from the outgoing request, so a bare jump re-sends the
            # prefix unchanged rather than replaying the failure back at the
            # model.
            if kind == _DROPPED_CALL:
                out["messages"] = [
                    HumanMessage(
                        _UNPARSABLE_TOOL_CALL_NUDGE.content,
                        additional_kwargs={"lc_source": _REQUIRE_RESULT_SRC},
                    )
                ]
            return out
        n = state.get(REASKS_KEY, 0)
        if n >= self._max:
            # Exhausted the re-asks: let the turn end verdict-less (the caller
            # surfaces None as incomplete, never as a clean pass). Bounded so a
            # model that refuses to call the tool cannot spin.
            log.warning(
                "require-result: still no structured result after %d re-asks;"
                " giving up",
                n,
            )
            return None
        log.info(
            "require-result: turn ended in plain text with no result;"
            " re-asking (%d/%d)",
            n + 1,
            self._max,
        )
        reminder = HumanMessage(
            self._reminder, additional_kwargs={"lc_source": _REQUIRE_RESULT_SRC}
        )
        return {"jump_to": "model", REASKS_KEY: n + 1, "messages": [reminder]}

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime) -> dict[str, Any] | None:
        return self._reask(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state, runtime) -> dict[str, Any] | None:
        return self._reask(state)


class _TimeBudgetState(AgentState):
    # UntrackedValue: per-turn (per graph invocation), NEVER checkpointed -- so the
    # turn-start stamp resets each turn instead of pinning the first turn's clock
    # for the whole conversation's lifetime.
    turn_start: NotRequired[Annotated[float, UntrackedValue]]


class TimeBudgetMiddleware(AgentMiddleware):
    """Steer a long-running turn toward converging via escalating wrap-up nudges at
    given wall-clock elapsed times -- the CallBudgetMiddleware pattern keyed on
    seconds rather than model-call count.

    The failure this guards against: an expensive tool-using turn hitting the
    orchestrator's hard per-turn timeout, which kills the turn and discards all the
    work it gathered. Nudging the model to wrap up before that deadline gets a reply
    out of the context already paid for. Stamp the thresholds below the hard timeout
    so convergence happens first; the timeout stays only as a backstop.

    Like CallBudgetMiddleware, the nudge is appended to the one model request
    (ephemeral) and never to graph state, so on a checkpointed persona graph it does
    not bake "stop using tools" into every future turn. Elapsed is measured only
    between model calls, so a single slow call or tool cannot be interrupted -- that
    remains the backstop timeout's job. Repeating the current stage's nudge on each
    call past its threshold is intended: sustained pressure to converge, at the cost
    of one ephemeral message per call.
    """

    state_schema = _TimeBudgetState

    def __init__(self, stages: list[tuple[float, str]]) -> None:
        super().__init__()
        # (elapsed_seconds, nudge) ascending; the most-advanced crossed stage wins.
        self._stages = sorted(stages)

    def before_model(self, state, runtime) -> dict[str, Any] | None:
        # Stamp the turn clock on the first call of this invocation only.
        if not state.get("turn_start"):
            return {"turn_start": time.monotonic()}
        return None

    async def abefore_model(self, state, runtime) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    async def awrap_model_call(self, request, handler):
        start = request.state.get("turn_start")
        if start is not None:
            elapsed = time.monotonic() - start
            nudge = next(
                (t for secs, t in reversed(self._stages) if elapsed >= secs), None
            )
            if nudge is not None:
                log.info("time budget: wrap-up nudge at %.0fs elapsed", elapsed)
                request = request.override(
                    messages=[*request.messages, HumanMessage(nudge)]
                )
        return await handler(request)


# The critical rule re-injected into a deep conversation. Formulated to name the
# exact slop mode (recalling vs reading = a guess) and the wanted behaviour (verify
# or flag), not a generic "ground everything" that the buried system prompt
# already says. The "[system reminder]" prefix matters: this is persisted as a
# HumanMessage in the agent's own history, so the prefix marks it as out-of-band
# and keeps it from reading as another user turn in the conversation.
_GROUNDING_REMINDER = (
    "[system reminder] Grounding check before you reply: every claim in your answer"
    " must trace to this thread or a tool result you ran this turn. Anything you are"
    " recalling rather than reading -- a symbol, path, line, number, or mechanism --"
    " is a guess; verify it or say plainly you are unsure. Never present a guess as a"
    " confident conclusion."
)
# Marker so a reminder is recognizable in the message list (and excluded from the
# tokens-since-last-reminder measure) without matching on its text.
_GROUNDING_SRC = "grounding_reminder"
# Insert one reminder per this many tokens of context growth. Once a thread is
# this deep the system prompt sits far from the tail and loses weight (recency /
# lost-in-the-middle); a fresh copy near the working tail every interval keeps the
# rule salient. 0 disables it.
_GROUNDING_REMINDER_TOKENS = 200_000


class GroundingReminderMiddleware(AgentMiddleware):
    """Insert a standing rule into the conversation once per `interval` tokens of
    context growth, so a long thread keeps a recent copy near the working tail
    rather than only at the (increasingly buried) system prompt. Defaults to the
    grounding rule; `reminder`/`src` make it reusable for any such rule.

    Writes the reminder into graph state via the messages reducer -- a TAIL append,
    never a mid-history insert (which would rewrite and poison the cached prefix).
    So it settles into the growing prefix cache like any other message and costs a
    cache-read, not a full re-send, thereafter. Self-tracked: it measures tokens
    since the last reminder in the current message list, so it re-arms on its own
    after a summarization compaction drops earlier reminders (no absolute counter
    to reset). 0 disables it.
    """

    def __init__(
        self,
        interval: int,
        reminder: str = _GROUNDING_REMINDER,
        src: str = _GROUNDING_SRC,
    ) -> None:
        super().__init__()
        self._interval = interval
        self._reminder = reminder
        # Per-instance, so two reminders on different intervals do not read each
        # other's inserts as their own and each suppress the other -- silently,
        # and only in threads long enough for both to fire.
        self._src = src

    @property
    def name(self) -> str:
        # create_agent both rejects duplicate middleware names and keys graph
        # nodes on them, and the default name is the CLASS name -- so
        # base_middleware's grounding reminder plus an application reminder
        # (the harness appends one per `reminders` entry) failed the whole
        # graph at compile time. src is already the per-instance identity
        # (_is_reminder keys on it), so the name rides the same uniqueness.
        return f"GroundingReminderMiddleware({self._src})"

    def _is_reminder(self, m) -> bool:
        return (
            isinstance(m, HumanMessage)
            and m.additional_kwargs.get("lc_source") == self._src
        )

    def _due(self, messages: list) -> bool:
        # Walk back from the tail: due once `interval` tokens of content accrue
        # without hitting a reminder (or reaching the start). A reminder within
        # that window means one is still recent -- not due.
        chars = 0
        for m in reversed(messages):
            if self._is_reminder(m):
                return False
            chars += len(str(m.content))
            if chars // _CHARS_PER_TOKEN >= self._interval:
                return True
        return False

    def before_model(self, state, runtime):
        if self._interval > 0 and self._due(state["messages"]):
            return {
                "messages": [
                    HumanMessage(
                        self._reminder,
                        additional_kwargs={"lc_source": self._src},
                    )
                ]
            }
        return None

    async def abefore_model(self, state, runtime):
        return self.before_model(state, runtime)


def _seed_vertex_cache_globals() -> str:
    """Seed `aiplatform.initializer.global_config` with location, project, and
    credentials before calling `CachedContent.create` or `.delete`.

    `langchain_google_vertexai.create_context_cache` is incomplete: it takes a
    ChatVertexAI model instance but never passes the model's location, project,
    or credentials to `caching.CachedContent.create`, which instead reads from
    the SDK initializer (`global_config`). `caching.CachedContent(name).delete`
    similarly relies on global initializer state.

    We seed `global_config` directly:
      - `_location`: must match the model's serving region (default us-central1).
      - `_project`: avoids lazy normalization that triggers an otherwise-needless
        Cloud Resource Manager `projects.get` call.
    Credentials are deliberately NOT seeded. Under the sandbox proxy
    (`AISAN_VERTEX_PROXY_ENDPOINT`) the box holds none: the cached-content client
    is a different stack from the chat client and insists on TLS, so it cannot
    use the plaintext loopback seam at all. The caller drives that path over REST
    instead; seeding here would only point a client we do not use at a credential
    we do not have. Everywhere else ADC applies as normal.
    """
    from google.cloud.aiplatform import initializer

    location = os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
    initializer.global_config._location = location
    if project := os.environ.get("GOOGLE_CLOUD_PROJECT"):
        initializer.global_config._project = project
    return location


# Summarization fires here, deliberately BELOW the shed's _HARD_FRACTION, and keeps
# the recent tail verbatim while compacting the rest. The two ceilings count on
# different bases -- the shed on the provider-exact input (system + tool schemas +
# messages, via the usage_metadata floor), summarization on a message-token
# estimate that may not see the fixed prefix -- so at an equal threshold the shed
# would pre-empt. Firing at 0.80 while the shed backstops at 0.90 leaves a margin
# wider than any realistic prefix, so summarization always leads; and the
# post-compaction request (~0.80*W) stays clear of the shed's 0.90 line, so a stale
# usage floor on the kept tail cannot make the shed re-fire right after a compaction.
_SUMMARY_TRIGGER_TOKENS = int(CONTEXT_WINDOW * 0.80)
# The verbatim recent tail kept after a compaction, in TOKENS not messages: a
# message count means ~10k tokens for chat turns but ~300k for tool-heavy ReAct
# rounds, so the same number behaves wildly differently per agent. A token budget
# is predictable, and the cutoff still lands on a message boundary (it snaps back
# to keep an AI/tool-call pair together), so a message is never split. 50k is a
# few recent rounds -- enough working set to continue coherently.
_SUMMARY_KEEP_TOKENS = 50_000
# How much of the dropped block the summarizer reads. langchain's default is 4000
# (strategy="last"), which at our trigger would summarize the final 4k of ~850k
# and silently discard the rest. We lift it to just cover the block. With a 50k
# keep the block is trigger-keep ~= 850k, so 0.85x reads the whole block without
# dropping anything; going higher buys nothing (the block sets the size, not this
# cap) and only widens the overflow ceiling. And overflow is the danger: if the
# XML-serialized block exceeds the summarizer's 1M window the call 400s, and the
# middleware swallows that and replaces the history with the error string. At 0.85
# the input is ~850k*~1.08 (XML tags + escaping; V8 C++ is <>-dense) ~= 918k, a
# safe ~80k under 1M. Invariant when tuning: keep + trim >= trigger (no silent
# drop) and trim*~1.1 < summarizer window (no overflow).
_SUMMARY_TRIM_TOKENS = int(CONTEXT_WINDOW * 0.85)


class _SkipOnSummaryFailure(SummarizationMiddleware):
    """SummarizationMiddleware that fails safe. The stock middleware swallows any
    summarize error and returns the exception text AS the summary, which then
    replaces the whole compacted block -- an overflow or transient error turns
    into catastrophic context loss. Here the summary call is allowed to raise, and
    the hook catches it and skips: the history is left intact, the _ContextBudget
    backstop sizes this turn's request, and compaction retries next turn.

    Only the async path is overridden -- airc drives agents via astream. If a sync
    caller ever appears, override _create_summary/before_model the same way.

    `model_id` is the configured id of `model`, tagged onto the summary call so
    the usage collector can price it: the call runs on the filter model, not
    the turn's, and the callback sees only the request."""

    def __init__(self, *args, model_id: str = "", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model_id = model_id

    async def _acreate_summary(self, messages_to_summarize):
        # The parent body, minus the try/except that swallows a failure into a
        # fake summary. Reimplemented (rather than sniffing the parent's error
        # string) so a genuine failure propagates for the hook to catch.
        if not messages_to_summarize:
            return "No previous conversation history."
        trimmed = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed:
            return "Previous conversation was too long to summarize."
        formatted = get_buffer_string(trimmed, format="xml")
        # This nested call inherits the turn's ambient callbacks (langchain merges
        # configs), and under astream(stream_mode="messages") that includes a
        # streaming handler -- which silently upgrades ainvoke to streaming, so
        # every summary token would be emitted into the message stream and
        # collected as reply text (the summary restates the conversation, so the
        # posted reply reads as an echoed prompt). TAG_NOSTREAM keeps the call out
        # of the message stream at the source; the metadata lets the usage
        # collector book it aside from the turn, on its own model.
        response = await self.model.ainvoke(
            self.summary_prompt.format(messages=formatted).rstrip(),
            config={
                "tags": [TAG_NOSTREAM],
                "metadata": {SOURCE_KEY: SUMMARIZATION, MODEL_KEY: self._model_id},
            },
        )
        return response.text.strip()

    async def abefore_model(self, state, runtime):
        try:
            return await super().abefore_model(state, runtime)
        except Exception as e:
            log.warning(
                "summarize: failed (%s); kept history, backstop covers the turn", e
            )
            return None


# Anthropic accepts only "5m" and "1h" here. Not the [caching] ttl_minutes
# knob, which is a free int for Vertex cachedContents.
_ANTHROPIC_CACHE_TTL = "5m"

# Session affinity: keeps a conversation's turns together so the prefix one
# turn cached is readable by the next. One header for both Vertex dialects:
# Claude's explicit breakpoints and Gemini's implicit cache are both served by
# the replica that holds the prefix, and without the routing key a turn lands
# wherever the balancer puts it and reads nothing back.
_VERTEX_SESSION_HEADER = "X-Vertex-Ai-Session-Id"


def _vertex_session_id(key: object) -> str:
    """A stable, opaque routing id for the conversation `key` names.

    Hashed so the value is always a legal header, survives a restart, and
    carries no internal identifier off the process. 32 hex chars only has to
    separate concurrent conversations."""
    return hashlib.sha256(str(key).encode()).hexdigest()[:32]


def _session_for(request, fallback: str) -> str:
    """The session id for this request's conversation: keyed on the thread, or
    `fallback` (one per middleware instance) for a checkpointer-less graph."""
    key = _thread_key(request)
    return fallback if key is None else _vertex_session_id(key)


def _with_header(settings: dict | None, key: str, header: str, value: str) -> dict:
    """model_settings with `header` added under settings[key] (a dict of
    headers), merged rather than assigned: neither the settings nor the headers
    already there are ours to drop."""
    settings = settings or {}
    return {**settings, key: {**(settings.get(key) or {}), header: value}}


class _GeminiVertexSession(AgentMiddleware):
    """The session-affinity header on every Gemini-on-Vertex call.

    Gemini's prompt cache is implicit and server-side, so there is nothing to
    mark on the request; what a turn needs is to reach the replica that holds
    the prefix the previous turn wrote. Observed without it, 2026-09: passes at
    ~400k context flipping between normal hit rates and runs of "0 cached"
    within one pass, both directions, at full input rates on every miss.

    The header rides `http_options`, the per-call hook langchain-google-genai
    exposes: it pops that bind kwarg and merges it into the request's own
    HttpOptions, whose headers the google-genai client merges into the wire
    request. Gated on the Vertex backend, since the Developer API has no such
    routing key; and the legacy ChatVertexAI stack is not covered -- it fixes
    its metadata at construction and takes no per-call header.
    """

    def __init__(self) -> None:
        super().__init__()
        self._fallback_session = uuid.uuid4().hex

    def _applies(self, model) -> bool:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            return False
        if not isinstance(model, ChatGoogleGenerativeAI):
            return False
        return bool(getattr(getattr(model, "client", None), "vertexai", False))

    def _settings(self, request) -> dict:
        settings = request.model_settings or {}
        opts = settings.get("http_options")
        # A caller may pass the SDK's HttpOptions object; the merge below wants
        # the dict form, which _prepare_request accepts too.
        if opts is not None and not isinstance(opts, dict):
            settings = {**settings, "http_options": opts.model_dump(exclude_none=True)}
        opts = dict(settings.get("http_options") or {})
        opts["headers"] = {
            **(opts.get("headers") or {}),
            _VERTEX_SESSION_HEADER: _session_for(request, self._fallback_session),
        }
        return {**settings, "http_options": opts}

    def wrap_model_call(self, request, handler):
        if not self._applies(request.model):
            return handler(request)
        return handler(request.override(model_settings=self._settings(request)))

    async def awrap_model_call(self, request, handler):
        if not self._applies(request.model):
            return await handler(request)
        return await handler(request.override(model_settings=self._settings(request)))


@dataclass
class _AnthropicPrefix:
    """Per-conversation state for the advancing history breakpoint.

    Deliberately thinner than _PrefixState: an Anthropic breakpoint is a mark in
    a request we were sending anyway, not a server-side object, so there is no
    name to hold, no model to rebind, and nothing to delete on eviction -- an
    abandoned entry simply ages out at the TTL.
    """

    # messages[:boundary] sits behind the history mark. 0 = no history mark yet,
    # so only the system+tools prefix is cached.
    boundary: int = 0
    # The provider's own count of the cached span (cache_read + cache_creation),
    # not an estimate. Compared across an advance by _record's not-reaching-
    # the-wire check.
    prefix_tokens: int = 0
    # Model calls since the mark last moved: 0 on the call that moved it, which
    # is how _record knows an advance happened.
    calls_since: int = 0
    seen_len: int = 0
    # Last placement decision, for the log line.
    why: str = "init"


class _AnthropicVertexCaching(AgentMiddleware):
    """Prompt caching for Claude on Vertex: two cache_control marks plus a
    session affinity header. The header is not optional decoration -- a mark
    with no affinity writes an entry the next turn may not find, and affinity
    with nothing tagged has nothing to read.

    The two marks are not redundant, and neither subsumes the other:

    * a STATIC mark closing system+tools. Identical across every conversation
      this agent runs, so a brand-new conversation hits it on its FIRST call.
      Review keys one thread per claim, so this is the only mark those
      short-lived threads can ever read from.
    * an ADVANCING mark closing history, moved forward when it pays. Private to
      one conversation, and the one that matters on long flows, where history
      is ~90% of the prompt.

    An advancing mark alone would cover the static span too (a cached span is
    always a prefix of the request), but only for a conversation that already
    has history -- which is exactly the case the static mark is not needed for.


    Separate from AnthropicPromptCachingMiddleware because that one gates on
    isinstance(model, ChatAnthropic), which ChatAnthropicVertex is not, so
    under our "ignore" setting it silently does nothing here. The two type
    gates are exclusive, so at most one fires per request. Subclassing would
    override a private method and reintroduce the same silent failure on the
    next upstream refactor.

    The omission is silent and expensive: Anthropic has no implicit cache, so
    a request without a breakpoint re-bills the whole prefix every turn
    (measured: cache_read stays 0 across identical calls). Gemini's implicit
    cache hides the same omission.

    Placement:

    * The static mark goes on the system message, and covers the tools too. The
      wire order is tools, system, messages, and caching is prefix-based.
      Measured with 28 tools and a ~330-token system message: cache_creation
      13124, then cache_read 13124.
    * The advancing mark goes on the last message that can hold one
      (_last_markable), which in an agent loop is the newest tool result. It may
      be the final message: everything that appends to the conversation does so
      through the state reducer, so the marked message is still at that index
      next call. The one exception is the empty-candidate nudge, which lives in
      the request only, and _advance stands down while it is in play.
    * Which part of that message carries the mark is _mark_placement's
      decision, and _advance refuses to advance onto a message it cannot place
      one on. The two must agree: the first version of this chose the placement
      independently, always tagging the last content block, which on a
      tool-calling AIMessage is the tool_use -- discarded in serialization. The
      state then recorded a cached span that was never sent. _record's span
      check is the backstop.
    * No model_settings["cache_control"]. That tags the LAST content block, i.e.
      that same nudge. Measured to change nothing.

    When to advance is a cost decision, not an interval -- see _advance_pays.
    Its input is B, the cost of creating the cache, and for Anthropic that is
    measured: moving the mark forward bills only the delta, because the shorter
    prefix is still cached and is read at the read rate. So there is no prefix
    to re-buy, the EOQ cadence _recache_pays solves for does not apply, and the
    mark advances whenever the turn has another call to read it back.


    No token floor: Anthropic declines to cache prefixes under ~1024 tokens
    without erroring, so a floor here would duplicate a server-side rule.

    The affinity id is per conversation, not per process, so concurrent long
    flows (review, verify, worker loops) do not share one. Inside an
    icompleteu box the sandbox proxy drops every client header and sends its
    own per-box id, which is the same granularity there: one box runs one
    conversation. Sent on every call, with or without a system message.

    If affinity does not take, a write is billed at Anthropic's 1.25x premium
    with no later read. The premium applies only to the cached span, so a long
    turn costs ~2.5% more, not 25%. Watch cache_write_tokens against
    cache_read in the ledger.
    """

    def __init__(
        self, ttl: str = _ANTHROPIC_CACHE_TTL, *, max_calls: int | None = None
    ):
        super().__init__()
        self._cache_control = {"type": "ephemeral", "ttl": ttl}
        # For a request with no thread id at all: a graph that configures none
        # (a direct, serial CommitReview, tests), where one middleware instance
        # is one conversation. A checkpointer-less graph still reports the
        # thread_id its config named, and the review graph names one per claim.
        self._fallback_session = uuid.uuid4().hex
        # The turn's model-call cap, for the end-of-turn brake in _advance_pays.
        # Optional: callers that do not track one still get history caching,
        # just without the brake (see _calls_left).
        self._max_calls = max_calls
        # Keyed by thread id, same single-writer argument as _GrowingPrefixCache:
        # the orchestrator serializes turns per (thread, agent), and review gives
        # each concurrent claim its own thread id.
        self._prefixes: OrderedDict[object, _AnthropicPrefix] = OrderedDict()

    def _blocks(self, request):
        """The system message as content blocks, or None if there is nothing
        to tag."""
        system = request.system_message
        if system is None:
            return None
        content = system.content
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else None
        if isinstance(content, list) and content:
            return list(content)
        return None

    def _tagged_system(self, blocks):
        """`blocks` with cache_control on the last one."""
        blocks = list(blocks)
        last = blocks[-1]
        blocks[-1] = (
            {**last, "cache_control": self._cache_control}
            if isinstance(last, dict)
            else {
                "type": "text",
                "text": str(last),
                "cache_control": self._cache_control,
            }
        )
        return SystemMessage(content=blocks)

    def _session_id(self, request) -> str:
        return _session_for(request, self._fallback_session)

    def _with_session_header(self, request):
        """model_settings plus the affinity header. extra_headers is a bind
        kwarg passed straight to messages.create, the only per-call hook; the
        model's additional_headers is fixed at construction."""
        return _with_header(
            request.model_settings,
            "extra_headers",
            _VERTEX_SESSION_HEADER,
            self._session_id(request),
        )

    def _state(self, request, messages) -> _AnthropicPrefix:
        """This conversation's breakpoint state, LRU-bounded.

        A shorter history than last seen means the boundary indexes messages
        that no longer exist -- summarization compacted them, or a fresh run is
        reusing this graph (review). Either way the old mark is meaningless, so
        start the key over rather than tag an arbitrary point. Eviction just
        drops the entry: there is nothing server-side to delete, unlike
        _GrowingPrefixCache, so a lost entry costs one re-write and no leak.
        """
        key = _thread_key(request)
        st = self._prefixes.get(key)
        if st is None or len(messages) < st.seen_len:
            st = _AnthropicPrefix()
        self._prefixes[key] = st
        self._prefixes.move_to_end(key)
        while len(self._prefixes) > _GROWING_MAX_STATES:
            self._prefixes.popitem(last=False)
        st.seen_len = len(messages)
        return st

    def _calls_left(self, request) -> float:
        """Model calls left in this turn, THIS ONE INCLUDED, for the end-of-turn
        brake in _advance_pays.

        BudgetMiddleware publishes it directly under CALLS_LEFT_KEY when a turn
        is bounded by dollars: remaining budget over what the last call cost,
        which is the same question ("how many more calls like this one") asked
        in the unit that actually ends the turn. It is unset before the turn's
        first call, and on a stack without that middleware at all.

        The fallback is the call cap: state.model_calls is the count of calls
        already completed (CallBudgetMiddleware increments it in after_model),
        so cap minus count is the calls still permitted, of which the one being
        wrapped is the first. 1 therefore means "this is the last call".

        math.inf when no cap was configured. The brake exists to refuse an
        advance no later call can read back; with no cap there is no horizon to
        compare against, and returning 0 instead would suppress advancement
        permanently rather than merely mistime it -- disabling history caching
        outright for every caller that does not pass max_calls.
        """
        state = getattr(request, "state", None) or {}
        if (left := state.get(CALLS_LEFT_KEY)) is not None:
            return left
        if self._max_calls is None:
            return math.inf
        calls = state.get("model_calls")
        if calls is None:
            return self._max_calls
        return max(0, self._max_calls - calls)

    def _advance(self, st: _AnthropicPrefix, messages: list, request) -> None:
        """Move the history mark forward if it pays. Mutates `st`."""
        if _empty_retry.get():
            # This call carries _EMPTY_NUDGE, appended to the request and not to
            # state, so the tail is not where the next call's history will be.
            # A mark placed into it would cache a prefix nothing else shares.
            # The existing mark still ships: unlike the Vertex cache there is no
            # stored prefix to step around, only a billing annotation.
            st.why = "empty retry"
            return
        target = _last_markable(messages)
        if target == 0:
            # No message in the conversation can carry a mark that survives
            # serialization, so there is nowhere to record a span we would
            # actually send. Warn once per stretch: every shape we have seen
            # ends on a tool result or user text, both markable, so this means
            # something new is in the history.
            if st.why != _NO_SAFE_MARK:
                log.warning(
                    "anthropic cache: nothing in %d messages can carry a mark;"
                    " holding at %d. History caching is stalled.",
                    len(messages),
                    st.boundary,
                )
            st.why = _NO_SAFE_MARK
            return
        if target <= st.boundary:
            # Nothing new behind the mark since it last moved.
            st.why = "no new step"
            return
        if st.boundary == 0:
            # Nothing cached beyond system+tools yet, so there is no prefix to
            # re-buy: the first placement is free and unconditional.
            st.boundary, st.calls_since, st.why = target, 0, "first"
            return
        # Chars, not the provider count, because the delta is the part NOT yet
        # cached -- no usage report covers it.
        delta = (
            sum(len(str(m.content)) for m in messages[st.boundary : target])
            // _CHARS_PER_TOKEN
        )
        due, st.why = _advance_pays(delta, self._calls_left(request))
        if due:
            st.boundary, st.calls_since = target, 0

    def _tagged_history(self, messages: list, boundary: int) -> list:
        """`messages` with a mark closing messages[:boundary].

        Copied, never mutated: the list is graph state and the mark belongs to
        this request only (request.override is per call).

        Where the mark goes is _mark_placement's call, not ours, and _advance
        has already refused to move the boundary onto a message it returns None
        for. Both sides asking the same function is the point: when the choice
        of placement and the decision to advance disagreed, the state recorded
        spans that were never sent.
        """
        i = boundary - 1
        msg = messages[i]
        where = _mark_placement(
            msg,
            messages[i - 1] if i else None,
            messages[i + 1] if i + 1 < len(messages) else None,
        )
        if where is None:
            # _advance should have prevented this. Leave the history alone
            # rather than emit a mark we know will be dropped.
            return messages
        if where is _MARK_IN_KWARGS:
            marked = msg.model_copy(
                update={
                    "additional_kwargs": {
                        **(msg.additional_kwargs or {}),
                        "cache_control": self._cache_control,
                    }
                }
            )
        else:
            blocks = list(msg.content)
            block = blocks[where]
            blocks[where] = (
                {**block, "cache_control": self._cache_control}
                if isinstance(block, dict)
                else {
                    "type": "text",
                    "text": str(block),
                    "cache_control": self._cache_control,
                }
            )
            marked = msg.model_copy(update={"content": blocks})
        return [*messages[:i], marked, *messages[i + 1 :]]

    def _record(self, st: _AnthropicPrefix, response) -> None:
        """Fold the provider's own counts back into the state, and log them.

        read+create is the exact cached span; the check below compares it
        across an advance.

        Debug, not info. This line was info on a call that advanced the mark,
        as the instrument for B -- whether creation bills the whole span or
        only the delta -- on the reasoning that advances were rare enough for
        the noise to be worth it. Neither premise holds now. B is settled
        against the provider's own token meter (cache_write_input came in ~47x
        under what a full re-bill of the span would have required, so creation
        bills the delta), and since the mark stopped stalling every call
        advances, which made this one info line per model call.

        The warning below stays: it is the detector for the mark silently not
        reaching the wire, and that is not something to find in a debug log.
        """
        read, create = _response_cache_stats(response)
        advanced = st.calls_since == 0
        before = st.prefix_tokens
        if read or create:
            st.prefix_tokens = read + create
        log.debug(
            "anthropic cache: read=%d create=%d span=%d mark=%d/%d (%s)",
            read,
            create,
            st.prefix_tokens,
            st.boundary,
            st.seen_len,
            st.why,
        )
        if advanced and before and st.prefix_tokens == before:
            # We moved the mark and the provider's measured span did not grow.
            # That is impossible if the mark arrived: a later cut point covers
            # strictly more. So it was dropped somewhere between here and the
            # wire -- which is exactly how the tool_use placement bug looked,
            # silently, for a full day (span pinned at the system+tools size
            # across every call while the boundary advanced past it).
            #
            # Deliberately compares the span rather than testing create == 0:
            # an advance onto a span that happens to be cached already returns
            # read=span, create=0 and is perfectly healthy.
            log.warning(
                "anthropic cache: advanced to %d/%d but the cached span is"
                " still %d -- the mark is not reaching the wire.",
                st.boundary,
                st.seen_len,
                before,
            )

    async def awrap_model_call(self, request, handler):
        # Lazy import: an install without langchain_google_vertexai should run
        # uncached, not fail to import the agent.
        try:
            from langchain_google_vertexai.model_garden import ChatAnthropicVertex
        except ImportError:
            return await handler(request)
        if not isinstance(request.model, ChatAnthropicVertex):
            return await handler(request)
        # The header goes on every Claude-on-Vertex call; the static mark only
        # when there is a system message to tag.
        overrides = {"model_settings": self._with_session_header(request)}
        blocks = self._blocks(request)
        if blocks is not None:
            overrides["system_message"] = self._tagged_system(blocks)

        messages = request.messages or []
        st = self._state(request, messages)
        st.calls_since += 1
        self._advance(st, messages, request)
        if st.boundary > 0:
            overrides["messages"] = self._tagged_history(messages, st.boundary)

        response = await handler(request.override(**overrides))
        self._record(st, response)
        return response


def base_middleware(
    model_id: str,
    system_prompt: str,
    tools: list,
    *,
    summarizer_model_id: str | None = None,
    grounding_tokens: int = _GROUNDING_REMINDER_TOKENS,
    max_calls: int | None = None,
):
    """The model-call middleware every agent graph shares: ceiling summarization,
    context-window sizing, empty-response stripping, transient-error retry, and
    prompt caching.

    Both agent builders (persona turns in AgentRunner, the commit-review graph
    in processors) compose this with their own control middleware so the cost
    and robustness behavior cannot drift between them. Explicit context caching
    is the caller's job (append _GrowingPrefixCache via growing_cache_middleware)
    since a request references one cached_content and the cache it builds depends
    on the caller's system prompt and tools.

    When summarizer_model_id is given, SummarizationMiddleware is the ceiling
    action: at the hard fraction it compacts old history into a summary (a
    before_model state mutation, so the growing cache sees the shrink and rebuilds
    on the smaller prefix -- compress, re-cache, continue). It runs on the cheap
    model over the whole dropped block (see _SUMMARY_TRIM_TOKENS -- the langchain
    default would summarize only the last 4k of it), and the _ContextBudget shed
    stays a last-resort backstop for when a turn still overflows. Below the
    threshold nothing fires and the prefix cache is never poisoned.
    """
    stack: list = []
    if summarizer_model_id:
        stack.append(
            _SkipOnSummaryFailure(
                model=make_model(summarizer_model_id),
                model_id=summarizer_model_id,
                trigger=("tokens", _SUMMARY_TRIGGER_TOKENS),
                keep=("tokens", _SUMMARY_KEEP_TOKENS),
                trim_tokens_to_summarize=_SUMMARY_TRIM_TOKENS,
            )
        )
    stack += [
        # Outermost of the request shapers so each model call (and its retries)
        # sends a request sized to the context window: at the hard threshold a
        # turn is shed to the last result and forced to wrap up (the backstop
        # when summarization could not bring it under).
        _ContextBudget(),
        # Strip a terminal empty response so it never reaches the checkpoint and
        # poisons the next turn (Gemini rejects empty parts).
        _DropEmptyResponses(),
        # Backoff (~4min worst case per model call, see _RETRY_MAX) outlasts a
        # per-minute 429 quota window, then fails the turn visibly. Only
        # transient provider errors; a zero-part empty candidate is owned by
        # _EmptyCandidateRetry (which mutates the request on its retry), not
        # retried here -- an identical resend reproduces the deterministic empty.
        ModelRetryMiddleware(
            retry_on=_is_retryable,
            on_failure="error",
            max_retries=_RETRY_MAX,
            initial_delay=_RETRY_INITIAL_DELAY,
            backoff_factor=_RETRY_BACKOFF_FACTOR,
            max_delay=_RETRY_MAX_DELAY,
        ),
        # After ModelRetryMiddleware (innermore) so it wraps the cache, whose
        # step-aside its retry depends on. It owns the empty-candidate retry
        # itself (one mutated call, uncached + nudged); the raise that follows
        # is NOT retryable, so the retry layer passes it straight through to
        # the harness rather than resending.
        _EmptyCandidateRetry(),
        # One caching middleware per Anthropic transport: upstream's gates on
        # ChatAnthropic, ours on ChatAnthropicVertex. At most one fires.
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        _AnthropicVertexCaching(max_calls=max_calls),
        # Gemini on Vertex has no request-side cache mark, only the routing key.
        _GeminiVertexSession(),
    ]
    # After summarization in the stack, so its before_model inserts against the
    # post-compaction state (a tail append via the messages reducer -- it settles
    # into the cache, never poisons it).
    if grounding_tokens > 0:
        stack.append(GroundingReminderMiddleware(grounding_tokens))
    return stack


# Vertex bills cache creation at the full input rate and cached reads at a
# fraction of it, so a re-cache re-buys the ENTIRE prefix to move only the tail
# behind the boundary. The Gemini models this was tuned on priced reads at 10%
# of input.
#
# TODO(jgruber): this is a price outside airc_core.pricing, and the only one.
# The explicit Vertex cache has been inactive since 2026-09 (no google_vertexai
# model configured), so it stays as the number it was tuned with. If the path
# is enabled again, derive it from price_for(model_id).rate (cache_read over
# input) at _GrowingPrefixCache construction and check the horizon rule below
# against the listed rates; load_common warns when a config would enable it.
_CACHE_READ_RATIO = 0.1
# The cache serves its FULL prefix (ContextBudget cannot shed inside an immutable
# cache), so cap the cached prefix, and serve uncached if prefix+tail would
# exceed a larger fraction -- otherwise the re-inflated prefix plus a recent tail
# ContextBudget refuses to shed could exceed the window. The total cap leaves
# ContextBudget headroom to shed the (uncached) tail.
_GROWING_MAX_PREFIX = int(CONTEXT_WINDOW * 0.6)
_GROWING_MAX_TOTAL = int(CONTEXT_WINDOW * _HARD_FRACTION)
# Vertex's minimum cacheable size; below it create_context_cache rejects.
_CACHE_FLOOR_TOKENS = 4096
# Bound on live per-conversation cache states (and the server-side caches they
# pin). The least-recently-used is evicted and its cache deleted.
_GROWING_MAX_STATES = 256


def _recache_pays(prefix_tokens: int, delta: int, calls_since: int, calls_left: float):
    """Whether re-caching a `prefix_tokens` prefix to absorb a `delta`-token tail
    is worth its creation cost. Returns (due, reason) -- reason for the log.

    Per call, a cache of size B costs `r*B + (P - B)`: the cached prefix at the
    read rate plus the uncached tail at full rate. Re-caching every k calls with
    the prompt growing g tokens/call therefore averages

        f(k) = B/k + g*k/2 + r*B      (creation, tail growth, reads)

    where only the first two terms depend on k -- there is one live cache of
    ~size B either way, so the read rate and the token-hour storage rate drop out
    of the cadence entirely (they decide only WHETHER to cache at all, which for
    our turn lengths is always yes). Minimizing gives the EOQ square-root law
    k* = sqrt(2B/g), and substituting the measured g = delta/calls_since removes
    the growth estimate, leaving `delta * calls_since >= 2B`.

    A fixed message-count trigger ignored B: on a ~120k prefix growing ~400
    tokens/call it re-cached every 4 calls to recoup over ~24, running ~2.6x the
    cost of never re-caching at all.

    The horizon check is the same trade over the calls actually left in the turn
    rather than the steady state: creation costs B now to save (1-r)*delta on
    each remaining call, so near a turn's end a re-cache can never repay and is
    pure loss.
    """
    if delta <= 0:
        return False, "no growth"
    if delta * calls_since < 2 * prefix_tokens:
        return False, "below payback"
    if calls_left * (1 - _CACHE_READ_RATIO) * delta <= prefix_tokens:
        return False, "turn ending"
    return True, "due"


def _advance_pays(delta: int, calls_left: float) -> tuple[bool, str]:
    """Whether to move an Anthropic breakpoint onto `delta` new tokens.
    Returns (due, reason) -- reason for the log.

    Deliberately not _recache_pays. That rule solves for a cache whose creation
    re-bills the whole prefix, which is what a Vertex cachedContents create
    does. Anthropic does not: measured against claude-opus-5, moving the mark
    one step forward reported read=21230 -- the previous span, still cached and
    served at the read rate -- and create=10856, the new part alone. Reproduced
    over three runs, and again with the mark jumped two steps at once (read
    32086, create 21712), so a mark that lags the tail still finds the older
    prefix.

    So there is no prefix to re-buy and no cadence to optimize. Advancing costs
    the write premium on the delta and saves (1 - read rate) * delta on every
    later call of the turn. At every listed price the premium is below what a
    single read saves (a 5m write is 1.25x input against a read at 0.1x or
    less), so one further call repays it and the rule needs no rate: the only
    losing case is a turn that ends immediately after. `calls_left` counts the
    call being made, so the brake fires when it is 1: the write would be paid
    and nothing would read it. A 1h write (2x) would want the brake one call
    earlier; the TTL is fixed at 5m (_ANTHROPIC_CACHE_TTL).

    The older prefix is found by Anthropic's lookback from the new mark, which
    reaches at most 20 positions back (a run of tool_use blocks is one
    position, a run of tool_result blocks another). Advancing every call keeps
    the jump at a step or two, far inside that. A rule that let the mark lag
    for many steps would fall off the lookback and re-bill the whole span --
    the very cost this rule assumes away.

    This was the one number _AnthropicVertexCaching assumed rather than knew.
    Assuming the worse branch was the right call while it was unmeasured -- the
    error was one-sided -- but it cost most of the benefit: on the reviewed
    commit the rule held the mark for eleven calls after the first placement
    while a re-cache had been due since roughly the eighth.
    """
    if delta <= 0:
        return False, "no growth"
    if calls_left <= 1:
        return False, "turn ending"
    return True, "due"


# Where a cache_control mark has to sit on a message for it to survive
# serialization. _mark_placement returns one of these, or the index of the
# content block to tag, or None for "nowhere safe".
_MARK_IN_KWARGS = "kwargs"

# _AnthropicPrefix.why when the boundary is held back because the step we would
# move onto cannot carry a mark. Also the once-per-stretch warning latch.
_NO_SAFE_MARK = "no safe mark"

# Content blocks langchain_google_vertexai carries cache_control through on,
# verified against _anthropic_utils._format_message_anthropic:
#   text  :193-200 (only when non-empty -- an empty text block is dropped)
#   thinking / redacted_thinking / reasoning  :203-235
# Deliberately not a catch-all. Unknown block types do reach the wire intact
# (:257), but "we have not checked this one" is exactly the reasoning that hid
# the tool_use bug for a day, so an unrecognized block counts as unsafe and the
# boundary waits for a step we can vouch for.
_MARKABLE_BLOCKS = ("thinking", "redacted_thinking", "reasoning")


def _block_takes_a_mark(block) -> bool:
    if isinstance(block, str):
        # _tagged_history promotes a bare string to a text block, which carries
        # the mark; an empty one would be dropped at :198.
        return bool(block.strip())
    if not isinstance(block, dict):
        return False
    if block.get("type") == "text":
        return bool(str(block.get("text", "")).strip())
    return block.get("type") in _MARKABLE_BLOCKS


# TODO(upstream): report this to langchain_google_vertexai. In
# _anthropic_utils._format_message_anthropic, a tool_use content block whose id
# matches one of the message's tool_calls is discarded (:247-255) and the block
# is rebuilt from the tool_call by _lc_tool_call_to_anthropic_tool_use_block
# (:262-265). The rebuilt block copies none of the superseded block's
# keys, so a caller-set cache_control breakpoint disappears with no error and
# no warning -- the request succeeds and simply caches nothing, which is
# indistinguishable from not having cached. A fix would carry cache_control (at
# minimum) across from the block being replaced. Until then _mark_placement
# routes around it by never choosing a tool_use block.
def _mark_placement(msg, prev=None, nxt=None):
    """Where to put a cache_control mark on `msg`, or None if nowhere is safe.

    Returns _MARK_IN_KWARGS, or the index of the content block to tag. `prev`
    and `nxt` are the messages either side of it; a user-role neighbour decides
    one of the cases below.

    This exists because the mark is a passenger inside the request body, and
    langchain_google_vertexai rewrites that body on the way out: it drops keys
    it does not recognize, skips empty text blocks, merges runs of user-role
    messages, and -- the one that bit us -- discards any tool_use block whose id
    matches a tool_call and rebuilds it from the tool_call instead. See the TODO
    above.

    Callers must consult this BEFORE recording that the boundary moved: the
    decision to advance and the ability to mark have to be taken together, or
    the state claims a cached span that was never sent.
    """
    content = msg.content
    if isinstance(msg, ToolMessage):
        # A tool result is formatted into a tool_result block that takes the
        # mark from additional_kwargs (:480-484) -- unless the content already
        # IS tool_result blocks, which takes an earlier branch (:458-468) that
        # never looks at additional_kwargs. The mark rides inside the block, so
        # it survives the merge into a neighbouring user turn.
        already_formatted = (
            isinstance(content, list)
            and content
            and all(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            )
        )
        return None if already_formatted else _MARK_IN_KWARGS
    if isinstance(content, str):
        user_role = (HumanMessage, ToolMessage)
        if isinstance(msg, HumanMessage) and (
            isinstance(prev, user_role) or isinstance(nxt, user_role)
        ):
            # _merge_messages folds a run of user-role messages into one message
            # with LIST content, and the formatter's list branch never looks at
            # additional_kwargs. So a kwargs mark on ANY member of the run is
            # dropped, the same way the tool_use rebuild drops it -- checked
            # against the formatter for both the first and the last of two
            # HumanMessages. Whichever side the run continues on, this message
            # cannot carry the mark; a tool result or assistant text next to the
            # run still can, and a mark inside a content block survives the
            # merge, which is why the list branch below needs no such check.
            return None
        # additional_kwargs is only read for string content (:159-168), and
        # only an actual text block can carry the mark, so empty content has
        # nowhere to put it even though the message is still sent.
        return _MARK_IN_KWARGS if content.strip() else None
    if isinstance(content, list):
        for i in range(len(content) - 1, -1, -1):
            if _block_takes_a_mark(content[i]):
                return i
    return None


def _last_markable(messages: list) -> int:
    """Largest p such that a mark closing messages[:p] reaches the wire, for
    Anthropic. 0 when no message can carry one.

    Anthropic's only placement rule is that the mark sits on a content block the
    client will not rebuild -- _mark_placement. It does not need a conversational
    rest point: the prefix is re-sent in full on every call and the mark only
    says where to cut.

    So this deliberately does NOT reuse _last_step_boundary. That function
    encodes Gemini's wire restrictions, and the shape it selects for a tail of
    [..., AI(tool_use), Tool] is the cut BETWEEN the call and its result -- the
    one message that can never hold a mark. On a turn that makes one tool call
    at a time, which is what a review turn does from start to finish, it
    proposes only that cut and history caching stops for the whole turn.

    p may be len(messages), marking the newest tool result. Measured on Vertex
    against claude-opus-5 over three runs: a mark in a final ToolMessage's
    additional_kwargs writes the prefix (create 21230, read 0) and the next call
    reads it back whole (read 21230, create 0). A mark placed the old way, on
    the assistant turn's tool_use block, wrote nothing at all (create 0) -- the
    control that says these numbers measure the mark and not the weather.

    Marking the final message is safe because the list is append-only from here:
    every middleware that adds a message does it through the state reducer, so
    the message keeps its index next call. The exception is _EMPTY_NUDGE, which
    _EmptyCandidateRetry overrides into the request alone; _advance stands down
    for that call rather than encoding the exception here.
    """
    for p in range(len(messages), 0, -1):
        prev = messages[p - 2] if p > 1 else None
        nxt = messages[p] if p < len(messages) else None
        if _mark_placement(messages[p - 1], prev, nxt) is not None:
            return p
    return 0


def _last_step_boundary(messages: list) -> int:
    """Largest slice point p where caching messages[:p] ends the prefix on a
    wire shape Gemini accepts (_GrowingPrefixCache).

    An AIMessage holding a function call (its ToolMessage then opens the tail,
    which the tool-first guard in model.py makes sendable) or plain user text.
    Never on a ToolMessage: some models (gemini-3.8-flash) reject a cached prefix
    ending on a function response, reporting it as "ending with a model turn".
    A HumanMessage directly after a ToolMessage is no rest-point either --
    consecutive user-role contents merge into one on the wire, so that prefix
    would still end on the function response.

    p is never len(messages): Gemini needs a non-empty tail to send.

    0 when no point qualifies (then the prefix is just [system], i.e. the
    system+tools cache).
    """
    for p in range(len(messages) - 1, 0, -1):
        prev = messages[p - 1]
        if isinstance(prev, AIMessage) and isinstance(messages[p], ToolMessage):
            return p
        if isinstance(prev, HumanMessage) and (
            p < 2 or not isinstance(messages[p - 2], ToolMessage)
        ):
            return p
    return 0


# The cached-content API version the proxy allowlists. Measured against real
# Vertex: v1beta1 serves cachedContents, v1 does not.
_CACHE_API_VERSION = "v1beta1"


def _cache_base(endpoint: str) -> str:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
    return f"{endpoint}/{_CACHE_API_VERSION}/projects/{project}/locations/{location}"


async def _cache_create_rest(endpoint, model_id, prefix, oai_tools, ttl_minutes) -> str:
    """Create cached content over REST, through the sandbox proxy.

    The SDK client cannot be used here: it forces TLS (and, on a corp box, mTLS
    via a device-cert provider), so it cannot reach the plaintext loopback seam
    the chat client uses. The API itself is ordinary REST on the same host, so
    the blocker is the client rather than the endpoint -- we build the request
    proto locally (no network) and post it ourselves.

    Returns the BARE id, not the resource path. ChatVertexAI wraps whatever it is
    given as projects/<p>/locations/<l>/cachedContents/<value>, and the create
    response names the project by NUMBER while the model is built with the
    project NAME -- so returning the full path yields a doubled, mixed-identity
    name and a 400. The SDK hides this by exposing name.split("/")[-1]; this must
    do the same.
    """
    import aiohttp
    from google.protobuf.json_format import MessageToDict
    from langchain_google_vertexai.functions_utils import _format_to_gapic_tool
    from vertexai.caching._caching import _prepare_create_request
    from vertexai.generative_models import Content, Part

    # _prepare_create_request builds the proto locally but still reads project
    # and location off the initializer, so they must be seeded even though
    # nothing here touches the network.
    location = _seed_vertex_cache_globals()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    model = model_id.split(":", 1)[1]
    system, contents = "", []
    for m in prefix:
        if isinstance(m, SystemMessage):
            system = str(m.content)
        else:
            role = "model" if isinstance(m, AIMessage) else "user"
            contents.append(Content(role=role, parts=[Part.from_text(str(m.content))]))
    req = _prepare_create_request(
        f"projects/{project}/locations/{location}/publishers/google/models/{model}",
        system_instruction=system or None,
        contents=contents or None,
        tools=[_format_to_gapic_tool(oai_tools)] if oai_tools else None,
        ttl=timedelta(minutes=ttl_minutes),
    )
    body = MessageToDict(req._pb.cached_content)
    url = f"{_cache_base(endpoint)}/cachedContents"
    async with aiohttp.ClientSession() as s, s.post(url, json=body) as r:
        r.raise_for_status()
        return (await r.json())["name"].rsplit("/", 1)[-1]


async def _cache_delete_rest(endpoint: str, name: str) -> None:
    import aiohttp

    url = f"{_cache_base(endpoint)}/cachedContents/{name.rsplit('/', 1)[-1]}"
    async with aiohttp.ClientSession() as s, s.delete(url) as r:
        r.raise_for_status()


def _genai_client():
    """google-genai client mirroring ChatGoogleGenerativeAI's backend choice:
    a set GOOGLE_CLOUD_PROJECT selects Vertex over ADC (prod), else the
    Developer API via GOOGLE_API_KEY -- the local-verification path, running
    the identical client code against the other backend. The sandbox proxy
    rides base_url with a placeholder credential (the proxy attaches the real
    bearer; see model.proxy_placeholder_credentials).

    The async transport is pinned to httpx: genai prefers aiohttp whenever it
    is importable, and newer releases run client-cert (mTLS) discovery on that
    path -- the machinery GOOGLE_API_USE_CLIENT_CERTIFICATE=false exists to
    keep away from the segfaulting corp device-cert provider. An explicit
    transport is the SDK's own escape hatch, and makes the behavior the same
    across genai versions."""
    import httpx
    from google import genai
    from google.genai import types

    opts: dict = {"async_client_args": {"transport": httpx.AsyncHTTPTransport()}}
    credentials = None
    if endpoint := os.environ.get(_VERTEX_PROXY_ENV):
        from .model import proxy_placeholder_credentials

        opts["base_url"] = endpoint
        credentials = proxy_placeholder_credentials()
    http_options = types.HttpOptions(**opts)
    if proj := os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return genai.Client(
            vertexai=True,
            project=proj,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION") or "global",
            credentials=credentials,
            http_options=http_options,
        )
    return genai.Client(http_options=http_options)


async def _genai_cache_create(model_id, prefix, tools, ttl_minutes) -> str:
    """Create cached content through the google-genai SDK.

    Much less machinery than the vertexai path: the langchain-google-genai
    converter emits google.genai types directly (no proto hand-assembly, no
    aiplatform initializer globals to seed) and the client is natively async.
    Returns the FULL resource name: ChatGoogleGenerativeAI passes
    cached_content through verbatim, so the bare-id dance _cache_create_rest
    documents does not apply here.
    """
    from google.genai import types
    from langchain_google_genai import chat_models as cm
    from langchain_google_genai._function_utils import (
        _format_to_genai_function_declaration,
    )

    name = model_id.split(":", 1)[1]
    system, contents = cm._parse_chat_history(prefix, model=name)
    decls = [_format_to_genai_function_declaration(t) for t in tools]
    # Bind the client to a local for the whole await. genai.Client.__del__ closes
    # the underlying httpx client, and the sub-clients hold the api client but not
    # the Client itself -- so `_genai_client().aio.caches.create(...)` drops the
    # last reference while building the coroutine, and refcounting closes the
    # transport before the request is ever sent ("Cannot send a request, as the
    # client has been closed").
    client = _genai_client()
    cache = await client.aio.caches.create(
        model=name,
        config=types.CreateCachedContentConfig(
            system_instruction=system,
            contents=contents or None,
            tools=[types.Tool(function_declarations=decls)] if decls else None,
            ttl=f"{ttl_minutes * 60}s",
        ),
    )
    return cache.name


async def _genai_cache_delete(name: str) -> None:
    # By bare id, letting the client rebuild the path under ITS project NAME:
    # create returns a name carrying the project NUMBER, and the sandbox
    # proxy's allowlist is anchored to the configured name, so a number-path
    # delete would be refused in the box. The same mixed-identity dance
    # _cache_create_rest documents, at the other end of the lifecycle.
    client = _genai_client()  # a local for the await; see _genai_cache_create
    await client.aio.caches.delete(name=name.rsplit("/", 1)[-1])


def _growing_cache_fns(model_id, tools, ttl_minutes):
    """Return (create, delete, model_for, tools_tokens) for the growing-prefix
    cache.

    create(prefix) caches a [system, ...history] list and returns its name;
    delete(name) removes a superseded cache; model_for(name) yields a model bound
    to it; tools_tokens is a rough token estimate of the tool schemas, which live
    in the cache (not in the messages) and the floor/size math otherwise cannot
    see. Vertex-only (create_context_cache).
    """
    import json

    from langchain_core.utils.function_calling import convert_to_openai_tool

    oai_tools = [convert_to_openai_tool(t) for t in tools]
    # json.dumps, not str(dict): the JSON is closer to what Vertex tokenizes than
    # the Python repr (which inflates with quotes/spaces and would over-count).
    tools_tokens = sum(len(json.dumps(t)) for t in oai_tools) // _CHARS_PER_TOKEN

    async def create(prefix: list) -> str:
        if _google_sdk() == "genai":
            return await _genai_cache_create(model_id, prefix, tools, ttl_minutes)
        from langchain_google_vertexai import create_context_cache

        if endpoint := os.environ.get(_VERTEX_PROXY_ENV):
            return await _cache_create_rest(
                endpoint, model_id, prefix, oai_tools, ttl_minutes
            )
        _seed_vertex_cache_globals()
        return await asyncio.to_thread(
            create_context_cache,
            make_model(model_id),
            prefix,
            tools=oai_tools,
            time_to_live=timedelta(minutes=ttl_minutes),
        )

    async def delete(name: str) -> None:
        if _google_sdk() == "genai":
            await _genai_cache_delete(name)
            return
        from vertexai.preview import caching

        if endpoint := os.environ.get(_VERTEX_PROXY_ENV):
            await _cache_delete_rest(endpoint, name)
            return
        _seed_vertex_cache_globals()
        await asyncio.to_thread(caching.CachedContent(name).delete)

    def model_for(name: str):
        return make_model(model_id, cached_content=name)

    return create, delete, model_for, tools_tokens


@dataclass
class _PrefixState:
    """Per-conversation growing-cache state (keyed by LangGraph thread id)."""

    name: str | None = None
    model: object | None = None
    boundary: int = 0
    prefix_tokens: int = 0
    # Calls this generation has served -- the k in the payback rule, so a cache
    # earns its keep by being read, not by the history happening to grow.
    calls_since: int = 0
    seen_len: int = 0


def _thread_key(request) -> object:
    """The conversation a request belongs to: the LangGraph thread id, or None
    for a checkpointer-less run (review), where one Semaphore(1)-serialized graph
    instance is one logical conversation. execution_info is populated inside a
    running model node but typed Optional, so guard it, and runtime too: this
    runs on every Vertex call, and a missing attribute must mean "no thread",
    not a failed call."""
    runtime = getattr(request, "runtime", None)
    ei = getattr(runtime, "execution_info", None) if runtime else None
    return getattr(ei, "thread_id", None) if ei else None


class _GrowingPrefixCache(AgentMiddleware):
    """Cache a conversation's growing [system + history] prefix and send only the
    uncached tail (the cache supplies the rest; the probe measured ~99%
    cache_read on the tail). One mechanism for both callers:

    - A short turn caches [system] (boundary 0) -- the system+tools cache,
      available from the first call, sending the full history as the tail.
    - A long tool-using turn or a long conversation grows the cached prefix as
      history accrues, re-caching at rest-points where the prefix ends on a
      shape every model accepts (see _last_step_boundary); the tail then opens
      on the tool response answering the cached function call. When to grow is
      a cost decision, not a fixed interval -- see _recache_pays.

    State is per conversation, keyed by thread id, because the persona graph (and
    this middleware) is shared across all of a persona's threads. No locks: the
    orchestrator serializes turns per (thread, agent), so each key's state has a
    single writer at a time; review runs concurrently but passes a per-commit
    thread id (the hash) so each concurrent run gets its own key and single writer
    -- a run keyed to None instead (a direct, serial CommitReview) is safe only
    because nothing else shares that key. The message list shrinking detects a
    fresh run reusing a key. An LRU bound + delete-on-evict caps live caches;
    reactive recovery
    (_is_cache_gone) rebuilds a cache that vanished/expired between turns; an
    instance-level cooldown backs off a permanent create failure. One
    cached_content per request, so this is the sole cache overlay (no _PersonaCache).
    """

    def __init__(
        self,
        create,
        delete,
        model_for,
        system_message,
        tools_tokens,
        *,
        max_calls,
        model_id="",
        ttl_minutes=0,
    ):
        self._create = create
        self._delete = delete
        self._model_for = model_for
        self._system = system_message
        self._tools_tokens = tools_tokens
        # For booking a creation: it is billed as input plus storage over the
        # TTL, and never reaches a model callback. Tests that exercise the
        # cache mechanics alone leave them unset and book nothing priced.
        self._model_id = model_id
        self._ttl_minutes = ttl_minutes
        # The turn's model-call cap, so the horizon check knows how many calls a
        # new cache could still be read over. Mirrors the ModelCallLimitMiddleware
        # run_limit its caller installs.
        self._max_calls = max_calls
        self._states: OrderedDict[object, _PrefixState] = OrderedDict()
        # Monotonic deadline before which no create is attempted. Carries the
        # backoff duration set at failure time (short for transient, long for
        # permanent), so the gate is a single deadline compare.
        self._cooldown_until = -math.inf

    def _prefix_size(self, prefix: list) -> int:
        # Char estimate of the content PLUS the tool schemas (which live in the
        # cache, not the messages), floored by the provider's own previous count
        # (which already counts tools) once a model call has happened.
        chars = sum(len(str(m.content)) for m in prefix) // _CHARS_PER_TOKEN
        return max(chars + self._tools_tokens, _prev_input_tokens(prefix))

    async def _delete_quietly(self, name: str) -> None:
        try:
            await self._delete(name)
        except Exception as e:
            # WARNING, not debug: a delete that fails leaks a cached_content
            # (billable, and -- if it were poisoned -- able to be re-read). The
            # empty-candidate step-aside logs "dropping" before this; pairing a
            # failed delete with a visible warning is how an operator tells a
            # busted delete from a successful one at default log level.
            log.warning(
                "growing cache delete failed (%s: %s); relies on TTL",
                type(e).__name__,
                e,
            )

    async def _evict(self) -> None:
        while len(self._states) > _GROWING_MAX_STATES:
            _, victim = self._states.popitem(last=False)
            if victim.name:
                await self._delete_quietly(victim.name)

    def _calls_left(self, request) -> float:
        """Model calls remaining in this turn, in whichever unit bounds it.

        BudgetMiddleware publishes remaining dollars over the last call's cost
        under CALLS_LEFT_KEY, which is the horizon on a turn the dollar limit
        ends; the call cap is the fallback for a stack without it.

        CallBudgetMiddleware keeps the per-turn count in graph state, and this
        middleware is installed outside it, so the key is visible here. Both are
        UntrackedValues (never checkpointed), so a resumed thread starts a turn
        with them unset -- which lands on the same branch as a graph built
        without either middleware. All of those mean "no count to go on", and the
        safe reading is the full cap: assuming zero calls left would permanently
        suppress re-caching rather than merely mistime it.
        """
        if (left := request.state.get(CALLS_LEFT_KEY)) is not None:
            return left
        calls = request.state.get("model_calls")
        if calls is None:
            return self._max_calls
        return max(0, self._max_calls - calls)

    async def _recache(
        self, st: _PrefixState, prefix: list, target: int, ptok: int, why: str
    ):
        try:
            name = await self._create(prefix)
        except Exception as e:
            # Back off (instance-wide) so the failure does not retry every growth
            # interval. A transient overload -- the prefill storm the cache exists
            # to relieve -- backs off only briefly, so the next turn rebuilds once
            # it clears; a permanent failure (floor/API/IAM) backs off for the long
            # window so a long-lived conversation is not poisoned by inline retries.
            transient = _is_transient(e)
            cooldown = (
                _CACHE_TRANSIENT_COOLDOWN_S if transient else _CACHE_FAIL_COOLDOWN_S
            )
            self._cooldown_until = time.monotonic() + cooldown
            log.warning(
                "growing cache create failed (%s); uncached ~%ds",
                _short_error(e),
                cooldown,
            )
            return
        old = st.name
        st.name, st.boundary = name, target
        st.model, st.prefix_tokens = self._model_for(name), ptok
        # ptok is the estimate the create was sized from; the provider's own
        # count arrives with the first cached call, after the money is spent.
        book_aside(
            Usage.of_cache_creation(self._model_id, ptok, self._ttl_minutes / 60)
        )
        # A new generation restarts the payback clock: the next re-cache must be
        # earned over the calls THIS one serves.
        st.calls_since = 0
        log.info(
            "growing cache gen at boundary %d, %d tokens (%s, %s)",
            target,
            ptok,
            why,
            name,
        )
        if old:
            await self._delete_quietly(old)

    async def awrap_model_call(self, request, handler):
        messages = request.messages
        key = _thread_key(request)
        st = self._states.get(key)
        if st is None:
            st = self._states[key] = _PrefixState()
            await self._evict()
        else:
            self._states.move_to_end(key)

        # An empty candidate's mutated retry (_RETRY_EMPTY): serve this ONE call
        # uncached, so the retry cannot re-read the prefix that may have produced
        # the empty. Step aside rather than delete -- an empty candidate is usually
        # a one-off flake, and tearing the cache down would make a single flake
        # cost a full uncached prefix resend on every later call in the turn.
        #
        # An unparsable tool call retry (_RETRY_UNPARSABLE) also carries an
        # ephemeral nudge that never enters graph state, so it too skips seen_len
        # and recache bookkeeping below (recording seen_len would read as a shrink
        # on the next real call and delete the cache). Unlike an empty candidate,
        # a truncated output stream is not caused by the cached prefix, so it
        # continues serving from st.name below instead of paying an uncached send.
        retry_kind = _empty_retry.get()
        if retry_kind == _RETRY_EMPTY:
            if st.name is not None:
                log.warning(
                    "growing cache: serving uncached past %s for an"
                    " empty-candidate retry; cache kept",
                    st.name,
                )
            return await handler(request)

        if not retry_kind:
            if len(messages) < st.seen_len:
                # History shrank: a fresh run reusing this graph (review). Drop the
                # prior run's cache and start this key over.
                if st.name:
                    await self._delete_quietly(st.name)
                st = self._states[key] = _PrefixState()
            st.seen_len = len(messages)

            st.calls_since += 1
            target = _last_step_boundary(messages)
            if time.monotonic() >= self._cooldown_until:
                prefix = [self._system, *messages[:target]]
                ptok = self._prefix_size(prefix)
                if st.name is None:
                    # No cache yet: the first one is unconditional. It costs one
                    # prefill that this call pays anyway, and every later call reads
                    # it at the discount, so it repays within a few calls at any
                    # plausible read/storage rate.
                    due, why = True, "first"
                else:
                    # prefix_tokens is the provider's exact cache_read after the
                    # first cached call, so the delta is measured, not estimated.
                    due, why = _recache_pays(
                        st.prefix_tokens,
                        ptok - st.prefix_tokens,
                        st.calls_since,
                        self._calls_left(request),
                    )
                if due and _CACHE_FLOOR_TOKENS <= ptok <= _GROWING_MAX_PREFIX:
                    await self._recache(st, prefix, target, ptok, why)

        if st.name is not None and st.boundary < len(messages):
            tail = messages[st.boundary :]
            # Window guard: the cache serves its full (un-sheddable) prefix, so if
            # prefix + tail would exceed the window, step aside and send the full
            # request uncached -- ContextBudget then sheds it normally.
            #
            # Estimate the total as max(prefix + tail chars, the last call's full
            # reported input), NOT prefix_tokens + _estimate_input_tokens(tail):
            # the tail's own usage floor (_prev_input_tokens) is already the last
            # FULL prompt (prefix + tail), so adding prefix_tokens on top double-
            # counts the prefix and wrongly steps aside on large contexts -- the
            # very turns the cache exists for. (The prefix is tools+system, not in
            # the message content, so it cannot come from a char count of tail.)
            tail_chars = sum(len(str(m.content)) for m in tail) // _CHARS_PER_TOKEN
            total = max(st.prefix_tokens + tail_chars, _prev_input_tokens(tail))
            if total <= _GROWING_MAX_TOTAL:
                try:
                    resp = await handler(
                        request.override(model=st.model, messages=tail)
                    )
                except Exception as e:
                    if _is_cache_gone(e, st.name):
                        # Vanished/expired; uncached this call, rebuild next.
                        log.info("growing cache gone (%s); uncached this call", st.name)
                        self._states[key] = _PrefixState(seen_len=st.seen_len)
                        return await handler(request)
                    if _is_transient(e):
                        raise
                    # A permanent error that may be the cached prefix rather than
                    # the request. A model can reject a cached prefix on its shape
                    # alone, in wording that names neither the cache nor its id
                    # ("Requests ending with a model turn are not supported"), and
                    # the tail cannot be what it describes -- a model call is only
                    # ever made with a human or tool turn last. Wording therefore
                    # cannot separate the two, so decide it by experiment: resend
                    # once without the cache. Succeeding proves the prefix was at
                    # fault (drop it, back off, and let the turn continue uncached
                    # rather than die); failing proves the request was, and the
                    # original error stands. Gated to permanent errors because a
                    # transient one is the retry middleware's to handle, and a
                    # full uncached resend is the wrong answer to an overload.
                    try:
                        resp = await handler(request)
                    except Exception:
                        # from None: the probe re-raises the same error the
                        # cached call already reported, and chaining it would
                        # bury the original under a duplicate.
                        raise e from None
                    log.warning(
                        "growing cache %s rejected at serve time; dropped,"
                        " uncached ~%ds: %s",
                        st.name,
                        _CACHE_FAIL_COOLDOWN_S,
                        _short_error(e),
                    )
                    self._cooldown_until = time.monotonic() + _CACHE_FAIL_COOLDOWN_S
                    rejected = st.name
                    self._states[key] = _PrefixState(seen_len=st.seen_len)
                    await self._delete_quietly(rejected)
                    return resp
                # Replace the prefix-size estimate with the provider's exact count
                # so the window guard cannot under-count a token-dense prefix and
                # let the un-sheddable cache overflow on a later, larger tail.
                if actual := _response_cache_read(resp):
                    st.prefix_tokens = actual
                return resp
        return await handler(request)


def growing_cache_middleware(
    model_id: str,
    system_prompt: str,
    tools: list,
    caching_explicit: bool,
    cache_ttl_minutes: int,
    max_calls: int,
):
    """The growing-prefix cache overlay for an agent graph, or None when caching
    is off or the model is not Vertex (create_context_cache is Vertex-only, a
    no-op on the dev google_genai model). Both agent builders append it as the
    sole cache overlay.

    max_calls is the turn's model-call cap (the run_limit of the caller's
    ModelCallLimitMiddleware); the re-cache horizon check needs it to know how
    many calls a new cache could still be read over."""
    if not (caching_explicit and model_id.startswith("google_vertexai:")):
        return None
    create, delete, model_for, tools_tokens = _growing_cache_fns(
        model_id, tools, cache_ttl_minutes
    )
    return _GrowingPrefixCache(
        create,
        delete,
        model_for,
        SystemMessage(system_prompt),
        tools_tokens,
        max_calls=max_calls,
        model_id=model_id,
        ttl_minutes=cache_ttl_minutes,
    )

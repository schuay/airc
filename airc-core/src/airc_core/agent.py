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
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import timedelta
from typing import Annotated, Any

from langgraph.channels.untracked_value import UntrackedValue
from langgraph.constants import TAG_NOSTREAM
from typing_extensions import NotRequired

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    hook_config,
)
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.callbacks import (
    BaseCallbackHandler,
    UsageMetadataCallbackHandler,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import get_buffer_string

from .model import make_model

log = logging.getLogger(__name__)

# The context-window size assumed for every model. Every budget and cache
# threshold below is a fraction of this. We deliberately assume 1M (Gemini, and
# Claude's 1M tier) rather than deriving it per model: the shedders are
# safe-biased and the growing cache measures the real prefix exactly after the
# first cached call, so the slack absorbs the difference for the models we run.
CONTEXT_WINDOW = 1_000_000


def _usage_from_response(response) -> dict:
    """Pull usage_metadata off an LLMResult, via the generation message.

    The aggregating UsageMetadataCallbackHandler sums across the turn and hides
    per-call detail; this reads the single call's own counts so a tracer can log
    the growth curve. Returns {} if the provider attached no usage.
    """
    for gens in getattr(response, "generations", None) or []:
        for gen in gens:
            usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
            if usage:
                return usage
    return {}


class _CallTrace(BaseCallbackHandler):
    """Per-model-call tracer: logs each call's request shape and reported usage.

    The token_usage row sums every call in a turn, so a turn that grew from 8k
    to 240k tokens over fifty tool-calling rounds and one that made a single
    240k-token call look identical there. This logs each call as it returns --
    its input/output/cached tokens next to how many messages and tool results
    the request carried -- making the within-turn growth visible. It also tallies
    the call count and the largest single-call input for the row the caller
    persists. Model calls in a turn are sequential, so the plain counter is safe;
    request shape is keyed by run_id to pair on_chat_model_start with on_llm_end.
    """

    def __init__(self, agent: str, kind: str) -> None:
        self._agent = agent
        self._kind = kind
        self.calls = 0
        self.max_input_tokens = 0
        # Turn totals, so a caller can report the cache hit rate and uncached
        # (full-price) tokens -- the real cost signal once prefix caching is on,
        # since a re-sent tool result behind the cache boundary is a cheap
        # cache_read, not a full re-charge.
        self.total_input = 0
        self.total_output = 0
        self.total_cached = 0
        self._shape: dict[object, tuple[int, int, int]] = {}
        # run_ids of ceiling-summarization calls (tagged lc_source), so their
        # tokens/calls are not booked against this turn -- they run on the cheap
        # filter model and are not part of the persona's work.
        self._skip: set = set()

    def summary(self) -> dict:
        """Turn aggregates plus the derived cache signals."""
        hit = (
            round(100 * self.total_cached / self.total_input) if self.total_input else 0
        )
        return {
            "calls": self.calls,
            "input": self.total_input,
            "output": self.total_output,
            "cached": self.total_cached,
            "uncached": self.total_input - self.total_cached,
            "hit_pct": hit,
            "max_call_input": self.max_input_tokens,
        }

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        if (kwargs.get("metadata") or {}).get("lc_source") == "summarization":
            self._skip.add(run_id)
            return
        msgs = messages[0] if messages else []
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        tool_chars = sum(len(str(m.content)) for m in tool_msgs)
        self._shape[run_id] = (len(msgs), len(tool_msgs), tool_chars)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        # A call that errors (e.g. retried by ModelRetryMiddleware) fires start
        # but not end; drop its shape so the dict does not leak across retries.
        self._shape.pop(run_id, None)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        if run_id in self._skip:
            self._skip.discard(run_id)
            return
        usage = _usage_from_response(response)
        in_tok = int(usage.get("input_tokens", 0))
        out_tok = int(usage.get("output_tokens", 0))
        cached = int(usage.get("input_token_details", {}).get("cache_read", 0))
        self.calls += 1
        self.max_input_tokens = max(self.max_input_tokens, in_tok)
        self.total_input += in_tok
        self.total_output += out_tok
        self.total_cached += cached
        n_msgs, n_tool, tool_chars = self._shape.pop(run_id, (0, 0, 0))
        # Per-call cache hit %, so the growing cache warming over a turn (first
        # call ~0%, later calls high) is visible at a glance -- a flat-low series
        # means the cache is not taking (churn, or a non-Vertex model).
        hit = round(100 * cached / in_tok) if in_tok else 0
        log.info(
            "call %s/%s #%d: %d in (%d cached, %d%%) / %d out; %d msgs, %d tool"
            " results (%d chars)",
            self._agent,
            self._kind,
            self.calls,
            in_tok,
            cached,
            hit,
            out_tok,
            n_msgs,
            n_tool,
            tool_chars,
        )


class TurnUsageHandler(UsageMetadataCallbackHandler):
    """UsageMetadataCallbackHandler that ignores ceiling-summarization calls, so a
    compaction (run on the cheap filter model inside before_model) is not booked
    against the persona's turn token row. Keyed on the lc_source metadata tag, so
    it is robust regardless of whether the nested call inherits the turn's
    callbacks (langchain merges rather than replaces them, so passing callbacks=[]
    to the summarizer would not isolate it)."""

    def __init__(self) -> None:
        super().__init__()
        self._skip: set = set()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        if (kwargs.get("metadata") or {}).get("lc_source") == "summarization":
            self._skip.add(run_id)
            return
        parent = getattr(super(), "on_chat_model_start", None)
        if parent is not None:
            parent(serialized, messages, run_id=run_id, **kwargs)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        if run_id in self._skip:
            self._skip.discard(run_id)
            return
        super().on_llm_end(response, run_id=run_id, **kwargs)


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


def _response_cache_read(response) -> int:
    """The cache_read tokens the model reported for a ModelResponse -- i.e. the
    true size of the cached prefix it just served. 0 if unavailable. Lets the
    serve-time window guard replace its char estimate of the cached prefix with
    the provider's exact count after the first cached call."""
    mr = getattr(response, "model_response", response)  # unwrap ExtendedModelResponse
    msgs = getattr(mr, "result", None)
    if msgs is None:
        msgs = [response] if isinstance(response, AIMessage) else []
    for m in reversed(msgs):
        usage = getattr(m, "usage_metadata", None)
        if usage:
            return int(usage.get("input_token_details", {}).get("cache_read", 0))
    return 0


def _short_error(exc: Exception) -> str:
    """A one-line reason for a model/cache error, for logs.

    Vertex wraps failures in multi-kilobyte RPC dumps (nested original error,
    stack, source-location trace). The signal is the status and a short cause;
    this pulls those out so a transient overload does not print a 3k-char block
    on every occurrence. Falls back to a truncated first line."""
    s = " ".join(str(exc).split())
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
            return f"{type(exc).__name__}: {status}"
    return f"{type(exc).__name__}: {s[:120]}"


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        k in msg
        for k in (
            "429",
            "quota",
            "rate limit",
            "resource_exhausted",
            "exhausted",
            "503",
            "unavailable",
            "overloaded",
            "internal server error",
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
# Six attempts also cover an empty candidate (Gemini's zero-part bug): the first
# retry is a fast 5s re-roll; only a deterministic cache-fault empty exhausts.
_RETRY_MAX = 6
_RETRY_INITIAL_DELAY = 5.0
_RETRY_BACKOFF_FACTOR = 3.0
_RETRY_MAX_DELAY = 60.0


class EmptyCandidateError(Exception):
    """A model call returned a zero-part candidate: no text and no tool calls.

    Gemini's known bug: returns finish_reason=STOP with zero parts -- reads as a
    benign end-of-turn but carries nothing (the silent-dead-turn shape). Raised
    by _EmptyCandidateRetry after inspecting the response, caught by
    ModelRetryMiddleware (via _is_retryable), and retried with the same backoff
    as a 429/5xx. On exhaustion, on_failure="error" re-raises; the harness
    catches it by type and surfaces a named diagnostic instead of a generic
    traceback. Treated as equivalent to HTTP 5xx / empty response.text, per the
    observed provider-flake family.
    """


def _is_retryable(exc: Exception) -> bool:
    """Retry policy for ModelRetryMiddleware: transient provider errors (429/503/
    overloaded) OR a zero-part empty candidate (Gemini's bug). Unifies the flake
    family under one backoff. _is_transient stays string-matched (provider-
    dependent wording) for the bare retrying() helper which never sees empty
    candidates (single-shot calls, no ToolStrategy)."""
    if isinstance(exc, EmptyCandidateError):
        return True
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

    return RunnableLambda(_call)


_ELIDED_TOOL_RESULT = "[tool result from an earlier turn elided to save context]"
# Backstop ceiling on a single intact tool result's characters in a request.
# The eliding pruners always keep the most recent result intact, so one result
# larger than the window overflows the request no matter how much else is shed,
# and every retry 400s. This caps that survivor. Set above the source cap in
# mcptools (_MAX_TOOL_RESULT_CHARS, ~200k) so a normally-capped result passes
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
    """Treat a provider-side empty candidate as a retriable transient error.

    Gemini's known bug: returns a zero-part response (finish_reason=STOP or
    SAFETY/RECITATION, but no content and no tool calls) that reads as a benign
    end-of-turn but carries nothing -- the silent-dead-turn shape. Without this,
    the empty AIMessage sails through (or is scrubbed by _DropEmptyResponses)
    and the turn ends with no report, scored as a dead turn by the reentry loop.
    Raising it as a transient error routes it through the existing
    ModelRetryMiddleware backoff (via _is_retryable), unifying it with
    5xx/429/overloaded -- one retry policy, one failure signal. On exhaustion
    the turn errors visibly (the harness names the empty candidate) instead of
    going silent.

    Detection scope: empty candidate (0 parts: no text AND no tool calls) and
    STOP-with-no-content only -- the flake family. A SAFETY/RECITATION block
    WITH content is a genuine refusal, not a flake; not retried here.

    Note there is no legitimately-empty reply to churn on: an agent with nothing
    to say answers with a sentinel (the room's NOTHING_TO_ADD) or calls its
    report tool, both non-empty. A zero-part candidate is always the bug.

    Sits in front of _DropEmptyResponses, which stays for the request-scrubbing
    half its docstring describes (durable checkpoints poisoned before either
    middleware existed).

    Placement: listed AFTER ModelRetryMiddleware in base_middleware, so the
    retry layer wraps this and catches its raise. The growing cache (appended
    by the harness after base_middleware) sits inside this middleware, so a
    retry resends the same cached request -- cheap on a transient flake, but
    a cache-fault empty (mismatched prefix) exhausts retries against the same
    cache.
    TODO: if wild logs show cache-shaped empties, force uncached on retry via
    the growing cache's step-aside path. For now we accept the risk -- the
    failure is at least named, not silent.
    """

    async def awrap_model_call(self, request, handler):
        resp = await handler(request)
        for msg in resp.result:
            if not isinstance(msg, AIMessage):
                continue
            tool_calls = msg.tool_calls or []
            content = str(msg.content or "").strip()
            if not tool_calls and not content:
                meta = msg.response_metadata or {}
                reason = meta.get("finish_reason", "")
                raise EmptyCandidateError(
                    f"empty candidate (finish_reason={reason or 'unknown'})"
                )
        return resp


class _CallBudgetState(AgentState):
    # UntrackedValue: per-turn (per graph invocation), NEVER checkpointed -- so on
    # a checkpointed persona graph the count resets each turn instead of
    # accumulating across the conversation's lifetime.
    model_calls: NotRequired[Annotated[int, UntrackedValue]]


class CallBudgetMiddleware(AgentMiddleware):
    """Steer a long tool-using turn toward converging, via escalating wrap-up
    nudges at given model-call counts.

    Counts model calls per turn and, at each scheduled threshold, appends a
    wrap-up instruction to that one model request -- NOT to graph state, so on a
    checkpointed persona graph the nudge never persists into the conversation
    (which would bake "stop using tools" into every future turn). The hard cap
    itself is ModelCallLimitMiddleware; these nudges only get the model to wrap
    up before it. The schedule is caller-supplied -- a persona reply uses two
    thresholds, a commit review a longer escalation -- so both share the
    mechanism. List the final nudge a few calls below the cap and make it
    insistent (e.g. "produce your result now"), so the run rarely ends with
    nothing to show.
    """

    state_schema = _CallBudgetState

    def __init__(self, stages: list[tuple[int, str]]) -> None:
        super().__init__()
        # threshold -> nudge; each fires once (model_calls steps by 1 per call,
        # so an exact-equality lookup hits each threshold exactly once).
        self._stages = dict(stages)

    def after_model(self, state, runtime) -> dict[str, Any]:
        return {"model_calls": state.get("model_calls", 0) + 1}

    async def aafter_model(self, state, runtime) -> dict[str, Any]:
        return self.after_model(state, runtime)

    async def awrap_model_call(self, request, handler):
        # state.model_calls is the count of calls already completed this turn.
        # Append the nudge to this request only (ephemeral), never to state.
        n = request.state.get("model_calls", 0)
        nudge = self._stages.get(n)
        if nudge is not None:
            log.info("call budget: wrap-up nudge at %d model calls", n)
            request = request.override(
                messages=[*request.messages, HumanMessage(nudge)]
            )
        return await handler(request)


# Tag on the re-ask reminders this middleware injects. No longer load-bearing
# for the bound (the re-ask count is an UntrackedValue counter, below) -- kept
# so a reminder is identifiable in logs/checkpoints and a test can assert it was
# injected, the same self-tagging scheme GroundingReminderMiddleware uses.
_REQUIRE_RESULT_SRC = "require_result_reask"


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
        # Only a terminal plain-text answer qualifies. An AIMessage carrying tool
        # calls is an intermediate step (read tools, or a pending structured call)
        # the loop handles itself; a delivered verdict sets structured_response.
        # A failed structured call leaves a ToolMessage last (handle_errors
        # re-prompts) -- also not ours. So: last is an AIMessage, no tool calls,
        # and no structured_response yet.
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        if state.get("structured_response") is not None:
            return None
        n = state.get("reasks", 0)
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
        return {"jump_to": "model", "reasks": n + 1, "messages": [reminder]}

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
    """Insert the grounding rule into the conversation once per `interval` tokens of
    context growth, so a long thread keeps a recent copy near the working tail
    rather than only at the (increasingly buried) system prompt.

    Writes the reminder into graph state via the messages reducer -- a TAIL append,
    never a mid-history insert (which would rewrite and poison the cached prefix).
    So it settles into the growing prefix cache like any other message and costs a
    cache-read, not a full re-send, thereafter. Self-tracked: it measures tokens
    since the last reminder in the current message list, so it re-arms on its own
    after a summarization compaction drops earlier reminders (no absolute counter
    to reset). 0 disables it.
    """

    def __init__(self, interval: int, reminder: str = _GROUNDING_REMINDER) -> None:
        super().__init__()
        self._interval = interval
        self._reminder = reminder

    @staticmethod
    def _is_reminder(m) -> bool:
        return (
            isinstance(m, HumanMessage)
            and m.additional_kwargs.get("lc_source") == _GROUNDING_SRC
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
                        additional_kwargs={"lc_source": _GROUNDING_SRC},
                    )
                ]
            }
        return None

    async def abefore_model(self, state, runtime):
        return self.before_model(state, runtime)


def _set_vertex_cache_region() -> str:
    """Pin the aiplatform initializer to the model's serving region and return
    it. CachedContent.create reads the region from the initializer (default
    us-central1), not the model, and a cache must share the model's serving
    region. Set it directly -- aiplatform.init() would too but eagerly builds a
    client that imports pyOpenSSL (absent in some envs). Pre-set the project too
    so reading global_config.project does not trigger the SDK's lazy project-id
    normalization (an otherwise-needless Cloud Resource Manager projects.get).
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
    caller ever appears, override _create_summary/before_model the same way."""

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
        # of the message stream at the source; lc_source keeps it out of the
        # usage/trace books.
        response = await self.model.ainvoke(
            self.summary_prompt.format(messages=formatted).rstrip(),
            config={
                "tags": [TAG_NOSTREAM],
                "metadata": {"lc_source": "summarization"},
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


def base_middleware(
    model_id: str,
    system_prompt: str,
    tools: list,
    *,
    summarizer_model_id: str | None = None,
    grounding_tokens: int = _GROUNDING_REMINDER_TOKENS,
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
        # per-minute 429 quota window, then fails the turn visibly. _is_retryable
        # extends _is_transient with EmptyCandidateError so a zero-part Gemini
        # response retries through the same path as a 429.
        ModelRetryMiddleware(
            retry_on=_is_retryable,
            on_failure="error",
            max_retries=_RETRY_MAX,
            initial_delay=_RETRY_INITIAL_DELAY,
            backoff_factor=_RETRY_BACKOFF_FACTOR,
            max_delay=_RETRY_MAX_DELAY,
        ),
        # After ModelRetryMiddleware (innermore) so its raise is caught and
        # retried. Inspects each response; raises EmptyCandidateError on a
        # zero-part candidate, letting the retry layer treat it as transient.
        _EmptyCandidateRetry(),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
    ]
    # After summarization in the stack, so its before_model inserts against the
    # post-compaction state (a tail append via the messages reducer -- it settles
    # into the cache, never poisons it).
    if grounding_tokens > 0:
        stack.append(GroundingReminderMiddleware(grounding_tokens))
    return stack


# Re-cache the growing prefix once the message list has grown by this many
# messages (~2 per ReAct step) since the last cache, so each generation is read
# several calls before it is superseded. Smaller = more cache writes; larger = a
# longer uncached tail re-sent each call.
_GROWING_RECACHE_GROWTH = 8
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


def _last_step_boundary(messages: list) -> int:
    """Largest index b where messages[b] begins a completed ReAct step: an
    AIMessage preceded by a tool response or the initial human turn. Caching
    messages[:b] leaves a tail [ai, tool, ...] that starts on a model turn (a
    request may not begin with a tool response). 0 before any model turn (then
    the prefix is just [system], i.e. the system+tools cache)."""
    for i in range(len(messages) - 1, 0, -1):
        if isinstance(messages[i], AIMessage) and isinstance(
            messages[i - 1], (ToolMessage, HumanMessage)
        ):
            return i
    return 0


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
        from langchain_google_vertexai import create_context_cache

        _set_vertex_cache_region()
        return await asyncio.to_thread(
            create_context_cache,
            make_model(model_id),
            prefix,
            tools=oai_tools,
            time_to_live=timedelta(minutes=ttl_minutes),
        )

    async def delete(name: str) -> None:
        from vertexai.preview import caching

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
    attempt_len: int = 0
    seen_len: int = 0


def _thread_key(request) -> object:
    """The conversation a request belongs to: the LangGraph thread id, or None
    for a checkpointer-less run (review), where one Semaphore(1)-serialized graph
    instance is one logical conversation. execution_info is populated inside a
    running model node but typed Optional, so guard it."""
    ei = getattr(request.runtime, "execution_info", None)
    return getattr(ei, "thread_id", None) if ei else None


class _GrowingPrefixCache(AgentMiddleware):
    """Cache a conversation's growing [system + history] prefix and send only the
    uncached tail (the cache supplies the rest; the probe measured ~99%
    cache_read on the tail). One mechanism for both callers:

    - A short turn caches [system] (boundary 0) -- the system+tools cache,
      available from the first call, sending the full history as the tail.
    - A long tool-using turn or a long conversation grows the cached prefix as
      history accrues, re-caching at ReAct step rest-points so the tail always
      starts on a model turn (a request may not begin with a tool response).

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
        growth=_GROWING_RECACHE_GROWTH,
    ):
        self._create = create
        self._delete = delete
        self._model_for = model_for
        self._system = system_message
        self._tools_tokens = tools_tokens
        self._growth = growth
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
            log.debug(
                "growing cache delete failed (%s: %s); relies on TTL",
                type(e).__name__,
                e,
            )

    async def _evict(self) -> None:
        while len(self._states) > _GROWING_MAX_STATES:
            _, victim = self._states.popitem(last=False)
            if victim.name:
                await self._delete_quietly(victim.name)

    async def _recache(self, st: _PrefixState, prefix: list, target: int, ptok: int):
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
        log.info("growing cache gen at boundary %d (%s)", target, name)
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

        if len(messages) < st.seen_len:
            # History shrank: a fresh run reusing this graph (review). Drop the
            # prior run's cache and start this key over.
            if st.name:
                await self._delete_quietly(st.name)
            st = self._states[key] = _PrefixState()
        st.seen_len = len(messages)

        target = _last_step_boundary(messages)
        due = st.name is None or len(messages) - st.attempt_len >= self._growth
        if due and time.monotonic() >= self._cooldown_until:
            prefix = [self._system, *messages[:target]]
            ptok = self._prefix_size(prefix)
            if _CACHE_FLOOR_TOKENS <= ptok <= _GROWING_MAX_PREFIX:
                st.attempt_len = len(messages)
                await self._recache(st, prefix, target, ptok)

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
                    if not _is_cache_gone(e, st.name):
                        raise
                    # Vanished/expired; uncached this call, rebuild next.
                    log.info("growing cache gone (%s); uncached this call", st.name)
                    self._states[key] = _PrefixState(seen_len=len(messages))
                    return await handler(request)
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
):
    """The growing-prefix cache overlay for an agent graph, or None when caching
    is off or the model is not Vertex (create_context_cache is Vertex-only, a
    no-op on the dev google_genai model). Both agent builders append it as the
    sole cache overlay."""
    if not (caching_explicit and model_id.startswith("google_vertexai:")):
        return None
    create, delete, model_for, tools_tokens = _growing_cache_fns(
        model_id, tools, cache_ttl_minutes
    )
    return _GrowingPrefixCache(
        create, delete, model_for, SystemMessage(system_prompt), tools_tokens
    )

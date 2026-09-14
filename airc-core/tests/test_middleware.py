# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""base_middleware (shared cost/robustness stack) and the growing-cache overlay.

The drift guard: every agent graph gets context-window sizing, empty-response
stripping, retry, and Anthropic caching from base_middleware; the explicit
context cache is the caller-appended growing-prefix overlay, gated to Vertex.
"""

import contextlib
from types import SimpleNamespace

import pytest
from airc_core.agent import (
    CallBudgetMiddleware,
    EmptyCandidateError,
    RequireStructuredResultMiddleware,
    _DropEmptyResponses,
    _SkipOnSummaryFailure,
    base_middleware,
    growing_cache_middleware,
    retrying,
)
from langchain.agents.middleware import (
    ModelRequest,
    ModelRetryMiddleware,
    SummarizationMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

_NON_VERTEX = "google_genai:gemini-3.1-flash-lite"
_VERTEX = "google_vertexai:gemini-2.5-flash"


def _names(middleware):
    return [type(m).__name__ for m in middleware]


@pytest.fixture
def middleware_with_summarizer(monkeypatch):
    """Build the stack without asking a real provider for credentials."""
    from langchain_core.language_models import GenericFakeChatModel

    model = GenericFakeChatModel(messages=iter([]))
    monkeypatch.setattr("airc_core.agent.make_model", lambda _model_id: model)
    return base_middleware(_NON_VERTEX, "s", [], summarizer_model_id=_NON_VERTEX)


# ── retrying() bare-model wrapper ────────────────────────────────────────────


class _FlakyModel:
    """Chat model stand-in whose ainvoke raises a scripted sequence, then a
    value. Records how many times it was called."""

    def __init__(self, errors):
        self._errors = list(errors)
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return AIMessage("ok")


async def test_retrying_retries_transient_then_succeeds(monkeypatch):
    from airc_core import agent

    monkeypatch.setattr(agent.asyncio, "sleep", _noop_sleep)
    model = _FlakyModel(
        [RuntimeError("429 resource_exhausted"), RuntimeError("503 unavailable")]
    )
    out = await retrying(model).ainvoke("hi")
    assert out.text == "ok"
    assert model.calls == 3  # two transient failures, then success


async def test_retrying_does_not_retry_non_transient(monkeypatch):
    from airc_core import agent

    monkeypatch.setattr(agent.asyncio, "sleep", _noop_sleep)
    model = _FlakyModel([ValueError("malformed request")])
    try:
        await retrying(model).ainvoke("hi")
        raised = False
    except ValueError:
        raised = True
    assert raised
    assert model.calls == 1  # propagated immediately, no retry


async def test_retrying_propagates_after_exhausting(monkeypatch):
    from airc_core import agent

    monkeypatch.setattr(agent.asyncio, "sleep", _noop_sleep)
    # Always transient: exhausts the budget and re-raises the last error.
    model = _FlakyModel([RuntimeError("overloaded")] * 99)
    try:
        await retrying(model).ainvoke("hi")
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    assert model.calls == agent._RETRY_MAX + 1  # initial try + _RETRY_MAX retries


async def _noop_sleep(_):
    return None


def test_is_transient_decides_on_structured_code_when_present():
    from airc_core.agent import _is_transient
    from google.api_core import exceptions as gexc
    from google.genai import errors as generr

    # The finding-7 regression: digits inside a request id or token count no
    # longer launder a permanent 400 into a transient, whatever the text says.
    dump = "model turn not supported; request_id=4297bd11 prompt_tokens=8503"
    assert not _is_transient(gexc.InvalidArgument(dump))
    assert not _is_transient(generr.APIError(400, {"error": {"message": dump}}))
    # Both stacks' real transients decide on the code alone.
    assert _is_transient(gexc.ResourceExhausted("busy"))
    assert _is_transient(gexc.ServiceUnavailable("overloaded"))
    assert _is_transient(gexc.DeadlineExceeded("hung stream reaped"))
    assert _is_transient(generr.APIError(429, {"error": {"message": "quota"}}))
    assert _is_transient(generr.APIError(503, {"error": {"message": "busy"}}))


def test_is_transient_knows_the_anthropic_sdk_error_shapes():
    """529 carries a status_code, so the code set must name it; the "overloaded"
    word is never reached. A timeout or dropped connection has no code."""
    import httpx
    from airc_core.agent import _is_transient
    from anthropic import (
        APIConnectionError,
        APITimeoutError,
        BadRequestError,
        OverloadedError,
    )

    req = httpx.Request("POST", "http://x")

    def status(cls, code):
        return cls("boom", response=httpx.Response(code, request=req), body=None)

    assert _is_transient(status(OverloadedError, 529))
    assert not _is_transient(status(BadRequestError, 400))
    assert _is_transient(APITimeoutError(request=req))
    assert _is_transient(APIConnectionError(request=req))


def test_is_transient_and_short_error_follow_the_cause_chain():
    # The genai stack re-raises API errors as a plain wrapper with the
    # structured original chained underneath; both helpers must see through it.
    from airc_core.agent import _is_transient, _short_error
    from google.genai import errors as generr

    def wrapped(code, status, message):
        cause = generr.APIError(code, {"error": {"message": message, "status": status}})
        exc = RuntimeError(f"Error calling model 'm' ({status}): {cause}")
        exc.__cause__ = cause
        return exc

    assert _is_transient(wrapped(429, "RESOURCE_EXHAUSTED", "quota"))
    exc = wrapped(400, "INVALID_ARGUMENT", "model turn; request_id=4297bd11")
    assert not _is_transient(exc)
    out = _short_error(exc)
    assert out.startswith("RuntimeError: INVALID_ARGUMENT: ")
    assert "model turn" in out and "4297bd11" in out


def test_is_transient_text_fallback_needs_words_not_digits():
    from airc_core.agent import _is_transient

    assert _is_transient(RuntimeError("rate limit exceeded, retry later"))
    assert _is_transient(RuntimeError("the model is overloaded"))
    assert not _is_transient(RuntimeError("request 4297bd11 failed writing 8503"))


def test_short_error_extracts_status_from_verbose_dump():
    from airc_core.agent import _SHORT_ERROR_CAUSE_CHARS, _short_error

    huge = RuntimeError("500 RpcClientException ... RESOURCE_EXHAUSTED " + "x" * 3000)
    out = _short_error(huge)
    prefix = "RuntimeError: RESOURCE_EXHAUSTED: "
    assert out.startswith(prefix)
    # Never the 3k dump: the cap bounds the cause, and the cut is marked.
    assert len(out) == len(prefix) + _SHORT_ERROR_CAUSE_CHARS + len("...")
    assert out.endswith("...")
    # No known status: falls back to a truncated first line, still bounded.
    other = _short_error(RuntimeError("something unusual happened " * 50))
    assert len(other) == len("RuntimeError: ") + 120 + len("...")
    assert other.endswith("...")


def test_short_error_keeps_the_cause_of_a_permanent_400():
    """A bare status cannot diagnose a 400 -- the reason is the whole signal, and
    every 400 report (cache create, lost pass, re-dispatch) logs through here."""
    from airc_core.agent import _SHORT_ERROR_CAUSE_CHARS, _short_error
    from google.api_core.exceptions import InvalidArgument

    exc = InvalidArgument(
        "Requests ending with a model turn are not supported."
        " RPC to BAG server failed: INVALID_ARGUMENT: " + "x" * 3000
    )
    out = _short_error(exc)
    prefix = "InvalidArgument: INVALID_ARGUMENT: "
    assert out.startswith(prefix)
    assert "Requests ending with a model turn are not supported." in out
    assert len(out) == len(prefix) + _SHORT_ERROR_CAUSE_CHARS + len("...")


class _Req:
    """Minimal ModelRequest stand-in for the call-budget middleware: a per-turn
    model_calls count, messages, and override(messages=)."""

    def __init__(self, model_calls: int, messages=None):
        self.state = {"model_calls": model_calls}
        self.messages = messages or [HumanMessage("hi")]

    def override(self, *, messages):
        return _Req(self.state["model_calls"], messages)


async def _appended(mw, n):
    """The messages the handler saw for a request at call-count n -- i.e. the
    request after any nudge the middleware appended."""
    seen = {}

    async def handler(req):
        seen["msgs"] = req.messages
        return "ok"

    await mw.awrap_model_call(_Req(n), handler)
    return [str(m.content) for m in seen["msgs"]]


def test_call_budget_counts_each_call():
    mw = CallBudgetMiddleware([(2, "x")])
    assert mw.after_model({"model_calls": 7}, None) == {"model_calls": 8}
    assert mw.after_model({}, None) == {"model_calls": 1}


async def test_call_budget_nudges_fire_only_on_schedule():
    mw = CallBudgetMiddleware([(2, "converge now"), (5, "stop now")])
    # Off a threshold: nothing appended.
    assert not any("converge" in m or "stop" in m for m in await _appended(mw, 1))
    assert not any("converge" in m or "stop" in m for m in await _appended(mw, 3))
    # Each threshold appends its own nudge to that one request.
    assert any("converge now" in m for m in await _appended(mw, 2))
    assert any("stop now" in m for m in await _appended(mw, 5))


def test_shared_stack_present_for_any_model():
    # base_middleware carries no cache overlay -- that is the caller's job.
    # Both Anthropic caching middlewares are always listed; they gate on
    # mutually exclusive client types, so at most one acts on a given request.
    assert _names(base_middleware(_NON_VERTEX, "sys", [])) == [
        "_ContextBudget",
        "_DropEmptyResponses",
        "ModelRetryMiddleware",
        "_EmptyCandidateRetry",
        "AnthropicPromptCachingMiddleware",
        "_AnthropicVertexCaching",
        "GroundingReminderMiddleware",  # inner to _ContextBudget, on by default
    ]


# ── _AnthropicVertexCaching (Claude on Vertex) ───────────────────────────────
#
# The upstream middleware gates on isinstance(model, ChatAnthropic), which
# ChatAnthropicVertex is not, so under "ignore" caching silently stops. The
# stack assertion above passes either way; these assert the breakpoint reaches
# the request.


def _CacheReq(
    model, system_message, model_settings=None, thread_id="t1", messages=None
):
    """A real ModelRequest with a stubbed runtime, whose execution_info carries
    the thread id the session header is keyed on."""
    return ModelRequest(
        model=model,
        messages=[HumanMessage("hi")] if messages is None else messages,
        system_message=system_message,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={},
        runtime=SimpleNamespace(
            execution_info=(
                SimpleNamespace(thread_id=thread_id) if thread_id is not None else None
            )
        ),
        model_settings=model_settings or {},
    )


def _usage(read=0, create=0):
    """An AIMessage reporting cache counters, as a real response does."""
    return AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": read + create,
            "output_tokens": 1,
            "total_tokens": read + create + 1,
            "input_token_details": {"cache_read": read, "cache_creation": create},
        },
    )


async def _delivered(
    model,
    system_message,
    model_settings=None,
    thread_id="t1",
    mw=None,
    messages=None,
    response="ok",
):
    """The (system_message, model_settings, messages) the handler saw. `mw`
    reuses one middleware instance across calls, as base_middleware does per
    agent."""
    from airc_core.agent import _AnthropicVertexCaching

    seen = {}

    async def handler(req):
        seen["system"] = req.system_message
        seen["settings"] = req.model_settings
        seen["messages"] = req.messages
        return response

    await (mw or _AnthropicVertexCaching()).awrap_model_call(
        _CacheReq(model, system_message, model_settings, thread_id, messages), handler
    )
    return seen


def _fake_vertex_anthropic():
    """A real ChatAnthropicVertex without credentials; only its type matters."""
    pytest.importorskip("langchain_google_vertexai")
    from langchain_google_vertexai.model_garden import ChatAnthropicVertex

    return ChatAnthropicVertex.model_construct()


async def test_anthropic_vertex_caching_tags_the_system_message():
    seen = await _delivered(_fake_vertex_anthropic(), SystemMessage("you are helpful"))
    blocks = seen["system"].content
    # A str system prompt is promoted to one text block carrying the breakpoint.
    assert blocks == [
        {
            "type": "text",
            "text": "you are helpful",
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        }
    ]


async def test_anthropic_vertex_caching_sets_no_tail_breakpoint():
    """No cache_control in model_settings: that tags the last content block,
    a per-turn budget nudge, so the breakpoint would move every call."""
    seen = await _delivered(_fake_vertex_anthropic(), SystemMessage("sys"))
    assert "cache_control" not in seen["settings"]
    assert set(seen["settings"]) == {"extra_headers"}


async def test_anthropic_vertex_caching_ignores_other_providers():
    """Other providers pass through untouched: Vertex rejects an unknown
    cache_control key, and ChatAnthropic is upstream's job."""
    from langchain_core.language_models import GenericFakeChatModel

    seen = await _delivered(
        GenericFakeChatModel(messages=iter([])), SystemMessage("sys")
    )
    assert seen["system"].content == "sys"
    assert seen["settings"] == {}


async def test_anthropic_vertex_caching_passes_through_without_a_system_message():
    seen = await _delivered(_fake_vertex_anthropic(), None)
    assert seen["system"] is None


# ── session affinity ─────────────────────────────────────────────────────────
#
# One conversation keeps one id, separate conversations get separate ones, and
# a header somebody else set survives.

_SESSION_HEADER = "X-Vertex-Ai-Session-Id"


async def _session_of(system_message=None, **kw):
    seen = await _delivered(
        _fake_vertex_anthropic(), system_message or SystemMessage("sys"), **kw
    )
    return seen["settings"]["extra_headers"][_SESSION_HEADER]


async def test_session_id_is_stable_across_turns_of_one_conversation():
    """Turn two must carry the id turn one cached under."""
    assert await _session_of(thread_id="thread-a") == await _session_of(
        thread_id="thread-a"
    )


async def test_session_id_differs_between_conversations():
    """One id for every conversation is the pile-up the per-conversation key
    exists to avoid."""
    assert await _session_of(thread_id="thread-a") != await _session_of(
        thread_id="thread-b"
    )


async def test_session_id_does_not_leak_the_thread_id():
    """A hex digest, not the internal id."""
    sid = await _session_of(thread_id="thread-a")
    assert "thread-a" not in sid
    assert len(sid) == 32
    assert all(c in "0123456789abcdef" for c in sid)


async def test_session_header_is_sent_without_a_system_message():
    """Affinity does not depend on this turn having anything to cache."""
    seen = await _delivered(_fake_vertex_anthropic(), None)
    assert seen["settings"]["extra_headers"][_SESSION_HEADER]


async def test_session_header_preserves_headers_the_caller_already_set():
    seen = await _delivered(
        _fake_vertex_anthropic(),
        SystemMessage("sys"),
        model_settings={"extra_headers": {"X-Other": "keep"}, "temperature": 0},
    )
    headers = seen["settings"]["extra_headers"]
    assert headers["X-Other"] == "keep"
    assert headers[_SESSION_HEADER]
    # The rest of model_settings survives the merge too.
    assert seen["settings"]["temperature"] == 0


async def test_session_id_falls_back_when_there_is_no_thread():
    """A graph that configures no thread id (a direct, serial CommitReview, a
    test) is one conversation per middleware instance, and two instances must
    not collide."""
    from airc_core.agent import _AnthropicVertexCaching

    async def sid_from(mw):
        seen = {}

        async def handler(req):
            seen["settings"] = req.model_settings
            return "ok"

        await mw.awrap_model_call(
            _CacheReq(_fake_vertex_anthropic(), SystemMessage("sys"), thread_id=None),
            handler,
        )
        return seen["settings"]["extra_headers"][_SESSION_HEADER]

    one, other = _AnthropicVertexCaching(), _AnthropicVertexCaching()
    assert await sid_from(one) == await sid_from(one)
    assert await sid_from(one) != await sid_from(other)


async def test_claude_and_gemini_can_share_one_deployment():
    """A mixed deploy (Gemini for the room, Claude for a long flow) shares one
    base_middleware instance; the type gate keeps them apart. Uses the real
    google-genai class because that is what the gate must hold against."""
    pytest.importorskip("langchain_google_genai")
    from airc_core.agent import _AnthropicVertexCaching
    from langchain_google_genai import ChatGoogleGenerativeAI

    gemini = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key="x")
    claude = _fake_vertex_anthropic()
    # One instance for every call; a per-call one would not exercise sharing.
    mw = _AnthropicVertexCaching()

    for _ in range(2):  # interleaved, on one middleware instance
        g = await _delivered(gemini, SystemMessage("sys"), mw=mw)
        c = await _delivered(claude, SystemMessage("sys"), mw=mw)
        # Gemini is untouched: Vertex rejects an unknown cache_control key.
        assert g["settings"] == {} and g["system"].content == "sys"
        assert _SESSION_HEADER in c["settings"]["extra_headers"]


# ── the advancing history breakpoint ─────────────────────────────────────────
#
# The static mark covers system+tools only, ~10% of a long prompt. These pin the
# second, advancing mark: where it lands, and that it moves on the payback rule
# rather than on every call (moving it every call re-buys the whole prefix).

_CC = {"type": "ephemeral", "ttl": "5m"}


def _step(n, content=None):
    """One completed tool step: the model calls a tool, the tool answers."""
    return [
        AIMessage(
            content=content if content is not None else f"calling {n}",
            tool_calls=[{"name": "t", "args": {}, "id": f"c{n}"}],
        ),
        ToolMessage(content=f"result {n}", tool_call_id=f"c{n}"),
    ]


def _marks(messages):
    """Indices of messages carrying a cache_control mark, either placement."""
    out = []
    for i, m in enumerate(messages):
        block = isinstance(m.content, list) and any(
            isinstance(b, dict) and "cache_control" in b for b in m.content
        )
        if block or (m.additional_kwargs or {}).get("cache_control"):
            out.append(i)
    return out


async def test_no_history_mark_before_a_completed_step():
    """A first call has nothing behind it worth closing, so only the static
    mark ships. Placing one anyway would cache a prefix no later call shares."""
    seen = await _delivered(_fake_vertex_anthropic(), SystemMessage("sys"))
    assert _marks(seen["messages"]) == []


async def test_history_mark_closes_the_last_completed_step():
    """It lands on the AIMessage that opened the step, never on the trailing
    user turn -- which may be a per-turn nudge that moves every call."""
    messages = [HumanMessage("q"), *_step(1), HumanMessage("next")]
    seen = await _delivered(
        _fake_vertex_anthropic(), SystemMessage("sys"), messages=messages
    )
    assert _marks(seen["messages"]) == [1]
    # str content takes the additional_kwargs path (_get_cache_control).
    assert seen["messages"][1].additional_kwargs["cache_control"] == _CC
    # The originals are untouched: the list is graph state.
    assert _marks(messages) == []


async def test_history_mark_on_block_content_goes_in_the_block():
    """List content is read from the block itself, not additional_kwargs, so
    the placement has to follow the content type or the mark is dropped."""
    messages = [
        HumanMessage("q"),
        *_step(1, content=[{"type": "text", "text": "calling 1"}]),
        HumanMessage("next"),
    ]
    seen = await _delivered(
        _fake_vertex_anthropic(), SystemMessage("sys"), messages=messages
    )
    assert seen["messages"][1].content[-1]["cache_control"] == _CC
    assert "cache_control" not in (seen["messages"][1].additional_kwargs or {})


async def test_history_mark_holds_until_the_delta_repays():
    """Advancing re-buys the whole prefix, so a mark that chased every step
    would cost more than caching nothing. It moves only when
    delta * calls_since >= 2 * prefix."""
    from airc_core.agent import _AnthropicVertexCaching

    mw = _AnthropicVertexCaching()
    model = _fake_vertex_anthropic()
    history = [HumanMessage("q"), *_step(1), HumanMessage("next")]

    # First placement is unconditional: no prefix exists yet to re-buy.
    seen = await _delivered(
        model,
        SystemMessage("sys"),
        mw=mw,
        messages=history,
        response=_usage(create=50_000),
    )
    assert _marks(seen["messages"]) == [1]

    # A big measured prefix against a tiny new step: not worth moving.
    history = [*history, *_step(2), HumanMessage("next")]
    seen = await _delivered(
        model,
        SystemMessage("sys"),
        mw=mw,
        messages=history,
        response=_usage(read=50_000),
    )
    assert _marks(seen["messages"]) == [1], "moved before it could repay"

    # A step large enough to clear 2 * prefix does move it. The mark closes the
    # step, so it lands on the AIMessage that opened it.
    big = len(history)
    history = [
        *history,
        AIMessage("x" * 800_000, tool_calls=[{"name": "t", "args": {}, "id": "big"}]),
        ToolMessage(content="r", tool_call_id="big"),
        HumanMessage("next"),
    ]
    seen = await _delivered(
        model,
        SystemMessage("sys"),
        mw=mw,
        messages=history,
        response=_usage(read=50_000),
    )
    assert _marks(seen["messages"]) == [big]


async def test_history_mark_restarts_when_history_shrinks():
    """Summarization (or a fresh run reusing the graph) leaves the boundary
    pointing at messages that no longer exist; the state must start over rather
    than mark an arbitrary position."""
    from airc_core.agent import _AnthropicVertexCaching

    mw = _AnthropicVertexCaching()
    model = _fake_vertex_anthropic()
    long_history = [HumanMessage("q"), *_step(1), *_step(2), HumanMessage("next")]
    await _delivered(
        model,
        SystemMessage("sys"),
        mw=mw,
        messages=long_history,
        response=_usage(create=9_000),
    )
    # Two completed steps, so the mark closes the second: messages[:4].
    assert mw._prefixes["t1"].boundary == 4

    await _delivered(
        model, SystemMessage("sys"), mw=mw, messages=[HumanMessage("compacted")]
    )
    st = mw._prefixes["t1"]
    assert st.boundary == 0 and st.prefix_tokens == 0


async def test_measured_span_replaces_the_estimate():
    """read+create is the cached span whichever way the provider reports it.
    This is also how B -- whether advancing re-bills the prefix or the delta --
    becomes readable from production instead of needing another probe."""
    from airc_core.agent import _AnthropicVertexCaching

    mw = _AnthropicVertexCaching()
    model = _fake_vertex_anthropic()
    messages = [HumanMessage("q"), *_step(1), HumanMessage("next")]
    await _delivered(
        model,
        SystemMessage("sys"),
        mw=mw,
        messages=messages,
        response=_usage(create=13_124),
    )
    assert mw._prefixes["t1"].prefix_tokens == 13_124
    await _delivered(
        model,
        SystemMessage("sys"),
        mw=mw,
        messages=messages,
        response=_usage(read=13_000, create=124),
    )
    assert mw._prefixes["t1"].prefix_tokens == 13_124


# ── _EmptyCandidateRetry (Gemini's zero-part candidate) ──────────────────────
#
# The stack-order assertion above only pins that it is listed AFTER the retry
# layer. These pin the behavior that placement buys: a zero-part candidate is
# retried through the shared backoff and finally raised by type, while a
# candidate that carries anything at all is passed straight through.


class _CandidateModel(BaseChatModel):
    """Returns a scripted AIMessage kwargs dict per call, repeating the last."""

    scripted: list = []
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "candidate"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        spec = self.scripted[min(self.calls, len(self.scripted) - 1)]
        object.__setattr__(self, "calls", self.calls + 1)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(**spec))])


def _candidate_agent(scripted):
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def read(x: str) -> str:
        """A read tool, so a no-tool-call turn exits the graph."""
        return x

    model = _CandidateModel(scripted=scripted)
    agent = create_agent(
        model,
        tools=[read],
        system_prompt="sys",
        middleware=base_middleware(_NON_VERTEX, "sys", []),
    )
    return model, agent


_EMPTY_STOP = {"content": "", "response_metadata": {"finish_reason": "STOP"}}


async def test_empty_candidate_retries_then_succeeds(monkeypatch):
    from airc_core import agent

    monkeypatch.setattr(agent.asyncio, "sleep", _noop_sleep)
    # A zero-part STOP (the silent-dead-turn shape) is a flake, not an answer:
    # re-rolled through the same backoff as a 429, and the retry's real reply
    # is what the turn returns.
    model, graph = _candidate_agent([_EMPTY_STOP, {"content": "the answer"}])
    state = await graph.ainvoke({"messages": [{"role": "user", "content": "go"}]})
    assert state["messages"][-1].content == "the answer"
    assert model.calls == 2


async def test_empty_candidate_exhausts_and_raises_by_type(monkeypatch):
    from airc_core import agent

    monkeypatch.setattr(agent.asyncio, "sleep", _noop_sleep)
    # A deterministic empty (a cache-fault or history poison, not a flake)
    # exhausts _EmptyCandidateRetry's own retry -- one identical call, then one
    # uncached+nudged call -- and raises EmptyCandidateError, a NAMED failure the
    # harness catches by type, never a silent dead turn. Two calls total, not the
    # six identical resends the old ModelRetryMiddleware loop churned through.
    model, graph = _candidate_agent([_EMPTY_STOP])
    try:
        await graph.ainvoke({"messages": [{"role": "user", "content": "go"}]})
        raised = False
    except EmptyCandidateError:
        raised = True
    assert raised
    assert model.calls == 2


async def test_a_candidate_with_content_or_tool_calls_is_never_retried(monkeypatch):
    from airc_core import agent

    monkeypatch.setattr(agent.asyncio, "sleep", _noop_sleep)
    # The discrimination the detection scope rests on: a SAFETY/RECITATION block
    # that CARRIES content is a genuine refusal, not the zero-part flake, so it
    # passes through untouched on the first call.
    refusal = {
        "content": "I cannot help with that.",
        "response_metadata": {"finish_reason": "SAFETY"},
    }
    model, graph = _candidate_agent([refusal])
    state = await graph.ainvoke({"messages": [{"role": "user", "content": "go"}]})
    assert state["messages"][-1].content == "I cannot help with that."
    assert model.calls == 1
    # Likewise a tool-calling step: empty text, but parts are present.
    model, graph = _candidate_agent(
        [
            {
                "content": "",
                "tool_calls": [{"name": "read", "args": {"x": "f"}, "id": "c1"}],
            },
            {"content": "done"},
        ]
    )
    state = await graph.ainvoke({"messages": [{"role": "user", "content": "go"}]})
    assert state["messages"][-1].content == "done"
    assert model.calls == 2


async def _retry_over_empty(responses):
    """Compose ModelRetryMiddleware over _EmptyCandidateRetry the way
    base_middleware nests them, and return (per inner call) the _empty_retry
    count the cache would key its step-aside on and the request's message count
    (so a mutated retry is visible). Exercised at the middleware level, not
    through a graph: langgraph runs nodes in a copied context, so a contextvar
    set inside the node is invisible to the caller. What matters is propagation
    to the middleware NESTED inside the retry (the cache), which shares this
    context.
    """
    from airc_core import agent

    empty = agent._EmptyCandidateRetry()
    retry = ModelRetryMiddleware(
        retry_on=agent._is_retryable,
        on_failure="error",
        max_retries=2,
        initial_delay=0,
        backoff_factor=0,
        max_delay=0,
    )
    seen, box = [], list(responses)

    async def inner(req):
        # Stands in for the growing cache: records the count it would key its
        # step-aside on at the moment it is asked to serve the call, and the
        # request length (the nudge _EmptyCandidateRetry appends grows it).
        seen.append((agent._empty_retry.get(), len(req.messages)))
        return type("R", (), {"result": [AIMessage(**box.pop(0))]})()

    agent._empty_retry.set(0)
    with contextlib.suppress(EmptyCandidateError):
        await retry.awrap_model_call(
            _Req(0, [HumanMessage("go")]), lambda r: empty.awrap_model_call(r, inner)
        )
    return seen


async def test_a_repeated_empty_is_retried_once_uncached_with_a_nudge():
    # The count the cache keys its step-aside on, and the mutation. The first
    # call sees _empty_retry 0 (serve cached) on the original 1-message request;
    # the retry sees 1 (step aside, so the cache stops serving the prefix that
    # may be producing the empty) on a request grown by the nudge. Bounded to
    # those two calls: no sixth identical resend.
    assert await _retry_over_empty([_EMPTY_STOP] * 3) == [(0, 1), (1, 2)]


async def test_a_non_empty_response_resets_the_retry_counter():
    from airc_core import agent

    # One empty then a real reply: the episode is over, so a later empty in the
    # same turn starts fresh against a warm cache rather than stepping aside
    # immediately.
    seen = await _retry_over_empty([_EMPTY_STOP, {"content": "the answer"}])
    assert seen == [(0, 1), (1, 2)]
    assert agent._empty_retry.get() == 0


async def test_an_empty_candidate_is_never_handed_back_to_the_retry_layer():
    from airc_core import agent

    # _is_retryable must reject EmptyCandidateError by TYPE. The raise embeds
    # the provider's finish_reason, so a reason naming an overload or an
    # unavailable region would match the string-based transient test and
    # re-drive the whole two-call episode per outer attempt -- 14 model calls
    # and the full backoff ladder, worse than the wedge this path prevents.
    for reason in ("STOP", "MODEL_OVERLOADED", "unavailable in region", "429 quota"):
        exc = agent.EmptyCandidateError(f"empty candidate (finish_reason={reason})")
        assert not agent._is_retryable(exc), reason
    # A genuine transient still retries, by message.
    assert agent._is_retryable(RuntimeError("503 model overloaded"))


async def test_the_nudge_is_not_reappended_on_an_empty_candidate_retry():
    from airc_core import agent

    # after_model never runs when _EmptyCandidateRetry raises, so model_calls
    # freezes and the threshold would re-fire on every retry of the same call --
    # stacking copies of the nudge onto a request the model already refused.
    agent._empty_retry.set(0)
    mw = CallBudgetMiddleware([(2, "converge now")])
    assert any("converge now" in m for m in await _appended(mw, 2))
    agent._empty_retry.set(1)  # now retrying that same call
    assert not any("converge now" in m for m in await _appended(mw, 2))
    agent._empty_retry.set(0)


def test_summarization_added_only_with_a_summarizer_model(
    middleware_with_summarizer,
):
    # No summarizer -> no compaction middleware (unchanged stack).
    assert not any(
        isinstance(m, SummarizationMiddleware)
        for m in base_middleware(_NON_VERTEX, "s", [])
    )
    # With one, it leads (a before_model state mutation, so the request shapers
    # and the growing cache see the already-compacted state).
    mw = middleware_with_summarizer
    assert isinstance(mw[0], SummarizationMiddleware)
    assert type(mw[1]).__name__ == "_ContextBudget"  # still the outermost shaper


def test_summarizer_reads_the_whole_block_not_langchains_4k_default(
    middleware_with_summarizer,
):
    # langchain trims the to-summarize block to its last 4000 tokens by default,
    # which at our ~900k trigger would summarize almost none of what it drops. We
    # lift that cap so the summary actually covers the block.
    from airc_core.agent import _SUMMARY_TRIM_TOKENS

    mw = middleware_with_summarizer
    summ = next(m for m in mw if isinstance(m, SummarizationMiddleware))
    assert summ.trim_tokens_to_summarize == _SUMMARY_TRIM_TOKENS
    assert summ.trim_tokens_to_summarize > 4000
    # Capped under the summarizer's window so an overflow can't turn into the
    # error-string-as-history failure mode.
    from airc_core.agent import CONTEXT_WINDOW

    assert summ.trim_tokens_to_summarize < CONTEXT_WINDOW


def test_keep_is_token_based_and_tuning_invariants_hold(middleware_with_summarizer):
    # keep is a token budget, not a message count, so the verbatim tail is
    # predictable across chat and tool-heavy agents.
    from airc_core.agent import (
        _SUMMARY_KEEP_TOKENS,
        _SUMMARY_TRIGGER_TOKENS,
        _SUMMARY_TRIM_TOKENS,
    )

    mw = middleware_with_summarizer
    summ = next(m for m in mw if isinstance(m, SummarizationMiddleware))
    assert summ.keep == ("tokens", _SUMMARY_KEEP_TOKENS)
    # No silent drop: the summarizer's cap covers the whole block (trigger - keep).
    assert _SUMMARY_KEEP_TOKENS + _SUMMARY_TRIM_TOKENS >= _SUMMARY_TRIGGER_TOKENS
    # No overflow: even with ~10% XML expansion the input stays under the window.
    from airc_core.agent import CONTEXT_WINDOW

    assert _SUMMARY_TRIM_TOKENS * 1.1 < CONTEXT_WINDOW


class _RaisingModel:
    """A summarizer whose invoke always fails, with the one attribute
    SummarizationMiddleware inspects at construction."""

    _llm_type = "chat-fake"

    async def ainvoke(self, *a, **k):
        raise RuntimeError("boom")


async def test_summary_failure_keeps_history_instead_of_replacing_it():
    # A tiny trigger so a couple of messages force summarization.
    cfg = {
        "model": _RaisingModel(),
        "trigger": ("tokens", 5),
        "keep": ("tokens", 1),
    }
    msgs = [
        HumanMessage("hello there " * 20),
        AIMessage("general kenobi " * 20),
        HumanMessage("a question " * 20),
        AIMessage("an answer " * 20),
    ]
    state = {"messages": list(msgs)}

    # Stock middleware swallows the error and returns a mutation that replaces the
    # history with the exception text -- the failure mode we are guarding against.
    stock = await SummarizationMiddleware(**cfg).abefore_model(state, None)
    assert stock is not None

    # Our subclass instead skips: no state mutation, history left intact.
    safe = await _SkipOnSummaryFailure(**cfg).abefore_model(state, None)
    assert safe is None


async def test_summary_call_does_not_leak_into_the_message_stream():
    # astream(stream_mode="messages") attaches a streaming callback handler,
    # which silently upgrades the summarizer's nested ainvoke to streaming --
    # untagged, every summary token is then emitted as an AIMessageChunk, and a
    # reply collector posts the conversation restatement as the agent's message
    # (the echoed-prompt artifact). The nostream tag must keep the whole call
    # out of the message stream while the real reply still streams.
    from langchain.agents import create_agent
    from langchain_core.language_models import GenericFakeChatModel
    from langchain_core.messages import AIMessageChunk

    agent = create_agent(
        GenericFakeChatModel(messages=iter([AIMessage("the real reply")])),
        tools=[],
        system_prompt="sys",
        middleware=[
            _SkipOnSummaryFailure(
                model=GenericFakeChatModel(messages=iter([AIMessage("a summary")])),
                trigger=("tokens", 30),
                keep=("tokens", 5),
            )
        ],
    )
    input = {
        "messages": [
            HumanMessage("long filler prose " * 40),
            AIMessage("more filler prose " * 40),
            HumanMessage("reply now"),
        ]
    }
    texts: list[str] = []
    async for _mode, (chunk, _meta) in agent.astream(input, stream_mode=["messages"]):
        if isinstance(chunk, AIMessageChunk) and isinstance(chunk.content, str):
            texts.append(chunk.content)
    joined = "".join(texts)
    assert "a summary" not in joined
    assert "the real reply" in joined


def test_growing_cache_only_for_vertex_when_enabled():
    assert growing_cache_middleware(_VERTEX, "sys", [], True, 30, 70) is not None
    # Non-Vertex model: create_context_cache is unavailable, so no overlay.
    assert growing_cache_middleware(_NON_VERTEX, "sys", [], True, 30, 70) is None


def test_growing_cache_omitted_when_disabled():
    assert growing_cache_middleware(_VERTEX, "sys", [], False, 30, 70) is None


def test_context_budget_is_outermost():
    # Both builders compose base_middleware + (cache?) + governor; the crash-fix
    # guarantee is that _ContextBudget leads regardless of the appended tail.
    composed = [*base_middleware(_VERTEX, "sys", []), "governor"]
    assert type(composed[0]).__name__ == "_ContextBudget"


async def test_grounding_reminder_inserts_once_per_interval():
    from airc_core.agent import (
        _CHARS_PER_TOKEN,
        _GROUNDING_REMINDER,
        _GROUNDING_SRC,
        GroundingReminderMiddleware,
    )

    mw = GroundingReminderMiddleware(interval=1000)

    def _msg(tokens):
        return HumanMessage("x" * (tokens * _CHARS_PER_TOKEN))

    # Shallow: below the interval -> no insert.
    assert await mw.abefore_model({"messages": [_msg(500)]}, None) is None

    # Deep: >= interval of content -> insert one marked reminder (a tail append).
    out = await mw.abefore_model({"messages": [_msg(1500)]}, None)
    reminder = out["messages"][0]
    assert reminder.content == _GROUNDING_REMINDER
    assert reminder.additional_kwargs["lc_source"] == _GROUNDING_SRC

    # A reminder within the last interval -> not due again (self-tracked).
    assert await mw.abefore_model({"messages": [_msg(1500), reminder]}, None) is None

    # Once another interval of content follows it -> due again.
    grown = [_msg(1500), reminder, _msg(1500)]
    assert await mw.abefore_model({"messages": grown}, None) is not None


def test_grounding_reminder_in_base_and_disablable():
    assert "GroundingReminderMiddleware" in _names(
        base_middleware(_NON_VERTEX, "s", [])
    )
    off = _names(base_middleware(_NON_VERTEX, "s", [], grounding_tokens=0))
    assert "GroundingReminderMiddleware" not in off


async def test_two_reminders_do_not_suppress_each_other():
    # The marker used to be a module-level constant matched by a staticmethod, so
    # a second instance read the FIRST one's inserts as its own and went quiet --
    # silently, and only in threads long enough for both to fire. Each instance
    # now carries its own src, so each tracks only its own reminders.
    from airc_core.agent import _CHARS_PER_TOKEN, GroundingReminderMiddleware

    tools = GroundingReminderMiddleware(1000, "[system reminder] tools", "tools")
    ground = GroundingReminderMiddleware(1000, "[system reminder] ground", "ground")
    deep = [HumanMessage("x" * (1500 * _CHARS_PER_TOKEN))]

    inserted = (await tools.abefore_model({"messages": deep}, None))["messages"]
    assert inserted[0].additional_kwargs["lc_source"] == "tools"

    # The other reminder's insert is content to us, not a reminder: still due.
    both = [*deep, *inserted]
    assert await ground.abefore_model({"messages": both}, None) is not None
    # ...while our own suppresses us.
    assert await tools.abefore_model({"messages": both}, None) is None


def test_two_reminders_compile_into_one_agent():
    # The harness composes base_middleware (which carries the grounding
    # reminder) with one GroundingReminderMiddleware per application reminder.
    # create_agent both asserts on duplicate middleware names and keys graph
    # nodes on them, and the name defaults to the CLASS name -- so exactly this
    # composition raised "Please remove duplicate middleware instances." at
    # graph build and killed every icompleteu goal turn. Must drive the real
    # create_agent validation: a name-set comparison here could drift from
    # whatever the factory checks. Fails against the class-name default.
    from airc_core.agent import GroundingReminderMiddleware
    from langchain.agents import create_agent
    from langchain_core.language_models import GenericFakeChatModel

    create_agent(
        GenericFakeChatModel(messages=iter([])),
        tools=[],
        system_prompt="sys",
        middleware=[
            *base_middleware(_NON_VERTEX, "sys", []),
            GroundingReminderMiddleware(150_000, "use the tools", "tools_reminder"),
        ],
    )


# ── RequireStructuredResultMiddleware ────────────────────────────────────────
#
# The guard against a silent no-verdict: a ToolStrategy turn that answers in
# plain text (no result tool call) exits with structured_response unset, which a
# review verifier reads as "no verdict" and drops the finding. These cover the
# _reask decision in isolation, then the whole graph re-asking and terminating.


def test_require_result_fires_only_on_a_plain_text_ending():
    mw = RequireStructuredResultMiddleware("remind", max_reasks=3)
    # Plain-text terminal, no structured_response yet: re-ask (jump back to model).
    out = mw.after_model({"messages": [HumanMessage("go"), AIMessage("prose")]}, None)
    assert out is not None and out["jump_to"] == "model"
    # The re-ask is the caller's reminder, tail-appended and marked as ours.
    (msg,) = out["messages"]
    assert msg.content == "remind"
    assert mw._is_reask(msg)
    # The per-turn UntrackedValue counter increments from 0 -> 1 (the bound).
    assert out["reasks"] == 1


def test_require_result_ignores_a_delivered_or_in_progress_verdict():
    mw = RequireStructuredResultMiddleware("remind")
    # A delivered verdict (structured_response set): nothing to recover.
    done = {"messages": [AIMessage("prose")], "structured_response": object()}
    assert mw.after_model(done, None) is None
    # An AIMessage carrying tool calls is an intermediate step, not a terminal.
    mid = {
        "messages": [AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": "1"}])]
    }
    assert mw.after_model(mid, None) is None


def test_require_result_is_bounded_by_the_per_turn_counter():
    # The bound is a per-turn UntrackedValue counter (reasks): resets each ainvoke,
    # survives the jump's supersteps within the turn. At or above the cap the
    # middleware gives up; below it re-asks and increments the counter.
    mw = RequireStructuredResultMiddleware("remind", max_reasks=2)
    at_cap = {"messages": [AIMessage("still prose")], "reasks": 2}
    assert mw.after_model(at_cap, None) is None
    below = {"messages": [AIMessage("still prose")], "reasks": 1}
    out = mw.after_model(below, None)
    assert out is not None and out["reasks"] == 2


class _ScriptedToolModel(BaseChatModel):
    """Chat model stand-in that replays a script of AIMessage kwargs, one per
    call (repeating the last), and treats bind_tools as a passthrough -- the
    stock fakes raise NotImplementedError on bind_tools, which create_agent needs.
    A fresh AIMessage per call gives each a distinct id so the add_messages
    reducer accumulates them, as a real model's replies do."""

    scripted: list = []
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-tool"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        spec = self.scripted[min(self.calls, len(self.scripted) - 1)]
        object.__setattr__(self, "calls", self.calls + 1)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(**spec))])


def _verdict_agent(scripted, max_reasks=3, with_governors=False):
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware
    from langchain.agents.structured_output import ToolStrategy
    from langchain_core.tools import tool
    from pydantic import BaseModel

    @tool
    def read(x: str) -> str:
        """A read tool, so the graph exits on a no-tool-call turn (as the real
        review graph does) rather than always retrying a missing structured
        output (the no-tools model_to_model edge)."""
        return x

    class Verdict(BaseModel):
        verdict: str

    # with_governors mirrors the review graph's control tail and ORDER: the
    # re-ask middleware sits before the two governors in the list so its
    # after_model runs after theirs (last-appended runs first), leaving it an
    # intermediate after_model node -- not the routing owner it is in isolation.
    require = RequireStructuredResultMiddleware("CALL THE TOOL", max_reasks)
    middleware = [require]
    if with_governors:
        middleware += [
            _DropEmptyResponses(),  # a real after_model peer that owns the edge
            CallBudgetMiddleware([(10, "converge")]),
            ModelCallLimitMiddleware(run_limit=200, exit_behavior="end"),
        ]
    model = _ScriptedToolModel(scripted=scripted)
    agent = create_agent(
        model,
        tools=[read],
        system_prompt="sys",
        middleware=middleware,
        response_format=ToolStrategy(schema=Verdict, handle_errors=True),
    )
    return model, agent


_TOOL_TURN = {
    "content": "",
    "tool_calls": [{"name": "Verdict", "args": {"verdict": "rejected"}, "id": "c1"}],
}


def _reminders(state):
    return sum(1 for m in state["messages"] if "CALL THE TOOL" in str(m.content))


async def test_require_result_recovers_a_plain_text_turn_end_to_end():
    # Prose first (would have ended as no-verdict), then the tool call the re-ask
    # elicited: the verdict is recovered instead of lost.
    model, agent = _verdict_agent([{"content": "prose"}, _TOOL_TURN])
    state = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
    assert state.get("structured_response").verdict == "rejected"
    assert model.calls == 2
    assert _reminders(state) == 1


async def test_require_result_gives_up_bounded_when_never_complies():
    # A model that only ever answers in prose is re-asked exactly max_reasks
    # times, then the turn ends verdict-less (None -> the caller's incomplete,
    # never a clean pass). The bound is what keeps it off the recursion limit.
    _model, agent = _verdict_agent([{"content": "always prose"}], max_reasks=3)
    state = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
    assert state.get("structured_response") is None
    assert _reminders(state) == 3


async def test_require_result_is_inert_when_the_tool_is_called():
    # A turn that calls the result tool immediately is untouched: no re-ask, no
    # injected reminder.
    model, agent = _verdict_agent([_TOOL_TURN])
    state = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
    assert state.get("structured_response").verdict == "rejected"
    assert model.calls == 1
    assert _reminders(state) == 0


async def test_require_result_composes_with_the_review_governors():
    # The review graph wires the re-ask BEFORE the call-budget and call-limit
    # governors, so it is an intermediate after_model node (not the routing
    # owner). Its jump back to the model must still fire there, and the bound must
    # still terminate -- otherwise a prose-forever turn spins to the recursion
    # limit even though the isolated middleware is correct.
    _model, agent = _verdict_agent(
        [{"content": "prose"}, _TOOL_TURN], with_governors=True
    )
    state = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
    assert state.get("structured_response").verdict == "rejected"
    assert _reminders(state) == 1

    _model, agent = _verdict_agent(
        [{"content": "always prose"}], max_reasks=3, with_governors=True
    )
    state = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
    assert state.get("structured_response") is None
    assert _reminders(state) == 3

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""The callback that turns one graph invocation into a `Usage`.

Kept apart from usage.py because a callback subclasses the framework's
handler, and the value it produces has to stay importable without the
framework (it is part of the wire spec).
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage

from .usage import MODEL_KEY, SOURCE_KEY, SUMMARIZATION, Usage

log = logging.getLogger(__name__)


def usage_from_response(response) -> Mapping[str, Any]:
    """The single call's usage_metadata off an LLMResult, via the generation
    message; {} if the provider attached none. The aggregating callback hides
    per-call detail, and per-call is where the price tier is decided."""
    for gens in getattr(response, "generations", None) or []:
        for gen in gens:
            usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
            if usage:
                return usage
    return {}


# The collector whose invocation is running, for spend that is not a model
# call and so never reaches a callback: an explicit cache the middleware
# creates. Set by UsageCollector.active() around the invoke; langgraph copies
# the context into its tasks, so the middleware sees it.
_active: contextvars.ContextVar[UsageCollector | None] = contextvars.ContextVar(
    "usage_collector", default=None
)


def book_aside(usage: Usage) -> None:
    """Add spend that is not one of the invocation's model calls to the active
    collector's `aside`. No collector active means nobody is booking, which is
    the same as before the collector existed; the caller need not care."""
    if (c := _active.get()) is not None:
        c.aside = c.aside + usage


class UsageCollector(BaseCallbackHandler):
    """Books every model call of one graph invocation.

    `total` is the agent's own calls on `model_id`. `aside` is spend the
    invocation caused on top of them: the ceiling summarization the middleware
    runs on the filter model, and an explicit cache it created. Different
    models doing different work, kept off the agent's row but not dropped,
    since they are paid for. Pass callbacks=[collector] on the invoke and run
    the invoke inside `active()`; nested model calls inherit the callback,
    and non-call spend reaches `aside` through book_aside.

    Model calls in an invocation are sequential, so plain accumulation is
    safe. A call that errors fires start but not end and books nothing, which
    matches the providers: a failed request is not billed. Request details are
    keyed by run_id to pair start with end, and dropped on error so a retry
    does not inherit it.
    """

    def __init__(self, agent: str, kind: str, model_id: str) -> None:
        super().__init__()
        self._agent = agent
        self._kind = kind
        self._model_id = model_id
        self.total = Usage(model=model_id)
        self.aside = Usage()
        # The aggregate keeps max_call_input, but cadence design needs the
        # growth curve. Keep the provider-reported prompt size of every agent
        # call; callers that persist usage can store it beside the total.
        self.call_input_tokens: list[int] = []
        self._shape: dict[object, tuple[int, int, int]] = {}
        self._aside_model: dict[object, str] = {}

    @contextlib.contextmanager
    def active(self):
        token = _active.set(self)
        try:
            yield self
        finally:
            _active.reset(token)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        meta = kwargs.get("metadata") or {}
        if meta.get(SOURCE_KEY) == SUMMARIZATION:
            self._aside_model[run_id] = str(meta.get(MODEL_KEY) or "")
            return
        msgs = messages[0] if messages else []
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        tool_chars = sum(len(str(m.content)) for m in tool_msgs)
        self._shape[run_id] = (len(msgs), len(tool_msgs), tool_chars)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._shape.pop(run_id, None)
        self._aside_model.pop(run_id, None)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        usage = usage_from_response(response)
        if run_id in self._aside_model:
            model = self._aside_model.pop(run_id)
            self.aside = self.aside + Usage.of_call(usage, model)
            return
        call = Usage.of_call(usage, self._model_id)
        self.total = self.total + call
        self.call_input_tokens.append(call.input)
        n_msgs, n_tool, tool_chars = self._shape.pop(run_id, (0, 0, 0))
        # Per call, so the growth curve within a turn is visible: what this
        # call cost and what the turn has cost so far, then the prompt's size
        # and hit rate.
        log.info(
            "call %s/%s #%d: %s (turn %s): %s; %d msgs, %d tool results (%d chars)",
            self._agent,
            self._kind,
            self.total.calls,
            call.cost(),
            self.total.cost(),
            call.shape(),
            n_msgs,
            n_tool,
            tool_chars,
        )

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Bounded, tool-less model judgments with validated structured output.

This is the small-task counterpart to deepagent. A deep-agent turn owns tools,
a worktree, conversation state and possibly several resumable turns. A
structured task receives all of its evidence from the caller, has no tools or
checkpoint, and returns one typed value from one bounded graph invocation.

The graph may make a few model calls to recover a malformed or plain-text
answer. "One task" means no application-visible conversation or reentry loop,
not that provider validation gets only one chance.
"""

from __future__ import annotations

import asyncio
import logging
import zlib
from collections import OrderedDict
from typing import TypeVar

from pydantic import BaseModel

from .agent import (
    RequireStructuredResultMiddleware,
    base_middleware,
)
from .collector import UsageCollector
from .config import CommonConfig, profile_for
from .model import make_model
from .tokens import TokenLog

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_MAX_GRAPHS = 32
_DEFAULT_MAX_MODEL_CALLS = 4
_DEFAULT_MAX_REASKS = 2
_RECURSION_LIMIT = 64
_REQUIRE_RESULT = (
    "Your previous answer was not recorded because it did not use the required"
    " structured response. Return the required structured response now."
)


class StructuredTaskError(RuntimeError):
    """A structured task timed out, failed, or produced no validated result."""


class StructuredTaskRunner:
    """Run independent, tool-less, typed model tasks against shared suite config.

    Build one runner per process and reuse it. Compiled graphs are cached by
    system prompt and schema, while every invocation starts with fresh message
    state: graph reuse saves setup cost but never turns tasks into a conversation.

    The runner deliberately accepts text rather than arbitrary LangChain
    messages. A task's capability boundary is that the caller supplies all
    evidence; accepting prior AI/tool messages would quietly grow this into an
    agent transcript without the controls deepagent applies to one.
    """

    def __init__(
        self,
        common: CommonConfig,
        *,
        model_key: str = "default",
        max_model_calls: int = _DEFAULT_MAX_MODEL_CALLS,
        max_reasks: int = _DEFAULT_MAX_REASKS,
    ) -> None:
        # The profile, not the id: the entry's effort is part of what the
        # operator configured, and reading `models` would drop it silently.
        self._profile = profile_for(common, model_key, "")
        self._model_id = self._profile.id
        if max_model_calls < 1:
            raise ValueError("max_model_calls must be positive")
        if max_reasks < 0:
            raise ValueError("max_reasks cannot be negative")
        self._max_model_calls = max_model_calls
        self._max_reasks = max_reasks
        self._tokens = TokenLog(common.token_db_path)
        self._graphs: OrderedDict[tuple[str, type[BaseModel]], object] = OrderedDict()

    def close(self) -> None:
        self._tokens.close()

    def _graph_for(self, system_prompt: str, schema: type[T]):
        key = (system_prompt, schema)
        if key in self._graphs:
            self._graphs.move_to_end(key)
            return self._graphs[key]

        from langchain.agents import create_agent
        from langchain.agents.middleware import ModelCallLimitMiddleware
        from langchain.agents.structured_output import ToolStrategy

        # No explicit context cache: a task has no cross-call history, and the
        # small validation-retry tail does not justify the lifecycle and tool
        # binding complexity of a provider cache. base_middleware still supplies
        # the shared context bound, empty-candidate handling and transient retry.
        middleware = base_middleware(
            self._model_id, system_prompt, [], grounding_tokens=0
        )
        # Ordering matches the agent runtimes: after_model hooks run in reverse,
        # so the call limit records the call before the result middleware decides
        # whether a plain-text terminal deserves one bounded re-ask.
        middleware += [
            RequireStructuredResultMiddleware(
                _REQUIRE_RESULT, max_reasks=self._max_reasks
            ),
            ModelCallLimitMiddleware(
                run_limit=self._max_model_calls, exit_behavior="end"
            ),
        ]
        graph = create_agent(
            make_model(self._model_id, **self._profile.call_kwargs),
            tools=[],
            system_prompt=system_prompt,
            middleware=middleware,
            response_format=ToolStrategy(schema=schema, handle_errors=True),
        ).with_config({"recursion_limit": _RECURSION_LIMIT})
        self._graphs[key] = graph
        while len(self._graphs) > _MAX_GRAPHS:
            self._graphs.popitem(last=False)
        return graph

    async def run(
        self,
        input: str,
        *,
        schema: type[T],
        system_prompt: str = "",
        task: str = "structured-task",
        run_key: str = "",
        timeout_s: float = 300.0,
    ) -> T:
        """Return one validated result or raise StructuredTaskError.

        `task` is the stable operation name in logs and token accounting;
        `run_key` identifies this subject (job id, commit hash, etc.) without
        becoming model-visible state. Cancellation propagates unchanged so a
        daemon shutdown never reads as an inference failure.
        """
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        graph = self._graph_for(system_prompt, schema)
        usage = UsageCollector(task, "structured-task", self._model_id)
        try:
            with usage.active():
                state = await asyncio.wait_for(
                    graph.ainvoke(
                        {"messages": [{"role": "user", "content": input}]},
                        config={"callbacks": [usage]},
                    ),
                    timeout=timeout_s,
                )
        except asyncio.CancelledError:
            raise
        except TimeoutError as e:
            raise StructuredTaskError(f"{task} timed out after {timeout_s:.0f}s") from e
        except Exception as e:
            raise StructuredTaskError(f"{task} failed: {e}") from e
        finally:
            thread = zlib.crc32(run_key.encode()) if run_key else 0
            self._tokens.add(
                usage.total, thread_id=thread, agent=task, kind="structured-task"
            )
            if not usage.aside.empty:
                self._tokens.add(
                    usage.aside, thread_id=thread, agent=task, kind="overhead"
                )

        result = state.get("structured_response")
        if not isinstance(result, schema):
            raise StructuredTaskError(
                f"{task} produced no valid {schema.__name__} after"
                f" {usage.total.calls} model call(s)"
            )
        log.info("%s: complete, %s", task, usage.total.line())
        return result

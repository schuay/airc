# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""LangGraphHarness: the in-process coding agent, one bounded turn per call.

Kept out of harness.py so that module (the Protocol, AgentResult, MockHarness)
stays langchain-free and cheap to import. All heavy imports (langchain,
langgraph, airc-core, airc-tools) live here.

The agent reuses airc's model-call stack -- base_middleware (context budget,
empty-response strip, retry, Anthropic caching), the Vertex growing-prefix
cache, and the call-budget governor -- so caching and token accounting behave
identically to the rest of the suite. One graph is built per per-stage thread
and reused across that loop's turns, so the growing cache warms across turns.

Domain-agnostic: the system prompt and the per-agent verdict schemas are
injected by the application. The turn ends when the agent calls the structured
report tool (ToolStrategy); the parsed schema is flattened into AgentResult.data.
A turn that hits the model-call cap ends without a report and is resumed,
continuing the same thread.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import zlib
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field

from airc_core import (
    CommonConfig,
    EmptyCandidateError,
    MCPToolset,
    TokenLog,
    apply_gcp_env_defaults,
    base_middleware,
    growing_cache_middleware,
    make_model,
)
from airc_tools.edit import apply_edits as _apply_edits
from airc_tools.edit import write_file as _write_file
from airc_tools.read import read_file as _read_file
from airc_tools.sandbox import Sandbox
from airc_tools.shell import run_shell as _run_shell

from .harness import REPORT_TOOL_NAME, AgentResult, HarnessRun, Report, to_result
from .journal import EventKind, Journal

log = logging.getLogger(__name__)

# Tool args and results in the journal are truncated: the full picture lives in
# the worktree and blobs; the journal is a following stream, not an archive.
_JOURNAL_TRUNC = 2000


def _trunc(s: str, n: int = _JOURNAL_TRUNC) -> str:
    return s if len(s) <= n else s[:n] + f" ...[+{len(s) - n} chars]"


# Model-call budget for a coding turn: nudge toward wrapping up, then a hard cap
# that ends the turn (no report -> the loop resumes the thread, continuing the
# accumulated context). Generous: a coding turn legitimately runs many tool calls.
#
# The cap, not the context window, is what ends a long turn. A measured draft
# turn grew ~3.3k tokens per call (174k peak input at call 52), so summarization
# (_SUMMARY_TRIGGER_TOKENS, 80% of a 1M window) would not trigger until roughly
# call 240 -- the cap binds ~3x earlier than the window can support. 100 buys
# turns that end on their own work rather than mid-investigation; the nudges
# keep their old distance from it (cap minus 30 / minus 10).
_SOFT_NUDGE_CALLS = 70
_HARD_NUDGE_CALLS = 90
_MAX_MODEL_CALLS = 100
_SOFT_NUDGE = (
    "You have done substantial work this turn. Wrap up THIS turn: finish the"
    f" current build/test, then call the `{REPORT_TOOL_NAME}` tool. If you are"
    " not finished, report with disposition=continue -- you resume next turn"
    " with your worktree and context intact, so this is a checkpoint, not a"
    " verdict. Do not open a new line of investigation in this turn."
)
_HARD_NUDGE = (
    "Stop starting new work. Finish the in-flight build/test if any, then call"
    f" the `{REPORT_TOOL_NAME}` tool now -- disposition=continue if you still"
    " have work (you resume with context intact), a terminal disposition only if"
    " you are genuinely done."
)
# The ToolMessage the model sees in its history after calling the report tool.
# The langchain default ("Returning structured response: <report>") reads as a
# finished answer, so on the next turn the model concludes it already reported
# and does nothing -- an empty turn. Reframe a `continue` report as a checkpoint
# so the resumed turn keeps working. A terminal report ends the loop and is
# never resumed, so tuning the wording for the continue case is safe. Also drops
# the echoed report body, which was pure context bloat (the harness reads the
# parsed report from state, not from this message).
_REPORT_ACK = (
    "Report recorded. If your disposition was `continue`, this was a checkpoint,"
    " not completion -- you have not finished. On your next turn, pick up where"
    f" you left off with the next concrete step, then call the `{REPORT_TOOL_NAME}`"
    " tool again."
)
# Recovery re-ask when a ToolStrategy turn ends in plain text with no report tool
# call. Without it the loop exits the instant a turn has no tool calls, so a model
# that writes its conclusion as prose ends the turn with structured_response unset
# -- the harness reads None and the reentry loop counts a dead turn toward its cap
# (observed: a turn that stops one call past the soft nudge without reporting).
# RequireStructuredResultMiddleware appends this and jumps back to the model.
# Distinct from the wrap-up nudges -- those steer a still-running turn; this
# recovers one that already stopped -- and it names the concrete failure so the
# model calls the tool rather than re-explaining in prose.
_REQUIRE_RESULT_REMINDER = (
    "You ended your turn with a plain-text message and did not CALL the"
    f" `{REPORT_TOOL_NAME}` tool, so nothing was recorded and the turn is lost. Do"
    " not answer in prose. CALL the tool now: disposition=continue if you have"
    " more work (you resume with context intact), a terminal disposition if you"
    " are genuinely done. Put every conclusion into its fields."
)
# Bound on re-asks per turn: enough to recover a chatty model that forgot the
# tool more than once before complying, low enough that one which will not comply
# cannot spin. The hard call cap is the outer backstop. (Per-turn, not per job:
# the bound is an UntrackedValue counter, never checkpointed, so it resets on
# each ainvoke instead of accumulating across a stage-loop's turns on a shared
# thread. Nothing to clear on resume.)
_MAX_REASKS = 5
# Graph nodes per model call (middleware hooks), so keep the recursion limit well
# above _MAX_MODEL_CALLS * nodes/call: the call cap should govern, not a
# GraphRecursionError. Scaled with the cap (the old 800 was ~11x a 70-call cap);
# a GraphRecursionError is a strictly worse ending than the cap, since it ends
# the turn with no report and lands in the loop's dead-turn path.
_RECURSION_LIMIT = 1200
# Bound on live per-thread graphs (each pins an InMemorySaver conversation and a
# growing-cache state). Evicting frees the saver; the server-side Vertex caches
# it created lapse on their TTL.
_MAX_GRAPHS = 16

# A minimal generic system prompt; applications pass their own (identity +
# conventions + skill index) via LangGraphHarness(system_prompt=...).
_DEFAULT_SYSTEM = f"""\
You are an autonomous coding agent working in one git worktree. Understand the
task, edit code, build and test in the worktree, and iterate until it is
correct. shell/read_file/edit_file/write_file act in your worktree; end every turn by
calling the `{REPORT_TOOL_NAME}` tool exactly once (with disposition `continue`
if you need another turn -- the worktree and your context persist).
"""


class _Edit(BaseModel):
    search: str = Field(
        description=(
            "Exact text to find, verbatim including indentation. Empty to create"
            " a new file or append to an existing one."
        )
    )
    replace: str = Field(description="Text to substitute for the search text.")


def _abs(workdir: Path, path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else workdir / p)


def _worktree_tools(
    workdir: Path, shell_timeout_s: float, sandbox: Sandbox | None = None
) -> list:
    """airc-tools shell/read/edit bound to one job's worktree.

    cwd and relative-path resolution are bound in the closure (not tool args), so
    the tool schemas are identical across jobs -- the growing prefix cache sees a
    stable [system + tools] prefix -- while each job's tools act only in its own
    tree. Per-thread binding also means no shared mutable cwd/env across the
    concurrent jobs a scheduler may run.

    With a sandbox, shell commands run inside its bwrap/cgroup wrapper, and
    read/edit -- which execute in this trusted process -- enforce the same
    boundary via realpath containment. The latter matters because casefile
    documents flow out to chat: an unconfined read_file would let injected
    content exfiltrate credentials without any network egress.
    """
    from langchain_core.tools import StructuredTool

    wt = Path(workdir)

    async def shell(command: str, timeout: float = shell_timeout_s) -> str:
        return await _run_shell(command, cwd=str(wt), timeout=timeout, sandbox=sandbox)

    def read_file(path: str, offset: int = 1, limit: int = 2000) -> str:
        p = _abs(wt, path)
        if sandbox is not None and (deny := sandbox.check(p, write=False)):
            return deny
        return _read_file(p, offset, limit)

    def edit_file(path: str, edits: list[_Edit]) -> str:
        p = _abs(wt, path)
        if sandbox is not None and (deny := sandbox.check(p, write=True)):
            return deny
        return _apply_edits(p, [(e.search, e.replace) for e in edits])

    def write_file(path: str, content: str) -> str:
        p = _abs(wt, path)
        if sandbox is not None and (deny := sandbox.check(p, write=True)):
            return deny
        return _write_file(p, content)

    return [
        StructuredTool.from_function(
            coroutine=shell,
            name="shell",
            description=(
                "Run a command in a fresh `bash -lc` in the worktree (stateless:"
                " no cd/env persists): builds, test binaries, git state, and"
                " listing/searching files. NOT for writing file content -- author"
                " files with write_file/edit_file; a shell write into a source"
                " file is refused. Redirecting a command's output to a log is"
                f" fine. The default timeout is {shell_timeout_s:g}s: raise it for"
                " work that is honestly long (a build, a test suite, a benchmark),"
                " and lower it for a command that may hang (e.g. a repro)."
            ),
        ),
        StructuredTool.from_function(
            func=read_file,
            name="read_file",
            description=(
                "Read a file verbatim from 1-based line `offset` for `limit`"
                " lines. Relative paths resolve in the worktree; other paths are"
                " absolute. No line-number gutter, so output pastes into an edit"
                " search."
            ),
        ),
        StructuredTool.from_function(
            func=edit_file,
            name="edit_file",
            description=(
                "Apply exact SEARCH/REPLACE `edits` to one file, all-or-nothing."
                " Keep each search small and exact. For a whole new file or a full"
                " rewrite, use write_file instead. If a SEARCH does not match,"
                " read_file that region again and retry with the exact bytes --"
                " a failed match is a stale search, not a reason to fall back to"
                " a shell rewrite (which is refused anyway)."
            ),
        ),
        StructuredTool.from_function(
            func=write_file,
            name="write_file",
            description=(
                "Create or overwrite a whole file with `content`. Use for a new"
                " file (a test, a scratch script) or a full rewrite -- never a"
                " shell heredoc/redirect. Use edit_file for a partial change."
            ),
        ),
    ]


class _JournalCallback(BaseCallbackHandler):
    """Stream a turn's agent activity into the job journal as it happens.

    Tool starts/ends and per-model-call text/thinking are emitted live, so a
    human `icu tail`s the turn and the reentry loop sees journal growth as the
    liveness signal (a turn mid-build advances the journal even without a
    result). Every hook is best-effort: a journal hiccup must never break the
    agent turn, and the tool name is carried start->end by run_id."""

    def __init__(self, journal: Journal, agent: str, turn: int) -> None:
        self._j = journal
        self._agent = agent
        self._turn = turn
        self._tool_names: dict[str, str] = {}  # run_id -> tool name

    def on_tool_start(self, serialized, input_str, *, run_id=None, **kwargs) -> None:
        name = (serialized or {}).get("name", "tool")
        self._tool_names[str(run_id)] = name
        self._emit(EventKind.TOOL_START, name=name, text=_trunc(str(input_str)))

    def on_tool_end(self, output, *, run_id=None, **kwargs) -> None:
        name = self._tool_names.pop(str(run_id), "tool")
        self._emit(EventKind.TOOL_END, name=name, text=_trunc(str(output)))

    def on_llm_end(self, response, **kwargs) -> None:
        # One model call finished: surface its reasoning and response text. The
        # generation carries a message whose content is either a plain string or
        # a list of blocks (Anthropic thinking + text); handle both.
        try:
            for gen in (response.generations or [[]])[0]:
                msg = getattr(gen, "message", None)
                content = getattr(msg, "content", gen.text if gen else "")
                self._emit_content(content)
        except Exception:  # defensive: callback errors must not fail the turn
            pass

    def _emit_content(self, content) -> None:
        if isinstance(content, str):
            if content.strip():
                self._emit(EventKind.MESSAGE, text=_trunc(content))
            return
        for block in content or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype in ("thinking", "reasoning"):
                body = block.get("thinking") or block.get("reasoning") or ""
                if body.strip():
                    self._emit(EventKind.THINKING, text=_trunc(body))
            elif btype == "text":
                body = block.get("text", "")
                if body.strip():
                    self._emit(EventKind.MESSAGE, text=_trunc(body))

    def _emit(self, kind: EventKind, **fields) -> None:
        self._j.emit(kind, agent=self._agent, turn=self._turn, **fields)


class _StopReasonCallback(BaseCallbackHandler):
    """Record the last model call's finish_reason and whether its candidate was
    empty, read at the raw model boundary.

    Why a callback and not state.messages[-1]: a turn that stops without a report
    is exactly the case we most need to diagnose (an empty candidate -- Gemini
    returns zero parts with finish_reason SAFETY/RECITATION/MALFORMED_FUNCTION_CALL
    /MAX_TOKENS, or a zero-part STOP that reads as benign), and _DropEmptyResponses
    strips that empty AIMessage from the returned state, so it is gone by the time
    run_once inspects the result. on_llm_end sees the generation before any
    middleware scrub. Keeps only the latest -- the terminating call is the one
    whose reason explains why the turn produced no report. Best-effort: a parse
    miss must never fail the turn. Skips ceiling-summarization calls (run on the
    filter model inside before_model) so a nested summarize does not overwrite
    the real terminating call's reason before it fires.
    """

    def __init__(self) -> None:
        self.finish_reason = ""
        self.empty = False
        self._skip: set = set()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        if (kwargs.get("metadata") or {}).get("lc_source") == "summarization":
            self._skip.add(run_id)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        if run_id in self._skip:
            self._skip.discard(run_id)
            return
        try:
            for gen in (response.generations or [[]])[0]:
                msg = getattr(gen, "message", None)
                meta = getattr(msg, "response_metadata", None) or {}
                # langchain-google puts it here; some providers use generation_info.
                reason = meta.get("finish_reason") or (
                    (getattr(gen, "generation_info", None) or {}).get("finish_reason")
                )
                if reason:
                    self.finish_reason = str(reason)
                # A zero-part candidate: no tool calls AND no text. STOP with empty
                # content is the silent-dead-turn shape -- benign by finish_reason
                # alone, pathological by content. Track it so run_once can name it.
                tool_calls = getattr(msg, "tool_calls", None) or []
                content = str(getattr(msg, "content", "") or "").strip()
                self.empty = not tool_calls and not content
        except Exception:  # defensive: callback errors must not fail the turn
            pass


class _TurnUsage:
    input_tokens = 0
    output_tokens = 0
    cached_in = 0
    calls = 0
    max_call_input = 0


class LangGraphHarness:
    """Runs one bounded coding turn per run_once against a per-stage thread.

    Async-lazy: MCP sessions and the shared tool set are opened on the first
    turn and held for the process's life; call `aclose()` on shutdown. Build one
    instance and share it across jobs. The application injects `system_prompt`
    (identity + conventions + skill index) and `schemas` (agent-name -> verdict
    Report subclass).
    """

    def __init__(
        self,
        common: CommonConfig,
        *,
        system_prompt: str = _DEFAULT_SYSTEM,
        schemas: Mapping[str, type[Report]] | None = None,
        coding_model_key: str = "default",
        coding_tool_groups: tuple[str, ...] = ("read", "active"),
        sandboxed_tool_groups: tuple[str, ...] = ("read",),
        shell_timeout_s: float = 300.0,
    ) -> None:
        self._common = common
        self._model_id = common.models.get(coding_model_key) or common.models.get(
            "default", ""
        )
        if not self._model_id:
            raise ValueError(
                f"no model configured: [models].{coding_model_key} or .default"
            )
        self._groups = list(coding_tool_groups)
        # Sandboxed jobs get a narrower MCP surface: `active`-group tools
        # (run_d8, benchmarks) execute in this trusted process outside any
        # sandbox and take arbitrary paths, so a sandboxed job keeps only the
        # read group and runs d8/tests/gdb/perf through its confined shell.
        self._sandboxed_groups = list(sandboxed_tool_groups)
        self._shell_timeout_s = shell_timeout_s
        self._schemas: dict[str, type[Report]] = dict(schemas or {})
        self._system_base = system_prompt
        self._system = system_prompt
        self._tokens = TokenLog(common.token_db_path)
        self._stack = contextlib.AsyncExitStack()
        self._toolset: MCPToolset | None = None
        self._v8_tools: list = []
        self._v8_tools_sandboxed: list = []
        self._graphs: OrderedDict[str, object] = OrderedDict()
        self._init_lock = asyncio.Lock()

    async def _ensure_init(self) -> None:
        if self._toolset is not None:
            return
        async with self._init_lock:
            if self._toolset is not None:
                return
            apply_gcp_env_defaults(self._common.gcp)
            ts = MCPToolset(self._common.mcp_servers, self._common.tool_groups)
            await self._stack.enter_async_context(ts)
            self._v8_tools = ts.tools_for(ts.resolve_patterns(self._groups))
            self._v8_tools_sandboxed = ts.tools_for(
                ts.resolve_patterns(self._sandboxed_groups)
            )
            if ts.instructions:
                self._system = (
                    f"{self._system_base}\n\n## MCP server instructions\n\n"
                    f"{ts.instructions}"
                )
            self._toolset = ts
            log.info(
                "langgraph harness: model=%s mcp-tools=%d",
                self._model_id,
                len(self._v8_tools),
            )

    async def aclose(self) -> None:
        await self._stack.aclose()
        self._tokens.close()

    def _graph_for(
        self,
        thread_id: str,
        workdir: Path,
        schema: type[Report],
        sandbox: Sandbox | None = None,
    ):
        # A sandbox is a per-job constant, so the thread cache never mixes
        # sandboxed and unsandboxed tool sets for one thread.
        if thread_id in self._graphs:
            self._graphs.move_to_end(thread_id)
            return self._graphs[thread_id]

        from langchain.agents import create_agent
        from langchain.agents.middleware import ModelCallLimitMiddleware
        from langchain.agents.structured_output import ToolStrategy
        from langgraph.checkpoint.memory import InMemorySaver

        from airc_core import (
            CallBudgetMiddleware,
            RequireStructuredResultMiddleware,
        )

        mcp_tools = self._v8_tools_sandboxed if sandbox is not None else self._v8_tools
        tools = [
            *mcp_tools,
            *_worktree_tools(workdir, self._shell_timeout_s, sandbox),
        ]
        mw = base_middleware(
            self._model_id,
            self._system,
            tools,
            # Compact old history at the ceiling on the cheap model; falls back to
            # the agent model if no filter model is configured.
            summarizer_model_id=self._common.models.get("filter")
            or self._common.models.get("default"),
        )
        if cache := growing_cache_middleware(
            self._model_id,
            self._system,
            tools,
            self._common.caching_explicit,
            self._common.cache_ttl_minutes,
            _MAX_MODEL_CALLS,
        ):
            mw.append(cache)
        # Order matters: the after_model chain runs last-appended-first, so
        # RequireStructuredResultMiddleware is listed BEFORE the governors to make
        # its hook run AFTER theirs -- the call counters increment first, then it
        # inspects the finished turn and, on a plain-text ending with no report
        # tool call, jumps back to re-ask (bounded by _MAX_REASKS) instead of
        # letting the turn end as a silent no-result the reentry loop scores dead.
        # It composes with the cap: a re-ask re-enters ModelCallLimitMiddleware
        # .before_model, so an exhausted turn still ends at the cap.
        mw += [
            RequireStructuredResultMiddleware(
                _REQUIRE_RESULT_REMINDER, max_reasks=_MAX_REASKS
            ),
            CallBudgetMiddleware(
                [(_SOFT_NUDGE_CALLS, _SOFT_NUDGE), (_HARD_NUDGE_CALLS, _HARD_NUDGE)]
            ),
            ModelCallLimitMiddleware(run_limit=_MAX_MODEL_CALLS, exit_behavior="end"),
        ]
        # ToolStrategy names the structured-output tool after the schema class
        # (DraftReport/ReviewReport/...). Override it to one fixed name so every
        # stage's prompt can name the exact tool -- schema_specs[0].name is what
        # langchain builds the tool with (structured_output.py: name=spec.name).
        # The schema (its fields) is unchanged, so structured_response is still
        # parsed into the Report subclass.
        strategy = ToolStrategy(
            schema=schema, handle_errors=True, tool_message_content=_REPORT_ACK
        )
        strategy.schema_specs[0].name = REPORT_TOOL_NAME
        graph = create_agent(
            make_model(self._model_id),
            tools=tools,
            system_prompt=self._system,
            middleware=mw,
            checkpointer=InMemorySaver(),  # per-thread; freed when the graph evicts
            response_format=strategy,
        ).with_config({"recursion_limit": _RECURSION_LIMIT})

        self._graphs[thread_id] = graph
        while len(self._graphs) > _MAX_GRAPHS:
            self._graphs.popitem(last=False)  # LRU; its InMemorySaver is GC'd
        return graph

    async def run_once(
        self,
        *,
        prompt_path: Path,
        workdir: Path,
        result_path: Path,
        timeout_s: float,
        agent: str = "",
        resume: bool = False,
        resume_prompt: str = "",
        casefile: Path | None = None,
        journal: Journal | None = None,
        sandbox: Sandbox | None = None,
    ) -> HarnessRun:
        await self._ensure_init()
        # One thread per stage-loop (its control dir); stable across the loop's
        # turns so context and cache warmth accumulate.
        thread_id = str(result_path.parent)
        schema = self._schemas.get(agent, Report)
        # Whether the thread actually exists BEFORE _graph_for may (re)build it.
        # Thread continuity is a cache-warmth optimization, never correctness: if
        # the graph was LRU-evicted or the process restarted, the InMemorySaver is
        # gone and "continue where you left off" would address an empty thread. So
        # resume only sends the short continue prompt when the thread is really
        # live; otherwise it re-sends the full prompt (the worktree + casefile
        # carry the real state), with any phase instruction (reflect/final)
        # appended so a cache eviction cannot swallow it.
        thread_live = thread_id in self._graphs
        graph = self._graph_for(thread_id, workdir, schema, sandbox)

        default_resume = (
            "Continue from where you left off -- your previous report was a"
            " checkpoint, not completion, so do the next concrete step of work"
            f" now. When you pause or finish, call the `{REPORT_TOOL_NAME}` tool."
        )
        if resume and thread_live:
            turn = resume_prompt or default_resume
        elif resume:
            full = prompt_path.read_text()
            turn = f"{full}\n\n{resume_prompt}" if resume_prompt else full
        else:
            turn = prompt_path.read_text()
        turn_index = (
            int(result_path.stem.split(".")[-1]) if "." in result_path.stem else 0
        )
        if journal is not None:
            journal.emit(EventKind.TURN, agent=agent or "turn", turn=turn_index)
        log_path = result_path.with_suffix(".log")
        result_path.parent.mkdir(parents=True, exist_ok=True)

        from langchain_core.callbacks import UsageMetadataCallbackHandler

        from airc_core.agent import _CallTrace

        usage_cb = UsageMetadataCallbackHandler()
        trace_cb = _CallTrace(agent or "turn", "turn")
        stop_cb = _StopReasonCallback()
        callbacks = [usage_cb, trace_cb, stop_cb]
        if journal is not None:
            callbacks.append(_JournalCallback(journal, agent or "turn", turn_index))
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": callbacks,
            "recursion_limit": _RECURSION_LIMIT,
        }
        # structured_response is a checkpointed channel with no cross-turn reducer
        # that resets it. A resumed turn that ends without calling the report tool
        # (the model just acknowledges the checkpoint and stops -- no tool call, so
        # the graph takes the classic 0-tool-calls exit) never overwrites it, so we
        # would read the PRIOR turn's report back as this turn's verdict: a valid
        # CONTINUE that the loop resets its dead-turn cap on, spinning to max_iters
        # doing nothing. Blank the channel before a live-thread resume so only a
        # report produced THIS turn is accepted; a dead resume then reads None and
        # falls into the loop's dead-turn guard. Can't clear via ainvoke input --
        # structured_response is OmitFromInput -- so update the checkpoint directly.
        #
        # No analogous clear for RequireStructuredResultMiddleware: its re-ask
        # bound is an UntrackedValue counter (per-turn, resets each ainvoke), so
        # the budget is already fresh on resume -- nothing to clear here.
        if resume and thread_live:
            await graph.aupdate_state(config, {"structured_response": None})
        start = time.monotonic()
        result: AgentResult | None = None
        code = 0
        try:
            state = await asyncio.wait_for(
                graph.ainvoke(
                    {"messages": [{"role": "user", "content": turn}]}, config
                ),
                timeout=timeout_s,
            )
            report = state.get("structured_response")
            if isinstance(report, Report):
                result = to_result(report)
            else:
                # No structured report this turn. Usually benign (ran out of
                # model-call budget mid-work; the loop resumes the thread), but a
                # provider-side empty candidate lands here too and looks identical
                # unless we name the finish_reason -- a deterministic SAFETY/
                # RECITATION/MALFORMED_FUNCTION_CALL block reproduces every resume
                # and burns the dead-turn cap. A zero-part STOP is the same failure
                # wearing a benign label: finish_reason reads STOP (so an allowlist
                # on reason alone skips it) but the candidate had no text and no
                # tool calls, leaving the turn with nothing to report -- the
                # silent-dead-turn shape. Surface both the reason and the empty
                # flag so the log and the loop's abandon reason say why, instead
                # of an opaque "exit 1".
                code = 1
                benign = {"STOP", "TOOL_CALLS", "TOOL_CALL", "END_TURN"}
                reason = stop_cb.finish_reason
                if stop_cb.empty or (reason and reason.upper() not in benign):
                    log.warning(
                        "%s: turn produced no report; finish_reason=%s%s",
                        agent or "turn",
                        reason or "(none)",
                        " (empty candidate)" if stop_cb.empty else "",
                    )
        except asyncio.TimeoutError:
            log.warning("%s: turn timed out after %.0fs", agent or "turn", timeout_s)
            code = -1
        except EmptyCandidateError:
            # _EmptyCandidateRetry retried the zero-part candidate through the
            # shared backoff and exhausted it. Surface a named diagnostic (not a
            # generic traceback) so the loop's dead-turn abandon says why. The
            # finish_reason from the last empty model call is already in stop_cb.
            log.warning(
                "%s: empty candidate; retries exhausted -- finish_reason=%s",
                agent or "turn",
                stop_cb.finish_reason or "unknown",
            )
            code = 1
        except Exception:
            log.exception("%s: turn errored", agent or "turn")
            code = -1

        usage = self._aggregate(usage_cb, trace_cb)
        self._tokens.add(
            zlib.crc32(thread_id.encode()),
            agent or "turn",
            "turn",
            usage.input_tokens,
            usage.output_tokens,
            usage.cached_in,
            self._model_id,
            model_calls=usage.calls,
            max_call_input_tokens=usage.max_call_input,
        )
        if journal is not None:
            journal.emit(
                EventKind.USAGE,
                agent=agent or "turn",
                turn=turn_index,
                data={
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cached_in": usage.cached_in,
                    "model_calls": usage.calls,
                    # model + max_call_input let an out-of-process consumer (the
                    # sandboxed worker's runner) credit the ledger fully from the
                    # journal, since tokens.db is never bound into the box.
                    "model": self._model_id,
                    "max_call_input": usage.max_call_input,
                    # Only when the turn produced no report -- on a clean report
                    # the reason is STOP/TOOL_CALLS and carries no diagnostic value.
                    # empty_candidate names the silent-dead-turn shape (a zero-part
                    # STOP) that finish_reason alone cannot.
                    **(
                        {
                            "finish_reason": stop_cb.finish_reason,
                            "empty_candidate": stop_cb.empty,
                        }
                        if result is None and (stop_cb.finish_reason or stop_cb.empty)
                        else {}
                    ),
                },
            )
            if result is not None:
                journal.emit(
                    EventKind.REPORT,
                    agent=agent or "turn",
                    turn=turn_index,
                    name=result.disposition.value,
                    text=result.summary or result.reason,
                )
                # A separate event so friction is greppable across jobs and does
                # not get folded into the report line; only when the agent had
                # something to flag, to keep quiet turns quiet.
                if result.friction:
                    journal.emit(
                        EventKind.FRICTION,
                        agent=agent or "turn",
                        turn=turn_index,
                        text=result.friction,
                    )
        if result is not None:
            with contextlib.suppress(OSError):
                result_path.write_text(result.model_dump_json())
        return HarnessRun(
            exit_code=code,
            result=result,
            log_path=log_path,
            duration_s=time.monotonic() - start,
            finish_reason=stop_cb.finish_reason,
            empty_candidate=stop_cb.empty,
        )

    @staticmethod
    def _aggregate(usage_cb, trace_cb) -> _TurnUsage:
        u = _TurnUsage()
        for meta in usage_cb.usage_metadata.values():
            u.input_tokens += meta.get("input_tokens", 0)
            u.output_tokens += meta.get("output_tokens", 0)
            u.cached_in += meta.get("input_token_details", {}).get("cache_read", 0)
        u.calls = trace_cb.calls
        u.max_call_input = trace_cb.max_input_tokens
        return u

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""A durable thread stays live across a restart, and its verdict does not.

run_once decides two things from `thread_live`: whether to send the short
continue prompt or the whole original one, and whether to blank
structured_response before invoking. Both were keyed on a per-process dict,
which the durable saver (phase 5) turned into the wrong question -- after a
restart the checkpoint is intact but the dict is empty, so the full prompt went
into a thread that already contained it AND the stale-verdict clear was skipped.
A restarted turn that made no report then read its PRE-CRASH verdict back as its
own; if that was COMPLETE, a turn that did nothing persisted a success.
"""

from pathlib import Path

from airc_core import CommonConfig
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from deepagent import REPORT_TOOL_NAME, LangGraphHarness, Report


class _Scripted(GenericFakeChatModel):
    """Reports once if asked to, then only ever answers in plain text -- the
    ghost-turn shape that leaves structured_response holding the old report."""

    def __init__(self, report_first: bool):
        super().__init__(messages=iter([]))
        self._report_first = report_first
        self._turns: list[str] = []

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self._turns.append(str(messages[-1].content))
        if self._report_first and len(self._turns) == 1:
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": REPORT_TOOL_NAME,
                        "args": {"disposition": "complete", "summary": "pre-crash"},
                        "id": "call_1",
                    }
                ],
            )
        else:
            msg = AIMessage(content="acknowledged")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kw):
        return self

    @property
    def turns(self) -> list[str]:
        return self._turns


def _graph(model, saver):
    strat = ToolStrategy(schema=Report, handle_errors=True)
    strat.schema_specs[0].name = REPORT_TOOL_NAME
    return create_agent(
        model, tools=[], response_format=strat, checkpointer=saver
    ).with_config({"recursion_limit": 100})


def _harness(tmp_path: Path, model, saver) -> LangGraphHarness:
    """A harness whose graph is scripted but whose run_once is the real one --
    the thread-liveness decision is what is under test."""
    common = CommonConfig()
    common.models = {"default": "google_genai:gemini-3.1-flash-lite"}
    common.token_db_path = tmp_path / "tokens.db"
    h = LangGraphHarness(common)

    async def _no_init() -> None:
        return None

    h._ensure_init = _no_init
    graph = _graph(model, saver)
    h._graph_for = lambda *a, **k: graph
    return h


def _saver(db: Path):
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    conn = aiosqlite.connect(str(db))
    return AsyncSqliteSaver(conn), conn


async def _turn(h, tmp_path: Path, *, resume: bool, n: int):
    prompt = tmp_path / "JOB.md"
    prompt.write_text("THE WHOLE ORIGINAL PROMPT")
    return await h.run_once(
        prompt_path=prompt,
        workdir=tmp_path,
        result_path=tmp_path / f"result.{n}.json",
        timeout_s=30,
        agent="draft",
        resume=resume,
    )


async def test_a_restart_does_not_inherit_the_pre_crash_verdict(tmp_path):
    db = tmp_path / "checkpoints.db"
    m1 = _Scripted(report_first=True)
    saver, conn = _saver(db)
    h1 = _harness(tmp_path, m1, saver)
    run = await _turn(h1, tmp_path, resume=False, n=0)
    assert run.result is not None and run.result.summary == "pre-crash"
    await conn.close()

    # The restart: a new harness (empty _graphs) over the same checkpoint file.
    m2 = _Scripted(report_first=False)  # this turn reports nothing
    saver2, conn2 = _saver(db)
    h2 = _harness(tmp_path, m2, saver2)
    run2 = await _turn(h2, tmp_path, resume=True, n=1)
    await conn2.close()

    assert run2.result is None, "a turn that made no report has no verdict"
    assert run2.exit_code == 1  # the loop's dead-turn path, not a false success
    # ...and the thread it resumed was recognized as live, so the turn is the
    # short continue prompt rather than the whole job re-sent into a thread that
    # already holds it.
    assert "THE WHOLE ORIGINAL PROMPT" not in m2.turns[0]
    assert "Continue from where you left off" in m2.turns[0]


async def test_a_genuinely_new_thread_still_gets_the_full_prompt(tmp_path):
    # The other side: liveness must be read from the checkpoint, not assumed. A
    # resume whose thread is gone (a fresh DB, an LRU eviction, a saver that
    # degraded to memory) has to re-send the prompt -- "continue where you left
    # off" would address an empty conversation.
    m = _Scripted(report_first=False)
    saver, conn = _saver(tmp_path / "empty.db")
    h = _harness(tmp_path, m, saver)
    await _turn(h, tmp_path, resume=True, n=0)
    await conn.close()
    assert "THE WHOLE ORIGINAL PROMPT" in m.turns[0]

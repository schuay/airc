# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""StructuredTaskRunner: one typed, tool-less, stateless model task."""

import pytest
from airc_core import CommonConfig, StructuredTaskError, StructuredTaskRunner
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel


class Verdict(BaseModel):
    verdict: str


class _ScriptedModel(BaseChatModel):
    scripted: list = []
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "structured-task-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        spec = self.scripted[min(self.calls, len(self.scripted) - 1)]
        object.__setattr__(self, "calls", self.calls + 1)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(**spec))])


def _tool(verdict: str = "infrastructure") -> dict:
    return {
        "content": "",
        "tool_calls": [
            {"name": "Verdict", "args": {"verdict": verdict}, "id": "result"}
        ],
    }


def _runner(tmp_path, monkeypatch, scripted, **kwargs):
    from airc_core import structured

    model = _ScriptedModel(scripted=scripted)
    monkeypatch.setattr(structured, "make_model", lambda _model_id: model)
    common = CommonConfig(
        models={"judge": "test:model"}, token_db_path=tmp_path / "tokens.db"
    )
    return StructuredTaskRunner(common, model_key="judge", **kwargs), model


async def test_returns_validated_schema(tmp_path, monkeypatch):
    runner, model = _runner(tmp_path, monkeypatch, [_tool()])
    result = await runner.run(
        "classify", schema=Verdict, system_prompt="judge", task="triage"
    )
    assert result == Verdict(verdict="infrastructure")
    assert model.calls == 1
    runner.close()


async def test_plain_text_gets_a_bounded_structured_reask(tmp_path, monkeypatch):
    runner, model = _runner(
        tmp_path, monkeypatch, [{"content": "I think it is infra"}, _tool()]
    )
    assert (await runner.run("classify", schema=Verdict)).verdict == "infrastructure"
    assert model.calls == 2
    runner.close()


async def test_missing_result_fails_instead_of_becoming_a_clean_verdict(
    tmp_path, monkeypatch
):
    runner, model = _runner(
        tmp_path,
        monkeypatch,
        [{"content": "only prose"}],
        max_model_calls=3,
        max_reasks=2,
    )
    with pytest.raises(StructuredTaskError, match="produced no valid Verdict"):
        await runner.run("classify", schema=Verdict)
    assert model.calls == 3
    runner.close()


def test_requires_a_configured_model(tmp_path):
    with pytest.raises(ValueError, match="no model configured"):
        StructuredTaskRunner(CommonConfig(token_db_path=tmp_path / "tokens.db"))

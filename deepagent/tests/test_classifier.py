# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Classifier metadata must survive the coding graph and worker contract."""

import pytest
from airc_core import CommonConfig, ModelProfile
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from deepagent import Journal, LoopSpec, Report, read_outcome, run_loop_from_spec
from deepagent import langgraph_harness as lh


class StoppedModel(GenericFakeChatModel):
    def __init__(self, reason):
        super().__init__(messages=iter([]))
        self._reason = reason
        self._calls = 0

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self._calls += 1
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        response_metadata={"stop_reason": self._reason},
                        usage_metadata={
                            "input_tokens": 100,
                            "output_tokens": 1,
                            "total_tokens": 101,
                        },
                    )
                )
            ]
        )


@pytest.mark.parametrize("reason", ["refusal", "SAFETY", "RECITATION"])
async def test_classifier_exits_once_and_accounts_usage(tmp_path, monkeypatch, reason):
    model = StoppedModel(reason)
    monkeypatch.setattr(lh, "make_model", lambda *a, **k: model)
    common = CommonConfig()
    common.models = {"default": "anthropic:test"}
    common.token_db_path = tmp_path / "tokens.db"
    harness = lh.LangGraphHarness(common)
    # Keep the real middleware stack, including the empty-candidate retry.
    common.models.pop("default")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("continue")
    spec = LoopSpec(
        prompt_path=str(prompt),
        workdir=str(tmp_path),
        control_dir=str(tmp_path / "goal"),
        journal_path=str(tmp_path / "events.jsonl"),
        casefile=str(tmp_path),
        agent="test",
        max_iters=5,
    )
    try:
        result = await run_loop_from_spec(harness, spec)
    finally:
        await harness.aclose()
    assert model._calls == 1
    assert result.classifier_stop == reason.lower()
    assert read_outcome(tmp_path / "goal") == result
    usage = [
        e.data["usage"] for e in Journal.read(spec.journal_path) if e.kind == "usage"
    ]
    assert sum(u["calls"] for u in usage) == 1
    assert sum(u["input"] for u in usage) == 100
    assert "classifier_stop" not in Report.model_fields


def test_saved_profile_reaches_adapter(tmp_path, monkeypatch):
    common = CommonConfig()
    common.models = {"default": "anthropic:primary", "alternate": "anthropic:changed"}
    common.token_db_path = tmp_path / "tokens.db"
    saved = ModelProfile(key="alternate", id="anthropic:saved", effort="high")
    seen = []
    monkeypatch.setattr(
        lh, "make_model", lambda *a, **k: seen.append((a, k)) or StoppedModel("refusal")
    )
    monkeypatch.setattr(lh, "base_middleware", lambda *a, **k: [])
    harness = lh.LangGraphHarness(common, coding_profile=saved)
    harness._graph_for("t", tmp_path, Report)
    assert seen == [((saved.id,), {"effort": "high"})]

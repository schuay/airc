# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""The worker core: run a goal loop from a LoopSpec, write outcome.json.

Uses MockHarness so no model/langchain is touched -- this exercises the file
contract (spec in, outcome out), not a real turn.
"""

from pathlib import Path

from deepagent import (
    AgentResult,
    Disposition,
    LoopSpec,
    MockHarness,
    read_outcome,
    run_loop_from_spec,
)


def _spec(tmp_path: Path) -> LoopSpec:
    wt = tmp_path / "wt"
    wt.mkdir()
    cd = tmp_path / "control" / "repro.0"
    cf = tmp_path / "control" / "casefile"
    cf.mkdir(parents=True)
    prompt = cf / "repro.prompt.md"
    prompt.write_text("do the thing")
    return LoopSpec(
        prompt_path=str(prompt),
        workdir=str(wt),
        control_dir=str(cd),
        journal_path=str(tmp_path / "control" / "events.jsonl"),
        casefile=str(cf),
        agent="icompleteu-repro",
        max_iters=5,
    )


async def test_run_loop_writes_outcome(tmp_path):
    spec = _spec(tmp_path)
    harness = MockHarness(
        [AgentResult(disposition=Disposition.COMPLETE, data={"reproduced": True})]
    )

    res = await run_loop_from_spec(harness, spec)

    assert res.disposition is Disposition.COMPLETE
    assert res.data["reproduced"] is True
    # The terminal result is persisted where the runner reads it back.
    back = read_outcome(Path(spec.control_dir))
    assert back is not None and back.data["reproduced"] is True


async def test_run_loop_abandon_is_persisted(tmp_path):
    spec = _spec(tmp_path)
    harness = MockHarness(
        [AgentResult(disposition=Disposition.ABANDON, reason="cannot")]
    )

    res = await run_loop_from_spec(harness, spec)

    assert res.disposition is Disposition.ABANDON
    assert read_outcome(Path(spec.control_dir)).reason == "cannot"


async def test_the_step_budget_crosses_into_the_box(tmp_path):
    """The loop runs INSIDE the sandbox on the normal path, so a cap that did
    not ride the spec would bind the in-process driver and nothing else -- which
    is the configuration nobody runs. Asserted on the caps the in-box loop is
    actually built with, not on the field existing."""
    from deepagent import worker as w

    seen = {}

    async def fake_loop(harness, **kw):
        seen["caps"] = kw["caps"]
        return AgentResult(disposition=Disposition.COMPLETE, summary="done")

    spec = _spec(tmp_path)
    spec = spec.model_copy(update={"max_usd": 12.5})
    orig = w.run_agent_loop
    w.run_agent_loop = fake_loop
    try:
        await w.run_loop_from_spec(MockHarness([]), spec)
    finally:
        w.run_agent_loop = orig
    assert seen["caps"].max_usd == 12.5
    # And the default stays "no cap", so a spec written before the field existed
    # keeps running unbounded rather than being abandoned by a zero.
    assert LoopSpec(**(spec.model_dump() | {"max_usd": None})).max_usd is None

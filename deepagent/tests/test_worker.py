"""The worker core: run a goal loop from a LoopSpec, write outcome.json.

Uses MockHarness so no model/langchain is touched -- this exercises the file
contract (spec in, outcome out) and the sandbox=None in-box tool posture, not a
real turn.
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


async def test_run_loop_runs_tools_unwrapped(tmp_path):
    # In the box the harness must run each turn with sandbox=None -- the process
    # boundary is the confinement, so no per-call wrapper.
    spec = _spec(tmp_path)
    harness = MockHarness([AgentResult(disposition=Disposition.COMPLETE)])

    await run_loop_from_spec(harness, spec)

    assert harness.sandboxes == [None]


async def test_run_loop_abandon_is_persisted(tmp_path):
    spec = _spec(tmp_path)
    harness = MockHarness(
        [AgentResult(disposition=Disposition.ABANDON, reason="cannot")]
    )

    res = await run_loop_from_spec(harness, spec)

    assert res.disposition is Disposition.ABANDON
    assert read_outcome(Path(spec.control_dir)).reason == "cannot"

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field

from deepagent import (
    AgentResult,
    Disposition,
    HarnessRun,
    Journal,
    LoopCaps,
    MockHarness,
    run_agent_loop,
)
from deepagent.journal import EventKind


async def _run(results, *, max_iters=20, tmp):
    h = MockHarness(results=results)
    out = await run_agent_loop(
        h,
        prompt_path=tmp / "p.md",
        workdir=tmp / "wt",
        control_dir=tmp / "ctl",
        caps=LoopCaps(max_iters=max_iters),
        agent="agent",
    )
    return h, out


async def test_stops_on_complete(tmp_path):
    _, out = await _run(
        [AgentResult(disposition=Disposition.COMPLETE, summary="done")], tmp=tmp_path
    )
    assert out.disposition is Disposition.COMPLETE
    assert out.summary == "done"


async def test_loops_through_continue(tmp_path):
    h, out = await _run(
        [
            AgentResult(disposition=Disposition.CONTINUE),
            AgentResult(disposition=Disposition.CONTINUE),
            AgentResult(disposition=Disposition.COMPLETE, data={"verdict": "accept"}),
        ],
        tmp=tmp_path,
    )
    assert out.disposition is Disposition.COMPLETE
    assert out.data["verdict"] == "accept"
    assert h.calls == 3
    # turn 0 is a fresh send; the rest resume
    assert [inv[1] for inv in h.invocations] == [False, True, True]


async def test_cap_synthesizes_abandon(tmp_path):
    _, out = await _run(
        [AgentResult(disposition=Disposition.CONTINUE)], max_iters=3, tmp=tmp_path
    )
    assert out.disposition is Disposition.ABANDON
    assert "did not converge" in out.reason


async def test_resume_prompts_escalate(tmp_path):
    # A loop that never converges walks plain-continue -> reflection checkpoint
    # -> final-turn verdict force, with turn awareness throughout.
    h = MockHarness(results=[AgentResult(disposition=Disposition.CONTINUE)])
    await run_agent_loop(
        h,
        prompt_path=tmp_path / "p.md",
        workdir=tmp_path / "wt",
        control_dir=tmp_path / "ctl",
        caps=LoopCaps(max_iters=8, checkpoint_turn=4),
        agent="agent",
    )
    rp = h.resume_prompts
    assert rp[0] == ""  # turn 0 sends the task, not a resume prompt
    assert "turn 2 of 8" in rp[1]  # turn awareness
    assert "Step back" in rp[4] and "achievability" in rp[4]  # checkpoint
    assert "final turn" in rp[7]  # forced verdict, not a silent ABANDON
    # The checkpoint fires once, only at its turn.
    assert sum("Step back" in p for p in rp) == 1


async def test_no_checkpoint_when_unset(tmp_path):
    h = MockHarness(results=[AgentResult(disposition=Disposition.CONTINUE)])
    await run_agent_loop(
        h,
        prompt_path=tmp_path / "p.md",
        workdir=tmp_path / "wt",
        control_dir=tmp_path / "ctl",
        caps=LoopCaps(max_iters=5),  # checkpoint_turn None -> no reflection
        agent="agent",
    )
    assert not any("Step back" in p for p in h.resume_prompts)


async def test_terminal_abandon_passes_through(tmp_path):
    _, out = await _run(
        [AgentResult(disposition=Disposition.ABANDON, reason="cannot fix")],
        tmp=tmp_path,
    )
    assert out.disposition is Disposition.ABANDON
    assert out.reason == "cannot fix"


@dataclass
class _ScriptedHarness:
    """Per-turn (result, advances_journal) script, to exercise liveness.

    Emits the harness's own TURN/USAGE bookkeeping every turn like the real
    run_once does -- unconditionally, even on a dead turn -- so the liveness
    tests exercise the production event pattern. A turn that "advances" also
    emits a work event (TOOL_START); a dead turn emits only the bookkeeping.
    A liveness signal that keyed on raw journal length rather than work events
    would see every turn as alive here and never abandon (the P0)."""

    # (AgentResult | None, advances_journal), optionally + finish_reason,
    # optionally + empty_candidate.
    script: list[tuple]
    calls: int = 0
    invocations: list = field(default_factory=list)

    async def run_once(
        self, *, result_path, journal=None, agent="", resume=False, **kw
    ) -> HarnessRun:
        entry = self.script[min(self.calls, len(self.script) - 1)]
        result, advance = entry[0], entry[1]
        finish_reason = entry[2] if len(entry) > 2 else ""
        empty_candidate = entry[3] if len(entry) > 3 else False
        self.calls += 1
        self.invocations.append(resume)
        if journal is not None:
            journal.emit(EventKind.TURN, agent=agent, turn=self.calls - 1)
        if advance and journal is not None:
            journal.emit(EventKind.TOOL_START, agent=agent, name="shell")
        if result is not None:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(result.model_dump_json())
        if journal is not None:
            journal.emit(EventKind.USAGE, agent=agent, turn=self.calls - 1)
        return HarnessRun(
            exit_code=0 if result else 1,
            result=result,
            log_path=result_path,
            duration_s=0.0,
            finish_reason=finish_reason,
            empty_candidate=empty_candidate,
        )


async def test_journal_growth_is_liveness_free_retry(tmp_path):
    # A turn that produces no result but advances the journal (a build still
    # running) is alive: retried for free, not counted against the dead-turn cap.
    journal = Journal(tmp_path / "events.jsonl")
    h = _ScriptedHarness(
        script=[
            (None, True),  # no result, but journal advanced -> alive, free retry
            (None, True),  # again alive
            (AgentResult(disposition=Disposition.COMPLETE, summary="ok"), True),
        ]
    )
    out = await run_agent_loop(
        h,
        prompt_path=tmp_path / "p.md",
        workdir=tmp_path / "wt",
        control_dir=tmp_path / "ctl",
        caps=LoopCaps(max_iters=10, no_result_cap=2),
        journal=journal,
    )
    assert out.disposition is Disposition.COMPLETE
    assert h.calls == 3  # the two empty-but-alive turns did not abandon


async def test_an_exhausted_empty_candidate_is_not_liveness(tmp_path):
    # A turn that advanced the journal AND ended in an exhausted empty candidate
    # is dead, not alive. progress is measured against the START of the turn, so
    # a tool call minutes before the model went silent would otherwise score it
    # alive and reset the streak -- observed in the wild as an unbounded grind
    # (two turns, ~5.5 min each, the second empty from its first call).
    journal = Journal(tmp_path / "events.jsonl")
    h = _ScriptedHarness(script=[(None, True, "STOP", True)])
    out = await run_agent_loop(
        h,
        prompt_path=tmp_path / "p.md",
        workdir=tmp_path / "wt",
        control_dir=tmp_path / "ctl",
        caps=LoopCaps(max_iters=10, no_result_cap=2),
        journal=journal,
    )
    assert out.disposition is Disposition.ABANDON
    assert "empty candidate" in out.reason
    assert h.calls == 2  # bounded by no_result_cap, not grinding to max_iters
    assert journal.progress > 0  # it DID advance -- and was still scored dead


async def test_dead_turns_without_journal_growth_abandon(tmp_path):
    # No result and no WORK event -- only the harness's own TURN/USAGE
    # bookkeeping (which _ScriptedHarness now emits every turn, as the real
    # harness does). The dead-turn cap must still fire: the P0 was that the
    # liveness signal keyed on raw journal length, which those two bookkeeping
    # events grow every turn, so the cap never fired in production.
    journal = Journal(tmp_path / "events.jsonl")
    h = _ScriptedHarness(script=[(None, False)])  # never does work
    out = await run_agent_loop(
        h,
        prompt_path=tmp_path / "p.md",
        workdir=tmp_path / "wt",
        control_dir=tmp_path / "ctl",
        caps=LoopCaps(max_iters=10, no_result_cap=2),
        journal=journal,
    )
    assert out.disposition is Disposition.ABANDON
    assert "no valid result" in out.reason  # the dead-turn reason, not max-iters
    assert h.calls == 2  # hit no_result_cap, well before max_iters=10
    # Sanity: the bookkeeping DID grow the raw count, proving the cap now keys
    # on work events (progress), not length.
    assert journal.count > journal.progress


async def test_dead_turn_abandon_names_finish_reason(tmp_path):
    # A provider-side empty candidate reproduces every resume (deterministic
    # SAFETY/RECITATION block); the abandon reason must name it so the failure is
    # not an opaque "exit 1" the operator cannot act on.
    journal = Journal(tmp_path / "events.jsonl")
    h = _ScriptedHarness(script=[(None, False, "SAFETY")])
    out = await run_agent_loop(
        h,
        prompt_path=tmp_path / "p.md",
        workdir=tmp_path / "wt",
        control_dir=tmp_path / "ctl",
        caps=LoopCaps(max_iters=10, no_result_cap=2),
        journal=journal,
    )
    assert out.disposition is Disposition.ABANDON
    assert "finish_reason=SAFETY" in out.reason


async def test_interjection_reaches_the_next_turn(tmp_path):
    # The orchestrator learns something after a goal started (new review
    # comments). It must reach the agent at the next turn boundary -- not
    # interrupt the running turn, and not wait for the goal to finish.
    news = ["", "NEW REVIEW COMMENTS: use a scope here"]
    harness = MockHarness(
        results=[
            AgentResult(disposition=Disposition.CONTINUE),
            AgentResult(disposition=Disposition.COMPLETE),
        ]
    )
    await run_agent_loop(
        harness,
        prompt_path=tmp_path / "p.md",
        workdir=tmp_path,
        control_dir=tmp_path / "ctl",
        caps=LoopCaps(max_iters=3, timeout_s=1.0),
        interject=lambda: news.pop(0) if news else "",
    )
    # Turn 0 saw nothing; turn 1 carries the news APPENDED to its resume prompt,
    # so the loop's own turn-awareness framing is not replaced by it.
    assert "REVIEW COMMENTS" not in harness.resume_prompts[0]
    assert "NEW REVIEW COMMENTS: use a scope here" in harness.resume_prompts[1]
    assert "Continue from where you left off" in harness.resume_prompts[1]


async def test_interjection_on_the_first_turn(tmp_path):
    # News already waiting when the goal starts rides turn 0, whose resume
    # prompt is empty -- otherwise a comment that landed between steps would sit
    # unseen until the second turn.
    harness = MockHarness(results=[AgentResult(disposition=Disposition.COMPLETE)])
    await run_agent_loop(
        harness,
        prompt_path=tmp_path / "p.md",
        workdir=tmp_path,
        control_dir=tmp_path / "ctl",
        caps=LoopCaps(max_iters=2, timeout_s=1.0),
        interject=lambda: "PENDING: a reviewer commented",
    )
    assert harness.resume_prompts[0] == "PENDING: a reviewer commented"


async def test_no_interjection_leaves_prompts_untouched(tmp_path):
    harness = MockHarness(
        results=[
            AgentResult(disposition=Disposition.CONTINUE),
            AgentResult(disposition=Disposition.COMPLETE),
        ]
    )
    await run_agent_loop(
        harness,
        prompt_path=tmp_path / "p.md",
        workdir=tmp_path,
        control_dir=tmp_path / "ctl",
        caps=LoopCaps(max_iters=3, timeout_s=1.0),
        interject=lambda: "",
    )
    assert harness.resume_prompts[0] == ""
    assert "Continue from where you left off" in harness.resume_prompts[1]

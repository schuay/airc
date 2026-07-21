"""The reentry loop: re-invoke the harness until it signals termination.

This is the heart of the control structure. The caller runs the harness, reads
its `disposition`, and loops while it is CONTINUE; COMPLETE/ABANDON/BLOCKED are
terminal. An iteration cap guarantees the loop terminates even if the agent
never stops signalling CONTINUE -- termination is a property of this code, not a
property the agent has to be trusted for. Continuity between iterations lives in
the worktree and a scratchpad under the control dir; the backend decides whether
a turn resumes the prior context or starts fresh.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from typing import TYPE_CHECKING

from .harness import REPORT_TOOL_NAME, AgentResult, Disposition, Harness
from .journal import Journal

if TYPE_CHECKING:
    from airc_tools.sandbox import Sandbox

log = logging.getLogger(__name__)


# A mid-run reflection turn, injected once at caps.checkpoint_turn. It exists
# because "Continue from where you left off" every turn is pure momentum: a loop
# that cannot reach its goal keeps trying tactical variants instead of stepping
# back to reason about whether the goal is reachable at all. This forces that
# mode switch once, without capping effort -- "one specific new hypothesis" is a
# valid outcome, so a hard-but-reachable goal gets a sharper attempt, not an
# early abort. Deliberately generic (achievability, not domain specifics): the
# stage's own system prompt defines what a negative verdict means (for a repro,
# reproduced=false with evidence).
_REFLECT = (
    "Step back before continuing. You have made several attempts -- assess"
    " achievability, not tactics: is the goal actually reachable here, or are"
    " you accumulating evidence that it cannot be (an upstream guard, a flag"
    " default, a type check, a mistaken premise in the task)? Record that"
    " assessment in your casefile, then decide explicitly: either state one"
    " specific new hypothesis worth another turn, or report your terminal"
    " verdict now with the evidence. Repeating the same approach is not a valid"
    " choice."
)
# The last turn: the loop is about to synthesize an ABANDON, which discards the
# reasoned negative verdict the agent was likely converging toward. Force the
# verdict instead, so the ceiling yields an earned answer, not a non-answer.
_FINAL = (
    "This is your final turn -- do not continue. Commit to a terminal verdict"
    f" now via the `{REPORT_TOOL_NAME}` tool, backed by your evidence: your best"
    " result if you have one, or a reasoned negative verdict (why the goal could"
    " not be achieved) if you do not."
)


@dataclass
class LoopCaps:
    max_iters: int = 20
    timeout_s: float = 3600.0  # per harness invocation
    # Consecutive *dead* turns (no result AND no progress) tolerated before
    # abandoning. A turn that advanced its progress file is alive (e.g. a long
    # build still running) and is retried for free -- max_iters is the ceiling on
    # progressing work. This cap only catches a harness that produces nothing.
    no_result_cap: int = 3
    # Turn index at which to inject the reflection prompt (0-based). None keeps
    # the pre-checkpoint behavior (plain continue every turn). Applications that
    # want the mode switch set it, typically to max_iters // 2.
    checkpoint_turn: int | None = None


def _resume_prompt(i: int, caps: LoopCaps) -> str:
    """The instruction for resume turn `i` (i > 0): a plain continue with turn
    awareness, escalating to a reflection checkpoint and then a final-turn
    verdict force. Turn awareness lets the model pace itself against the cap
    instead of being surprised by it."""
    n = caps.max_iters
    if i >= n - 1:
        return _FINAL
    head = (
        f"Continue from where you left off (turn {i + 1} of {n}) -- your previous"
        " report was a checkpoint, not completion, so do the next concrete step of"
        f" work now. When you pause or finish, call the `{REPORT_TOOL_NAME}` tool."
    )
    if caps.checkpoint_turn is not None and i == caps.checkpoint_turn:
        return f"{head}\n\n{_REFLECT}"
    return head


async def run_agent_loop(
    harness: Harness,
    *,
    prompt_path: Path,
    workdir: Path,
    control_dir: Path,
    caps: LoopCaps,
    agent: str = "",
    casefile: Path | None = None,
    journal: Journal | None = None,
    sandbox: "Sandbox | None" = None,
) -> AgentResult:
    """Drive the harness to a terminal disposition (or synthesize an ABANDON).

    Returns the agent's terminal AgentResult. If the cap fires first, returns a
    synthetic ABANDON so the caller always sees a terminal result and never an
    open-ended CONTINUE. When a journal is given, the harness streams the turn's
    events into it and its work-event count (`progress`) is the liveness signal
    (a turn that produced no result but emitted agent output/tool calls is alive
    -- a long build still running).
    """
    control_dir.mkdir(parents=True, exist_ok=True)
    consecutive_empty = 0
    for i in range(caps.max_iters):
        result_path = control_dir / f"result.{i:03d}.json"
        before = journal.progress if journal is not None else 0
        run = await harness.run_once(
            prompt_path=prompt_path,
            workdir=workdir,
            result_path=result_path,
            timeout_s=caps.timeout_s,
            agent=agent,
            resume=i > 0,  # turn 0 sends the task; later turns continue the thread
            resume_prompt=_resume_prompt(i, caps) if i > 0 else "",
            casefile=casefile,
            journal=journal,
            sandbox=sandbox,
        )
        if run.result is None:
            # No valid result: a turn that was cut (timeout) or violated the
            # contract. If the turn advanced the journal it was alive (a long
            # build still running, the common case), so retry for free and let
            # the worktree's build resume; max_iters bounds it. Only a turn that
            # produced nothing at all -- no result, no journal growth -- counts
            # as a dead harness toward the cap.
            alive = journal is not None and journal.progress > before
            if alive:
                log.info(
                    "%s turn %d: no result but journal advanced; retrying free",
                    agent or "agent",
                    i,
                )
                consecutive_empty = 0  # alive: breaks the dead-turn streak
                continue
            consecutive_empty += 1
            # finish_reason distinguishes a provider-side empty candidate (a
            # deterministic SAFETY/RECITATION/MALFORMED_FUNCTION_CALL block that
            # reproduces every resume) from a plain cut turn; empty_candidate names
            # the silent shape (a zero-part STOP). Name both in the log and the
            # abandon reason so a dead-turn abandon is not opaque.
            why = f"exit {run.exit_code}"
            if run.finish_reason:
                why += f", finish_reason={run.finish_reason}"
            if run.empty_candidate:
                why += " (empty candidate)"
            log.warning(
                "%s turn %d: no result and no progress (%s), retry %d/%d -- inspect %s",
                agent or "agent",
                i,
                why,
                consecutive_empty,
                caps.no_result_cap,
                run.log_path,
            )
            if consecutive_empty >= caps.no_result_cap:
                return AgentResult(
                    disposition=Disposition.ABANDON,
                    reason=f"no valid result after {consecutive_empty} dead attempts "
                    f"({why})",
                )
            continue
        consecutive_empty = 0
        if run.result.disposition is not Disposition.CONTINUE:
            return run.result
    return AgentResult(
        disposition=Disposition.ABANDON,
        reason=f"did not converge in {caps.max_iters} iterations",
    )

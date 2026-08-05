# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Run one goal loop to completion, driven entirely by a file contract.

This is the in-box half of the whole-process sandbox: the orchestrator spawns a
worker under the job's bwrap+cgroup wrapper, and the worker runs the *entire*
reentry loop here -- so every agent turn, its tools, and any d8/gdb/perf run are
confined by the process boundary, not per shell call. The tools therefore run
free inside the box, and the job keeps its full tool groups; the confinement is
the process, not a narrowed surface. This is the only supported way to run a
turn against untrusted input -- there is no per-call confinement to fall back
on, by design.

The contract is exactly the file-contract the loop already uses: a `LoopSpec`
in (paths, agent, caps), `events.jsonl` appended live under the bound-rw control
dir, and a terminal `outcome.json` out. No runtime coupling to the orchestrator:
it reads a spec, runs, writes the outcome, exits. The application supplies the
Harness (its system prompt + verdict schemas); deepagent supplies the loop.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from pydantic import BaseModel

from .harness import AgentResult, Harness
from .journal import Journal
from .loop import LoopCaps, run_agent_loop

# The terminal AgentResult, written into the loop's control dir. The runner
# reads exactly this file back; its absence means the worker died before
# converging (a synthesized ABANDON on the runner side).
OUTCOME_FILE = "outcome.json"


class LoopSpec(BaseModel):
    """The goal-loop invocation, serialized across the process boundary.

    Every path is absolute and bound at the same location inside the sandbox as
    on the host (the worktree and control dir are bind mounts at their real
    paths), so the worker and the orchestrator name the same files.
    """

    prompt_path: str  # the stage prompt the loop sends on turn 0
    workdir: str  # the job worktree (the tools' cwd)
    control_dir: str  # result.NNN.json + outcome.json live here
    journal_path: str  # events.jsonl, shared with the orchestrator's view
    casefile: str  # the cross-phase document dir
    agent: str  # stage/agent name (keys the verdict schema + ledger)
    max_iters: int = 20
    timeout_s: float = 3600.0  # per turn
    no_result_cap: int = 3
    checkpoint_turn: int | None = None  # reflection-turn index; see LoopCaps
    # Absolute path the orchestrator drops out-of-band news into (see
    # run_agent_loop's `interject`). A FILE rather than a callback because the
    # worker is a subprocess: the loop runs in the box and the orchestrator does
    # not, so the only channel between them is the shared mount -- which is the
    # file contract this whole design rests on, not an exception to it. Read and
    # then REMOVED between turns, so each interjection is delivered exactly once.
    interject_path: str = ""


def write_outcome(control_dir: Path, result: AgentResult) -> None:
    control_dir.mkdir(parents=True, exist_ok=True)
    (control_dir / OUTCOME_FILE).write_text(result.model_dump_json())


def read_outcome(control_dir: Path) -> AgentResult | None:
    path = Path(control_dir) / OUTCOME_FILE
    try:
        return AgentResult.model_validate_json(path.read_text())
    except (OSError, ValueError):
        return None


def _file_interjection(path: str):
    """Read-and-consume the interjection file, or None when none is configured.

    Consuming (unlink) is what makes delivery exactly-once: the loop asks every
    turn, and a file left in place would repeat the same news until the goal
    ended. A read error is swallowed -- an interjection is an optimization, and
    losing one costs a round, while raising here would kill a turn that was
    doing real work.
    """
    if not path:
        return None
    p = Path(path)

    def read() -> str:
        try:
            text = p.read_text()
        except OSError:
            return ""
        with contextlib.suppress(OSError):
            p.unlink()
        return text.strip()

    return read


async def run_loop_from_spec(harness: Harness, spec: LoopSpec) -> AgentResult:
    """Drive the reentry loop to a terminal result and persist it.

    The tools run unwrapped: we are already inside the box, so the mount
    namespace is the boundary and a per-call wrapper would only duplicate it.
    The journal is the same events.jsonl the orchestrator tails, so `icu tail`
    keeps working live across the boundary.
    """
    control_dir = Path(spec.control_dir)
    result = await run_agent_loop(
        harness,
        prompt_path=Path(spec.prompt_path),
        workdir=Path(spec.workdir),
        control_dir=control_dir,
        caps=LoopCaps(
            max_iters=spec.max_iters,
            timeout_s=spec.timeout_s,
            no_result_cap=spec.no_result_cap,
            checkpoint_turn=spec.checkpoint_turn,
        ),
        agent=spec.agent,
        casefile=Path(spec.casefile),
        journal=Journal(spec.journal_path),
        interject=_file_interjection(spec.interject_path),
    )
    write_outcome(control_dir, result)
    return result

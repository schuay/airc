"""The turn contract: Protocol, verdict types, and the test double.

Kept langchain-free so it is cheap to import for the loop and for application
tests; the in-process backend lives in langgraph_harness. Carries no domain
constants -- stage names and verdict fields belong to the application.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
    from airc_tools.sandbox import Sandbox

    from .journal import Journal

log = logging.getLogger(__name__)

# The single, fixed name of the structured-output tool the loop watches for a
# turn's verdict. ToolStrategy would otherwise name it after each stage's Report
# subclass (DraftReport/ReviewReport/...), so the prompt could not name it
# statically and a model that took "the report tool" literally could not find it
# (the observed failure: it wrote a file instead and the turn never terminated).
# Pinning one distinctive name lets every prompt name the exact tool. Not a
# domain constant -- it is part of the turn contract, so it lives here.
REPORT_TOOL_NAME = "submit_turn_report"


class Disposition(StrEnum):
    CONTINUE = "continue"  # not done; re-invoke to make more progress
    COMPLETE = "complete"  # the step is finished (verdict in data)
    ABANDON = "abandon"  # give up on this step (reason set)
    BLOCKED = "blocked"  # needs a human decision


class AgentResult(BaseModel):
    """One turn's flattened result -- the termination signal the loop acts on.

    `data` carries the application's stage-specific verdict fields; the runtime
    stays agnostic about them. The harness fills this by flattening a Report
    subclass; MockHarness constructs it directly in tests.
    """

    disposition: Disposition
    summary: str = ""
    reason: str = ""
    friction: str = ""  # operator diagnostics; see Report.friction
    data: dict = {}


class Report(BaseModel):
    """Base structured verdict a turn reports via ToolStrategy.

    Applications subclass it with stage-specific verdict fields (explicit scalar
    fields, so a provider's structured output accepts the tool schema); the
    harness flattens the extra fields into AgentResult.data.
    """

    disposition: Disposition
    summary: str = ""
    reason: str = ""
    # A friction log: what fought the agent this turn (a broken build, a flaky
    # tool, a misleading casefile, wasted rounds and why). Diagnostics for the
    # operator, never a control signal -- the machine does not read it; the
    # harness journals it so `icu tail` and a cross-job grep surface recurring
    # environment problems no other channel exposes. Empty when nothing snagged.
    friction: str = ""


def to_result(report: Report) -> AgentResult:
    d = report.model_dump()
    return AgentResult(
        disposition=d.pop("disposition"),
        summary=d.pop("summary", ""),
        reason=d.pop("reason", ""),
        friction=d.pop("friction", ""),
        data=d,
    )


@dataclass
class HarnessRun:
    exit_code: int
    result: AgentResult | None  # None on a cut/errored turn (no structured report)
    log_path: Path
    duration_s: float
    # The terminating model call's finish_reason, when the backend can read it
    # (empty on a timeout, an infra error, or a provider that omits it). The one
    # signal that distinguishes why a turn produced no report -- a model-call-cap
    # stop from a provider-side empty candidate (SAFETY/RECITATION/MALFORMED_
    # FUNCTION_CALL/MAX_TOKENS) -- which otherwise both surface as exit_code 1.
    finish_reason: str = ""
    # The terminating candidate carried no text and no tool calls: a zero-part
    # reply. This is the silent-dead-turn signature -- a provider-side empty
    # candidate that still reads finish_reason=STOP (so finish_reason alone looks
    # benign). Lets the loop's abandon reason name it rather than a bare STOP.
    empty_candidate: bool = False


class Harness(Protocol):
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
        journal: "Journal | None" = None,
        sandbox: "Sandbox | None" = None,
    ) -> HarnessRun: ...


@dataclass
class MockHarness:
    """Test double: returns scripted results, writing each to the result path.

    The script is consumed one entry per call; once exhausted it repeats the
    last entry, so a runaway `continue` script still lets the loop's cap fire.
    """

    results: list[AgentResult]
    calls: int = 0
    invocations: list[tuple[str, bool]] = field(default_factory=list)  # (agent, resume)
    resume_prompts: list[str] = field(default_factory=list)  # resume_prompt per call
    sandboxes: list[object] = field(default_factory=list)  # sandbox arg per call

    async def run_once(
        self,
        *,
        prompt_path,
        workdir,
        result_path,
        timeout_s,
        agent="",
        resume=False,
        resume_prompt="",
        casefile=None,
        journal=None,
        sandbox=None,
    ) -> HarnessRun:
        self.invocations.append((agent, resume))
        self.resume_prompts.append(resume_prompt)
        self.sandboxes.append(sandbox)
        r = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(r.model_dump_json())
        return HarnessRun(exit_code=0, result=r, log_path=result_path, duration_s=0.0)

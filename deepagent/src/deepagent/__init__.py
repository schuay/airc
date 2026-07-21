"""deepagent: a reusable runtime for an in-process coding agent.

Import the turn engine, the tools, the caching/accounting, and the robustness;
bring your own state machine, job spec, prompts, and verdict schemas. See
DESIGN.md for the package/application boundary and how to build a new app.
"""

# Re-exported so applications can build confinement profiles for run_once /
# run_agent_loop without depending on airc-tools directly.
from airc_tools.sandbox import Sandbox

from .harness import (
    REPORT_TOOL_NAME,
    AgentResult,
    Disposition,
    Harness,
    HarnessRun,
    MockHarness,
    Report,
    to_result,
)
from .journal import Event, EventKind, Journal
from .langgraph_harness import LangGraphHarness
from .loop import LoopCaps, run_agent_loop
from .skills import render_skill_index
from .worker import (
    OUTCOME_FILE,
    LoopSpec,
    read_outcome,
    run_loop_from_spec,
    write_outcome,
)

__all__ = [
    "AgentResult",
    "Disposition",
    "Harness",
    "HarnessRun",
    "MockHarness",
    "Report",
    "REPORT_TOOL_NAME",
    "to_result",
    "Event",
    "EventKind",
    "Journal",
    "LangGraphHarness",
    "LoopCaps",
    "run_agent_loop",
    "render_skill_index",
    "Sandbox",
    "LoopSpec",
    "run_loop_from_spec",
    "read_outcome",
    "write_outcome",
    "OUTCOME_FILE",
]

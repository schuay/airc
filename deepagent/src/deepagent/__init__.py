# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""deepagent: a reusable runtime for an in-process coding agent.

Import the turn engine, the tools, the caching/accounting, and the robustness;
bring your own state machine, job spec, prompts, and verdict schemas. See
DESIGN.md for the package/application boundary and how to build a new app.
"""

# Re-exported so applications can build the confinement profile they hand to a
# worker (see worker.py) without depending on airc-tools directly. Confinement
# is whole-process: a Sandbox wraps the worker subprocess, and neither run_once
# nor run_agent_loop takes one.
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
    "OUTCOME_FILE",
    "REPORT_TOOL_NAME",
    "AgentResult",
    "Disposition",
    "Event",
    "EventKind",
    "Harness",
    "HarnessRun",
    "Journal",
    "LangGraphHarness",
    "LoopCaps",
    "LoopSpec",
    "MockHarness",
    "Report",
    "Sandbox",
    "read_outcome",
    "render_skill_index",
    "run_agent_loop",
    "run_loop_from_spec",
    "to_result",
    "write_outcome",
]

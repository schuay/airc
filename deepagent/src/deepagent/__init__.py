# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""deepagent: a reusable runtime for an in-process coding agent.

Import the turn engine, the tools, the caching/accounting, and the robustness;
bring your own state machine, job spec, prompts, and verdict schemas. See
DESIGN.md for the package/application boundary and how to build a new app.
"""

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
from .loop import LoopCaps, run_agent_loop
from .skills import render_skill_index
from .worker import (
    OUTCOME_FILE,
    LoopSpec,
    read_outcome,
    run_loop_from_spec,
    write_outcome,
)


def __getattr__(name: str):
    # LangGraphHarness is the only backend behind the Harness protocol, and the
    # only thing here that costs langchain/langgraph/anthropic/mcp to import --
    # 785ms against 58-72ms for every other module in this package. Deferring it
    # is what lets a consumer that wants a Journal, a Report or a LoopCaps take
    # the protocol and the data types without paying for an implementation it
    # never builds. A missing langgraph install now fails when the harness is
    # first constructed rather than at import; that is still daemon startup.
    if name == "LangGraphHarness":
        from .langgraph_harness import LangGraphHarness

        return LangGraphHarness
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "read_outcome",
    "render_skill_index",
    "run_agent_loop",
    "run_loop_from_spec",
    "to_result",
    "write_outcome",
]

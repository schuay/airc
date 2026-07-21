"""The turn-report tool has one fixed name, and every prompt names that tool.

The failure this guards against: ToolStrategy names the structured-output tool
after the schema class (DraftReport/ReviewReport/...), while the prompt said "the
report tool" -- so a model that took the name literally could not find the tool,
never reported, and the loop ran to max_iters. The fix pins one tool name and
threads it through the prompts; these tests lock both halves together.
"""

import re

from langchain.agents.structured_output import OutputToolBinding, ToolStrategy

from deepagent import REPORT_TOOL_NAME, Report
from deepagent.langgraph_harness import _DEFAULT_SYSTEM, _HARD_NUDGE, _SOFT_NUDGE


class _StageReport(Report):
    built: bool | None = None


def test_name_is_a_valid_distinct_tool_name():
    # Provider tool-name grammar, and not the too-generic "report".
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", REPORT_TOOL_NAME)
    assert REPORT_TOOL_NAME != "report"


def test_override_renames_the_structured_output_tool():
    # Default: ToolStrategy names the tool after the schema class. The harness
    # override pins it to REPORT_TOOL_NAME; assert langchain actually builds the
    # tool with that name (the mechanism the fix rests on).
    strat = ToolStrategy(schema=_StageReport, handle_errors=True)
    assert strat.schema_specs[0].name == "_StageReport"
    strat.schema_specs[0].name = REPORT_TOOL_NAME
    tool = OutputToolBinding.from_schema_spec(strat.schema_specs[0]).tool
    assert tool.name == REPORT_TOOL_NAME


def test_prompts_name_the_exact_tool():
    for prompt in (_DEFAULT_SYSTEM, _SOFT_NUDGE, _HARD_NUDGE):
        assert REPORT_TOOL_NAME in prompt

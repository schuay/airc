"""An exhausted empty candidate must surface as a NAMED harness failure.

_EmptyCandidateRetry (airc-core) retries a zero-part Gemini candidate through
the shared backoff and, on exhaustion, raises EmptyCandidateError. run_once
catches it by type so the abandon reason says "empty candidate" rather than an
opaque exit -- and, crucially, reports exit_code 1 (a dead turn the loop may
retry) rather than -1 (a hard error), which is what distinguishes it from a
genuine crash in the loop's accounting.
"""

from airc_core import CommonConfig, EmptyCandidateError

from deepagent import LangGraphHarness


class _RaisingGraph:
    """Stands in for the compiled agent graph: its ainvoke raises the way an
    exhausted _EmptyCandidateRetry does at the end of the retry budget."""

    def __init__(self, exc):
        self._exc = exc

    async def ainvoke(self, _input, _config):
        raise self._exc


def _harness(tmp_path, exc) -> LangGraphHarness:
    common = CommonConfig()
    common.models = {"default": "google_genai:gemini-3.1-flash-lite"}
    common.token_db_path = tmp_path / "tokens.db"
    h = LangGraphHarness(common)

    async def _no_init() -> None:
        return None

    h._ensure_init = _no_init  # MCP servers are irrelevant to this path
    h._graph_for = lambda *a, **k: _RaisingGraph(exc)
    return h


async def _run(tmp_path, exc):
    prompt = tmp_path / "JOB.md"
    prompt.write_text("do the thing")
    return await _harness(tmp_path, exc).run_once(
        prompt_path=prompt,
        workdir=tmp_path,
        result_path=tmp_path / "result.0.json",
        timeout_s=30,
        agent="repro",
    )


async def test_exhausted_empty_candidate_is_a_dead_turn_not_a_crash(tmp_path, caplog):
    run = await _run(tmp_path, EmptyCandidateError("empty candidate (STOP)"))
    # exit 1 = no result this turn (the loop's dead-turn path); -1 is reserved
    # for a timeout or an unexpected exception.
    assert run.exit_code == 1
    assert run.result is None
    assert "empty candidate" in caplog.text.lower()


async def test_any_other_exception_is_still_a_hard_error(tmp_path):
    # The by-type catch must not swallow real crashes into the softer code.
    run = await _run(tmp_path, RuntimeError("boom"))
    assert run.exit_code == -1

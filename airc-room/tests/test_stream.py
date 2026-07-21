"""_stream aggregation: only the final answer is posted, not tool-step preamble."""

from types import SimpleNamespace

from langchain_core.messages import AIMessageChunk

from airc_room.runner import AgentRunner


def _chunk(
    text: str = "", tool: str | None = None, id: str | None = None
) -> AIMessageChunk:
    tcc = []
    if tool:
        tcc = [{"name": tool, "args": "", "id": "t1", "index": 0}]
    return AIMessageChunk(content=text, tool_call_chunks=tcc, id=id)


async def test_stream_drops_preamble_before_tool_calls():
    # A ReAct turn: the model narrates before a tool call, then answers. Only the
    # answer (text after the last tool call) should be returned/posted.
    async def _emit(*a):
        pass

    async def astream(input, config=None, stream_mode=None):
        yield "messages", (_chunk(text="Let me check foo.cc."), {})
        yield "messages", (_chunk(tool="read_file"), {})
        yield "messages", (_chunk(text="The bug is an off-by-one in the loop."), {})

    graph = SimpleNamespace(astream=astream)
    text, _usage = await AgentRunner._stream(
        SimpleNamespace(_emit=_emit), graph, "perf", {}, {}
    )
    assert text == "The bug is an off-by-one in the loop."
    assert "Let me check" not in text


async def test_stream_skips_summarization_chunks():
    # A ceiling-summarization call nested in the turn streams into messages
    # mode with lc_source metadata; its conversation restatement must never be
    # collected as the reply (the echoed-prompt artifact).
    async def _emit(*a):
        pass

    async def astream(input, config=None, stream_mode=None):
        yield (
            "messages",
            (
                _chunk(text="You were asked to comment on a commit..."),
                {"lc_source": "summarization"},
            ),
        )
        yield "messages", (_chunk(text="The real answer."), {})

    text, _usage = await AgentRunner._stream(
        SimpleNamespace(_emit=_emit), SimpleNamespace(astream=astream), "perf", {}, {}
    )
    assert text == "The real answer."


async def test_stream_keeps_a_plain_answer_with_no_tools():
    async def _emit(*a):
        pass

    async def astream(input, config=None, stream_mode=None):
        yield "messages", (_chunk(text="Short "), {})
        yield "messages", (_chunk(text="answer."), {})

    text, _usage = await AgentRunner._stream(
        SimpleNamespace(_emit=_emit), SimpleNamespace(astream=astream), "perf", {}, {}
    )
    assert text == "Short answer."


async def test_stream_drops_trailing_text_after_a_tool_call():
    # Gemini orders parts model-side and can emit a self-note AFTER its tool
    # call, past the preamble clear; only the final response's text may be
    # posted, or the note leaks concatenated before the answer (the observed
    # "Do not set timers or search chat unless strictly required. gandalf: ..."
    # artifact).
    async def _emit(*a):
        pass

    async def astream(input, config=None, stream_mode=None):
        yield "messages", (_chunk(tool="search_chat", id="m1"), {})
        yield "messages", (_chunk(text="Do not set timers here. ", id="m1"), {})
        yield "messages", (_chunk(text="gandalf: the real answer", id="m2"), {})

    text, _usage = await AgentRunner._stream(
        SimpleNamespace(_emit=_emit), SimpleNamespace(astream=astream), "perf", {}, {}
    )
    assert text == "gandalf: the real answer"


async def test_stream_drops_partial_text_from_a_retried_call():
    # A model call that dies mid-stream has already emitted chunks; the retry
    # is a fresh response (new id) and must replace, not extend, the partial.
    async def _emit(*a):
        pass

    async def astream(input, config=None, stream_mode=None):
        yield "messages", (_chunk(text="The bug is in fo", id="m1"), {})
        yield "messages", (_chunk(text="The bug is in foo.cc.", id="m2"), {})

    text, _usage = await AgentRunner._stream(
        SimpleNamespace(_emit=_emit), SimpleNamespace(astream=astream), "perf", {}, {}
    )
    assert text == "The bug is in foo.cc."


async def test_stream_turn_ending_on_a_tool_call_keeps_only_post_call_text():
    # A turn cut off at the call cap ends ON a tool-calling response; the
    # within-response preamble clear still governs there, so only text after
    # the call is posted.
    async def _emit(*a):
        pass

    async def astream(input, config=None, stream_mode=None):
        yield "messages", (_chunk(text="Let me look. ", id="m1"), {})
        yield "messages", (_chunk(tool="read_file", id="m1"), {})
        yield "messages", (_chunk(text="Checking now.", id="m1"), {})

    text, _usage = await AgentRunner._stream(
        SimpleNamespace(_emit=_emit), SimpleNamespace(astream=astream), "perf", {}, {}
    )
    assert text == "Checking now."


async def test_stream_id_less_chunks_never_reset_the_buffer():
    # Continuation chunks may omit the id; a None id must not read as a new
    # response, or a mid-answer reset would drop the answer's own head.
    async def _emit(*a):
        pass

    async def astream(input, config=None, stream_mode=None):
        yield "messages", (_chunk(text="Short ", id="m1"), {})
        yield "messages", (_chunk(text="answer."), {})

    text, _usage = await AgentRunner._stream(
        SimpleNamespace(_emit=_emit), SimpleNamespace(astream=astream), "perf", {}, {}
    )
    assert text == "Short answer."

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Place the memory index in a persona's conversation, from the conversation.

Injecting the index into every turn's text costs a copy per turn: each lands in
a user-role message the growing prefix cache absorbs as ordinary history, so a
long thread pays for the same table of contents repeatedly and reaches the
summarization threshold sooner.

Whether the conversation already has the index is read from the message list
itself. A side record of past injections goes wrong when SummarizationMiddleware
compacts old history: it keeps suppressing a block whose message no longer
exists, and the index is gone for the rest of the thread. Reading the messages
makes the check self-correcting: absence triggers re-injection, so compaction, a
fresh checkpoint after a generation bump, a turn that crashed before its block
was checkpointed, and a restart all resolve the same way with no state to reset.

The index itself is not computed here: it arrives per turn through the
memory_index state key, so the git grep runs once per turn in the runner
instead of once per model call. The middleware owns placement (and the block's
framing), the runner owns content.
"""

from __future__ import annotations

from typing import Annotated, Any, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.channels.untracked_value import UntrackedValue

# Marks our own inserts, so finding the last one is a keyed lookup instead of a
# search for our wording in message text. Mirrors GroundingReminderMiddleware.
_SRC = "memory_index"

# Rough chars-per-token, for measuring how far back the last block has drifted.
# Only ever compared against the interval below, so the estimate needs to be the
# right scale, not exact.
_CHARS_PER_TOKEN = 4

# Re-inject once the newest block is this far back, even when the store has not
# changed: the system prompt loses weight with distance, and a persona that
# cannot see the index stops consulting memory at all. Sized at half of
# airc_core's _SUMMARY_KEEP_TOKENS (50k, the tail a compaction keeps), so the
# newest block is normally inside what compaction preserves and the absence path
# stays a backstop, not the common case. If either number moves, this is
# the relationship to re-check -- correctness does not depend on it, only how
# often a conversation falls back to absence.
_REMINDER_TOKENS = 25_000


class _MemoryIndexState(AgentState):
    # UntrackedValue: supplied fresh by the runner each turn and never
    # checkpointed. The block that lands in the conversation is the durable
    # record; persisting the input as well would put a second, silently
    # divergent copy in the checkpoint.
    memory_index: NotRequired[Annotated[str, UntrackedValue]]


class MemoryIndexMiddleware(AgentMiddleware):
    """Append the memory index to the conversation when it is not already there,
    has changed, or has drifted too far back. Granted only to memory-enabled
    personas, so a persona without the tools never carries the block."""

    state_schema = _MemoryIndexState

    @property
    def name(self) -> str:
        # create_agent keys graph nodes on middleware names and rejects
        # duplicates; the default is the class name, which is already unique here.
        return "MemoryIndexMiddleware"

    def _block(self, index: str) -> str:
        # The framing travels with the block: it is a table of contents, and a
        # persona that reads a hook as the fact is the failure MEMORY_RULES warns
        # about. Deterministic in index, so the staleness check below can compare
        # rendered blocks instead of tracking the raw index separately.
        return f"Memory (read a note with memory_read before relying on it):\n{index}"

    def _is_block(self, m) -> bool:
        return (
            isinstance(m, HumanMessage) and m.additional_kwargs.get("lc_source") == _SRC
        )

    def _due(self, messages: list, block: str) -> bool:
        """Whether this conversation needs the block, walking back from the tail.

        The first block found decides: identical and recent means no, changed or
        buried means yes. Reaching the start without finding one means the
        conversation has never had it or no longer does -- both need it now.
        """
        chars = 0
        for m in reversed(messages):
            if self._is_block(m):
                if str(m.content) != block:
                    return True
                return chars // _CHARS_PER_TOKEN >= _REMINDER_TOKENS
            chars += len(str(m.content))
        return True

    def before_model(self, state, runtime) -> dict[str, Any] | None:
        index = state.get("memory_index") or ""
        if not index:  # memory off for this persona, or an empty store
            return None
        block = self._block(index)
        if not self._due(state["messages"], block):
            return None
        # A tail append, never a mid-history insert: rewriting history would
        # poison the cached prefix. The block settles into the cache like any
        # other message and costs a cache-read from the next call on.
        return {
            "messages": [HumanMessage(block, additional_kwargs={"lc_source": _SRC})]
        }

    async def abefore_model(self, state, runtime) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

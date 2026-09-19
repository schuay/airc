# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""The turn identity a local tool reads out of its injected RunnableConfig.

LangGraph keys a checkpoint on one opaque string, `configurable.thread_id`, and
the runner packs (thread, persona, context generation) into it. That string is a
checkpoint key, not a data channel: parsing the parts back out ties every
consumer to a format the runner alone owns, and broke when the runner added the
generation suffix (the parse returned "perf:g0" as the persona, so no live
persona matched and every timer wake was dropped).

So the parts travel as their own configurable keys, built and read here.
LangGraph passes unknown `configurable` keys through untouched (they reach a
tool's injected config) and keys its checkpoint on `thread_id` alone.

There is no fallback to parsing the composite. A turn that did not come from
turn_config() has no identity to offer (the forced-JSON structured turn, which
holds no local tools because it belongs to no thread), and "missing turn
context" is the correct answer there.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

# Namespaced so they cannot collide with a LangGraph/LangChain reserved key.
THREAD_KEY = "airc_thread"
AGENT_KEY = "airc_agent"
# The message that caused this turn. Separate from the identity pair below: a
# turn always has an identity, but not always a trigger (a timer wake has none),
# so it is read through its own accessor and never folded into turn_context's
# all-or-nothing check.
TRIGGER_KEY = "airc_trigger"


def turn_config(
    thread_id: int, agent_key: str, generation: int, trigger_id: int | None = None
) -> dict:
    """The `configurable` a persona's turn runs under.

    `thread_id` is the composite LangGraph checkpoints on: the room thread, the
    persona's stable key (not its addressable name, so a nickname toggle does not
    orphan a checkpoint), and the context generation, which is what a memory
    compaction bumps to start the persona from a fresh checkpoint. The same two
    identity parts ride alongside under their own keys for tools to read.

    `trigger_id` is the message that caused this turn, when one did. A tool whose
    job is "did a human ask for this" would otherwise scan the thread backwards
    guessing which message meant it; guessing resolved two open requests in one
    thread to whichever was newest. Optional because not every turn has a
    trigger: a timer wake is driven by a note, not a message.
    """
    cfg = {
        "thread_id": f"{thread_id}:{agent_key}:g{generation}",
        THREAD_KEY: thread_id,
        AGENT_KEY: agent_key,
    }
    if trigger_id is not None:
        cfg[TRIGGER_KEY] = trigger_id
    return cfg


def turn_context(config: RunnableConfig | None) -> tuple[int | None, str]:
    """The (thread id, persona stable key) a tool was called in, or (None, "")
    when the turn carries no identity. Tools treat that as a refusal, not a
    default -- acting on a guessed thread is worse than declining.

    All or nothing: a config carrying only one of the two parts yields no
    identity at all. Returning the half that is present reproduces the
    failure above: an empty agent still reads as
    "present" to a caller that only checks the thread id, so timer_create would
    report success, persist a timer no persona can own, and have its wake
    dropped at fire time. A partial identity is a bug upstream; the only safe
    reading of it is none.
    """
    configurable = (config or {}).get("configurable") or {}
    thread_id = configurable.get(THREAD_KEY)
    agent = configurable.get(AGENT_KEY)
    # bool is an int subclass, and True would sail through as thread 1.
    if not isinstance(thread_id, int) or isinstance(thread_id, bool):
        return None, ""
    if not isinstance(agent, str) or not agent:
        return None, ""
    return thread_id, agent


def turn_trigger(config: RunnableConfig | None) -> int | None:
    """The id of the message that caused this turn, or None when it had no one
    message behind it (a timer wake, a structured turn, an older config).

    Read on its own rather than as a third element of turn_context, because it
    is genuinely optional where the identity pair is not: making it part of the
    all-or-nothing check would turn every triggerless turn into "no identity"
    and silently disable the tools that read it.
    """
    trigger = ((config or {}).get("configurable") or {}).get(TRIGGER_KEY)
    # bool is an int subclass, and True would sail through as message 1.
    if not isinstance(trigger, int) or isinstance(trigger, bool):
        return None
    return trigger

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Per-persona LangGraph agents and turn execution.

Each persona gets one create_agent() graph, built at startup with its
filtered MCP toolset. Conversation state is per (room thread, persona): the
LangGraph thread id is "<thread>:<persona>", checkpointed in SQLite, so an
agent keeps its own tool-call history across turns. Messages from other
participants are injected as user-role transcript lines the first time the
agent takes a turn after they were posted (offset-tracked in the store).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable


from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
)
from langchain_core.messages import (
    AIMessageChunk,
)

from .config import Config
from airc_core import MCPToolset, TokenLog, make_model, missing_key
from airc_core.agent import (
    CallBudgetMiddleware,
    TimeBudgetMiddleware,
    _CallTrace,
    base_middleware,
    growing_cache_middleware,
)
from .personas import Persona
from .store import Message, MessageKind, Store

log = logging.getLogger(__name__)

# Event callback: (agent_name, event, detail) — e.g. ("perf", "tool", "run_d8").
EventHook = Callable[[str, str, str], Awaitable[None]]


ROOM_RULES = """\
## Room rules

You are one participant in a shared chat room with humans and other agents.
Other participants' messages are shown to you as "[sender] text". Write your
reply as plain message text; never prefix it with your own name, and never
quote, fabricate, or continue other participants' lines -- you write exactly
one message, as yourself. To pull a specific agent in, refer to them by name
and they will join if they can add value; to require a reply, write their
handle (as listed under "Other agents") followed by a colon, anywhere in your
message. Do not use @.

- Ground non-trivial claims in evidence: cite file:line, a commit hash, or a
  CL link you actually looked up with tools. Distinguish clearly between what
  you verified and what you are inferring ("this looks like" vs "this is").
- Before sending, spend your last thinking on verification: the humans in
  this room are experts and immediately recognize guesses and mistakes.
  Re-check every hash, number, and code claim against what you actually read
  in this conversation; label anything unverified as such, or cut it.
- Strongly prefer conciseness: a few short sentences, plain language, no
  headers, minimal bullets. AI overload is real -- go long only when the
  content requires it or detail was explicitly requested, and even then,
  tight prose beats structure.
- If you have nothing substantive to add, reply with exactly NOTHING_TO_ADD.
- Tools that execute code or benchmarks are expensive; use them only when a
  concrete question justifies it, never speculatively.
"""


def _identity_section(persona: Persona, all_personas: dict[str, Persona]) -> str:
    others = "\n".join(
        f"- {p.name} -- {p.description}"
        for p in all_personas.values()
        if p.name != persona.name
    )
    return (
        f"## Identity\n"
        f'You are "{persona.display_name}" (handle {persona.name}).\n'
        f"{persona.description}\n\n"
        f"## Other agents in this room\n{others or '(none)'}\n"
    )


def build_system_prompt(
    persona: Persona,
    all_personas: dict[str, Persona],
    mcp_instructions: str,
    room_prompt: str = "",
    voice: str = "",
) -> str:
    parts = [_identity_section(persona, all_personas), ROOM_RULES]
    if room_prompt:
        parts.append(room_prompt)
    parts.append(persona.system_prompt)
    if mcp_instructions:
        parts.append(f"## MCP server instructions\n\n{mcp_instructions}")
    # Voice goes last: it is a style overlay, and trailing position gives the tone
    # reference recency weight without displacing the role or the grounding rules.
    if voice:
        parts.append(_voice_section(voice))
    return "\n\n".join(parts)


def _voice_section(voice: str) -> str:
    return (
        "## Voice\n"
        "Write your messages in the voice below. It governs TONE ONLY -- never"
        " your expertise, your conclusions, or what you choose to flag, only how"
        " it sounds. Do not mention this guide or the person it is modeled on;"
        " just sound like it.\n"
        "Take from the guide ONLY the abstract register -- sentence length,"
        " directness, how it reasons and hedges, how warm or terse it is. The"
        " specific phrases, sample lines, interjections, and tics in it are"
        " EVIDENCE of that register, not material to reuse: never reproduce them,"
        " verbatim or adapted, and never adopt a catchphrase or signature tic."
        " Sound like the same kind of writer, entirely in your own words.\n\n"
        f"{voice}"
    )


# Provenance patterns redacted from a voice guide as a backstop. Deployed guides
# are authored without identifiers; this only bites if a raw source guide (which
# still carries name/email/CL provenance) is pointed at by mistake. Not a name
# detector -- a bare surname in prose is exactly what the source scrub is for.
_VOICE_EMAIL_RE = re.compile(r"\s*<?\b[\w.+-]+@[\w.-]+\.\w+\b>?")
_VOICE_HASH_RE = re.compile(r"\b(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b")


def voice_body(text: str) -> str:
    """The tone-bearing body of a voice guide, provenance stripped.

    Deployed guides are authored clean; this is defense in depth so a raw source
    guide cannot carry provenance into a prompt. Drops the YAML frontmatter, a
    leading `# ... voice guide` title, and the trailing `## Sources mined` corpus
    note, then redacts stray emails and commit-hash tokens. It cannot strip a
    bare surname in prose. Robust to any part being absent."""
    t = text.strip()
    if t.startswith("---"):
        end = t.find("\n---", 3)
        if end != -1:
            t = t[end + len("\n---") :].lstrip()
    # A leading H1 ("# <handle> voice guide") names the source; drop it.
    if t.startswith("# "):
        nl = t.find("\n")
        t = t[nl + 1 :].lstrip() if nl != -1 else ""
    marker = t.find("\n## Sources mined")
    if marker != -1:
        t = t[:marker].rstrip()
    t = _VOICE_HASH_RE.sub("", _VOICE_EMAIL_RE.sub("", t))
    return t.strip()


def format_transcript(messages: list[Message]) -> str:
    # Bracketed attribution, NOT "sender: text": the colon form is reserved
    # for addressing, so transcript lines can never read as (or be echoed
    # into) addresses.
    return "\n\n".join(f"[{m.sender}] {m.text}" for m in messages)


def sanitize_reply(text: str, self_name: str, others: set[str]) -> str:
    """Strip fabricated transcript artifacts from a reply.

    The format split makes echoed lines inert for routing; this keeps them out
    of the room entirely: drop lines impersonating another participant
    ("[compiler] ..." / "[users/...] ..." / raw "users/...: ...") and a leading
    self-attribution prefix. `others` should be EVERY participant name (including
    disabled personas), not just live ones, or a line attributed to a real but
    disabled agent would pass through.
    """
    text = text.strip()
    lowered = text.lower()
    for prefix in (f"[{self_name}]", f"{self_name}:"):
        if lowered.startswith(prefix):
            text = text[len(prefix) :].lstrip()
            break
    # Tolerate bracket whitespace ("[ compiler]") so a spaced variant cannot slip
    # an attribution past a literal "[compiler]" match.
    fabricated = (
        re.compile(r"\[\s*(?:%s)\s*\]" % "|".join(re.escape(o) for o in others), re.I)
        if others
        else None
    )
    lines = [
        ln
        for ln in text.splitlines()
        if not (
            (fabricated and fabricated.match(ln.lstrip()))
            or re.match(r"\s*(\[)?users/\S+[\]:]", ln)
        )
    ]
    return "\n".join(lines).strip()


def build_turn_content(
    unseen: list[Message],
    addressed: bool = False,
    task_prompt: str | None = None,
    now: datetime | None = None,
    memory_index: str | None = None,
) -> str:
    """The user-role content injected for a turn.

    The framing matters: without it, models continue the "sender: text"
    transcript format in their reply, fabricating other participants' lines
    inside one message. It is a leading label on the transcript, not a trailing
    parenthetical meta-instruction -- an aside like "(do not quote this...)"
    reads to the model as a command it can echo, and did (a persona once replied
    with the instruction itself). The behavioral rule (one message, no quoting
    or continuing others' lines) is stated once, authoritatively, in ROOM_RULES;
    here we only frame the transcript as input.

    A current-time line leads the content. Agents need it constantly -- "this
    week's offers", timer math, dating a memory entry -- and it goes HERE, in the
    per-turn tail, deliberately NOT in the cached system prompt: a timestamp that
    changed every turn would bust the prefix cache. As uncached tail content it is
    free. `now` is injectable for tests; None reads the wall clock.

    memory_index, when memory is enabled, is the derived table of contents of the
    room's long-term memory (a line per note). It rides the same uncached tail for
    the same reason -- and there it stays fresh (a note written this turn shows up
    next turn), which a cached system-prompt placement could not do.

    When addressed is set, a human named this agent directly, so the
    NOTHING_TO_ADD escape hatch is withdrawn for this turn. task_prompt carries a
    per-turn task brief (e.g. commit commentary) appended only when the
    orchestrator routes this turn to that task; it stays out of the cached system
    prompt so the cache is unaffected.
    """
    stamp = (now or datetime.now().astimezone()).strftime("%A %Y-%m-%d %H:%M %Z")
    when = f"Current time: {stamp.strip()}.\n\n"
    if memory_index:
        # The memory table of contents, between the time line and the transcript.
        # A hook, not the fact: the persona reads a note in full before relying on
        # it (stated in MEMORY_RULES).
        when += f"Memory (read a note with memory_read before relying on it):\n{memory_index}\n\n"
    if not unseen:
        body = when + "(You were asked to respond; see the conversation above.)"
    else:
        body = (
            when + "New messages in the room since your last turn -- read them and"
            " reply as yourself, in one message:\n\n"
            f"{format_transcript(unseen)}"
        )
    if addressed:
        body += (
            "\n\nA human addressed you directly (your handle followed by a"
            " colon). Answer their question or request substantively; do not"
            " reply NOTHING_TO_ADD -- that option does not apply to a direct"
            " human address."
        )
    if task_prompt:
        body += f"\n\n{task_prompt}"
    return body


@dataclass
class _AgentEntry:
    persona: Persona
    graph: object


@dataclass
class _TurnUsage:
    """Token usage aggregated over a turn's model calls, plus the per-call shape.

    input/output/cached are summed across the turn (and across providers);
    calls and max_call_input come from the per-call tracer and separate a long
    accumulating loop from one large prompt.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_in: int = 0
    calls: int = 0
    max_call_input: int = 0


# Per-turn model-call budget for persona turns: nudge toward wrapping up, then a
# graceful cap, so a sprawling tool-using turn converges instead of spinning to
# the recursion limit. Mirrors the review budget; the cap stays well under the
# recursion limit so it, not a GraphRecursionError, governs.
_TURN_SOFT_NUDGE_CALLS = 25
_TURN_HARD_NUDGE_CALLS = 40
_TURN_MAX_MODEL_CALLS = 50
_TURN_SOFT_NUDGE = (
    "You have gathered substantial context. Begin converging on your reply now; "
    "only keep using tools to confirm a specific point, not to open new lines of "
    "investigation."
)
_TURN_HARD_NUDGE = (
    "Stop using tools. Write your reply now using what you have already gathered."
)
# The same two nudges keyed on wall-clock, as fractions of turn_timeout: the
# orchestrator's hard timeout kills the turn and discards all its work, so nudge
# it to converge and produce a reply from the context already paid for before that
# deadline. Stamped below 1.0 so the final converging call still lands in time.
_TURN_SOFT_NUDGE_TIME_FRAC = 0.55
_TURN_HARD_NUDGE_TIME_FRAC = 0.80


class AgentRunner:
    """Builds and runs one agent per persona.

    Use as an async context manager; personas whose model provider key is
    missing or whose model fails to construct are skipped with a warning.
    """

    def __init__(
        self,
        cfg: Config,
        personas: dict[str, Persona],
        toolset: MCPToolset,
        store: Store,
        on_event: EventHook | None = None,
        room_prompt: str = "",
        timer_scheduler=None,
        local_tool_groups: dict | None = None,
    ) -> None:
        self._cfg = cfg
        self._personas = personas
        self._toolset = toolset
        # Plugin-supplied local (non-MCP) tools, keyed by tool_group name. A
        # persona gets a group's tools iff the group is in its tool_groups -- the
        # same gate MCP tools use. Empty for a bare room or a plugin that ships none.
        self._local_tool_groups = local_tool_groups or {}
        self._store = store
        self._tokens = TokenLog(cfg.token_db_path)
        self._on_event = on_event
        self._room_prompt = room_prompt
        # Optional TimerScheduler: when set, every chat persona gets the local
        # (non-MCP) timer tools (create/list/cancel). None disables timers.
        self._timer_scheduler = timer_scheduler
        self._stack = contextlib.AsyncExitStack()
        self._agents: dict[str, _AgentEntry] = {}
        # Per-persona structured-turn graphs (see run_structured_turn): a fresh
        # JSON-forcing graph kept off the chat checkpointer, built lazily, reused.
        self._structured_agents: dict[str, object] = {}
        self._structured_locks: dict[str, asyncio.Lock] = {}
        self._voice_cache: dict[str, str] = {}  # state_key -> cleaned voice body

    @property
    def agents(self) -> dict[str, Persona]:
        return {name: e.persona for name, e in self._agents.items()}

    def name_for_key(self, key: str) -> str | None:
        """The live addressable agent name for a persona's stable key, or None if
        the persona is not live. Timer wakes persist the stable key (the
        checkpoint identity); with use_nicknames on the addressable name differs
        from the key, so a wake must translate before routing. With nicknames off
        name == key, so this is an identity lookup."""
        for name, e in self._agents.items():
            if e.persona.key == key:
                return name
        return None

    def _voice_for(self, persona: Persona) -> str:
        """The persona's configured voice guide body, or "" for a neutral voice.
        Keyed on the stable handle so a nickname toggle does not change the
        mapping; a missing or unreadable file logs and degrades to neutral."""
        key = persona.state_key
        path = self._cfg.voices.get(key)
        if not path:
            return ""
        if key not in self._voice_cache:
            try:
                self._voice_cache[key] = voice_body(path.read_text())
                log.info("voice: %s <- %s", persona.name, path)
            except OSError as e:
                log.warning("voice: %s: cannot read %s: %s", persona.name, path, e)
                self._voice_cache[key] = ""
        return self._voice_cache[key]

    def _memory_enabled(self, persona: Persona) -> bool:
        """Whether this persona gets long-term memory: the feature is on and the
        persona lists the memory tool_group (no point priming or ruling a persona
        that cannot read entries)."""
        from .memory import MEMORY_GROUP

        return self._cfg.memory.enabled and MEMORY_GROUP in persona.tool_groups

    async def __aenter__(self) -> AgentRunner:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        ckpt_path = self._cfg.db_path.with_suffix(".ckpt.db")
        checkpointer = await self._stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(str(ckpt_path))
        )

        available = {
            name: p
            for name, p in self._personas.items()
            if not self._skip_for_missing_key(p)
        }
        for name, persona in available.items():
            try:
                graph = self._build_agent(
                    persona, available, checkpointer, voice=self._voice_for(persona)
                )
            except Exception:
                log.exception("failed to build agent %s; disabling", name)
                continue
            self._agents[name] = _AgentEntry(persona=persona, graph=graph)
        if not self._agents:
            raise RuntimeError("no usable agents (check model provider keys)")
        log.info("agents ready: %s", ", ".join(self._agents))
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    def _skip_for_missing_key(self, persona: Persona) -> bool:
        model_id = self._cfg.resolve_model(persona.model_id)
        if key := missing_key(model_id):
            log.warning("agent %s disabled: %s is not set", persona.name, key)
            return True
        return False

    def _build_agent(
        self,
        persona: Persona,
        available: dict[str, Persona],
        checkpointer,
        *,
        extra_system: str = "",
        voice: str = "",
    ) -> object:
        model_id = self._cfg.resolve_model(persona.model_id)
        # A persona's tool_groups mix MCP groups (resolved to patterns below) and
        # plugin-local groups (resolved to local tools further down). Keep the
        # local ones out of the MCP resolver so they are not logged as unknown
        # groups -- they are known, just to the plugin, not the MCP toolset.
        mcp_groups = [
            g for g in persona.tool_groups if g not in self._local_tool_groups
        ]
        patterns = self._toolset.resolve_patterns(
            mcp_groups, persona.tools, persona.name
        )
        tools = self._toolset.tools_for(patterns)
        # Local tools (not MCP), added for conversational builds only (extra_system
        # marks the forced-JSON digest turn, where they have no place). They read
        # their thread/agent from the run config, so they need no ambient wiring.
        if not extra_system:
            local: list = []
            # timer tools: every chat persona (create steered to rare use by its
            # docstring; list/cancel let it manage what it scheduled).
            if self._timer_scheduler is not None:
                from .timers import make_timer_tools

                local.extend(make_timer_tools(self._timer_scheduler))
            # search_chat: every chat persona gets it by default, no tool_groups
            # grant needed. It is read-only over the room's own history and scopes
            # to the caller's space, so there is nothing to gate on.
            from .chat_search import make_search_chat_tool

            local.append(make_search_chat_tool(str(self._cfg.db_path)))
            # Plugin local tools, gated by the persona's tool_groups exactly like
            # MCP tools: a persona gets a group's local tools only if it lists the
            # group (e.g. "memory" for the grocery akbase tools). These groups are
            # plugin-owned and separate from the MCP [tool_groups], so naming one
            # here is not an unknown-MCP-group warning.
            for group in persona.tool_groups:
                local.extend(self._local_tool_groups.get(group, []))
            tools = [*tools, *local]
        log.info("agent %s: model=%s tools=%d", persona.name, model_id, len(tools))
        system_prompt = build_system_prompt(
            persona,
            available,
            self._toolset.instructions,
            self._room_prompt,
            voice=voice,
        )
        # Memory write-discipline: appended (cached with the prefix) for a
        # conversational build whose persona holds the memory tools. Skipped on the
        # forced-JSON digest turn (extra_system), which has no tools and no memory.
        if not extra_system and self._memory_enabled(persona):
            from .memory import MEMORY_RULES

            system_prompt = f"{system_prompt}\n\n{MEMORY_RULES}"
        if extra_system:
            system_prompt = f"{system_prompt}\n\n{extra_system}"
        middleware = base_middleware(
            model_id,
            system_prompt,
            tools,
            summarizer_model_id=self._cfg.filter_model,
            grounding_tokens=self._cfg.grounding_reminder_tokens,
        )
        # The growing-prefix cache, then the call budget. The cache is outer to
        # the budget so the ephemeral nudge the budget appends lands in the tail,
        # never the cached prefix.
        if cache := growing_cache_middleware(
            model_id,
            system_prompt,
            tools,
            self._cfg.caching_explicit,
            self._cfg.cache_ttl_minutes,
        ):
            middleware.append(cache)
        turn_timeout = self._cfg.orchestrator.turn_timeout
        middleware += [
            # Converge a sprawling tool-using turn before it spins to the
            # recursion limit (a single turn re-sends its whole growing history
            # per call, so a runaway turn is the dominant cost). The cap stays
            # below the recursion limit so it governs gracefully.
            CallBudgetMiddleware(
                [
                    (_TURN_SOFT_NUDGE_CALLS, _TURN_SOFT_NUDGE),
                    (_TURN_HARD_NUDGE_CALLS, _TURN_HARD_NUDGE),
                ]
            ),
            # Wall-clock analog of the call budget: converge before the
            # orchestrator's hard turn_timeout kills the turn and wastes its work.
            TimeBudgetMiddleware(
                [
                    (turn_timeout * _TURN_SOFT_NUDGE_TIME_FRAC, _TURN_SOFT_NUDGE),
                    (turn_timeout * _TURN_HARD_NUDGE_TIME_FRAC, _TURN_HARD_NUDGE),
                ]
            ),
            ModelCallLimitMiddleware(
                run_limit=_TURN_MAX_MODEL_CALLS, exit_behavior="end"
            ),
        ]
        return create_agent(
            make_model(model_id),
            tools=tools,
            system_prompt=system_prompt,
            middleware=middleware,
            checkpointer=checkpointer,
            name=persona.name,
        ).with_config({"recursion_limit": 500})

    # ── turn execution ───────────────────────────────────────────────────────

    async def run_turn(
        self,
        agent_name: str,
        thread_id: int,
        *,
        addressed: bool = False,
        task_prompt: str | None = None,
    ) -> str | None:
        """Run one agent turn against a thread; return the reply text.

        Returns None if the agent declined (NOTHING_TO_ADD) or produced no
        text -- except when addressed is set (a human named this agent
        directly), where the NOTHING_TO_ADD escape hatch is withdrawn and only
        a genuinely empty reply yields None. The agent's own messages are
        filtered at injection (they are
        already in its LangGraph thread as AI messages), so the seen offset
        never needs to skip them -- and a peer reply landing while this turn
        streams is injected on the next turn, in every interleaving. The
        offset advances only after the stream completes, so a crash mid-turn
        re-injects the same context when the message is replayed.

        Concurrency contract: the orchestrator never runs two turns for the
        same (thread, agent) simultaneously; this method relies on that for
        the offset read-advance and the checkpoint.
        """
        entry = self._agents[agent_name]
        # Persisted state (seen offset, checkpoint) keys on the stable identity so
        # a nickname toggle does not orphan it; agent_name (the live handle) still
        # drives sender filtering and attribution.
        skey = entry.persona.state_key
        seen = self._store.get_agent_seen(thread_id, skey)
        unseen = [
            m
            for m in self._store.thread_messages(thread_id)
            if m.id > seen and m.sender != agent_name and m.kind != MessageKind.PING
        ]
        # Prime recall: inject the memory index into the uncached tail when this
        # persona has memory. One git-grep per memory turn, threaded; empty for a
        # fresh store, in which case nothing is injected.
        mem_index = None
        if self._memory_enabled(entry.persona):
            from .memory import memory_index

            mem_index = await memory_index(self._cfg.memory.path)
        content = build_turn_content(
            unseen,
            addressed=addressed,
            task_prompt=task_prompt,
            memory_index=mem_index,
        )

        # The context generation folds into the checkpoint id, so a memory
        # compaction (which bumps the generation after summarizing the thread to
        # durable memory) starts this persona from a fresh checkpoint on its next
        # turn. Race-free: an in-flight turn keeps writing to the old id; only the
        # next turn reads the bumped one.
        gen = self._store.context_generation(thread_id)
        config = {"configurable": {"thread_id": f"{thread_id}:{skey}:g{gen}"}}
        text, usage = await self._stream(
            entry.graph,
            agent_name,
            {"messages": [{"role": "user", "content": content}]},
            config,
        )
        self._tokens.add(
            thread_id,
            agent_name,
            "turn",
            usage.input_tokens,
            usage.output_tokens,
            usage.cached_in,
            self._cfg.resolve_model(entry.persona.model_id),
            model_calls=usage.calls,
            max_call_input_tokens=usage.max_call_input,
        )
        log.info(
            "agent %s thread %d: %d in (%d cached) / %d out tokens"
            " over %d calls (max %d in/call)",
            agent_name,
            thread_id,
            usage.input_tokens,
            usage.cached_in,
            usage.output_tokens,
            usage.calls,
            usage.max_call_input,
        )
        if unseen:
            self._store.set_agent_seen(thread_id, skey, unseen[-1].id)
        # Every persona, including ones disabled this run, so an attribution to a
        # real but disabled agent is still stripped.
        text = sanitize_reply(text, agent_name, set(self._personas) - {agent_name})
        if addressed:
            # The directive told the agent to answer; honor whatever it
            # produced, only stripping a stray sentinel so it never reaches
            # the room. None only if the reply is genuinely empty.
            text = text.replace("NOTHING_TO_ADD", "").strip()
            return text or None
        if not text or "NOTHING_TO_ADD" in text:
            return None
        return text

    async def run_structured_turn(
        self,
        agent_name: str,
        content: str,
        *,
        extra_system: str,
        label: str = "structured",
    ) -> str | None:
        """Run a structured turn (extra_system forces a machine-readable result)
        and return the raw model text; None if the agent is unknown.

        A separate fresh graph from the persona's conversational one (it forces a
        structured result, so it must not share the chat checkpointer), built once
        per persona and reused -- its [persona system prompt + tools] prefix is
        constant across a stream of these turns, so it caches. The caller supplies
        the instruction (extra_system) and parses the returned text: the graph,
        lock, cache, and token accounting here are domain-neutral, the prompt and
        parse are the caller's. label tags token accounting for the caller's use.
        """
        entry = self._agents.get(agent_name)
        if entry is None:
            log.warning("%s: unknown agent %s", label, agent_name)
            return None
        graph = self._structured_agents.get(agent_name)
        if graph is None:
            available = {n: e.persona for n, e in self._agents.items()}
            graph = self._build_agent(
                entry.persona, available, None, extra_system=extra_system
            )
            self._structured_agents[agent_name] = graph
        config = {
            "configurable": {"thread_id": f"structured:{entry.persona.state_key}"}
        }
        # One structured turn at a time per persona: the shared thread id is what
        # lets the prefix cache accumulate across the stream, but the growing-cache
        # middleware keys its state on it -- two concurrent runs would interleave
        # one _PrefixState, trip each other's shrink detection, and worst case
        # send one turn's tail on top of a cache built from the other's prefix.
        lock = self._structured_locks.setdefault(agent_name, asyncio.Lock())
        async with lock:
            text, usage = await self._stream(
                graph,
                agent_name,
                {"messages": [{"role": "user", "content": content}]},
                config,
            )
        self._tokens.add(
            0,
            agent_name,
            label,
            usage.input_tokens,
            usage.output_tokens,
            usage.cached_in,
            self._cfg.resolve_model(entry.persona.model_id),
            model_calls=usage.calls,
            max_call_input_tokens=usage.max_call_input,
        )
        return text

    async def _emit(self, agent: str, event: str, detail: str) -> None:
        if self._on_event:
            try:
                await self._on_event(agent, event, detail)
            except Exception:
                log.exception("event hook failed")

    async def _stream(
        self, graph, agent_name: str, input: dict, config: dict
    ) -> tuple[str, _TurnUsage]:
        """Drive astream to completion, emitting tool events.

        Returns (text, usage). The usage callback aggregates input/output/cached
        across the turn's model calls and providers; the per-call tracer supplies
        the call count and the largest single-call input. cached_in is the
        prompt-cache-served subset of input_tokens (provider implicit/explicit
        caching), 0 when unsupported.
        """
        from airc_core.agent import TurnUsageHandler

        usage_cb = TurnUsageHandler()  # skips ceiling-summarization calls
        trace_cb = _CallTrace(agent_name, "turn")
        config = {**config, "callbacks": [usage_cb, trace_cb]}
        parts: list[str] = []
        seen_tool_ids: set[str] = set()
        cur_msg_id: str | None = None
        async for mode, data in graph.astream(
            input, config=config, stream_mode=["messages"]
        ):
            chunk, meta = data
            if not isinstance(chunk, AIMessageChunk):
                continue
            # A ceiling-summarization call nested in the turn streams through
            # messages mode too; its output restates the conversation, so
            # collected as reply text it posts as an echoed prompt. The call is
            # tagged nostream at the source; this backstop mirrors the lc_source
            # skip in the usage/trace handlers in case a chunk arrives anyway.
            if (meta or {}).get("lc_source") == "summarization":
                continue
            # Only the final model response's text is the answer (the agent loop
            # ends on a response without tool calls), and chunks of one response
            # share an id while a new model call gets a fresh one -- so a new id
            # resets the buffer. Without this, two leaks concatenate an earlier
            # response's text before the real answer: Gemini orders parts
            # model-side and can emit a self-note AFTER its tool call (past the
            # clear below), and a call retried mid-stream leaves the failed
            # attempt's partial text behind. Id-less chunks never reset --
            # continuation chunks may omit the id -- degrading to the old
            # text-after-last-tool-call semantics, never worse.
            if chunk.id is not None:
                if cur_msg_id is not None and chunk.id != cur_msg_id:
                    parts.clear()
                cur_msg_id = chunk.id
            for tc in chunk.tool_call_chunks or []:
                if tc.get("name"):
                    # Text preceding a tool call within a response is preamble
                    # ("Let me check foo.cc..."), not answer material; drop it.
                    # With the id reset above this scopes within one response,
                    # which still matters when the turn ends ON a tool-calling
                    # response (call cap) and as the id-less fallback.
                    parts.clear()
                    if tc.get("id") not in seen_tool_ids:
                        seen_tool_ids.add(tc.get("id"))
                        await self._emit(agent_name, "tool", tc["name"])
            blocks = (
                chunk.content
                if isinstance(chunk.content, list)
                else [{"type": "text", "text": chunk.content}]
            )
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    if text := block.get("text", ""):
                        parts.append(text)
        usage = usage_cb.usage_metadata.values()
        return "".join(parts), _TurnUsage(
            input_tokens=sum(u.get("input_tokens", 0) for u in usage),
            output_tokens=sum(u.get("output_tokens", 0) for u in usage),
            cached_in=sum(
                u.get("input_token_details", {}).get("cache_read", 0) for u in usage
            ),
            calls=trace_cb.calls,
            max_call_input=trace_cb.max_input_tokens,
        )

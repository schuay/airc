# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

from airc_room import orchestrator as orch_mod
from airc_room.config import Config, OrchestratorConfig
from airc_room.orchestrator import Orchestrator, parse_mentions
from airc_room.store import Message


KNOWN = {"perf", "compiler", "security"}


def _msg(mid: int, kind: str) -> Message:
    return Message(id=mid, thread_id=1, sender="x", kind=kind, text="t", ts=0.0)


class _FakeRoom:
    def __init__(self, msgs):
        self._msgs = msgs

    def thread_messages(self, thread_id):
        return self._msgs


def _orch(monkeypatch, msgs, cfg=None):
    monkeypatch.setattr(orch_mod, "make_model", lambda mid: object())
    return Orchestrator(
        cfg or Config(), _FakeRoom(msgs), runner=object(), store=object()
    )


def test_address_prefix():
    assert parse_mentions("perf: why is this slow?", KNOWN) == ["perf"]


def test_address_prefix_multiple_and_dedup():
    assert parse_mentions("perf, compiler: thoughts?", KNOWN) == ["perf", "compiler"]
    assert parse_mentions("perf compiler: thoughts?", KNOWN) == ["perf", "compiler"]
    assert parse_mentions("perf, perf: x", KNOWN) == ["perf"]


def test_address_prefix_only_known_tokens():
    # Unknown words with colons, URLs, and plain prose never route.
    assert parse_mentions("note: this is slow", KNOWN) == []
    assert parse_mentions("see https://example.com/x for details", KNOWN) == []
    assert parse_mentions("anyone seen this deopt?", KNOWN) == []
    assert parse_mentions("meet at 12:30 sharp", KNOWN) == []
    # A known handle with a colon routes even when the list form is voided
    # by unknown tokens.
    assert parse_mentions("perf and compiler: thoughts?", KNOWN) == ["compiler"]


def test_address_prefix_case_insensitive():
    assert parse_mentions("Perf: check this", KNOWN) == ["perf"]


def test_at_is_not_an_address():
    # @ is dropped entirely: it never forces a reply, anywhere in the text.
    assert parse_mentions("@perf can you check this?", KNOWN) == []
    assert parse_mentions("perf: ping @security too", KNOWN) == ["perf"]
    assert parse_mentions("mail me at user@example.com", KNOWN) == []


def test_agent_streak_counts_trailing_agents(monkeypatch):
    msgs = [_msg(1, "human"), _msg(2, "agent"), _msg(3, "agent")]
    assert _orch(monkeypatch, msgs)._agent_streak(1) == 2


def test_agent_streak_resets_after_human(monkeypatch):
    msgs = [_msg(1, "agent"), _msg(2, "agent"), _msg(3, "human")]
    assert _orch(monkeypatch, msgs)._agent_streak(1) == 0
    assert _orch(monkeypatch, [])._agent_streak(1) == 0


def test_pressure_empty_below_soft_budget(monkeypatch):
    cfg = Config(orchestrator=OrchestratorConfig(soft_turn_budget=8, max_turns=24))
    o = _orch(monkeypatch, [], cfg)
    assert o._pressure(7) == ""


def test_pressure_kicks_in_at_soft_budget(monkeypatch):
    cfg = Config(orchestrator=OrchestratorConfig(soft_turn_budget=8, max_turns=24))
    o = _orch(monkeypatch, [], cfg)
    p = o._pressure(8)
    assert "8 agent messages in a row" in p
    assert "converge" in p


def test_parse_coordinator_reply():
    from airc_room.orchestrator import parse_coordinator_reply

    known = {"perf", "compiler", "security"}
    assert parse_coordinator_reply("NOBODY", known, 2) == []
    assert parse_coordinator_reply("NOBODY -- humans talking", known, 2) == []
    assert parse_coordinator_reply("perf", known, 2) == ["perf"]
    assert parse_coordinator_reply("perf, compiler -- both relevant", known, 2) == [
        "perf",
        "compiler",
    ]
    # Cap, dedup, unknown names, garbage: all fail toward fewer/none.
    assert parse_coordinator_reply("perf, compiler, security", known, 2) == [
        "perf",
        "compiler",
    ]
    assert parse_coordinator_reply("perf, perf", known, 2) == ["perf"]
    assert parse_coordinator_reply("ghost, banshee", known, 2) == []
    assert parse_coordinator_reply("", known, 2) == []
    assert parse_coordinator_reply("Sure! I think maybe perf?", known, 2) == []
    # A unicode dash the model substitutes for " -- " still routes (not [] ).
    assert parse_coordinator_reply("perf — most relevant", known, 2) == ["perf"]
    assert parse_coordinator_reply("compiler – lowering", known, 2) == ["compiler"]


def test_humans_moved_on():
    from airc_room.orchestrator import humans_moved_on

    known = {"perf"}

    def m(kind, sender, text="t"):
        return Message(id=0, thread_id=1, sender=sender, kind=kind, text=text, ts=0.0)

    # Two humans past the last agent message, neither addressing an agent.
    msgs = [m("human", "a"), m("agent", "perf"), m("human", "a"), m("human", "b")]
    assert humans_moved_on(msgs, known) is True
    # Only one trailing human: not yet moved on.
    assert humans_moved_on(msgs[:-1], known) is False
    # A trailing human addressed an agent: engaged, not moved on.
    msgs2 = msgs[:-1] + [m("human", "b", "perf: what do you think?")]
    assert humans_moved_on(msgs2, known) is False
    # No agent has spoken at all.
    assert humans_moved_on([m("human", "a"), m("human", "b")], known) is False


def test_address_anywhere_in_text():
    text = "Long analysis here.\nMore detail. compiler: is there a plan for ICs?"
    assert parse_mentions(text, KNOWN) == ["compiler"]
    text = "perf: take a look\nsome detail\nsecurity: you too"
    assert parse_mentions(text, KNOWN) == ["perf", "security"]


def test_bracketed_attribution_is_inert():
    # Transcript/display attribution uses "[sender] text"; quoting or echoing
    # it must never force a reply (the colon belongs to addressing alone).
    text = "[perf] I measured it\n[users/42] are you sure?\nSo that is that."
    assert parse_mentions(text, KNOWN) == []

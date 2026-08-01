# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Autonomous long-term memory for the room.

A git repo of markdown entries with frontmatter (one fact each) that the personas
maintain themselves: they write when a human corrects them or a durable fact
appears, and recall on later turns. The MECHANISM lives here in core; the store's
SCHEMA (types, required frontmatter, validator hook, templates) travels with the
store repo, and an app is a thin pointer (config + a persona tool_group + a few
lines of persona prose). This is what lets a coding room and a grocery room share
one implementation with different memories.

Public surface:
- make_memory_tools(root) -- the jailed search/read/write/edit/delete tools
  (auto-commit).
- memory_index(root) -- the derived one-line index for per-turn injection.
- MEMORY_RULES -- the default write-discipline block appended to a memory-enabled
  persona's system prompt.
"""

from __future__ import annotations

from .index import memory_index
from .tools import make_memory_tools

# The tool_group name that grants the memory tools. Reserved by core when
# [airc.memory] is enabled; a persona lists it in agent.toml tool_groups exactly
# as for an MCP group.
MEMORY_GROUP = "memory"

# Appended to the system prompt of any memory-enabled persona (like ROOM_RULES).
# Always-active, so it is cached with the prefix. Generic: an app adds domain
# specifics in each persona's system.md. The two load-bearing disciplines
# (search-before-write, read-before-relying) live here because without them the
# store rots into near-duplicate one-liners or into confidently-recited hooks.
MEMORY_RULES = """\
## Long-term memory

You have a durable memory: a small store of notes you maintain yourself, one fact
per note. Its index (a line per note) is injected each turn under "Memory"; read a
note in full with memory_read before you rely on it -- the index line is a hook,
not the fact.

Write a note when, and only when, something durable is worth carrying to a later
conversation:
- a human CORRECTS you, or states a standing preference or convention -- record
  what to do differently, and why.
- a genuinely new, lasting fact about the people, the project, or how they want
  you to work appears.

Do NOT record what is already in the code, git history, or project docs; passing
task state; or anything you could re-derive next time. When in doubt, do not write
-- a noisy store is worse than a small one.

Before writing, memory_search first: prefer updating an existing note over
creating a near-duplicate. Keep each note to the durable substance. Writes are
committed for you; a malformed note is rejected with the reason, so fix and write
again.

When a note turns out to be WRONG or is superseded outright, delete it with
memory_delete -- a store that keeps stale facts around is worse than a small one,
and the entry stays in git history. If the fact merely changed, update the note
instead, so its history stays in one place. Never blank a note to "remove" it: an
empty entry still shows up in the index and reads as a fact you have forgotten."""

__all__ = ["MEMORY_GROUP", "MEMORY_RULES", "make_memory_tools", "memory_index"]

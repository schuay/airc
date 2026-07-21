"""airc-tools: local action primitives for airc coding agents.

Three tools, deliberately small: a stateless `shell` (which doubles as the
list/grep/read primitive), a verbatim `read_file` whose format is locked to the
editor, and an `edit_file` that ports aider's SEARCH/REPLACE apply engine behind
a structured tool call. These are library functions bound directly by deepagent's
worktree tools and airc-room's memory tools -- no MCP server wrapper.
"""

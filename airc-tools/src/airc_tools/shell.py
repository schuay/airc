# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""shell: a stateless `bash -lc` runner.

Stateless on purpose. Each call is a fresh process, so there is no session state
to demux -- no END-marker plumbing to know when a command finished (the process
boundary does that), and no surprising cwd/env carryover between calls. The cost,
no persistent cd/venv, is paid by composing one pipeline per call, which is also
the whole token-efficiency play: full shell means the model pipes rg/tail/head
itself instead of us wrapping read/list/grep.
"""

import asyncio
import contextlib
import os
import re
import signal
import time
from typing import Protocol

from .limits import MAX_SHELL_CAPTURE, MAX_SHELL_OUTPUT, head_tail


class Confinement(Protocol):
    """Whatever produces an argv prefix that runs a command confined.

    Structural on purpose: all this module ever does with a sandbox is prepend
    `wrapper()`, so naming a concrete class here would import a confinement
    implementation into the one module that does not need one. airc_tools.sandbox
    satisfies it; so does anything else a caller brings.
    """

    def wrapper(self) -> list[str]: ...


# Build footguns to refuse before running. A raw `ninja`/`siso ninja` bypasses
# autoninja (and raw ninja fails on the siso state file, then tempts `gn clean`);
# `gn clean` wipes the build into a slow cold rebuild. Match the program at a
# command position (start, or after a &&/;/| separator) so autoninja, a
# `build.ninja` path, or `grep ninja` are untouched. A deliberately simple
# regex, not a shell parser -- it only sees the agent's own command, never
# autoninja's internal ninja/siso calls.
# Anchored at a command position (start, or after a ;/&/| separator) and matched
# against the whole word, so `autoninja` -- the one sanctioned build entry point
# -- is not caught by the `ninja` alternative. gm.py and gclient are reachable by
# a path or through an interpreter, so allow a leading dir prefix and an optional
# python/vpython launcher; `cat tools/dev/gm.py` stays clear because `cat` is
# neither.
_BUILD_TRAP = re.compile(
    r"(?:^|[;&|]+)\s*"
    r"(?:(?:\S*/)?(?:v?python3?)\s+)?"
    r"(?:\S*/)?"
    r"(?:ninja|siso\s+ninja|gn\s+clean|gm\.py|gclient)\b",
    re.IGNORECASE,
)
_BUILD_TRAP_MSG = (
    "error: raw `ninja` / `siso ninja` / `gn clean` / `gm.py` / `gclient` is"
    " disabled -- builds go through autoninja. Use `autoninja -C out/<build>"
    " <target>`, or add `-o` (`autoninja -o -C ...`) for an offline (no-RBE)"
    " build. The worktree is already build-ready: gm.py and gclient would"
    " re-sync deps against the shared read-only checkout, and `gn clean` forces"
    " a slow cold rebuild."
)


# Set in the child, not a session (there is none): keep tools noninteractive so
# nothing blocks on a pager or a credential prompt waiting for a tty that will
# never answer -- that would just burn the whole timeout.
#
# AI_AGENT belongs to the same family: every shell run through these tools is an
# agent's shell, and autoninja/siso key off AI_AGENT (any non-empty value) to
# add --quiet, dropping the per-action build progress that would otherwise flood
# the model's context one line per compiled file. This module merges it over
# os.environ for the unsandboxed child; a sandboxed one starts from --clearenv
# and gets exactly the env its profile carries, so whoever BUILDS that profile
# has to merge this dict in -- which is why it is public. Splitting it that way
# is deliberate: a profile that states what the box's environment is, and then
# silently receives more of it from the module that runs the command, is a
# profile nobody can read. (The daemon's own prebuild is handled separately.)
DEFANG_ENV = {
    "DEBIAN_FRONTEND": "noninteractive",
    "GIT_PAGER": "cat",
    "PAGER": "cat",
    "GIT_TERMINAL_PROMPT": "0",
    "AI_AGENT": "airc-tools",
}


# A timeout is bounded but not free: it costs the caller whatever budget it was
# given, so the default is deliberately short and the agent raises it per call
# for work that is honestly long. That only works if the agent can tell the two
# cases apart, so the hint turns on the one thing it can read off the output it
# already has -- was the command still talking when we killed it -- rather than
# on a rule about hangs it has no way to evaluate. SLOW/HUNG are literal labels
# to give that decision something to match on.
#
# No fix-it flags are spelled out on purpose: named remedies get pasted at the
# next hang whether or not they address it. The cure depends on what is
# blocking, which only the caller can see.
#
# The last line closes the escalation loop -- without it, a hang that already
# survived one raise invites the next one.
#
# SLOW spells out the pipeline case because the observed failure was re-running
# only the tail of a chain: the kill is by process GROUP, so an earlier stage is
# neither still running nor holding output for the retry to consume.
_TIMEOUT_HINT = (
    "Which case is this? Look at the output below.\n\n"
    "SLOW -- output was still arriving when it was killed. The budget was just"
    " too short. Retry the WHOLE command with timeout=<seconds> (600 covers an"
    " incremental build). If it was a pipeline, the whole chain was killed"
    " together: re-run it entire, first command included -- nothing is still"
    " running and no earlier stage's output survived.\n\n"
    "HUNG -- no output, or it stopped long before the kill. Something is"
    " waiting forever: a prompt, a pager, a lock, a network call. A longer"
    " timeout waits longer and burns the turn. Do not raise it -- change the"
    " command so it cannot block, or run a smaller piece to find where it"
    " stalls.\n\n"
    "Already raised it once and hit this again? Treat it as HUNG."
)


def _killpg(proc):
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


async def run_shell(
    command: str,
    cwd: str | None = None,
    timeout: float = 30.0,
    sandbox: Confinement | None = None,
) -> str:
    if _BUILD_TRAP.search(command):
        return _BUILD_TRAP_MSG
    root = cwd or os.environ.get("AIRC_TOOLS_ROOT") or None
    env = {**os.environ, **DEFANG_ENV}
    # Under a sandbox the wrapper owns cwd (--chdir) and environment
    # (--clearenv + --setenv), so the host-side cwd/env are irrelevant to the
    # command; kill-by-group and the capture plumbing below are unchanged
    # because the whole wrapper chain shares the child's process group.
    try:
        argv = [*(sandbox.wrapper() if sandbox else []), "bash", "-lc", command]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=None if sandbox else root,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            # Own process group so a timeout kills the whole tree, not just the
            # top-level bash (which may be waiting on children).
            start_new_session=True,
        )
    except (FileNotFoundError, NotADirectoryError) as e:
        return f"error: could not start shell in {root!r}: {e}"

    captured = bytearray()
    hit_ceiling = False

    async def drain():
        # Stop the moment we cross the capture ceiling: an unbounded producer
        # would otherwise stream until the timeout, wasting CPU. We keep the head
        # and kill below.
        nonlocal hit_ceiling
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                return
            captured.extend(chunk)
            if len(captured) >= MAX_SHELL_CAPTURE:
                hit_ceiling = True
                return

    start = time.monotonic()
    timed_out = False
    try:
        await asyncio.wait_for(drain(), timeout=timeout)
    except TimeoutError:
        timed_out = True
    except BaseException:
        # External cancellation (an embedder's turn timeout or daemon shutdown)
        # or a drain error: do not leak the child. Kill its group before
        # re-raising, so a running build cannot outlive the cancelled call and
        # collide with a retry in the same directory. asyncio's child watcher
        # reaps the killed process; awaiting proc.wait() here could re-raise the
        # cancellation before the kill lands.
        _killpg(proc)
        raise
    elapsed = time.monotonic() - start

    if timed_out or hit_ceiling:
        _killpg(proc)
    try:
        # EOF on stdout is not exit: a child that closed its fds but lingers
        # (e.g. `exec >&-; sleep 60`) would otherwise hold this wait past the
        # timeout, unbounded. Give it the remaining budget, then kill. The
        # BaseException arm mirrors the drain guard above -- a cancellation
        # landing during this wait must not leak the child either.
        async with asyncio.timeout(max(0.1, timeout - elapsed)):
            rc = await proc.wait()
    except TimeoutError:
        timed_out = True
        _killpg(proc)
        rc = await proc.wait()
    except BaseException:
        _killpg(proc)
        raise

    output = head_tail(
        bytes(captured[:MAX_SHELL_CAPTURE]).decode("utf-8", errors="replace"),
        MAX_SHELL_OUTPUT,
    )
    if timed_out:
        status = (
            f"timeout after {timeout:g}s; the command and its process group"
            f" were killed.\n\n{_TIMEOUT_HINT}\n\n"
        )
        status += "partial output:" if output else "(no output before the kill)"
    elif hit_ceiling:
        status = (
            f"output exceeded {MAX_SHELL_CAPTURE} bytes (killed, head kept);"
            " pipe through rg/head/tail to narrow:"
        )
    else:
        status = f"exit {rc} in {elapsed:.1f}s:"
    return f"{status}\n{output}" if output else status

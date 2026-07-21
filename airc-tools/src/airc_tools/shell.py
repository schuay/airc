"""shell: a stateless `bash -lc` runner.

Stateless on purpose. Each call is a fresh process, so there is no session state
to demux -- no END-marker plumbing to know when a command finished (the process
boundary does that), and no surprising cwd/env carryover between calls. The cost,
no persistent cd/venv, is paid by composing one pipeline per call, which is also
the whole token-efficiency play: full shell means the model pipes rg/tail/head
itself instead of us wrapping read/list/grep.
"""

import asyncio
import os
import re
import signal
import time
from typing import TYPE_CHECKING

from .limits import MAX_SHELL_CAPTURE, MAX_SHELL_OUTPUT, head_tail

if TYPE_CHECKING:
    from .sandbox import Sandbox

# Build footguns to refuse before running. A raw `ninja`/`siso ninja` bypasses
# autoninja (and raw ninja fails on the siso state file, then tempts `gn clean`);
# `gn clean` wipes the build into a slow cold rebuild. Match the program at a
# command position (start, or after a &&/;/| separator) so autoninja, a
# `build.ninja` path, or `grep ninja` are untouched. A deliberately simple
# regex, not a shell parser -- it only sees the agent's own command, never
# autoninja's internal ninja/siso calls.
_BUILD_TRAP = re.compile(r"(?:^|[;&|]+)\s*(?:ninja|siso\s+ninja|gn\s+clean)\b", re.I)
_BUILD_TRAP_MSG = (
    "error: raw `ninja` / `siso ninja` / `gn clean` is disabled -- builds go"
    " through autoninja. Use `autoninja -C out/<build> <target>`, or add `-o`"
    " (`autoninja -o -C ...`) for an offline (no-RBE) build. Never `gn clean`:"
    " it forces a slow cold rebuild."
)

# Set in the child, not a session (there is none): keep tools noninteractive so
# nothing blocks on a pager or a credential prompt waiting for a tty that will
# never answer -- that would just burn the whole timeout.
#
# AI_AGENT belongs to the same family: every shell run through these tools is an
# agent's shell, and autoninja/siso key off AI_AGENT (any non-empty value) to
# add --quiet, dropping the per-action build progress that would otherwise flood
# the model's context one line per compiled file. Set here so it reaches BOTH the
# unsandboxed child (this dict is merged over os.environ below) and the sandboxed
# one (the wrapper --setenv's this dict after --clearenv), which is the only
# place that covers both -- the daemon's own prebuild is handled separately.
_DEFANG_ENV = {
    "DEBIAN_FRONTEND": "noninteractive",
    "GIT_PAGER": "cat",
    "PAGER": "cat",
    "GIT_TERMINAL_PROMPT": "0",
    "AI_AGENT": "airc-tools",
}


def _killpg(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


async def run_shell(
    command: str,
    cwd: str | None = None,
    timeout: float = 30.0,
    sandbox: "Sandbox | None" = None,
) -> str:
    if _BUILD_TRAP.search(command):
        return _BUILD_TRAP_MSG
    root = cwd or os.environ.get("AIRC_TOOLS_ROOT") or None
    env = {**os.environ, **_DEFANG_ENV}
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
    except asyncio.TimeoutError:
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
        status = f"timeout after {timeout:g}s (killed); partial output:"
    elif hit_ceiling:
        status = (
            f"output exceeded {MAX_SHELL_CAPTURE} bytes (killed, head kept);"
            " pipe through rg/head/tail to narrow:"
        )
    else:
        status = f"exit {rc} in {elapsed:.1f}s:"
    return f"{status}\n{output}" if output else status

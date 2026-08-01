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
    re.I,
)
_BUILD_TRAP_MSG = (
    "error: raw `ninja` / `siso ninja` / `gn clean` / `gm.py` / `gclient` is"
    " disabled -- builds go through autoninja. Use `autoninja -C out/<build>"
    " <target>`, or add `-o` (`autoninja -o -C ...`) for an offline (no-RBE)"
    " build. The worktree is already build-ready: gm.py and gclient would"
    " re-sync deps against the shared read-only checkout, and `gn clean` forces"
    " a slow cold rebuild."
)

# Authoring a SOURCE file through the shell, refused so the model uses the
# purpose-built tools. This is a repeat offender: the system prompt has forbidden
# it in prose since the tools existed, and agents still fall back to a heredoc
# after one failed edit_file match -- prose the model can rationalize past does
# not hold, a refused call does.
#
# The discriminator is the TARGET, not the command. Capturing a command's output
# into a log is legitimate and instructed elsewhere ("redirect a noisy build to a
# file under the casefile dir"), so a redirect is judged by what it writes to: a
# file with a source extension is authoring, anything else is capture. That keeps
# every build/test/benchmark invocation -- the expensive things to break -- out of
# the trap, because none of them redirect into a .cc or a .js.
#
# Deliberately conservative: a missed shell write costs one badly-authored file
# (and the prose still says not to), while a false positive blocks a legitimate
# command and burns a turn. Anything not clearly authoring is allowed through.
#
# So this is a nudge off the well-worn spellings, NOT a boundary. It matches
# literal suffixes on a statically visible redirect target, which `> patch.txt`,
# a computed name, `dd of=`, `ed`, or write-then-rename all walk around, and the
# apply side is not closed either. Do not build anything on top of it that
# assumes an agent CANNOT author a patch; real prevention would be removing
# `git apply`/`patch` from the sandbox PATH.
#
# .diff/.patch are the one place the target-not-command rule is knowingly bent:
# `git diff > review.diff` is output capture by that rule, but the same shape is
# how a heredoc patch gets staged, and the suffix cannot tell them apart. The
# refusal message names the .log alternative so a legitimate capture recovers in
# one turn.
_SOURCE_SUFFIXES = (
    # Source and headers.
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".py",
    ".rs",
    ".java",
    # V8/Chromium build and config files, which are code for our purposes.
    ".gn",
    ".gni",
    ".gyp",
    ".gypi",
    ".star",
    ".tq",
    # Patch and diff files: writing patch files via heredoc to apply later is
    # authoring source code through the shell.
    ".diff",
    ".patch",
)
# Redirect targets, ignoring fd duplication (`2>&1`, `>&2`) and fd-prefixed
# redirects (`2> x`). The lookbehind keeps `2>&1` out; `[^&]` on the target keeps
# `>&2` out. Quotes are stripped by the caller.
_REDIRECT = re.compile(r">>?\s*(?!&)(['\"]?)([^\s'\";|&<>]+)\1")
# `tee` and `tee -a`, whose non-flag arguments are the files it writes.
_TEE = re.compile(r"(?:^|[;&|]+)\s*tee\b([^;&|<>]*)")
# In-place editors and patch appliers: these only ever rewrite an existing file,
# so unlike a redirect there is no capture reading of them.
_INPLACE = re.compile(
    r"(?:^|[;&|]+)\s*(?:sed\s+(?:-[^\s]*\s+)*-i|perl\s+(?:-[^\s]*\s+)*-i"
    r"|git\s+apply|patch\b)",
    re.I,
)
# `git apply --check` / `--stat` / `--summary` only inspect a patch; they write
# nothing, and refusing them would block a legitimate way to test whether a diff
# applies. Same for `patch --dry-run`.
_DRY_RUN = re.compile(r"--(?:check|stat|summary|dry-run)\b")
_WRITE_TRAP_MSG = (
    "error: authoring a source file through the shell is disabled. Use"
    " write_file (whole file) or edit_file (part of one) -- they are the only"
    " ways to write file content here, and a shell-written file counts as a"
    " failed edit. If an edit_file SEARCH did not match, read_file that region"
    " again and retry with the exact bytes; do not rewrite the file from the"
    " shell. Redirecting a command's OUTPUT to a log is fine -- that is not"
    " what this refused; send it to the casefile dir with a .log name."
)


def _authors_source(command: str) -> bool:
    """Whether `command` writes CONTENT into a source file.

    Not a shell parser -- it reads the agent's own one-liner, and errs toward
    allowing: every branch here has to be a clear authoring shape, because the
    cost of a wrong refusal (a dead build command, a burnt turn) is far above
    the cost of a missed one (prose still forbids it).
    """
    if _INPLACE.search(command) and not _DRY_RUN.search(command):
        return True
    targets = [m.group(2) for m in _REDIRECT.finditer(command)]
    for m in _TEE.finditer(command):
        targets += [a for a in m.group(1).split() if not a.startswith("-")]
    return any(t.lower().endswith(_SOURCE_SUFFIXES) for t in targets)


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
_TIMEOUT_HINT = (
    "Which case is this? Look at the output below.\n\n"
    "SLOW -- output was still arriving when it was killed. The budget was just"
    " too short. Retry the same command with timeout=<seconds> (600 covers an"
    " incremental build).\n\n"
    "HUNG -- no output, or it stopped long before the kill. Something is"
    " waiting forever: a prompt, a pager, a lock, a network call. A longer"
    " timeout waits longer and burns the turn. Do not raise it -- change the"
    " command so it cannot block, or run a smaller piece to find where it"
    " stalls.\n\n"
    "Already raised it once and hit this again? Treat it as HUNG."
)


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
    if _authors_source(command):
        return _WRITE_TRAP_MSG
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

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

import asyncio
import time

import pytest

from airc_tools.limits import MAX_SHELL_CAPTURE
from airc_tools.shell import _BUILD_TRAP_MSG, _WRITE_TRAP_MSG, run_shell


async def test_basic_exit_zero():
    out = await run_shell("echo hello")
    assert "exit 0" in out
    assert "hello" in out


async def test_build_trap_blocks_raw_ninja_and_gn_clean():
    # These never run; the shell returns the redirect to autoninja.
    for cmd in (
        "ninja -C out/x64.optdebug d8",
        "siso ninja --offline -C out d8",
        "gn clean out/x64.optdebug",
        "cd out && ninja",
        "rm -rf x; gn clean out",
    ):
        assert await run_shell(cmd) == _BUILD_TRAP_MSG, cmd


async def test_build_trap_allows_autoninja_and_lookalikes():
    # autoninja, autoninja -o, a build.ninja path, gn gen, and a bare mention of
    # "ninja" mid-command must NOT be trapped -- they execute normally.
    for cmd in (
        "echo autoninja -C out d8",
        "echo autoninja -o -C out d8",
        "echo out/x64.optdebug/build.ninja",
        "echo gn gen out",
        "echo running ninja now",
    ):
        out = await run_shell(cmd)
        assert out != _BUILD_TRAP_MSG and "exit 0" in out, cmd


async def test_write_trap_blocks_authoring_a_source_file():
    # The failure this closes: the system prompt has forbidden shell-authored
    # files in prose since the tools existed, and agents still fall back to a
    # heredoc after one failed edit_file match. Prose the model can rationalize
    # past does not hold; a refused call does.
    for cmd in (
        "cat <<'EOF' > test/mjsunit/regress/regress-1.js\nassertTrue(1);\nEOF",
        "echo 'assertTrue(true);' > test/mjsunit/regress/regress-1.js",
        "printf '%s' 'int x;' > src/compiler/foo.cc",
        "sed -i 's/return 1/return 2/' src/compiler/globals.h",
        "sed -i.bak 's/a/b/' src/a.cc",
        "perl -i -pe 's/a/b/' src/a.cc",
        "git apply /cf/patch.diff",
        "patch -p1 < /cf/patch.diff",
        "cat <<'EOF' > patch.diff\n--- a/src/a.cc\n+++ b/src/a.cc\nEOF",
        "echo x > patch.patch",
        "tee src/compiler/foo.cc <<'EOF'\nint x;\nEOF",
        "echo x >> BUILD.gn",
    ):
        assert await run_shell(cmd) == _WRITE_TRAP_MSG, cmd


async def test_write_trap_allows_capturing_output():
    # The distinction the trap turns on: authoring CONTENT is refused, capturing
    # a command's OUTPUT is not -- the build skill explicitly tells the agent to
    # redirect a noisy build to a casefile log, so trapping redirects wholesale
    # would forbid instructed work and break every build/test invocation.
    # A false positive here costs a burnt turn; these must all run.
    for cmd in (
        "echo autoninja -C out d8 > /dev/null",
        "echo build output 2>&1 | tee /dev/null",
        "echo x > /dev/null 2>&1",
        "echo 'note' > /dev/null",
        # Reads and inspections that merely NAME source files.
        "echo grep -c foo src/a.cc",
        "echo diff src/a.cc src/b.cc",
        "echo git add test/mjsunit/regress/regress-1.js",
        "echo git checkout src/compiler/globals.h",
        "echo out/x64.optdebug/d8 test/mjsunit/regress/regress-1.js",
        # sed WITHOUT -i is a filter, not an in-place edit.
        "echo sed -n 1,50p src/a.cc",
        # Patch tools that only inspect.
        "git apply --check /dev/null",
        "echo patch --dry-run -p1",
    ):
        out = await run_shell(cmd)
        assert out != _WRITE_TRAP_MSG, cmd


async def test_nonzero_exit():
    out = await run_shell("exit 3")
    assert "exit 3" in out


async def test_stderr_merged():
    out = await run_shell("echo to_stderr 1>&2")
    assert "to_stderr" in out


async def test_defang_env():
    out = await run_shell("echo PAGER=$PAGER GIT_TERMINAL_PROMPT=$GIT_TERMINAL_PROMPT")
    assert "PAGER=cat" in out
    assert "GIT_TERMINAL_PROMPT=0" in out


async def test_cwd(tmp_path):
    (tmp_path / "marker").write_text("x")
    out = await run_shell("ls", cwd=str(tmp_path))
    assert "marker" in out


async def test_timeout_kills_fast():
    start = time.monotonic()
    out = await run_shell("sleep 10", timeout=0.5)
    assert time.monotonic() - start < 5  # killed, not waited out
    assert "timeout" in out


async def test_timeout_carries_the_hint_either_way():
    # The hint is the whole point of the short default: a timeout the agent
    # cannot classify just gets retried longer. It has to ride along even when
    # the command died silent (the hang case), where there is no partial output
    # to append it to.
    noisy = await run_shell("echo working; sleep 10", timeout=0.5)
    assert "SLOW" in noisy and "HUNG" in noisy
    assert "partial output:" in noisy and "working" in noisy

    silent = await run_shell("sleep 10", timeout=0.5)
    assert "SLOW" in silent and "HUNG" in silent
    assert "(no output before the kill)" in silent


async def test_output_ceiling():
    # Emit ~1MB, over the capture ceiling: killed, head kept, flagged.
    out = await run_shell("head -c 1000000 /dev/zero | tr '\\0' a")
    assert "exceeded" in out
    assert len(out) < MAX_SHELL_CAPTURE  # not the full megabyte


async def test_cancel_kills_child_process(tmp_path):
    # An embedder cancelling the turn (timeout/shutdown) must kill the child, not
    # leave it running detached. The child appends forever; after cancel + a
    # grace window its output must stop growing (killed), and CancelledError must
    # propagate.
    out = tmp_path / "out"
    task = asyncio.create_task(
        run_shell(f"while true; do echo x >> {out}; sleep 0.02; done", timeout=60)
    )
    for _ in range(500):  # wait until the child is up and writing
        if out.exists() and out.stat().st_size > 0:
            break
        await asyncio.sleep(0.01)
    assert out.exists() and out.stat().st_size > 0

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(0.3)  # any surviving loop would keep appending
    size1 = out.stat().st_size
    await asyncio.sleep(0.3)
    assert out.stat().st_size == size1  # writes stopped -> the group was killed


async def test_lingering_child_with_closed_fds_is_bounded():
    # EOF on stdout is not exit: a child that closes its fds but lingers used
    # to hold the exit wait unbounded, reporting "exit 0 in 0.1s" only after
    # the child finally died on its own.
    start = time.monotonic()
    out = await run_shell("exec >/dev/null 2>&1; sleep 30", timeout=0.5)
    assert time.monotonic() - start < 5  # killed, not waited out
    assert "timeout" in out

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""What the box may touch, asserted as effective mounts rather than as argv
trivia.

Most of these tests are about ORDER, because order is the whole mechanism: bwrap
mounts in sequence and a later mount wins where two overlap, so "which mount is
in effect at this path" is decided by nothing else. That makes them easy to write
badly. An assertion that merely finds both binds in the list passes whichever way
round they are, and an assertion phrased against the list a caller PASSED IN is a
tautology about that list, not a check on what the box gets. Every ordering test
here therefore reads indices out of the resolved mounts or the emitted argv (what
bwrap is actually handed), and the end-to-end ones run a real box and ask the
kernel.
"""

import dataclasses
import shutil
import socket
import threading
from pathlib import Path

import pytest
from airc_tools.sandbox import (
    RO,
    RW,
    Bind,
    BindOver,
    Overlay,
    Sandbox,
    Seal,
)
from airc_tools.shell import DEFANG_ENV, run_shell


def _dest_at(argv: list[str], flag: str, dst: str) -> int:
    """argv index of the mount spec with this flag and destination.

    Ordering assertions compare these: an index into the list bwrap is handed is
    the only thing that says which mount wins.
    """
    for i in range(len(argv)):
        if argv[i] != flag:
            continue
        # --tmpfs/--remount-ro take one operand (the destination); the bind forms
        # take two, destination last.
        n = 1 if flag in ("--tmpfs", "--remount-ro", "--tmp-overlay") else 2
        if argv[i + n] == dst:
            return i
    raise AssertionError(f"no {flag} at {dst} in argv")


def _ops(sb: Sandbox) -> list[tuple[str, str]]:
    """The resolved list as (op, destination) pairs -- the profile's own
    statement of what it mounts, in order."""
    return [(m.op, str(m.dst)) for m in sb.resolve()]


@pytest.fixture
def profile(tmp_path):
    root = tmp_path / "wt"
    casefile = tmp_path / "casefile"
    dep = tmp_path / "dep"
    for d in (root, casefile, dep):
        d.mkdir()
    return Sandbox(
        root=root,
        binds=(Bind(dep, RO, optional=True), Bind(casefile, RW, optional=True)),
        tmpfs=(("/tmp", 1 << 20),),
        # As a real caller builds it: the profile carries the whole environment,
        # defang defaults included, because the wrapper adds nothing to it. HOME
        # stays first -- tests below read env[0] for it.
        env=(("HOME", str(tmp_path)), *DEFANG_ENV.items()),
        memory_max="1G",
        tasks_max=64,
        use_cgroup=False,
    )


def test_ai_agent_is_in_the_defang_set(profile):
    # siso goes quiet on AI_AGENT (any non-empty value), and DEFANG_ENV is where
    # that lives for both shells: merged over os.environ for the unsandboxed one,
    # and merged into the PROFILE by whoever builds it for the sandboxed one --
    # the wrapper starts from --clearenv and adds nothing of its own.
    assert DEFANG_ENV.get("AI_AGENT")
    assert "--setenv AI_AGENT" in " ".join(profile.wrapper())


def test_the_profile_env_is_the_whole_env(tmp_path):
    # Verbatim, in both directions: everything the profile names is set, and
    # nothing it does not name is. A wrapper that quietly adds a variable makes
    # the profile unreadable as a statement of what the box gets -- and it is the
    # only place the defang defaults could come from, so their absence here is
    # what proves the merge really moved to the caller.
    root = tmp_path / "root"
    root.mkdir()
    argv = Sandbox(root=root, env=(("ONLY", "1"),), use_cgroup=False).wrapper()
    setenv = [argv[i + 1] for i, a in enumerate(argv) if a == "--setenv"]
    assert setenv == ["ONLY"]


def test_wrapper_shape(profile):
    argv = profile.wrapper()
    # bwrap-only (cgroup off); command slots append after the prefix.
    assert argv[0] == "bwrap"
    s = " ".join(argv)
    dep, casefile = profile.binds[0].path, profile.binds[1].path
    assert f"--bind {profile.root} {profile.root}" in s
    assert f"--ro-bind {dep} {dep}" in s
    assert f"--bind {casefile} {casefile}" in s
    assert "--cap-drop ALL" in s
    assert "--die-with-parent" in s
    assert "--clearenv" in s
    # Env allowlist includes the defang set and the profile's own vars.
    assert "--setenv PAGER cat" in s
    assert f"--setenv HOME {profile.env[0][1]}" in s
    assert f"--chdir {profile.root}" in s


def test_tmpfs_size_precedes_mount(profile):
    argv = profile.wrapper()
    i = argv.index("--tmpfs")
    assert argv[i - 2 : i + 2] == ["--size", str(1 << 20), "--tmpfs", "/tmp"]


def test_journal_socket_bound_when_present(profile, monkeypatch, tmp_path):
    # A login-shell profile may pipe startup through systemd-cat, which
    # needs /run/systemd/journal/stdout; bind it (when it exists) so the connect
    # succeeds instead of spraying "Failed to create stream fd" into every result.
    import socket as _socket
    from pathlib import Path

    from airc_tools import sandbox as sb

    sock_path = tmp_path / "journal-stdout"
    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    s.bind(str(sock_path))
    try:
        monkeypatch.setattr(sb, "_JOURNAL_STDOUT_SOCK", Path(sock_path))
        assert f"--ro-bind {sock_path} {sock_path}" in " ".join(profile.wrapper())
        # Absent socket -> no bind, no error.
        missing = tmp_path / "nope"
        monkeypatch.setattr(sb, "_JOURNAL_STDOUT_SOCK", missing)
        assert str(missing) not in " ".join(profile.wrapper())
    finally:
        s.close()


def test_missing_optional_binds_skipped(profile):
    dep, casefile = profile.binds[0].path, profile.binds[1].path
    shutil.rmtree(dep)
    shutil.rmtree(casefile)
    s = " ".join(profile.wrapper())
    assert str(dep) not in s
    assert str(casefile) not in s


def test_unreachable_optional_binds_skipped(profile, monkeypatch):
    # An optional bind source we cannot stat for a reason OTHER than absence --
    # the real case is a corp path on a credential-gated mount raising ENOKEY --
    # is skipped like a missing one. Path.exists() re-raises anything that is not
    # ENOENT/ENOTDIR/EBADF/ELOOP, so testing "missing" alone does not cover this:
    # the error propagated out of wrapper() and failed the whole job.
    import errno

    gated = profile.binds[0].path
    casefile = profile.binds[1].path
    real_stat = Path.stat

    def fake_stat(self, *a, **kw):
        if self == gated:
            raise OSError(errno.ENOKEY, "Required key not available", str(self))
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", fake_stat)
    s = " ".join(profile.wrapper())
    assert str(gated) not in s
    # The rest of the profile is unaffected: one unreachable bind is not fatal.
    assert f"--bind {profile.root} {profile.root}" in s
    assert f"--bind {casefile} {casefile}" in s


def test_unreachable_mandatory_bind_still_raises(profile, monkeypatch):
    # The inverse of the above, and the reason `optional` is a property of the
    # bind rather than of what it happens to point at: a pin is a guard, not a
    # convenience. Skipping an unstattable one would leave the path writable
    # under whatever it was pinned on top of -- exactly the hole it closes -- so
    # it must fail loudly rather than degrade.
    import errno

    guard = profile.root / "guarded"
    guard.touch()
    boxed = Sandbox(
        root=profile.root,
        binds=(Bind(guard, RO),),
        use_cgroup=False,
    )
    real_stat = Path.stat

    def fake_stat(self, *a, **kw):
        if self == guard:
            raise OSError(errno.ENOKEY, "Required key not available", str(self))
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", fake_stat)
    with pytest.raises(OSError):
        boxed.wrapper()


def test_missing_root_raises(profile):
    shutil.rmtree(profile.root)
    with pytest.raises(FileNotFoundError):
        profile.wrapper()


def test_cgroup_prefix(profile, monkeypatch):
    boxed = Sandbox(
        root=profile.root,
        memory_max="1G",
        cpu_quota="200%",
        tasks_max=64,
        use_cgroup=True,
    )
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/systemd-run")
    argv = boxed.wrapper()
    assert argv[:5] == ["systemd-run", "--user", "--scope", "-q", "--collect"]
    s = " ".join(argv[: argv.index("--")])
    assert "-p MemoryMax=1G" in s
    assert "-p CPUQuota=200%" in s
    assert "-p TasksMax=64" in s
    # Degrades to bwrap-only when systemd-run is absent.
    monkeypatch.setattr(shutil, "which", lambda _: None)
    assert boxed.wrapper()[0] == "bwrap"


needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None, reason="bubblewrap not installed"
)


@needs_bwrap
async def test_run_shell_sandboxed_end_to_end(profile):
    (profile.root / "marker").write_text("x")
    out = await run_shell("ls; echo home=$HOME; ls /tmp | wc -l", sandbox=profile)
    assert "marker" in out  # chdir'd into the root
    assert f"home={profile.env[0][1]}" in out  # clearenv + setenv applied
    assert "exit 0" in out


@needs_bwrap
async def test_run_shell_sandboxed_ro_enforced(profile):
    out = await run_shell(
        f"touch {profile.binds[0].path}/probe 2>&1; ls /home/*/.* 2>&1 | head -1",
        sandbox=profile,
    )
    assert "Read-only file system" in out


def test_ro_ancestor_of_root_is_hoisted_above_it(profile, tmp_path):
    # An RO bind that is an ANCESTOR of the rw root (an operator extra_ro, an
    # editable source root, a resolved launcher chain -- all layout-dependent and
    # none of them written with the root in mind) would remount the whole
    # worktree read-only if it landed after it. It is hoisted above the root
    # instead of being the caller's problem to order.
    casefile = profile.binds[1].path
    boxed = Sandbox(
        root=profile.root,
        binds=(Bind(tmp_path, RO), Bind(casefile, RW, optional=True)),
        use_cgroup=False,
    )
    argv = boxed.wrapper()
    ro_at = _dest_at(argv, "--ro-bind", str(tmp_path))
    root_at = _dest_at(argv, "--bind", str(profile.root))
    assert ro_at < root_at
    # ...and an rw bind written after it still lands after it: hoisting moves the
    # ancestor, not everything.
    assert _dest_at(argv, "--bind", str(casefile)) > ro_at


@needs_bwrap
async def test_root_stays_writable_under_ro_ancestor_end_to_end(profile, tmp_path):
    boxed = Sandbox(
        root=profile.root,
        binds=(Bind(tmp_path, RO),),  # ancestor of root
        tmpfs=(("/tmp", 1 << 20),),
        env=(("HOME", str(tmp_path)), *DEFANG_ENV.items()),
        use_cgroup=False,
    )
    out = await run_shell(
        f"touch probe && echo wrote; touch {tmp_path}/probe2 2>&1", sandbox=boxed
    )
    assert "wrote" in out  # the worktree root is rw despite the ro ancestor
    assert "Read-only file system" in out  # the ancestor itself stays ro


def _pinned_box(tmp_path, use_cgroup=False):
    """An rw dir with ro pins written after it, mirroring the main .git bound rw
    with config/hooks pinned on top."""
    root = tmp_path / "wt"
    gitdir = tmp_path / "gitdir"  # stands in for the common .git
    (gitdir / "hooks").mkdir(parents=True)
    (gitdir / "config").write_text("[core]\n")
    (gitdir / "refs").mkdir()
    root.mkdir()
    return gitdir, Sandbox(
        root=root,
        binds=(
            Bind(gitdir, RW),
            Bind(gitdir / "config", RO),
            Bind(gitdir / "hooks", RO),
        ),
        tmpfs=(("/tmp", 1 << 20),),
        env=(("HOME", str(tmp_path)), *DEFANG_ENV.items()),
        use_cgroup=use_cgroup,
    )


def test_a_pin_written_after_the_rw_parent_is_emitted_after_it(tmp_path):
    # The list order is the mount order, so a pin written after the rw parent
    # wins over it. Asserted against the emitted argv rather than against the
    # list that was passed in: the second would hold no matter what wrapper()
    # did with it.
    gitdir, boxed = _pinned_box(tmp_path)
    argv = boxed.wrapper()
    assert _dest_at(argv, "--ro-bind", str(gitdir / "config")) > _dest_at(
        argv, "--bind", str(gitdir)
    )


def test_a_pin_written_before_the_rw_parent_does_not_win(tmp_path):
    # The other half of "later wins", and the reason this model can be read as
    # policy: the SAME two binds in the other order give the other answer, and
    # nothing silently repairs it. If wrapper() sorted binds by mode -- the
    # field-shaped model this replaced -- the pin would survive its own
    # misplacement and the ordering would stop meaning anything.
    root = tmp_path / "wt"
    root.mkdir()
    gitdir = tmp_path / "gitdir"
    gitdir.mkdir()
    (gitdir / "config").write_text("[core]\n")
    boxed = Sandbox(
        root=root,
        binds=(Bind(gitdir / "config", RO), Bind(gitdir, RW)),
        use_cgroup=False,
    )
    argv = boxed.wrapper()
    assert _dest_at(argv, "--ro-bind", str(gitdir / "config")) < _dest_at(
        argv, "--bind", str(gitdir)
    )


@needs_bwrap
async def test_pin_over_an_rw_parent_is_read_only_end_to_end(tmp_path):
    gitdir, boxed = _pinned_box(tmp_path)
    out = await run_shell(
        f"touch {gitdir}/refs/probe && echo wrote_refs; "
        f"echo x >> {gitdir}/config 2>&1; "
        f"echo x > {gitdir}/hooks/post-commit 2>&1",
        sandbox=boxed,
    )
    assert "wrote_refs" in out  # the rw .git parent is writable
    assert out.count("Read-only file system") >= 2  # config and hooks both ro


def test_wrapper_refuses_a_missing_mandatory_bind_source(tmp_path):
    # A pin whose source does not exist must fail loud, not skip: the skipped
    # guard leaves the path writable under the rw parent (e.g. a plantable
    # .git/hooks), which is the hole it exists to close.
    gitdir, boxed = _pinned_box(tmp_path)
    (gitdir / "config").unlink()
    with pytest.raises(FileNotFoundError, match="config"):
        boxed.wrapper()


def _sealed_box(tmp_path, use_cgroup=False):
    """A sealed directory with one rw hole punched back through it, mirroring
    .git/worktrees with this job's own private worktree dir kept writable."""
    root = tmp_path / "wt"
    root.mkdir()
    gitdir = tmp_path / "gitdir"
    wts = gitdir / "worktrees"
    private = wts / "mine"
    private.mkdir(parents=True)
    # A name no path or message can collide with: the assertion below is
    # "absent", and an accidental substring hit would make it pass vacuously.
    (wts / "OTHERWT").mkdir()
    (wts / "OTHERWT" / "config.worktree").write_text("[core]\n")
    return (
        gitdir,
        private,
        Sandbox(
            root=root,
            binds=(Bind(gitdir, RW), Seal(wts), Bind(private, RW)),
            tmpfs=(("/tmp", 1 << 20),),
            env=(("HOME", str(tmp_path)),),
            use_cgroup=use_cgroup,
        ),
    )


def test_a_hole_through_a_seal_is_mounted_after_it(tmp_path):
    # Order IS the mechanism: emitted before the seal, the hole is buried by the
    # seal's empty tmpfs and the box gets a read-only private worktree dir, where
    # git then cannot create index.lock -- breaking every in-box commit.
    gitdir, private, boxed = _sealed_box(tmp_path)
    argv = boxed.wrapper()
    assert _dest_at(argv, "--bind", str(private)) > _dest_at(
        argv, "--tmpfs", str(gitdir / "worktrees")
    )


def test_a_seal_closes_only_after_every_later_mount(tmp_path):
    # The seal's ro remount is deferred to the END of the resolved list, not
    # emitted with its tmpfs. bwrap cannot create a mountpoint inside a
    # read-only tmpfs, so an atomic seal makes every hole through it die with
    # "Can't mkdir: Read-only file system" (measured against bwrap 0.11.2) --
    # which fails the box outright rather than leaking, but takes the .git dance
    # with it.
    gitdir, _, boxed = _sealed_box(tmp_path)
    ops = _ops(boxed)
    assert ops[-1] == ("seal-ro", str(gitdir / "worktrees"))
    argv = boxed.wrapper()
    binds = [i for i, a in enumerate(argv) if a in ("--ro-bind", "--bind", "--tmpfs")]
    assert _dest_at(argv, "--remount-ro", str(gitdir / "worktrees")) > max(binds)


@needs_bwrap
async def test_seal_hides_siblings_and_forbids_creation_end_to_end(tmp_path):
    # What a seal buys over pinning each entry: the sealed directory is EMPTY in
    # the box, so a sibling worktree is not read-only but absent, and nothing new
    # can be created beside it. Pins are a snapshot of what existed at assembly
    # time -- a job could otherwise plant a config.worktree in a sibling created
    # after the scan, and host-side git runs core.fsmonitor out of it.
    gitdir, private, boxed = _sealed_box(tmp_path)
    wts = gitdir / "worktrees"
    out = await run_shell(
        f"ls {wts}; mkdir {wts}/planted 2>&1; "
        f"touch {private}/index.lock && echo wrote_private",
        sandbox=boxed,
    )
    assert "OTHERWT" not in out  # not merely read-only: not there
    assert "mine" in out  # the hole is still visible
    assert "Read-only file system" in out  # and nothing new can appear
    assert "wrote_private" in out  # while the job's own dir is writable
    # None of it reached the host's copy.
    assert (wts / "OTHERWT" / "config.worktree").read_text() == "[core]\n"
    assert not (wts / "planted").exists()


@needs_bwrap
async def test_a_pin_inside_a_hole_holds_when_written_after_it(tmp_path):
    # The case that used to need a derived re-emission pass, and the best single
    # test of whether this model is more expressive than the fields it replaced:
    # pin, hole, pin again is just three binds in that order. The real .git shape
    # -- the private worktree dir must be writable for index.lock, while the
    # commondir and config.worktree inside it steer host-side git and must not be.
    root = tmp_path / "wt"
    root.mkdir()
    gitdir = tmp_path / "gitdir"
    wts = gitdir / "worktrees"
    private = wts / "mine"
    private.mkdir(parents=True)
    (private / "commondir").write_text("../..\n")
    (private / "config.worktree").write_text("")
    boxed = Sandbox(
        root=root,
        binds=(
            Bind(gitdir, RW),
            Seal(wts),
            Bind(private, RW),
            Bind(private / "commondir", RO),
            Bind(private / "config.worktree", RO),
        ),
        tmpfs=(("/tmp", 1 << 20),),
        env=(("HOME", str(tmp_path)), *DEFANG_ENV.items()),
        use_cgroup=False,
    )
    out = await run_shell(
        f"touch {private}/index.lock && echo wrote_lock; "
        f"echo x > {private}/config.worktree 2>&1; "
        f"echo x > {private}/commondir 2>&1",
        sandbox=boxed,
    )
    assert "wrote_lock" in out  # the hole still works
    assert out.count("Read-only file system") >= 2  # ...without un-pinning these


@needs_bwrap
async def test_a_pin_written_before_its_hole_is_re_opened_by_it(tmp_path):
    # The negative of the test above, and what makes it a claim about ordering
    # rather than about list membership: the SAME binds with the pins written
    # BEFORE the rw hole leave commondir writable, because the hole re-opens
    # everything under it. Nothing in the model repairs that, which is the point
    # -- the old field model papered over it with a second derived pass, so the
    # ordering the caller wrote and the ordering the box got were different
    # things.
    root = tmp_path / "wt"
    root.mkdir()
    gitdir = tmp_path / "gitdir"
    private = gitdir / "worktrees" / "mine"
    private.mkdir(parents=True)
    (private / "commondir").write_text("../..\n")
    boxed = Sandbox(
        root=root,
        binds=(
            Bind(gitdir, RW),
            Bind(private / "commondir", RO),  # pinned BEFORE the hole
            Bind(private, RW),
        ),
        tmpfs=(("/tmp", 1 << 20),),
        env=(("HOME", str(tmp_path)), *DEFANG_ENV.items()),
        use_cgroup=False,
    )
    out = await run_shell(
        f"echo x > {private}/commondir 2>&1 && echo REOPENED", sandbox=boxed
    )
    assert "REOPENED" in out
    assert "Read-only file system" not in out


def test_wrapper_refuses_a_missing_hole_source(tmp_path):
    # Same reasoning as a pin, opposite failure: a skipped rw hole leaves the
    # path read-only under its seal, which does not leak but silently breaks
    # in-box git. Fail at assembly instead.
    _, private, boxed = _sealed_box(tmp_path)
    private.rmdir()
    with pytest.raises(FileNotFoundError, match="mine"):
        boxed.wrapper()


def test_seal_source_must_exist(tmp_path):
    # bwrap would create the mountpoint on the HOST rather than fail (measured),
    # so a seal of a path that is not there quietly makes a directory in the
    # shared checkout as a side effect of building a profile. Refuse instead.
    root = tmp_path / "wt"
    root.mkdir()
    boxed = Sandbox(root=root, binds=(Seal(tmp_path / "gone"),), use_cgroup=False)
    with pytest.raises(FileNotFoundError, match="seal source missing"):
        boxed.wrapper()
    assert not (tmp_path / "gone").exists()


def test_ro_ancestor_of_tmpfs_phases_before_it(tmp_path):
    # The prod bug: a ro bind that is an ANCESTOR of a tmpfs mount must land
    # BEFORE the tmpfs, or it shadows the tmpfs read-only ($HOME/.cache, where
    # vpython takes its lock). /usr re-emitted by launcher interpreter-root
    # resolution after the $HOME tmpfs was the original failure.
    home = tmp_path / "home"  # stands in for $HOME, under the ro ancestor
    home.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    boxed = Sandbox(
        root=root,
        binds=(Bind(tmp_path, RO),),  # ro ancestor of the home tmpfs
        tmpfs=((str(home), 1 << 20),),
        env=(("HOME", str(home)),),
        use_cgroup=False,
    )
    argv = boxed.wrapper()
    ro_at = _dest_at(argv, "--ro-bind", str(tmp_path))
    assert ro_at < _dest_at(argv, "--tmpfs", str(home))


def test_descendant_of_a_tmpfs_is_not_hoisted(tmp_path):
    # The inverse, and the reason hoisting is a targeted exception rather than a
    # sort: depot_tools under $HOME is a DESCENDANT of the tmpfs and must land
    # AFTER it to stay visible on top of the blanked home. Hoisting it would make
    # the box's depot_tools vanish.
    home = tmp_path / "home"
    home.mkdir()
    depot = home / "depot_tools"
    depot.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    boxed = Sandbox(
        root=root,
        binds=(Bind(depot, RO),),
        tmpfs=((str(home), 1 << 20),),
        env=(("HOME", str(home)),),
        use_cgroup=False,
    )
    argv = boxed.wrapper()
    assert _dest_at(argv, "--ro-bind", str(depot)) > _dest_at(
        argv, "--tmpfs", str(home)
    )


@needs_bwrap
async def test_ro_ancestor_of_tmpfs_tmpfs_wins_end_to_end(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    boxed = Sandbox(
        root=root,
        binds=(Bind(tmp_path, RO),),  # ro ancestor of the home tmpfs
        tmpfs=((str(home), 1 << 20),),
        env=(("HOME", str(home)), ("PATH", "/usr/bin:/bin")),
        use_cgroup=False,
    )
    out = await run_shell(
        'mkdir -p "$HOME/.cache" && touch "$HOME/.cache/_p" && echo HOME_CACHE_WRITABLE;'
        f"touch {tmp_path}/probe 2>&1",
        sandbox=boxed,
    )
    assert "HOME_CACHE_WRITABLE" in out  # tmpfs won, not shadowed read-only
    assert "Read-only file system" in out  # the ro ancestor itself stays ro


def test_system_ro_root_is_dropped_not_reemitted(tmp_path):
    # /usr is already bound by _SYSTEM_ARGS; re-emitting it (as launcher
    # interpreter-root resolution did on prod) is waste, and after a tmpfs it
    # covers it is the leak. It is dropped -- exactly one /usr bind.
    root = tmp_path / "wt"
    root.mkdir()
    boxed = Sandbox(root=root, binds=(Bind(Path("/usr"), RO),), use_cgroup=False)
    argv = boxed.wrapper()
    usr_binds = sum(
        1 for i in range(len(argv)) if argv[i] == "--ro-bind" and argv[i + 1] == "/usr"
    )
    assert usr_binds == 1


def test_symlinked_alias_and_its_target_are_both_bound(tmp_path):
    # The prod shape: uv gives an unversioned interpreter root that symlinks to
    # a versioned one, and the venv names the alias. Both are real, distinct
    # destinations in the box, so deduping them to one leaves the other absent
    # and exec fails ENOENT. Keying dedup on the resolved path collapsed them.
    root = tmp_path / "wt"
    root.mkdir()
    versioned = tmp_path / "cpython-3.12.12"
    versioned.mkdir()
    alias = tmp_path / "cpython-3.12"
    alias.symlink_to(versioned)
    boxed = Sandbox(
        root=root,
        binds=(Bind(alias, RO), Bind(versioned, RO)),
        use_cgroup=False,
    )
    dests = {d for op, d in _ops(boxed) if op == "ro"}
    assert str(alias) in dests
    assert str(versioned) in dests


def test_a_repeated_bind_is_emitted_once(tmp_path):
    # Two callers naming the same dep is ordinary (a launcher chain and an
    # extra_ro often overlap), and re-emitting it is waste at best. Deduped only
    # while the earlier mount is still in effect -- see the test below for the
    # case where it is not.
    root = tmp_path / "wt"
    root.mkdir()
    dep = tmp_path / "dep"
    dep.mkdir()
    boxed = Sandbox(
        root=root,
        binds=(Bind(dep, RO), Bind(dep, RO)),
        use_cgroup=False,
    )
    assert [d for op, d in _ops(boxed) if d == str(dep)] == [str(dep)]


def test_a_pin_restated_after_a_hole_is_not_deduped_away(tmp_path):
    # The dedup must not eat a pin that is re-stated because something was
    # mounted over it in between -- that is precisely the .git shape, and
    # dropping the second one silently un-pins a host-code-exec vector. What
    # makes it correct is that a covering mount invalidates what it covers, so
    # "already in effect" stops being true.
    root = tmp_path / "wt"
    root.mkdir()
    gitdir = tmp_path / "gitdir"
    private = gitdir / "worktrees" / "mine"
    private.mkdir(parents=True)
    (private / "commondir").write_text("../..\n")
    boxed = Sandbox(
        root=root,
        binds=(
            Bind(gitdir, RW),
            Bind(private / "commondir", RO),
            Bind(private, RW),  # re-opens everything under it
            Bind(private / "commondir", RO),  # ...so this one must survive
        ),
        use_cgroup=False,
    )
    emitted = [d for op, d in _ops(boxed) if d == str(private / "commondir")]
    assert len(emitted) == 2


def test_assert_no_leak_raises_on_a_later_ancestor(tmp_path):
    # The guarantee: a bind list where a ro ancestor lands AFTER a tmpfs it
    # covers fails loud at assembly, rather than producing a silently-broken box.
    # Written as a BindOver because it is the one bind whose destination the
    # caller picks freely -- a same-path RO ancestor would be hoisted instead.
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    boxed = Sandbox(
        root=root,
        binds=(BindOver(src, tmp_path),),
        tmpfs=((str(home), 1 << 20),),
        use_cgroup=False,
    )
    with pytest.raises(ValueError, match="leaks"):
        boxed.wrapper()


def test_assert_no_leak_raises_on_a_later_bind_of_the_exact_path(tmp_path):
    # Equality, not just ancestry: a bind that remounts the tmpfs mount point
    # itself replaces the blanked scratch with the real directory underneath.
    # The guard leans on is_relative_to being true for equality, so a refactor
    # to a strict-ancestor test would open exactly this hole -- pin it.
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    boxed = Sandbox(
        root=root,
        binds=(BindOver(home, home),),
        tmpfs=((str(home), 1 << 20),),
        use_cgroup=False,
    )
    with pytest.raises(ValueError, match="leaks"):
        boxed.wrapper()
    # Same for the rw root: a later bind over it swaps the worktree for real
    # disk, which is writable and so hides the swap.
    with pytest.raises(ValueError, match="leaks"):
        dataclasses.replace(boxed, binds=(BindOver(home, root),)).wrapper()


def test_a_seal_does_not_count_as_covering_its_own_remount(tmp_path):
    # A seal's ro remount is deferred past every other mount, so it is the LAST
    # thing in the list -- but it modifies a mount already there rather than
    # covering the path anew. Treating it as a cover would make the leak check
    # fire on any seal of a directory under the rw root, i.e. on the .git dance
    # itself, and the natural "fix" for that is to weaken the check.
    root = tmp_path / "wt"
    root.mkdir()
    sealed = root / "worktrees"
    sealed.mkdir()
    Sandbox(root=root, binds=(Seal(sealed),), use_cgroup=False).wrapper()


def test_bind_over_places_a_file_at_a_different_path(tmp_path):
    """The one bind that is not src -> same path.

    Everything else here answers "let the box see this"; this answers "let the
    box see THIS where it expects THAT" -- a config override, which is how a
    per-job /etc/hosts or a build config reaches a checkout the job does not own.
    """
    src = tmp_path / "my-hosts"
    src.write_text("127.0.0.1 example\n")
    root = tmp_path / "wt"
    root.mkdir()
    argv = Sandbox(root=root, binds=(BindOver(src, Path("/etc/hosts")),)).wrapper()
    i = argv.index(str(src))
    assert argv[i - 1] == "--ro-bind"
    assert argv[i + 1] == "/etc/hosts"


def test_bind_over_written_last_lands_after_the_system_binds(tmp_path):
    """Order is the mechanism: bwrap mounts in sequence and a later mount wins.

    If this bind were emitted before `--ro-bind /etc /etc`, the system file
    would shadow the override and the box would silently read the wrong config
    -- the exact failure the bind exists to prevent, and an invisible one.
    """
    src = tmp_path / "my-hosts"
    src.write_text("x\n")
    root = tmp_path / "wt"
    root.mkdir()
    # Assert against the LAST bind of any kind, not just /etc: the requirement
    # is that nothing can be mounted after a mapped bind written last, and an
    # assertion naming one earlier mount passes even when the bind is emitted far
    # too early. (Measured: it did.)
    ro = tmp_path / "dep"
    ro.mkdir()
    argv = Sandbox(
        root=root,
        binds=(Bind(ro, RO), BindOver(src, Path("/etc/hosts"))),
    ).wrapper()
    binds = [i for i, a in enumerate(argv) if a in ("--ro-bind", "--bind", "--tmpfs")]
    assert argv.index(str(src)) > max(binds[:-1])


def test_bind_over_source_must_exist(tmp_path):
    # Skipping a missing source would leave the box reading the ORIGINAL file,
    # which is the misconfiguration this bind exists to prevent -- so it raises
    # rather than degrading, exactly as a pin does.
    root = tmp_path / "wt"
    root.mkdir()
    box = Sandbox(root=root, binds=(BindOver(tmp_path / "gone", Path("/etc/x")),))
    with pytest.raises(FileNotFoundError, match="bind-over source missing"):
        box.wrapper()


def test_bind_over_cannot_shadow_the_rw_root(tmp_path):
    """The new freedom is the SOURCE, not the destination.

    A mapped bind is still a mount, so it stays under the leak check: being able
    to choose where a file lands must not become a way to remount the rw root
    read-only, or to cover a tmpfs.
    """
    src = tmp_path / "f"
    src.write_text("x\n")
    root = tmp_path / "wt"
    root.mkdir()
    box = Sandbox(root=root, binds=(BindOver(src, root),))
    with pytest.raises(ValueError, match="leaks"):
        box.wrapper()


def test_unshare_net_is_off_by_default(profile):
    assert "--unshare-net" not in profile.wrapper()


def test_unshare_net_is_passed_when_set(profile):
    boxed = dataclasses.replace(profile, unshare_net=True)
    assert "--unshare-net" in boxed.wrapper()


@needs_bwrap
async def test_unshare_net_gives_the_box_its_own_loopback(profile):
    """The two properties the RBE/Vertex proxies are designed around.

    Asserted by running real processes, not by inspecting argv: what matters is
    not that the flag is present but that a host listener becomes unreachable
    while an in-box one still works. A relay that binds in-box depends on the
    second being true, and every host-side proxy port depends on the first being
    false -- so both are pinned here rather than left to bwrap's semantics.
    """
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    boxed = dataclasses.replace(profile, unshare_net=True)
    try:
        # The host's loopback is a different loopback now.
        out = await run_shell(
            f'python3 -c "'
            f"import socket;s=socket.socket();s.settimeout(3);"
            f"print('rc=%d' % s.connect_ex(('127.0.0.1',{port})))\"",
            sandbox=boxed,
        )
        # ECONNREFUSED: the port exists on the HOST's loopback, not this one.
        assert "rc=111" in out, out
        # But the box still HAS a loopback, so its own relay can bind and serve.
        out = await run_shell(
            'python3 -c "'
            "import socket,threading;"
            "srv=socket.socket();srv.bind(('127.0.0.1',0));srv.listen(1);"
            "threading.Thread(target=lambda: srv.accept().sendall(b'ok'),daemon=True).start();"
            "c=socket.socket();c.settimeout(3);c.connect(srv.getsockname());"
            'print(c.recv(8).decode())"',
            sandbox=boxed,
        )
        assert "ok" in out
    finally:
        srv.close()


@needs_bwrap
async def test_a_bound_unix_socket_crosses_the_network_namespace(profile, tmp_path):
    """The seam every host-side proxy reaches the isolated box through.

    A UNIX socket is a filesystem object, so a bind mount carries it across a
    boundary that no port survives. This is the whole reason proxy mode can keep
    working with the network taken away, so it is pinned by connecting to a real
    host listener from inside a real netns-isolated box.
    """
    sock_dir = tmp_path / "sock"
    sock_dir.mkdir()
    sock = sock_dir / "s.sock"
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(str(sock))
    srv.listen(1)

    def _serve():
        conn, _ = srv.accept()
        conn.sendall(b"HOST")
        conn.close()

    threading.Thread(target=_serve, daemon=True).start()
    boxed = dataclasses.replace(
        profile,
        unshare_net=True,
        binds=(*profile.binds, Bind(sock_dir, RO)),
    )
    try:
        out = await run_shell(
            f'python3 -c "'
            f"import socket;c=socket.socket(socket.AF_UNIX);c.settimeout(3);"
            f"c.connect('{sock}');print(c.recv(8).decode())\"",
            sandbox=boxed,
        )
        assert "HOST" in out
    finally:
        srv.close()


def test_overlay_source_must_exist(tmp_path):
    # Skipping it would leave the box with no cache where it expects a warm one.
    # For vpython that is not a degradation but a HANG (it tries to rebuild the
    # venv and, with no network, never finishes), so this fails at assembly.
    root = tmp_path / "wt"
    root.mkdir()
    box = Sandbox(root=root, binds=(Overlay(tmp_path / "gone"),))
    with pytest.raises(FileNotFoundError, match="tmp-overlay source missing"):
        box.wrapper()


def test_overlay_cannot_shadow_a_tmpfs(tmp_path):
    # An overlay covers its mount point exactly as a bind does, so it is subject
    # to the leak rule too -- one over $HOME would shadow the blanked home just
    # as silently as the ro bind that caused that bug.
    src = tmp_path / "cache"
    src.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    box = Sandbox(root=root, tmpfs=((str(src), 1 << 20),), binds=(Overlay(src),))
    with pytest.raises(ValueError, match="leaks"):
        box.wrapper()


@needs_bwrap
async def test_overlay_is_warm_to_read_and_writes_go_nowhere(profile, tmp_path):
    """The two halves that make a shared tool cache safe to expose.

    A cache the box must be able to WRITE to (vpython takes a lock file even to
    read) but must never actually CHANGE: it is 4 GB of interpreters that every
    `git cl` run executes, so a writable bind would let one box rewrite the
    python every other box -- and the host -- then runs.

    Both halves are asserted against a real box, because either alone is
    useless: a cache that cannot be written fails the lock, and one whose writes
    escape is the vulnerability.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "warm.txt").write_text("from the host\n")
    boxed = dataclasses.replace(profile, binds=(*profile.binds, Overlay(cache)))
    out = await run_shell(
        f"cat {cache}/warm.txt; echo poison > {cache}/evil.txt && echo WROTE",
        sandbox=boxed,
    )
    assert "from the host" in out  # warm: the host's contents are there
    assert "WROTE" in out  # writable: the box's own writes succeed
    # ...and none of it reached the host.
    assert not (cache / "evil.txt").exists()
    assert (cache / "warm.txt").read_text() == "from the host\n"

import shutil
from pathlib import Path

import pytest

from airc_tools.sandbox import Sandbox
from airc_tools.shell import run_shell


@pytest.fixture
def profile(tmp_path):
    root = tmp_path / "wt"
    casefile = tmp_path / "casefile"
    dep = tmp_path / "dep"
    creds = tmp_path / "creds"
    for d in (root, casefile, dep, creds):
        d.mkdir()
    return Sandbox(
        root=root,
        rw_paths=(casefile,),
        ro_paths=(dep,),
        opaque_ro_paths=(creds,),
        tmpfs=(("/tmp", 1 << 20),),
        env=(("HOME", str(tmp_path)),),
        memory_max="1G",
        tasks_max=64,
        use_cgroup=False,
    )


def test_ai_agent_reaches_both_shells(profile):
    # siso goes quiet on AI_AGENT (any non-empty value). _DEFANG_ENV is the single
    # source: it is merged over os.environ for the unsandboxed shell, and the
    # wrapper --setenv's it into the sandboxed one after --clearenv.
    from airc_tools.shell import _DEFANG_ENV

    assert _DEFANG_ENV.get("AI_AGENT")
    assert "--setenv AI_AGENT" in " ".join(profile.wrapper())


def test_wrapper_shape(profile):
    argv = profile.wrapper()
    # bwrap-only (cgroup off); command slots append after the prefix.
    assert argv[0] == "bwrap"
    s = " ".join(argv)
    assert f"--bind {profile.root} {profile.root}" in s
    assert f"--ro-bind {profile.ro_paths[0]} {profile.ro_paths[0]}" in s
    # Opaque paths are mounted ro like deps; the difference is check() below.
    assert f"--ro-bind {profile.opaque_ro_paths[0]}" in s
    assert f"--bind {profile.rw_paths[0]} {profile.rw_paths[0]}" in s
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
    shutil.rmtree(profile.ro_paths[0])
    shutil.rmtree(profile.opaque_ro_paths[0])
    s = " ".join(profile.wrapper())
    assert str(profile.ro_paths[0]) not in s
    assert str(profile.opaque_ro_paths[0]) not in s


def test_unreachable_optional_binds_skipped(profile, monkeypatch):
    # An optional bind source we cannot stat for a reason OTHER than absence --
    # the real case is a corp path on a credential-gated mount raising ENOKEY --
    # is skipped like a missing one. Path.exists() re-raises anything that is not
    # ENOENT/ENOTDIR/EBADF/ELOOP, so testing "missing" alone does not cover this:
    # the error propagated out of wrapper() and failed the whole job.
    import errno

    gated = profile.ro_paths[0]
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
    assert f"--ro-bind {profile.opaque_ro_paths[0]}" in s


def test_unreachable_ro_over_bind_still_raises(profile, monkeypatch):
    # The inverse of the above, and deliberate: a ro-over source is a guard, not
    # a convenience. Skipping an unstattable one would leave the path creatable
    # under its rw parent -- exactly the hole the field closes -- so it must fail
    # loudly rather than degrade.
    import errno

    guard = profile.root / "guarded"
    guard.touch()
    boxed = Sandbox(root=profile.root, ro_over_rw_paths=(guard,), use_cgroup=False)
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


def test_check_containment(profile, tmp_path):
    inside = profile.root / "a.cc"
    case_doc = profile.rw_paths[0] / "DRAFT.md"
    dep_file = profile.ro_paths[0] / "header.h"
    secret = profile.opaque_ro_paths[0] / "tokens.json"
    outside = tmp_path / "elsewhere.txt"

    assert profile.check(inside, write=True) is None
    assert profile.check(case_doc, write=True) is None
    assert profile.check(dep_file, write=False) is None
    # Deps are read-only; credentials and stray paths are refused both ways.
    assert "outside" in profile.check(dep_file, write=True)
    assert "outside" in profile.check(secret, write=False)
    assert "outside" in profile.check(outside, write=False)
    assert "outside" in profile.check("/home", write=False)


def test_check_resolves_symlink_escape(profile, tmp_path):
    target = tmp_path / "secret.txt"
    target.write_text("s")
    link = profile.root / "innocent.txt"
    link.symlink_to(target)
    assert "outside" in profile.check(link, write=False)


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
        f"touch {profile.ro_paths[0]}/probe 2>&1; ls /home/*/.* 2>&1 | head -1",
        sandbox=profile,
    )
    assert "Read-only file system" in out


def test_ro_ancestor_of_root_cannot_shadow_the_worktree(profile, tmp_path):
    # bwrap mounts in argv order and later mounts shadow earlier ones, so every
    # ro bind must precede the rw root: an ro path that is an ANCESTOR of the
    # root (operator extra_ro, editable source root, launcher chain -- all
    # layout-dependent) would otherwise remount the whole worktree read-only.
    boxed = Sandbox(
        root=profile.root,
        rw_paths=profile.rw_paths,
        ro_paths=(tmp_path,),  # ancestor of root
        use_cgroup=False,
    )
    argv = boxed.wrapper()
    ro_at = argv.index("--ro-bind")
    root_at = argv.index("--bind")
    assert ro_at < root_at
    # and the rw seams still land after the ro binds (rw-over-ro layering)
    assert argv.index(str(profile.rw_paths[0])) > ro_at


@needs_bwrap
async def test_root_stays_writable_under_ro_ancestor_end_to_end(profile, tmp_path):
    boxed = Sandbox(
        root=profile.root,
        rw_paths=(),
        ro_paths=(tmp_path,),  # ancestor of root
        tmpfs=(("/tmp", 1 << 20),),
        env=(("HOME", str(tmp_path)),),
        use_cgroup=False,
    )
    out = await run_shell(
        f"touch probe && echo wrote; touch {tmp_path}/probe2 2>&1", sandbox=boxed
    )
    assert "wrote" in out  # the worktree root is rw despite the ro ancestor
    assert "Read-only file system" in out  # the ancestor itself stays ro


def _ro_over_box(tmp_path, use_cgroup=False):
    """A profile with an rw dir that has a ro-over hole punched in it, mirroring
    the main .git bound rw with config/hooks pinned ro."""
    root = tmp_path / "wt"
    gitdir = tmp_path / "gitdir"  # stands in for the common .git
    (gitdir / "hooks").mkdir(parents=True)
    (gitdir / "config").write_text("[core]\n")
    (gitdir / "refs").mkdir()
    root.mkdir()
    return gitdir, Sandbox(
        root=root,
        rw_paths=(gitdir,),
        ro_over_rw_paths=(gitdir / "config", gitdir / "hooks"),
        tmpfs=(("/tmp", 1 << 20),),
        env=(("HOME", str(tmp_path)),),
        use_cgroup=use_cgroup,
    )


def test_ro_over_binds_land_after_the_rw_parent(tmp_path):
    # ro-over paths must be bound AFTER the rw parent so they win on overlap;
    # the fixed ro-first/rw-last layering would otherwise let the rw .git bind
    # shadow them and leave config/hooks writable.
    gitdir, boxed = _ro_over_box(tmp_path)
    argv = boxed.wrapper()
    gitdir_rw = next(
        i
        for i in range(len(argv))
        if argv[i] == "--bind" and argv[i + 1] == str(gitdir)
    )
    config_ro = next(
        i
        for i in range(len(argv))
        if argv[i] == "--ro-bind" and argv[i + 1] == str(gitdir / "config")
    )
    assert config_ro > gitdir_rw  # ro-over shadows the rw parent


def test_check_refuses_write_to_ro_over_but_allows_read_and_rw_siblings(tmp_path):
    gitdir, boxed = _ro_over_box(tmp_path)
    # refs under the rw .git are writable (repo state); config/hooks are not,
    # but stay readable.
    assert boxed.check(gitdir / "refs" / "heads" / "x", write=True) is None
    assert "read-only" in boxed.check(gitdir / "config", write=True)
    assert "read-only" in boxed.check(gitdir / "hooks" / "post-commit", write=True)
    assert boxed.check(gitdir / "config", write=False) is None


@needs_bwrap
async def test_ro_over_hole_is_read_only_end_to_end(tmp_path):
    gitdir, boxed = _ro_over_box(tmp_path)
    out = await run_shell(
        f"touch {gitdir}/refs/probe && echo wrote_refs; "
        f"echo x >> {gitdir}/config 2>&1; "
        f"echo x > {gitdir}/hooks/post-commit 2>&1",
        sandbox=boxed,
    )
    assert "wrote_refs" in out  # the rw .git parent is writable
    assert out.count("Read-only file system") >= 2  # config and hooks both ro


def test_wrapper_refuses_missing_ro_over_source(tmp_path):
    # A ro-over source that does not exist must fail loud, not skip: the
    # skipped guard leaves the path creatable under the rw parent (e.g. a
    # plantable .git/hooks), which is the hole the field exists to close.
    gitdir, boxed = _ro_over_box(tmp_path)
    (gitdir / "config").unlink()
    with pytest.raises(FileNotFoundError, match="config"):
        boxed.wrapper()

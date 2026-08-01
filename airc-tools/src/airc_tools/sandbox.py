# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Sandbox: per-call confinement for the shell tool, plus the matching path
policy for read/edit.

The mechanism is tier 0 of the icompleteu sandbox design: with an in-process
agent harness the model's whole capability surface is its tools, so wrapping
tool execution confines the agent without a process boundary. Each shell call
runs inside a fresh bubblewrap mount namespace (the job worktree rw at its real
absolute path, declared dependencies ro, everything else -- $HOME, other
worktrees, the main checkout -- absent, not merely denied), under a systemd
transient scope for cgroup limits. read/edit tools run in the trusted process,
so they get the same boundary as a realpath check against the declared mounts
(`check`): without it, injected content could exfiltrate ~/.ssh into a casefile
document that flows out to chat, no network needed.

Known give-ups of this tier, documented rather than papered over:
- Network stays the host's (no --unshare-net): the build needs RBE egress and
  the credential-helper socket + netns allowlist is a follow-up step.
- Build credentials bound via `opaque_ro_paths` (the luci token cache) are
  readable by in-sandbox code even though the read tool refuses them; the
  token broker replaces the bind later.
- No disk quota on the rw worktree (project quotas need root); only the tmpfs
  mounts are size-capped.
- /tmp is a fresh tmpfs per call: scratch there does not survive to the next
  shell call. Durable scratch belongs in the worktree or the casefile.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .shell import _DEFANG_ENV

# The fixed system surface every sandboxed command sees. /usr and /etc ro; the
# usual merged-usr symlinks recreated instead of bound so nothing else from /
# leaks in. No /opt, no /srv, no /home beyond the tmpfs + explicit binds.
_SYSTEM_ARGS = (
    "--ro-bind", "/usr", "/usr",
    "--symlink", "usr/bin", "/bin",
    "--symlink", "usr/lib", "/lib",
    "--symlink", "usr/lib64", "/lib64",
    "--symlink", "usr/sbin", "/sbin",
    "--ro-bind", "/etc", "/etc",
    "--proc", "/proc",
    "--dev", "/dev",
)  # fmt: skip

# The host journal's stdout stream socket. `bash -lc` sources /etc/profile.d,
# and on a systemd host an /etc/profile.d snippet may pipe shell startup through
# `systemd-cat`, which connects here; /run is unbound, so absent this bind the
# connect fails and prefixes every command with two benign "Failed to create
# stream fd" lines that are then re-fed to the model on each shell result.
_JOURNAL_STDOUT_SOCK = Path("/run/systemd/journal/stdout")

_ISOLATION_ARGS = (
    "--unshare-pid",
    "--unshare-ipc",
    "--unshare-uts",
    "--cap-drop",
    "ALL",
    "--die-with-parent",
)


def _bindable(p: Path) -> bool:
    """Whether an OPTIONAL bind source can be mounted, i.e. whether we can stat
    it at all -- not merely whether it exists.

    `Path.exists()` is not enough: it swallows ENOENT/ENOTDIR/EBADF/ELOOP and
    re-raises everything else, so a path we cannot reach for any other reason
    propagates an OSError out of wrapper() instead of being skipped. The case
    that bit us is a credential-gated network mount (a corp /google path with an
    expired ticket) raising ENOKEY: the caller had already decided that bind was
    optional, but sandbox assembly blew up and took the whole job with it.
    Unreachable and absent are the same thing for an optional bind -- bwrap
    cannot mount either -- so treat any stat failure as "skip".

    Only for optional sources. `ro_over_rw_paths` must keep raising: skipping one
    silently drops a guard while its rw parent stays writable.
    """
    try:
        p.stat()
    except OSError:
        return False
    return True


# The ro roots _SYSTEM_ARGS already binds (/usr, /etc). A ro/opaque path that
# resolves to one of these is already mounted (and mounted early, in the safe
# order), so emitting it again is pure waste -- and, when it lands after a tmpfs
# it covers (the prod bug: a launcher-resolved interpreter root of /usr re-emitted
# after the $HOME tmpfs), a leak. Parsed from _SYSTEM_ARGS so it cannot drift.
_SYSTEM_RO_ROOTS = frozenset(
    Path(_SYSTEM_ARGS[i + 2])
    for i in range(len(_SYSTEM_ARGS) - 2)
    if _SYSTEM_ARGS[i] == "--ro-bind"
)


def _strict_ancestor(a: Path, b: Path) -> bool:
    """Is `a` a strict ancestor of `b` (a covers b, a != b)? Both resolved, so a
    symlinked home still compares against its real target."""
    try:
        ra, rb = a.resolve(), b.resolve()
    except OSError:
        return False
    return ra != rb and rb.is_relative_to(ra)


@dataclass(frozen=True)
class Sandbox:
    """One job's confinement profile; `wrapper()` yields the argv prefix that
    runs a command inside it, `check()` is the same boundary for the in-process
    read/edit tools. Immutable so a profile can be shared across a job's calls.
    """

    # The rw root (the job worktree), bound at its real absolute path because
    # remote-exec resolves build inputs by absolute path. Also the cwd.
    root: Path
    rw_paths: tuple[Path, ...] = ()  # casefile + the main-.git write seams
    ro_paths: tuple[Path, ...] = ()  # dep symlink targets, depot_tools, ...
    # Bound ro for in-sandbox consumers (siso reading the token cache) but
    # refused by check(): the read tool must not copy credentials into casefile
    # documents that leave the machine. In-sandbox shell code can still read
    # them -- that is the documented give-up until the credential broker.
    opaque_ro_paths: tuple[Path, ...] = ()
    # Read-only holes punched into an otherwise-rw tree: bound AFTER the rw root
    # and seams so they win on overlap. The main checkout's .git is bound rw (so
    # in-worktree git can create packed-refs.lock; a ro .git root makes every
    # commit print a harmless-but-misleading "Read-only file system") with
    # config and hooks/ listed here -- a poisoned shared checkout is a
    # host-code-exec vector, writable refs/objects are only repo state.
    ro_over_rw_paths: tuple[Path, ...] = ()
    # (mount point, size in bytes). Mounted before the binds so a bind under a
    # tmpfs (the worktree under the blanked $HOME) lands on top of it.
    tmpfs: tuple[tuple[str, int], ...] = ()
    # The complete environment inside the sandbox (--clearenv first, so the
    # daemon's env -- credentials included -- never leaks through).
    env: tuple[tuple[str, str], ...] = ()
    # cgroup limits, applied via `systemd-run --user --scope` when available.
    # Empty string / 0 skips that property.
    memory_max: str = ""
    cpu_quota: str = ""
    tasks_max: int = 0
    use_cgroup: bool = True
    # systemd slice to place the scope under (`--slice=`). A stable cgroup anchor
    # so a host firewall rule can match every sandbox scope (e.g. drop the GCE
    # metadata IP from this slice's cgroup); empty leaves systemd's default.
    slice_unit: str = ""

    _resolved: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def wrapper(self) -> list[str]:
        """The argv prefix: [systemd-run ...] bwrap ... -- ready to have the
        actual command appended. Unreachable optional bind sources are skipped
        (bwrap errors out on a nonexistent source); a missing root or ro-over
        source raises -- skipping a ro-over guard would silently leave the
        path creatable under its rw parent."""
        if not self.root.is_dir():
            raise FileNotFoundError(f"sandbox root missing: {self.root}")
        argv = list(self._cgroup_args())
        argv += ["bwrap", *_SYSTEM_ARGS, *self._resolv_conf_args()]
        argv += self._journal_socket_args()
        # ro/opaque binds split into two phases around the tmpfs. bwrap mounts in
        # argv order and a later mount shadows an earlier one on overlap, so a ro
        # bind that is an ANCESTOR of a tmpfs mount -- the prod bug, where a
        # launcher-resolved interpreter root of /usr was re-emitted after the
        # $HOME tmpfs -- shadows the tmpfs and turns its scratch ($HOME/.cache,
        # where vpython takes its lock) read-only. Such ancestor binds go FIRST,
        # before the tmpfs, so the tmpfs wins at its own path; the rest stay
        # after the tmpfs so descendant binds (deps under $HOME) land on top of
        # the blanked home and stay visible. Paths already bound by _SYSTEM_ARGS
        # (/usr, /etc) are dropped: already bound, and bound early.
        early, late = self._split_ro()
        for p in early:
            argv += ["--ro-bind", str(p), str(p)]
        for mnt, size in self.tmpfs:
            argv += ["--size", str(size), "--tmpfs", mnt]
        # Mount order is the precedence order (bwrap: later mounts shadow
        # earlier ones where they overlap), so ro binds go FIRST and the rw
        # root and seams last. An ro path that happens to be an ancestor of
        # the worktree (an operator extra_ro, an editable source root, a
        # resolved launcher chain -- all layout-dependent) must never turn
        # the worktree read-only; conversely the rw .git seams must land on
        # top of the ro main-.git bind. The give-up is the inverse layering:
        # an ro bind INSIDE the worktree is shadowed by the root bind --
        # nothing uses that.
        for p in late:
            argv += ["--ro-bind", str(p), str(p)]
        argv += ["--bind", str(self.root), str(self.root)]
        for p in self.rw_paths:
            if _bindable(p):
                argv += ["--bind", str(p), str(p)]
        # ro-over binds land last so they shadow an overlapping rw parent (see
        # the field doc): the main .git is rw for packed-refs.lock, config and
        # hooks/ stay ro on top of it. Unlike the optional binds above, a
        # missing source here is not skipped: the guard would vanish while the
        # rw parent keeps the path creatable -- exactly the hole this field
        # exists to close. The caller must ensure the targets exist.
        for p in self.ro_over_rw_paths:
            if not p.exists():
                raise FileNotFoundError(f"ro-over bind source missing: {p}")
            argv += ["--ro-bind", str(p), str(p)]
        # Guarantee: no later mount may cover (be an ancestor-or-equal of) an
        # earlier tmpfs or the rw root. A violation is a silent sandbox leak --
        # the home-tmpfs shadow above if the phasing ever regresses, or a
        # worktree turned read-only (or replaced with real disk) by a later
        # ancestor bind. Fail the profile at assembly rather than hand the agent
        # a box that is not what it claims to be.
        self._assert_no_leak(argv)
        argv += _ISOLATION_ARGS
        argv += ["--clearenv"]
        for k, v in (*_DEFANG_ENV.items(), *self.env):
            argv += ["--setenv", k, v]
        argv += ["--chdir", str(self.root)]
        return argv

    def _split_ro(self) -> tuple[list[Path], list[Path]]:
        """Partition ro+opaque binds into early (before the tmpfs) and late.

        Early: a strict ancestor of any tmpfs mount -- it must precede that
        tmpfs or shadow it read-only (the home-tmpfs leak). Late: the rest,
        bound after the tmpfs so descendant binds (deps under $HOME) win on top
        of the blanked home. Drops anything already bound by _SYSTEM_ARGS
        (/usr, /etc): already mounted, and mounted in the safe early order.

        Two keys, deliberately: membership in _SYSTEM_RO_ROOTS asks "is this
        /usr under any name", which only the resolved path answers, while
        dedup asks "have I already emitted this mount destination", which is
        the literal one. Keying dedup on the resolved path drops aliases that
        are distinct destinations in the box -- and the interpreter roots are
        exactly that shape (a versioned dir plus the unversioned symlink the
        venv names), where binding only one leaves the other missing and every
        exec fails with ENOENT."""
        tmpfs_mounts = [Path(mnt) for mnt, _ in self.tmpfs]
        early: list[Path] = []
        late: list[Path] = []
        seen: set[Path] = set()
        for p in (*self.ro_paths, *self.opaque_ro_paths):
            if not _bindable(p):
                continue
            rp = p.resolve()
            if rp in _SYSTEM_RO_ROOTS or p in seen:
                continue
            seen.add(p)
            if any(_strict_ancestor(rp, t) for t in tmpfs_mounts):
                early.append(p)
            else:
                late.append(p)
        return early, late

    def _assert_no_leak(self, argv: list[str]) -> None:
        """No later mount may cover (be an ancestor-or-equal of) an earlier
        tmpfs or the rw root. Parses the mount specs wrapper() just emitted and
        raises naming the offending pair. A violation is a silent sandbox leak:
        the home-tmpfs shadow if phasing regresses, or the worktree turned
        read-only / swapped for real disk by a later ancestor bind."""
        root = self.root.resolve()
        mounts: list[tuple[int, Path]] = []
        protected: list[tuple[int, Path]] = []
        i = 0
        while i < len(argv):
            a = argv[i]
            if a == "--tmpfs" and i + 1 < len(argv):
                d = Path(argv[i + 1]).resolve()
                mounts.append((i, d))
                protected.append((i, d))
                i += 2
            elif a in ("--ro-bind", "--bind") and i + 2 < len(argv):
                d = Path(argv[i + 2]).resolve()
                mounts.append((i, d))
                if a == "--bind" and d == root:
                    protected.append((i, d))
                i += 3
            else:
                i += 1
        for pi, pp in protected:
            for mi, mp in mounts:
                # is_relative_to is True for equality too: a later bind that
                # remounts the exact path is just as much a shadow.
                if mi > pi and pp.is_relative_to(mp):
                    raise ValueError(
                        f"sandbox mount order leaks: {mp} (index {mi}) covers"
                        f" protected {pp} (index {pi}) -- a later bind shadows"
                        f" a tmpfs or the rw root"
                    )

    def _cgroup_args(self) -> list[str]:
        # systemd-run needs the user manager; when it is absent (a bare chroot,
        # a container) degrade to bwrap-only rather than failing every call.
        if not self.use_cgroup or shutil.which("systemd-run") is None:
            return []
        args = ["systemd-run", "--user", "--scope", "-q", "--collect"]
        if self.slice_unit:
            args += [f"--slice={self.slice_unit}"]
        if self.memory_max:
            args += ["-p", f"MemoryMax={self.memory_max}"]
        if self.cpu_quota:
            args += ["-p", f"CPUQuota={self.cpu_quota}"]
        if self.tasks_max:
            args += ["-p", f"TasksMax={self.tasks_max}"]
        return [*args, "--"]

    @staticmethod
    def _resolv_conf_args() -> list[str]:
        # systemd-resolved hosts symlink /etc/resolv.conf into /run, which is
        # not bound; bind the real file so DNS keeps working under host net.
        target = Path("/etc/resolv.conf").resolve()
        if target.exists() and not str(target).startswith("/etc/"):
            return ["--ro-bind", str(target), str(target)]
        return []

    @staticmethod
    def _journal_socket_args() -> list[str]:
        # Bind the host journal socket (when present) so the login shell's
        # systemd-cat connect succeeds instead of spraying "Failed to create
        # stream fd" into every result. Keeping `bash -lc` preserves the
        # corp-set build PATH; this just silences its one sandbox casualty.
        if _JOURNAL_STDOUT_SOCK.is_socket():
            return ["--ro-bind", str(_JOURNAL_STDOUT_SOCK), str(_JOURNAL_STDOUT_SOCK)]
        return []

    def check(self, path: str | Path, *, write: bool) -> str | None:
        """Containment for the in-process read/edit tools: None when `path` is
        inside the profile, else a refusal message (the tool's return value).
        Resolves symlinks first, so a link pointing out of the worktree cannot
        smuggle a read/write past the boundary."""
        p = Path(path).resolve()  # non-strict: also resolves not-yet-created files
        # ro-over paths (.git config/hooks) sit under an rw parent, so a plain
        # roots check would call them writable; refuse the write explicitly.
        if write and any(p.is_relative_to(r) for r in self._ro_over_roots()):
            return (
                f"error: {path} is read-only in this job's sandbox "
                f"(not writable: "
                f"{', '.join(str(r) for r in self.ro_over_rw_paths)})"
            )
        allowed = self._roots(write)
        if any(p.is_relative_to(r) for r in allowed):
            return None
        mode = "write" if write else "read"
        return (
            f"error: {path} is outside this job's sandbox ({mode} allowed under: "
            f"{', '.join(str(r) for r in allowed)})"
        )

    def _roots(self, write: bool) -> tuple[Path, ...]:
        # Resolve the policy roots once (the worktree path itself may contain
        # symlinks) and cache; the profile is frozen so this cannot go stale.
        key = "w" if write else "r"
        if key not in self._resolved:
            roots = (self.root, *self.rw_paths)
            if not write:
                roots += self.ro_paths
            self._resolved[key] = tuple(r.resolve() for r in roots)
        return self._resolved[key]

    def _ro_over_roots(self) -> tuple[Path, ...]:
        if "o" not in self._resolved:
            self._resolved["o"] = tuple(r.resolve() for r in self.ro_over_rw_paths)
        return self._resolved["o"]

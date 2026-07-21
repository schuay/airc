"""Path resolution shared by read_file and edit_file.

The server is long-lived and shared across worktrees, so no fixed root is baked
in. Relative paths resolve against AIRC_TOOLS_ROOT when set (the single-worktree
launch: the agent works one tree and passes bare paths), else the process cwd.
Callers that span trees pass absolute paths, which are used as-is.
"""

import functools
import os
from pathlib import Path


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    root = os.environ.get("AIRC_TOOLS_ROOT")
    return Path(root) / p if root else p.resolve()


def io_guarded(fn):
    """Turn an OSError from the underlying filesystem into the tool's own
    error-string contract instead of letting it escape.

    The per-tool guards catch the file's *shape* (dir, FIFO, too big), but the
    read/write itself can still fail for reasons the path alone does not show --
    a read-only mount, a permission bit, a symlink into a mirror the worktree
    only mounts ro. Uncaught, that raises out through the tool node and kills the
    whole agent turn; caught, it is one more message the model can act on (pick a
    different path, stop trying to edit a vendored dep). Every file tool takes
    `path` first, so the wrapper can name it."""

    @functools.wraps(fn)
    def wrapper(path: str, *args, **kwargs):
        try:
            return fn(path, *args, **kwargs)
        except OSError as e:
            return f"error: cannot access {path}: {e.strerror or e}"

    return wrapper

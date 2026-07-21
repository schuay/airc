"""Pure-seam coverage that needs no live model: the worktree-bound tools, the
Report -> AgentResult flattening, and the skill-index renderer."""

from deepagent import Disposition, Report, Sandbox, render_skill_index, to_result
from deepagent.langgraph_harness import _abs, _worktree_tools


class _DemoReport(Report):
    built: bool | None = None
    note: str = ""


def test_worktree_tools_names(tmp_path):
    tools = {t.name: t for t in _worktree_tools(tmp_path, shell_timeout_s=5.0)}
    assert set(tools) == {"shell", "read_file", "edit_file", "write_file"}


def test_bound_write_file_resolves_relative(tmp_path):
    tools = {t.name: t for t in _worktree_tools(tmp_path, shell_timeout_s=10.0)}
    tools["write_file"].invoke({"path": "sub/t.js", "content": "let x = 1;\n"})
    assert (tmp_path / "sub" / "t.js").read_text() == "let x = 1;\n"


async def test_bound_shell_runs_in_worktree(tmp_path):
    (tmp_path / "marker").write_text("x")
    tools = {t.name: t for t in _worktree_tools(tmp_path, shell_timeout_s=10.0)}
    out = await tools["shell"].ainvoke({"command": "ls"})
    assert "marker" in out


def test_bound_edit_and_read_resolve_relative(tmp_path):
    tools = {t.name: t for t in _worktree_tools(tmp_path, shell_timeout_s=10.0)}
    tools["edit_file"].invoke(
        {"path": "sub/new.py", "edits": [{"search": "", "replace": "x = 1\n"}]}
    )
    assert (tmp_path / "sub" / "new.py").read_text() == "x = 1\n"
    out = tools["read_file"].invoke({"path": "sub/new.py"})
    assert "x = 1" in out


def test_sandboxed_read_edit_contained(tmp_path):
    wt = tmp_path / "wt"
    case = tmp_path / "case"
    wt.mkdir()
    case.mkdir()
    (case / "JOB.md").write_text("job")
    secret = tmp_path / "secret.txt"
    secret.write_text("s3cret")
    box = Sandbox(root=wt, rw_paths=(case,), use_cgroup=False)
    tools = {t.name: t for t in _worktree_tools(wt, 10.0, sandbox=box)}

    # Inside the boundary: worktree-relative and casefile-absolute both work.
    tools["edit_file"].invoke(
        {"path": "a.py", "edits": [{"search": "", "replace": "x = 1\n"}]}
    )
    assert "x = 1" in tools["read_file"].invoke({"path": "a.py"})
    assert "job" in tools["read_file"].invoke({"path": str(case / "JOB.md")})

    # Outside: refused by the policy, and never touched.
    out = tools["read_file"].invoke({"path": str(secret)})
    assert "outside" in out and "s3cret" not in out
    out = tools["edit_file"].invoke(
        {"path": str(secret), "edits": [{"search": "s3cret", "replace": "gone"}]}
    )
    assert "outside" in out and secret.read_text() == "s3cret"
    out = tools["write_file"].invoke({"path": str(secret), "content": "gone"})
    assert "outside" in out and secret.read_text() == "s3cret"

    # A symlink inside the worktree cannot smuggle a read outside it.
    (wt / "link.txt").symlink_to(secret)
    assert "outside" in tools["read_file"].invoke({"path": "link.txt"})


def test_abs(tmp_path):
    assert _abs(tmp_path, "/etc/hosts") == "/etc/hosts"
    assert _abs(tmp_path, "rel/f") == str(tmp_path / "rel" / "f")


def test_to_result_flattens_extra_fields():
    res = to_result(
        _DemoReport(
            disposition=Disposition.COMPLETE,
            summary="s",
            friction="build was broken",
            built=True,
            note="hi",
        )
    )
    assert res.disposition is Disposition.COMPLETE
    assert res.summary == "s"
    assert res.friction == "build was broken"
    assert res.data == {"built": True, "note": "hi"}
    # summary/reason/friction are first-class, never in the stage data bag.
    assert "disposition" not in res.data and "friction" not in res.data


def test_render_skill_index(tmp_path):
    sk = tmp_path / "skills"
    sk.mkdir()
    (sk / "build.md").write_text(
        "---\nname: build\ndescription: how to build d8\n---\nlong body...\n"
    )
    (sk / "debug.md").write_text("---\nname: debug\ndescription: gdb tips\n---\nbody")
    idx = render_skill_index(sk, read_hint="Read via repo_git_show(...).")
    assert "build (build.md) -- how to build d8" in idx
    assert "debug (debug.md) -- gdb tips" in idx
    assert "Read via repo_git_show" in idx


def test_render_skill_index_absent_dir(tmp_path):
    assert render_skill_index(tmp_path / "nope") == ""

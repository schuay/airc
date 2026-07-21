"""edit_file: apply structured SEARCH/REPLACE edits to a single file.

One file per call, N chunks in `edits`, all-or-nothing. Edits apply to an
in-memory copy in order (a later edit sees an earlier one's change), and the
file is written only if every chunk matches. On any miss nothing is written and
a failure message names the failed block(s) with a did-you-mean hint -- the
message is the model's next prompt.

All-or-nothing, rather than aider's partial apply, because our chunks are small:
resending the fixed array is cheap, and it never leaves the file half-edited
under searches the model computed against the original. Cross-file edits are
separate calls (the model emits several in one turn); a single typed `path` per
call is what kills wrong-file application.
"""

from .editblock import find_similar_lines, replace_most_similar_chunk
from .limits import MAX_EDIT_FILE_BYTES, MAX_FAILURE_TEXT, clip
from .paths import io_guarded, resolve_path


def _ends_nl(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


@io_guarded
def write_file(path: str, content: str) -> str:
    """Create or overwrite `path` with the full `content`.

    The whole-file counterpart to edit_file: use it when you have the entire
    file (a new test, a scratch script), edit_file when changing part of an
    existing one. Shares path resolution and the size ceiling with edit_file;
    it does not go through the SEARCH/REPLACE engine because there is nothing to
    match -- and edit_file's empty-search path appends rather than overwrites,
    which is the wrong shape for "write this file".
    """
    p = resolve_path(path)
    if p.is_dir():
        return f"error: {path} is a directory"
    if p.exists() and not p.is_file():
        # Writing a FIFO/socket/device blocks the worker thread indefinitely.
        return f"error: {path} is not a regular file"
    size = len(content.encode())
    if size > MAX_EDIT_FILE_BYTES:
        return (
            f"error: content is {size} bytes, over the {MAX_EDIT_FILE_BYTES}"
            " byte limit; write a smaller file"
        )
    existed = p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"{'overwrote' if existed else 'created'} {path} ({size} bytes)"


@io_guarded
def apply_edits(path: str, edits: list[tuple[str, str]]) -> str:
    p = resolve_path(path)
    if p.is_dir():
        return f"error: {path} is a directory"
    if p.exists() and not p.is_file():
        # Reading a FIFO/socket/device blocks the worker thread forever
        # (uncancellably -- it runs via to_thread); a few of those wedge the
        # whole tool server.
        return f"error: {path} is not a regular file"
    if not edits:
        return f"error: no edits given for {path}"

    creating = not p.exists()
    if creating:
        # A non-existent file can only be created, not searched. Require the
        # explicit create shape so a typo'd path is a clear error, not a
        # surprise new file half-full of one edit's replace text.
        if any(search.strip() for search, _ in edits):
            return (
                f"error: {path} does not exist. To create it, send a single edit"
                " with an empty search and the full file content as replace."
            )
        p.parent.mkdir(parents=True, exist_ok=True)
        content = ""
    else:
        size = p.stat().st_size
        if size > MAX_EDIT_FILE_BYTES:
            return (
                f"error: {path} is {size} bytes, over the {MAX_EDIT_FILE_BYTES}"
                " byte edit limit; edit a smaller file or split the change"
            )
        content = p.read_text()

    # prep() in the matcher forces a trailing newline on the working buffer;
    # remember whether the file actually had one so we don't add diff noise.
    had_trailing_nl = content.endswith("\n") or content == ""

    working = content
    failures = []
    for idx, (search, replace) in enumerate(edits):
        if not search.strip():
            # Empty search: append to the file (or fill a freshly created one).
            working = (_ends_nl(working) if working else working) + replace
            continue
        res = replace_most_similar_chunk(working, search, replace)
        if res is None:
            failures.append((idx, search, replace))
        else:
            working = res

    if failures:
        # Hints resolve against the original content: since we write nothing,
        # that is exactly the file the model will see if it re-reads.
        return _failure_text(path, failures, content)

    if not had_trailing_nl and working.endswith("\n"):
        working = working[:-1]

    p.write_text(working)
    n = len(edits)
    return f"applied {n} edit{'s' if n != 1 else ''} to {path}"


def _failure_text(path, failures, content) -> str:
    n = len(failures)
    parts = [
        f"{n} SEARCH block{'s' if n != 1 else ''} failed to match in {path}."
        " No changes were written.\n"
    ]
    for idx, search, replace in failures:
        parts.append(
            f"\n## edit[{idx}] failed to match:\n"
            "<<<<<<< SEARCH\n"
            f"{_ends_nl(clip(search))}"
            "=======\n"
            f"{_ends_nl(clip(replace))}"
            ">>>>>>> REPLACE\n"
        )
        hint = find_similar_lines(search, content)
        if hint:
            parts.append(
                f"\nClosest lines already in {path}:\n```\n{clip(hint)}\n```\n"
            )
        if replace and replace in content:
            parts.append(
                f"\nNote: the REPLACE text for edit[{idx}] is already present in"
                f" {path}; this edit may be unnecessary.\n"
            )
    parts.append(
        "\nThe SEARCH text must match the file exactly -- whitespace, comments,"
        " everything. Fix the failed block(s) and resend all edits for this file.\n"
    )
    text = "".join(parts)
    if len(text) > MAX_FAILURE_TEXT:
        text = text[:MAX_FAILURE_TEXT] + "\n[... failure detail truncated ...]\n"
    return text

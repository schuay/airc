# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""airc-prune: age out the content of old threads, then reclaim the bytes.

What a retention policy is about is message content, and this room stores it
twice: verbatim in `messages.text` (airc.db) and again, per persona, inside the
LangGraph checkpoint blobs (airc.ckpt.db, which is the bulk of the volume). A
sweep that skips the checkpoints has not scrubbed anything.

Two retention classes: content in shared SPACES keeps a long window (default
18 months), content in DMs a short one (default 30 days). A thread earns the
space window only on a positive signal -- it was watcher-announced (SYSTEM
message, commit_threads, announcement_meta: announcements only ever post to
spaces) or it is linked to a chat thread recorded as is_dm=0 -- and an is_dm=1
link vetoes regardless. Everything else (unknown links, unlinked local threads)
falls to the short window: misclassification can only scrub early, never
retain DM content past its window.

The shape is REDACT, not delete. Blanking text keeps the rows, and with them the
five dedup keys that stop a real-world event being announced twice
(`commit_threads`, `chat_threads`, `chat_seen_messages`, `handover_jobs`,
`delivered_results`). That
matters concretely: an icompleteu CL job polls CQ for hours and can sit awaiting
review for days, so its result can arrive for a thread this sweep already
scrubbed. With the keys intact that result routes into the existing thread; with
the rows deleted it would create a new thread and post about a commit whose
discussion was just purged -- the opposite of the intent. `--hard` is available
for a policy that demands the stronger "the row is gone" story, and is not the
default for that reason.

SYSTEM messages survive: the commit digest and the perf summary. They are
rendered from public git data, and keeping them is what makes a scrubbed thread
intelligible in an audit. The line is drawn on kind alone because that is a rule
an operator can state and a reviewer can check -- notably it also catches review
findings, which post under kind AGENT (so a "keep the automated posts" rule would
have had to discriminate on sender to find them).

Not run in-process. A coroutine inside the room cannot vacuum files its own
connections hold open, nor delete checkpoints from under a live saver mid-turn;
it is a oneshot unit that stops the room, sweeps, and starts it again.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .config import CONFIG_DIR, load_config

# Every kind except this one is redacted. A str, not MessageKind, because the
# column stores the bare value and the pruner deliberately does not import the
# store (see _connect).
_RETAINED_KIND = "system"

# Thread-keyed tables with no content worth an audit trail and no dedup role:
# a persona's context generation, the Chat headline annotations and their
# per-finding dedup rows, an unresolved "thinking..." placeholder, a pending
# timer, and every plugin's own thread state. All are live-thread working state,
# meaningless once a thread is aged out, and all are safe to lose (the
# headline's Chat message is not edited again once nothing posts to the thread;
# a timer for a month-old thread has no one to wake; plugin_state holds
# working state for a conversation whose content is being redacted anyway).
# Ordered parent-last so a future FK cannot trip.
#
# plugin_state deliberately does NOT feed live_threads: its payload is opaque to
# core, so core cannot tell an open record from a spent one, and treating any row
# as liveness would make a thread with one stale row permanently unprunable.
_DROP_TABLES = (
    "context_generation",
    "chat_headline_findings",
    "chat_headlines",
    "chat_pending",
    "timers",
    "plugin_state",
)

_DURATION_RE = re.compile(r"^(\d+)([dwh])$")
_UNIT_SECONDS = {"h": 3600, "d": 86400, "w": 604800}


def parse_duration(text: str) -> float:
    """A retention window ("30d", "6w", "48h") in seconds.

    Deliberately a small closed set of units: a bare number would be ambiguous
    between seconds and days, and getting that wrong silently purges everything.
    """
    m = _DURATION_RE.match(text.strip())
    if not m:
        raise argparse.ArgumentTypeError(
            f"invalid duration {text!r}: want <n>h, <n>d or <n>w (e.g. 30d)"
        )
    return int(m.group(1)) * _UNIT_SECONDS[m.group(2)]


@dataclass
class Counts:
    """What one sweep touched, for the report and the audit note."""

    threads: int = 0
    messages_redacted: int = 0
    messages_deleted: int = 0
    titles_blanked: int = 0
    checkpoints: int = 0
    writes: int = 0
    rows_dropped: int = 0
    bugs_unlinked: int = 0

    def summary(self) -> str:
        verb = "deleted" if self.messages_deleted else "redacted"
        n = self.messages_deleted or self.messages_redacted
        return (
            f"{self.threads} thread(s), {n} message(s) {verb},"
            f" {self.titles_blanked} title(s) blanked,"
            f" {self.checkpoints} checkpoint(s) + {self.writes} write(s) dropped,"
            f" {self.rows_dropped} side row(s), {self.bugs_unlinked} bug(s) unlinked"
        )


def _connect(path: Path) -> sqlite3.Connection:
    """Open a database directly, NOT through Store.

    Store.__init__ runs the schema script plus _migrate() on connect, so opening
    through it would migrate a file this tool is about to vacuum, and it exposes
    no bulk-redaction API anyway. The one thing we do need from the schema (the
    thread_seen_floor table) is created here if absent, so the pruner works
    against a database written by a room that predates it.
    """
    db = sqlite3.connect(str(path))
    db.execute("PRAGMA busy_timeout=5000")
    return db


def check_writable(db: sqlite3.Connection, path: Path) -> str | None:
    """None if this database can be written exclusively, else why not.

    The sweep's whole premise is that the room is stopped: it deletes
    checkpoints, then redacts, then vacuums. Without this check a still-running
    room fails the run HALFWAY -- observed: the checkpoint delete and vacuum
    succeeded, then airc.db raised "database is locked", so the personas lost
    their context and not one byte of content was scrubbed. Loud, but the worst
    possible split.

    BEGIN IMMEDIATE takes the write lock without writing anything, which is
    exactly the question being asked, and the rollback leaves no trace. Checked
    on BOTH files before either is touched, so a locked airc.db is discovered
    while the checkpoints are still intact.
    """
    try:
        db.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as e:
        return f"{path} is locked by another process ({e})"
    db.execute("ROLLBACK")
    return None


def existing_tables(db: sqlite3.Connection) -> set[str]:
    """The tables this database actually has.

    Not a paranoia check: NOT opening through Store means the schema is never
    created or migrated, so a store written by an older room genuinely lacks
    tables this code names. The live store at the time of writing had no
    `timers`, `pending_bugs`, `chat_headline_findings`, or `thread_seen_floor` --
    and an unguarded query against one aborts the sweep (or, mid-transaction,
    rolls back a redaction that already reported its counts). Every table below
    threads/messages is therefore consulted through this set.
    """
    return {
        r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def _has_column(db: sqlite3.Connection, table: str, column: str) -> bool:
    """Whether the table has the column. Needed for the same reason as
    existing_tables: this tool never migrates, so a store written by an older
    room can have chat_threads without is_dm, and an unguarded reference
    aborts the sweep."""
    return column in {r[1] for r in db.execute(f"PRAGMA table_info({table})")}


def space_class_threads(db: sqlite3.Connection) -> set[int]:
    """Thread ids entitled to the space retention window.

    Positive signals only, because the two windows differ by an order of
    magnitude and the failure directions are not symmetric: granting the space
    window to a DM thread over-retains against policy, while denying it to a
    space thread merely scrubs early. So a thread qualifies on evidence it is
    space content -- it was announced (SYSTEM message, commit_threads,
    announcement_meta: announcements only ever post to spaces), or a transport
    recorded a link with is_dm=0 -- and any is_dm=1 link vetoes, so a stray
    announcement signal on a DM thread cannot extend it. Unknown links (NULL:
    pre-migration rows, transports that never say) grant nothing.
    """
    have = existing_tables(db)
    space = {
        r[0]
        for r in db.execute(
            "SELECT DISTINCT thread_id FROM messages WHERE kind = ?",
            (_RETAINED_KIND,),
        )
    }
    for t in ("commit_threads", "announcement_meta"):
        if t in have:
            space |= {r[0] for r in db.execute(f"SELECT thread_id FROM {t}")}
    if "chat_threads" in have and _has_column(db, "chat_threads", "is_dm"):
        space |= {
            r[0]
            for r in db.execute("SELECT thread_id FROM chat_threads WHERE is_dm = 0")
        }
        space -= {
            r[0]
            for r in db.execute("SELECT thread_id FROM chat_threads WHERE is_dm = 1")
        }
    return space


def aged_threads(
    db: sqlite3.Connection, cutoff: float, space_cutoff: float | None = None
) -> list[int]:
    """Thread ids aged past their class window and not already fully scrubbed.

    cutoff is the short (DM/default) window; space_cutoff, when given, is the
    long window applied to space_class_threads. None means one window for
    everything (the pre-split behavior, kept for callers that do not classify).

    The content check is what keeps a weekly timer cheap and its report honest:
    a thread whose non-SYSTEM messages are all blank has nothing left to redact,
    so re-reporting it every week would make the counts meaningless. The floor
    row is not the marker -- a thread can have one and still hold new messages
    posted since -- so the check is on the content itself.
    """
    space = space_class_threads(db) if space_cutoff is not None else set()
    rows = db.execute(
        "SELECT t.id, t.created FROM threads t"
        " WHERE EXISTS (SELECT 1 FROM messages m WHERE m.thread_id = t.id"
        "   AND m.kind != ? AND (m.text != '' OR m.sender != ''))"
        " ORDER BY t.id",
        (_RETAINED_KIND,),
    ).fetchall()
    return [
        tid
        for tid, created in rows
        if created < (space_cutoff if tid in space else cutoff)
    ]


def live_threads(db: sqlite3.Connection, control_root: Path | None) -> set[int]:
    """Threads with work still in flight, which the sweep skips.

    Age alone is not enough: a thread can be old and still be actively worked.
    Three signals, in increasing order of coupling:

    - a queued `pending_bugs` row (a bug awaiting a tracker retry, whose "filed"
      follow-up posts back into the thread),
    - a pending `timers` row (an agent asked to be woken there),
    - an unterminated icompleteu job, read from `control_root/<job>/state.json`.

    The third reaches into another component's on-disk layout, which is why it is
    best-effort and gated on control_root being passed: a missing or unreadable
    file skips the check rather than failing the sweep. That is acceptable because
    the exclusion is not what makes a late result safe -- redaction is, by keeping
    commit_threads resolving so the result still routes to its thread. This only
    avoids gutting the context of a thread someone is mid-way through.
    """
    # Queried separately and only when present: an older store lacks these
    # tables entirely (see existing_tables), and a single UNION would abort on
    # the first missing one -- taking the whole sweep with it.
    have = existing_tables(db)
    live: set[int] = set()
    if "pending_bugs" in have:
        live |= {
            r[0]
            for r in db.execute(
                "SELECT thread_id FROM pending_bugs WHERE thread_id IS NOT NULL"
            )
        }
    if "timers" in have:
        live |= {r[0] for r in db.execute("SELECT thread_id FROM timers")}
    if control_root is None:
        return live
    live |= _icompleteu_live(control_root)
    return live


def _icompleteu_live(control_root: Path) -> set[int]:
    """Thread ids of unterminated icompleteu jobs, best-effort.

    Reads the job state files directly rather than importing icompleteu: core
    must not depend on a plugin's component, and this tool has to run on a deploy
    where icompleteu is not installed at all. The cost is that the two terminal
    step names below duplicate icompleteu's `TERMINAL`; being wrong here only
    over- or under-skips a thread, never corrupts one.
    """
    import json

    terminal = {"ready-to-land", "blocked", "done", "abandoned", "error"}
    live: set[int] = set()
    try:
        state_files = sorted(control_root.glob("*/state.json"))
    except OSError as e:
        print(f"warning: cannot scan {control_root}: {e}", file=sys.stderr)
        return live
    for path in state_files:
        try:
            data = json.loads(path.read_bytes())
        except (OSError, ValueError) as e:
            # A half-written or corrupt state file must not decide retention
            # either way; say so and move on.
            print(f"warning: skipping {path}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, dict) or data.get("step") in terminal:
            continue
        tid = (data.get("job") or {}).get("thread_id")
        if isinstance(tid, int):
            live.add(tid)
    return live


def delete_checkpoints(
    ckpt: sqlite3.Connection, thread_ids: list[int]
) -> tuple[int, int]:
    """Drop every persona checkpoint for these threads.

    The LangGraph key is "<thread>:<persona>:g<gen>", so a prefix match covers
    every persona and every context generation of the thread. The blobs are
    msgpack under a custom serializer and cannot be surgically rewritten -- a
    wholesale delete per thread is the only safe operation, and it needs no
    finer grain: every announcement worth keeping lives in airc.db and is
    untouched by this.

    A persona then resumes the thread with empty context rather than a cold
    cache -- it never re-reads that history in any form, because the seen offset
    (in airc.db, floored by the sweep) keeps it from being re-injected. What is
    actually lost is the persona's own tool results, all re-derivable.
    """
    # Both tables are created by the saver on first use, so a checkpoint file
    # from a room that never completed a turn can be missing one. Skipping an
    # absent table is right; aborting here would take the redaction with it.
    have = existing_tables(ckpt)
    n_ckpt = n_writes = 0
    for tid in thread_ids:
        # No LIKE-escaping needed: the key is "<int>:<persona>[:g<gen>]", so the
        # prefix is digits and a colon -- no `_` or `%` can appear in it. The
        # colon is what makes this exact: '1:%' cannot reach thread 12.
        like = f"{tid}:%"
        if "checkpoints" in have:
            n_ckpt += ckpt.execute(
                "DELETE FROM checkpoints WHERE thread_id LIKE ?", (like,)
            ).rowcount
        if "writes" in have:
            n_writes += ckpt.execute(
                "DELETE FROM writes WHERE thread_id LIKE ?", (like,)
            ).rowcount
    ckpt.commit()
    return n_ckpt, n_writes


def redact_threads(
    db: sqlite3.Connection, thread_ids: list[int], *, hard: bool = False
) -> Counts:
    """Scrub the content of these threads in airc.db, in one transaction.

    Atomicity matters for one specific pairing: the seen floor must land with the
    redaction. A committed redaction with no floor leaves personas pointed at
    history that is now blank, which surfaces later as a wall of empty transcript
    lines in a turn -- long after this reported success.
    """
    c = Counts(threads=len(thread_ids))
    # An older store lacks some of these tables outright, and inside a
    # transaction a "no such table" is the worst case: it rolls the redaction
    # back after the run has already printed its counts. Resolve what exists
    # once, before opening the transaction.
    have = existing_tables(db)
    orchestrated = "orchestrated" in have
    # "Is this thread a watcher announcement rather than a human's?" -- OR'd over
    # whichever signals this store actually has (see the title UPDATE below).
    # A SYSTEM message needs no table check: messages is always present.
    tests = [
        f"EXISTS (SELECT 1 FROM messages WHERE thread_id = ? AND kind = '{_RETAINED_KIND}')"
    ]
    tests += [
        f"EXISTS (SELECT 1 FROM {t} WHERE thread_id = ?)"
        for t in ("commit_threads", "announcement_meta")
        if t in have
    ]
    announced_sql = " OR ".join(tests)
    # One binding per EXISTS test, plus the UPDATE's own `WHERE id = ?`. Every
    # placeholder in that statement takes the same thread id.
    announced_params = len(tests) + 1
    with db:  # one transaction; rolls back on any exception
        for tid in thread_ids:
            tip = db.execute(
                "SELECT COALESCE(MAX(id), 0) FROM messages WHERE thread_id = ?", (tid,)
            ).fetchone()[0]
            if hard:
                c.messages_deleted += db.execute(
                    "DELETE FROM messages WHERE thread_id = ? AND kind != ?",
                    (tid, _RETAINED_KIND),
                ).rowcount
            else:
                c.messages_redacted += db.execute(
                    "UPDATE messages SET text = '', sender = ''"
                    " WHERE thread_id = ? AND kind != ?"
                    "   AND (text != '' OR sender != '')",
                    (tid, _RETAINED_KIND),
                ).rowcount
            # An announced thread's title is the commit subject: public git data,
            # and what keeps a scrubbed row intelligible. Only a human-started
            # thread's title may be user-authored, so blank just those.
            #
            # "Announced" is tested three ways because no single one is reliable
            # across the store's history. commit_threads is the intended signal
            # but was added late -- it is EMPTY on a store whose 342 threads are
            # almost all commit threads, which would have blanked every subject.
            # announcement_meta covers more of them, and a SYSTEM message is the
            # announcement itself, so it covers any thread a watcher posted to.
            # A human thread has none of the three.
            c.titles_blanked += db.execute(
                "UPDATE threads SET title = '' WHERE id = ? AND title != ''"
                f" AND NOT ({announced_sql})",
                (tid,) * announced_params,
            ).rowcount
            # Pin both offsets to the thread's own tip, in this transaction.
            # The floor covers every persona including those with no agent_seen
            # row (which would read 0 and replay everything); `orchestrated`
            # is pinned for the same one-line cost rather than relying on a
            # restart's _recover to repair it, which a running daemon never does.
            db.execute(
                "INSERT INTO thread_seen_floor (thread_id, last_msg_id) VALUES (?, ?)"
                " ON CONFLICT(thread_id) DO UPDATE SET"
                " last_msg_id = MAX(last_msg_id, excluded.last_msg_id)",
                (tid, tip),
            )
            if orchestrated:
                db.execute(
                    "INSERT INTO orchestrated (thread_id, last_msg_id) VALUES (?, ?)"
                    " ON CONFLICT(thread_id) DO UPDATE SET"
                    " last_msg_id = MAX(last_msg_id, excluded.last_msg_id)",
                    (tid, tip),
                )
            for table in _DROP_TABLES:
                if table not in have:
                    continue
                c.rows_dropped += db.execute(
                    f"DELETE FROM {table} WHERE thread_id = ?", (tid,)
                ).rowcount
            # The bug still wants filing; it just loses its follow-up
            # destination. Null the link rather than dropping the row.
            if "pending_bugs" in have:
                c.bugs_unlinked += db.execute(
                    "UPDATE pending_bugs SET thread_id = NULL WHERE thread_id = ?",
                    (tid,),
                ).rowcount
    return c


def vacuum(db: sqlite3.Connection) -> None:
    """Reclaim the freed pages. Without this the content is still on disk in the
    freelist, and "the bytes are still there" is the wrong answer to a policy
    question -- which is why a failure here fails the whole run rather than being
    reported as a partial success. Truncates the WAL first so its frames are not
    left holding the old pages either.
    """
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.execute("VACUUM")


def _ensure_floor_table(db: sqlite3.Connection) -> None:
    """Create thread_seen_floor if the room that wrote this DB predates it. The
    sweep writes the floor, so it cannot wait for the room's own migration to run
    -- that happens on the next start, after this tool has already redacted."""
    db.execute(
        "CREATE TABLE IF NOT EXISTS thread_seen_floor ("
        " thread_id INTEGER PRIMARY KEY, last_msg_id INTEGER NOT NULL)"
    )
    db.commit()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="airc-prune",
        description="Age out the content of old threads and reclaim the bytes.",
    )
    p.add_argument("-c", "--config", type=Path, help="suite config (airc.toml)")
    p.add_argument("--db", type=Path, help="override the store path")
    p.add_argument(
        "--older-than",
        type=parse_duration,
        default=parse_duration("30d"),
        metavar="DURATION",
        # The floor is set by icompleteu, not policy: a CL job polls CQ for hours
        # and can await review for days, so a window under a week would purge
        # threads with work still live in them.
        help="retention window: <n>h, <n>d or <n>w (default 30d)",
    )
    p.add_argument(
        "--space-older-than",
        type=parse_duration,
        default=parse_duration("540d"),
        metavar="DURATION",
        # Defaulted long rather than to --older-than: the asymmetric footgun is
        # a hand-run without the flag wiping space threads 17 months early
        # (irreversible), not a deploy without the split retaining them (idle
        # bytes). A single-window sweep is --space-older-than equal to
        # --older-than, stated explicitly.
        help="retention window for space-class threads (default 540d, ~18 months)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be swept, change nothing",
    )
    p.add_argument(
        "--hard",
        action="store_true",
        help="delete message rows instead of blanking them (loses replay safety)",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirmation prompt (for the systemd timer, which has no stdin)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    db_path = args.db or cfg.db_path
    # Derived, not configured: the room derives the checkpoint path from db_path
    # the same way (runner.py), and there is no ckpt_path config key. Deriving it
    # identically is what keeps `--db` from vacuuming the wrong second file.
    ckpt_path = db_path.with_suffix(".ckpt.db")
    if not db_path.exists():
        print(f"no store at {db_path}", file=sys.stderr)
        return 1

    now = time.time()
    cutoff = now - args.older_than
    space_cutoff = now - args.space_older_than
    db = _connect(db_path)
    try:
        aged = aged_threads(db, cutoff, space_cutoff)
        live = live_threads(db, _control_root(args.config))
        targets = [t for t in aged if t not in live]
        skipped = len(aged) - len(targets)
        n_space = len(set(targets) & space_class_threads(db))

        days = args.older_than / 86400
        space_days = args.space_older_than / 86400
        print(f"store:      {db_path}")
        print(f"checkpoints:{ckpt_path}")
        print(
            f"window:     {days:.1f} day(s) DM/default, {space_days:.0f} day(s) space"
        )
        print(
            f"threads:    {len(targets)} to sweep"
            f" ({n_space} space-class, {len(targets) - n_space} dm/default),"
            f" {skipped} skipped (work in flight)"
        )
        if not targets:
            print("nothing to do")
            return 0
        print(_preview(db, targets))
        if args.dry_run:
            print("dry run: nothing changed")
            return 0

        # Preflight BOTH files before touching either. The sweep assumes the room
        # is stopped; if it is not, failing here costs nothing, whereas failing
        # partway leaves the checkpoints deleted and the content unscrubbed --
        # the one outcome worse than not running at all.
        problems = [p for p in (check_writable(db, db_path),) if p]
        if ckpt_path.exists():
            probe = _connect(ckpt_path)
            try:
                if p := check_writable(probe, ckpt_path):
                    problems.append(p)
            finally:
                probe.close()
        if problems:
            for p in problems:
                print(f"error: {p}", file=sys.stderr)
            print(
                "refusing to sweep: stop airc.service first"
                " (the unit's ExecStartPre does this)",
                file=sys.stderr,
            )
            return 1

        if not args.yes and not _confirm():
            print("aborted")
            return 1

        # Checkpoints FIRST, deliberately. A crash between the two files leaves
        # orphaned checkpoints, which are harmless (nothing reads a checkpoint
        # without a live thread row) and a re-run clears them. The reverse order
        # would leave live threads with amputated persona state.
        n_ckpt = n_writes = 0
        if ckpt_path.exists():
            ckpt = _connect(ckpt_path)
            try:
                n_ckpt, n_writes = delete_checkpoints(ckpt, targets)
                vacuum(ckpt)
            finally:
                ckpt.close()
        else:
            print(f"warning: no checkpoint db at {ckpt_path}", file=sys.stderr)

        _ensure_floor_table(db)
        counts = redact_threads(db, targets, hard=args.hard)
        counts.checkpoints, counts.writes = n_ckpt, n_writes
        # Audit trail: thread ids and counts, never content. After the redaction,
        # so a failed sweep leaves no note claiming it happened.
        if "source_notes" in existing_tables(db):
            db.execute(
                "INSERT INTO source_notes (source, ts, note) VALUES (?, ?, ?)",
                (
                    "prune",
                    time.time(),
                    (
                        f"swept {counts.summary()}"
                        f" (window {days:.0f}d/{space_days:.0f}d,"
                        f" threads {_compact_ids(targets)})"
                    ),
                ),
            )
            db.commit()
        vacuum(db)
        print(f"swept: {counts.summary()}")
        return 0
    finally:
        db.close()


def _control_root(config_path: Path | None) -> Path | None:
    """icompleteu's control root from the shared suite config, or None.

    Read straight out of the TOML rather than via Config or icompleteu's own
    parser: `[icompleteu]` is a top-level suite section that core does not
    interpret (Config keeps only the `[airc]` leftovers in plugin_config), and
    core must not import a plugin's component -- this tool has to run on a deploy
    where icompleteu is not installed at all. Any read failure yields None, which
    only downgrades the best-effort live-job check.
    """
    import tomllib

    path = config_path or CONFIG_DIR / "airc.toml"
    try:
        raw = tomllib.loads(path.read_text())
    except (OSError, ValueError):
        return None
    section = raw.get("icompleteu")
    root = section.get("control_root") if isinstance(section, dict) else None
    return Path(str(root)).expanduser() if root else None


def _preview(db: sqlite3.Connection, thread_ids: list[int]) -> str:
    """Per-kind message counts across the target threads, so an operator sees
    what is about to go and what survives before confirming."""
    marks = ",".join("?" * len(thread_ids))
    rows = db.execute(
        f"SELECT kind, COUNT(*) FROM messages WHERE thread_id IN ({marks})"
        " GROUP BY kind ORDER BY kind",
        thread_ids,
    ).fetchall()
    lines = ["messages by kind:"]
    for kind, n in rows:
        fate = "RETAINED" if kind == _RETAINED_KIND else "scrubbed"
        lines.append(f"  {kind:<8} {n:>6}  {fate}")
    return "\n".join(lines)


def _compact_ids(ids: list[int], limit: int = 20) -> str:
    """Thread ids for the audit note, truncated so one huge first sweep does not
    write a multi-KB row."""
    head = ",".join(str(i) for i in ids[:limit])
    return head if len(ids) <= limit else f"{head},... (+{len(ids) - limit})"


def _confirm() -> bool:
    """Interactive gate. This is irreversible and the room is stopped around it,
    so the default is to ask; the timer passes --yes because it has no stdin."""
    try:
        return input("proceed? this cannot be undone [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())

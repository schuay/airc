# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""A directory-backed channel with maildir-style claim semantics.

Layout under the channel root:
    tmp/          messages being written (never observed mid-write)
    incoming/     published, awaiting a consumer
    in-progress/  claimed by a consumer
    done/         completed (kept for audit)
    failed/       poison / failed handling (kept for inspection)

Publish is atomic (write to tmp, rename into incoming). A consumer claims a
message by renaming it from incoming to in-progress; the rename is the lock, so
exactly one of several racing consumers wins. Files are named "<ulid>.json", so
incoming sorts by time and the oldest is claimed first.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .envelope import Envelope

log = logging.getLogger(__name__)

_SUBDIRS = ("tmp", "incoming", "in-progress", "done", "failed")


class Channel:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        for d in _SUBDIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def publish(self, env: Envelope) -> str:
        name = f"{env.message_id}.json"
        tmp = self.root / "tmp" / name
        tmp.write_bytes(env.to_bytes())
        os.rename(tmp, self.root / "incoming" / name)  # atomic within the fs
        return env.message_id

    def pending(self) -> list[str]:
        return sorted(
            p.name for p in (self.root / "incoming").iterdir() if p.suffix == ".json"
        )

    def claim(self) -> Claim | None:
        for name in self.pending():  # ulid-sorted: oldest first
            src = self.root / "incoming" / name
            dst = self.root / "in-progress" / name
            try:
                os.rename(src, dst)  # the lock: the loser of a race raises
            except OSError:
                continue
            return Claim(self, dst, Envelope.from_bytes(dst.read_bytes()))
        return None

    def adopt(self, env: Envelope) -> Claim:
        """Publish a message straight into in-progress, skipping incoming.

        For seeding a job a consumer should *resume* (via in_progress()) rather
        than claim fresh: the message is never offered to a normal claim(), so it
        cannot race a running consumer. Returns the held Claim.
        """
        name = f"{env.message_id}.json"
        tmp = self.root / "tmp" / name
        tmp.write_bytes(env.to_bytes())
        dst = self.root / "in-progress" / name
        os.rename(tmp, dst)  # atomic within the fs
        return Claim(self, dst, env)

    def in_progress(self) -> list[Claim]:
        """Re-adopt messages left in in-progress/ by a prior run (crash recovery).

        A claim moves a message to in-progress/ and only leaves on complete/fail,
        so anything still here when a consumer restarts is unfinished work it
        should resume.

        A file that cannot be parsed is QUARANTINED rather than raised past: a
        consumer re-lists this directory on every tick, so one torn write or one
        envelope from a drifted producer would otherwise abort the listing
        forever -- the whole daemon crash-looping under systemd, and its
        diagnostics (`icu ps` reads the same directory) down with it. Renamed
        out of the way like Subscription.poll does, so the bad file survives for
        inspection and the good ones are still returned.
        """
        d = self.root / "in-progress"
        out = []
        for name in sorted(p.name for p in d.iterdir() if p.suffix == ".json"):
            path = d / name
            try:
                env = Envelope.from_bytes(path.read_bytes())
            except Exception as e:
                bad = path.with_suffix(".json.bad")
                try:
                    os.rename(path, bad)
                except OSError:
                    pass
                log.error("ERROR: bus: unparseable %s; quarantined: %s", name, e)
                continue
            out.append(Claim(self, path, env))
        return out


class Claim:
    """A claimed message held in in-progress, moved to a terminal dir when done."""

    def __init__(self, channel: Channel, path: Path, env: Envelope) -> None:
        self.channel = channel
        self.path = path
        self.env = env

    def _move(self, sub: str) -> None:
        dst = self.channel.root / sub / self.path.name
        try:
            os.rename(self.path, dst)
        except FileNotFoundError:
            # The message already left in-progress: this claim was resolved
            # twice, or a second consumer/daemon is running against the same
            # bus. The terminal dirs are audit-only, so the intent (mark this
            # claim done/failed) is already satisfied or moot -- warn, so a
            # stray second daemon shows up in the logs, but never crash the
            # consumer's task with an unretrieved FileNotFoundError.
            log.warning(
                "claim %s already gone from in-progress (concurrent consumer?);"
                " skipping move to %s",
                self.path.name,
                sub,
            )
            return
        self.path = dst

    def complete(self) -> None:
        self._move("done")

    def fail(self) -> None:
        self._move("failed")

    def release(self) -> None:
        self._move("incoming")

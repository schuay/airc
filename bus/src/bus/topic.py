# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Directory-backed publish/subscribe topics: an append-only log per topic plus
independent per-subscriber cursors.

Distinct from Channel (the maildir work-queue, claim-once). A topic is broadcast:
every subscriber reads the whole stream at its own pace, tracking its own
position. Layout under the bus root:

    topics/<domain>/<topic>/<seq>.json     append-only messages (Envelope bytes)
    cursors/<subscriber>/<domain>/<topic>  one int: the last seq handled

Each topic has a single publisher, so the next sequence number is allocated
without coordination (highest present + 1). seq is contiguous: a subscriber
detects a gap, and the cursor is a plain integer compare. Publish is atomic
(write a temp file, rename into place -- a half-written message is never
observed). Delivery is at-least-once: the cursor advances only after a message
is handled, so a crash between handling and ack reprocesses, and consumers must
be idempotent.

The Envelope's message_id (a ulid) is the global identity used for tracing; the
per-topic seq is purely the on-disk position and cursor unit.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from .envelope import Envelope

log = logging.getLogger(__name__)

_SEQ_WIDTH = 8
_MSG_RE = re.compile(r"^(\d+)\.json$")


def _safe(segment: str) -> str:
    # Topic/domain/subscriber names become path segments; keep them to a tame
    # charset so a stray name can never escape the root or collide with tmp files.
    if not segment or not re.fullmatch(r"[A-Za-z0-9._-]+", segment):
        raise ValueError(f"invalid name segment: {segment!r}")
    return segment


class Topic:
    """The publish side of one topic: an append-only, single-publisher log."""

    def __init__(self, root: Path | str, domain: str, topic: str) -> None:
        self.domain = _safe(domain)
        self.topic = _safe(topic)
        self.dir = Path(root) / "topics" / self.domain / self.topic
        self.dir.mkdir(parents=True, exist_ok=True)

    def latest(self) -> int:
        """Highest published seq, or 0 when the topic is empty."""
        seqs = [
            int(m.group(1)) for p in self.dir.iterdir() if (m := _MSG_RE.match(p.name))
        ]
        return max(seqs, default=0)

    def publish(self, env: Envelope) -> int:
        """Append `env` and return its seq. Atomic: write tmp, rename into place."""
        seq = self.latest() + 1
        name = f"{seq:0{_SEQ_WIDTH}d}.json"
        tmp = self.dir / f".tmp-{env.message_id}.json"
        tmp.write_bytes(env.to_bytes())
        os.rename(tmp, self.dir / name)  # atomic within the fs
        return seq

    def read(self, seq: int) -> Envelope:
        return Envelope.from_bytes(
            (self.dir / f"{seq:0{_SEQ_WIDTH}d}.json").read_bytes()
        )


class Subscription:
    """The consume side: one subscriber's cursor over one topic. Independent of
    every other subscriber -- fan-out is just each cursor advancing on its own."""

    def __init__(
        self, root: Path | str, subscriber: str, domain: str, topic: str
    ) -> None:
        root = Path(root)
        self.subscriber = _safe(subscriber)
        self._topic = Topic(root, domain, topic)
        self._cursor_path = (
            root / "cursors" / self.subscriber / self._topic.domain / self._topic.topic
        )
        self._cursor_path.parent.mkdir(parents=True, exist_ok=True)

    def cursor(self) -> int:
        try:
            return int(self._cursor_path.read_text().strip())
        except (FileNotFoundError, ValueError):
            return 0

    def poll(self) -> list[tuple[int, Envelope]]:
        """All messages with seq > cursor, in order. Does not advance the cursor;
        call ack(seq) after a message is durably handled."""
        cur = self.cursor()
        out: list[tuple[int, Envelope]] = []
        for p in self._topic.dir.iterdir():
            m = _MSG_RE.match(p.name)
            if not (m and (seq := int(m.group(1))) > cur):
                continue
            try:
                env = Envelope.from_bytes(p.read_bytes())
            except Exception as e:
                # A corrupt/half-written file (or an envelope a newer publisher
                # wrote that this reader cannot parse) must not abort the whole
                # poll and wedge every later message. Quarantine it out of the seq
                # namespace and keep going; the cursor advances past it when a
                # later message is acked. Loud (ERROR:) so the bug still surfaces.
                bad = p.parent / (p.name + ".bad")
                try:
                    p.rename(bad)
                except OSError:
                    pass
                log.error("ERROR: bus: unparseable %s; quarantined: %s", p.name, e)
                continue
            out.append((seq, env))
        out.sort(key=lambda t: t[0])
        return out

    def ack(self, seq: int) -> None:
        """Advance the cursor to seq (atomic). Idempotent; never moves backwards."""
        if seq <= self.cursor():
            return
        tmp = self._cursor_path.with_suffix(".tmp")
        tmp.write_text(str(seq))
        os.rename(tmp, self._cursor_path)

    def reset(self, seq: int = 0) -> None:
        """Rewind (or fast-forward) the cursor -- e.g. to replay from the start."""
        tmp = self._cursor_path.with_suffix(".tmp")
        tmp.write_text(str(seq))
        os.rename(tmp, self._cursor_path)

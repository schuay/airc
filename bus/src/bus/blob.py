# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Content-addressed blob store for fat payloads (patch diffs, build logs).

Keeps the channel messages small and greppable: a message references a blob by
its sha256, and the store deduplicates identical content. Writes are atomic
(tmp + rename) and idempotent (the ref is the content hash).
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

_REF_RE = re.compile(r"^[0-9a-f]{64}$")


class BlobStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        (self.root / "tmp").mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes) -> str:
        ref = hashlib.sha256(data).hexdigest()
        dst = self.root / ref
        if not dst.exists():
            tmp = self.root / "tmp" / ref
            tmp.write_bytes(data)
            os.rename(tmp, dst)
        return ref

    def get(self, ref: str) -> bytes:
        # Refs arrive off the bus in message fields, written by another process.
        # Validate before touching the filesystem: an unchecked ref like "/etc/
        # passwd" or "../../secret" would read an arbitrary file (put's ref is a
        # content hash, so it is always well-formed; only get trusts its caller).
        if not _REF_RE.match(ref):
            raise ValueError(f"invalid blob ref: {ref!r}")
        return (self.root / ref).read_bytes()

    def put_text(self, text: str) -> str:
        return self.put(text.encode())

    def get_text(self, ref: str) -> str:
        return self.get(ref).decode()

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""The transport envelope: the common wrapper every message on the bus shares.

The envelope is type-agnostic (payload is an opaque dict); a plugin's payload
models wrap and unwrap typed payloads. message_id identifies this message; a job's
many messages (job, progress, result) share a correlation_id (the job_id) and a
trace_id that spans the whole chain from source commit to chat notice.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .ids import ulid

SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Envelope(BaseModel):
    type: str
    payload: dict
    message_id: str = Field(default_factory=ulid)
    schema_version: int = SCHEMA_VERSION
    ts: str = Field(default_factory=_now_iso)
    producer: str = ""
    trace_id: str = ""
    correlation_id: str = ""

    def to_bytes(self) -> bytes:
        return self.model_dump_json().encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "Envelope":
        return cls.model_validate_json(data)

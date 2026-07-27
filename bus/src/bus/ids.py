# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""ULID generation: a 128-bit id whose 48-bit millisecond prefix makes the
Crockford-base32 string sort by creation time. The 80-bit random tail breaks
ties within a millisecond, so ordering between two ids minted in the same
millisecond is arbitrary (the channel relies on time order across milliseconds,
not strict per-call FIFO).
"""

from __future__ import annotations

import os
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid() -> str:
    ms = int(time.time() * 1000)
    n = (ms << 80) | int.from_bytes(os.urandom(10), "big")
    out = bytearray(26)
    for i in range(25, -1, -1):
        out[i] = ord(_CROCKFORD[n & 0x1F])
        n >>= 5
    return out.decode("ascii")

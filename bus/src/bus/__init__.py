# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Directory-backed message bus: the domain-neutral transport primitives.

`bus` is pure transport -- an Envelope (type + opaque JSON payload), append-only
Topics with per-subscriber cursors, a claim Channel, a content-addressed
BlobStore, and ulids. It knows nothing about what any payload means.

Typed domain payloads belong to the application that defines them, not here: an
app declares its own protocol and event schemas and hands `bus` the serialized
form. That keeps `bus` at the base of the dependency graph, consumable as a
standalone core package.
"""

from .blob import BlobStore
from .channel import Channel, Claim
from .envelope import SCHEMA_VERSION, Envelope
from .ids import ulid
from .topic import Subscription, Topic

# The permanent, domain-neutral surface. Only these belong to `bus`.
__all__ = [
    "BlobStore",
    "Channel",
    "Claim",
    "Envelope",
    "SCHEMA_VERSION",
    "Topic",
    "Subscription",
    "ulid",
]

# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

import pytest

from bus.blob import BlobStore


def test_blob_roundtrip(tmp_path):
    b = BlobStore(tmp_path)
    ref = b.put(b"hello world")
    assert b.get(ref) == b"hello world"
    assert b.put(b"hello world") == ref  # content-addressed, deduped


def test_blob_get_rejects_non_hash_ref(tmp_path):
    # Refs arrive off the bus from another process; get must not read an arbitrary
    # file when handed a path-shaped or otherwise-malformed ref.
    b = BlobStore(tmp_path)
    for bad in ("/etc/passwd", "../secret", "..", "abc", "", "g" * 64):
        with pytest.raises(ValueError):
            b.get(bad)

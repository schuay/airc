#!/usr/bin/env python3
# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Check license headers on the staged source paths supplied by a hook."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MARKER = re.compile(r"^# SPDX-License-Identifier:")
HEAD_LINES = 4
SUFFIXES = {".py", ".toml", ".sh", ".service"}
NAMES = {"Dockerfile"}
EXCLUDE_NAMES = {"uv.lock"}


def is_candidate(path: Path) -> bool:
    if path.name in EXCLUDE_NAMES or "vendor" in path.parts:
        return False
    return path.suffix in SUFFIXES or path.name in NAMES


def problem(path: Path) -> str | None:
    lines = path.read_text().split("\n")
    for i, line in enumerate(lines[:HEAD_LINES]):
        if MARKER.match(line):
            if i + 1 < len(lines) and lines[i + 1].strip():
                return "header not separated"
            return None
    return "missing header"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--files", type=Path, nargs="+")
    args = parser.parse_args()
    bad = 0
    for path in args.files:
        if not is_candidate(path) or not path.is_file():
            continue
        if defect := problem(path):
            print(f"{defect}: {path}")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

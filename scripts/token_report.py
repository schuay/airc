#!/usr/bin/env python3
# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Summarize token_usage from the suite's token ledger (tokens.db).

Groupings answer "where do tokens go": by kind (chat turn vs coordinator vs
triage vs review), by agent, by model, by day, by thread, plus the heaviest
single rows. Uncached input is reported everywhere because with prefix caching
cache reads are cheap -- uncached input, not raw input, is what costs money.

Read-only; safe to run against a live service.

Usage:
    scripts/token_report.py                # config token_db_path, all time
    scripts/token_report.py --today
    scripts/token_report.py --days 7
    scripts/token_report.py --db path.db

Always prints the rolling daily/weekly spend windows the fleet caps read
(daily_usd_cap / weekly_usd_cap), including when the next dollars free up: a
rolling window never visibly resets, so a bound fleet otherwise reads as a
stuck one.
"""

import argparse
import sqlite3
import time
from pathlib import Path

import tomllib

# The suite file every daemon is pointed at (see packaging/*.service), then the
# older name, so a deploy that still uses it keeps resolving. This used to name
# config.toml alone, which was survivable while token_db_path was the only key
# read here -- the DB default is right on a standard layout, so a miss was
# invisible. It stopped being survivable with the spend caps: a miss reads them
# as unset and the report says the fleet is unbound when it is not.
CONFIG_PATHS = (
    Path("~/.config/airc/airc.toml").expanduser(),
    Path("~/.config/airc/config.toml").expanduser(),
)
DEFAULT_DB = Path("~/.local/share/airc/tokens.db").expanduser()

COLUMNS = (
    "ts, thread_id, agent, kind, input_tokens, output_tokens,"
    " cached_input_tokens, model, model_calls, max_call_input_tokens, usd, estimated"
)


def _suite_config() -> tuple[dict, Path | None]:
    """The parsed suite file and where it came from. The path is returned so the
    report can SAY which file the caps below were read from -- "no cap set" and
    "no config found" look identical in the output otherwise, and they are very
    different states to be in."""
    for path in CONFIG_PATHS:
        if path.exists():
            with open(path, "rb") as f:
                return tomllib.load(f), path
    return {}, None


def resolve_db(cfg: dict) -> Path:
    # Same resolution as airc_core.load_common, without needing the venv: the
    # top-level token_db_path key, else the code default next to the store.
    if v := cfg.get("token_db_path"):
        return Path(v).expanduser()
    return DEFAULT_DB


# (label, seconds), matching airc_core.tokens.SpendWindows.
WINDOWS = (("daily", 86400.0), ("weekly", 7 * 86400.0))


def windows(path: Path, cfg: dict, source: Path | None) -> None:
    """What the rolling fleet caps see right now.

    A rolling window never visibly resets, so a bound fleet reads as a stuck
    one and the only other sign is a single log line. "next free" is what falls
    out of the window in the coming hour -- the dollars that come back without
    anybody doing anything -- which is the question an operator staring at a
    quiet queue actually has.
    """
    now = time.time()
    print("\n== rolling spend windows ==")
    print(
        f"{'window':<8} {'spent':>10} {'cap':>10} {'remaining':>10} {'next free':>10}"
    )
    for label, seconds in WINDOWS:
        cap = cfg.get(f"{label}_usd_cap")
        spent, estimated = _sum_usd(path, now - seconds, now)
        # The rows about to roll out: the leaky bucket's next drip.
        freeing, _ = _sum_usd(path, now - seconds, now - seconds + 3600.0)
        mark = "~" if estimated else ""
        print(
            f"{label:<8} {mark + f'${spent:.2f}':>10}"
            f" {(f'${cap:g}' if cap is not None else '-'):>10}"
            f" {(f'${cap - spent:.2f}' if cap is not None else '-'):>10}"
            f" {mark + f'${freeing:.2f}':>10}"
        )
    if not any(cfg.get(f"{label}_usd_cap") is not None for label, _ in WINDOWS):
        where = source or " or ".join(str(p) for p in CONFIG_PATHS)
        print(f"(no daily_usd_cap / weekly_usd_cap in {where}: nothing is bound)")


def _sum_usd(path: Path, since: float, until: float) -> tuple[float, bool]:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = db.execute(
            "SELECT COALESCE(SUM(usd), 0), COALESCE(MAX(estimated), 0)"
            " FROM token_usage WHERE ts >= ? AND ts < ?",
            (since, until),
        ).fetchone()
    finally:
        db.close()
    return float(row[0]), bool(row[1])


def load_rows(path: Path, since: float) -> list[sqlite3.Row]:
    # Immutable read would skip the WAL; a plain ro open replays it, which
    # matters because the live service commits there continuously.
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        return db.execute(
            f"SELECT {COLUMNS} FROM token_usage WHERE ts >= ?", (since,)
        ).fetchall()
    finally:
        db.close()


def fmt(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(int(n))


def day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


class Agg:
    def __init__(self) -> None:
        self.n = 0
        self.inp = 0
        self.out = 0
        self.cached = 0
        self.calls = 0
        self.usd = 0.0
        self.estimated = False

    def add(self, r: sqlite3.Row) -> None:
        self.n += 1
        self.inp += r["input_tokens"]
        self.out += r["output_tokens"]
        self.cached += r["cached_input_tokens"]
        self.calls += r["model_calls"]
        self.usd += r["usd"]
        self.estimated = self.estimated or bool(r["estimated"])

    @property
    def cost(self) -> str:
        return f"{'~' if self.estimated else ''}${self.usd:.2f}"

    @property
    def uncached(self) -> int:
        return self.inp - self.cached


def group(rows: list[sqlite3.Row], key) -> dict[str, Agg]:
    out: dict[str, Agg] = {}
    for r in rows:
        out.setdefault(key(r), Agg()).add(r)
    return out


def table(title: str, groups: dict[str, Agg], total_inp: int, limit: int = 0) -> None:
    print(f"\n== {title} ==")
    print(
        f"{'':<42} {'cost':>9} {'rows':>5} {'input':>9} {'uncached':>9} {'hit%':>5}"
        f" {'output':>8} {'in%':>5}"
    )
    items = sorted(groups.items(), key=lambda kv: -kv[1].usd)
    if limit:
        items = items[:limit]
    for name, a in items:
        hit = f"{100 * a.cached / a.inp:.0f}" if a.inp else "-"
        share = f"{100 * a.inp / total_inp:.1f}" if total_inp else "-"
        print(
            f"{name:<42} {a.cost:>9} {a.n:>5} {fmt(a.inp):>9} {fmt(a.uncached):>9}"
            f" {hit:>5} {fmt(a.out):>8} {share:>5}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--db",
        type=Path,
        help="token db path; default: token_db_path from the suite config",
    )
    ap.add_argument("--days", type=float, help="only include the last N days")
    ap.add_argument(
        "--today",
        action="store_true",
        help="only include rows since local midnight",
    )
    ap.add_argument("--top", type=int, default=10, help="row count for top-N sections")
    args = ap.parse_args()

    since = time.time() - args.days * 86400 if args.days else 0.0
    if args.today:
        now = time.localtime()
        midnight = time.struct_time(
            (now.tm_year, now.tm_mon, now.tm_mday, 0, 0, 0, 0, 0, -1)
        )
        since = max(since, time.mktime(midnight))

    cfg, source = _suite_config()
    path = args.db or resolve_db(cfg)
    if not path.exists():
        raise SystemExit(f"no token db at {path}")
    rows = load_rows(path, since)
    print(f"source: {path} ({len(rows)} rows)")
    # Before the early return and outside `since`: the windows are about the
    # caps as they stand now, not about whatever slice the arguments selected,
    # and "the fleet is bound" is the one thing worth saying even when the
    # selected range is empty.
    windows(path, cfg, source)
    if not rows:
        return

    total = Agg()
    for r in rows:
        total.add(r)
    first = time.strftime("%Y-%m-%d %H:%M", time.localtime(min(r["ts"] for r in rows)))
    last = time.strftime("%Y-%m-%d %H:%M", time.localtime(max(r["ts"] for r in rows)))
    hit = 100 * total.cached / total.inp if total.inp else 0
    print(
        f"{first} .. {last}\n"
        f"{total.cost}: input {fmt(total.inp)} ({fmt(total.uncached)} uncached,"
        f" {hit:.0f}% cache hit), output {fmt(total.out)}"
    )

    table("by kind", group(rows, lambda r: r["kind"]), total.inp)
    table(
        "by agent",
        group(rows, lambda r: f"{r['agent']} ({r['kind']})"),
        total.inp,
    )
    table("by model", group(rows, lambda r: r["model"]), total.inp)
    table("by day", group(rows, lambda r: day(r["ts"])), total.inp)
    table(
        f"top {args.top} threads by input",
        group(rows, lambda r: f"thread {r['thread_id']}"),
        total.inp,
        limit=args.top,
    )

    # Heaviest single rows: a max_call near the row's own input means one huge
    # prompt; a large input spread over many calls means an accumulating
    # tool-call loop re-sending its context every call.
    print(f"\n== top {args.top} single rows by input ==")
    print(
        f"{'when':<17} {'agent':<12} {'kind':<12} {'input':>9} {'calls':>6}"
        f" {'max/call':>9} {'output':>8}"
    )
    for r in sorted(rows, key=lambda r: -r["input_tokens"])[: args.top]:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))
        print(
            f"{when:<17} {r['agent']:<12} {r['kind']:<12}"
            f" {fmt(r['input_tokens']):>9} {r['model_calls'] or '-':>6}"
            f" {fmt(r['max_call_input_tokens']):>9} {fmt(r['output_tokens']):>8}"
        )


if __name__ == "__main__":
    main()

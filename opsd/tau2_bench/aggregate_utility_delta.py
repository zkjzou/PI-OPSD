#!/usr/bin/env python
"""Paired utility Δ for τ²: success(with-hint) − success(no-hint), per task.

    python -m opsd.tau2_bench.aggregate_utility_delta \\
        --root ./runs/tau2_pi_utility/pi_utility \\
        --baseline nohint --conditions nohint actions raw_no_think raw \\
        --out utility_delta.json

Reads ``<root>/<condition>/trajectories.jsonl`` (whatever ``baseline.py --output-dir`` wrote) and
emits a ``{"table": [{condition, ...}]}`` JSON (the schema the PI-attribution tables read) with
``utility_delta_paired_mean`` / ``utility_delta_pooled`` / ``tasks_improved`` / ``tasks_hurt``.

⚠️ THE BASELINE MUST BE IN --conditions. Omitting it raised a KeyError that discarded a whole run on
the ALFWorld side; here it is asserted up front instead, and the baseline row is emitted with Δ=0 so
the table always carries its own reference point.

⚠️ PAIRED, NOT POOLED, IS THE HEADLINE. τ² task difficulty spans a wide range, so a pooled difference
of means is dominated by which tasks happened to appear in each condition. The paired mean averages
per-task differences over the INTERSECTION of tasks present in both conditions, which is the only
comparison that holds difficulty fixed. Both are reported; ``paired`` is the one to quote.

⚠️ ONLY ``terminated_*`` EPISODES COUNT. An infra failure (dead endpoint, rate-limited sim, judge
error) is a missing measurement, not a zero -- scoring it as failure would make a throttled with-hint
leg look like a harmful hint.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path


def _load(path: Path) -> dict[str, list[float]]:
    """task_id -> [reward, ...] over terminated episodes, deduped on (task, trial) keeping the last."""
    by: dict[str, dict[int, float]] = collections.defaultdict(dict)
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001 - a torn final line must not kill the aggregation
                continue
            if not str(r.get("status", "")).startswith("terminated_"):
                continue
            by[str(r["task_id"])][int(r.get("trial") or 0)] = float(r.get("reward") or 0.0)
    return {t: [d[k] for k in sorted(d)] for t, d in by.items()}


def _mean(xs) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _t_stat(diffs: list[float]) -> tuple[float | None, int]:
    n = len(diffs)
    if n < 2:
        return None, max(n - 1, 0)
    m = sum(diffs) / n
    sd = statistics.stdev(diffs)
    if sd == 0:
        return (math.inf if m > 0 else (-math.inf if m < 0 else 0.0)), n - 1
    return m / (sd / math.sqrt(n)), n - 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--baseline", default="nohint")
    ap.add_argument("--traj-name", default="trajectories.jsonl")
    ap.add_argument("--out", default="utility_delta.json")
    a = ap.parse_args()

    if a.baseline not in a.conditions:
        raise SystemExit(
            f"ERROR: baseline '{a.baseline}' is not in --conditions {a.conditions}. Every Δ is measured "
            f"against it, so omitting it discards the entire run."
        )

    per = {c: _load(a.root / c / a.traj_name) for c in a.conditions}
    base = per[a.baseline]
    if not base:
        raise SystemExit(f"ERROR: no terminated episodes for the baseline at " f"{a.root / a.baseline / a.traj_name}")
    print(f"baseline '{a.baseline}': {len(base)} tasks, {sum(len(v) for v in base.values())} episodes")

    table = []
    for c in a.conditions:
        cur = per[c]
        shared = sorted(set(cur) & set(base))
        diffs = [(_mean(cur[t]) or 0.0) - (_mean(base[t]) or 0.0) for t in shared]
        t, df = _t_stat(diffs)
        pooled_cur = _mean(x for v in cur.values() for x in v)
        pooled_base = _mean(x for v in base.values() for x in v)
        row = {
            "condition": c,
            "n_tasks": len(cur),
            "n_tasks_paired": len(shared),
            "n_episodes": sum(len(v) for v in cur.values()),
            "success_rate": pooled_cur,
            "utility_delta_paired_mean": _mean(diffs),
            "utility_delta_pooled": (None if pooled_cur is None or pooled_base is None else pooled_cur - pooled_base),
            "t_paired": t,
            "df": df,
            "tasks_improved": sum(1 for d in diffs if d > 0),
            "tasks_hurt": sum(1 for d in diffs if d < 0),
            "tasks_unchanged": sum(1 for d in diffs if d == 0),
        }
        table.append(row)
        star = (
            ""
            if c == a.baseline
            else (
                "  SIGNIFICANT"
                if t is not None and abs(t) >= 2 and df >= 4
                else ("  below df>=4" if df < 4 else "  n.s.")
            )
        )
        print(
            f"  {c:14s} n={row['n_tasks']:>4} paired={row['n_tasks_paired']:>4} "
            f"succ={pooled_cur if pooled_cur is None else round(pooled_cur, 4)} "
            f"Δpaired={row['utility_delta_paired_mean'] if row['utility_delta_paired_mean'] is None else round(row['utility_delta_paired_mean'], 4)} "
            f"t={t if t is None else round(t, 2)} (df={df}) "
            f"+{row['tasks_improved']}/-{row['tasks_hurt']}{star}"
        )

    Path(a.out).write_text(json.dumps({"baseline": a.baseline, "table": table}, indent=1))
    print(f"\nwrote {a.out}")
    print(
        "⚠️ quote utility_delta_paired_mean, not the pooled value: τ² task difficulty spans a wide "
        "range and the pooled difference is dominated by task mix."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

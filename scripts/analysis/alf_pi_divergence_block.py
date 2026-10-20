#!/usr/bin/env python
"""ALFWorld PI DIVERGENCE block (KL, d, d_s, o50) for the ALFWorld PI-metrics table.

WHY A DIVERGENCE-ONLY SCRIPT. The ranking study's metrics script (not released) computes ranking and
divergence together, which forces the balanced-8W+8L subsample and the both-class filter on both.
Divergence needs none of that: it is not a ranking metric, so it takes no pool, no filter and no
seed, and restricting it COSTS accuracy -- up to 8.7% on this corpus (30bi glm_raw KL 0.1806
all-tasks vs 0.1964 balanced-16). This script therefore reports divergence over ALL rollouts of ALL
tasks, which is also what the table caption claims ("over all 25,600 rollouts" = 200 tasks x 128).

TOKEN-WEIGHTED, NOT ROW-MEAN. rows.parquet holds one row per (rollout, PI, position); a plain mean
over rows would weight a 40-token step the same as a 4,000-token one. Every statistic here is
token-weighted, matching that ranking-study script exactly.

--agg (how rollouts are combined; see pi_src_lib.agg_divergence):
  token_pooled (default) np.average(..., weights=n_tokens) over all rows of all rollouts -- what
                         produced the paper's numbers.
  traj_mean              the paper's equation for D: per-rollout token mean, then the mean over rollouts.

THE uid INTERSECTION IS NOT COSMETIC. Arms can differ in which rollouts they cover (a task with no
hint is skipped by some arms), and divergence is compared BETWEEN arms in the table. Restricting to
uids present for every PI arm makes the columns commensurable; without it a cheaper arm can look
different purely because it was measured on an easier subset.

WHICH CELL SUPPLIES WHICH ARMS is a data-availability constraint, not a choice. The base
alfworld_train200_* cells have only rollout_attrs.parquet -- their rows.parquet (which carries the
per-position divergences) was not kept -- so the 6 reference-trajectory arms can only come from the
cap24k cells. The 8 analyzer/summariser/checklist arms come from the base new8 cells. That is exactly
what the table's `src` column records.

o50 (top-50 overlap) is reported when the cell carries it; the released cells keep only
pi/task_id/uid/success/step/channel/n_tokens/d/dsigned/kl, so it prints as '-' there.

    python alf_pi_divergence_block.py \\
        --cells 30bi:alfworld_new8_train200_30bi:base,30bi:alfworld_train200_30bi_cap24k:cap24k
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pi_src_lib as S  # noqa: E402

COLS = ["uid", "pi", "kl", "d", "dsigned", "o50", "n_tokens"]
STATS = ("kl", "d", "dsigned", "o50")


def divergence(cell_dir: str, agg: str = "token_pooled") -> tuple[dict, int, int]:
    """Token-weighted divergence per PI arm over the rollouts common to every arm."""
    attrs = f"{cell_dir}/rollout_attrs.parquet"
    # (uid, pi) coverage: rollout_attrs.parquet when the cell has it (the released cells do), else the
    # distinct (uid, pi) pairs of rows.parquet -- the same set, since a scored rollout has >= 1 step row.
    src = attrs if os.path.exists(attrs) else f"{cell_dir}/rows.parquet"
    tok = pd.read_parquet(src, columns=["uid", "pi"]).drop_duplicates()
    pis = list(tok.pi.unique())
    common = set.intersection(*[set(tok[tok.pi == p].uid) for p in pis])
    have = set(pq.read_schema(f"{cell_dir}/rows.parquet").names)
    stats = [z for z in STATS if z in have]
    rw = pd.read_parquet(f"{cell_dir}/rows.parquet", columns=[c for c in COLS if c in have])
    rw = rw[rw.uid.isin(common)]
    out = {}
    for pi in sorted(p for p in pis if not str(p).endswith("_null")):
        b = rw[rw.pi == pi]
        if b.empty:
            continue
        out[str(pi)] = {z: S.agg_divergence(b, z, agg) for z in stats}
        out[str(pi)]["n_rows"] = int(len(b))
    return out, len(common), len(pis)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--grid",
        default=None,
        help="directory holding the cells (default: $PI_GRID_ROOT or ./outputs/pi_cells/alfworld)",
    )
    ap.add_argument(
        "--cells",
        required=True,
        help="comma-separated model:cell:src triples, e.g. 30bi:alfworld_new8_train200_30bi:base",
    )
    ap.add_argument(
        "--agg",
        choices=S.AGG_CHOICES,
        default="token_pooled",
        help="token_pooled = the paper numbers; traj_mean = the paper's per-trajectory equation",
    )
    ap.add_argument("--out", default="", help="optional JSON dump")
    args = ap.parse_args()
    grid = args.grid or S.grid_root("alfworld")

    result = {}
    for spec in args.cells.split(","):
        model, cell, src = spec.split(":")
        d, n_roll, n_pi = divergence(f"{grid}/{cell}", args.agg)
        print(f"\n=== {model} | {cell} | src={src} | agg={args.agg} | {n_roll} rollouts common to {n_pi} arms ===")
        print(f"{'arm':22s}{'KL':>9}{'d':>9}{'d_s':>10}{'o50':>8}{'rows':>12}")
        for pi, v in sorted(d.items(), key=lambda kv: -kv[1]["kl"]):
            o50 = f"{v['o50']:>8.3f}" if "o50" in v else f"{'-':>8}"
            print(f"{pi:22s}{v['kl']:>9.4f}{v['d']:>9.4f}{v['dsigned']:>+10.4f}{o50}{v['n_rows']:>12,}")
            result[f"{model}|{pi}"] = dict(v, src=src, cell=cell, rollouts=n_roll, agg=args.agg)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out} ({len(result)} arm rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

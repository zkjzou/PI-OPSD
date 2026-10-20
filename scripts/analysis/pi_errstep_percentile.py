#!/usr/bin/env python
"""Error-step PERCENTILE: is the judge-flagged error step the most-suppressed step of its own trajectory?

Per scored step j of rollout τ (arm I, task x), with the published position baseline
b_I(x, j) = mean d⁻ over all rollouts of x at step j:
    Δd_j    = d⁻_j - b_I(x, j)
    ratio_j = d⁻_j / b_I(x, j)
Within each LOST labelled rollout, rank its steps by the score; a flagged step's percentile is the share of the
rollout's OTHER steps it beats (ties count 1/2): (average rank - 1) / (n_steps - 1). 1.0 = the error step is the
single most-suppressed step, 0.5 = chance. The metric is the mean over flagged steps (each flagged step weighted
equally, as in Δd⁻_err). It is a within-trajectory AUC of flagged vs other steps, so it is SCALE-FREE: a hint that
suppresses more everywhere cannot raise it.

Also: top-1 rate (flagged step is the rollout's max), raw d⁻ percentile (no baseline), the same percentile at
flagged steps of WON rollouts (the labeller null), and a task-cluster bootstrap 95% CI for the Δd percentile.

    python pi_errstep_percentile.py                  # the 145-task self/published set (8 arms)
    python pi_errstep_percentile.py --multimodel     # the 200-task multi-author set
Output: --out (default ./outputs/pi_errstep_percentile/{self145,multimodel}/errstep_percentile.json).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alf_study_common as C  # noqa: E402
import pi_src_lib as S  # noqa: E402
from pi_glm8_report import CAP24K, COLS, LABELS  # noqa: E402
from pi_self4_report import load_chunks  # noqa: E402

N = str(C.PI_STUDY_ROOT)


def load_self145() -> pd.DataFrame:
    rc = load_chunks(Path(f"{N}/pi_selfcorrect200")).assign(pi="self_correct_raw")
    rw = load_chunks(Path(f"{N}/pi_selfwrong200")).assign(pi="self_wrong_raw")
    uids = set(rc.uid) & set(rw.uid)
    pub = pd.read_parquet(CAP24K, columns=COLS, filters=[("task_id", "in", sorted(set(rc.task_id)))])
    rows = pd.concat([rc, rw, pub], ignore_index=True)
    return rows[rows.uid.isin(uids)]


def load_multimodel(dirs=None) -> pd.DataFrame:
    rows = pd.concat([load_chunks(Path(d)) for d in (dirs or [f"{N}/pi_mm_offline200"])], ignore_index=True)
    pub = pd.read_parquet(CAP24K, columns=COLS)
    return pd.concat([rows, pub[pub.uid.isin(set(rows.uid))]], ignore_index=True)


def percentiles(ps: pd.DataFrame, score: str) -> pd.Series:
    g = ps.groupby(["pi", "uid"], observed=True)[score]
    n = g.transform("size")
    return ((g.rank(method="average") - 1) / (n - 1)).where(n > 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--multimodel", action="store_true")
    ap.add_argument("--mm-dir", action="append", default=None, help="multi-author scoring dir(s) (repeatable)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument(
        "--task-ids",
        default=None,
        help="JSON list: restrict flagged/null steps to these tasks (common set)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = load_multimodel(args.mm_dir) if args.multimodel else load_self145()
    labels = json.load(open(LABELS))

    ps = S.per_step(rows)
    ps["b"] = ps.groupby(["pi", "task_id", "step"], observed=True)["dminus"].transform("mean")
    ps = ps[ps.uid.isin(set(labels))].copy()  # only labelled rollouts are ranked
    ps["delta"] = ps["dminus"] - ps["b"]
    ps["ratio"] = ps["dminus"] / ps["b"].where(ps["b"] > 0)
    for s in ("delta", "ratio", "dminus"):
        ps[f"pct_{s}"] = percentiles(ps, s)
    ps["top1_delta"] = ps["delta"] == ps.groupby(["pi", "uid"], observed=True)["delta"].transform("max")
    ps["win"] = ps.uid.map(ps.groupby("uid").success.first() > 0)
    flagged = pd.DataFrame([(u, int(s)) for u, ss in labels.items() for s in set(ss)], columns=["uid", "step"])
    ps = ps.merge(flagged.assign(hit=True), on=["uid", "step"], how="left")
    ps["hit"] = ps["hit"].astype("boolean").fillna(False).astype(bool)

    # within-rollout ranks and the position baseline are per task, so filtering here is exact
    if args.task_ids:
        ps = ps[ps.task_id.isin(set(json.load(open(args.task_ids))))]
    rng = np.random.default_rng(0)
    res = {}
    for p, g in ps.groupby("pi"):
        fl = g[g.hit & ~g.win & g.pct_delta.notna()]
        nul = g[g.hit & g.win & g.pct_delta.notna()]
        per_task = fl.groupby("task_id").pct_delta.agg(["sum", "size"])
        boots = []
        for _ in range(args.boot):
            s = per_task.sample(len(per_task), replace=True, random_state=int(rng.integers(1 << 31)))
            boots.append(s["sum"].sum() / s["size"].sum())
        res[p] = {
            "pct_delta": fl.pct_delta.mean(),
            "pct_delta_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "pct_ratio": fl.pct_ratio.mean(),
            "pct_raw_dminus": fl.pct_dminus.mean(),
            "top1_delta": fl.top1_delta.mean(),
            "null_won_pct_delta": nul.pct_delta.mean(),
            "flagged_steps": len(fl),
            "rollouts": fl.uid.nunique(),
            "null_steps": len(nul),
        }
    t = pd.DataFrame(res).T.sort_values("pct_delta")
    out = Path(
        args.out
        or C.default_out("pi_errstep_percentile")
        / ("multimodel" if args.multimodel else "self145")
        / "errstep_percentile.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    pd.set_option("display.width", 220)
    t["pct_delta_ci"] = t["pct_delta_ci"].map(lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]")
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()

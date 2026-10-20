"""Error-step columns for the AppWorld train90+dev table (appworld_traindev_pi_metrics.tex).

Same estimator that reproduces Table 5's published err_F 42/42 (appworld_pi_metrics.tex):
  1. per split, per_step() over the split's hint cell + trajectory cell (4bt traj from thinkfix);
  2. baseline b[a,t,s] = mean d- over the FULL cell (all 128 rollouts/task), before any subsetting;
  3. keep labelled rollouts present in every one of the 14 arms (the table's own basis);
  4. concatenate train90 and dev (task ids are split-specific, so baselines never mix);
  5. per arm: err_F / err_W = mean d- at flagged steps on lost / won rollouts,
     Delta-err_F/W/all = mean (d- - b) at flagged steps on lost / won / all rollouts,
     net = Delta-err_F minus the same excess on unflagged steps of lost rollouts.
Labels: pi_error_step_label.py --cell appworld_{dev,train90}_{4bi,4bt,30bi}, k=2, medium.
Cells resolve through pi_src_lib (PI_GRID_ROOT / OPSD_DATA_ROOT); labels default to the AppWorld labels
root. The dataset bundle carries the train90 cells and labels only, so the dev split is reported missing
unless you regenerate it.

    python pi_error_step_appworld_traindev.py [--labels DIR] [--out CSV]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pi_src_lib as S  # noqa: E402

COLS = ["pi", "task_id", "uid", "success", "step", "channel", "n_tokens", "kl", "d", "dsigned"]


def cells(split, m):
    traj = f"appworld_{split}_{m}_thinkfix" if m == "4bt" else f"appworld_{split}_{m}"
    return [f"appworld_hint128_{split}_{m}", traj]


def split_frame(split, m, lab):
    full = S.per_step(S.load_cells(cells(split, m), COLS))
    full["_b"] = full.groupby(["pi", "task_id", "step"], observed=True)["dminus"].transform("mean")
    ps = full[full.uid.isin(set(lab))].copy()
    cov = ps.groupby("uid").pi.nunique()
    ps = ps[ps.uid.isin(set(cov[cov == ps.pi.nunique()].index))].copy()
    ps["split"] = split
    return ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--labels",
        default=None,
        help="label directory (default: $PI_LABELS_ROOT or <appworld>/pi_metrics/error_labels)",
    )
    ap.add_argument(
        "--out",
        default="./outputs/pi_error_step_appworld_traindev/appworld_traindev_errsteps.csv",
    )
    ap.add_argument("--models", default="4bi,4bt,30bi")
    ap.add_argument("--splits", default="train90,dev")
    a = ap.parse_args()
    a.labels = a.labels or S.labels_root("appworld")
    rows = []
    for m in a.models.split(","):
        parts = []
        for sp in a.splits.split(","):
            p = f"{a.labels}/labels_appworld_{sp}_{m}_k2_medium.json"
            if not os.path.exists(p):
                print(f"  {m} {sp}: no labels at {p}")
                continue
            parts.append(split_frame(sp, m, json.load(open(p))))
        if not parts:
            continue
        ps = pd.concat(parts, ignore_index=True)
        lab = {}
        for sp in a.splits.split(","):
            p = f"{a.labels}/labels_appworld_{sp}_{m}_k2_medium.json"
            if os.path.exists(p):
                lab.update(json.load(open(p)))
        su = ps.groupby("uid").success.first()
        ps["win"] = ps.uid.map(su > 0)
        ps["hit"] = [s in set(lab[u]) for u, s in zip(ps.uid, ps.step)]
        ps["dev"] = ps["dminus"] - ps["_b"]
        print(
            f"  {m}: arms {ps.pi.nunique()}, rollouts {ps.uid.nunique():,} "
            f"({int((~ps.groupby('uid').win.first()).sum()):,} lost / {int(ps.groupby('uid').win.first().sum()):,} won), "
            f"tasks {ps.task_id.nunique()} ({', '.join(f'{k} {v}' for k, v in ps.groupby('split').task_id.nunique().items())})"
        )
        for pi in sorted(ps.pi.unique()):
            g = ps[ps.pi == pi]
            F, W = g[~g.win], g[g.win]
            fe, we = F[F.hit], W[W.hit]
            rows.append(
                dict(
                    model=m,
                    arm=pi,
                    errF=fe.dminus.mean(),
                    errW=we.dminus.mean() if len(we) else np.nan,
                    derrF=fe.dev.mean(),
                    derrW=we.dev.mean() if len(we) else np.nan,
                    derrALL=g[g.hit].dev.mean(),
                    netF=fe.dev.mean() - F[~F.hit].dev.mean(),
                    n_flag_F=len(fe),
                    n_flag_W=len(we),
                )
            )
    D = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    D.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(D)} rows)")


if __name__ == "__main__":
    main()

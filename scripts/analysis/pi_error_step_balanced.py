"""Balanced error-step metrics: equal numbers of won and lost rollouts per task (16 + 16).

The natural-mix labels (32 rollouts/task) are mostly lost rollouts on hard tasks and mostly won on easy
ones, so Delta-err_F and its labeller null Delta-err_W are measured on different task mixes. Here every
task contributes k = min(K, #won labelled, #lost labelled) rollouts from EACH side (seeded draw), so the
two sides share one task mix and one weight per task.

  bal     every task with k > 0 (k <= K; hard/easy tasks contribute fewer, but equally on both sides)
  bal16   only tasks with k == K (the strict 16 + 16 design; mid-difficulty tasks only)
  natural the existing natural-mix labels, unbalanced -- the tables' basis, kept as the reference

Labels:
  ALFWorld / AppWorld : pi_error_step_label.py --cell <cell> --balance 16  (labels_<cell>_k2_medium_bal16.json)
  tau2                : every one of the 32 rollouts/task is already labelled, so no new calls are needed
Estimator (same as the tables): baseline b[a,t,s] = mean d- over the FULL cell (all rollouts of each
(arm, task, step)) before any subsetting; ALFWorld/AppWorld keep rollouts present in all 14 arms;
dev = d- - b; err_F/W = mean d- at flagged steps of lost/won rollouts; Delta-err_F/W/all = mean dev
there; net_F/W = Delta-err minus the same excess on unflagged steps; gap = Delta-err_F - Delta-err_W.
Correlation: Spearman over the 14 arms against OPSD best-window pass@1 of your own runs (--targets <csv>,
columns bench,student,pi,pass1; see pi_src_lib.opsd_best_window).

Inputs (see pi_src_lib.py): your scored cells under $PI_GRID_ROOT or ./outputs/pi_cells/<bench> (score the
PIs first with python -m opsd.alfworld.analyze_pi_positions), labels under $PI_LABELS_ROOT or
$OPSD_DATA_ROOT/<bench>/pi_metrics/error_labels. A bench whose inputs are not present is skipped with
a message.

    python pi_error_step_balanced.py [--k 16] [--benches alfworld,appworld_traindev,...] [--out-dir DIR]
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
METRICS = ["errF", "errW", "derrF", "derrW", "derrALL", "netF", "netW", "gap"]


def tf(m, name):
    return f"{name}_thinkfix" if m == "4bt" else name


def parts(bench, m, bal):
    """[(cells, label files, task filter, common-rollout filter)] per independently-baselined part."""
    sfx = "_bal16" if bal else ""
    LK = S.labels_root("alfworld" if bench == "alfworld" else "appworld")
    LT = S.labels_root("tau2")
    TG = S.grid_root("tau2")  # tau2 cells: <pool>/<student>/rows.parquet
    if bench == "alfworld":
        return [
            (
                [f"alfworld_new8_train200_{m}", f"alfworld_train200_{m}_cap24k"],
                [f"{LK}/labels_alfworld_{m}_k2_medium{sfx}.json"],
                None,
                True,
            )
        ]
    if bench == "appworld_normal":
        return [
            (
                [f"appworld_hint128_normal_{m}", tf(m, f"appworld_passk128_normal_{m}")],
                [f"{LK}/labels_appworld_{m}_k2_medium{sfx}.json"],
                None,
                True,
            )
        ]
    if bench == "appworld_traindev":
        return [
            (
                [f"appworld_hint128_{s}_{m}", tf(m, f"appworld_{s}_{m}")],
                [f"{LK}/labels_appworld_{s}_{m}_k2_medium{sfx}.json"],
                None,
                True,
            )
            for s in ["train90", "dev"]
        ]
    if bench == "tau2_base":
        return [
            (
                [f"{TG}/luna32/{m}/rows.parquet"],
                [f"{LT}/labels_tau2_{m}_luna_k1_medium.json"],
                None,
                False,
            )
        ]
    if bench == "tau2_train132":
        keep = set(json.load(open(os.path.join(S.bench_dir("tau2"), "splits", "matched132_tasks.json"))))
        return [
            (
                [f"{TG}/t211/{m}/rows.parquet"],
                [f"{LT}/labels_tau2_{m}_t211_k1_medium.json"],
                keep,
                False,
            ),
            (
                [f"{TG}/m75/{m}/rows.parquet"],
                [f"{LT}/labels_tau2_{m}_m75_k1_medium.json"],
                keep,
                False,
            ),
        ]
    raise ValueError(bench)


def frame(bench, m, bal):
    out, lab = [], {}
    for cells, lps, keep, common in parts(bench, m, bal):
        full = S.per_step(S.load_cells(cells, COLS, task_ids=keep))
        full = full[~full.pi.str.endswith("_null")]
        full["_b"] = full.groupby(["pi", "task_id", "step"], observed=True)["dminus"].transform("mean")
        lb = S.load_labels(lps)
        ps = full[full.uid.isin(set(lb))].copy()
        if common:
            cov = ps.groupby("uid").pi.nunique()
            ps = ps[ps.uid.isin(set(cov[cov == ps.pi.nunique()].index))]
        out.append(ps)
        lab.update(lb)
    ps = pd.concat(out, ignore_index=True)
    ps["win"] = ps.uid.map(ps.groupby("uid").success.first() > 0)
    ps["hit"] = [s in set(lab[u]) for u, s in zip(ps.uid, ps.step)]
    ps["dev"] = ps["dminus"] - ps["_b"]
    return ps


def balance(ps, k, seed=0):
    """Per task, k_t = min(k, #won, #lost) rollouts from each side; returns (uids, k_t per task)."""
    rng = np.random.default_rng(seed)
    ro = ps.groupby("uid").agg(t=("task_id", "first"), w=("win", "first")).reset_index()
    keep, kt = [], {}
    for t, g in ro.groupby("t"):
        won, lost = sorted(g.uid[g.w]), sorted(g.uid[~g.w])
        kt[t] = n = min(k, len(won), len(lost))
        if n:
            keep += list(rng.choice(won, n, replace=False)) + list(rng.choice(lost, n, replace=False))
    return set(keep), pd.Series(kt)


def arm_metrics(g):
    F, W = g[~g.win], g[g.win]
    fe, we = F[F.hit], W[W.hit]
    r = dict(
        errF=fe.dminus.mean(),
        errW=we.dminus.mean(),
        derrF=fe.dev.mean(),
        derrW=we.dev.mean(),
        derrALL=g[g.hit].dev.mean(),
        netF=fe.dev.mean() - F[~F.hit].dev.mean(),
        netW=we.dev.mean() - W[~W.hit].dev.mean(),
    )
    r["gap"] = r["derrF"] - r["derrW"]
    return r


# ---- OPSD best-window pass@1 of your runs (--targets) ----
def opsd(bench, targets=None):
    """{student: {arm: best-window pass@1}} for `bench` (see pi_src_lib.opsd_best_window)."""
    return S.opsd_best_window(bench, targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--benches", default="alfworld,appworld_normal,appworld_traindev,tau2_base,tau2_train132")
    ap.add_argument("--models", default="4bi,4bt,30bi")
    ap.add_argument("--out-dir", default="./outputs/pi_error_step_balanced")
    ap.add_argument("--targets", default=None, help="CSV bench,student,pi,pass1 of your OPSD runs ($OPSD_TARGETS_CSV)")
    a = ap.parse_args()
    rows, cors = [], []
    for bench in a.benches.split(","):
        O = opsd(bench.split("_")[0], a.targets)
        for m in a.models.split(","):
            aw = not bench.startswith("tau2")
            try:
                designs = {"natural": frame(bench, m, bal=False)}
                bal = frame(bench, m, bal=True) if aw else designs["natural"]
            except OSError as e:  # FileNotFoundError from open() / pyarrow
                print(f"  [skip] {bench} {m}: missing input ({e})", flush=True)
                continue
            keep, kt = balance(bal, a.k)
            designs["bal"] = bal[bal.uid.isin(keep)]
            full_k = set(kt[kt == a.k].index)
            designs[f"bal{a.k}"] = designs["bal"][designs["bal"].task_id.isin(full_k)]
            for dn, ps in designs.items():
                w = ps.groupby("uid").win.first()
                info = dict(
                    bench=bench,
                    model=m,
                    design=dn,
                    tasks=ps.task_id.nunique(),
                    won=int(w.sum()),
                    lost=int((~w).sum()),
                    arms=ps.pi.nunique(),
                )
                per = {pi: arm_metrics(g) for pi, g in ps.groupby("pi")}
                for pi, r in per.items():
                    rows.append({**info, "arm": pi, **r})
                arms = [pi for pi in per if pi in O.get(m, {})]
                c = dict(info, n_arms=len(arms))
                for k in METRICS:
                    s = S.spearman([per[pi][k] for pi in arms], [O[m][pi] for pi in arms])
                    c[f"rho_{k}"], c[f"p_{k}"] = s["rho"], s["p"]
                cors.append(c)
                print(
                    f"  {bench:18s}{m:5s}{dn:8s} tasks {info['tasks']:3d} won {info['won']:5,} lost {info['lost']:5,}  "
                    + "  ".join(f"{k} {c['rho_' + k]:+.2f}{'*' if c['p_' + k] < 0.05 else ' '}" for k in METRICS),
                    flush=True,
                )
    os.makedirs(a.out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out_dir}/errsteps_balanced_k{a.k}_arms.csv", index=False)
    pd.DataFrame(cors).to_csv(f"{a.out_dir}/errsteps_balanced_k{a.k}_vs_opsd.csv", index=False)
    print(f"wrote {a.out_dir}/errsteps_balanced_k{a.k}_{{arms,vs_opsd}}.csv")


if __name__ == "__main__":
    main()

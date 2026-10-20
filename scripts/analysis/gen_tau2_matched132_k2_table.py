"""Emit tables/tau2_matched132_pi_metrics.tex: tau2 retail-train PI diagnostics on matched-132, k=2 labels.

POOL. matched-132 = the 150 retail-train tasks drawn to mirror retail-base's per-student difficulty,
less the 18 with no reference trajectory: 75 carried from the 211-task cell (t211) and
57 generated separately (m75). $OPSD_DATA_ROOT/tau2/splits/matched132_tasks.json

COLUMNS as Tables 11/12 (tau2_retail_pi_metrics.tex / tau2_train_pi_metrics.tex):
    utility   Delta pass@1 vs the student's own no-hint arm, n=4/task, paired per task; sign test
    KL, d, d_s, o50   token-weighted over all rollouts (32/task) of each arm
    err_F .. net      natural-sample error-step metrics from the k=2 judge labels
                      (pi_error_step_label_tau2.py --pool {t211,m75} --tasks matched132 --k 2);
                      baseline b[a,t,s] = mean d- over each cell's rollouts of (arm, task, step)
Every cell and every number in the caption is computed here; the caption also quotes the same
pool's k=1 correlations so the effect of k is visible.

INPUTS (public dataset layout, see pi_src_lib.py / gen_tau2_median200_table.py): scored cells
<tau2 scores>/{t211,m75}/<student>/rows.parquet, k=1/k=2 judge labels under <tau2 error_labels>, per-arm
utility rollouts (TAU2_UTIL_ROOT) and the OPSD best-window pass@1 targets of your runs (--targets <csv>,
columns bench,student,pi,pass1; see pi_src_lib.opsd_best_window).

    python gen_tau2_matched132_k2_table.py --targets my_opsd_targets.csv [--out TEX]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_tau2_median200_table as G  # noqa: E402  (ROWS, MODELS, rates, util_path, sign_p, sgn)
import pi_src_lib as S  # noqa: E402

GRID = S.grid_root("tau2")
L = S.labels_root("tau2")
A = os.path.join(S.bench_dir("tau2"), "splits")
OUT = "./outputs/gen_tau2_matched132_k2_table/tau2_matched132_pi_metrics.tex"
COLS = [
    "pi",
    "task_id",
    "uid",
    "success",
    "step",
    "channel",
    "n_tokens",
    "kl",
    "d",
    "dsigned",
    "o50",
]
BASE_PASS1 = {"4bi": 0.405, "4bt": 0.668, "30bi": 0.493}  # retail-base 103, Table 11 headers


def errsteps(g, lab):
    """Natural-sample error-step metrics of one arm's per-step frame under one label set."""
    hit = np.array([s in lab.get(u, ()) for u, s in zip(g.uid, g.step)])
    labelled = g.uid.isin(set(lab)).values
    g = g[labelled]
    hit = hit[labelled]
    F, W = ~g.win.values, g.win.values
    fe, we = F & hit, W & hit
    return dict(
        errF=g.dminus.values[fe].mean(),
        errW=g.dminus.values[we].mean(),
        derrF=g.dev.values[fe].mean(),
        derrW=g.dev.values[we].mean(),
        derrALL=g.dev.values[hit].mean(),
        netF=g.dev.values[fe].mean() - g.dev.values[F & ~hit].mean(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--targets", default=None, help="CSV bench,student,pi,pass1 of your OPSD runs ($OPSD_TARGETS_CSV)")
    a = ap.parse_args()
    pool = set(json.load(open(f"{A}/matched132_tasks.json")))
    t211 = set(pd.read_parquet(f"{GRID}/t211/4bi/rows.parquet", columns=["task_id"]).task_id)
    hA, hB = pool - t211, pool & t211
    assert len(pool) == 132 and len(hA) == 57 and len(hB) == 75
    O = S.opsd_best_window("tau2", a.targets)  # {student: {arm: OPSD best-window pass@1}}
    # The published cells are slimmed (no o50): read it only when the cells carry it.
    have = set(pq.ParquetFile(f"{GRID}/t211/4bi/rows.parquet").schema_arrow.names)
    cols = [c for c in COLS if c in have]
    attrs = [c for c in ("kl", "d", "dsigned", "o50") if c in have]

    body, nohint, pass1, rho, flags = [], {}, {}, {}, {}
    for tag, pretty in G.MODELS:
        base = {
            **G.rates(G.util_path("A", tag, "nohint"), hA),
            **G.rates(G.util_path("B", tag, "nohint"), hB),
        }
        assert len(base) == len(pool), (tag, len(base))
        nohint[tag] = sum(base.values()) / len(base)
        util = {}
        for _, arms in G.ROWS:
            for arm, _ in arms:
                r = {
                    **G.rates(G.util_path("A", tag, arm), hA),
                    **G.rates(G.util_path("B", tag, arm), hB),
                }
                d_ = [r[t] - base[t] for t in r if t in base]
                util[arm] = (
                    float(np.mean(d_)),
                    G.sign_p(sum(x > 0 for x in d_), sum(x < 0 for x in d_)) < 0.05,
                )
        parts = []
        labs = {1: {}, 2: {}}
        for cell, keep, pk in [
            (f"{GRID}/t211/{tag}/rows.parquet", hB, "t211"),
            (f"{GRID}/m75/{tag}/rows.parquet", hA, "m75"),
        ]:
            ps = S.per_step(S.load_cells([cell], cols, task_ids=keep), attrs=attrs)
            ps["_b"] = ps.groupby(["pi", "task_id", "step"], observed=True)["dminus"].transform("mean")
            parts.append(ps)
            for k in (1, 2):
                lb = json.load(open(f"{L}/labels_tau2_{tag}_{pk}_k{k}_medium.json"))
                labs[k].update({u: set(v) for u, v in lb.items() if u.rsplit("#", 1)[0] in keep})
        ps = pd.concat(parts, ignore_index=True)
        ps["dev"] = ps.dminus - ps._b
        ps["win"] = ps.uid.map(ps.groupby("uid").success.first() > 0)
        pass1[tag] = float(ps.groupby("uid").win.first().mean())
        flags[tag] = float(np.mean([len(v) for v in labs[2].values()]))
        per = {}
        for pi, g in ps.groupby("pi"):
            w = g.n_tokens
            per[pi] = dict(
                KL=np.average(g.kl, weights=w),
                d=np.average(g.d, weights=w),
                ds=np.average(g.dsigned, weights=w),
                o50=np.average(g.o50, weights=w) if "o50" in g else np.nan,
                n=g.task_id.nunique(),
                k2=errsteps(g, labs[2]),
                k1=errsteps(g, labs[1]),
            )
        arms = [x for x in O[tag] if x in per]
        y = [O[tag][x] for x in arms]
        rho[tag] = {
            "u": S.spearman([util[x][0] for x in arms], y),
            "d": S.spearman([per[x]["d"] for x in arms], y),
            "f2": S.spearman([per[x]["k2"]["derrF"] for x in arms], y),
            "w2": S.spearman([per[x]["k2"]["derrW"] for x in arms], y),
            "f1": S.spearman([per[x]["k1"]["derrF"] for x in arms], y),
            "w1": S.spearman([per[x]["k1"]["derrW"] for x in arms], y),
        }
        best = max(util, key=lambda k_: util[k_][0])

        body.append(
            rf"      \multicolumn{{13}}{{l}}{{\textbf{{{pretty}}} "
            rf"\emph{{(no-hint pass@1 $={nohint[tag]:.4f}$)}}}} \\"
        )
        for src, arm_rows in G.ROWS:
            body.append(rf"      \multirow{{{len(arm_rows)}}}{{*}}{{\hspace{{1em}}{src}}}")
            for arm, label in arm_rows:
                delta, sig = util[arm]
                g, e = per[arm], per[arm]["k2"]
                u = G.sgn(delta)
                if arm == best:
                    u = rf"\textbf{{{u}}}"
                if sig:
                    u += "$^{*}$"
                o = ("--" if pd.isna(g["o50"]) else f"{g['o50']:.3f}") + (
                    rf" \tiny({int(g['n'])})" if int(g["n"]) != len(pool) else ""
                )
                es = (
                    f"{e['errF']:.4f} & {e['errW']:.4f} & {G.sgn(e['derrF'])} & {G.sgn(e['derrW'])} & "
                    f"{G.sgn(e['derrALL'])} & {G.sgn(e['netF'])}"
                )
                body.append(
                    rf"         &  {label:<22s} &  {u}  &  {g['KL']:.4f}  &  {g['d']:.4f}  &  "
                    rf"{G.sgn(g['ds'])}  &  {o} & {es} \\"
                )
            if src != "Self":
                body.append(r"      \addlinespace")
        body.append(r"      \midrule")
    body[-1] = r"      \bottomrule"

    def trio(k):
        out = []
        for tag in ["4bi", "4bt", "30bi"]:
            s = rho[tag][k]
            out.append(f"{s['rho']:+.2f}".replace("+", "{+}").replace("-", "{-}") + ("^{*}" if s["p"] < 0.05 else ""))
        return "$" + "/".join(out) + "$"

    p1 = "/".join(f"{pass1[t]:.3f}" for t in ["4bi", "4bt", "30bi"])
    pb = "/".join(f"{BASE_PASS1[t]:.3f}" for t in ["4bi", "4bt", "30bi"])
    fl = "--".join(f"{flags[t]:.2f}" for t in sorted(flags, key=lambda t: flags[t])[:: len(flags) - 1])
    caption = rf"""\caption{{\small $\tau^2$-bench (retail-\emph{{train}}) PI diagnostics on the \textbf{{matched-132}}
  pool, with error steps labelled at \textbf{{$k{{=}}2$}}. The pool is the scorable part of 150
  training tasks drawn to mirror retail-base's per-student difficulty: 18 of the 150 have no reference
  trajectory and so admit no PI arm. It has two halves with no shared task ids --- 75 from the 211-task
  training cell and 57 generated separately. Pass@1 on its 32-rollout corpus is {p1}
  (4B-Instruct/4B-Thinking/30B-A3B) against {pb} on retail-base, so it runs somewhat harder.
  Same 14 PIs, 3 students and column definitions as Table~\ref{{tab:tau2_retail_pi_metrics}}.
  \textbf{{utility}} is $\Delta$pass@1 against each model's own no-hint arm at $n{{=}}4$/task, paired
  per task; $^{{*}}$ is $p<0.05$, two-sided exact sign test. \textbf{{Divergence}} is token-weighted
  over all rollouts at 32 traj./task; arms not covering all 132 tasks carry their task count in the
  o50 cell. \textbf{{Error steps}}: the judge (\texttt{{gpt-5.6}}) marks \emph{{up to two}} decisive
  mistakes per trajectory ({fl} per rollout on average), not the single step of
  Tables~\ref{{tab:tau2_retail_pi_metrics}}--\ref{{tab:tau2_train_pi_metrics}}, so the error-step
  values here are not directly comparable to those tables'. Against \method{{}} post-training
  performance: utility {trio('u')}, $d$ {trio('d')}, and $\Delta d^{{-}}_{{\mathrm{{err}},F}}$
  {trio('f2')} against its labeller null $\Delta d^{{-}}_{{\mathrm{{err}},W}}$ {trio('w2')};
  the same pool labelled at $k{{=}}1$ gives {trio('f1')} against {trio('w1')}. Best utility per
  model in \textbf{{bold}}.}}"""

    tex = (
        r"""\begin{table}[t]
  """
        + caption
        + r"""
  \label{tab:tau2_matched132_pi_metrics}
  \begin{center}
    \small
    \setlength{\tabcolsep}{3pt}
    \renewcommand{\arraystretch}{0.775}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{ll c cccc cccccc}
      \toprule
      \multicolumn{2}{l}{\bf $\tau^2$-retail-train PI diagnostics (matched-132)} & {\bf utility} & """
        r"""\multicolumn{4}{c}{\bf Divergence (32 traj./task)} & \multicolumn{6}{c}{\bf Error steps ($k{=}2$, 32 traj./task)} \\
      \cmidrule(lr){3-3} \cmidrule(lr){4-7} \cmidrule(lr){8-13}
      \bf Model / Source & \bf PI & \bf $\Delta$p@1 & \bf KL & \bf $d$ & \bf $d_{\rm s}$ & \bf o50 & """
        r"""\bf err$_F$ & \bf err$_W$ & \bf $\Delta d^{-}_{\mathrm{err},F}$ & \bf $\Delta d^{-}_{\mathrm{err},W}$ & """
        r"""\bf $\Delta d^{-}_{\mathrm{err,all}}$ & \bf net \\
      \midrule
"""
        + "\n".join(body)
        + r"""
    \end{tabular}}
  \end{center}
\end{table}
"""
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write(tex)
    print(f"wrote {a.out}  ({len(tex)} B)")
    for tag, _ in G.MODELS:
        print(
            f"  {tag:5s} no-hint {nohint[tag]:.4f} pass@1 {pass1[tag]:.3f} flags/rollout {flags[tag]:.2f} | "
            + " ".join(f"{k} {rho[tag][k]['rho']:+.3f}" for k in ["u", "d", "f2", "w2", "f1", "w1"])
        )


if __name__ == "__main__":
    main()

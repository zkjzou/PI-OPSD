"""Emit tables/tau2_train_pi_metrics.tex (Table 7) on the median-200 retail-train pool.

POOL. 268 retail-train tasks have divergence scores and judge labels: the 211-task cell
(pi_grid_tau2_t211) plus 57 generated separately (pi_m75_inputs). The 200 are a uniform random
subset -- the one of 2,000 random 200-task draws closest to the median of every correlation with
OPSD best-window pass@1 (d, label-free d-, err_F, balanced Delta-err_F, its null, the gap), so it
is a representative draw, not a selected one. Task list:
    <tau2 error_labels>/analysis/tau2_train268_median200.json

COLUMNS (same definitions as Table 6, tables/tau2_retail_pi_metrics.tex):
    utility   Delta pass@1 vs the student's own no-hint arm, n=4/task, paired per task; exact sign test
    KL, d, d_s, o50   token-weighted over all rollouts (32/task) of each arm
    err_F .. net      natural-sample error-step metrics, read from the published
        <tau2 error_labels>/analysis/tau2_median200_errsteps.csv (computed from the published k=1
        error-step labels and the scored cells, restricted to the median-200 tasks; the helper that
        wrote it is not part of this release)

WHY GENERATED, NOT HAND-EDITED: 14 arms x 3 students x 11 columns; a transposed digit is invisible.
Helpers (ROWS, rates, util_path, sign_p) follow the earlier matched-132 table generator, which built
the previous version of this table.

INPUTS (see pi_src_lib.py): your scored cells <grid>/{t211,m75}/<student>/
rows.parquet, the analysis files under <tau2 error_labels>/analysis, and the per-arm utility rollouts.
Utility is read from TAU2_UTIL_ROOT/<tree>/<arm>/trajectories.jsonl when that file exists (trees
m75_boxb/utility_<s>, tau2_util_t123_<s>, tau2_util_pi8_211_<s>, as written by run_tau2_m75.sh /
run_tau2_util_t123_luna.sh / run_tau2_util_pi8.sh), else from slim per-episode records
TAU2_UTIL_ROOT/slim.parquet (default TAU2_UTIL_ROOT: ./outputs/utility/tau2).

    python gen_tau2_median200_table.py [--out TEX]
"""

import argparse
import collections
import json
import os
import sys
from math import comb

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pi_src_lib as S  # noqa: E402  (public data-layout helpers)

GRID = S.grid_root("tau2")
A = os.path.join(S.labels_root("tau2"), "analysis")
UTIL = os.environ.get("TAU2_UTIL_ROOT") or os.path.join("outputs", "utility", "tau2")
OUT = "./outputs/gen_tau2_median200_table/tau2_train_pi_metrics.tex"
N_TASKS = 200

ROWS = [
    (
        "GLM-5.2",
        [
            ("glm_actions", "Action"),
            ("glm_raw_no_think", "Action + Obs."),
            ("glm_raw", "Action + Obs. + Thou."),
            ("anlz_glm", "On-policy skill"),
            ("summ_glm", "Summarised ref."),
            ("chk_glm", "Checklist"),
        ],
    ),
    (
        "Qwen3.7-Plus",
        [
            ("qwen37_actions", "Action"),
            ("qwen37_raw_no_think", "Action + Obs."),
            ("qwen37_raw", "Action + Obs. + Thou."),
            ("anlz_qwn", "On-policy skill"),
            ("summ_qwn", "Summarised ref."),
            ("chk_qwn", "Checklist"),
        ],
    ),
    ("Self", [("anlz_self", "On-policy skill"), ("summ_self", "Summarised ref.")]),
]
MODELS = [
    ("4bi", "Qwen3-4B-Instruct-2507"),
    ("4bt", "Qwen3-4B-Thinking-2507"),
    ("30bi", "Qwen3-30B-A3B-Instruct-2507"),
]


def sign_p(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


_SLIM = None


def _slim():
    """The published per-episode utility records (tree, arm, task_id, status, reward, ...), or empty."""
    global _SLIM
    if _SLIM is None:
        f = os.path.join(UTIL, "slim.parquet")
        cols = ["tree", "arm", "task_id", "status", "reward"]
        _SLIM = pd.read_parquet(f, columns=cols) if os.path.exists(f) else pd.DataFrame(columns=cols)
    return _SLIM


def _tree_arm(path):
    """<UTIL>/<tree>/<arm>/trajectories.jsonl -> (tree, arm); tree may itself contain a '/'."""
    rel = os.path.relpath(os.path.dirname(path), UTIL)
    return os.path.dirname(rel), os.path.basename(rel)


def _slim_rows(path):
    tree, arm = _tree_arm(path)
    s = _slim()
    return s[(s.tree == tree) & (s.arm == arm)]


def rates(path, keep):
    d = collections.defaultdict(list)
    if not os.path.exists(path):
        # Same rule as the jsonl branch below: terminated episodes only, success := reward >= 1.
        g = _slim_rows(path)
        g = g[g.status.astype(str).str.startswith("terminated") & g.task_id.isin(keep)]
        for t, r in zip(g.task_id, g.reward):
            d[t].append(1 if (0 if pd.isna(r) else r) >= 1 else 0)
        return {k: sum(v) / len(v) for k, v in d.items()}
    for line in open(path, errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not str(r.get("status", "")).startswith("terminated"):
            continue
        if r["task_id"] in keep:
            d[r["task_id"]].append(1 if (r.get("reward") or 0) >= 1 else 0)
    return {k: sum(v) / len(v) for k, v in d.items()}


def util_path(half, tag, arm):
    """half A = the 57 separately generated tasks, half B = tasks of the 211-task cell.

    A run of opsd/tau2_bench/run_tau2_pi_utility.sh over all of matched-132 with OUT_DIR=$TAU2_UTIL_ROOT/<student>
    covers both halves in one tree, <student>/<arm>/trajectories.jsonl; it is used when present."""
    flat = f"{UTIL}/{tag}/{arm}/trajectories.jsonl"
    if os.path.exists(flat):
        return flat
    if half == "A":
        return f"{UTIL}/m75_boxb/utility_{tag}/{arm}/trajectories.jsonl"
    p = f"{UTIL}/tau2_util_t123_{tag}/{arm}/trajectories.jsonl"
    return (
        p if (os.path.exists(p) or len(_slim_rows(p))) else f"{UTIL}/tau2_util_pi8_211_{tag}/{arm}/trajectories.jsonl"
    )


def sgn(x):
    return f"{x:+.4f}".replace("+", "$+$").replace("-", "$-$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    pool = set(json.load(open(f"{A}/tau2_train268_median200.json")))
    assert len(pool) == N_TASKS
    t211 = set(pd.read_parquet(f"{GRID}/t211/4bi/rows.parquet", columns=["task_id"]).task_id)
    hA, hB = pool - t211, pool & t211
    err = pd.read_csv(f"{A}/tau2_median200_errsteps.csv").set_index(["model", "arm"])
    cols = ["pi", "task_id", "n_tokens", "kl", "d", "dsigned", "o50"]
    # The published cells are slimmed to pi/task_id/uid/success/step/channel/n_tokens/d/dsigned/kl: o50
    # is read only when the cell carries it (a full scorer output does) and renders as "--" otherwise.
    have = set(pq.ParquetFile(f"{GRID}/t211/4bi/rows.parquet").schema_arrow.names)
    cols = [c for c in cols if c in have]

    body, nohint = [], {}
    for tag, pretty in MODELS:
        base = {
            **rates(util_path("A", tag, "nohint"), hA),
            **rates(util_path("B", tag, "nohint"), hB),
        }
        assert len(base) == N_TASKS, (tag, len(base))
        nohint[tag] = sum(base.values()) / len(base)
        dv = pd.concat(
            [
                pd.read_parquet(f"{GRID}/m75/{tag}/rows.parquet", columns=cols).query("task_id in @hA"),
                pd.read_parquet(f"{GRID}/t211/{tag}/rows.parquet", columns=cols).query("task_id in @hB"),
            ],
            ignore_index=True,
        )
        agg = dv.groupby("pi").apply(
            lambda x: pd.Series(
                {
                    "KL": np.average(x["kl"], weights=x["n_tokens"]),
                    "d": np.average(x["d"], weights=x["n_tokens"]),
                    "ds": np.average(x["dsigned"], weights=x["n_tokens"]),
                    "o50": np.average(x["o50"], weights=x["n_tokens"]) if "o50" in x else np.nan,
                    "n": x.task_id.nunique(),
                }
            ),
            include_groups=False,
        )
        cells = {}
        for _, arms in ROWS:
            for arm, _ in arms:
                r = {**rates(util_path("A", tag, arm), hA), **rates(util_path("B", tag, arm), hB)}
                common = [t for t in r if t in base]
                if not common:
                    continue
                delta = sum(r[t] - base[t] for t in common) / len(common)
                pos = sum(1 for t in common if r[t] > base[t])
                neg = sum(1 for t in common if r[t] < base[t])
                cells[arm] = (delta, sign_p(pos, neg) < 0.05, len(common))
        best = max(cells, key=lambda k: cells[k][0])

        body.append(
            rf"      \multicolumn{{13}}{{l}}{{\textbf{{{pretty}}} "
            rf"\emph{{(no-hint pass@1 $={nohint[tag]:.4f}$)}}}} \\"
        )
        for src, arms in ROWS:
            body.append(rf"      \multirow{{{len(arms)}}}{{*}}{{\hspace{{1em}}{src}}}")
            for arm, label in arms:
                delta, sig, _ = cells[arm]
                g, e = agg.loc[arm], err.loc[(tag, arm)]
                u = sgn(delta)
                if arm == best:
                    u = rf"\textbf{{{u}}}"
                if sig:
                    u += "$^{*}$"
                o = ("--" if pd.isna(g.o50) else f"{g.o50:.3f}") + (
                    rf" \tiny({int(g.n)})" if int(g.n) != N_TASKS else ""
                )
                es = (
                    f"{e.errF:.4f} & {e.errW:.4f} & {sgn(e.derrF)} & {sgn(e.derrW)} & "
                    f"{sgn(e.derrALL)} & {sgn(e.netF)}"
                )
                body.append(
                    rf"         &  {label:<22s} &  {u}  &  {g.KL:.4f}  &  {g.d:.4f}  &  {sgn(g.ds)}  &  {o} & {es} \\"
                )
            if src != "Self":
                body.append(r"      \addlinespace")
        body.append(r"      \midrule")
    body[-1] = r"      \bottomrule"

    tex = (
        r"""\begin{table}[t]
  \caption{\small $\tau^2$-bench (retail-\emph{train}) PI diagnostics on \textbf{200 training tasks}:
  the same 14 PIs, 3 students and column definitions as Table~\ref{tab:tau2_retail_pi_metrics}, on
  \texttt{retail\_N} tasks disjoint from retail-base. \textbf{Pool.} 268 training tasks carry
  divergence scores and judge labels (a 211-task cell plus 57 tasks generated separately); these 200
  are a uniform random subset, namely the one of $2{,}000$ random draws closest to the median of
  every correlation discussed here, so it is neither a lucky nor an unlucky draw. Per-model no-hint
  pass@1 on the 200 is within $0.01$ of all 268.
  \textbf{Ranking the 14 arms by $d$ is a property of the PI, not the pool:} the order is identical
  here, on the difficulty-matched 132 and on all 268, for every student, so $\rho(d,\text{\method{}})$
  is ${+}0.645^{*}/{-}0.767^{*}/{+}0.538$ ($p{=}0.055$) on each (4B-Instruct/4B-Thinking/30B-A3B).
  \textbf{utility} is $\Delta$pass@1 against each model's own no-hint arm at $n{=}4$/task, paired per
  task; $^{*}$ is $p<0.05$, two-sided exact sign test. \textbf{Divergence} is token-weighted over all
  rollouts at 32 traj./task; arms not covering all 200 tasks carry their task count in the o50 cell.
  \textbf{Error steps} follow Table~\ref{tab:tau2_retail_pi_metrics} (\texttt{gpt-5.6}, $k{=}1$, all
  32 rollouts/task labelled). Against \method{} post-training performance, $\Delta d^{-}_{\mathrm{err},F}$
  gives $\rho={+}0.34/{+}0.53/{+}0.57^{*}$; reweighting won and lost rollouts so that every task
  contributes $\min(\#\text{won},\#\text{lost})$ to each side (it and its null then share one task mix) gives
  ${+}0.61^{*}/{+}0.56^{*}/{+}0.74^{*}$ against a null of ${+}0.50/{-}0.27/{+}0.64^{*}$: only
  4B-Thinking separates from its null (gap $\rho={+}0.68^{*}$).
  \textbf{The reasoning-bearing hint inverts on the Thinking student:} \emph{Action + Obs. + Thou.}
  is the best arm for both Instruct models, and its two versions are 4B-Thinking's two worst
  (${+}0.0112$, ${-}0.0697^{*}$), while the same hint with thinking stripped is its best. Rolled out
  under that arm, 4B-Thinking makes \emph{zero tool calls} in $26.3\%$ (GLM-5.2) and $37.8\%$ (Qwen3.7-Plus) of episodes, against $0.6\%$/$0.1\%$ for
  4B-Instruct on the identical hint files and $0.1\%$/$1.3\%$ for the stripped variants: it answers
  from the teacher's reasoning instead of acting. Best utility per model in \textbf{bold}.}
  \label{tab:tau2_train_pi_metrics}
  \begin{center}
    \small
    \setlength{\tabcolsep}{3pt}
    \renewcommand{\arraystretch}{0.775}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{ll c cccc cccccc}
      \toprule
      \multicolumn{2}{l}{\bf $\tau^2$-retail-train PI diagnostics (200 tasks)} & {\bf utility} & """
        r"""\multicolumn{4}{c}{\bf Divergence (32 traj./task)} & \multicolumn{6}{c}{\bf Error steps (32 traj./task)} \\
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
    for tag, _ in MODELS:
        print(f"  {tag:5s} no-hint pass@1 {nohint[tag]:.4f}")


if __name__ == "__main__":
    main()

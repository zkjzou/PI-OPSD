#!/usr/bin/env python
"""Build the AppWorld train90+dev PI-diagnostics table (LaTeX rows) -- AppWorld PI diagnostics on train90 + dev ONLY.

WHY A SEPARATE TABLE RATHER THAN A FOOTNOTE. Table~\\ref{tab:appworld_pi_metrics} pools whatever
splits each model happens to have finished, so its rows rest on DIFFERENT bases: 296 tasks for the
trajectory arms, 244 for 4B-Instruct's hint arms, 250 for 30B-A3B's, 147 for 4B-Thinking's. The
caption has to warn "read it within a block, since a smaller, easier pool lifts every arm in it" --
which is exactly the comparison a diagnostics table exists to support. Restricting to the two splits
every model completed makes all 42 rows share one base: 146 tasks (train90 90 + dev 56).

THE UID INTERSECTION IS PER CELL, THEN CONCATENATED -- the same rule the by-outcome divergence analysis used
and what reproduces the published pooled `d` to 4 decimals. Intersecting across the pooled frame
instead would drop any rollout missing from either split for unrelated reasons.

D AGGREGATION (--agg, see pi_src_lib.agg_divergence): token_pooled (default) is the token-weighted mean
over all rows of all rollouts and is what produced the published table; traj_mean is the paper's
equation for D (per-rollout token mean, then the mean over rollouts).

4B-THINKING'S TRAJECTORY ARMS MUST COME FROM THE thinkfix CELLS. Without that adapter fix the scorer
drops 80.6% of a Thinking trajectory's context (717 tokens instead of 30,287) and inflates divergence
roughly fourfold.

UTILITY IS DELTA-pass@1, paired per task. Layout (the released dataset bundle):
  <utility-root>/<student>/<arm>/trajectories.jsonl            hinted arm, n=4 per task
  <utility-root>/<student>/nohint_passk128/trajectories.jsonl  the no-hint baseline
Every published row sits on the full train90+dev base. A row whose arm covers fewer tasks is marked with
its task count. Task ids are split-specific, so one file per arm holding both splits gives exactly the
per-split deltas concatenated. The dataset's labels cover train90 only, so a rerun
on them covers the train90 subset.

THE BASELINE IS THE n=128 PASS@K CORPUS, not a fresh n=4 draw: the pass@k corpus IS the no-hint arm at
this sampling profile, so the baseline side is far less noisy than the n=4 hinted side. Delta is paired
per task over the arm's own covered tasks.

NO ERROR-STEP BLOCK: those labels were drawn on the pooled sample, not per split.

Usage:
  python build_appworld_traindev_pi_table.py [--grid DIR] [--utility-root DIR] [--agg token_pooled] \
      [--out outputs/build_appworld_traindev_pi_table/appworld_traindev_pi_metrics.tex]
"""

import argparse
import collections
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pi_src_lib as S  # noqa: E402

GRID = None  # --grid
UTIL = None  # --utility-root
AGG = "token_pooled"  # --agg
SPLITS = ("train90", "dev")

MODELS = [("4bi", "Qwen3-4B-Instruct-2507"), ("4bt", "Qwen3-4B-Thinking-2507"), ("30bi", "Qwen3-30B-A3B-Instruct-2507")]

LAYOUT = [
    (
        "GLM-5.2",
        [
            ("Action", "glm52_actions"),
            ("Action + Obs.", "glm52_raw_no_think"),
            ("Action + Obs. + Thou.", "glm52_raw"),
            ("On-policy skill", "anlz_glm"),
            ("Summarised ref.", "summ_glm"),
            ("Checklist", "chk_glm"),
        ],
    ),
    (
        "Qwen3.7-Plus",
        [
            ("Action", "qwen37_actions"),
            ("Action + Obs.", "qwen37_raw_no_think"),
            ("Action + Obs. + Thou.", "qwen37_raw"),
            ("On-policy skill", "anlz_qwn"),
            ("Summarised ref.", "summ_qwn"),
            ("Checklist", "chk_qwn"),
        ],
    ),
    ("Self", [("On-policy skill", "anlz_self"), ("Summarised ref.", "summ_self")]),
]

COLS = ["uid", "pi", "task_id", "kl", "d", "dsigned", "o50", "n_tokens"]


def traj_cell(split, tag):
    # 4bt's trajectory rows are only valid from the thinkfix cell; the others have one name.
    return f"appworld_{split}_{tag}_thinkfix" if tag == "4bt" else f"appworld_{split}_{tag}"


def hint_cell(split, tag):
    return f"appworld_hint128_{split}_{tag}"


def collect(cells):
    """Per-cell uid intersection across that cell's arms, then concatenate the cells."""
    parts, ntask, nroll = {}, set(), 0
    for c in cells:
        p = f"{GRID}/{c}/rows.parquet"
        if not os.path.exists(p):
            print(f"  [skip] {c}: no rows.parquet", file=sys.stderr)
            continue
        have = set(pq.read_schema(p).names)  # released cells carry no o50 column
        rw = pd.read_parquet(p, columns=[c_ for c_ in COLS if c_ in have])
        pis = list(rw.pi.unique())
        common = set.intersection(*[set(rw[rw.pi == q].uid) for q in pis])
        rw = rw[rw.uid.isin(common)]
        nroll += len(common)
        ntask |= set(rw.task_id.unique())
        for pi, b in rw.groupby("pi"):
            parts.setdefault(str(pi), []).append(b)
    out = {}
    for pi, ps in parts.items():
        b = pd.concat(ps)
        out[pi] = dict(
            kl=S.agg_divergence(b, "kl", AGG),
            d=S.agg_divergence(b, "d", AGG),
            ds=S.agg_divergence(b, "dsigned", AGG),
            o50=S.agg_divergence(b, "o50", AGG) if "o50" in b.columns else None,
            ntask=int(b.task_id.nunique()),
            nroll=int(b.uid.nunique()),
        )
    return out, nroll, len(ntask)


# ---------------------------------------------------------------------------------------------
# utility
# ---------------------------------------------------------------------------------------------
BASELINE_ARM = "nohint_passk128"


def rollouts(path):
    """task_id -> [0/1,...]. `reward`>0 or `success`>0; both conventions appear in these dirs."""
    if not os.path.exists(path):
        return None
    per = collections.defaultdict(list)
    for line in open(path, errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        v = r.get("reward", r.get("success", 0))
        try:
            v = float(v or 0)
        except Exception:
            v = 0.0
        per[str(r.get("task_id"))].append(1 if v > 0 else 0)
    return per or None


def arm_path(tag, arm):
    return f"{UTIL}/{tag}/{arm}/trajectories.jsonl"


def sign_test(diffs):
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))


def utility(tag):
    """{arm: d1 / sign-test p / n tasks} and the baseline's task count (the full-coverage base)."""
    base = rollouts(arm_path(tag, BASELINE_ARM))
    rate = lambda per, t: sum(per[t]) / len(per[t])
    out = {}
    if base is None:
        return out, 0
    for _, arms in LAYOUT:
        for _, arm in arms:
            per = rollouts(arm_path(tag, arm))
            if per is None:
                continue
            ts = sorted(set(per) & set(base))
            diffs = [rate(per, t) - rate(base, t) for t in ts]
            if diffs:
                out[arm] = dict(d1=float(np.mean(diffs)), p1=sign_test(diffs), n=len(ts))
    return out, len(base)


def sg(v):
    return f"$+${v:.4f}" if v >= 0 else f"$-${abs(v):.4f}"


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--grid", default=None, help="scored cells (default: $PI_GRID_ROOT or ./outputs/pi_cells/appworld)")
ap.add_argument("--utility-root", default=None, help="utility rollouts (default: ./outputs/utility/appworld)")
ap.add_argument("--splits", default=",".join(SPLITS), help="splits whose divergence cells are pooled")
ap.add_argument(
    "--agg",
    choices=S.AGG_CHOICES,
    default="token_pooled",
    help="token_pooled = the published numbers; traj_mean = the paper's per-trajectory equation for D",
)
ap.add_argument("--out", default=None, help="write the LaTeX rows here (default: stdout)")
args = ap.parse_args()
GRID = args.grid or S.grid_root("appworld")
UTIL = args.utility_root or os.path.join("outputs", "utility", "appworld")
AGG = args.agg
SPLITS = tuple(args.splits.split(","))

rows, notes = [], []
for tag, full in MODELS:
    T, tn, tt = collect([traj_cell(s, tag) for s in SPLITS])
    H, hn, ht = collect([hint_cell(s, tag) for s in SPLITS])
    D = {**T, **H}
    if not D:
        print(f"  [skip] {tag}: nothing collected", file=sys.stderr)
        continue
    # UBASE = tasks in the no-hint baseline (90 train90 + 57 dev = 147 for the published table)
    Uv, UBASE = utility(tag)
    src = " \\emph{(traj.\\ arms from \\texttt{thinkfix})}" if tag == "4bt" else ""
    best = max((Uv[a]["d1"] for _, arms in LAYOUT for _, a in arms if a in Uv), default=None)
    rows.append(f"      \\multicolumn{{7}}{{l}}{{\\textbf{{{full}}}{src}}} \\\\")
    for si, (source, arms) in enumerate(LAYOUT):
        rows.append(f"      \\multirow{{{len(arms)}}}{{*}}{{\\hspace{{1em}}{source}}}")
        for label, arm in arms:
            v = D.get(arm)
            if v is None:
                rows.append(f"        & {label:<22}& --- & --- & --- & --- & --- \\\\")
                continue
            u = Uv.get(arm)
            if u is None:
                c1 = "---"
            else:
                c1 = "\\textbf{" + sg(u["d1"]) + "}" if u["d1"] == best else sg(u["d1"])
                c1 += "$^{*}$" if u["p1"] < 0.05 else ""
                # An arm missing a split would average over fewer tasks; say so rather than let it sit
                # unmarked beside full-base ones.
                if u["n"] != UBASE:
                    c1 += f" \\tiny({u['n']})"
            base = tt if arm in T else ht
            nn = f" \\tiny({v['ntask']})" if v["ntask"] != base else ""
            o50 = "---" if v["o50"] is None else f"{v['o50']:.3f}"
            rows.append(
                f"        & {label:<22}& {c1} & {v['kl']:.4f} & {v['d']:.4f} " f"& {sg(v['ds'])} & {o50}{nn} \\\\"
            )
        if si < len(LAYOUT) - 1:
            rows.append("      \\addlinespace")
    rows.append("      \\midrule")
    notes.append(f"% {tag}: traj {tn} rollouts / {tt} tasks; hint {hn} rollouts / {ht} tasks")
if rows and rows[-1] == "      \\midrule":
    rows[-1] = "      \\bottomrule"
notes.append(f"% D aggregation: {AGG}")
if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
else:
    print("\n".join(rows))
for n in notes:
    print(n, file=sys.stderr)

#!/usr/bin/env python
"""GLM-8 hint-variance report: how much do d and Δd⁻_err move with WHICH GLM-5.2 trajectory is the hint?

Both metrics use the published estimators exactly:
  d        np.average(d, weights=n_tokens) over every scored row of the task's 128 student rollouts,
           restricted to rollouts common to every arm (alf_pi_divergence_block.py).
  Δd⁻_err  per_step() -> baseline b = mean d⁻ over ALL rollouts of each (arm, task, step); then over the
           judge-flagged steps of LOST labelled rollouts, the mean of d⁻ − b, each flagged step weighted
           equally (pi_error_step_balanced.py: frame + arm_metrics["derrF"]).
Pooled values over the 20 tasks use the same definitions over all their rows / flagged steps.

The 9th arm glm_ref is the published hint, and the published cap24k cell's glm_raw arm on the same 20
tasks is re-derived here as a pipeline check (same hint for every task whose render is under 24k tokens).
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

LABELS = str(C.LABELS_4BI)
CAP24K = str(C.CAP24K_4BI)
COLS = ["pi", "task_id", "uid", "success", "step", "channel", "n_tokens", "kl", "d", "dsigned"]


def remap(rows: pd.DataFrame, index: list) -> pd.DataFrame:
    """Scorer uid = task#<line in corpus_20>; labels use task#<line in the full dump>."""
    k = rows["uid"].str.rsplit("#", n=1).str[1].astype(int)
    rows = rows.copy()
    rows["uid"] = rows["task_id"] + "#" + pd.Series(np.asarray(index)[k.to_numpy()], index=rows.index).astype(str)
    return rows


def common(rows: pd.DataFrame) -> pd.DataFrame:
    pairs = rows[["uid", "pi"]].drop_duplicates()
    cov = pairs.groupby("uid").pi.size()
    return rows[rows.uid.isin(set(cov[cov == rows.pi.nunique()].index))]


def d_by(rows: pd.DataFrame, keys: list) -> pd.Series:
    """np.average(d, weights=n_tokens) per group, vectorised."""
    m = rows.assign(_m=rows["d"] * rows["n_tokens"]).groupby(keys, observed=True)[["_m", "n_tokens"]].sum()
    return m["_m"] / m["n_tokens"]


def derr_frame(rows: pd.DataFrame, labels: dict) -> pd.DataFrame:
    ps = S.per_step(rows)
    ps["_b"] = ps.groupby(["pi", "task_id", "step"], observed=True)["dminus"].transform("mean")
    ps = ps[ps.uid.isin(set(labels))].copy()
    ps["win"] = ps.uid.map(ps.groupby("uid").success.first() > 0)
    flagged = pd.DataFrame([(u, int(s)) for u, ss in labels.items() for s in set(ss)], columns=["uid", "step"])
    ps = ps.merge(flagged.assign(hit=True), on=["uid", "step"], how="left")
    ps["hit"] = ps["hit"].astype("boolean").fillna(False).astype(bool)
    ps["dev"] = ps["dminus"] - ps["_b"]
    return ps[ps.hit & ~ps.win]


def spread(s: pd.Series) -> dict:
    return {
        "mean": s.mean(),
        "sd": s.std(ddof=1),
        "min": s.min(),
        "max": s.max(),
        "range": s.max() - s.min(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--w",
        default=str(C.PI_STUDY_ROOT / "pi_glm8"),
        help="study dir: task_ids_20.json + corpus_20_index.json (pi_glm8_prepare.py), hints/manifest.json "
        "(pi_glm8_build_hints.py), score/rows.parquet (run_pi_glm8_score.sh)",
    )
    ap.add_argument("--out", default=str(C.default_out("pi_glm8_report")))
    args = ap.parse_args()
    W = Path(args.w)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks = json.load(open(W / "task_ids_20.json"))
    labels = json.load(open(LABELS))
    rows = remap(
        pd.read_parquet(W / "score/rows.parquet", columns=COLS),
        json.load(open(W / "corpus_20_index.json")),
    )
    rows = common(rows)
    slots = [f"glm_s{j}" for j in range(8)]
    arms = slots + ["glm_ref"]
    assert set(rows.pi.unique()) == set(arms), rows.pi.unique()

    cap = pd.read_parquet(CAP24K, columns=COLS, filters=[("task_id", "in", tasks), ("pi", "==", "glm_raw")])
    cap = cap[cap.uid.isin(set(rows.uid))].assign(pi="pub_cap24k")

    both = pd.concat([rows, cap], ignore_index=True)
    d_task = d_by(both, ["pi", "task_id"]).unstack(0)  # task x arm
    d_pool = d_by(both, ["pi"])
    fe = derr_frame(both, labels)
    e_task = fe.groupby(["task_id", "pi"]).dev.mean().unstack(1)
    e_pool = fe.groupby("pi").dev.mean()
    n_flag = fe[fe.pi == "glm_s0"].groupby("task_id").size()

    man = json.load(open(W / "hints/manifest.json"))["tasks"]
    per_task = []
    for t in tasks:
        m = man[t]
        chars = pd.Series([x["chars"] for x in m], index=slots)
        seqs = {tuple(x["actions"]) for x in m}
        per_task.append(
            {
                "task_id": t,
                "flagged_steps": int(n_flag.get(t, 0)),
                "turns_min": min(x["num_turns"] for x in m),
                "turns_max": max(x["num_turns"] for x in m),
                "chars_min": int(chars.min()),
                "chars_max": int(chars.max()),
                "distinct_action_seqs": len(seqs),
                **{f"d_{k}": v for k, v in spread(d_task.loc[t, slots]).items()},
                **{f"e_{k}": v for k, v in spread(e_task.loc[t, slots]).items()},
                "d_ref": d_task.loc[t, "glm_ref"],
                "d_pub": d_task.loc[t, "pub_cap24k"],
                "e_ref": e_task.loc[t, "glm_ref"],
                "e_pub": e_task.loc[t, "pub_cap24k"],
                "rho_d_chars": d_task.loc[t, slots].rank().corr(chars.rank()),  # Spearman (no scipy here)
            }
        )
    pt = pd.DataFrame(per_task).set_index("task_id")

    # variance decomposition over the 20 x 8 grid: task vs slot-within-task
    def icc(mat: pd.DataFrame) -> dict:
        within = mat.var(axis=1, ddof=1).mean()
        between = mat.mean(axis=1).var(ddof=1)
        return {
            "within_task_sd": within**0.5,
            "between_task_sd": between**0.5,
            "task_share": between / (between + within),
        }

    summary = {
        "d_pooled_per_slot": d_pool[slots].to_dict(),
        "d_pooled_ref": d_pool["glm_ref"],
        "d_pooled_pub": d_pool["pub_cap24k"],
        "d_pooled_slot_spread": spread(d_pool[slots]),
        "e_pooled_per_slot": e_pool[slots].to_dict(),
        "e_pooled_ref": e_pool["glm_ref"],
        "e_pooled_pub": e_pool["pub_cap24k"],
        "e_pooled_slot_spread": spread(e_pool[slots]),
        "d_icc": icc(d_task[slots]),
        "e_icc": icc(e_task[slots]),
        "check_ref_vs_pub_max_abs": {
            "d": float((pt.d_ref - pt.d_pub).abs().max()),
            "e": float((pt.e_ref - pt.e_pub).abs().max()),
        },
        "median_rho_d_vs_hint_chars": float(pt.rho_d_chars.median()),
    }
    pt.to_csv(out / "per_task.csv")
    d_task.to_csv(out / "d_task_by_arm.csv")
    e_task.to_csv(out / "derr_task_by_arm.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(
        pt[
            [
                "flagged_steps",
                "turns_min",
                "turns_max",
                "distinct_action_seqs",
                "d_mean",
                "d_sd",
                "d_range",
                "d_ref",
                "d_pub",
                "e_mean",
                "e_sd",
                "e_range",
                "e_ref",
                "e_pub",
            ]
        ]
        .round(4)
        .to_string()
    )
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()

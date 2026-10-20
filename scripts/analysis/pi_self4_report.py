#!/usr/bin/env python
"""Self-hint variance report: d and Δd⁻_err across k self-winning raw hints per task (pi_self4_prepare.py).

Estimators are pi_glm8_report's (the published ones). Several design directories can be pooled (their
task sets must be disjoint), and --k restricts every task to its first k hint slots, so a nested 16-hint
design also yields the 4- and 8-hint numbers on the same tasks and rollouts.

Adds a like-for-like comparison with the GLM-8 study on the tasks both cover, with min(k, 8) GLM hints, read
from --glm-report: pi_glm8_report.py's output dir (default ./outputs/pi_glm8_report; run that report first).

    python pi_self4_report.py --w $PI_STUDY_ROOT/pi_self4 --k 4
    python pi_self4_report.py --w $PI_STUDY_ROOT/pi_self16 --w $PI_STUDY_ROOT/pi_self8x --k 8
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alf_study_common as C  # noqa: E402
from pi_glm8_report import (  # noqa: E402
    COLS,
    LABELS,
    common,
    d_by,
    derr_frame,
    remap,
    spread,
)


def done_chunks(w: Path) -> list:
    """Finished chunk dirs: those the scorer marked DONE. A published copy of a finished study (the data
    bundle) carries no DONE markers, so when a dir has none at all every chunk with rows.parquet counts."""
    done = sorted(p.parent for p in (w / "score").glob("chunk_*/DONE"))
    if not done and not list((w / "score").glob("chunk_*/FAILED")):
        done = sorted(p.parent for p in (w / "score").glob("chunk_*/rows.parquet"))
    return done


def load_chunks(w: Path) -> pd.DataFrame:
    parts = []
    for d in done_chunks(w):
        c = d.name
        idx = json.load(open(w / "chunks" / f"{c}_index.json"))
        parts.append(remap(pd.read_parquet(d / "rows.parquet", columns=COLS), idx))
    return pd.concat(parts, ignore_index=True)


def glm_report_dir() -> Path:
    """pi_glm8_report.py's default output (run that report first)."""
    local = C.default_out("pi_glm8_report")
    if not (local / "d_task_by_arm.csv").exists():
        raise FileNotFoundError(f"{local}/d_task_by_arm.csv: run pi_glm8_report.py first, or pass --glm-report")
    return local


def read_task_table(d: Path, stem: str) -> pd.DataFrame:
    """A task x arm table written by pi_glm8_report.py (<stem>.csv), or its published .parquet copy."""
    if (d / f"{stem}.csv").exists():
        return pd.read_csv(d / f"{stem}.csv", index_col=0)
    t = pd.read_parquet(d / f"{stem}.parquet")  # the published copy keeps task_id as a column
    return t.set_index("task_id") if "task_id" in t.columns else t


def spread_table(d_task: pd.DataFrame, e_task: pd.DataFrame, slots: list) -> dict:
    sd_d, sd_e = d_task[slots].std(axis=1, ddof=1), e_task[slots].std(axis=1, ddof=1)
    rg_d = d_task[slots].max(axis=1) - d_task[slots].min(axis=1)
    rg_e = e_task[slots].max(axis=1) - e_task[slots].min(axis=1)

    def share(mat):
        within, between = mat.var(axis=1, ddof=1).mean(), mat.mean(axis=1).var(ddof=1)
        return between / (between + within)

    return {
        "n_tasks": len(d_task),
        "d_mean": float(d_task[slots].values.mean()),
        "d_sd_median": float(sd_d.median()),
        "d_sd_max": float(sd_d.max()),
        "d_range_median": float(rg_d.median()),
        "d_range_max": float(rg_d.max()),
        "d_task_share": float(share(d_task[slots])),
        "e_mean": float(np.nanmean(e_task[slots].values)),
        "e_sd_median": float(sd_e.median()),
        "e_sd_max": float(sd_e.max()),
        "e_range_median": float(rg_e.median()),
        "e_range_max": float(rg_e.max()),
        "e_task_share": float(share(e_task[slots].dropna())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", action="append", default=None, help="design dir(s); default pi_self4")
    ap.add_argument("--k", type=int, default=None, help="use each task's first k hint slots (default all)")
    ap.add_argument(
        "--glm-report",
        default=None,
        help="pi_glm8_report.py output dir (d_task_by_arm / derr_task_by_arm); default ./outputs/pi_glm8_report",
    )
    ap.add_argument("--out", default=None, help="default ./outputs/pi_self4_report/<first --w name>/report_k<k>")
    args = ap.parse_args()
    ws = [Path(w) for w in (args.w or [str(C.PI_STUDY_ROOT / "pi_self4")])]
    G = Path(args.glm_report) if args.glm_report else glm_report_dir()

    labels = json.load(open(LABELS))
    tasks, man, texts, parts = [], {}, {}, []
    for w in ws:
        tasks += json.load(open(w / "task_ids.json"))
        m = json.load(open(w / "hints/manifest.json"))
        assert not set(m) & set(man), f"task sets overlap between design dirs ({w})"
        man.update(m)
        files = sorted((w / "hints").glob("hints_self_s*.json"), key=lambda p: int(p.stem.rsplit("s", 1)[1]))
        hs = [json.load(open(f)) for f in files]
        texts.update({t: [h[t]["task"] for h in hs] for t in m})
        parts.append(load_chunks(w))
    rows = pd.concat(parts, ignore_index=True)
    slots = sorted(rows.pi.unique(), key=lambda s: int(s.rsplit("s", 1)[1]))
    if args.k:
        slots = slots[: args.k]
    rows = common(rows[rows.pi.isin(slots)])  # every kept rollout is scored under all k slots
    done_tasks = sorted(rows.task_id.unique())
    k = len(slots)
    out = Path(args.out or C.default_out("pi_self4_report") / ws[0].name / f"report_k{k}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"scored tasks {len(done_tasks)}/{len(tasks)}, k={k} slots")

    d_task = d_by(rows, ["pi", "task_id"]).unstack(0)[slots]
    fe = derr_frame(rows, labels)
    e_task = fe.groupby(["task_id", "pi"]).dev.mean().unstack(1).reindex(index=d_task.index, columns=slots)
    d_pool, e_pool = d_by(rows, ["pi"])[slots], fe.groupby("pi").dev.mean().reindex(slots)
    n_flag = fe[fe.pi == slots[0]].groupby("task_id").size()

    identical_text_pairs = sum(
        texts[t][a] == texts[t][b] for t in done_tasks for a, b in itertools.combinations(range(k), 2)
    )
    same, diff = {"d": [], "e": []}, {"d": [], "e": []}
    for t in done_tasks:
        acts = [tuple(h["actions"]) for h in man[t]["hints"][:k]]
        for a, b in itertools.combinations(range(k), 2):
            tgt = same if acts[a] == acts[b] else diff
            tgt["d"].append(abs(d_task.loc[t, slots[a]] - d_task.loc[t, slots[b]]))
            if not np.isnan(e_task.loc[t, slots[a]]) and not np.isnan(e_task.loc[t, slots[b]]):
                tgt["e"].append(abs(e_task.loc[t, slots[a]] - e_task.loc[t, slots[b]]))

    summary = {
        "k": k,
        "all": spread_table(d_task, e_task, slots),
        "pooled_per_slot": {"d": d_pool.to_dict(), "e": e_pool.to_dict()},
        "pooled_slot_spread": {"d": spread(d_pool), "e": spread(e_pool)},
        "pair_gaps_same_actions": {m: float(np.median(v)) if v else None for m, v in same.items()},
        "pair_gaps_diff_actions": {m: float(np.median(v)) if v else None for m, v in diff.items()},
        "n_pairs": {
            "same": len(same["d"]),
            "diff": len(diff["d"]),
            "identical_text": identical_text_pairs,
        },
        "tasks_without_flagged_lost_steps": int(e_task[slots[0]].isna().sum()),
    }

    # like-for-like vs GLM on the shared tasks, min(k, 8) hints each
    gd, ge = read_task_table(G, "d_task_by_arm"), read_task_table(G, "derr_task_by_arm")
    shared = sorted(set(gd.index) & set(d_task.index))
    gk = [f"glm_s{j}" for j in range(min(k, 8))]
    if shared:
        summary["shared_tasks"] = {
            "n": len(shared),
            "self": spread_table(d_task.loc[shared], e_task.loc[shared], slots[: len(gk)]),
            f"glm_{len(gk)}": spread_table(gd.loc[shared], ge.loc[shared], gk),
        }
    wins = pd.Series({t: man[t]["wins"] for t in done_tasks})
    per = pd.DataFrame(
        {
            "wins": wins,
            "flagged": n_flag.reindex(done_tasks).fillna(0).astype(int),
            "d_mean": d_task.mean(axis=1),
            "d_sd": d_task.std(axis=1, ddof=1),
            "e_mean": e_task.mean(axis=1),
            "e_sd": e_task.std(axis=1, ddof=1),
        }
    )
    bins = pd.cut(per.wins, [4, 16, 64, 127], labels=["5-16", "17-64", "65-127"])
    summary["by_wins"] = (
        per.groupby(bins, observed=True)[["d_mean", "d_sd", "e_mean", "e_sd", "flagged"]]
        .median()
        .round(4)
        .to_dict(orient="index")
    )
    per.to_csv(out / "per_task.csv")
    d_task.to_csv(out / "d_task_by_slot.csv")
    e_task.to_csv(out / "derr_task_by_slot.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()

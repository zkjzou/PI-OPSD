#!/usr/bin/env python
"""Per-task PI selection over the 8 PI sources for OPSD on train1600: random / top-3 by d / top-3 by Δd⁻_err.

Per (task, source), d and Δd⁻_err are the published estimators (pi_glm8_report.d_by / derr_frame) on the
untrained Qwen3-4B-Instruct-2507's rollouts, scored with that source's hint in context:
  offline 200  N=128 rollouts, natural-mix judge labels (~32 per task)   cap24k grid + the multi-author dirs
  rem1400      N=16 rollouts, judge labels on <=8 LOST rollouts per task  pi_mm_rem1400
A task's candidates are the sources whose won-trajectory hint was scored on it. Each arm fixes ONE hint per
task for the whole run (the draw is seeded by seed, arm and task):
  random8    uniform over the candidates
  top3_d     uniform over the 3 candidates with the highest d
  top3_derr  uniform over the 3 candidates with the highest Δd⁻_err; a task with no labelled lost rollout
             (Δd⁻_err undefined) keeps its random8 draw
The hint text is exactly the text that was scored (cap24k render for glm_raw / qwen37_raw, the uncapped won
trajectory for the frontier authors).

    python pi_mix_select_build.py --out outputs/pi_mix_select     # --ref defaults to the bundle's cap24k renders
    # same-source 8 slots (pi_flash8_prepare.py):
    python pi_mix_select_build.py --out outputs/pi_flash8_select \
      --sources glm53flash,glm53flash_k1,...,glm53flash_k7 \
      --offline-dir $PI_STUDY_ROOT/pi_mm_glm53flash200 --offline-dir outputs/pi_f8/pi_f8_offline200 \
      --rem-dir $PI_STUDY_ROOT/pi_mm_rem1400 --rem-dir outputs/pi_f8/pi_f8_rem1400     # run_flash8_score_build.sh

The study dirs (default under $PI_STUDY_ROOT) need their hints/ (the prepare scripts write them) as well as
the scores; the bundle's copies carry both. glm_raw / qwen37_raw texts come from --ref (default
alfworld/hints/{glm-5.2,qwen3.7-plus}/hints_traj_withthink_cap24k.json, the renders the cap24k cell scored; the
rem1400 dir's own copies of them are identical on all 1400 tasks and are not in the bundle). The published draws
are alfworld/selection/hints_mix{,_s1,_s2,_s3}/ (--seed 0..3; the arms' replicate seed=n uses --seed n-1).
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alf_study_common as C  # noqa: E402
from pi_glm8_report import CAP24K, COLS, LABELS, d_by, derr_frame  # noqa: E402
from pi_self4_report import load_chunks  # noqa: E402

REPO = C.repo_root()
sys.path.insert(0, str(REPO))
from opsd.pi.artifact import save_hint_artifact  # noqa: E402
from opsd.pi.base import HintRecord  # noqa: E402

N = str(C.PI_STUDY_ROOT)
SOURCES = [
    "glm_raw",
    "qwen37_raw",
    "qwen38max",
    "kimik3",
    "glm53",
    "minimaxm3",
    "dsv41flash",
    "glm53flash",
]
REFS = ["glm_raw", "qwen37_raw"]
OFFLINE = [f"{N}/pi_mm_offline200", f"{N}/pi_mm_topup200", f"{N}/pi_mm_glm53flash200"]
REM = f"{N}/pi_mm_rem1400"
REM_LABELS = str(C.LABELS_REM1400_4BI)
TRAIN1600 = str(C.TRAIN1600)
ARMS = ["random8", "top3_d", "top3_derr"]


def task_metrics(rows: pd.DataFrame, labels: dict) -> pd.DataFrame:
    d = d_by(rows, ["pi", "task_id"]).rename("d")
    e = derr_frame(rows, labels).groupby(["pi", "task_id"]).dev.mean().rename("derr")
    return pd.concat([d, e], axis=1).reset_index().rename(columns={"pi": "source"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--ref",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="published cap24k renders (default: glm_raw / qwen37_raw from the bundle, for the ones in --sources); "
        "a --ref overrides only the source it names",
    )
    ap.add_argument("--offline-dir", action="append", default=None, help="multi-author offline-200 dirs")
    ap.add_argument(
        "--sources",
        default=",".join(SOURCES),
        help="comma list of PI names (hint-file stems); e.g. glm53flash,glm53flash_k1..k7 for 8 same-source slots",
    )
    ap.add_argument("--rem-dir", action="append", default=None, help="rem1400 scoring dir(s)")
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="omit tasks with no scored hint (a same-source arm whose author never solved them) instead of failing",
    )
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--random-only",
        action="store_true",
        help="write only hints_mix_random8.json, before the rem1400 scoring ends: the draw needs just each "
        "task's candidates, and every rem1400 hint is being scored, so it equals the full build's draw",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = sorted(json.load(open(TRAIN1600)))
    sources = args.sources.split(",")
    refs = [s for s in REFS if s in sources]
    offline, rems = args.offline_dir or OFFLINE, args.rem_dir or [REM]

    # 1. the exact scored hint texts per (source, task)
    text = {}
    # Merge PER NAME: a --ref replaces only the source it names; every other ref source keeps its cap24k
    # default. (Defaults used to apply only when --ref was omitted entirely, so passing one --ref silently
    # left the other ref source with no text -- has_hint False on every task -- and it dropped out of the
    # candidate pools, changing the random8/top3 draws without an error.)
    given = {spec.split("=", 1)[0] for spec in args.ref or []}
    for spec in list(args.ref or []) + [f"{n}={C.CAP24K_REFS[n]}" for n in refs if n not in given]:
        name, path = spec.split("=", 1)
        for t, v in json.load(open(path)).items():
            if t != "__schema__":
                text.setdefault((name, t), v["task"] if isinstance(v, dict) else v)
    for w in offline + rems:
        for f in sorted(Path(w, "hints").glob("hints_*.json")):
            name = f.stem[len("hints_") :]
            for t, v in json.load(open(f)).items():
                if t != "__schema__" and name in sources:
                    text.setdefault((name, t), v["task"] if isinstance(v, dict) else v)

    # 2. per-(task, source) metrics: offline 200 (cap24k refs + author dirs) and rem1400
    pub = [pd.read_parquet(CAP24K, columns=COLS, filters=[("pi", "in", refs)])] if refs else []
    rows200 = pd.concat(pub + [load_chunks(Path(w)) for w in offline], ignore_index=True)
    rows200 = rows200[rows200.pi.isin(sources)]
    m200 = task_metrics(rows200, json.load(open(LABELS)))
    # random-only: candidates = every rem1400 hint (all of them are being scored); no metrics yet
    if args.random_only:
        man = {}
        for w in rems:
            man.update(json.load(open(Path(w, "hints", "manifest.json"))))
        m1400 = pd.DataFrame(
            [(s, t) for s, v in man.items() if s in sources for t in v["tasks"]],
            columns=["source", "task_id"],
        ).assign(d=float("nan"), derr=float("nan"))
    else:
        rows1400 = pd.concat([load_chunks(Path(w)) for w in rems], ignore_index=True)
        m1400 = task_metrics(rows1400[rows1400.pi.isin(sources)], json.load(open(REM_LABELS)))
    met = pd.concat([m200.assign(pool="offline200"), m1400.assign(pool="rem1400")], ignore_index=True)
    assert not met.duplicated(["source", "task_id"]).any()
    met["has_hint"] = [(s, t) in text for s, t in zip(met.source, met.task_id)]
    met.to_csv(out / "per_task_metrics.csv", index=False)
    by_task = {t: g for t, g in met[met.has_hint].groupby("task_id")}
    missing = [t for t in tasks if t not in by_task]
    assert args.allow_missing or not missing, f"{len(missing)} train tasks have no scored hint, e.g. {missing[:3]}"
    tasks = [t for t in tasks if t in by_task]  # --allow-missing: these get no hint (skipped in OPSD)

    # 3. one draw per arm and task
    picks = {a: {} for a in ARMS}
    fallback = 0
    for t in tasks:
        g = by_task[t].set_index("source")
        cands = sorted(g.index)

        def draw(arm: str, pool: list) -> str:
            return random.Random(f"{args.seed}:{arm}:{t}").choice(sorted(pool))

        picks["random8"][t] = draw("random8", cands)
        if args.random_only:
            continue
        picks["top3_d"][t] = draw("top3_d", list(g.d.sort_values(ascending=False).index[: args.k]))
        e = g.derr.dropna()
        if e.empty:
            picks["top3_derr"][t] = picks["random8"][t]
            fallback += 1
        else:
            picks["top3_derr"][t] = draw("top3_derr", list(e.sort_values(ascending=False).index[: args.k]))

    # 4. artifacts + summary
    summary = {
        "tasks": len(tasks),
        "tasks_without_hint": missing,
        "top3_derr_fallback_to_random8": fallback,
        "arms": {},
    }
    cand_n = collections.Counter(len(by_task[t]) for t in tasks)
    summary["candidates_per_task"] = dict(sorted(cand_n.items()))
    arms = ["random8"] if args.random_only else ARMS
    for a in arms:
        recs = {}
        for t in tasks:
            s = picks[a][t]
            row = by_task[t].set_index("source").loc[s]
            meta = {"arm": a, "source": s, "pool": row.pool}
            if pd.notna(row.d):
                meta["d"] = round(float(row.d), 5)
            if pd.notna(row.derr):
                meta["derr"] = round(float(row.derr), 5)
            recs[t] = HintRecord(task=text[(s, t)], meta=meta)
        save_hint_artifact(str(out / f"hints_mix_{a}.json"), recs)
        chosen = pd.DataFrame([{"task_id": t, "source": picks[a][t]} for t in tasks]).merge(
            met, on=["task_id", "source"]
        )
        summary["arms"][a] = {
            "source_share": chosen.source.value_counts(normalize=True).round(3).to_dict(),
            "mean_d": round(float(chosen.d.mean()), 4),
            "mean_derr": round(float(chosen.derr.mean()), 4),
            "mean_chars": round(float(sum(len(text[(picks[a][t], t)]) for t in tasks) / len(tasks))),
        }
    summary["same_source_share"] = {
        f"{x}~{y}": round(sum(picks[x][t] == picks[y][t] for t in tasks) / len(tasks), 3)
        for i, x in enumerate(arms)
        for y in arms[i + 1 :]
    }
    name = "summary_random_only.json" if args.random_only else "summary.json"
    (out / name).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

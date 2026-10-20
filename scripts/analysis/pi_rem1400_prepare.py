#!/usr/bin/env python
"""Scoring + labelling inputs for the 8 PI sources on the OTHER 1,400 train1600 tasks (rem1400).

The offline 200 already carry per-(task, source) d and Δd⁻_err; this extends both to the rest of train1600,
so a per-task PI selection (random / top-3 by d / top-3 by Δd⁻_err) can cover the whole OPSD training set.

Student rollouts: the untrained Qwen3-4B-Instruct-2507 rem1400 N=32 dumps (7 shards, same contract as the
train200 N=128 dump), subsampled to --per-task rollouts per task with a fixed seed -> corpus.jsonl, in which
uid = task#<line>.

Hints, rendered exactly as in the offline-200 analysis:
  --ref NAME=PATH     a published render (opsd_hints/v2), e.g. glm_raw = hints_traj_withthink_cap24k.json
  --author NAME=DIR   a frontier author's won trajectory (main pass c*.jsonl + every topup_r*.jsonl), rendered
                      with render_uncapped (each task has at most one won trajectory per author)
Chunks are by task, each with its corpus line index, so scorer rows join the labels (pi_glm8_report.remap).

cell/rows.parquet (pi="corpus", one row per assistant step 1..n, success = max(score) >= 10, which equals the
scorer's `success` on the train200 cell, 3000/3000) lets pi_error_step_label.py sample LOST rollouts while
the scoring runs; check its steps against the scorer rows once those exist.

    python pi_rem1400_prepare.py --out outputs/pi_mm_rem1400 \
      --ref glm_raw=<glm hints_traj_withthink_cap24k.json> --ref qwen37_raw=<qwen37 ...> \
      --author qwen38max=outputs/pi_multimodel/qwen3.8-max --author ...
"""

import argparse
import collections
import glob
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alf_study_common as C  # noqa: E402
from pi_glm8_build_hints import render_uncapped  # noqa: E402
from pi_multimodel_offline_prepare import SCHEMA, TASKS200  # noqa: E402

SHARDS = str(C.REM1400_SHARDS_4BI)
TRAIN1600 = str(C.TRAIN1600)


def n_assistant(resp: str) -> int:
    """Assistant turns in a dumped output_response -- the same split as pi_error_step_label.alf_runs."""
    n = 0
    for i, ch in enumerate(re.split(r"<\|im_start\|>", resp or "")):
        n += i == 0 or ch.startswith("assistant")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--author", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--per-task", type=int, default=16)
    ap.add_argument("--chunks", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shards", default=SHARDS, help="glob of the rem1400 N=32 no-hint student corpus shards")
    ap.add_argument(
        "--hints-only",
        action="store_true",
        help="rebuild hints/ only (e.g. after more top-up rounds); corpus, cell and chunks are left as they are",
    )
    args = ap.parse_args()
    out = Path(args.out)
    for sub in ("hints", "chunks", "cell"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    t1600, t200 = set(json.load(open(TRAIN1600))), set(json.load(open(TASKS200)))
    tasks = sorted(t1600 - t200)
    assert len(tasks) == 1400 and t200 <= t1600, (len(tasks), len(t200 - t1600))
    if not args.hints_only:
        write_corpus(out, tasks, args)
    build_hints(out, tasks, args)


def write_corpus(out: Path, tasks: list, args) -> None:
    by_task = collections.defaultdict(list)
    for f in sorted(glob.glob(args.shards)):
        for line in open(f):
            d = json.loads(line)
            by_task[str((d["env_extras"].get("extra_info") or {}).get("task_id"))].append(line.rstrip("\n"))
    assert set(by_task) == set(tasks), len(set(by_task) ^ set(tasks))
    assert all(len(v) == 32 for v in by_task.values())

    rng = random.Random(args.seed)
    corpus, task_of, cell = [], [], []
    for t in tasks:
        for p in sorted(rng.sample(range(len(by_task[t])), args.per_task)):
            line = by_task[t][p]
            d = json.loads(line)
            sc = d["score"]
            success = int((max(sc) if isinstance(sc, list) else sc) >= 10)
            i = len(corpus)
            corpus.append(line)
            task_of.append(t)
            cell += [(f"{t}#{i}", t, success, k) for k in range(1, n_assistant(d.get("output_response")) + 1)]
    (out / "corpus.jsonl").write_text("\n".join(corpus) + "\n")
    cell = pd.DataFrame(cell, columns=["uid", "task_id", "success", "step"]).assign(pi="corpus")
    cell.to_parquet(out / "cell" / "rows.parquet")
    per_uid = cell.groupby("uid").agg(task_id=("task_id", "first"), success=("success", "first"))
    lost_per_task = per_uid.groupby("task_id").success.apply(lambda s: int((s == 0).sum()))
    print(
        f"corpus: {len(corpus)} rollouts = {len(tasks)} tasks x {args.per_task}; success {per_uid.success.mean():.3f}; "
        f"tasks with no lost rollout {int((lost_per_task == 0).sum())}; "
        f"lost rollouts to label at <=8/task {int(lost_per_task.clip(upper=8).sum())}"
    )

    for c in range(args.chunks):
        ts = set(tasks[c :: args.chunks])
        idx = [i for i, t in enumerate(task_of) if t in ts]
        (out / "chunks" / f"chunk_{c}.jsonl").write_text("\n".join(corpus[i] for i in idx) + "\n")
        (out / "chunks" / f"chunk_{c}_index.json").write_text(json.dumps(idx))
    print(f"{args.chunks} chunks written to {out / 'chunks'}")
    (out / "task_ids.json").write_text(json.dumps(tasks, indent=1))
    meta = {"shards": args.shards, "per_task": args.per_task, "seed": args.seed, "rollouts": len(corpus)}
    (out / "corpus_meta.json").write_text(json.dumps(meta))


def build_hints(out: Path, tasks: list, args) -> None:
    tset, manifest = set(tasks), {}
    for spec in args.ref:
        name, path = spec.split("=", 1)
        src = json.load(open(path))
        hints = {"__schema__": SCHEMA, **{t: src[t] for t in tasks if t in src}}
        manifest[name] = {"source": path}
        (out / "hints" / f"hints_{name}.json").write_text(json.dumps(hints))
        manifest[name].update(hinted_tasks=len(hints) - 1, tasks=sorted(k for k in hints if k != "__schema__"))
    for spec in args.author:
        name, d = spec.split("=", 1)
        hints = {"__schema__": SCHEMA}
        for f in sorted(Path(d).glob("c*.jsonl")) + sorted(Path(d).glob("topup_r*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if r["task_id"] in tset and r["won"] and r["task_id"] not in hints:
                    hints[r["task_id"]] = {
                        "task": render_uncapped(r),
                        "meta": {
                            "pi_format": "traj_withthink",
                            "author": name,
                            "num_turns": r["num_turns"],
                        },
                    }
        (out / "hints" / f"hints_{name}.json").write_text(json.dumps(hints))
        manifest[name] = {
            "source": d,
            "hinted_tasks": len(hints) - 1,
            "tasks": sorted(k for k in hints if k != "__schema__"),
        }
    for name, v in manifest.items():
        print(f"{name}: {v['hinted_tasks']}/{len(tasks)} tasks hinted")
    (out / "hints" / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()

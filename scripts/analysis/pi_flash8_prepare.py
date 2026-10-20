#!/usr/bin/env python
"""Scoring inputs for the SAME-SOURCE 8-slot PI: 8 won GLM-5.3-flash trajectories per train1600 task.

Slot 0 is the author's original won trajectory (run_alf_pi_multimodel.sh, already scored as `glm53flash` in
pi_mm_glm53flash200 and pi_mm_rem1400). Slots 1..K come from the k8 generation dir (N_SAMPLES=7 main pass +
WINS_TARGET=7 top-up): each task's won trajectories in file order (c0..c7, then topup_r1, r2, ...), line order
within a file, the first K of them. Rendered with render_uncapped, exactly like slot 0.

Two scoring dirs, each reusing an existing dir's rollout chunks (symlinked, so the scorer rows join the same
judge labels):
  <out>_offline200   chunks of the train200 N=128 dump (from pi_mm_glm53flash200)
  <out>_rem1400      chunks of the rem1400 N=16 corpus (from pi_mm_rem1400)

    python pi_flash8_prepare.py --gen outputs/pi_multimodel/glm-5.3-flash-k8 \
        --out outputs/pi_f8/pi_f8      # reads the scored pools from $PI_STUDY_ROOT (with their chunk .jsonl)

STATUS: in progress. The same-source 8-slot OPSD arms (f8) are planned, not run.
"""

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alf_study_common as C  # noqa: E402
from pi_glm8_build_hints import render_uncapped  # noqa: E402
from pi_multimodel_offline_prepare import SCHEMA, TASKS200  # noqa: E402

N = str(C.PI_STUDY_ROOT)
POOLS = {
    "offline200": f"{N}/pi_mm_glm53flash200",
    "rem1400": f"{N}/pi_mm_rem1400",
}
TRAIN1600 = str(C.TRAIN1600)
NAME = "glm53flash"


def file_order(p: Path) -> tuple:
    m = re.match(r"(c|topup_r)(\d+)\.jsonl$", p.name)
    return (0 if m.group(1) == "c" else 1, int(m.group(2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="k8 author dir with c*.jsonl + topup_r*.jsonl")
    ap.add_argument("--out", required=True, help="prefix: writes <out>_offline200 and <out>_rem1400")
    ap.add_argument("--k", type=int, default=7, help="new slots per task (slot 0 is already scored)")
    args = ap.parse_args()

    files = sorted(
        (p for p in Path(args.gen).iterdir() if re.match(r"(c|topup_r)\d+\.jsonl$", p.name)),
        key=file_order,
    )
    wins = collections.defaultdict(list)
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r["won"] and len(wins[r["task_id"]]) < args.k:
                wins[r["task_id"]].append(r)
    t1600, t200 = json.load(open(TRAIN1600)), set(json.load(open(TASKS200)))
    have = collections.Counter(min(len(wins[t]), args.k) for t in t1600)
    print(
        f"{len(files)} files; tasks by number of new won trajectories (cap {args.k}):",
        dict(sorted(have.items())),
    )

    for pool, src in POOLS.items():
        out = Path(f"{args.out}_{pool}")
        (out / "hints").mkdir(parents=True, exist_ok=True)
        for link in ("chunks", "corpus.jsonl"):
            if (Path(src) / link).exists() and not (out / link).exists():
                os.symlink(Path(src) / link, out / link)
        tasks = sorted(t for t in t1600 if (t in t200) == (pool == "offline200"))
        manifest = {}
        for k in range(1, args.k + 1):
            name = f"{NAME}_k{k}"
            hints = {"__schema__": SCHEMA}
            for t in tasks:
                if len(wins[t]) >= k:
                    r = wins[t][k - 1]
                    hints[t] = {
                        "task": render_uncapped(r),
                        "meta": {
                            "pi_format": "traj_withthink",
                            "author": NAME,
                            "slot": k,
                            "num_turns": r["num_turns"],
                        },
                    }
            (out / "hints" / f"hints_{name}.json").write_text(json.dumps(hints))
            manifest[name] = {
                "hinted_tasks": len(hints) - 1,
                "tasks": sorted(x for x in hints if x != "__schema__"),
            }
        (out / "hints" / "manifest.json").write_text(json.dumps(manifest, indent=1))
        (out / "task_ids.json").write_text(json.dumps(tasks, indent=1))
        print(f"{out}: {len(tasks)} tasks;", {k: v["hinted_tasks"] for k, v in manifest.items()})


if __name__ == "__main__":
    main()

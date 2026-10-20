#!/usr/bin/env python
"""Offline-metric inputs for the multi-author PI (run_alf_pi_multimodel.sh output) on the 200 offline tasks.

Per author, the hint for a task is that author's WON trajectory rendered raw, the same render as the
published GLM raw arm (hints_traj_withthink): role-tagged [OBSERVATION]/[AGENT] lines, hidden
reasoning_content folded back as <think>...</think> before each turn's visible text, no cap. A task the
author did not solve gets no hint (a failed trajectory is not a reference).

The student rollouts are the N=128 corpus of the 200 tasks, ALL scored: no author hint is a student
rollout, so nothing needs excluding. Chunked by task, with each chunk's full-dump line index, so the
published error labels join (see pi_glm8_prepare.py).
"""

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alf_study_common as C  # noqa: E402
from pi_glm8_build_hints import render_uncapped  # noqa: E402

DUMP = str(C.DUMP200_4BI)
TASKS200 = str(C.TRAIN200)
SCHEMA = "opsd_hints/v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gen",
        default=str(C.MULTI_AUTHOR),
        help="generation root: <author>/c*.jsonl + topup_r*.jsonl (run_alf_pi_multimodel.sh / "
        "run_alf_pi_topup.sh OUT; default outputs/pi_multimodel)",
    )
    ap.add_argument("--models", default="dsv41flash,minimaxm3,qwen38max,kimik3")
    ap.add_argument("--dump", default=DUMP, help="the N=128 no-hint student corpus on the 200 tasks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument(
        "--skip-manifest",
        default=None,
        help="hints/manifest.json of an already-scored dir: an author's tasks listed there are skipped "
        "here, so only newly covered tasks (e.g. solved in a top-up round) are scored",
    )
    args = ap.parse_args()
    skip = json.load(open(args.skip_manifest)) if args.skip_manifest else {}
    tasks = sorted(json.load(open(TASKS200)))
    tset = set(tasks)
    out = Path(args.out)
    (out / "hints").mkdir(parents=True, exist_ok=True)
    (out / "chunks").mkdir(parents=True, exist_ok=True)

    manifest = {}
    for m in args.models.split(","):
        hints, stats = {"__schema__": SCHEMA}, collections.Counter()
        files = sorted(C.author_dir(args.gen, m).glob("c*.jsonl")) + sorted(
            C.author_dir(args.gen, m).glob("topup_r*.jsonl")
        )
        done_before = set(skip.get(m, {}).get("tasks", []))
        seen_tasks = set()
        for f in files:
            for line in open(f):
                r = json.loads(line)
                if r["task_id"] not in tset:
                    continue
                seen_tasks.add(r["task_id"])
                if not r["won"] or r["task_id"] in done_before or r["task_id"] in hints:
                    continue
                text = render_uncapped(r)
                hints[r["task_id"]] = {
                    "task": text,
                    "meta": {
                        "pi_format": "traj_withthink",
                        "author": m,
                        "num_turns": r["num_turns"],
                    },
                }
                stats["chars"] += len(text)
                stats["reasoning_turns"] += sum(bool(t.get("reasoning")) for t in r["turns"] if "assistant" in t)
                stats["turns"] += r["num_turns"]
        n = len(hints) - 1
        assert len(seen_tasks) == 200, (m, len(seen_tasks))
        (out / "hints" / f"hints_{m}.json").write_text(json.dumps(hints))
        manifest[m] = {
            "hinted_tasks": n,
            "mean_chars": stats["chars"] / max(n, 1),
            "mean_turns": stats["turns"] / max(n, 1),
            "reasoning_turn_share": stats["reasoning_turns"] / max(stats["turns"], 1),
            "tasks": sorted(k for k in hints if k != "__schema__"),
        }
        print(
            f"{m}: {n}/200 tasks hinted, mean {manifest[m]['mean_chars']:.0f} chars, "
            f"{manifest[m]['mean_turns']:.1f} turns"
        )
    (out / "hints" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "task_ids.json").write_text(json.dumps(tasks, indent=1))

    lines = open(args.dump).read().splitlines()
    assert len(lines) == 25600
    task_of = [json.loads(line)["env_extras"]["extra_info"]["task_id"] for line in lines]
    for c in range(args.chunks):
        ts, idx = set(tasks[c :: args.chunks]), []
        with open(out / "chunks" / f"chunk_{c}.jsonl", "w") as f:
            for k, line in enumerate(lines):
                if task_of[k] in ts:
                    f.write(line + "\n")
                    idx.append(k)
        (out / "chunks" / f"chunk_{c}_index.json").write_text(json.dumps(idx))
    common = set.intersection(*(set(v["tasks"]) for v in manifest.values()))
    print(f"tasks hinted by every author: {len(common)}; {len(lines)} rollouts in {args.chunks} chunks")


if __name__ == "__main__":
    main()

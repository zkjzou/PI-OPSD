#!/usr/bin/env python
"""Render each task's first 8 WON GLM-5.2 trajectories as 8 raw-trajectory hint files (slots s0..s7).

Format = the published uncapped raw arm, the uncapped GLM-5.2 `hints_traj_withthink.json`:
role-tagged [OBSERVATION]/[AGENT] lines, GLM's out-of-band reasoning folded back in as
<think>...</think> (stripped of surrounding whitespace), and NO token cap. `render_uncapped` reproduces
all 1,369 published hints of the 2026-07-12 top-up byte-for-byte (checked 2026-09-29).

Also writes `hints_glm_ref.json`, the published hint for the same tasks, as a 9th arm, plus a manifest
with per-trajectory descriptors (turns, length, action sequence) for the diversity summary.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alf_study_common as C  # noqa: E402
from alf_study_common import ACT_RE, fold_reasoning  # noqa: E402

SCHEMA = "opsd_hints/v2"
# The 9th arm is the published uncapped glr hint (hints_traj_withthink.json, the REBUILT_glm__traj_withthink
# render of alfworld_rebuild_hints.py) restricted to the study's tasks (only those tasks are read).
PUBLISHED = str(C.GLM_HINTS / "hints_traj_withthink.json")


def render_uncapped(rec: dict) -> str:
    turns = [dict(t, reasoning=(t.get("reasoning") or "").strip()) for t in rec["turns"]]
    lines = []
    for m in fold_reasoning(rec["messages"], turns):
        content = str(m.get("content", ""))
        if not content:
            continue
        tag = {"user": "OBSERVATION", "assistant": "AGENT", "system": "SYSTEM"}.get(m.get("role", "?"), "?")
        lines.append(f"[{tag}] {content}")
    return "\n".join(lines)


def actions(rec: dict) -> list:
    return [
        a.strip()
        for m in rec["messages"]
        if m.get("role") == "assistant"
        for a in ACT_RE.findall(str(m.get("content", "")))
        if a.strip()
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", action="append", required=True, help="GLM driver output(s), in order")
    ap.add_argument("--task-ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument(
        "--published",
        default=PUBLISHED,
        help="published uncapped glr hints (only the study's tasks are read)",
    )
    args = ap.parse_args()

    tasks = json.load(open(args.task_ids))
    won, lost = collections.defaultdict(list), collections.Counter()
    for path in args.trajectories:
        for line in open(path):
            r = json.loads(line)
            if r["task_id"] not in tasks:
                continue
            if r.get("won"):
                won[r["task_id"]].append(r)
            else:
                lost[r["task_id"]] += 1
    short = {t: len(won[t]) for t in tasks if len(won[t]) < args.slots}
    if short:
        raise SystemExit(f"tasks with < {args.slots} won trajectories (top up first): {short}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"slots": args.slots, "lost_per_task": dict(lost), "tasks": {}}
    files = [{"__schema__": SCHEMA} for _ in range(args.slots)]
    for t in tasks:
        manifest["tasks"][t] = []
        for j, r in enumerate(won[t][: args.slots]):
            text = render_uncapped(r)
            files[j][t] = {
                "task": text,
                "meta": {
                    "pi_format": "traj_withthink",
                    "source_rep": r["repetition_id"],
                    "num_turns": r["num_turns"],
                },
            }
            manifest["tasks"][t].append(
                {
                    "slot": j,
                    "rep": r["repetition_id"],
                    "num_turns": r["num_turns"],
                    "chars": len(text),
                    "reasoning_chars": sum(len(x.get("reasoning") or "") for x in r["turns"]),
                    "actions": actions(r),
                }
            )
    for j, f in enumerate(files):
        (out / f"hints_glm_s{j}.json").write_text(json.dumps(f))

    pub = json.load(open(args.published))
    ref = {"__schema__": SCHEMA}
    for t in tasks:
        if t in pub:
            ref[t] = pub[t]
    (out / "hints_glm_ref.json").write_text(json.dumps(ref))
    manifest["ref_coverage"] = len(ref) - 1
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        f"wrote {args.slots} slot files for {len(tasks)} tasks; published ref covers {len(ref) - 1}; "
        f"lost GLM trajectories: {sum(lost.values())}"
    )


if __name__ == "__main__":
    main()

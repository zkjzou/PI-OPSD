#!/usr/bin/env python
"""Per-task SELF PI selection for OPSD on the 483 tasks with >= 8 self wins (the pi_self16_prepare.py set).

Two candidate pools per task, both from the 16 self rollouts sampled once per task:
  sw8   the 8 won rollouts (self_w1..w8)
  sa16  all 16: the 8 won + the 8 lost slots (on 105 tasks some lost slots hold least-efficient WINS as
        backups; they are candidates like any slot)
Per (task, slot): d and Δd⁻_err with the published estimators (pi_glm8_report.d_by / derr_frame) on the task's
16 scored rollouts MINUS the slot's own trajectory (self_pairs.json), so every candidate is scored on the
other 15. Backups count as LOSSES for Δd⁻_err (their judge labels were drawn for exactly that); d is unaffected.
Arms per pool, one hint per task, fixed for the run (the draw is seeded by seed, pool, arm and task):
  rand   uniform over the pool
  top3d  uniform over the 3 slots with the highest d
  top3e  uniform over the 3 with the highest Δd⁻_err (the rand draw where Δd⁻_err is undefined)

    python pi_self16_select_build.py --seed 0 --out outputs/pi_self16_select

Reads (--w, default $PI_STUDY_ROOT/pi_self16b): hints/hints_self_{w,l}{1..8}.json, self_pairs.json, selection.csv,
labels_merged.json (pi_self16_prepare.py's labels.json merged with the judge labels of its to_label.json) and the
scored chunks. The published draws are alfworld/selection/hints_self16{,_s1,_s2}/ (--seed 0..2).
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
from pi_glm8_report import d_by, derr_frame  # noqa: E402
from pi_self4_report import done_chunks, load_chunks  # noqa: E402

REPO = C.repo_root()
sys.path.insert(0, str(REPO))
from opsd.pi.artifact import save_hint_artifact  # noqa: E402
from opsd.pi.base import HintRecord  # noqa: E402

W = C.PI_STUDY_ROOT / "pi_self16b"
SLOTS_W = [f"self_w{k}" for k in range(1, 9)]
SLOTS_L = [f"self_l{k}" for k in range(1, 9)]
POOLS = {"sw8": SLOTS_W, "sa16": SLOTS_W + SLOTS_L}
ARMS = ["rand", "top3d", "top3e"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", default=str(W), help="scoring dir from pi_self16_prepare.py")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--partial", action="store_true", help="allow unfinished chunks (smoke test only)")
    args = ap.parse_args()
    w, out = Path(args.w), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    hints = {s: json.load(open(w / "hints" / f"hints_{s}.json")) for s in SLOTS_W + SLOTS_L}
    pairs = json.load(open(w / "self_pairs.json"))  # slot -> task -> uid of the slot's own rollout
    own = {f"{s}|{t}": u for s, m in pairs.items() for t, u in m.items()}
    backup = {
        own[f"{s}|{t}"] for s in SLOTS_L for t, v in hints[s].items() if t != "__schema__" and v["meta"].get("backup")
    }
    tasks = sorted(pd.read_csv(w / "selection.csv").task_id)

    rows = load_chunks(w)
    # planned chunks: the prepare script writes chunk_<i>.jsonl + chunk_<i>_index.json; the published copy
    # carries only the index files (no rollout text), so count those when the .jsonl are absent
    n_chunks = len(list((w / "chunks").glob("chunk_*.jsonl"))) or len(list((w / "chunks").glob("chunk_*_index.json")))
    n_done = len(done_chunks(w))
    assert args.partial or n_done == n_chunks, f"only {n_done}/{n_chunks} chunks scored"
    key = rows.pi.astype(str) + "|" + rows.task_id.astype(str)
    rows = rows[rows.uid != key.map(own)]  # never a hint against its own trajectory
    rows_e = rows.copy()
    rows_e.loc[rows_e.uid.isin(backup), "success"] = 0  # backups count as losses for Δd⁻_err
    labels = json.load(open(w / "labels_merged.json"))
    met = pd.concat(
        [
            d_by(rows, ["pi", "task_id"]).rename("d"),
            derr_frame(rows_e, labels).groupby(["pi", "task_id"]).dev.mean().rename("derr"),
        ],
        axis=1,
    ).reset_index()
    met = met.rename(columns={"pi": "slot"})
    met.to_csv(out / "per_slot_metrics.csv", index=False)
    by_task = {t: g.set_index("slot") for t, g in met.groupby("task_id")}
    scored_tasks = [t for t in tasks if t in by_task]

    summary = {"tasks": len(scored_tasks), "backup_slots": len(backup), "pools": {}}
    for pool, slots in POOLS.items():
        picks = {a: {} for a in ARMS}
        fallback = 0
        for t in scored_tasks:
            g = by_task[t].loc[[s for s in slots if s in by_task[t].index]]

            def draw(arm: str, cands: list) -> str:
                return random.Random(f"{args.seed}:{pool}_{arm}:{t}").choice(sorted(cands))

            picks["rand"][t] = draw("rand", list(g.index))
            picks["top3d"][t] = draw("top3d", list(g.d.sort_values(ascending=False).index[: args.k]))
            e = g.derr.dropna()
            if e.empty:
                picks["top3e"][t], fallback = picks["rand"][t], fallback + 1
            else:
                picks["top3e"][t] = draw("top3e", list(e.sort_values(ascending=False).index[: args.k]))
        summary["pools"][pool] = {"top3e_fallback_to_rand": fallback, "arms": {}}
        for a in ARMS:
            recs = {}
            for t in scored_tasks:
                s = picks[a][t]
                v, r = hints[s][t], by_task[t].loc[s]
                meta = {
                    "arm": f"{pool}_{a}",
                    "slot": s,
                    "src": v["meta"]["src"],
                    "backup": v["meta"].get("backup", False),
                }
                meta["d"] = round(float(r.d), 5)
                if pd.notna(r.derr):
                    meta["derr"] = round(float(r.derr), 5)
                recs[t] = HintRecord(task=v["task"], meta=meta)
            save_hint_artifact(str(out / f"hints_{pool}_{a}.json"), recs)
            kinds = collections.Counter(
                "backup" if hints[picks[a][t]][t]["meta"].get("backup") else picks[a][t][5] for t in scored_tasks
            )
            chosen = pd.DataFrame([by_task[t].loc[picks[a][t]] for t in scored_tasks])
            summary["pools"][pool]["arms"][a] = {
                "slot_kind_share": {k: round(c / len(scored_tasks), 3) for k, c in kinds.items()},
                "mean_d": round(float(chosen.d.mean()), 4),
                "mean_derr": round(float(chosen.derr.mean()), 4),
                "mean_chars": round(sum(len(hints[picks[a][t]][t]["task"]) for t in scored_tasks) / len(scored_tasks)),
            }
        summary["pools"][pool]["same_slot_share"] = {
            f"{x}~{y}": round(sum(picks[x][t] == picks[y][t] for t in scored_tasks) / len(scored_tasks), 3)
            for i, x in enumerate(ARMS)
            for y in ARMS[i + 1 :]
        }
    (out / "task_ids.json").write_text(json.dumps(scored_tasks, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

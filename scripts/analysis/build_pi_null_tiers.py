#!/usr/bin/env python
"""Build the TIERED mismatched-PI nulls (n1/n2/n3) for ALFWorld.

Three nulls of increasing SEMANTIC DISTANCE. Every tier keeps the hint strings, register and length
multiset byte-identical -- only task correspondence is destroyed, with progressively stricter mismatch:

    n1  fixed-point-free derangement (no task gets its own hint)
    n2  n1 + different task TYPE
    n3  n2 + different ROOM (scene) + zero object overlap

WHY A SEPARATE SCRIPT FROM build_pi_null_hints.py. That one derives the mismatch constraint by PARSING
THE TASK ID (`<scenario>_<variant>`), which is right for AppWorld and unsatisfiable for ALFWorld: every
ALFWorld id is `trial_T<ts>_<id>`, one prefix, so its scenario constraint degenerates and it correctly
falls back to a plain derangement -- i.e. it can only ever build n1 here. ALFWorld DOES have the
structure, it just is not in the id: it is in the train parquet, where
`reward_spec.ground_truth` encodes `<task_type>-<object>-<movable>-<target_recep>-<scene>`. So the
grouping is LOOKED UP, not parsed.

FEASIBILITY IS NOT ASSUMED, IT IS PROVEN. Legal partners per task over train1600, measured:
    n1  min 1599   n2  min 1191   n3  min 1015   (of 1599 candidates)
Huge slack at every tier, so the room/object constraints never need the "secondary" relaxation. The
assignment is still solved EXACTLY rather than by rejection sampling, and verified after the fact.

WHAT `scene` IS, AND IS NOT. It is the scene INSTANCE (108 of them), not the room CLASS. Two different
scenes can both be kitchens with heavily overlapping receptacle vocabulary, so n3's room constraint is
weaker than "different room" reads. --room-class buckets the scene number into ALFWorld's standard
blocks (kitchen / living room / bedroom / bathroom) if you want the stronger reading.

  python scripts/analysis/build_pi_null_tiers.py --inp <hints.json> --out-dir <dir> \
      --parquet <ALFWORLD_PARQUET_DIR>/train.parquet --tiers n1 n2 n3

The paper's mismatched-trajectory null (the ALFWorld `null_task` arm, glt_n3_from_glm.json) is tier n3 of the
GLM-5.2 Action + Obs. + Thou. hints over the 1,600 training tasks, seed 0:

  python scripts/analysis/build_pi_null_tiers.py \
      --inp $OPSD_DATA_ROOT/alfworld/hints/glm-5.2/hints_traj_thoughts_cap24k.json \
      --task-ids $OPSD_DATA_ROOT/alfworld/splits/task_ids_train1600.json \
      --parquet <ALFWORLD_PARQUET_DIR>/train.parquet --tiers n3 --out-dir outputs/nulls
  # -> outputs/nulls/hints_null_n3.json (same donor and hint text for all 1,600 tasks)

train.parquet is written by scripts/setup/setup_alfworld.sh (default $HOME/data/alfworld/parquet).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

_GT = re.compile(r"/([a-z_]+)-([A-Za-z]+)-([A-Za-z]+|None)-([A-Za-z]+)-(\d+)/")


def room_class(scene: int) -> str:
    """ALFWorld's standard scene-number blocks. Only used with --room-class."""
    s = int(scene)
    if 1 <= s <= 30:
        return "kitchen"
    if 201 <= s <= 230:
        return "living_room"
    if 301 <= s <= 330:
        return "bedroom"
    if 401 <= s <= 430:
        return "bathroom"
    return f"other_{s // 100}"


def load_structure(parquet: str, keys: set) -> dict:
    import pandas as pd
    df = pd.read_parquet(parquet)
    out = {}
    for _, r in df.iterrows():
        ei = r["extra_info"]
        tid = ei.get("task_id")
        if tid not in keys:
            continue
        m = _GT.search(str(r["reward_spec"].get("ground_truth", "")))
        if not m:
            continue
        ttype, obj, mov, recep, scene = m.groups()
        out[tid] = {"type": ei.get("task_type") or ttype, "scene": scene,
                    "objs": {x for x in (obj, mov, recep) if x and x != "None"}}
    return out


def legal(a: dict, b: dict, tier: str, use_room_class: bool) -> bool:
    if tier == "n1":
        return True
    if a["type"] == b["type"]:
        return False
    if tier == "n2":
        return True
    ra = room_class(a["scene"]) if use_room_class else a["scene"]
    rb = room_class(b["scene"]) if use_room_class else b["scene"]
    return ra != rb and not (a["objs"] & b["objs"])


def solve(keys: list, st: dict, tier: str, seed: int, use_room_class: bool) -> dict:
    """Exact assignment by augmenting-path matching (Hopcroft-Karp style), not rejection sampling.

    Rejection sampling would work here given the measured slack, but it cannot DISTINGUISH "infeasible"
    from "unlucky" -- and the whole point of a null is that you can state what it guarantees. An
    augmenting-path search either returns a perfect matching or proves none exists.
    """
    rng = random.Random(seed)
    idx = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    # adjacency, randomised so different seeds give different valid matchings
    adj = []
    for k in keys:
        cand = [idx[j] for j in keys if j != k and legal(st[k], st[j], tier, use_room_class)]
        rng.shuffle(cand)
        adj.append(cand)
    match_to = [-1] * n          # hint index -> task index

    def try_assign(u: int, seen: list) -> bool:
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match_to[v] == -1 or try_assign(match_to[v], seen):
                match_to[v] = u
                return True
        return False

    order = list(range(n))
    rng.shuffle(order)
    sys.setrecursionlimit(10000)
    for u in order:
        if not try_assign(u, [False] * n):
            raise SystemExit(f"FATAL: {tier} has no perfect matching (task {keys[u]} unassignable)")
    return {keys[match_to[v]]: keys[v] for v in range(n)}   # task -> donor task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="source hint artifact to permute")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--parquet", default=os.path.join(os.environ.get("ALFWORLD_PARQUET_DIR", os.path.expanduser("~/data/alfworld/parquet")), "train.parquet"))
    ap.add_argument("--task-ids", default=None, help="restrict to these ids (JSON list); default = all in parquet")
    ap.add_argument("--tiers", nargs="+", default=["n1", "n2", "n3"])
    ap.add_argument("--room-class", action="store_true", help="n3 room constraint = room CLASS not scene")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src = json.loads(Path(a.inp).read_text())
    schema = src.pop("__schema__", None)
    want = set(json.loads(Path(a.task_ids).read_text())) if a.task_ids else None
    st = load_structure(a.parquet, set(src) if want is None else (set(src) & want))
    keys = sorted(k for k in src if k in st)
    print(f"  source {Path(a.inp).name}: {len(src)} keys -> {len(keys)} with parquet structure")
    if want:
        miss = want - set(keys)
        print(f"  requested {len(want)} ids, {len(keys)} usable, {len(miss)} missing" + (f" !! {sorted(miss)[:3]}" if miss else ""))

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for tier in a.tiers:
        assign = solve(keys, st, tier, a.seed, a.room_class)
        body = {}
        for task, donor in assign.items():
            v = src[donor]
            rec = dict(v) if isinstance(v, dict) else {"task": v}
            meta = dict(rec.get("meta") or {})
            meta.update({"null_tier": tier, "donor_task_id": donor,
                         "room_constraint": "room_class" if a.room_class else "scene",
                         "source_artifact": Path(a.inp).name})
            rec["meta"] = meta
            body[task] = rec
        # VERIFY before writing: fixed points, tier constraint, and the length multiset
        fixed = sum(1 for t, d in assign.items() if t == d)
        bad = [t for t, d in assign.items() if not legal(st[t], st[d], tier, a.room_class)]
        li = sorted(len((src[k]["task"] if isinstance(src[k], dict) else src[k])) for k in keys)
        lo = sorted(len(v["task"]) for v in body.values())
        assert fixed == 0, f"{tier}: {fixed} fixed points"
        assert not bad, f"{tier}: {len(bad)} pairs violate the constraint"
        assert li == lo, f"{tier}: length multiset not preserved"
        same_type = sum(1 for t, d in assign.items() if st[t]["type"] == st[d]["type"])
        same_scene = sum(1 for t, d in assign.items() if st[t]["scene"] == st[d]["scene"])
        overlap = sum(1 for t, d in assign.items() if st[t]["objs"] & st[d]["objs"])
        p = out / f"hints_null_{tier}.json"
        p.write_text(json.dumps(({"__schema__": schema} if schema else {}) | body))
        print(f"  {tier}: {len(body)} tasks | fixed 0 | same-type {same_type} | same-scene {same_scene} "
              f"| object-overlap {overlap} | lengths preserved -> {p.name}")


if __name__ == "__main__":
    main()

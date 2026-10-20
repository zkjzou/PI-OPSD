#!/usr/bin/env python
"""Split the ALFWorld train1600 task ids into the 8 generation chunks of the multi-author PI pipeline.

run_alf_pi_multimodel.sh / run_alf_pi_topup.sh read $TASKS/train1600_c{0..7}.parquet: chunk k is the k-th
consecutive slice of 200 ids of task_ids_train1600.json (the file's order, which is sorted), and each chunk
parquet holds those tasks' rows of the ALFWorld train parquet (alfworld_dataset.py output, whose
reward_spec.ground_truth points at THIS machine's game files) in chunk-id order.

Writes, under --out-dir:
  train1600_c<k>_task_ids.json   the chunk's ids
  train1600_c<k>.parquet         the chunk's rows (only with --parquet)
  task_ids_train1600.json        a copy of the input id list (run_alf_pi_topup.sh reads $OUT/tasks/)

--verify-dir DIR compares against an existing chunk set: train1600_c<k>_task_ids.json and/or
train1600_c<k>.parquet in DIR (ids per chunk and their order; for parquets also every column, with the game
path compared relative to its json_2.1.1/ root). Exit 1 on any difference.

  python scripts/analysis/make_train_chunks.py --parquet <alfworld parquet dir>/train.parquet \\
      --out-dir outputs/pi_multimodel/tasks \\
      --verify-dir <an existing chunk set>
"""

import argparse
import json
import os
import sys
from pathlib import Path

_DATA = os.environ.get("OPSD_DATA_ROOT") or os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "opsd", "data"
)
TASK_IDS = os.path.join(_DATA, "alfworld", "splits", "task_ids_train1600.json")


def chunk_ids(ids: list, n_chunks: int) -> list:
    """n_chunks consecutive slices; equal sizes when they divide, else the first len % n are one longer."""
    q, r = divmod(len(ids), n_chunks)
    out, i = [], 0
    for k in range(n_chunks):
        size = q + (1 if k < r else 0)
        out.append(ids[i : i + size])
        i += size
    return out


def _rel_game(path: str) -> str:
    """Game path relative to its json_2.1.1/ root, so chunk files from different machines compare."""
    path = str(path)
    i = path.find("json_2.1.1/")
    return path[i:] if i >= 0 else path


def _norm_row(row: dict) -> dict:
    row = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in row.items()}
    rs = dict(row.get("reward_spec") or {})
    if "ground_truth" in rs:
        rs["ground_truth"] = _rel_game(rs["ground_truth"])
    row["reward_spec"] = rs
    row["prompt"] = [dict(m) for m in (row.get("prompt") or [])]
    row["extra_info"] = dict(row.get("extra_info") or {})
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-ids", default=TASK_IDS, help="train1600 task ids (JSON list)")
    ap.add_argument("--parquet", default=None, help="ALFWorld SkyRL train parquet; omit to write id files only")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-chunks", type=int, default=8)
    ap.add_argument("--prefix", default="train1600")
    ap.add_argument("--verify-dir", default=None, help="existing chunk set to compare against")
    a = ap.parse_args()

    ids = json.load(open(a.task_ids))
    assert len(ids) == len(set(ids)), "duplicate task ids"
    chunks = chunk_ids(ids, a.n_chunks)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "task_ids_train1600.json").write_text(json.dumps(ids))
    for k, c in enumerate(chunks):
        (out / f"{a.prefix}_c{k}_task_ids.json").write_text(json.dumps(c))
    print(f"{len(ids)} ids -> {a.n_chunks} chunks of {[len(c) for c in chunks]} in {out}")

    frames = None
    if a.parquet:
        import pandas as pd

        df = pd.read_parquet(a.parquet)
        tid = df["extra_info"].map(lambda e: e["task_id"])
        assert not tid.duplicated().any(), f"duplicate task ids in {a.parquet}"
        row_of = {t: i for i, t in enumerate(tid)}
        missing = [t for t in ids if t not in row_of]
        if missing:
            print(f"FATAL: {len(missing)} task ids are not in {a.parquet}, e.g. {missing[:3]}", file=sys.stderr)
            return 1
        frames = []
        for k, c in enumerate(chunks):
            sub = df.iloc[[row_of[t] for t in c]].reset_index(drop=True)
            sub.to_parquet(out / f"{a.prefix}_c{k}.parquet")
            frames.append(sub)
        print(f"wrote {a.n_chunks} chunk parquets from {a.parquet}")

    if not a.verify_dir:
        return 0
    ref, ok, checked = Path(a.verify_dir), True, 0
    for k, c in enumerate(chunks):
        jf, pf = ref / f"{a.prefix}_c{k}_task_ids.json", ref / f"{a.prefix}_c{k}.parquet"
        if jf.exists():
            same = json.load(open(jf)) == c
            ok &= same
            checked += 1
            print(f"  [{'OK  ' if same else 'FAIL'}] c{k} ids vs {jf.name}")
        if pf.exists():
            import pandas as pd

            theirs = pd.read_parquet(pf)
            t_ids = [e["task_id"] for e in theirs["extra_info"]]
            same = t_ids == c
            detail = "ids + order"
            if same and frames is not None:
                mine = frames[k]
                same = list(mine.columns) == list(theirs.columns) and all(
                    _norm_row(x) == _norm_row(y) for x, y in zip(mine.to_dict("records"), theirs.to_dict("records"))
                )
                detail = "ids + order + rows"
            ok &= same
            checked += 1
            print(f"  [{'OK  ' if same else 'FAIL'}] c{k} {detail} vs {pf.name}")
    if not checked:
        print(f"FATAL: nothing to verify in {ref}", file=sys.stderr)
        return 1
    print("VERIFY OK" if ok else "VERIFY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Per-run ALFWorld parquets for scripts/alfworld/run.sh.

The ALFWorld parquets (``train.parquet`` = the whole train split, ``validation.parquet`` = valid_seen,
``validation_unseen.parquet`` = valid_unseen) are built once per machine by
``scripts/setup/setup_alfworld.sh``. A run trains on a task-id subset of the train split and may hold
out a dev split from it, so this helper writes the run's own copies into ``--out-dir``:

  train.parquet      the train split filtered to ``--task-ids`` (file order is irrelevant: rows keep the
                     parquet's order), optionally truncated to the first ``--max-train-tasks`` ids of the list
  dev.parquet        (``--dev``) the train-split rows of ``--dev-task-ids``, relabelled
                     ``data_source=alfworld_dev`` so they land in their own eval bucket (eval/alfworld_dev/*),
                     written with the arrow schema of the peer eval parquet; the dev ids must not overlap the
                     training rows (asserted)
  validation*.parquet  (``--max-eval-tasks``) the first N rows of each eval parquet (smoke runs only)

Only the files that differ from the shared parquets are written; run.sh derives the same paths. The last
stdout line is a JSON object naming every file written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

EVAL_FILES = ("validation.parquet", "validation_unseen.parquet")


def load_task_ids(path: str) -> List[str]:
    """A JSON list of task ids, or one id per line."""
    text = Path(path).expanduser().read_text().strip()
    try:
        ids = json.loads(text)
    except json.JSONDecodeError:
        ids = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not isinstance(ids, list) or not ids:
        raise SystemExit(f"{path}: expected a non-empty JSON list (or newline-delimited file) of task ids")
    return [str(i) for i in ids]


def _task_id(extra_info) -> Optional[str]:
    return (extra_info or {}).get("task_id")


def write_train(parquet_dir: Path, out: Path, task_ids_path: Optional[str], max_tasks: Optional[int]) -> dict:
    import pandas as pd

    df = pd.read_parquet(parquet_dir / "train.parquet")
    if task_ids_path:
        ids = load_task_ids(task_ids_path)
        if max_tasks:
            ids = ids[:max_tasks]
        wanted = set(ids)
        df = df[df["extra_info"].map(lambda e: _task_id(e) in wanted)].reset_index(drop=True)
        print(f"train subset rows: {len(df)} / requested {len(wanted)}")
        if len(df) < len(wanted):
            print(f"WARNING: {len(wanted) - len(df)} requested task ids are not in the train parquet")
    elif max_tasks:
        df = df.iloc[:max_tasks].reset_index(drop=True)
        print(f"train rows (first {max_tasks}): {len(df)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return {"train": str(out)}


def write_dev(parquet_dir: Path, out: Path, dev_ids_path: str, train_used: Path, max_tasks: Optional[int]) -> dict:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    ids = set(load_task_ids(dev_ids_path))
    df = pd.read_parquet(parquet_dir / "train.parquet")
    df = df[df["extra_info"].map(lambda e: _task_id(e) in ids)].reset_index(drop=True)
    df["data_source"] = "alfworld_dev"
    if max_tasks:
        df = df.iloc[:max_tasks].reset_index(drop=True)
    # Write with the arrow schema of a PEER eval parquet: dev.parquet is concatenated with the validation
    # parquets, which needs identical feature types, and a pandas round-trip downgrades large_string.
    peer = pq.read_schema(parquet_dir / "validation.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, schema=peer, preserve_index=False), out)
    print(f"dev rows: {len(df)} / requested {len(ids)} | data_source type: {peer.field('data_source').type}")
    # Leakage guard: dev ids must not appear in the train parquet actually being used.
    tr = pd.read_parquet(train_used)
    bad = set(tr["extra_info"].map(_task_id)) & ids
    assert not bad, f"LEAKAGE: {len(bad)} dev task_ids present in the train parquet"
    print(f"leakage check OK: train rows {len(tr)} | train n dev = 0")
    return {"dev": str(out)}


def write_eval_heads(parquet_dir: Path, out_dir: Path, max_tasks: int) -> dict:
    import pyarrow.parquet as pq

    written = {}
    for name in EVAL_FILES:
        src = parquet_dir / name
        if not src.is_file():
            continue
        table = pq.read_table(src)
        out = out_dir / name
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.slice(0, max_tasks), out)
        print(f"{name}: first {min(max_tasks, table.num_rows)} of {table.num_rows} rows")
        written[name.removesuffix(".parquet")] = str(out)
    return written


def main(argv: Optional[Sequence[str]] = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", required=True, help="directory holding train/validation/validation_unseen")
    ap.add_argument("--out-dir", required=True, help="the run's data directory")
    ap.add_argument("--task-ids", default=None, help="train task ids (JSON list); omit to train on the split")
    ap.add_argument("--max-train-tasks", type=int, default=None, help="keep the first N train task ids")
    ap.add_argument("--dev", action="store_true", help="also write dev.parquet from --dev-task-ids")
    ap.add_argument("--dev-task-ids", default=None, help="dev task ids (JSON list), disjoint from --task-ids")
    ap.add_argument("--max-eval-tasks", type=int, default=None, help="keep the first N rows of each eval pool")
    a = ap.parse_args(argv)

    parquet_dir = Path(a.parquet_dir).expanduser()
    out_dir = Path(a.out_dir).expanduser()
    for name in ("train.parquet",) + EVAL_FILES:
        if not (parquet_dir / name).is_file():
            raise SystemExit(f"missing {parquet_dir / name}: build it with scripts/setup/setup_alfworld.sh")
    written: dict = {}
    train_used = parquet_dir / "train.parquet"
    if a.task_ids or a.max_train_tasks:
        written.update(write_train(parquet_dir, out_dir / "train.parquet", a.task_ids, a.max_train_tasks))
        train_used = out_dir / "train.parquet"
    if a.dev:
        if not a.dev_task_ids:
            raise SystemExit("--dev needs --dev-task-ids")
        written.update(write_dev(parquet_dir, out_dir / "dev.parquet", a.dev_task_ids, train_used, a.max_eval_tasks))
    if a.max_eval_tasks:
        written.update(write_eval_heads(parquet_dir, out_dir, a.max_eval_tasks))
    print(json.dumps(written, sort_keys=True))
    return written


if __name__ == "__main__":
    main()

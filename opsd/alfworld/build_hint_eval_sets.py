"""Build PI-utility eval parquets: one parquet per hint set, all over the SAME task ids.

Each output row is a copy of a base-parquet row (normally ``train.parquet`` from ``alfworld_dataset.py``,
since the task ids come from the train pool) with two columns set:

  data_source = "alfworld_<set>"   -> each set lands in its own eval bucket (eval/alfworld_<set>/*)
  hint        = that set's task-level hint ("" for the ``nohint`` set)

SkyRL's dataset hands every non-prompt column to ``env_extras``, and ``AlfworldEnv`` puts ``extras["hint"]``
into the STUDENT's prompt: appended to the first user turn (``--placement user``), or as a leading system
turn (``--placement system``), for which this script pre-wraps the hint exactly as the OPSD teacher's
<Secret information> block. ``--placement`` must therefore match the env's
``environment.skyrl_gym.alfworld.hint_placement``. Evaluating these parquets with training disabled
(EVAL_ONLY) measures in-context PI utility under the exact training env contract, e.g. the ALFWorld-Passcode
sets ``nohint+pc_key+pc_rawkey+pc_rawwrong+pc_wrongraw``.

Inputs:
  --base-parquet   ALFWorld parquet holding every requested task id (one row per task id)
  --hint-dir       directory with one artifact per set: hints_<set>.json      (default: $EVAL_HINT_DIR)
  --sets           set names separated by '+', ',' or spaces; `nohint` needs no file (default: $EVAL_HINT_SETS)
  --task-ids       JSON list (or newline-delimited file) of task ids         (default: $EVAL_HINT_TASK_IDS_PATH)
Every requested task must have a non-empty hint in every hinted set: a hint-less row would be scored as hinted.

Outputs: <out-dir>/eval_hint_<set>.parquet (default out-dir: the base parquet's directory), written with the
arrow schema of the peer eval parquet (``validation.parquet`` next to the base parquet, else the base
parquet's own) plus a large_string ``hint`` column, so ``datasets.concatenate_datasets`` can align the sets.
The LAST stdout line is the hydra list of the written files, ready for ``data.val_data=...``:

  VAL_DATA=$(python -m opsd.alfworld.build_hint_eval_sets \\
      --base-parquet "$DATA_DIR/train.parquet" --sets "nohint+pc_key" | tail -1)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

NOHINT = "nohint"
_SET_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_sets(raw: Sequence[str]) -> List[str]:
    """Split set names on '+', ',' or whitespace; keep first-seen order, drop duplicates."""
    names: List[str] = []
    for chunk in raw:
        for name in re.split(r"[+,\s]+", str(chunk).strip()):
            if not name:
                continue
            if not _SET_NAME_RE.match(name):
                raise ValueError(f"invalid hint-set name {name!r} (allowed: letters, digits, '_', '.', '-')")
            if name not in names:
                names.append(name)
    if not names:
        raise ValueError("no hint sets given (pass --sets or set EVAL_HINT_SETS)")
    return names


def load_task_ids(path: str) -> List[Any]:
    """A JSON list of task ids, or one id per line."""
    text = Path(path).expanduser().read_text().strip()
    try:
        ids = json.loads(text)
    except json.JSONDecodeError:
        ids = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"{path}: expected a non-empty JSON list (or newline-delimited file) of task ids")
    return ids


def hint_file(hint_dir: Optional[str], name: str) -> Path:
    if not hint_dir:
        raise ValueError(f"set {name!r} needs a hint artifact: pass --hint-dir (or set EVAL_HINT_DIR)")
    path = Path(hint_dir).expanduser() / f"hints_{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"hint artifact for set {name!r} not found: {path}")
    return path


def build_eval_set(base_df, name: str, task_ids: Sequence[Any], hints_path: Optional[Path], placement: str):
    """The rows of ``base_df`` for ``task_ids`` (base order), relabelled for set ``name`` with its hint."""
    wanted = set(task_ids)
    df = base_df[base_df["extra_info"].map(lambda e: (e or {}).get("task_id") in wanted)].reset_index(drop=True)
    if len(df) != len(task_ids):
        raise ValueError(
            f"{name}: {len(df)} base-parquet rows for {len(task_ids)} requested task ids "
            "(missing or duplicated task ids)"
        )
    df["data_source"] = f"alfworld_{name}"
    hints: Dict[str, str] = {}
    if name != NOHINT:
        from opsd.pi.artifact import load_task_hints

        hints = load_task_hints(str(hints_path))
        missing = [t for t in task_ids if not str(hints.get(t) or "").strip()]
        if missing:
            raise ValueError(
                f"{name}: {len(missing)} tasks have no hint (a hint-less row would score as hinted), "
                f"e.g. {missing[:3]}"
            )
        if placement == "system":
            from opsd.pi.signals.inject import secret_system_content

            hints = {k: secret_system_content(v) for k, v in hints.items()}
    df["hint"] = df["extra_info"].map(lambda e: hints.get(e["task_id"], ""))
    return df


def write_eval_parquet(df, out_path: Path, schema_parquet: Path) -> None:
    """Write with the peer eval parquet's arrow schema + a large_string ``hint`` column.

    A pandas read_parquet -> to_parquet round-trip downgrades large_string to string, and
    datasets.concatenate_datasets then refuses to align the eval sets; casting to the peer schema avoids it.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pq.read_schema(schema_parquet)
    if "hint" in schema.names:
        schema = schema.remove(schema.get_field_index("hint"))
    schema = schema.append(pa.field("hint", pa.large_string()))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False), out_path)


def build_hint_eval_sets(
    base_parquet: str,
    sets: Sequence[str],
    task_ids_path: str,
    hint_dir: Optional[str] = None,
    out_dir: Optional[str] = None,
    placement: str = "user",
    schema_parquet: Optional[str] = None,
) -> List[Path]:
    import pandas as pd

    if placement not in ("user", "system"):
        raise ValueError(f"placement must be 'user' or 'system', got {placement!r}")
    base = Path(base_parquet).expanduser()
    if not base.is_file():
        raise FileNotFoundError(f"base parquet not found: {base}")
    names = parse_sets(sets)
    task_ids = load_task_ids(task_ids_path)
    # Resolve every input before writing anything, so a bad set never leaves a partial output.
    hint_paths = {n: (None if n == NOHINT else hint_file(hint_dir, n)) for n in names}
    if schema_parquet:
        schema_src = Path(schema_parquet).expanduser()
    else:
        peer = base.parent / "validation.parquet"
        schema_src = peer if peer.is_file() else base
    out = Path(out_dir).expanduser() if out_dir else base.parent

    base_df = pd.read_parquet(base)
    # Build (and validate) every set first: a set with a missing hint must not leave the others written.
    frames = {name: build_eval_set(base_df, name, task_ids, hint_paths[name], placement) for name in names}
    written: List[Path] = []
    for name, df in frames.items():
        path = out / f"eval_hint_{name}.parquet"
        write_eval_parquet(df, path, schema_src)
        print(
            f"PI-utility eval set {name}: {len(df)} tasks, placement={placement}, "
            f"hinted rows={int((df['hint'] != '').sum())} -> {path}"
        )
        written.append(path.resolve())
    return written


def hydra_list(paths: Sequence[Path]) -> str:
    return "[" + ",".join(f"'{p}'" for p in paths) + "]"


def main(argv: Optional[Sequence[str]] = None) -> List[Path]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--base-parquet",
        required=True,
        help="ALFWorld parquet with one row per requested task id (normally train.parquet).",
    )
    ap.add_argument(
        "--sets",
        nargs="+",
        default=[os.environ["EVAL_HINT_SETS"]] if os.environ.get("EVAL_HINT_SETS") else None,
        help="hint-set names ('+', ',' or space separated). Default: $EVAL_HINT_SETS.",
    )
    ap.add_argument(
        "--hint-dir",
        default=os.environ.get("EVAL_HINT_DIR"),
        help="directory holding hints_<set>.json for every hinted set. Default: $EVAL_HINT_DIR.",
    )
    ap.add_argument(
        "--task-ids",
        default=os.environ.get("EVAL_HINT_TASK_IDS_PATH"),
        help="JSON list of task ids shared by every set. Default: $EVAL_HINT_TASK_IDS_PATH.",
    )
    ap.add_argument("--out-dir", default=None, help="output directory (default: the base parquet's directory).")
    ap.add_argument(
        "--placement",
        choices=("user", "system"),
        default=os.environ.get("ALFWORLD_HINT_PLACEMENT") or "user",
        help="must match environment.skyrl_gym.alfworld.hint_placement. "
        "Default: $ALFWORLD_HINT_PLACEMENT, else user.",
    )
    ap.add_argument(
        "--schema-parquet",
        default=None,
        help="peer eval parquet whose arrow schema the outputs share "
        "(default: validation.parquet next to the base parquet, else the base parquet).",
    )
    args = ap.parse_args(argv)
    if not args.sets:
        ap.error("--sets is required (or set EVAL_HINT_SETS)")
    if not args.task_ids:
        ap.error("--task-ids is required (or set EVAL_HINT_TASK_IDS_PATH)")
    written = build_hint_eval_sets(
        base_parquet=args.base_parquet,
        sets=args.sets,
        task_ids_path=args.task_ids,
        hint_dir=args.hint_dir,
        out_dir=args.out_dir,
        placement=args.placement,
        schema_parquet=args.schema_parquet,
    )
    print(hydra_list(written))
    return written


if __name__ == "__main__":
    main()

"""Build SkyRL-format parquet for ALFWorld (text-only AlfredTWEnv) RL training.

Scans ``$ALFWORLD_DATA/json_2.1.1/<split>/**/traj_data.json`` (the ALFRED game layout) and emits
one row per game. The prompt is left EMPTY and filled at runtime by ``AlfworldEnv.init()`` (which
resets the live game over HTTP and renders the SDAR ALFWorld template), so the parquet only needs
to pin each rollout to its game via ``reward_spec.ground_truth`` = abs path to that game's
``traj_data.json`` (its ``game.tw-pddl`` is the sibling the TextWorld engine loads).

Game paths come from ``$ALFWORLD_DATA`` (or ``--alfworld_data``) at BUILD time and are stored absolute,
so the parquet is machine-specific: rebuild it whenever ALFWORLD_DATA moves. The env servers load the
game from that path, and a stale path fails every ``/reset``. Rows are matched to hint artifacts and
task-id lists by ``extra_info.task_id``, which is portable.

Only games that already have a ``game.tw-pddl`` (i.e. generated & solvable) are emitted — that is
exactly the set the server's ``AlfredTWEnv.collect_game_files`` would register. Games whose path
contains ``movable`` or ``Sliced`` are skipped (AlfredTWEnv skips them too).

Splits (mapped to SkyRL file names like the AppWorld prep):
  train      -> train.parquet            ($ALFWORLD_DATA/json_2.1.1/train)
  valid_seen -> validation.parquet       ($ALFWORLD_DATA/json_2.1.1/valid_seen)
  valid_unseen (optional) -> validation_unseen.parquet

Prereq (one-time): the ALFWorld game data, downloaded by ``scripts/setup/setup_alfworld.sh``
(``alfworld-download -f`` into ``$ALFWORLD_DATA``). This script needs only pandas + pyarrow, so it runs
in either the SkyRL or the ALFWorld venv.

Usage:
    python opsd/alfworld/alfworld_dataset.py --local_dir ~/data/alfworld_parquet \
        --splits train valid_seen --task_types 1 2 3 4 5 6
"""

import argparse
import json
import logging
import os
from typing import List

import pandas as pd

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ALFRED task-type id -> traj_data["task_type"] string (alfworld TASK_TYPES).
TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}

# split name -> output parquet filename (validation.parquet is the SkyRL `data.val_data` default).
SPLIT_TO_FILE = {
    "train": "train.parquet",
    "valid_seen": "validation.parquet",
    "valid_unseen": "validation_unseen.parquet",
}


def _is_solvable_game(game_dir: str) -> bool:
    """Mirror AlfredTWEnv.collect_game_files: needs a game.tw-pddl with solvable=true."""
    pddl = os.path.join(game_dir, "game.tw-pddl")
    if not os.path.exists(pddl):
        return False
    try:
        with open(pddl) as f:
            gamedata = json.load(f)
    except Exception:
        return False
    return bool(gamedata.get("solvable", False))


def build_rows(data_root: str, split: str, allowed_types: List[str]) -> pd.DataFrame:
    split_dir = os.path.join(data_root, "json_2.1.1", split)
    if not os.path.isdir(split_dir):
        raise RuntimeError(f"Split dir not found: {split_dir} (check $ALFWORLD_DATA and alfworld-download).")

    rows = []
    skipped_type = skipped_unsolvable = skipped_movable = 0
    for root, _dirs, files in os.walk(split_dir):
        if "traj_data.json" not in files:
            continue
        if "movable" in root or "Sliced" in root:
            skipped_movable += 1
            continue
        json_path = os.path.join(root, "traj_data.json")
        try:
            with open(json_path) as f:
                traj = json.load(f)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to read %s: %s", json_path, e)
            continue
        task_type = traj.get("task_type")
        if task_type not in allowed_types:
            skipped_type += 1
            continue
        if not _is_solvable_game(root):
            skipped_unsolvable += 1
            continue

        task_id = traj.get("task_id") or os.path.basename(root)
        rows.append(
            {
                # In-domain (train + valid_seen) keeps data_source="alfworld" so the in-domain eval
                # bucket stays eval/alfworld/* — matching earlier runs (was renamed to
                # alfworld_valid_seen, which broke overlay + doubled the eval namespace). Only the OOD
                # split gets its own bucket (eval/alfworld_valid_unseen/*), and only when EVAL_OOD=true.
                "data_source": "alfworld" if split in ("train", "valid_seen") else f"alfworld_{split}",
                # Prompt filled at runtime by AlfworldEnv.init() (resets the live game over HTTP).
                "prompt": [{"role": "user", "content": ""}],
                "env_class": "alfworld",
                "reward_spec": {"method": "rule", "ground_truth": os.path.abspath(json_path)},
                "extra_info": {"task_type": task_type, "task_id": task_id, "split": split},
            }
        )

    logger.info(
        "split=%s: %d games kept (skipped %d wrong-type, %d unsolvable/no-pddl, %d movable/sliced)",
        split,
        len(rows),
        skipped_type,
        skipped_unsolvable,
        skipped_movable,
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SkyRL parquet from the ALFWorld game files.")
    parser.add_argument("--local_dir", default="~/data/alfworld_parquet", help="Output dir for the parquet files.")
    parser.add_argument(
        "--alfworld_data",
        default=os.environ.get("ALFWORLD_DATA") or "~/data/alfworld",
        help="ALFWorld data root holding json_2.1.1/ (default: $ALFWORLD_DATA, else ~/data/alfworld). "
        "Game paths in the parquet are absolute paths under it.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid_seen"],
        choices=list(SPLIT_TO_FILE.keys()),
        help="Splits to build (train, valid_seen, valid_unseen).",
    )
    parser.add_argument(
        "--task_types",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6],
        help="ALFRED task-type ids to include (1..6).",
    )
    parser.add_argument("--max_rows", type=int, default=None, help="Cap rows per split (debugging).")
    args = parser.parse_args()

    data_root = os.path.expanduser(args.alfworld_data)
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    logger.info("ALFWorld data root: %s", os.path.abspath(data_root))
    allowed_types = [TASK_TYPES[t] for t in args.task_types if t in TASK_TYPES]
    if not allowed_types:
        raise ValueError(f"No valid task_types in {args.task_types}; valid ids are {sorted(TASK_TYPES)}.")

    for split in args.splits:
        df = build_rows(data_root, split, allowed_types)
        if args.max_rows is not None:
            df = df.iloc[: args.max_rows]
        out = os.path.join(local_dir, SPLIT_TO_FILE[split])
        df.to_parquet(out, index=False)
        logger.info("Wrote %d rows -> %s", len(df), out)


if __name__ == "__main__":
    main()

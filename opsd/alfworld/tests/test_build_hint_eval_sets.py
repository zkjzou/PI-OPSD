"""CPU tests for build_hint_eval_sets: per-set PI-utility eval parquets over a shared task-id list."""

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from opsd.alfworld import build_hint_eval_sets as bhe
from opsd.pi.signals.inject import secret_system_content

TASK_IDS = ["t0", "t1", "t2", "t3"]


def _rows(split: str, ids):
    return [
        {
            "data_source": "alfworld",
            "prompt": [{"role": "user", "content": ""}],
            "env_class": "alfworld",
            "reward_spec": {"method": "rule", "ground_truth": f"/data/json_2.1.1/{split}/{tid}/traj_data.json"},
            "extra_info": {"task_type": "pick_and_place_simple", "task_id": tid, "split": split},
        }
        for tid in ids
    ]


@pytest.fixture
def data(tmp_path):
    data_dir = tmp_path / "parquet"
    data_dir.mkdir()
    pd.DataFrame(_rows("train", TASK_IDS)).to_parquet(data_dir / "train.parquet", index=False)
    pd.DataFrame(_rows("valid_seen", ["v0"])).to_parquet(data_dir / "validation.parquet", index=False)
    hint_dir = tmp_path / "hints"
    hint_dir.mkdir()
    # v1 (plain strings) and v2 ({"task": ...}) artifacts both load.
    (hint_dir / "hints_pc_key.json").write_text(json.dumps({t: f"key hint {t}" for t in TASK_IDS}))
    (hint_dir / "hints_raw.json").write_text(
        json.dumps({"__schema__": "opsd_hints/v2", **{t: {"task": f"raw hint {t}", "meta": {}} for t in TASK_IDS}})
    )
    ids_path = tmp_path / "task_ids.json"
    ids_path.write_text(json.dumps(["t2", "t0", "t3"]))  # a subset, not in base order
    return data_dir, hint_dir, ids_path


def test_parse_sets_splits_dedupes_and_validates():
    assert bhe.parse_sets(["nohint+pc_key,raw", "raw pc_key"]) == ["nohint", "pc_key", "raw"]
    with pytest.raises(ValueError):
        bhe.parse_sets(["ok+../evil"])
    with pytest.raises(ValueError):
        bhe.parse_sets(["+,"])


def test_builds_one_parquet_per_set(data, capsys):
    data_dir, hint_dir, ids_path = data
    written = bhe.main(
        [
            "--base-parquet",
            str(data_dir / "train.parquet"),
            "--sets",
            "nohint+pc_key+raw",
            "--hint-dir",
            str(hint_dir),
            "--task-ids",
            str(ids_path),
            "--placement",
            "user",
        ]
    )
    assert [p.name for p in written] == [
        "eval_hint_nohint.parquet",
        "eval_hint_pc_key.parquet",
        "eval_hint_raw.parquet",
    ]
    assert all(p.parent == data_dir.resolve() for p in written)
    # The LAST stdout line is the hydra list for data.val_data.
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last == "[" + ",".join(f"'{p}'" for p in written) + "]"

    frames = {p.name: pd.read_parquet(p) for p in written}
    for name, df in frames.items():
        set_name = name[len("eval_hint_") : -len(".parquet")]
        # Same tasks in every set, in base-parquet order; each set has its own eval bucket.
        assert [e["task_id"] for e in df["extra_info"]] == ["t0", "t2", "t3"]
        assert set(df["data_source"]) == {f"alfworld_{set_name}"}
        # Everything else is carried over from the base rows.
        assert list(df["env_class"]) == ["alfworld"] * 3
        assert df["reward_spec"].iloc[0]["ground_truth"].endswith("/train/t0/traj_data.json")
    assert list(frames["eval_hint_nohint.parquet"]["hint"]) == ["", "", ""]
    assert list(frames["eval_hint_pc_key.parquet"]["hint"]) == ["key hint t0", "key hint t2", "key hint t3"]
    assert list(frames["eval_hint_raw.parquet"]["hint"]) == ["raw hint t0", "raw hint t2", "raw hint t3"]

    # Peer (validation.parquet) schema + a large_string hint column, identical across sets.
    peer = pq.read_schema(data_dir / "validation.parquet")
    for p in written:
        schema = pq.read_schema(p)
        assert schema.field("hint").type == pa.large_string()
        assert schema.field("data_source").type == peer.field("data_source").type
        assert schema.names == peer.names + ["hint"]


def test_system_placement_wraps_the_teacher_block(data, tmp_path):
    data_dir, hint_dir, ids_path = data
    out = tmp_path / "out"
    (p,) = bhe.build_hint_eval_sets(
        str(data_dir / "train.parquet"),
        ["pc_key"],
        str(ids_path),
        hint_dir=str(hint_dir),
        out_dir=str(out),
        placement="system",
    )
    assert p.parent == out.resolve()
    df = pd.read_parquet(p)
    assert df["hint"].iloc[0] == secret_system_content("key hint t0")
    assert "<Secret information>" in df["hint"].iloc[0]


def test_missing_hint_fails_and_writes_nothing(data, tmp_path):
    data_dir, hint_dir, ids_path = data
    (hint_dir / "hints_partial.json").write_text(json.dumps({"t0": "only t0", "t2": "   "}))
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="no hint"):
        bhe.build_hint_eval_sets(
            str(data_dir / "train.parquet"),
            ["nohint", "partial"],
            str(ids_path),
            hint_dir=str(hint_dir),
            out_dir=str(out),
        )
    assert not out.exists() or not list(out.iterdir())


def test_unknown_task_id_and_missing_artifact_fail(data, tmp_path):
    data_dir, hint_dir, ids_path = data
    bad_ids = tmp_path / "bad_ids.json"
    bad_ids.write_text(json.dumps(["t0", "not_in_base"]))
    with pytest.raises(ValueError, match="requested task ids"):
        bhe.build_hint_eval_sets(str(data_dir / "train.parquet"), ["nohint"], str(bad_ids))
    with pytest.raises(FileNotFoundError):
        bhe.build_hint_eval_sets(
            str(data_dir / "train.parquet"), ["no_such_set"], str(ids_path), hint_dir=str(hint_dir)
        )
    with pytest.raises(ValueError, match="--hint-dir"):
        bhe.build_hint_eval_sets(str(data_dir / "train.parquet"), ["pc_key"], str(ids_path), hint_dir=None)

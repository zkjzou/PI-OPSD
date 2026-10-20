"""opsd.skyrl_ext.evaluate.dump_per_dataset_eval_results -- the fork-added cases of SkyRL
tests/train/test_trainer_utils.py: every dumped row carries its rollout's env_metrics."""

import json
from unittest.mock import Mock, mock_open, patch

import numpy as np

from opsd.skyrl_ext.evaluate import dump_per_dataset_eval_results


def _rows(mock_file):
    return [
        json.loads(c[0][0])
        for c in mock_file.return_value.write.call_args_list
        if c[0][0].strip().startswith("{") and "input_prompt" in c[0][0]
    ]


def _dump_dir():
    mock_dump_dir = Mock()
    mock_dump_dir.__truediv__ = Mock(side_effect=lambda x: f"mock_path/{x}")
    return mock_dump_dir


def _tokenizer():
    tok = Mock()
    tok.decode.side_effect = lambda x: f"decoded_{x}"
    return tok


@patch("builtins.open", new_callable=mock_open)
def test_dump_per_dataset_eval_results_includes_env_metrics(mock_file):
    """Per-rollout env_metrics reach the dump (the logged eval metrics are only their means), numpy -> python."""
    outs = {
        "prompt_token_ids": [[1], [2]],
        "response_ids": [[3], [4]],
        "rewards": [1.0, 0.0],
        "stop_reasons": ["stop", "stop"],
        "env_metrics": [{"success": np.float64(1.0), "task_completed": 1.0}, {"success": 0.0}],
    }
    dump_per_dataset_eval_results(_dump_dir(), _tokenizer(), outs, ["ds", "ds"], ["e", "e"], [{}, {}], {})
    rows = _rows(mock_file)
    assert [r["env_metrics"] for r in rows] == [{"success": 1.0, "task_completed": 1.0}, {"success": 0.0}]
    # the new field sits between env_extras and data_source, as in the fork
    assert list(rows[0]) == [
        "input_prompt",
        "output_response",
        "score",
        "stop_reason",
        "env_class",
        "env_extras",
        "env_metrics",
        "data_source",
    ]


@patch("builtins.open", new_callable=mock_open)
def test_dump_per_dataset_eval_results_without_env_metrics(mock_file):
    outs = {"prompt_token_ids": [[1]], "response_ids": [[3]], "rewards": [1.0], "stop_reasons": ["stop"]}
    dump_per_dataset_eval_results(_dump_dir(), _tokenizer(), outs, ["ds"], ["e"], [{}], {})
    rows = _rows(mock_file)
    assert rows[0]["env_metrics"] is None

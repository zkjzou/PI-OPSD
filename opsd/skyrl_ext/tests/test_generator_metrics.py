"""opsd.skyrl_ext.evaluate.get_metrics_from_generator_output: env-success pass@n and pass_at_1.

The first test is the fork-added case of SkyRL tests/train/generators/test_generator_output_utils.py (plus
pass_at_1); the golden test replays the fork's function on the edge inputs in fork_parity/eval/helpers.json.
"""

import json
import math

import pytest

from opsd.skyrl_ext.evaluate import get_metrics_from_generator_output
from opsd.skyrl_ext.tests.test_eval_support import fixture_dir
from skyrl.train.generators.base import GeneratorOutput
from skyrl.train.generators.utils import (
    get_metrics_from_generator_output as upstream_get_metrics_from_generator_output,
)


def test_pass_at_n_uses_env_success_when_present():
    """pass@n counts an example solved only if an env reports trajectory-level success==1 for one of
    its rollouts (e.g. AppWorld: all unit tests pass), NOT merely reward>0 (fractional partial reward).
    """
    # 2 examples, n=2. All rewards are fractional and > 0 (each rollout passed >= 1 subtest), so the
    # legacy reward>0 rule would give pass@2 = 1.0. But env success says only one rollout of 'a' fully
    # solved its task -> success-based pass@2 = 0.5.
    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1], [2], [3], [4]],
        "response_ids": [[1], [2], [3], [4]],
        "rewards": [1.0, 0.5, 0.3, 0.4],
        "loss_masks": [[1], [1], [1], [1]],
        "stop_reasons": ["stop"] * 4,
        "rollout_logprobs": None,
        "env_metrics": [
            {"success": 1.0, "partial_completion": 1.0},  # a: fully solved
            {"success": 0.0, "partial_completion": 0.5},  # a: partial only
            {"success": 0.0, "partial_completion": 0.3},  # b: partial only
            {"success": 0.0, "partial_completion": 0.4},  # b: partial only
        ],
    }
    uids = ["a", "a", "b", "b"]
    metrics = get_metrics_from_generator_output(generator_output, uids)
    assert metrics["avg_score"] == pytest.approx(0.55)  # reward-based means are unchanged
    assert metrics["mean_positive_reward"] == pytest.approx(0.55)
    assert metrics["pass_at_n"] == 0.5  # success-based: only 'a' has a solved rollout
    assert metrics["pass_at_1"] == 0.25  # 1 of 4 rollouts solved

    # Falls back to reward>0 when env_metrics is absent -> all rewards > 0 -> pass@n = 1.0
    go_no_env = {k: v for k, v in generator_output.items() if k != "env_metrics"}
    assert get_metrics_from_generator_output(go_no_env, uids)["pass_at_n"] == 1.0
    assert get_metrics_from_generator_output(go_no_env, uids)["pass_at_1"] == 1.0

    # Falls back to reward>0 when the env does not report a "success" key (e.g. gsm8k)
    go_no_success = dict(generator_output)
    go_no_success["env_metrics"] = [{"partial_completion": 0.5}] * 4
    assert get_metrics_from_generator_output(go_no_success, uids)["pass_at_n"] == 1.0

    # ... and when only SOME rollouts report it
    go_mixed = dict(generator_output)
    go_mixed["env_metrics"] = [{"success": 1.0}, {}, {"success": 0.0}, {"success": 0.0}]
    assert get_metrics_from_generator_output(go_mixed, uids)["pass_at_n"] == 1.0


def test_reward_rule_matches_upstream_without_env_success():
    """Without env success, avg_score / pass_at_n / mean_positive_reward are exactly upstream's (incl. token-level
    rewards, where the last token's reward decides)."""
    go = {
        "response_ids": [[1, 2], [3], [4, 5, 6], [7]],
        "rewards": [[0.0, 1.0], [-1.0], [0.5, 0.5, 0.0], [0.0]],
        "env_metrics": [{"num_turns": 1.0}] * 4,
    }
    uids = ["a", "a", "b", "c"]
    ours = get_metrics_from_generator_output(go, uids)
    theirs = upstream_get_metrics_from_generator_output(go, uids)
    assert {k: ours[k] for k in theirs} == dict(theirs)
    assert ours["pass_at_n"] == pytest.approx(1 / 3)  # only a's first rollout ends on a positive token reward
    assert ours["pass_at_1"] == pytest.approx(1 / 4)


def test_empty_rewards_still_raise():
    with pytest.raises(ValueError):
        get_metrics_from_generator_output({"rewards": [], "env_metrics": []}, [])


def _same(a, b):
    return (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)) or a == b


def test_matches_fork_on_helper_goldens():
    cases = json.loads((fixture_dir() / "helpers.json").read_text())["get_metrics_from_generator_output"]
    assert len(cases) == 24
    for case in cases:
        got = dict(get_metrics_from_generator_output(case["generator_output"], case["uids"]))
        assert list(got) == list(case["result"])
        assert all(_same(got[k], case["result"][k]) for k in got), (case, got)

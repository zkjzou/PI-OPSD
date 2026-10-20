"""Unit tests for opsd.algorithms.grpo_signal.gradient_signal_metrics (moved from the fork's
skyrl/train/utils/grpo_signal.py, unchanged).

The trainer-level parity (the same metrics the fork's RayPPOTrainer logged, on real batches through the real
estimator) lives in opsd/skyrl_ext/tests/test_trainer_grpo_signal.py; these pin the pure function itself, plus
the step-wise last-step row selection that feeds it.
"""

import math

import pytest

from opsd.algorithms.grpo_signal import gradient_signal_metrics


def test_groups_are_classified_signal_solved_failed():
    scores = [1, 1, 1, 1] + [0, 0, 0, 0] + [1, 0, 1, 0] + [1, 0.5, 0, 0.5]
    index = ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4
    out = gradient_signal_metrics(scores, index)
    assert out == {
        "grpo/groups": 4.0,
        "grpo/groups_with_signal": 2.0,
        "grpo/signal_frac": 0.5,
        "grpo/all_solved_frac": 0.25,
        "grpo/all_failed_frac": 0.25,
        # population std over the two live groups: 0.5 and sqrt(0.125)
        "grpo/reward_std_mean": (0.5 + math.sqrt(0.125)) / 2,
        "grpo/group_size_mean": 4.0,
    }


def test_token_dilution_figures():
    scores = [1, 1, 1, 0, 0, 1]
    index = ["dead", "dead", "dead", "live", "live", "live"]
    tokens = [10, 20, 30, 1, 2, 3]
    out = gradient_signal_metrics(scores, index, tokens)
    assert out["grpo/signal_token_frac"] == pytest.approx(6 / 66)
    assert out["grpo/dead_tokens"] == 60.0
    assert out["grpo/all_solved_frac"] == 0.5


def test_no_token_keys_without_counts_or_with_zero_tokens():
    scores, index = [1, 0, 1, 0], ["a", "a", "b", "b"]
    assert "grpo/signal_token_frac" not in gradient_signal_metrics(scores, index)
    out = gradient_signal_metrics(scores, index, [0, 0, 0, 0])
    assert "grpo/signal_token_frac" not in out and "grpo/dead_tokens" not in out


@pytest.mark.parametrize(
    "scores,index",
    [
        ([], []),  # empty batch
        ([1, 0], ["a"]),  # length mismatch
        ([1, 0, 1], ["a", "b", "c"]),  # every group has one member: a non-group estimator
    ],
)
def test_returns_empty_when_not_applicable(scores, index):
    assert gradient_signal_metrics(scores, index) == {}


def test_float_rounding_is_not_signal():
    # 0.1 + 0.2 != 0.3 in floating point; the variance is ~1e-34, far below eps
    out = gradient_signal_metrics([0.1 + 0.2, 0.3, 0.3, 0.1 + 0.2], ["g"] * 4)
    assert out["grpo/groups_with_signal"] == 0.0
    assert out["grpo/all_solved_frac"] == 1.0
    assert out["grpo/reward_std_mean"] == 0.0


def test_constant_non_positive_group_counts_as_failed():
    out = gradient_signal_metrics([-1, -1, 0, 0], ["neg", "neg", "zero", "zero"])
    assert out["grpo/all_failed_frac"] == 1.0
    assert out["grpo/all_solved_frac"] == 0.0


def test_uneven_groups_and_prefix():
    out = gradient_signal_metrics([1, 0, 1, 1, 0], [7, 7, 8, 8, 9], prefix="x/")
    assert set(out) == {
        "x/groups",
        "x/groups_with_signal",
        "x/signal_frac",
        "x/all_solved_frac",
        "x/all_failed_frac",
        "x/reward_std_mean",
        "x/group_size_mean",
    }
    assert out["x/groups"] == 3.0
    assert out["x/group_size_mean"] == pytest.approx(5 / 3)
    assert out["x/signal_frac"] == pytest.approx(1 / 3)


def test_stepwise_inputs_keep_last_steps_and_drop_padding():
    import torch

    from opsd.skyrl_ext.trainer import gradient_signal_inputs
    from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch

    # two trajectories of uid "a" (2 steps, 1 step) and two of uid "b" (1 step, 2 steps), then 2 padding rows.
    # Intermediate steps carry a reward that must not count; padding rows (copies of row 0) must not either.
    rewards = torch.tensor([[0.5, 0], [0, 1], [0, 0], [1, 0], [0.5, 0], [0, 1], [0.5, 0], [0.5, 0]])
    mask = torch.tensor([[1, 1], [1, 0], [1, 1], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1]], dtype=torch.float)
    data = TrainingInputBatch({"rewards": rewards, "response_mask": mask})
    data.metadata = {
        "uids": ["a", "a", "a", "b", "b", "b", "pad0", "pad1"],
        "is_last_step": [False, True, True, True, False, True, True, True],
        "pad_size": 2,
    }
    scores, uids, tokens = gradient_signal_inputs(data, step_wise=True)
    assert scores == [1.0, 0.0, 1.0, 1.0]
    assert uids == ["a", "a", "b", "b"]
    assert tokens == [1.0, 2.0, 1.0, 2.0]
    # trajectory-level groups: "a" is mixed, "b" all solved
    out = gradient_signal_metrics(scores, uids, tokens)
    assert out["grpo/groups"] == 2.0 and out["grpo/group_size_mean"] == 2.0
    assert out["grpo/signal_frac"] == 0.5 and out["grpo/all_solved_frac"] == 0.5

    # the non-step-wise path keeps every unpadded row
    scores, uids, tokens = gradient_signal_inputs(data, step_wise=False)
    assert uids == ["a", "a", "a", "b", "b", "b"] and len(scores) == len(tokens) == 6

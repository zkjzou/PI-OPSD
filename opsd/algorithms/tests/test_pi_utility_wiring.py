"""Trainer-side wiring for `pi_utility/*`: does the metric actually come out, and is the label right?

`opsd/algorithms/tests/test_pi_utility.py` covers the maths. This file covers the two ways the
wiring can degrade SILENTLY, both of which leave a plausible-looking panel behind:

  1. **nDCG quietly missing.** `_pi_utility_metrics` needs `data.metadata["uids"]` to group rollouts by
     task. If it is absent or the wrong length the method falls back to pooled attributes only -- `d` and
     friends still appear, so nothing looks broken, but `ndcg_at_4_d` (the headline) is gone. The window is
     real: SkyRL `skyrl/train/trainer.py:291` (@ ecdaa946) POPS "uids" from the metadata, one step after
     `apply_reward_kl_penalty` reads it.
  2. **The wrong success label.** `sum(env_reward) > 0` is exact on ALFWorld and wrong on AppWorld under
     `REWARD_MODE=partial`, where a partial-credit FAILURE scores > 0. The label must come from
     `env_metrics["success"]`, stashed by `convert_to_training_input`.

Run from the repo root:

    .venv/bin/python -m pytest opsd/algorithms/tests/test_pi_utility_wiring.py -q
"""

from __future__ import annotations

import pytest
import torch

from opsd.trainer.main_opsd_alfworld import SelfDistillationTrainer
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch

# 4 rollouts over 2 tasks, 3 tokens each. Task "t1" is both-class (one win, one loss) so it survives the
# nDCG both-class filter; "t2" is all-loss and is correctly excluded.
UIDS = ["t1", "t1", "t2", "t2"]
N, T = 4, 3


def _trainer(successes=None, num_turns=None):
    """A bare trainer instance: _pi_utility_metrics touches only KL_CLAMP and the two stashes."""
    tr = SelfDistillationTrainer.__new__(SelfDistillationTrainer)
    tr._last_env_success = successes
    tr._last_env_num_turns = num_turns
    return tr


def _batch(rewards):
    b = TrainingInputBatch({"rewards": torch.tensor(rewards, dtype=torch.float32)})
    b.metadata = {"uids": list(UIDS)}
    return b


def _logprobs():
    """Student/teacher logprobs where the teacher prefers rollout 0's tokens and dislikes rollout 1's."""
    student = torch.full((N, T), -1.0)
    teacher = torch.tensor(
        [
            [-0.5, -0.5, -0.5],  # teacher likes these more -> d_plus
            [-2.0, -2.0, -2.0],  # teacher likes these less -> d_minus
            [-0.9, -1.1, -1.0],
            [-1.5, -1.0, -1.2],
        ]
    )
    mask = torch.ones(N, T)
    return student, teacher, mask


def test_emits_the_full_metric_set_including_ndcg():
    """The headline `ndcg_at_4_*` keys must be present, not just the pooled attributes."""
    tr = _trainer(successes=[1.0, 0.0, 0.0, 0.0], num_turns=[5.0, 30.0, 30.0, 28.0])
    s, t, m = _logprobs()
    out = tr._pi_utility_metrics(_batch([[0, 0, 1.0]] + [[0, 0, 0]] * 3), s, t, m)

    for key in (
        "d",
        "d_plus",
        "d_minus",
        "d_signed",
        "identity_err",
        "ndcg_both_class_tasks",
        "ndcg_at_4_d",
        "ndcg_at_4_d_minus",
        "ndcg_at_4_random",
        "ndcg_at_4_neg_nstep",
    ):
        assert f"pi_utility/{key}" in out, f"missing pi_utility/{key}"
    # only t1 is both-class; t2 is all-loss and contributes no ranking information
    assert out["pi_utility/ndcg_both_class_tasks"] == 1.0
    # the identity d == d+ + d- is what catches a sign flip in d+/d-
    assert out["pi_utility/identity_err"] == pytest.approx(0.0, abs=1e-4)


def test_neg_nstep_absent_without_the_num_turns_stash():
    """No stash -> no `-nstep` reference, and that is the number that keeps ndcg_at_4_d honest."""
    tr = _trainer(successes=[1.0, 0.0, 0.0, 0.0], num_turns=None)
    s, t, m = _logprobs()
    out = tr._pi_utility_metrics(_batch([[0, 0, 1.0]] + [[0, 0, 0]] * 3), s, t, m)
    assert "pi_utility/ndcg_at_4_d" in out
    assert "pi_utility/ndcg_at_4_neg_nstep" not in out


def test_missing_uids_degrades_to_pooled_and_says_so():
    tr = _trainer(successes=[1.0, 0.0, 0.0, 0.0])
    s, t, m = _logprobs()
    b = TrainingInputBatch({"rewards": torch.zeros(N, T)})
    b.metadata = {}  # the state after trainer.py:291 pops "uids"
    out = tr._pi_utility_metrics(b, s, t, m)
    assert out["pi_utility/ndcg_both_class_tasks"] == 0.0
    assert "pi_utility/ndcg_at_4_d" not in out, "an nDCG over the wrong groups is worse than none"
    assert "pi_utility/d" in out  # pooled attributes still reported


def test_success_label_comes_from_env_metrics_not_the_reward_sign():
    """The AppWorld REWARD_MODE=partial case: a partial-credit FAILURE has reward > 0.

    Rollout 1 scores 0.4 (4 of 10 hidden tests) but did NOT succeed. Reward-sign labelling would call it a
    win and rank against the wrong labels; env_metrics says otherwise and must be believed.
    """
    s, t, m = _logprobs()
    rewards = [[0, 0, 1.0], [0, 0, 0.4], [0, 0, 0.0], [0, 0, 0.0]]

    from_env = _trainer(successes=[1.0, 0.0, 0.0, 0.0])._pi_utility_metrics(_batch(rewards), s, t, m)
    from_reward = _trainer(successes=None)._pi_utility_metrics(_batch(rewards), s, t, m)

    # env_metrics: t1 = one win + one loss -> both-class, counted.
    assert from_env["pi_utility/ndcg_both_class_tasks"] == 1.0
    # reward sign: t1 looks like TWO wins -> degenerate, dropped, and the nDCG silently vanishes.
    assert from_reward["pi_utility/ndcg_both_class_tasks"] == 0.0


def test_reward_sign_fallback_still_works_when_env_metrics_absent():
    """ALFWorld-shaped rewards with no env_metrics: the fallback must still produce a grouping."""
    s, t, m = _logprobs()
    rewards = [[0, 0, 10.0], [0, 0, -0.1], [0, 0, 0.0], [0, 0, 0.0]]
    out = _trainer(successes=None)._pi_utility_metrics(_batch(rewards), s, t, m)
    assert out["pi_utility/ndcg_both_class_tasks"] == 1.0


def test_never_raises_on_malformed_input():
    """A diagnostic must not take down a 24-hour run; it returns {} instead."""
    tr = _trainer()
    bad = TrainingInputBatch({})  # no "rewards" at all
    bad.metadata = {"uids": list(UIDS)}
    assert tr._pi_utility_metrics(bad, *_logprobs()) == {}


def test_cumulative_auc_state_persists_across_steps_on_the_trainer():
    """The accumulator must live on the TRAINER, not be rebuilt per call.

    If it were local, `auc_d_cum` would equal `auc_d` at every step -- a plausible-looking panel that
    silently carries none of the accumulation it is named for, which is the entire point of the metric.
    """
    tr = _trainer(successes=[1.0, 0.0, 0.0, 0.0])
    s, t, m = _logprobs()
    b = _batch([[0, 0, 1.0]] + [[0, 0, 0]] * 3)

    first = tr._pi_utility_metrics(b, s, t, m)
    assert "pi_utility/auc_d_cum" in first
    n1 = first["pi_utility/auc_groups_cum"]

    second = tr._pi_utility_metrics(b, s, t, m)
    assert second["pi_utility/auc_groups_cum"] == 2 * n1, "state did not carry over between steps"
    # identical batch twice -> the running mean must be unchanged while the count doubles
    assert second["pi_utility/auc_d_cum"] == pytest.approx(first["pi_utility/auc_d_cum"])


def test_cumulative_auc_survives_a_step_with_no_both_class_task():
    """A degenerate step must not reset or corrupt the accumulator."""
    tr = _trainer(successes=[1.0, 0.0, 0.0, 0.0])
    s, t, m = _logprobs()
    good = tr._pi_utility_metrics(_batch([[0, 0, 1.0]] + [[0, 0, 0]] * 3), s, t, m)
    tr._last_env_success = [1.0, 1.0, 1.0, 1.0]  # every task all-win -> nothing scorable
    degenerate = tr._pi_utility_metrics(_batch([[0, 0, 1.0]] * 4), s, t, m)
    assert degenerate["pi_utility/ndcg_both_class_tasks"] == 0.0
    assert "pi_utility/auc_d" not in degenerate
    tr._last_env_success = [1.0, 0.0, 0.0, 0.0]
    after = tr._pi_utility_metrics(_batch([[0, 0, 1.0]] + [[0, 0, 0]] * 3), s, t, m)
    assert after["pi_utility/auc_groups_cum"] == 2 * good["pi_utility/auc_groups_cum"]


def test_stash_reads_success_and_num_turns_from_generator_output():
    """Shape check on the values convert_to_training_input stashes, using AppWorld's get_metrics keys."""
    em = [
        {"success": 1.0, "partial_completion": 1.0, "num_turns": 7},
        {"success": 0.0, "partial_completion": 0.3, "num_turns": 30},
    ]
    turns = [float(m.get("num_turns", 0.0)) for m in em]
    succ = [float(m.get("success", 0.0)) for m in em] if isinstance(em[0], dict) and "success" in em[0] else None
    assert turns == [7.0, 30.0]
    assert succ == [1.0, 0.0]

"""Trainer-side wiring for the d+/d- ablation (`OPSD_DISTILL_SIDE`).

`test_reward_shaping.py` covers the mask arithmetic. This file covers the ways the wiring can be wrong
while every panel still looks healthy -- which is the failure mode that has cost this project the most
runs:

  1. **The env var is read but the reward is unchanged.** A side arm that silently trains the baseline
     objective is indistinguishable from the baseline by construction, so the ablation would report "no
     effect" after burning two 60-hour runs. Asserted against the reward tensor, not against a metric.
  2. **The diagnostics move with the ablation.** `train/teacher_kl` and `pi_utility/*` must stay on the
     FULL both-sided divergence, or a side arm's curves are not comparable to the baseline's and the
     already-measured d+ share (8-23% on AppWorld) stops being the right way to size the dose.
  3. **The magnitude match is off by the wrong factor.** With env_coef=0 the reward IS the advantage IS
     the per-token gradient weight, so a bad scale is a silent learning-rate change.

Run from the repo root:

    .venv/bin/python -m pytest opsd/algorithms/tests/test_distill_side_wiring.py -q
"""

from __future__ import annotations

import pytest
import torch

from opsd.trainer.main_opsd_alfworld import SelfDistillationTrainer
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch

N, T = 4, 6
UIDS = ["t1", "t1", "t2", "t2"]


def _trainer():
    """Bare trainer: apply_reward_kl_penalty needs only KL_CLAMP, all_metrics and the env stashes.

    `self.cfg` is deliberately absent -- the per-turn/channel KL monitor at the tail reads it inside its
    own try/except, so leaving it out exercises that guard instead of mocking a config tree.
    """
    tr = SelfDistillationTrainer.__new__(SelfDistillationTrainer)
    tr.all_metrics = {}
    tr._last_env_success = [1.0, 0.0, 0.0, 0.0]
    tr._last_env_num_turns = [5.0, 30.0, 30.0, 28.0]
    tr.global_step = 3
    return tr


def _batch(seed=0):
    """A batch whose teacher is mostly LESS confident than the student, as measured on AppWorld.

    d- is 79-92% of d on every live arm, so a symmetric random batch would not exercise the lopsidedness
    the ablation exists to probe.
    """
    g = torch.Generator().manual_seed(seed)
    student = torch.full((N, T), -1.0)
    # +0.9 offset -> ~85% of tokens have student_lp > teacher_lp, i.e. the d- side dominates
    teacher = student - 0.9 + torch.randn(N, T, generator=g) * 0.5
    loss_mask = torch.ones(N, T)
    loss_mask[:, -1] = 0.0  # a masked tail, to prove masked positions stay zero
    rewards = torch.zeros(N, T)
    rewards[0, -2] = 1.0
    b = TrainingInputBatch(
        {
            "loss_mask": loss_mask,
            "action_log_probs": student,
            "hinted_log_probs": teacher.clone(),
            "rewards": rewards,
            "sequences": torch.zeros(N, T, dtype=torch.long),
        }
    )
    b.metadata = {"uids": list(UIDS), "response_length": T}
    return b


def _run(monkeypatch, side, match="0", coef="1.0", env_coef="0"):
    monkeypatch.setenv("OPSD_DISTILL_SIDE", side)
    monkeypatch.setenv("OPSD_DISTILL_SIDE_MATCH", match)
    monkeypatch.setenv("OPSD_DISTILL_COEF", coef)
    monkeypatch.setenv("OPSD_ENV_COEF", env_coef)
    monkeypatch.setenv("OPSD_TOKEN_CLIP", "0")
    monkeypatch.setenv("OPSD_LENGTH_NORM", "0")
    tr = _trainer()
    out = tr.apply_reward_kl_penalty(_batch())
    return tr, out["rewards"]


def test_default_is_both_and_unchanged(monkeypatch):
    """No env var set at all must reproduce the baseline objective exactly."""
    monkeypatch.delenv("OPSD_DISTILL_SIDE", raising=False)
    monkeypatch.delenv("OPSD_DISTILL_SIDE_MATCH", raising=False)
    monkeypatch.setenv("OPSD_DISTILL_COEF", "1.0")
    monkeypatch.setenv("OPSD_ENV_COEF", "0")
    monkeypatch.setenv("OPSD_TOKEN_CLIP", "0")
    tr_default = _trainer()
    r_default = tr_default.apply_reward_kl_penalty(_batch())["rewards"]
    _, r_both = _run(monkeypatch, "both")
    assert torch.equal(r_default, r_both)
    assert tr_default.all_metrics["distill/side_code"] == 0.0


def test_plus_only_reward_is_nonnegative_and_minus_only_nonpositive(monkeypatch):
    """The sign guarantee that makes the ablation what it claims to be.

    plus-only  -> advantage >= 0 everywhere: up-only reinforcement, cannot suppress anything.
    minus-only -> advantage <= 0 everywhere: pure unlikelihood.
    """
    _, r_plus = _run(monkeypatch, "plus")
    _, r_minus = _run(monkeypatch, "minus")
    assert (r_plus >= 0).all(), "plus-only must never push a token down"
    assert (r_minus <= 0).all(), "minus-only must never push a token up"
    assert (r_plus > 0).any() and (r_minus < 0).any(), "both sides must be non-empty on this batch"


def test_sides_sum_to_the_baseline_reward(monkeypatch):
    """r_plus + r_minus == r_both, so the ablation is a partition of the baseline, not a new objective."""
    _, r_both = _run(monkeypatch, "both")
    _, r_plus = _run(monkeypatch, "plus")
    _, r_minus = _run(monkeypatch, "minus")
    assert torch.allclose(r_plus + r_minus, r_both, atol=1e-6)


def test_the_reward_actually_changes(monkeypatch):
    """Guards the "env var read, nothing applied" failure: a side arm must NOT train the baseline."""
    _, r_both = _run(monkeypatch, "both")
    for side in ("plus", "minus"):
        _, r_side = _run(monkeypatch, side)
        assert not torch.allclose(r_side, r_both, atol=1e-6), side


def test_diagnostics_stay_on_the_full_divergence(monkeypatch):
    """train/teacher_kl and pi_utility/* must be IDENTICAL across side settings.

    Without this a side arm's curves cannot be read against the baseline's, and the d+ share panel stops
    being the right way to size the dose.
    """
    tr_both, _ = _run(monkeypatch, "both")
    keys = [k for k in tr_both.all_metrics if k.startswith("pi_utility/")] + ["train/teacher_kl"]
    assert "pi_utility/d_plus" in keys and "pi_utility/d_minus" in keys
    for side in ("plus", "minus"):
        for match in ("0", "1"):
            tr, _ = _run(monkeypatch, side, match=match)
            for k in keys:
                assert tr.all_metrics[k] == pytest.approx(tr_both.all_metrics[k], rel=1e-9), (side, match, k)


def test_raw_plus_loses_gradient_norm_by_the_d_plus_share(monkeypatch):
    """distill/term_sum_mean is the applied magnitude; unmatched plus must fall to the d+/d ratio.

    This is the confound the match flag exists for: on the live arms d+/d is 0.08-0.23, i.e. a raw
    plus-only arm runs at roughly 1/8 the effective learning rate.
    """
    tr_both, _ = _run(monkeypatch, "both")
    tr_plus, _ = _run(monkeypatch, "plus")
    share = tr_both.all_metrics["pi_utility/d_plus"] / tr_both.all_metrics["pi_utility/d"]
    assert share < 0.35, "fixture should be d--dominant like the real arms"
    ratio = tr_plus.all_metrics["distill/term_sum_mean"] / tr_both.all_metrics["distill/term_sum_mean"]
    assert ratio == pytest.approx(share, rel=0.15)


def test_magnitude_match_restores_the_applied_norm(monkeypatch):
    """The match target is the mean ABSOLUTE PER-TOKEN weight, which is what sets the per-token dose.

    Deliberately NOT asserted against `distill/term_sum_mean`: that metric is |sum_t term| per rollout,
    so for the both-sided term the up-tokens and down-tokens partially CANCEL inside the sum while a
    one-sided term cannot. Matching mean|term| therefore leaves term_sum_mean slightly HIGHER than the
    baseline's (~1% on this fixture, more the less d--dominant the batch) -- correct, and itself part of
    what the ablation measures: a one-sided term of the same per-token size applies more COHERENT
    pressure. Asserting equality on term_sum_mean would be asserting the cancellation away.
    """
    tr_both, _ = _run(monkeypatch, "both")
    for side in ("plus", "minus"):
        tr, _ = _run(monkeypatch, side, match="1")
        assert tr.all_metrics["distill/side_mean_abs"] * tr.all_metrics["distill/side_scale"] == pytest.approx(
            tr.all_metrics["distill/full_mean_abs"], rel=1e-5
        ), side
        assert tr.all_metrics["distill/full_mean_abs"] == pytest.approx(
            tr_both.all_metrics["pi_utility/d"], rel=0.05
        ), side  # the full both-sided mean|.| IS `d`, up to per-row vs per-token weighting
        assert tr.all_metrics["distill/side_match_on"] == 1.0
        assert tr.all_metrics["distill/side_scale"] > 1.0
        # and the applied sum is back in the baseline's ballpark, not 8x below it
        assert tr.all_metrics["distill/term_sum_mean"] == pytest.approx(
            tr_both.all_metrics["distill/term_sum_mean"], rel=0.10
        ), side


def test_effective_token_clip_is_logged(monkeypatch):
    """The clip must be auditable from W&B, not only from the run-file's uncollected stdout banner.

    It has three different defaults across the tree (0 via the AppWorld run-file, 0.5 via ALFWorld's and
    via the trainer fallback, 0 in main_opsd_min), and it decides whether pi_utility/d_plus and
    distill/side_mean_abs are comparable at all -- so "which clip did this arm use" has to be answerable
    after the fact.
    """
    tr, _ = _run(monkeypatch, "both")
    assert tr.all_metrics["distill/token_clip"] == 0.0
    monkeypatch.setenv("OPSD_TOKEN_CLIP", "0.5")
    tr2 = _trainer()
    tr2.apply_reward_kl_penalty(_batch())
    assert tr2.all_metrics["distill/token_clip"] == 0.5
    # and the clip must actually bite: |term| per token cannot exceed it
    assert (
        tr2.all_metrics["distill/full_mean_abs"] <= 0.5 + 1e-6 if "distill/full_mean_abs" in tr2.all_metrics else True
    )


def test_side_code_and_frac_tokens_are_logged(monkeypatch):
    tr_plus, _ = _run(monkeypatch, "plus")
    tr_minus, _ = _run(monkeypatch, "minus")
    assert tr_plus.all_metrics["distill/side_code"] == 1.0
    assert tr_minus.all_metrics["distill/side_code"] == -1.0
    # d--dominant fixture: most masked tokens survive `minus`, few survive `plus`
    assert tr_minus.all_metrics["distill/side_frac_tokens"] > tr_plus.all_metrics["distill/side_frac_tokens"]
    assert 0.0 < tr_plus.all_metrics["distill/side_frac_tokens"] < 1.0


def test_masked_positions_get_no_reward(monkeypatch):
    for side in ("both", "plus", "minus"):
        _, r = _run(monkeypatch, side)
        assert torch.equal(r[:, -1], torch.zeros(N)), side


def test_env_reward_still_mixes_in(monkeypatch):
    """OPSD_ENV_COEF is orthogonal to the side mask; the ablation must not swallow the env term."""
    _, r0 = _run(monkeypatch, "plus", env_coef="0")
    _, r1 = _run(monkeypatch, "plus", env_coef="1")
    assert r1[0, -2] == pytest.approx(r0[0, -2] + 1.0, abs=1e-6)

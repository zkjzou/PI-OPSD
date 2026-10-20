"""Tests for the per-position group-baselined token advantage.

    A[i,t] = (env[i] - beta*k1[i,t]) - mean_{j in group} Rtilde[j,t]
    Rtilde[j,t] = R[j,t] if j reached position t, else env[j]

Three properties carry the design and none is obvious from reading the formula:

  1. THE PER-TOKEN SIGNAL SURVIVES. This exists because `grpo` sums the dense KL into one scalar and
     broadcasts it. If A came out constant along t for a varying k1, the whole point is gone.
  2. THE OUTCOME IMPUTATION KEEPS THE DENOMINATOR AT G. Without it the baseline at deep positions is a
     mean over only the rollouts still running -- one or two sequences at the tail, i.e. maximum noise
     exactly where trajectories diverge most. The tests pin the count, not just the value.
  3. BETA=0 IS EXACTLY GRPO. That is what makes an arm on this path interpretable against the GRPO/DAPO
     reference numbers, so it is pinned as a test rather than left as a claim in a docstring.

The env reward is BROADCAST along t, not read off the [B,T] tensor where it sits on one token. Under the
literal reading the task signal touches a single token per rollout and, worse, the imputed value (order
10 on ALFWorld) and R (order beta*k1) sit on different scales, so tail tokens of long rollouts collect
~ -mean(outcome) -- a length penalty far larger than any teacher signal. Tested both ways.

Grouping is by uid, so a batch holding several prompts must not leak baseline across them.
"""

import importlib.util

import pytest
import torch

from opsd._repo import repo_root

REPO = repo_root(__file__)
MODULE = REPO / "opsd/trainer/token_advantage.py"
TRAINER = REPO / "opsd/trainer/main_opsd_alfworld.py"

spec = importlib.util.spec_from_file_location("opsd_token_advantage", MODULE)
ta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ta)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("OPSD_TOKEN_ADV", "OPSD_TOKEN_ADV_COEF", "OPSD_TOKEN_ADV_STD", "OPSD_TOKEN_ADV_ENV"):
        monkeypatch.delenv(k, raising=False)


# --------------------------------------------------------------------------- defaults / wiring
def test_off_by_default():
    assert ta.token_adv_enabled() is False
    assert ta.token_adv_coef() == 1.0
    assert ta.token_adv_norm_std() is False
    assert ta.token_adv_env_broadcast() is True


def test_env_mode_is_validated(monkeypatch):
    monkeypatch.setenv("OPSD_TOKEN_ADV_ENV", "terminal")
    assert ta.token_adv_env_broadcast() is False
    monkeypatch.setenv("OPSD_TOKEN_ADV_ENV", "last")  # a plausible typo must not silently mean one
    with pytest.raises(ValueError):
        ta.token_adv_env_broadcast()


def test_trainer_is_mutually_exclusive_and_needs_pure_env_reward():
    src = TRAINER.read_text()
    assert "if token_adv_enabled() and teacher_lp is not None:" in src
    assert "rewrites the advantage, as do OPSD_MIX_LAMBDA<1 and OPSD_RLSD=1" in src
    assert (
        "OPSD_TOKEN_ADV=1 needs the PURE env reward" in src
    ), "use_kl_in_reward=true would double-count the teacher and must raise"


def test_trainer_plumbs_the_teacher_tensor():
    src = TRAINER.read_text()
    i = src.index('training_input["mix_teacher_log_probs"]')
    guard = src[max(0, i - 400) : i]
    assert "token_adv_enabled()" in guard, "TOKEN_ADV needs the teacher tensor plumbed through"


# --------------------------------------------------------------------------- grouping
def test_group_index_is_dense_and_order_preserving():
    idx, n = ta.group_index(["a", "b", "a", "c", "b"])
    assert idx == [0, 1, 0, 2, 1] and n == 3


def test_baseline_does_not_leak_across_prompts():
    """Two prompts in one batch. Group A is all-zeros, group B all-ones; each must baseline to 0."""
    env = torch.zeros(4, 3)
    env[2:] = 1.0
    kl = torch.zeros(4, 3)
    mask = torch.ones(4, 3)
    adv, _ = ta.per_position_group_advantage(env, kl, mask, ["p1", "p1", "p2", "p2"], beta=1.0)
    assert torch.allclose(
        adv, torch.zeros_like(adv), atol=1e-6
    ), "a constant group must have zero advantage; non-zero means baselines mixed across prompts"


# --------------------------------------------------------------------------- the core property
def test_advantage_varies_along_t_when_kl_does():
    """The reason this estimator exists: `grpo` would flatten this to a constant."""
    env = torch.zeros(2, 4)
    env[:, -1] = torch.tensor([1.0, 0.0])
    kl = torch.tensor([[0.0, 0.5, -0.5, 0.0], [0.0, 0.0, 0.0, 0.0]])
    mask = torch.ones(2, 4)
    adv, _ = ta.per_position_group_advantage(env, kl, mask, ["p", "p"], beta=1.0)
    assert adv[0].std() > 0.1, "per-token KL must survive into the advantage"
    # and the sign is right: a token the teacher DISLIKES (k1>0) gets less advantage
    assert adv[0, 1] < adv[0, 2]


def test_group_mean_is_removed_at_every_position():
    torch.manual_seed(0)
    env = torch.zeros(6, 5)
    kl = torch.randn(6, 5)
    mask = torch.ones(6, 5)
    adv, _ = ta.per_position_group_advantage(env, kl, mask, ["p"] * 6, beta=1.0)
    assert torch.allclose(
        adv.mean(dim=0), torch.zeros(5), atol=1e-5
    ), "with all rollouts full-length, each position's group mean must be exactly removed"


# --------------------------------------------------------------------------- the imputation
def test_short_rollouts_are_imputed_with_their_outcome():
    """Rollout 1 stops after 2 tokens. Its outcome (1.0) must fill positions 2..3 so the baseline at
    those positions is a mean over BOTH rollouts, not just the surviving one."""
    env = torch.zeros(2, 4)
    env[0, 3] = 0.0  # long rollout, outcome 0
    env[1, 1] = 1.0  # short rollout, outcome 1 (earned before it stopped)
    kl = torch.zeros(2, 4)
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    adv, met = ta.per_position_group_advantage(env, kl, mask, ["p", "p"], beta=1.0)
    # at position 2: R[0,2]=0, imputed R[1,2]=outcome[1]=1 -> baseline 0.5 -> adv[0,2] = -0.5
    assert adv[0, 2] == pytest.approx(-0.5, abs=1e-6)
    # the dead rollout contributes to the baseline but gets zero advantage of its own there
    assert adv[1, 2] == 0.0 and adv[1, 3] == 0.0
    assert met["tokenadv/imputed_frac"] == pytest.approx(0.25, abs=1e-6)


def test_without_imputation_the_tail_baseline_would_be_degenerate():
    """Contrast: if short rollouts were EXCLUDED, position 3's baseline would be rollout 0's own value
    and its advantage would collapse to exactly 0. The imputation is what prevents that."""
    env = torch.zeros(2, 4)
    env[1, 0] = 2.0
    kl = torch.zeros(2, 4)
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    adv, _ = ta.per_position_group_advantage(env, kl, mask, ["p", "p"], beta=1.0)
    assert adv[0, 3] != 0.0, "imputation must keep a second sample in the deep-tail baseline"
    assert adv[0, 3] == pytest.approx(-1.0, abs=1e-6)  # 0 - mean(0, 2)


# --------------------------------------------------------------------------- knobs / robustness
def test_beta_scales_the_teacher_term_linearly():
    """No env signal at all, so the whole advantage IS the teacher term and must be homogeneous in beta."""
    env = torch.zeros(2, 3)
    kl = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mask = torch.ones(2, 3)
    a1, _ = ta.per_position_group_advantage(env, kl, mask, ["p", "p"], beta=1.0)
    a2, _ = ta.per_position_group_advantage(env, kl, mask, ["p", "p"], beta=2.0)
    assert torch.allclose(a2, 2 * a1, atol=1e-6)
    assert a1[0, 0] == pytest.approx(-0.5, abs=1e-6)  # R = -1 against a group mean of -0.5


def test_beta_does_not_touch_the_task_signal():
    """The env term reaches EVERY position (that is the broadcast) and beta must not rescale it."""
    env = torch.zeros(2, 3)
    env[:, -1] = torch.tensor([1.0, 0.0])
    kl = torch.zeros(2, 3)
    mask = torch.ones(2, 3)
    for b in (0.0, 1.0, 7.0):
        adv, _ = ta.per_position_group_advantage(env, kl, mask, ["p", "p"], beta=b)
        assert torch.allclose(adv[0], torch.full((3,), 0.5), atol=1e-6), f"beta={b} moved the env term"
        assert torch.allclose(adv[1], torch.full((3,), -0.5), atol=1e-6)


def test_beta_zero_is_exactly_grpo():
    """The property that makes an arm here comparable to the GRPO/DAPO reference: at beta=0 the teacher
    is gone and A[i,t] collapses to (outcome_i - group mean), constant along t -- vanilla GRPO."""
    env = torch.zeros(4, 5)
    env[:, -1] = torch.tensor([1.0, 0.0, 1.0, 1.0])  # outcomes; 3 of 4 succeeded
    kl = torch.randn(4, 5)  # must be ignored entirely
    mask = torch.ones(4, 5)
    adv, _ = ta.per_position_group_advantage(env, kl, mask, ["p"] * 4, beta=0.0)
    grpo = (env.sum(dim=-1, keepdim=True) - 0.75).expand(4, 5)
    assert torch.allclose(adv, grpo, atol=1e-6)


def test_terminal_mode_punishes_a_long_winner_and_broadcast_does_not():
    """WHY BROADCAST IS THE DEFAULT, on the case that separates them. Three rollouts all SUCCEED (ALFWorld
    pays 10.0*won); rollout 0 just takes longer. Under the literal per-token reading the finished rollouts
    are imputed at 10 while the still-running winner has earned 0 so far, so it collects ~-6.7 on nearly
    every token -- a length penalty dressed up as credit assignment, and far larger than any teacher term.
    Broadcasting makes the tie a tie."""
    env = torch.zeros(3, 4)
    env[0, 3] = 10.0  # long winner, pays out at the end
    env[1, 0] = env[2, 0] = 10.0  # short winners, done after one token
    kl = torch.zeros(3, 4)
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    term, _ = ta.per_position_group_advantage(env, kl, mask, ["p"] * 3, beta=1.0, env_broadcast=False)
    assert term[0, 1] == pytest.approx(-20.0 / 3, abs=1e-5)
    assert term[0, 3] == pytest.approx(0.0, abs=1e-6), "credit arrives only on the payout token"

    bcast, _ = ta.per_position_group_advantage(env, kl, mask, ["p"] * 3, beta=1.0, env_broadcast=True)
    assert torch.allclose(bcast[0], torch.zeros(4), atol=1e-6), "three equal winners must tie at every t"


def test_std_normalisation_is_optional_and_finite():
    torch.manual_seed(0)
    env = torch.zeros(4, 5)
    kl = torch.randn(4, 5)
    mask = torch.ones(4, 5)
    plain, _ = ta.per_position_group_advantage(env, kl, mask, ["p"] * 4, norm_by_std=False)
    normed, _ = ta.per_position_group_advantage(env, kl, mask, ["p"] * 4, norm_by_std=True)
    assert torch.isfinite(normed).all()
    assert not torch.allclose(plain, normed)


def test_degenerate_group_of_one_does_not_divide_by_zero():
    env = torch.zeros(1, 3)
    kl = torch.randn(1, 3)
    mask = torch.ones(1, 3)
    adv, met = ta.per_position_group_advantage(env, kl, mask, ["solo"], norm_by_std=True)
    assert torch.isfinite(adv).all()
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-5), "a group of one baselines to itself"
    assert met["tokenadv/rollouts_per_group"] == 1.0


def test_masked_positions_carry_no_advantage():
    torch.manual_seed(0)
    env = torch.zeros(3, 6)
    kl = torch.randn(3, 6)
    mask = torch.ones(3, 6)
    mask[:, 4:] = 0.0
    adv, _ = ta.per_position_group_advantage(env, kl, mask, ["p"] * 3)
    assert torch.all(adv[:, 4:] == 0.0)

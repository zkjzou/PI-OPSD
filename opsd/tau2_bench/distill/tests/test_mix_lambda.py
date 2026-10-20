"""DAPO x OPSD advantage mix (OPSD_MIX_LAMBDA) on the τ² trainer.

    A = λ · A_GRPO + (1−λ) · A_SDPO,   A_SDPO = −k1(student ‖ frozen hinted teacher)   (per token, masked)

Before the integration, ``Tau2OPSDTrainer`` re-implemented the mix, its λ schedule and the live
``pi_utility/*`` diagnostics as overrides on an older ``SelfDistillationTrainer``. The shared base now does
all of that itself, so those overrides would apply the rewrite a second time inside ``super()``. They are
deleted; these tests pin the result:

* STRUCTURE -- the subclass no longer overrides the advantage / reward / training-input path.
* GOLDEN PARITY -- on fixed inputs (``golden_mix_lambda.json``) the merged trainer reproduces, exactly, the
  advantages and metrics the pre-integration τ² trainer produced (λ ∈ {0, 0.5, 0.99}, the 0.5 -> 0.99
  schedule across its window, a ramp that starts at λ=1, and the λ=1 / vanilla-OPSD controls) -- and the
  fixture is sensitive enough that applying the mix twice fails it.
* EXACTLY ONCE -- one ``pi_utility_metrics`` call and one mix per step.
* GUARDS -- every recipe error the τ² subclass refused (no_op estimator, use_kl_in_reward, a frozen teacher
  that ran on the live policy, a missing teacher tensor) still fails loudly, now from the shared base.
"""

import pytest
import torch

from opsd.algorithms import pi_utility
from opsd.algorithms.reward_shaping import per_token_reverse_kl
from opsd.tau2_bench.distill import trainer as T
from opsd.tau2_bench.distill.tests._golden import (
    assert_matches_golden,
    clear_opsd_env,
    load,
    make_batch,
    make_trainer,
    run_case,
    set_env,
)
from opsd.trainer import main_opsd_alfworld as base_mod
from opsd.trainer.main_opsd_alfworld import SelfDistillationTrainer
from opsd.trainer.mix_schedule import mix_lambda
from skyrl.train.trainer import RayPPOTrainer

FIX = load("golden_mix_lambda.json")
CASES = sorted(FIX["cases"])
INPUTS = FIX["inputs"]
MIX_CASES = [n for n in CASES if FIX["cases"][n]["metrics"].get("mix/grpo_over_sdpo") is not None]
# λ ∉ {0, 1}: there a second application changes the advantages (λ=0 and λ=1 are idempotent).
NON_IDEMPOTENT = [n for n in MIX_CASES if 0.0 < FIX["cases"][n]["metrics"]["mix/lambda"] < 1.0]

# The `mix/*` panel the τ² dashboards read (same keys as the ALFWorld/AppWorld arms).
TAU2_MIX_KEYS = (
    "mix/lambda",
    "mix/a_grpo_abs",
    "mix/a_sdpo_abs",
    "mix/a_sdpo_max_abs",
    "mix/contrib_grpo",
    "mix/contrib_sdpo",
    "mix/grpo_over_sdpo",
    "mix/advantage_abs",
    "mix/sign_disagree_frac",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    clear_opsd_env(monkeypatch)


# --- structure ------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    ["compute_advantages_and_returns", "convert_to_training_input", "apply_reward_kl_penalty", "_tau2_pi_utility"],
)
def test_deleted_overrides_are_gone(name):
    assert name not in T.Tau2OPSDTrainer.__dict__, f"{name} re-overridden: the base already does it"


def test_advantage_path_is_inherited_from_the_shared_base():
    assert T.Tau2OPSDTrainer.compute_advantages_and_returns is SelfDistillationTrainer.compute_advantages_and_returns
    assert T.Tau2OPSDTrainer.apply_reward_kl_penalty is SelfDistillationTrainer.apply_reward_kl_penalty
    assert T.Tau2OPSDTrainer.convert_to_training_input is SelfDistillationTrainer.convert_to_training_input
    # the two τ²-specific overrides stay
    assert {"eval", "postprocess_generator_output"} <= set(T.Tau2OPSDTrainer.__dict__)
    # the duplicate helpers are gone too: one implementation of the mix maths and its metrics
    assert not hasattr(T, "mix_metrics") and not hasattr(T, "opsd_mix_lambda")


# --- golden parity --------------------------------------------------------------------------------
@pytest.mark.parametrize("name", CASES)
def test_reproduces_the_pre_integration_tau2_trainer(monkeypatch, name):
    case = FIX["cases"][name]
    t, out, rewards_in = run_case(monkeypatch, FIX, name)
    assert_matches_golden(case, out["advantages"], t.all_metrics)
    assert torch.equal(out["advantages"], out["returns"])
    if "rewards_after_shaping" in case:  # the vanilla reward path: k1 shaping, OPSD_ENV_COEF=0
        assert torch.allclose(rewards_in, torch.tensor(case["rewards_after_shaping"]), rtol=1e-6, atol=1e-6)
    # every teacher tensor is consumed before the batch reaches the policy workers
    assert "hinted_log_probs" not in out and "mix_teacher_log_probs" not in out


def _mix_again(lam, advantages, token_clip):
    """What the pre-fix subclass did on top of the merged base: mix the ALREADY-mixed advantage again."""
    student = torch.tensor(INPUTS["student_lp"])
    teacher = torch.tensor(INPUTS["teacher_lp"])
    loss = torch.tensor(INPUTS["loss_mask"], dtype=torch.float32)
    resp = torch.tensor(INPUTS["response_mask"], dtype=torch.float32)
    k1, _ = per_token_reverse_kl(student, teacher, loss, kl_clamp=T.Tau2OPSDTrainer.KL_CLAMP, token_clip=token_clip)
    return (lam * advantages + (1.0 - lam) * -k1) * resp


@pytest.mark.parametrize("name", NON_IDEMPOTENT)
def test_a_double_application_fails_the_golden(monkeypatch, name):
    case = FIX["cases"][name]
    _, out, _ = run_case(monkeypatch, FIX, name)
    token_clip = float({**FIX["common_env"], **case["env"]}["OPSD_TOKEN_CLIP"])
    doubled = _mix_again(case["metrics"]["mix/lambda"], out["advantages"], token_clip)
    with pytest.raises(AssertionError, match="pre-integration"):
        assert_matches_golden(case, doubled)


# --- exactly once ---------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name", ["mix_l05", "mix_l099", "sched_l05to099_s50", "sched_l05to099_s101", "vanilla_reward_path"]
)
def test_pi_utility_and_the_mix_run_exactly_once_per_step(monkeypatch, name):
    calls = {"pi_utility": 0, "k1": 0}
    real_pu, real_k1 = pi_utility.pi_utility_metrics, base_mod.per_token_reverse_kl

    def pu_spy(*a, **k):
        calls["pi_utility"] += 1
        return real_pu(*a, **k)

    def k1_spy(*a, **k):
        calls["k1"] += 1
        return real_k1(*a, **k)

    monkeypatch.setattr(pi_utility, "pi_utility_metrics", pu_spy)
    monkeypatch.setattr(base_mod, "per_token_reverse_kl", k1_spy)
    case = FIX["cases"][name]
    t, out, _ = run_case(monkeypatch, FIX, name)
    assert calls["pi_utility"] == 1
    # one per-token k1: the mix's A_SDPO (λ<1), or the reward shaping on the vanilla path; none at λ=1
    lam = case["metrics"].get("mix/lambda")
    want_k1 = 1 if (case["kind"] == "reward" or (lam is not None and lam < 1.0)) else 0
    assert calls["k1"] == want_k1
    assert_matches_golden(case, out["advantages"], t.all_metrics)


# --- the τ² dashboard keys --------------------------------------------------------------------------
def test_mix_panel_keys_are_stable(monkeypatch):
    t, _, _ = run_case(monkeypatch, FIX, "mix_l05")
    for k in TAU2_MIX_KEYS:
        assert k in t.all_metrics, k
    assert 0.0 <= t.all_metrics["mix/sign_disagree_frac"] <= 1.0


def test_lambda_one_tail_logs_lambda_but_no_inf_ratio(monkeypatch):
    t, out, _ = run_case(monkeypatch, FIX, "sched_l05to099_s101")
    assert t.all_metrics["mix/lambda"] == 1.0
    assert "mix/grpo_over_sdpo" not in t.all_metrics  # no inf logged on the DAPO tail
    # λ=1.0: A_mix == A_GRPO, bit for bit
    _, grpo, _ = run_case(monkeypatch, FIX, "grpo_no_rewrite")
    assert torch.equal(out["advantages"], grpo["advantages"])


def test_a_ramp_that_starts_at_one_is_still_a_mix_arm(monkeypatch):
    # the mode is decided over the whole schedule: at step 0 λ=1, but the arm must already be in mix mode
    set_env(monkeypatch, FIX, FIX["cases"]["sched_ramp_from1_s0"]["env"])
    assert T.env_anchored_mode() == "mix"


def test_lambda_zero_is_not_silently_one(monkeypatch):
    # the old `float(x) or 1.0` parse turned λ=0 (pure teacher) into pure GRPO
    monkeypatch.setenv("OPSD_MIX_LAMBDA", "0")
    assert mix_lambda(0) == 0.0


# --- guards: every recipe error fails LOUDLY at step 1 ----------------------------------------------
def _run(monkeypatch, env, *, with_teacher=True, step=0, **trainer_kw):
    set_env(monkeypatch, FIX, env)
    t = make_trainer("adv", step, INPUTS, **trainer_kw)
    return t, t.compute_advantages_and_returns(make_batch(INPUTS, with_teacher=with_teacher))


def test_vanilla_config_without_a_mix_is_the_base_path(monkeypatch):
    # a VANILLA OPSD config (no_op + use_kl_in_reward) with no mix must not trip any guard
    t, out = _run(monkeypatch, {}, with_teacher=False, use_kl_in_reward=True, estimator="no_op")
    assert torch.equal(out["advantages"], out["rewards"])  # no_op: advantages are the token rewards
    assert not any(k.startswith("mix/") for k in t.all_metrics)


def test_refuses_kl_in_reward(monkeypatch):
    with pytest.raises(ValueError, match="use_kl_in_reward"):
        _run(monkeypatch, {"OPSD_MIX_LAMBDA": "0.5"}, use_kl_in_reward=True)


def test_refuses_kl_in_reward_even_at_a_lambda_one_step_of_a_schedule(monkeypatch):
    env = FIX["cases"]["sched_l05to099_s101"]["env"]
    with pytest.raises(ValueError, match="use_kl_in_reward"):
        _run(monkeypatch, env, step=101, use_kl_in_reward=True)


def test_refuses_no_op_estimator(monkeypatch):
    with pytest.raises(ValueError, match="no_op has no A_GRPO"):
        _run(monkeypatch, {"OPSD_MIX_LAMBDA": "0.5"}, estimator="no_op")


def test_token_advantage_no_op_refusal_has_its_own_message(monkeypatch):
    # The token advantage never reads A_GRPO, so its grpo requirement is a policy choice; the error must
    # say so instead of claiming there is no A_GRPO to rewrite.
    with pytest.raises(ValueError, match="run the token advantage with advantage_estimator=grpo") as exc:
        _run(monkeypatch, {"OPSD_TOKEN_ADV": "1"}, estimator="no_op")
    assert "no A_GRPO to rewrite" not in str(exc.value)


def test_refuses_live_teacher_when_frozen_requested(monkeypatch):
    # no ref worker -> the teacher was scored by the live policy: the collapse config, must not train
    with pytest.raises(RuntimeError, match="live policy"):
        _run(monkeypatch, {"OPSD_MIX_LAMBDA": "0.5"}, frozen=0.0)


def test_missing_teacher_logprobs_is_loud(monkeypatch):
    with pytest.raises(KeyError, match="mix_teacher_log_probs"):
        _run(monkeypatch, {"OPSD_MIX_LAMBDA": "0.5"}, with_teacher=False)


def test_a_schedule_and_rlsd_refuse_together(monkeypatch):
    set_env(monkeypatch, FIX, {**FIX["cases"]["sched_ramp_from1_s0"]["env"], "OPSD_RLSD": "1"})
    with pytest.raises(ValueError, match="pick one"):
        T.env_anchored_mode()
    with pytest.raises(ValueError, match="pick one"):
        make_trainer("adv", 0, INPUTS).compute_advantages_and_returns(make_batch(INPUTS))


def test_mix_and_token_advantage_refuse_together(monkeypatch):
    set_env(monkeypatch, FIX, {"OPSD_MIX_LAMBDA": "0.5", "OPSD_TOKEN_ADV": "1"})
    with pytest.raises(ValueError, match="pick one"):
        T.env_anchored_mode()


def test_parity_runs_the_real_group_relative_estimator():
    # the parity above runs the REAL GRPO estimator, not a stub -- keep it that way
    assert RayPPOTrainer.compute_advantages_and_returns.__module__ == "skyrl.train.trainer"

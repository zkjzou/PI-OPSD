"""RLSD (OPSD_RLSD=1, arXiv:2604.03128) on the τ² OPSD trainer.

rlsd.py's math is tested in opsd/algorithms/tests/test_opsd_rlsd.py. Since the integration the τ²
trainer no longer wires RLSD itself: it inherits the shared base's ``compute_advantages_and_returns`` (a
τ² override on top of it would rewrite the advantage twice). These pin, for the τ² subclass:

* GOLDEN PARITY -- on fixed inputs (``golden_rlsd.json``) the merged trainer reproduces, exactly, the
  advantages and ``rlsd/*`` / ``pi_utility/*`` metrics the pre-integration τ² trainer produced at steps 0,
  25 and 60 of the default schedule (λ 0.5 -> 0 over 50 steps), and a double application fails that check;
* EXACTLY ONCE -- one ``rlsd_advantage`` and one ``pi_utility_metrics`` call per step;
* the shared guards, the mutual exclusion with the λ-mix, an untouched vanilla arm, and that DAPO's overlong
  penalty is re-applied on every env-anchored arm (RLSD, mix, token advantage) and only there.
"""

import types

import pytest
import torch

from opsd.algorithms import pi_utility
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
from opsd.trainer import rlsd
from skyrl.train.trainer import RayPPOTrainer

FIX = load("golden_rlsd.json")
CASES = sorted(FIX["cases"])
INPUTS = FIX["inputs"]
RLSD_ENV = FIX["cases"]["rlsd_s0"]["env"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    clear_opsd_env(monkeypatch)


def _tensors():
    return (
        torch.tensor(INPUTS["student_lp"]),
        torch.tensor(INPUTS["teacher_lp"]),
        torch.tensor(INPUTS["loss_mask"], dtype=torch.float32),
        torch.tensor(INPUTS["response_mask"], dtype=torch.float32),
    )


def _a_grpo(monkeypatch):
    """The real estimator's A_GRPO on the same inputs (no rewrite on)."""
    set_env(monkeypatch, FIX, {})
    t = make_trainer("adv", 0, INPUTS)
    return t.compute_advantages_and_returns(make_batch(INPUTS))["advantages"]


# --- golden parity --------------------------------------------------------------------------------
@pytest.mark.parametrize("name", CASES)
def test_reproduces_the_pre_integration_tau2_trainer(monkeypatch, name):
    case = FIX["cases"][name]
    t, out, _ = run_case(monkeypatch, FIX, name)
    assert_matches_golden(case, out["advantages"], t.all_metrics)
    assert torch.equal(out["advantages"], out["returns"])
    assert "hinted_log_probs" not in out and "mix_teacher_log_probs" not in out
    assert t.all_metrics["rlsd/lambda"] == pytest.approx(rlsd.rlsd_lambda(case["step"]))
    assert not any(k.startswith("mix/") for k in t.all_metrics)


@pytest.mark.parametrize("name", [n for n in CASES if rlsd.rlsd_lambda(FIX["cases"][n]["step"]) > 0])
def test_a_double_application_fails_the_golden(monkeypatch, name):
    case = FIX["cases"][name]
    _, out, _ = run_case(monkeypatch, FIX, name)
    student, teacher, loss, resp = _tensors()
    doubled, _ = rlsd.rlsd_advantage(
        out["advantages"], student, teacher, loss, resp, lam=rlsd.rlsd_lambda(case["step"]), eps_w=rlsd.rlsd_clip()
    )
    with pytest.raises(AssertionError, match="pre-integration"):
        assert_matches_golden(case, doubled)


def test_after_the_decay_window_it_is_exactly_dapo(monkeypatch):
    # λ_s = 0 from step 50: A_hat == A_GRPO bit for bit (and a re-application would be invisible here,
    # which is why the double-application check above uses the λ_s > 0 steps)
    _, out, _ = run_case(monkeypatch, FIX, "rlsd_s60")
    assert torch.equal(out["advantages"], _a_grpo(monkeypatch))


def test_sign_of_every_token_follows_the_env_advantage(monkeypatch):
    t, out, _ = run_case(monkeypatch, FIX, "rlsd_s0")
    a_grpo = _a_grpo(monkeypatch)
    rm = torch.tensor(INPUTS["response_mask"]).bool()
    assert torch.equal(torch.sign(out["advantages"][rm]), torch.sign(a_grpo[rm]))
    assert t.all_metrics["rlsd/sign_preserved"] == 1.0


# --- exactly once ---------------------------------------------------------------------------------
@pytest.mark.parametrize("name", CASES)
def test_rlsd_and_pi_utility_run_exactly_once_per_step(monkeypatch, name):
    calls = {"rlsd": 0, "pi_utility": 0}
    real_rlsd, real_pu = base_mod.rlsd_advantage, pi_utility.pi_utility_metrics

    def rlsd_spy(*a, **k):
        calls["rlsd"] += 1
        return real_rlsd(*a, **k)

    def pu_spy(*a, **k):
        calls["pi_utility"] += 1
        return real_pu(*a, **k)

    monkeypatch.setattr(base_mod, "rlsd_advantage", rlsd_spy)
    monkeypatch.setattr(pi_utility, "pi_utility_metrics", pu_spy)
    t, out, _ = run_case(monkeypatch, FIX, name)
    assert calls == {"rlsd": 1, "pi_utility": 1}
    assert_matches_golden(FIX["cases"][name], out["advantages"], t.all_metrics)


# --- exclusivity + the shared guards --------------------------------------------------------------
def _run(monkeypatch, env, **trainer_kw):
    set_env(monkeypatch, FIX, env)
    t = make_trainer("adv", 0, INPUTS, **trainer_kw)
    return t, t.compute_advantages_and_returns(make_batch(INPUTS))


def test_rlsd_and_mix_together_refuse(monkeypatch):
    with pytest.raises(ValueError, match="pick one"):
        _run(monkeypatch, {**RLSD_ENV, "OPSD_MIX_LAMBDA": "0.5"})


def test_refuses_kl_in_reward(monkeypatch):
    with pytest.raises(ValueError, match="OPSD_RLSD=1 needs use_kl_in_reward=false"):
        _run(monkeypatch, RLSD_ENV, use_kl_in_reward=True)


def test_refuses_no_op_estimator(monkeypatch):
    with pytest.raises(ValueError, match="no_op"):
        _run(monkeypatch, RLSD_ENV, estimator="no_op")


def test_refuses_live_teacher_when_frozen_requested(monkeypatch):
    with pytest.raises(RuntimeError, match="live policy"):
        _run(monkeypatch, RLSD_ENV, frozen=0.0)


def test_vanilla_opsd_is_untouched(monkeypatch):
    # a vanilla OPSD config (no_op + use_kl_in_reward) with RLSD off must not raise or log rlsd/mix
    set_env(monkeypatch, FIX, {"OPSD_RLSD": "0"})
    t = make_trainer("adv", 0, INPUTS, use_kl_in_reward=True, estimator="no_op")
    out = t.compute_advantages_and_returns(make_batch(INPUTS, with_teacher=False))
    assert torch.equal(out["advantages"], out["rewards"])
    assert not any(k.startswith(("rlsd/", "mix/")) for k in t.all_metrics)


# --- DAPO's overlong penalty is re-applied on the env-anchored arms ---------------------------------
def _postprocess(monkeypatch, env):
    monkeypatch.setattr(RayPPOTrainer, "postprocess_generator_output", lambda self, g, u: (g, u))
    set_env(monkeypatch, FIX, env)
    t = object.__new__(T.Tau2OPSDTrainer)
    t.cfg = types.SimpleNamespace(
        tau2=types.SimpleNamespace(max_response_tokens=33000),
        trainer=types.SimpleNamespace(
            algorithm=types.SimpleNamespace(
                overlong_buffer=types.SimpleNamespace(len=13000, penalty_factor=1.0, reward_range=1.0)
            )
        ),
    )
    t.all_metrics = {}
    g = {"rewards": [1.0, 1.0], "response_ids": [[0] * 18000, [0] * 26500]}
    out, _ = t.postprocess_generator_output(g, ["a", "b"])
    return t, [round(float(x), 4) for x in out["rewards"]]


@pytest.mark.parametrize(
    "env", [RLSD_ENV, {"OPSD_MIX_LAMBDA": "0.5"}, {"OPSD_TOKEN_ADV": "1"}], ids=["rlsd", "mix", "token_adv"]
)
def test_overlong_penalty_applies_on_env_anchored_arms(monkeypatch, env):
    t, rewards = _postprocess(monkeypatch, env)
    assert rewards == [1.0, 0.75]  # onset 20k; 26.5k -> -0.25
    assert "overlong_buffer/frac_penalized" in t.all_metrics


def test_overlong_penalty_skipped_on_vanilla_opsd(monkeypatch):
    # OPSD_ENV_COEF=0 discards the env reward there, so a penalty on it would be multiplied by zero
    t, rewards = _postprocess(monkeypatch, {})
    assert rewards == [1.0, 1.0]
    assert "overlong_buffer/frac_penalized" not in t.all_metrics

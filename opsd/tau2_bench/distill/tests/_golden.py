"""Shared harness for the τ² trainer's golden-parity tests (test_mix_lambda.py, test_rlsd_tau2.py).

The fixtures (``golden_*.json`` next to this file) hold fixed inputs plus the advantages and metrics the
PRE-INTEGRATION τ² trainer produced on them: its own λ-mix / RLSD / pi_utility overrides on its pre-merge
``SelfDistillationTrainer`` base, with the real ``RayPPOTrainer`` GRPO estimator. ``run_case`` replays a
fixture case through the merged trainer the same way: real estimator, a ``TrainingInputBatch`` carrying
the keys ``fwd_logprobs_values_reward`` publishes, and the ``convert_to_training_input`` stashes.
"""

from __future__ import annotations

import json
import os
import types
from pathlib import Path

import pytest
import torch

from opsd.tau2_bench.distill import trainer as T
from opsd.trainer.mix_schedule import mix_lambda_configured
from opsd.trainer.rlsd import rlsd_enabled
from opsd.trainer.token_advantage import token_adv_enabled
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch

HERE = Path(__file__).resolve().parent
TOL = dict(rtol=1e-6, atol=1e-6)


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def clear_opsd_env(monkeypatch) -> None:
    for k in list(os.environ):
        if k.startswith("OPSD_"):
            monkeypatch.delenv(k, raising=False)


def set_env(monkeypatch, fixture: dict, env: dict) -> None:
    clear_opsd_env(monkeypatch)
    for k, v in {**fixture["common_env"], **env}.items():
        monkeypatch.setenv(k, v)


def make_trainer(kind: str, step: int, inputs: dict, *, use_kl_in_reward=None, estimator=None, frozen=1.0):
    """kind "adv" = an env-anchored arm (grpo, use_kl_in_reward=false); "reward" = vanilla OPSD (no_op,
    use_kl_in_reward=true, apply_reward_kl_penalty runs first)."""
    reward_path = kind == "reward"
    alg = types.SimpleNamespace(
        use_kl_in_reward=reward_path if use_kl_in_reward is None else use_kl_in_reward,
        advantage_estimator=("no_op" if reward_path else "grpo") if estimator is None else estimator,
        gamma=1.0,
        lambd=1.0,
        grpo_norm_by_std=True,  # the τ² recipe's value
    )
    t = object.__new__(T.Tau2OPSDTrainer)
    t.cfg = types.SimpleNamespace(
        trainer=types.SimpleNamespace(algorithm=alg, eval_interval=0),
        generator=types.SimpleNamespace(step_wise_trajectories=False, max_turns=32),
    )
    t.global_step = step
    t.all_metrics = {"distill/teacher_is_frozen": frozen}
    # What convert_to_training_input stashes from env_metrics (τ²: success = raw_reward > 0).
    t._last_env_num_turns = list(inputs["num_turns"])
    t._last_env_success = [float(s) for s in inputs["success"]]
    return t


def make_batch(inputs: dict, *, with_teacher: bool = True) -> TrainingInputBatch:
    """The batch as fwd_logprobs_values_reward leaves it. Rewards: the binary outcome on the last
    response token. `mix_teacher_log_probs` is published under the same switches as in the trainer."""
    resp = torch.tensor(inputs["response_mask"], dtype=torch.float32)
    rewards = torch.zeros_like(resp)
    for i, row in enumerate(inputs["response_mask"]):
        last = max(j for j, m in enumerate(row) if m)
        rewards[i, last] = float(inputs["success"][i])
    teacher = torch.tensor(inputs["teacher_lp"], dtype=torch.float32)
    d = {
        "rewards": rewards,
        "response_mask": resp,
        "loss_mask": torch.tensor(inputs["loss_mask"], dtype=torch.float32),
        "action_log_probs": torch.tensor(inputs["student_lp"], dtype=torch.float32),
        "base_action_log_probs": torch.tensor(inputs["student_lp"], dtype=torch.float32),
        "values": None,
    }
    if with_teacher:
        d["hinted_log_probs"] = teacher.clone()
        if mix_lambda_configured() or rlsd_enabled() or token_adv_enabled():
            d["mix_teacher_log_probs"] = teacher.clone()
    b = TrainingInputBatch(d)
    b.metadata = {
        "uids": list(inputs["uids"]),
        "avg_response_length": float(resp.sum(-1).mean()),
        "pad_size": 0,
        "response_length": resp.shape[1],
    }
    return b


def run_case(monkeypatch, fixture: dict, name: str, **trainer_kw):
    """Set the case's env, run one step's reward + advantage path, return (trainer, batch, rewards_in)."""
    case = fixture["cases"][name]
    set_env(monkeypatch, fixture, case["env"])
    t = make_trainer(case["kind"], case["step"], fixture["inputs"], **trainer_kw)
    data = make_batch(fixture["inputs"])
    if case["kind"] == "reward":
        data = t.apply_reward_kl_penalty(data)
    rewards_in = data["rewards"].clone()
    out = t.compute_advantages_and_returns(data)
    return t, out, rewards_in


def assert_matches_golden(case: dict, advantages: torch.Tensor, metrics: dict | None = None) -> None:
    want = torch.tensor(case["advantages"], dtype=advantages.dtype)
    assert torch.allclose(
        advantages, want, **TOL
    ), f"advantages differ from the pre-integration τ² trainer:\n got {advantages.tolist()}\nwant {want.tolist()}"
    if metrics is not None:
        for k, v in case["metrics"].items():
            assert k in metrics, f"{k} no longer logged (τ² dashboard key)"
            assert metrics[k] == pytest.approx(v, **{"rel": TOL["rtol"], "abs": TOL["atol"]}), k

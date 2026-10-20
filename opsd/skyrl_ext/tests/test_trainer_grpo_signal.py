"""grpo/* gradient-signal parity: OPSDRayPPOTrainer vs the fork's edited RayPPOTrainer.

The golden (opsd/tests/fixtures/fork_parity/trainer/grpo_signal.json) holds, per case, a batch and what the fork's
``RayPPOTrainer.compute_advantages_and_returns`` produced on it: the full ``all_metrics`` dict (upstream's
``loss/*`` plus the fork's ``grpo/*``), the advantages and ``data.metadata["metrics"]``. Each case is replayed
through ``OPSDRayPPOTrainer`` on the same minimally stubbed trainer (``object.__new__`` + a SimpleNamespace cfg)
with the real GRPO / RLOO estimators. Cases: all-solved / all-failed / mixed / partial groups of 4, padding,
uneven and singleton groups, float-rounding ties, step-wise batches (last-step rows only, with padding).
"""

import json
import types

import pytest
import torch

from opsd._repo import fixtures_dir
from opsd.skyrl_ext.trainer import OPSDRayPPOTrainer
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.trainer import RayPPOTrainer

FIXTURE = fixtures_dir() / "fork_parity" / "trainer" / "grpo_signal.json"
GOLDEN = json.loads(FIXTURE.read_text())
CASES = sorted(GOLDEN["cases"])


def make_trainer(config: dict) -> OPSDRayPPOTrainer:
    t = object.__new__(OPSDRayPPOTrainer)
    alg = types.SimpleNamespace(
        advantage_estimator=config["advantage_estimator"],
        gamma=1.0,
        lambd=1.0,
        grpo_norm_by_std=config["grpo_norm_by_std"],
        use_kl_in_reward=False,
    )
    t.cfg = types.SimpleNamespace(
        trainer=types.SimpleNamespace(algorithm=alg),
        generator=types.SimpleNamespace(step_wise_trajectories=config["step_wise_trajectories"]),
    )
    t.all_metrics = {"pre/existing": 1.0}  # the capture's sentinel: metrics are merged, not replaced
    return t


def make_batch(inputs: dict) -> TrainingInputBatch:
    resp = torch.tensor(inputs["response_mask"], dtype=torch.float32)
    b = TrainingInputBatch(
        {"rewards": torch.tensor(inputs["rewards"], dtype=torch.float32), "response_mask": resp, "values": None}
    )
    b.metadata = {
        "uids": list(inputs["uids"]),
        "avg_response_length": inputs["avg_response_length"],
        "pad_size": inputs["pad_size"],
    }
    if inputs["is_last_step"] is not None:
        b.metadata["is_last_step"] = list(inputs["is_last_step"])
    return b


def run_case(name: str):
    case = GOLDEN["cases"][name]
    t = make_trainer(case["config"])
    out = t.compute_advantages_and_returns(make_batch(case["inputs"]))
    return case, t, out


@pytest.mark.parametrize("name", CASES)
def test_grpo_metrics_match_fork(name):
    case, t, _ = run_case(name)
    want = case["expected"]["all_metrics"]
    got_grpo = {k: v for k, v in t.all_metrics.items() if k.startswith("grpo/")}
    want_grpo = {k: v for k, v in want.items() if k.startswith("grpo/")}
    # pure-python arithmetic on identical inputs: bit-identical
    assert got_grpo == want_grpo
    # same keys in the same order (loss/* from upstream, then grpo/*), the rest numerically equal
    assert list(t.all_metrics) == list(want)
    assert t.all_metrics == pytest.approx(want, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("name", CASES)
def test_advantages_and_batch_metrics_unchanged(name):
    case, _, out = run_case(name)
    # the same upstream estimator on the same inputs: bit-identical (the other metrics within 1e-6)
    assert torch.equal(out["advantages"], torch.tensor(case["expected"]["advantages"]))
    assert out.metadata["metrics"] == pytest.approx(case["expected"]["metadata_metrics"], rel=1e-6, abs=1e-9)
    # train() pops these right after compute_advantages_and_returns; the override must leave them in place
    assert "rewards" in out and "uids" in out.metadata


def test_golden_covers_the_interesting_shapes():
    cases = GOLDEN["cases"]
    assert any(c["inputs"]["pad_size"] > 0 and c["config"]["step_wise_trajectories"] for c in cases.values())
    assert any(c["inputs"]["pad_size"] > 0 and not c["config"]["step_wise_trajectories"] for c in cases.values())
    singleton = cases["grpo_singleton_groups"]["expected"]["all_metrics"]
    assert not any(k.startswith("grpo/") for k in singleton)  # non-group shape: nothing logged
    mixed = cases["grpo_mixed_groups"]["expected"]["all_metrics"]
    assert mixed["grpo/all_solved_frac"] == mixed["grpo/all_failed_frac"] == 0.25


def test_instrumentation_never_fails_a_step(monkeypatch):
    import opsd.skyrl_ext.trainer as trainer_mod

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(trainer_mod, "gradient_signal_metrics", boom)
    case, t, out = run_case("grpo_mixed_groups")
    assert not any(k.startswith("grpo/") for k in t.all_metrics)
    assert torch.allclose(out["advantages"], torch.tensor(case["expected"]["advantages"]), rtol=1e-6, atol=1e-6)


def test_upstream_estimator_is_still_skyrl():
    # tau2's golden tests (test_mix_lambda.py) assume the real estimator lives in skyrl.train.trainer
    assert RayPPOTrainer.compute_advantages_and_returns.__module__ == "skyrl.train.trainer"
    assert OPSDRayPPOTrainer.compute_advantages_and_returns.__module__ == "opsd.skyrl_ext.trainer"

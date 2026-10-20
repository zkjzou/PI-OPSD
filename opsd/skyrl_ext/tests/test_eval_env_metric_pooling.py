"""Regression test for the eval `environment/*` aggregation bug.

`concatenate_generator_outputs` combines per-batch `rollout_metrics` with a name-based heuristic whose
`else` branch SUMS unknown keys. The `environment/*` metrics (per-trajectory means) fall into that
branch, so across eval batches they were summed -> inflated values (e.g. `eval/all/environment/success`
> 1, `num_turns` > max_turns). `opsd.skyrl_ext.evaluate.evaluate()` re-pools them from the raw per-row
`env_metrics` via `get_rollout_metrics`. This test pins both facts: the summing in the concat, and the fix,
including through evaluate() itself over two eval batches.
"""

import asyncio

import numpy as np
import pytest

import skyrl_gym
from opsd.skyrl_ext.evaluate import evaluate
from opsd.skyrl_ext.tests.test_eval_support import (
    EVAL_ENV_KNOBS,
    StubGenerator,
    StubLoader,
    StubTokenizer,
    make_cfg,
)
from skyrl.train.generators.utils import (
    concatenate_generator_outputs,
    get_rollout_metrics,
)

FAKE_ENV = "_eval_pool_faketest"


class _FakeEnv:
    """Minimal env whose aggregate_metrics returns the per-key MEAN over trajectory metric dicts."""

    @staticmethod
    def aggregate_metrics(metrics):
        keys = metrics[0].keys() if metrics else []
        return {k: float(np.mean([m[k] for m in metrics])) for k in keys}


@pytest.fixture(autouse=True)
def _fake_env():
    from skyrl_gym.envs.registration import deregister, registry

    added = FAKE_ENV not in registry
    if added:
        skyrl_gym.register(id=FAKE_ENV, entry_point=_FakeEnv)
    yield
    if added:
        deregister(FAKE_ENV)


def _batch(success, num_turns):
    """One per-batch GeneratorOutput carrying per-trajectory env_metrics + its rollout_metrics."""
    n = len(success)
    responses = [[1, 2] for _ in range(n)]
    rewards = [float(s) * 10.0 for s in success]
    loss_masks = [[1, 1] for _ in range(n)]
    env_metrics = [{"success": float(s), "num_turns": float(t)} for s, t in zip(success, num_turns)]
    env_classes = [FAKE_ENV] * n
    return {
        "prompt_token_ids": [[1] for _ in range(n)],
        "response_ids": responses,
        "rewards": rewards,
        "loss_masks": loss_masks,
        "stop_reasons": ["stop"] * n,
        "rollout_logprobs": None,
        "env_metrics": env_metrics,
        "rollout_metrics": get_rollout_metrics(responses, rewards, env_metrics, env_classes, loss_masks),
    }


def test_concat_sums_environment_metrics_across_batches():
    """Documents the bug: concat SUMS the per-batch environment/* means."""
    b1 = _batch(success=[1, 0], num_turns=[30, 10])  # mean success 0.5, mean turns 20
    b2 = _batch(success=[1, 1], num_turns=[20, 20])  # mean success 1.0, mean turns 20
    concat = concatenate_generator_outputs([b1, b2])
    rm = concat["rollout_metrics"]
    assert rm["environment/success"] == 1.5  # 0.5 + 1.0  (a rate > 1 -> clearly a sum)
    assert rm["environment/num_turns"] == 40.0  # 20 + 20    (exceeds any per-episode cap)


def test_pooled_recompute_is_correct_and_bounded():
    """The fix: re-pool environment/* from the concatenated raw env_metrics -> count-weighted mean."""
    b1 = _batch(success=[1, 0], num_turns=[30, 10])
    b2 = _batch(success=[1, 1], num_turns=[20, 20])
    concat = concatenate_generator_outputs([b1, b2])

    pooled = get_rollout_metrics(
        concat["response_ids"],
        concat["rewards"],
        env_metrics=concat["env_metrics"],
        env_classes=[FAKE_ENV] * len(concat["response_ids"]),
        loss_masks=concat["loss_masks"],
    )
    # pooled over all 4 trajectories: success [1,0,1,1] -> 0.75 ; turns [30,10,20,20] -> 20
    assert pooled["environment/success"] == 0.75
    assert pooled["environment/success"] <= 1.0
    assert pooled["environment/num_turns"] == 20.0


def test_env_metrics_none_yields_no_environment_keys():
    """If a generator leaves env_metrics unset, pooled simply omits environment/* (no crash, no wrong sum)."""
    pooled = get_rollout_metrics(
        [[1, 2], [1, 2]], [10.0, 0.0], env_metrics=None, env_classes=None, loss_masks=[[1, 1], [1, 1]]
    )
    assert not any(k.startswith("environment/") for k in pooled)


def test_evaluate_pools_environment_metrics_over_two_batches(tmp_path, monkeypatch):
    """Through evaluate(): eval/all/environment/* is the pooled mean; custom generator keys survive the concat."""
    for knob in EVAL_ENV_KNOBS:
        monkeypatch.delenv(knob, raising=False)
    b1 = _batch(success=[1, 0], num_turns=[30, 10])
    b2 = _batch(success=[1, 1], num_turns=[20, 20])
    b1["rollout_metrics"]["custom/infra_failures"] = 1.0
    b2["rollout_metrics"]["custom/infra_failures"] = 2.0

    def _prompts(start, n):
        return [
            {
                "prompt": [{"role": "user", "content": f"q{k}"}],
                "env_class": None,
                "env_extras": {"data_source": "ds_a" if k % 2 else "ds_b"},
                "uid": f"uid-{k}",
            }
            for k in range(start, start + n)
        ]

    metrics = asyncio.run(
        evaluate(
            eval_dataloader=StubLoader([_prompts(0, 2), _prompts(2, 2)]),
            generator=StubGenerator([b1, b2]),
            cfg=make_cfg(FAKE_ENV, 1, tmp_path, dump_eval_results=False),
            global_step=1,
            tokenizer=StubTokenizer(),
        )
    )

    assert metrics["eval/all/environment/success"] == 0.75
    assert metrics["eval/all/environment/num_turns"] == 20.0
    assert metrics["eval/all/custom/infra_failures"] == 3.0  # concat heuristic (sum) kept for custom keys
    # per data source: ds_b = rows 0, 2 (success 1, 1); ds_a = rows 1, 3 (success 0, 1)
    assert metrics["eval/ds_b/environment/success"] == 1.0
    assert metrics["eval/ds_a/environment/success"] == 0.5

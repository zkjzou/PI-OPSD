"""opsd.skyrl_ext.evaluate.evaluate -- ported from SkyRL tests/train/test_eval.py.

``test_evaluate_computes_expected_metrics`` is upstream's skeleton test, retargeted at the OPSD copy so the copied
evaluate() skeleton (prepare_generator_input wiring, env classes, batch metadata) stays covered.
``test_evaluate_logs_pooled_pass_at_1_when_n_gt_1`` was added by the fork.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from opsd.skyrl_ext.evaluate import evaluate
from opsd.skyrl_ext.tests.test_eval_support import (
    EVAL_ENV_KNOBS,
    StubGenerator,
    StubLoader,
    make_cfg,
)
from skyrl.train.config import SamplingParams
from skyrl.train.generators.base import GeneratorOutput


@pytest.fixture(autouse=True)
def _no_eval_knobs(monkeypatch):
    for knob in EVAL_ENV_KNOBS:
        monkeypatch.delenv(knob, raising=False)


def _cfg(n, tmp_path):
    cfg = make_cfg("gsm8k", n, tmp_path, dump_eval_results=False)
    cfg.generator.eval_sampling_params = SamplingParams(
        max_generate_length=20, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0, logprobs=None, stop=None
    )
    return cfg


def test_evaluate_computes_expected_metrics(tmp_path):
    cfg = _cfg(1, tmp_path)
    prompts_batch = [
        {
            "prompt": [{"role": "user", "content": "question-1"}],
            "env_class": None,
            "env_extras": {"data_source": "dataset/a"},
            "uid": "uid-1",
        },
        {
            "prompt": [{"role": "user", "content": "question-2"}],
            "env_class": "custom_env",
            "env_extras": {"data_source": "dataset/b"},
            "uid": "uid-2",
        },
    ]
    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[101], [102]],
        "response_ids": [[201], [202]],
        "rewards": [1.0, 0.0],
        "loss_masks": [[1], [1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_logprobs": None,
    }
    generator = StubGenerator([generator_output])
    tokenizer = MagicMock()
    tokenizer.decode.side_effect = lambda tokens: "decoded"

    metrics = asyncio.run(
        evaluate(
            eval_dataloader=StubLoader([prompts_batch]),
            generator=generator,
            cfg=cfg,
            global_step=5,
            tokenizer=tokenizer,
        )
    )

    expected_metrics = {
        "eval/dataset_a/avg_score": 1.0,
        "eval/dataset_a/pass_at_1": 1.0,
        "eval/dataset_b/avg_score": 0.0,
        "eval/dataset_b/pass_at_1": 0.0,
        "eval/all/avg_score": 0.5,
        "eval/all/pass_at_1": 0.5,
    }
    for key, expected_value in expected_metrics.items():
        assert metrics[key] == pytest.approx(expected_value)

    assert len(generator.seen_inputs) == 1
    seen_batch = generator.seen_inputs[0]
    assert seen_batch["prompts"] == [prompt["prompt"] for prompt in prompts_batch]
    assert seen_batch["env_classes"] == ["gsm8k", "custom_env"]
    assert seen_batch["env_extras"] == [prompt["env_extras"] for prompt in prompts_batch]
    assert seen_batch["batch_metadata"].training_phase == "eval"


def test_evaluate_logs_pooled_pass_at_1_when_n_gt_1(tmp_path):
    """At N>1 the pooled bucket must carry pass_at_1 next to pass_at_N, like each data source does."""
    cfg = _cfg(2, tmp_path)
    prompts_batch = [
        {
            "prompt": [{"role": "user", "content": "q1"}],
            "env_class": None,
            "env_extras": {"data_source": "dataset/a"},
            "uid": "uid-1",
        },
        {
            "prompt": [{"role": "user", "content": "q2"}],
            "env_class": None,
            "env_extras": {"data_source": "dataset/b"},
            "uid": "uid-2",
        },
    ]
    # prepare_generator_input lays samples out per prompt: [uid-1, uid-1, uid-2, uid-2].
    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[101], [101], [102], [102]],
        "response_ids": [[201], [202], [203], [204]],
        "rewards": [1.0, 0.0, 0.0, 0.0],
        "loss_masks": [[1], [1], [1], [1]],
        "stop_reasons": ["stop"] * 4,
        "rollout_logprobs": None,
    }
    tokenizer = MagicMock()
    tokenizer.decode.side_effect = lambda tokens: "decoded"

    metrics = asyncio.run(
        evaluate(
            eval_dataloader=StubLoader([prompts_batch]),
            generator=StubGenerator([generator_output]),
            cfg=cfg,
            global_step=5,
            tokenizer=tokenizer,
        )
    )

    # uid-1 solved on 1 of 2 samples, uid-2 on 0 of 2.
    assert metrics["eval/all/pass_at_2"] == pytest.approx(0.5)  # any-of-2: 1 of 2 prompts
    assert metrics["eval/all/pass_at_1"] == pytest.approx(0.25)  # mean per-rollout: 1 of 4 rollouts
    assert metrics["eval/dataset_a/pass_at_1"] == pytest.approx(0.5)
    assert metrics["eval/dataset_b/pass_at_1"] == pytest.approx(0.0)

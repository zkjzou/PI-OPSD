"""OPSDEvalMixin routes RayPPOTrainer.eval: non-step-wise -> opsd evaluate(), step-wise -> upstream via super()."""

import asyncio
from types import SimpleNamespace

import pytest

import opsd.skyrl_ext.evaluate as opsd_eval
from opsd.skyrl_ext.evaluate import OPSDEvalMixin
from opsd.skyrl_ext.tests.test_eval_support import (
    EVAL_ENV_KNOBS,
    StubLoader,
    StubTokenizer,
    load_case,
    make_cfg,
    materialize_outputs,
    registered_eval_envs,
)

STEP_WISE_RESULT = {
    "eval/all/environment/success": 0.5,
    "eval/ds/generate/max_num_tokens": 3,
    "eval/ds/avg_score": 1.0,
}


@pytest.fixture(autouse=True)
def _no_eval_knobs(monkeypatch):
    for knob in EVAL_ENV_KNOBS:
        monkeypatch.delenv(knob, raising=False)


def _stub_evaluate(calls, result):
    async def _evaluate(**kwargs):
        calls.append(kwargs)
        return result

    return _evaluate


def _trainer(cls, step_wise, tmp_path):
    t = object.__new__(cls)  # several opsd tests build trainers this way; the mixin must not need __init__
    t.cfg = make_cfg("alfworld", 1, tmp_path)
    t.cfg.generator.step_wise_trajectories = step_wise
    t.eval_dataloader, t.generator, t.tokenizer, t.global_step = object(), object(), object(), 7
    return t


def test_mixin_is_stateless():
    assert "__init__" not in vars(OPSDEvalMixin)
    assert [k for k in vars(OPSDEvalMixin) if not k.startswith("__")] == ["eval"]


class _ToyBase:
    async def eval(self):
        return dict(STEP_WISE_RESULT)


class _ToyTrainer(OPSDEvalMixin, _ToyBase):
    pass


def test_non_step_wise_calls_opsd_evaluate(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(opsd_eval, "evaluate", _stub_evaluate(calls, {"eval/all/avg_score": 0.0}))
    t = _trainer(_ToyTrainer, step_wise=False, tmp_path=tmp_path)
    assert asyncio.run(t.eval()) == {"eval/all/avg_score": 0.0}
    assert calls == [
        dict(eval_dataloader=t.eval_dataloader, generator=t.generator, cfg=t.cfg, global_step=7, tokenizer=t.tokenizer)
    ]


def test_step_wise_delegates_to_super_and_postprocesses(monkeypatch, tmp_path):
    monkeypatch.setattr(opsd_eval, "evaluate", _stub_evaluate([], None))
    t = _trainer(_ToyTrainer, step_wise=True, tmp_path=tmp_path)
    # pass_at_1 is filled from environment/success, as the fork's post-processor did on this path
    assert asyncio.run(t.eval()) == {**STEP_WISE_RESULT, "eval/all/pass_at_1": 0.5}

    monkeypatch.setenv("SKYRL_EVAL_COMPACT", "1")
    monkeypatch.setenv("SKYRL_EVAL_NAMESPACE", "test")
    assert asyncio.run(t.eval()) == {
        "test/all/environment/success": 0.5,
        "test/ds/avg_score": 1.0,
        "test/all/pass_at_1": 0.5,
    }


def test_module_evaluate_step_wise_is_upstream_plus_postprocess(monkeypatch):
    """opsd evaluate_step_wise (for generation-only entrypoints) = upstream evaluate_step_wise + post-processors."""
    calls = []
    monkeypatch.setattr(opsd_eval, "_upstream_evaluate_step_wise", _stub_evaluate(calls, STEP_WISE_RESULT))
    monkeypatch.setenv("SKYRL_EVAL_COMPACT", "1")
    kwargs = dict(eval_dataloader=1, generator=2, cfg=3, global_step=None, tokenizer=4)
    out = asyncio.run(opsd_eval.evaluate_step_wise(**kwargs))
    assert calls == [kwargs]
    assert out == {"eval/all/environment/success": 0.5, "eval/ds/avg_score": 1.0, "eval/all/pass_at_1": 0.5}


@pytest.fixture(scope="module")
def ray_trainer_cls():
    from skyrl.train.trainer import RayPPOTrainer

    class _Trainer(OPSDEvalMixin, RayPPOTrainer):
        pass

    return _Trainer


def test_routes_through_real_ray_ppo_trainer(monkeypatch, tmp_path, ray_trainer_cls):
    import skyrl.train.trainer as upstream_trainer

    opsd_calls, upstream_calls, step_wise_calls = [], [], []
    monkeypatch.setattr(opsd_eval, "evaluate", _stub_evaluate(opsd_calls, {"eval/all/avg_score": 1.0}))
    monkeypatch.setattr(upstream_trainer, "evaluate", _stub_evaluate(upstream_calls, {}))
    monkeypatch.setattr(upstream_trainer, "evaluate_step_wise", _stub_evaluate(step_wise_calls, STEP_WISE_RESULT))

    t = _trainer(ray_trainer_cls, step_wise=False, tmp_path=tmp_path)
    assert asyncio.run(t.eval()) == {"eval/all/avg_score": 1.0}
    assert len(opsd_calls) == 1 and not upstream_calls and not step_wise_calls

    t = _trainer(ray_trainer_cls, step_wise=True, tmp_path=tmp_path)
    monkeypatch.setenv("SKYRL_EVAL_NAMESPACE", "test")
    out = asyncio.run(t.eval())
    assert len(step_wise_calls) == 1 and step_wise_calls[0]["global_step"] == 7
    assert out == {
        "test/all/environment/success": 0.5,
        "test/ds/generate/max_num_tokens": 3,
        "test/ds/avg_score": 1.0,
        "test/all/pass_at_1": 0.5,
    }
    assert len(opsd_calls) == 1 and not upstream_calls


def test_end_to_end_trainer_eval_matches_fork_golden(monkeypatch, tmp_path, ray_trainer_cls):
    """A trainer built on the mixin reproduces the fork's eval dict on a golden case (real opsd evaluate)."""
    record = load_case("appworld__n4__success")
    monkeypatch.setenv("SKYRL_EVAL_COMPACT", "1")
    t = object.__new__(ray_trainer_cls)
    t.cfg = make_cfg(record["env_class"], record["n_samples_per_prompt"], tmp_path)
    t.eval_dataloader = StubLoader(record["eval_batches"])
    t.tokenizer = StubTokenizer()
    t.global_step = record["global_step"]
    outputs = materialize_outputs(record)

    async def _generate(input_batch):
        return outputs.pop(0)

    t.generator = SimpleNamespace(generate=_generate)
    with registered_eval_envs():
        metrics = asyncio.run(t.eval())
    assert list(metrics.items()) == list(record["metrics"]["compact"].items())

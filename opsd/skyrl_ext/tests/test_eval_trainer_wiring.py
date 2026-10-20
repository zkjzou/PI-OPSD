"""Every release trainer's eval() must reach opsd.skyrl_ext.evaluate.evaluate (not upstream skyrl.train.evaluate).

Tau2OPSDTrainer is included explicitly: it overrides eval() to skip SelfDistillationTrainer's dump reader and
must still land on OPSDEvalMixin.eval, not on RayPPOTrainer.eval.
"""

import asyncio
import importlib

import pytest

import opsd.skyrl_ext.evaluate as opsd_eval
from opsd.skyrl_ext.tests.test_eval_support import EVAL_ENV_KNOBS, make_cfg

RELEASE_TRAINERS = [
    "opsd.trainer.main_opsd_alfworld:SelfDistillationTrainer",
    "opsd.alfworld.main_alfworld_dapo:ALFWorldDAPOTrainer",
    "opsd.appworld.main_appworld:AppBalancedTrainer",
    "opsd.trainer.main_opsd_appworld:AppWorldSelfDistillTrainer",
    "opsd.tau2_bench.main:Tau2DAPOTrainer",
    "opsd.tau2_bench.distill.trainer:Tau2OPSDTrainer",
]


def _load(path):
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("path", RELEASE_TRAINERS)
def test_trainer_eval_routes_to_opsd_evaluate(path, monkeypatch, tmp_path):
    import skyrl.train.trainer as upstream_trainer

    for knob in EVAL_ENV_KNOBS:
        monkeypatch.delenv(knob, raising=False)
    monkeypatch.setenv("OPSD_WANDB_TRAJ_N", "0")
    calls, upstream_calls = [], []

    async def _opsd(**kwargs):
        calls.append(kwargs)
        return {"eval/all/avg_score": 0.25}

    async def _upstream(**kwargs):
        upstream_calls.append(kwargs)
        return {}

    monkeypatch.setattr(opsd_eval, "evaluate", _opsd)
    monkeypatch.setattr(upstream_trainer, "evaluate", _upstream)

    cls = _load(path)
    from opsd.skyrl_ext.trainer import OPSDRayPPOTrainer

    assert issubclass(cls, OPSDRayPPOTrainer)
    t = object.__new__(cls)
    t.cfg = make_cfg("alfworld", 1, tmp_path, dump_eval_results=False)
    t.eval_dataloader, t.generator, t.tokenizer, t.global_step = object(), object(), object(), 3
    assert asyncio.run(t.eval()) == {"eval/all/avg_score": 0.25}
    assert len(calls) == 1 and not upstream_calls

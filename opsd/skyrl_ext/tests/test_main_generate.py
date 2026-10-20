"""opsd.skyrl_ext.main_generate: the generation-only entrypoint for OPSD configs.

Checks, with Ray / the model / the generator stubbed, that it parses with OPSDTrainConfig, starts Ray through the
OPSD initialize_ray, pins the eval task to the driver's node, builds an OPSDExpMixin Exp (env registration), and
scores with opsd.skyrl_ext.evaluate (never skyrl.train.evaluate). The upstream code it copies is drift-pinned in
test_eval_upstream_drift.py.
"""

import asyncio
import sys
from types import SimpleNamespace

import pytest
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

import opsd.skyrl_ext.evaluate as opsd_evaluate
from opsd.skyrl_ext import main_generate as mg
from opsd.skyrl_ext import ray_runtime
from opsd.skyrl_ext.config import OPSDTrainConfig
from opsd.skyrl_ext.exp import OPSDExpMixin
from skyrl.train.entrypoints.main_generate import EvalOnlyEntrypoint

_DRIVER_NODE_ID = "cd" * 28  # Ray validates node ids as 28-byte hex strings


def test_wiring():
    assert mg.CONFIG_CLS is OPSDTrainConfig
    assert mg.initialize_ray is ray_runtime.initialize_ray
    assert issubclass(mg.OPSDEvalOnlyEntrypoint, OPSDExpMixin)
    assert issubclass(mg.OPSDEvalOnlyEntrypoint, EvalOnlyEntrypoint)
    mro = mg.OPSDEvalOnlyEntrypoint.__mro__
    assert mro.index(OPSDExpMixin) < mro.index(EvalOnlyEntrypoint)  # the mixin's __init__ runs first


def test_main_parses_validates_initializes_and_pins(monkeypatch):
    calls, cfg = [], object()
    monkeypatch.setattr(sys, "argv", ["prog", "generator.eval_n_samples_per_prompt=4"])
    monkeypatch.setattr(
        mg, "CONFIG_CLS", SimpleNamespace(from_cli_overrides=lambda argv: calls.append(("parse", list(argv))) or cfg)
    )
    monkeypatch.setattr(mg, "validate_generator_cfg", lambda c: calls.append(("validate", c)))
    monkeypatch.setattr(mg, "initialize_ray", lambda c: calls.append(("init_ray", c)))

    class _Remote:
        def options(self, **kwargs):
            calls.append(("options", kwargs))
            return self

        def remote(self, c):
            calls.append(("remote", c))
            return "ref"

    monkeypatch.setattr(mg, "eval_entrypoint", _Remote())
    monkeypatch.setattr(
        mg,
        "ray",
        SimpleNamespace(
            get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: _DRIVER_NODE_ID),
            get=lambda ref: calls.append(("get", ref)) or {"eval/all/avg_score": 1.0},
        ),
    )

    mg.main()

    assert [c[0] for c in calls] == ["parse", "validate", "init_ray", "options", "remote", "get"]
    assert calls[0] == ("parse", ["generator.eval_n_samples_per_prompt=4"])
    strategy = calls[3][1]["scheduling_strategy"]
    assert isinstance(strategy, NodeAffinitySchedulingStrategy)
    assert strategy.node_id == _DRIVER_NODE_ID and strategy.soft is False
    assert calls[4] == ("remote", cfg) and calls[5] == ("get", "ref")


@pytest.mark.parametrize("step_wise", [False, True], ids=["evaluate", "evaluate_step_wise"])
def test_run_scores_with_the_opsd_evaluate(step_wise, monkeypatch):
    import skyrl.train.evaluate as upstream_evaluate

    cfg = OPSDTrainConfig()
    cfg.generator.step_wise_trajectories = step_wise
    used, logged, events = [], [], []

    async def _opsd(name, **kwargs):
        used.append((name, kwargs))
        return {"eval/all/avg_score": 0.5}

    async def _opsd_plain(**kwargs):
        return await _opsd("evaluate", **kwargs)

    async def _opsd_step_wise(**kwargs):
        return await _opsd("evaluate_step_wise", **kwargs)

    async def _upstream(**kwargs):
        raise AssertionError("upstream skyrl.train.evaluate must not run")

    monkeypatch.setattr(opsd_evaluate, "evaluate", _opsd_plain)
    monkeypatch.setattr(opsd_evaluate, "evaluate_step_wise", _opsd_step_wise)
    monkeypatch.setattr(upstream_evaluate, "evaluate", _upstream)
    monkeypatch.setattr(upstream_evaluate, "evaluate_step_wise", _upstream)
    monkeypatch.setattr(mg, "build_dataloader", lambda c, ds, is_train: ("loader", ds, is_train))

    class _Client:
        async def wake_up(self):
            events.append("wake_up")

    exp = object.__new__(mg.OPSDEvalOnlyEntrypoint)
    exp.cfg, exp.tokenizer, exp.eval_dataset = cfg, "tok", "eval-ds"
    exp.get_generator = lambda c, tok, client: events.append("get_generator") or "gen"
    exp.get_tracker = lambda: SimpleNamespace(log=lambda data, step, commit: logged.append((data, step, commit)))

    results = asyncio.run(exp.run(_Client()))

    assert results == {"eval/all/avg_score": 0.5}
    assert events == ["wake_up", "get_generator"]  # the engines are awake before the generator is built
    ((name, kwargs),) = used
    assert name == ("evaluate_step_wise" if step_wise else "evaluate")
    assert kwargs == dict(
        eval_dataloader=("loader", "eval-ds", False), generator="gen", cfg=cfg, global_step=None, tokenizer="tok"
    )
    # Logged once, unchanged (the opsd evaluate functions already applied SKYRL_EVAL_COMPACT / _NAMESPACE).
    assert logged == [({"eval/all/avg_score": 0.5}, 0, True)]


def test_run_requires_an_eval_dataset():
    exp = object.__new__(mg.OPSDEvalOnlyEntrypoint)
    exp.cfg, exp.eval_dataset = OPSDTrainConfig(), None
    with pytest.raises(AssertionError, match="requires an eval dataset"):
        asyncio.run(exp.run(SimpleNamespace()))

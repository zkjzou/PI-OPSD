"""AppWorld entrypoints pin the training loop to the driver's Ray node.

The AppWorld server pool listens on 127.0.0.1 of the node that started it and server recycling scans that
node's /proc, so the ``skyrl_entrypoint`` task must run on the driver's node. These tests check, with
``ray`` mocked out (no cluster is started), that ``run_on_driver_node`` uses HARD node affinity to the
driver's node and that both AppWorld entrypoints (GRPO/DAPO and OPSD) route through it.
"""

import sys
from types import SimpleNamespace

import pytest

# Plain imports, NOT pytest.importorskip: an entrypoint that fails to import (e.g. a dependency missing from
# the release) must FAIL this module, not skip it and let CI pass. The test env always has ray + skyrl.
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from opsd.appworld import main_appworld
from opsd.trainer import main_opsd_appworld


class _FakeRemoteFn:
    """Stands in for a ``@ray.remote`` function: records ``.options(...)`` and ``.remote(...)`` calls."""

    def __init__(self):
        self.options_kwargs = None
        self.remote_args = None

    def options(self, **kwargs):
        self.options_kwargs = kwargs
        return self

    def remote(self, *args):
        self.remote_args = args
        return "object-ref"


_DRIVER_NODE_ID = "ab" * 28  # Ray validates node ids as 28-byte hex strings


def test_run_on_driver_node_uses_hard_affinity_to_the_driver_node(monkeypatch):
    got = []
    fake_ray = SimpleNamespace(
        get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: _DRIVER_NODE_ID),
        get=lambda ref: got.append(ref) or "task-result",
    )
    monkeypatch.setattr(main_appworld, "ray", fake_ray)
    fn, cfg = _FakeRemoteFn(), object()

    assert main_appworld.run_on_driver_node(fn, cfg) == "task-result"

    strategy = fn.options_kwargs["scheduling_strategy"]
    assert set(fn.options_kwargs) == {"scheduling_strategy"}
    assert isinstance(strategy, NodeAffinitySchedulingStrategy)
    assert strategy.node_id == _DRIVER_NODE_ID
    assert strategy.soft is False  # never silently placed on another node
    assert fn.remote_args == (cfg,)
    assert got == ["object-ref"]  # blocks on the task (the driver waits for training to finish)


def test_opsd_entrypoint_shares_the_same_pinning_helper():
    assert main_opsd_appworld.run_on_driver_node is main_appworld.run_on_driver_node


@pytest.mark.parametrize("mod", [main_appworld, main_opsd_appworld], ids=["main_appworld", "main_opsd_appworld"])
def test_entrypoint_main_routes_through_run_on_driver_node(mod, monkeypatch):
    calls = []
    cfg = object()
    monkeypatch.setattr(sys, "argv", ["prog", "trainer.epochs=1"])
    # main() looks the config class up through the module global CONFIG_CLS (OPSDTrainConfig).
    monkeypatch.setattr(
        mod,
        "CONFIG_CLS",
        SimpleNamespace(from_cli_overrides=lambda argv: calls.append(("parse", list(argv))) or cfg),
    )
    monkeypatch.setattr(mod, "validate_cfg", lambda c: calls.append(("validate", c)))
    monkeypatch.setattr(mod, "initialize_ray", lambda c: calls.append(("init_ray", c)))
    monkeypatch.setattr(mod, "run_on_driver_node", lambda fn, c: calls.append(("run", fn, c)))

    mod.main()

    assert calls == [
        ("parse", ["trainer.epochs=1"]),
        ("validate", cfg),
        ("init_ray", cfg),
        ("run", mod.skyrl_entrypoint, cfg),
    ]
    # the pinned task is the module's own Ray remote function (each entrypoint runs its own Exp class)
    assert hasattr(mod.skyrl_entrypoint, "options") and hasattr(mod.skyrl_entrypoint, "remote")

"""Release-wide guards: every OPSD entrypoint runs on the OPSD extensions of SkyRL, not on bare upstream.

SkyRL is used unmodified, so OPSD behaviour only exists where an entrypoint opts into it:

* config: each entry module defines ``CONFIG_CLS`` (an ``OPSDTrainConfig``) and ``main()`` parses through that
  module global; upstream ``SkyRLTrainConfig`` rejects the OPSD keys (or, for a recipe that sets none of them,
  silently drops DAPO's ``overlong_buffer``);
* Ray: ``initialize_ray`` is ``opsd.skyrl_ext.ray_runtime.initialize_ray`` (project env-var forwarding);
* Exp: every ``BasePPOExp`` subclass lists ``OPSDExpMixin`` first (env registration in the Ray task, OPSD
  tracker, trainer-type guard) and its ``get_trainer`` returns an ``OPSDRayPPOTrainer``;
* trainer: every ``RayPPOTrainer`` subclass under ``opsd`` derives from ``OPSDRayPPOTrainer`` (OPSD eval,
  beta/SDAR worker selection, grpo/* signal, resume fix).

The static half scans every module of the ``opsd`` package (so a new entrypoint is covered without being
listed); the dynamic half imports the entry modules (slow: the trainer stack).
"""

import ast
import importlib
from pathlib import Path

import pytest

import opsd

PKG = Path(opsd.__file__).resolve().parent

ENTRY_MODULES = {
    "opsd.trainer.main_opsd_alfworld": "OPSDTrainConfig",
    "opsd.alfworld.main_alfworld_dapo": "OPSDTrainConfig",
    "opsd.trainer.main_opsd_appworld": "OPSDTrainConfig",
    "opsd.appworld.main_appworld": "OPSDTrainConfig",
    "opsd.tau2_bench.main": "Tau2SkyRLConfig",
    "opsd.tau2_bench.distill.main_tau2_opsd": "Tau2SkyRLConfig",
    "opsd.skyrl_ext.main_generate": "OPSDTrainConfig",
}

# Exp class -> the trainer class its get_trainer builds.
EXPS = {
    "opsd.trainer.main_opsd_alfworld:SelfDistillExp": "opsd.trainer.main_opsd_alfworld:SelfDistillationTrainer",
    "opsd.alfworld.main_alfworld_dapo:ALFWorldDAPOExp": "opsd.alfworld.main_alfworld_dapo:ALFWorldDAPOTrainer",
    "opsd.trainer.main_opsd_appworld:AppWorldSelfDistillExp": (
        "opsd.trainer.main_opsd_appworld:AppWorldSelfDistillTrainer"
    ),
    "opsd.appworld.main_appworld:AppWorldExp": "opsd.appworld.main_appworld:AppBalancedTrainer",
    "opsd.tau2_bench.main:Tau2Exp": "opsd.tau2_bench.main:Tau2DAPOTrainer",
    "opsd.tau2_bench.distill.main_tau2_opsd:Tau2OPSDExp": "opsd.tau2_bench.distill.trainer:Tau2OPSDTrainer",
}

UPSTREAM_EXPS = {"BasePPOExp", "EvalOnlyEntrypoint"}


def _load(path):
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


def _package_modules():
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG.parent)
        if "tests" in rel.parts:
            continue
        yield ".".join(rel.with_suffix("").parts).removesuffix(".__init__"), path


def _base_names(node: ast.ClassDef):
    for b in node.bases:
        if isinstance(b, ast.Name):
            yield b.id
        elif isinstance(b, ast.Attribute):
            yield b.attr


# ------------------------------------------------------------------------------------------------- static
def test_only_the_opsd_base_subclasses_ray_ppo_trainer_directly():
    offenders = []
    for module, path in _package_modules():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef) and "RayPPOTrainer" in set(_base_names(node)):
                if (module, node.name) != ("opsd.skyrl_ext.trainer", "OPSDRayPPOTrainer"):
                    offenders.append(f"{module}:{node.name}")
    assert not offenders, f"derive these from opsd.skyrl_ext.trainer.OPSDRayPPOTrainer instead: {offenders}"


def test_every_exp_lists_the_opsd_mixin_first():
    offenders, found = [], []
    for module, path in _package_modules():
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = list(_base_names(node))
            upstream = [i for i, b in enumerate(bases) if b in UPSTREAM_EXPS]
            if upstream:
                found.append(f"{module}:{node.name}")
                if "OPSDExpMixin" not in bases or bases.index("OPSDExpMixin") > upstream[0]:
                    offenders.append(f"{module}:{node.name}{tuple(bases)}")
    assert found, "the scan found no Exp class at all"
    assert not offenders, f"list OPSDExpMixin before the upstream Exp base: {offenders}"


def test_no_module_uses_upstream_initialize_ray_or_parses_with_upstream_config():
    bad = []
    for module, path in _package_modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("skyrl"):
                if any(a.name == "initialize_ray" for a in node.names):
                    bad.append(f"{module}: imports initialize_ray from {node.module}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_cli_overrides"
                and not (isinstance(node.func.value, ast.Name) and node.func.value.id == "CONFIG_CLS")
            ):
                bad.append(f"{module}:{node.lineno}: parses with {ast.unparse(node.func.value)}, not CONFIG_CLS")
    assert not bad, "\n".join(bad)


def test_the_entry_module_list_is_complete():
    """Every module with a `def main()` that parses a config is listed in ENTRY_MODULES."""
    parsing = set()
    for module, path in _package_modules():
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                if any(isinstance(n, ast.Attribute) and n.attr == "from_cli_overrides" for n in ast.walk(node)):
                    parsing.add(module)
    assert parsing == set(ENTRY_MODULES)


# ------------------------------------------------------------------------------------------------ dynamic
@pytest.mark.parametrize("module", sorted(ENTRY_MODULES))
def test_entry_module_defines_config_cls_and_parses_through_it(module, monkeypatch):
    import sys

    from opsd.skyrl_ext import ray_runtime
    from opsd.skyrl_ext.config import OPSDTrainConfig

    mod = importlib.import_module(module)
    assert mod.CONFIG_CLS.__name__ == ENTRY_MODULES[module]
    assert issubclass(mod.CONFIG_CLS, OPSDTrainConfig)
    assert mod.initialize_ray is ray_runtime.initialize_ray

    class _Parsed(Exception):
        pass

    def _parse(argv):
        raise _Parsed(list(argv))

    monkeypatch.setattr(sys, "argv", ["prog", "trainer.epochs=1"])
    monkeypatch.setattr(mod, "CONFIG_CLS", type("Stub", (), {"from_cli_overrides": staticmethod(_parse)}))
    with pytest.raises(_Parsed) as exc:
        mod.main()
    assert exc.value.args == (["trainer.epochs=1"],)


@pytest.fixture
def no_trainer_init(monkeypatch):
    from skyrl.train.trainer import RayPPOTrainer

    monkeypatch.setattr(RayPPOTrainer, "__init__", lambda self, **kwargs: None)


@pytest.mark.parametrize("exp_path", sorted(EXPS))
def test_every_exp_builds_an_opsd_trainer(exp_path, no_trainer_init, monkeypatch):
    from opsd.skyrl_ext.exp import OPSDExpMixin
    from opsd.skyrl_ext.trainer import OPSDRayPPOTrainer

    for knob in ("OPSD_TEACHER", "OPSD_TEACHER_ALPHA"):
        monkeypatch.delenv(knob, raising=False)
    exp_cls = _load(exp_path)
    assert issubclass(exp_cls, OPSDExpMixin)
    trainer = object.__new__(exp_cls).get_trainer(
        cfg=None,
        tracker=None,
        tokenizer=None,
        train_dataset=None,
        eval_dataset=None,
        inference_engine_client=None,
        generator=None,
        colocate_pg=None,
    )
    assert type(trainer) is _load(EXPS[exp_path])
    assert isinstance(trainer, OPSDRayPPOTrainer)


def test_every_opsd_trainer_class_derives_from_the_opsd_base():
    from opsd.skyrl_ext.trainer import OPSDRayPPOTrainer
    from skyrl.train.trainer import RayPPOTrainer

    for module in ENTRY_MODULES:
        importlib.import_module(module)
    seen, stack = [], list(RayPPOTrainer.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if cls.__module__.split(".")[0] == "opsd" and ".tests." not in cls.__module__:  # tests build stand-ins
            seen.append(cls)
    assert {c.__name__ for c in seen} >= {_load(t).__name__ for t in EXPS.values()}
    bad = [f"{c.__module__}:{c.__qualname__}" for c in seen if not issubclass(c, OPSDRayPPOTrainer)]
    assert not bad, f"not on OPSDRayPPOTrainer: {bad}"

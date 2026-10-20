"""OPSDExpMixin: config-type check, env registration before the Exp is built, OPSD tracker, trainer-type guard.

No Ray, GPU or model: ``BasePPOExp.__init__`` and ``RayPPOTrainer.__init__`` are stubbed where an Exp or a trainer
is built, and ``register_opsd_envs`` is replaced by a recorder (the real one is covered by opsd/envs/tests).
"""

import pytest

import opsd.skyrl_ext.exp as exp_mod
from opsd.skyrl_ext.config import OPSDTrainConfig
from opsd.skyrl_ext.exp import OPSDExpMixin, require_opsd_trainer
from opsd.skyrl_ext.tracking import OPSDTracking
from opsd.skyrl_ext.trainer import OPSDRayPPOTrainer
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.trainer import RayPPOTrainer

TRAINER_KWARGS = dict(
    cfg=None,
    tracker=None,
    tokenizer=None,
    train_dataset=None,
    eval_dataset=None,
    inference_engine_client=None,
    generator=None,
    colocate_pg=None,
)


@pytest.fixture
def no_trainer_init(monkeypatch):
    """RayPPOTrainer.__init__ -> record the kwargs only (it would build dataloaders, dispatch, ...)."""

    def _init(self, **kwargs):
        self.init_kwargs = kwargs

    monkeypatch.setattr(RayPPOTrainer, "__init__", _init)


@pytest.fixture
def recorded(monkeypatch):
    """Replace env registration and BasePPOExp.__init__ with recorders; returns the shared call log."""
    calls = []
    monkeypatch.setattr(exp_mod, "register_opsd_envs", lambda: calls.append("register"))

    def _base_init(self, cfg):
        calls.append(("base_init", cfg))
        self.cfg = cfg

    monkeypatch.setattr(BasePPOExp, "__init__", _base_init)
    return calls


class _Exp(OPSDExpMixin, BasePPOExp):
    pass


class _BadExp(OPSDExpMixin, BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return RayPPOTrainer(*args, **kwargs)


class _GoodExp(OPSDExpMixin, BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return OPSDRayPPOTrainer(*args, **kwargs)


class _InheritsBadExp(_GoodExp):
    def get_trainer(self, *args, **kwargs):  # an override further down the hierarchy is checked too
        return RayPPOTrainer(*args, **kwargs)


def test_init_registers_the_envs_before_building_the_exp(recorded):
    cfg = OPSDTrainConfig()
    exp = _Exp(cfg)
    assert recorded == ["register", ("base_init", cfg)]
    assert exp.cfg is cfg


def test_init_refuses_a_plain_skyrl_config(recorded):
    with pytest.raises(TypeError, match="needs an OPSDTrainConfig"):
        _Exp(SkyRLTrainConfig())
    assert recorded == []  # nothing registered, nothing built


def test_tau2_config_is_an_opsd_config():
    from opsd.tau2_bench.main import Tau2SkyRLConfig

    assert issubclass(Tau2SkyRLConfig, OPSDTrainConfig)


def test_default_get_trainer_builds_the_opsd_base(no_trainer_init):
    trainer = object.__new__(_Exp).get_trainer(**TRAINER_KWARGS)
    assert type(trainer) is OPSDRayPPOTrainer
    assert trainer.init_kwargs == TRAINER_KWARGS


@pytest.mark.parametrize("cls", [_BadExp, _InheritsBadExp])
def test_a_get_trainer_returning_a_plain_ray_ppo_trainer_raises(cls, no_trainer_init):
    with pytest.raises(TypeError, match="does not derive from opsd.skyrl_ext.trainer.OPSDRayPPOTrainer"):
        object.__new__(cls).get_trainer(**TRAINER_KWARGS)


def test_a_get_trainer_returning_an_opsd_trainer_passes(no_trainer_init):
    trainer = object.__new__(_GoodExp).get_trainer(**TRAINER_KWARGS)
    assert isinstance(trainer, OPSDRayPPOTrainer)


def test_subclass_get_trainer_is_wrapped_once():
    fn = _GoodExp.__dict__["get_trainer"]
    assert getattr(fn, exp_mod._CHECKED) is True
    assert exp_mod._checked_get_trainer(fn) is fn  # idempotent
    assert fn.__name__ == "get_trainer" and fn.__qualname__.endswith("_GoodExp.get_trainer")


def test_require_opsd_trainer_names_the_offender():
    with pytest.raises(TypeError, match=r"X\.get_trainer returned builtins\.object"):
        require_opsd_trainer(object(), "X.get_trainer")


def test_get_tracker_builds_opsd_tracking(monkeypatch):
    seen = {}

    def _init(self, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(OPSDTracking, "__init__", _init)
    exp = object.__new__(_Exp)
    exp.cfg = OPSDTrainConfig()
    tracker = exp.get_tracker()
    assert type(tracker) is OPSDTracking
    assert seen["config"] is exp.cfg and seen["backends"] == exp.cfg.trainer.logger


def test_by_value_pickled_exp_class_keeps_the_guard(no_trainer_init):
    """The entry modules run as __main__, so Ray pickles their Exp classes by value. The wrapped get_trainer
    must survive that round trip (cloudpickle rebuilds the class, then re-sets its attributes)."""
    from ray import cloudpickle

    def _make():
        class LocalBadExp(OPSDExpMixin, BasePPOExp):  # local class -> pickled by value
            def get_trainer(self, *args, **kwargs):
                return RayPPOTrainer(*args, **kwargs)

        return LocalBadExp

    original = _make()
    blob = cloudpickle.dumps(original)
    # In the same process cloudpickle hands back the tracked original; forget it so loads() rebuilds the class
    # the way a Ray worker does (skeleton class first, then the pickled class dict).
    tracker = getattr(cloudpickle.cloudpickle, "_DYNAMIC_CLASS_TRACKER_BY_ID", None)
    if tracker is None:
        pytest.skip("cloudpickle internals changed; cannot force a by-value rebuild in-process")
    for key in [k for k, v in list(tracker.items()) if v is original]:
        del tracker[key]
    cls = cloudpickle.loads(blob)
    assert cls is not original and issubclass(cls, OPSDExpMixin)
    assert getattr(cls.__dict__["get_trainer"], exp_mod._CHECKED) is True
    with pytest.raises(TypeError, match="OPSDRayPPOTrainer"):
        object.__new__(cls).get_trainer(**TRAINER_KWARGS)

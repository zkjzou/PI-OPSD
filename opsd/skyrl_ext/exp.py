"""``OPSDExpMixin``: what every OPSD experiment adds to SkyRL's ``BasePPOExp``.

SkyRL builds a run inside the entrypoint's Ray task from a ``BasePPOExp``: tokenizer and datasets in
``__init__``, then (``_setup_trainer``) tracker, inference client, generator, trainer and
``trainer.build_models(...)``. The mixin hooks that sequence only through ``BasePPOExp``'s own methods:

* ``__init__`` refuses a config that is not an :class:`~opsd.skyrl_ext.config.OPSDTrainConfig` (a plain
  ``SkyRLTrainConfig`` has no ``trainer.algorithm.overlong_buffer``, and the DAPO trainers read it with
  ``getattr(..., None)``, so DAPO would be a silent no-op), then registers the OPSD envs
  (:func:`opsd.envs.registry.register_opsd_envs`). The generator's env threads, the eval loop and
  ``skyrl_gym.metrics.aggregate_for_environment`` all run in this process and look ``alfworld`` /
  ``appworld`` up in the skyrl_gym registry; the Exp is constructed inside the Ray task, so registering here
  covers exactly the process that needs it. The policy / ref / vLLM workers never touch the registry.
* ``get_tracker`` builds :class:`~opsd.skyrl_ext.tracking.OPSDTracking` (``SKYRL_LOG_STEP_OVERRIDE``). An Exp
  that needs its own tracker overrides it with an ``OPSDTracking`` subclass.
* ``get_trainer`` must return an :class:`~opsd.skyrl_ext.trainer.OPSDRayPPOTrainer`: a trainer on plain
  ``RayPPOTrainer`` silently runs upstream eval (no per-dataset / env-success metrics), upstream workers (beta /
  SDAR become no-ops), no ``grpo/*`` signal and no epoch-boundary resume fix. Every ``get_trainer`` a subclass
  defines is wrapped when the class is created (``__init_subclass__``) with a check that raises ``TypeError``
  before ``build_models`` places a single GPU actor. The mixin's own default returns a bare
  ``OPSDRayPPOTrainer``.

Use it first in the bases: ``class MyExp(OPSDExpMixin, BasePPOExp)``. It lives in an importable module, not in
an entry module, because the entry modules run as ``python -m`` (``__main__``): Ray pickles their classes by
value, and anything they share must be importable by reference inside the worker.
"""

import functools

from opsd.envs.registry import register_opsd_envs
from opsd.skyrl_ext import tracking as opsd_tracking
from opsd.skyrl_ext.config import OPSDTrainConfig
from opsd.skyrl_ext.trainer import OPSDRayPPOTrainer

_CHECKED = "__opsd_trainer_checked__"


def require_opsd_trainer(trainer, where: str = "get_trainer"):
    """Return ``trainer`` if it is an ``OPSDRayPPOTrainer``; raise ``TypeError`` otherwise."""
    if not isinstance(trainer, OPSDRayPPOTrainer):
        cls = type(trainer)
        raise TypeError(
            f"{where} returned {cls.__module__}.{cls.__qualname__}, which does not derive from "
            "opsd.skyrl_ext.trainer.OPSDRayPPOTrainer. On plain RayPPOTrainer the run would silently lose OPSD's "
            "eval metrics, the beta/SDAR policy worker, the grpo/* signal and the epoch-boundary resume fix. "
            "Derive the trainer from OPSDRayPPOTrainer."
        )
    return trainer


def _checked_get_trainer(fn):
    if getattr(fn, _CHECKED, False):
        return fn

    @functools.wraps(fn)
    def get_trainer(self, *args, **kwargs):
        return require_opsd_trainer(fn(self, *args, **kwargs), f"{type(self).__qualname__}.get_trainer")

    setattr(get_trainer, _CHECKED, True)
    return get_trainer


class OPSDExpMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        fn = cls.__dict__.get("get_trainer")
        if fn is not None:
            cls.get_trainer = _checked_get_trainer(fn)

    def __init__(self, cfg, *args, **kwargs):
        if not isinstance(cfg, OPSDTrainConfig):
            raise TypeError(
                f"{type(self).__qualname__} needs an OPSDTrainConfig (parse with the entry module's CONFIG_CLS), "
                f"got {type(cfg).__module__}.{type(cfg).__qualname__}"
            )
        register_opsd_envs()
        super().__init__(cfg, *args, **kwargs)

    get_tracker = opsd_tracking.get_tracker

    def get_trainer(
        self, cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator, colocate_pg
    ):
        return OPSDRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )

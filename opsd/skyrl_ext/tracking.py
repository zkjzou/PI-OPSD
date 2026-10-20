# Adapted from SkyRL (Apache-2.0), skyrl/train/entrypoints/main_base.py @ ecdaa946; modified for OPSD:
# BasePPOExp.get_tracker builds OPSDTracking, a Tracking whose log() honours SKYRL_LOG_STEP_OVERRIDE.
"""Experiment tracking for the OPSD entrypoints: :class:`OPSDTracking` and a ``get_tracker`` that builds it."""

import os

from skyrl.train.utils.tracking import Tracking

LOG_STEP_OVERRIDE_ENV = "SKYRL_LOG_STEP_OVERRIDE"


def log_step_override(step):
    """``int(SKYRL_LOG_STEP_OVERRIDE)`` when it is set to an integer, else ``step`` unchanged."""
    override = os.environ.get(LOG_STEP_OVERRIDE_ENV, "").strip()
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    return step


class OPSDTracking(Tracking):
    """``Tracking`` whose every backend logs at ``SKYRL_LOG_STEP_OVERRIDE`` when that is set to an integer.

    The override lets an eval-only process log INTO a RESUMED W&B run at a step of its choosing. A chained
    final test resumes a checkpoint from step N while the parent run's history already reached a later step,
    and W&B silently DROPS log(step=N) when N is behind the run's step pointer (history must be monotonically
    increasing). Pointing the override just past the training tail lets the test's `test/*` metrics land on
    the same run as the dev curve. Unset, empty or non-integer => stock behaviour. The variable is read on
    every call, inside the entrypoint task; ``opsd.skyrl_ext.ray_runtime`` forwards it there.
    """

    def log(self, data, step, commit=False):
        super().log(data, step=log_step_override(step), commit=commit)


def get_tracker(self) -> OPSDTracking:
    """Drop-in for ``BasePPOExp.get_tracker`` (``self`` is the Exp) that returns an :class:`OPSDTracking`."""
    return OPSDTracking(
        project_name=self.cfg.trainer.project_name,
        experiment_name=self.cfg.trainer.run_name,
        backends=self.cfg.trainer.logger,
        config=self.cfg,
    )

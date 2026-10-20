"""Policy-worker selection for the OPSD trainers: ``OPSDWorkerMixin.build_models``.

SkyRL builds every worker inside Ray actors from the classes ``BasePPOExp._setup_trainer`` passes to
``RayPPOTrainer.build_models(PolicyWorker, CriticWorker, RefWorker)``. That argument is the seam: the mixin

  * swaps in ``OPSDFSDPPolicyWorker`` (opsd.skyrl_ext.policy_worker) whenever a loss-side teacher term is
    requested (OPSD_DISTILL_LOSS_COEF or OPSD_SDAR_LAMBDA != 0), whichever trainer runs, so a beta arm on a
    trainer that never ships ``distill_log_probs`` still logs ``distill_loss_attached=0``; with the knobs off
    every actor class is upstream's;
  * refuses those knobs with the Megatron strategy (its worker never sees the teacher tensor; in the fork they
    were a silent no-op there);
  * clamps ``total_training_steps`` to >= 1 for the duration of ``build_models`` on Megatron eval-only runs
    (EVAL_ONLY / EPOCHS=0 give 0 training steps, and megatron's OptimizerParamScheduler asserts
    ``lr_decay_steps > 0`` inside ``MegatronPolicyWorkerBase.init_model`` -- a bare AssertionError that killed
    every held-out Megatron eval BEFORE the checkpoint was read). Nothing trains, so the LR window is empty;
  * pushes the driver-resolved knobs to the policy actors through the ``pass_through`` dispatch, so a typo
    fails here at startup and the actors read no environment.

Mixed into ``OPSDRayPPOTrainer`` ahead of ``RayPPOTrainer``; ``build_models`` calls ``super()`` cooperatively.
"""

import ray
from loguru import logger

from opsd.algorithms.distill_loss import LossKnobs


class OPSDWorkerMixin:
    def build_models(self, PolicyWorker, CriticWorker, RefWorker):
        knobs = LossKnobs.from_env()
        strategy = self.cfg.trainer.strategy
        if knobs.active:
            PolicyWorker = _opsd_policy_worker(strategy, PolicyWorker, knobs)
        self.opsd_loss_knobs = knobs

        total_training_steps = self.total_training_steps
        clamp = strategy == "megatron" and total_training_steps is not None and total_training_steps < 1
        if clamp:
            logger.info(
                f"OPSD: Megatron eval-only run (total_training_steps={total_training_steps}); building the policy "
                "optimizer scheduler with 1 step so megatron's lr_decay_steps > 0 assertion holds"
            )
            self.total_training_steps = 1
        try:
            super().build_models(PolicyWorker, CriticWorker, RefWorker)
        finally:
            if clamp:
                self.total_training_steps = total_training_steps

        if knobs.active:
            ray.get(self.policy_model.async_run_ray_method("pass_through", "set_opsd_loss_knobs", knobs))


def _opsd_policy_worker(strategy: str, PolicyWorker, knobs: LossKnobs):
    """The policy actor class to build when a loss-side teacher term is requested."""
    if strategy == "megatron":
        raise ValueError(
            f"OPSD_DISTILL_LOSS_COEF={knobs.distill_coef} / OPSD_SDAR_LAMBDA={knobs.sdar_lambda} need the FSDP policy "
            "worker: the Megatron worker never receives distill_log_probs, so the term would silently do nothing. "
            "Use trainer.strategy=fsdp or unset both knobs."
        )
    if strategy != "fsdp":
        raise ValueError(f"OPSD loss-side teacher terms support trainer.strategy=fsdp only, got {strategy!r}")

    from opsd.skyrl_ext.policy_worker import OPSDFSDPPolicyWorker
    from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker

    if PolicyWorker is not fsdp_worker.PolicyWorker and PolicyWorker is not OPSDFSDPPolicyWorker:
        # Replacing a custom worker would silently drop whatever it adds.
        raise TypeError(
            f"OPSD would replace PolicyWorker={PolicyWorker!r}, which is not SkyRL's fsdp_worker.PolicyWorker; "
            "subclass OPSDFSDPPolicyWorkerBase instead"
        )
    logger.info(
        f"OPSD: policy worker = OPSDFSDPPolicyWorker (beta={knobs.distill_coef}, sdar_lambda={knobs.sdar_lambda}, "
        f"sdar_beta={knobs.sdar_beta}, sdar_gate={knobs.sdar_gate})"
    )
    return OPSDFSDPPolicyWorker

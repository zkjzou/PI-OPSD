"""OPSD trainer for τ²-bench.

Reuses OPSD's env-agnostic ``SelfDistillationTrainer`` (``opsd/trainer/main_opsd_alfworld.py``) unchanged: the
3-forward ``fwd_logprobs_values_reward`` (policy no-hint / frozen anchor no-hint / frozen teacher WITH-hint),
the ``apply_reward_kl_penalty`` reverse-KL shaping, ``convert_to_training_input`` (``ref_sequences`` from the
generator's ``ref_prompt_token_ids``), and ``compute_advantages_and_returns``, which owns every
advantage-space rewrite (the DAPO x OPSD λ-mix ``OPSD_MIX_LAMBDA<1`` and its step schedule, RLSD
``OPSD_RLSD=1``, the token advantage ``OPSD_TOKEN_ADV=1``), their hard guards, and the live ``pi_utility/*``
diagnostics. None of that is re-implemented here: an override on top of the shared base would apply the
rewrite a second time (the base already does it inside ``super()``).

Two overrides remain, both τ²-specific:

* ``eval()``. OPSD's ALFWorld eval override (``SelfDistillationTrainer.eval``) reads
  ``dumped_evals/<data_source>.jsonl`` to sync sampled trajectories to W&B, which τ² does not need, so we skip
  it and call ``OPSDRayPPOTrainer.eval`` (``OPSDEvalMixin``: opsd's ``evaluate()`` with the per-dataset and
  env-success metrics) directly. Calling ``RayPPOTrainer.eval`` instead would bypass that mixin and silently
  lose those metrics. (τ² registers its metrics-only skyrl_gym env itself in
  ``generator.py:_register_tau2_metrics_env``, which the eval needs to look up ``aggregate_metrics`` by the
  ``tau2_bench`` id.)

* ``postprocess_generator_output``: DAPO's Soft Overlong Punishment, reinstated on the env-anchored arms.
  NOT INHERITING ``Tau2DAPOTrainer`` IS DELIBERATE, and it drops the overlong-buffer penalty applied in
  its ``postprocess_generator_output``. That is correct for vanilla OPSD: ``OPSD_ENV_COEF=0`` discards the
  env reward entirely, so a penalty applied to it would be multiplied by zero. On an env-anchored arm
  (λ-mix, RLSD, token advantage) the reward-shaping path never runs (``use_kl_in_reward=false``), so the
  env reward IS the training reward again and the penalty has to come back. Inheriting from both trainers
  is the wrong fix, since ``postprocess_generator_output`` would then run before OPSD's own reward shaping
  rather than instead of it; this override calls the same ``apply_tau2_overlong_penalty`` helper
  ``Tau2DAPOTrainer`` uses, so the two cannot drift.

NOTE on the ``pi_utility/*`` success label: it comes from ``env_metrics["success"]`` (``raw_reward > 0``),
stashed by the base ``convert_to_training_input``. The pre-integration τ² subclass used ``sum(rewards) > 0``
on the TRAINED reward instead. The two differ for rollouts the generator hard-drops (``generator.py``): an
infrastructure failure, or, during training, a response longer than ``max_response_tokens``. Those get
reward 0 and an all-zero loss mask but keep their ``raw_reward``, so a dropped rollout that had succeeded is
now labelled a success with ``d = 0``, which shifts the nDCG and AUC diagnostics. (A success the overlong
penalty drives to ``<= 0`` would also differ, but at the paper settings, ``OVERLONG_PENALTY_FACTOR=1.0`` and
``OVERLONG_REWARD_RANGE=1.0``, the penalty saturates at 0.5 and a reward-1 success stays positive.)
Monitoring only -- no training tensor depends on the label.
"""

from __future__ import annotations

import torch

from opsd.skyrl_ext.trainer import OPSDRayPPOTrainer

# Importing the τ² generator registers τ²'s metrics-only skyrl_gym env as a side effect (see the module
# docstring); keep the import even though the symbol is unused here.
from opsd.tau2_bench import generator as _tau2_generator  # noqa: F401
from opsd.tau2_bench.main import apply_tau2_overlong_penalty
from opsd.trainer.main_opsd_alfworld import SelfDistillationTrainer
from opsd.trainer.mix_schedule import mix_lambda_configured
from opsd.trainer.rlsd import rlsd_enabled
from opsd.trainer.token_advantage import token_adv_enabled


def env_anchored_mode() -> str:
    """Which advantage rewrite is on: "" (vanilla OPSD / off), "mix" (OPSD_MIX_LAMBDA<1, or a schedule
    that ever leaves 1.0), "rlsd" (OPSD_RLSD=1) or "token_adv" (OPSD_TOKEN_ADV=1).

    All three rewrite the SAME tensor (`advantages`) from the env reward plus the frozen hinted teacher's
    log-probs, so they are mutually exclusive (the base trainer raises at advantage time too; this raises
    earlier, at postprocess time, before any penalty is applied). In every non-empty mode the env reward
    drives the advantage, which is what decides whether DAPO's overlong penalty applies.
    """
    # mix_lambda_configured(): the WHOLE schedule, not the current step -- an arm whose λ has reached 1.0
    # after its ramp is still a mix arm (it needs the teacher tensor and cannot be combined with RLSD).
    on = [
        name
        for name, enabled in (
            ("mix", mix_lambda_configured()),
            ("rlsd", rlsd_enabled()),
            ("token_adv", token_adv_enabled()),
        )
        if enabled
    ]
    if len(on) > 1:
        raise ValueError(
            f"{' + '.join(on)} enabled together, but each rewrites the advantage; pick one "
            "(OPSD_MIX_LAMBDA<1 / OPSD_RLSD=1 / OPSD_TOKEN_ADV=1). Leave OPSD_MIX_LAMBDA unset (=1.0) "
            "to run RLSD or the token advantage."
        )
    return on[0] if on else ""


class Tau2OPSDTrainer(SelfDistillationTrainer):
    async def eval(self):  # noqa: D401
        # Skip SelfDistillationTrainer.eval's `dumped_evals` W&B reader and run the OPSD eval (OPSDEvalMixin)
        # directly. NOT RayPPOTrainer.eval: that would bypass the mixin and its per-dataset / env-success metrics.
        return await OPSDRayPPOTrainer.eval(self)

    @torch.no_grad()
    def postprocess_generator_output(self, generator_output, uids):
        # On an env-anchored arm the env reward IS the training reward again, so DAPO's Soft Overlong
        # Punishment has to be reinstated (vanilla OPSD drops it deliberately -- see the module docstring).
        # It runs here, BEFORE advantages, exactly where Tau2DAPOTrainer applies it, and through the same
        # helper.
        if env_anchored_mode():
            apply_tau2_overlong_penalty(self.cfg, generator_output, self.all_metrics)
        return super().postprocess_generator_output(generator_output, uids)


__all__ = ["Tau2OPSDTrainer", "env_anchored_mode"]

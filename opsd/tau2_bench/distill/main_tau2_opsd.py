"""SkyRL OPSD entrypoint for τ²-bench (on-policy self-distillation).

Reuses τ²'s dataset/config plumbing (``Tau2Exp`` / ``Tau2SkyRLConfig``) and swaps in the OPSD generator
(emits the with-hint ``ref_prompt_token_ids``) and trainer (frozen teacher-with-hint forward +
reverse-KL reward). ``get_train_dataset`` / ``get_eval_dataset`` are inherited unchanged, so the pool
manifest and the ``rollout_batches``-driven repeat size behave exactly as under GRPO.

Required hydra/env combination for VANILLA (pure-distill) OPSD — the τ² launcher
(``scripts/tau2/run.sh``) forces the hydra half for the ``opsd`` recipe objective, so do not rely on
setting them by hand:

    trainer.algorithm.advantage_estimator=no_op
    trainer.algorithm.policy_loss_type=importance_sampling
    trainer.algorithm.use_kl_in_reward=true      # builds the ref worker AND gates the shaping
    trainer.algorithm.use_kl_loss=false          # mutually exclusive with the above (hard assert)
    OPSD_TEACHER=frozen  OPSD_ENV_COEF=0  OPSD_HINTS_PATH=<artifact>

⚠️ WITHOUT A REF WORKER THE "FROZEN" TEACHER SILENTLY BECOMES THE LIVE POLICY. The ref worker is built
only when ``use_kl_in_reward`` or ``use_kl_loss`` is true; if neither is,
``SelfDistillationTrainer.fwd_logprobs_values_reward`` falls back to cloning the policy's own log-probs
and you get the known-collapse configuration with no error. Check ``distill/teacher_is_frozen == 1.0``.

    python -m opsd.tau2_bench.distill.main_tau2_opsd <hydra overrides>
"""

import sys

import ray

from opsd.skyrl_ext.ray_runtime import initialize_ray
from opsd.tau2_bench.distill.generator import Tau2OPSDGenerator
from opsd.tau2_bench.distill.trainer import Tau2OPSDTrainer
from opsd.tau2_bench.main import Tau2Exp, Tau2SkyRLConfig, driver_node_strategy
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.train.utils import validate_cfg

# The config class main() parses with (looked up through this module global, so tests can swap it).
CONFIG_CLS = Tau2SkyRLConfig


class Tau2OPSDExp(Tau2Exp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return Tau2OPSDGenerator(
            generator_cfg=cfg.generator,
            tau2_cfg=cfg.tau2,
            tokenizer=tokenizer,
            policy_model_name=resolve_policy_model_name(cfg),
            inference_engine_client=inference_engine_client,
        )

    def get_trainer(self, *args, **kwargs):
        return Tau2OPSDTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: Tau2SkyRLConfig) -> None:
    Tau2OPSDExp(cfg).run()


def main() -> None:
    cfg = CONFIG_CLS.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    # Pinned to the driver node, like the GRPO entrypoint (see main.driver_node_strategy).
    ray.get(skyrl_entrypoint.options(scheduling_strategy=driver_node_strategy()).remote(cfg))


if __name__ == "__main__":
    main()

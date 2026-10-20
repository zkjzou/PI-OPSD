#!/usr/bin/env python
"""On-Policy Self-Distillation (OPSD) for AppWorld — the ALFWorld OPSD machinery + AppWorld's W&B sections.

Deliberately thin. Neither half needed forking:

  * `main_opsd_alfworld` is env-agnostic despite the name. It rides the generic
    `SkyRLGymGenerator` path, takes the env from the parquet's `env_class`, and keys hints via
    `_task_id_of`, which reads `extra_info.task_id` first — and AppWorld's parquet carries exactly that
    (`appworld_dataset.py` writes `extra_info.task_id` alongside `reward_spec.ground_truth`). So the
    trainer, the hint-injecting generator and the reverse-KL reward all work unchanged.
  * `main_appworld` adds nothing to training either: its only contribution is `AppWorldTracking`, which
    mirrors metrics under `AppWorld/*` so W&B renders a dedicated section.

Combining them is therefore a `get_tracker` override plus, since the DAPO work, a `get_trainer` that
mixes AppWorld's two trainer-level knobs into the OPSD trainer (see AppWorldSelfDistillTrainer;
without it `OVERLONG_BUFFER_LEN` is accepted and silently ignored). Everything else — the reverse-KL
reward, SelfDistillSkyRLGymGenerator / SelfDistillOnlineSkillGenerator (chosen by `OPSD_ONLINE_SKILL`),
the `no_op` advantage estimator — is inherited, so a fix to the ALFWorld OPSD path lands here for free.

Config comes from the AppWorld recipes (`opsd/recipes/appworld/`) via `scripts/appworld/run.sh`, parsed
with `CONFIG_CLS` (`OPSDTrainConfig`), like every OPSD entrypoint.
Note the one hard constraint: OPSD sets `use_kl_in_reward=true`, and `validate_cfg` asserts that is
mutually exclusive with `use_kl_loss`, so `USE_KL_LOSS=false` is REQUIRED or the job dies at startup
before a single rollout. The OPSD objective recipe sets it; don't override it to true.

Like `main_appworld`, the training loop is pinned to the driver's Ray node (`run_on_driver_node`),
because the AppWorld server pool is local to the node that started it.
"""

from __future__ import annotations

import sys

import ray

from opsd.appworld.main_appworld import (
    AppWorldTracking,
    AppWorldTrainerMixin,
    run_on_driver_node,
)
from opsd.skyrl_ext.config import OPSDTrainConfig
from opsd.skyrl_ext.ray_runtime import initialize_ray
from opsd.trainer.main_opsd_alfworld import (
    SelfDistillationTrainer,
    SelfDistillExp,
)
from skyrl.train.utils import validate_cfg

# The config class main() parses with (looked up through this module global, so tests can swap it).
CONFIG_CLS = OPSDTrainConfig


class AppWorldSelfDistillTrainer(AppWorldTrainerMixin, SelfDistillationTrainer):
    """OPSD trainer + AppWorld's two trainer-level knobs (DAPO soft overlong punishment, app-balanced
    train batches). Both are env/config gated and default OFF, so every arm submitted before this class
    existed is unaffected.

    Needed because those knobs are applied by a TRAINER override that previously lived only on the
    pure-GRPO trainer. Setting `OVERLONG_BUFFER_LEN` on an OPSD arm therefore wrote
    `trainer.algorithm.overlong_buffer.len` into the config and into W&B while changing no reward — a
    DAPO arm that looked configured and was not. There is no `overlong_buffer/*` metric on that path
    either, so the no-op left no trace to notice.

    ORDER, which matters for the reward-path (`use_kl_in_reward=true`) arms: the mixin prices length in
    `postprocess_generator_output`, i.e. on the ENV reward, before `apply_reward_kl_penalty` subtracts
    the teacher KL and before advantages are computed. That is DAPO's own ordering (length shaping is
    part of the outcome reward), not a penalty on the distillation signal.
    """


class AppWorldSelfDistillExp(SelfDistillExp):
    """OPSD experiment (generator from SelfDistillExp) with AppWorld's trainer knobs + W&B sectioning."""

    def get_trainer(
        self, cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator, colocate_pg
    ):
        return AppWorldSelfDistillTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )

    def get_tracker(self):
        return AppWorldTracking(
            project_name=self.cfg.trainer.project_name,
            experiment_name=self.cfg.trainer.run_name,
            backends=self.cfg.trainer.logger,
            config=self.cfg,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: OPSDTrainConfig):
    # Runs in a Ray worker, pinned to the driver's node by run_on_driver_node (same reason as main_appworld).
    AppWorldSelfDistillExp(cfg).run()


def main() -> None:
    cfg = CONFIG_CLS.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    run_on_driver_node(skyrl_entrypoint, cfg)


if __name__ == "__main__":
    main()

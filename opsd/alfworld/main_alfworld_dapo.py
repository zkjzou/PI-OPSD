"""ALFWorld DAPO entrypoint: SkyRL's ``main_base`` training loop on the OPSD trainer base + DAPO Soft Overlong
Punishment.

It runs ``skyrl.train.entrypoints.main_base``'s flow with the OPSD pieces every entrypoint shares: the
``OPSDTrainConfig`` parse (``CONFIG_CLS``; upstream ``main_base`` rejects ``trainer.algorithm.overlong_buffer.*``
and ``environment.skyrl_gym.alfworld.*``), ``OPSDExpMixin`` (env registration, tracker) and a trainer on
``OPSDRayPPOTrainer``. With ``trainer.algorithm.overlong_buffer.len=0`` (the default) the trainer adds no
penalty, so it is safe to point every ALFWorld GRPO arm at this module; the penalty only engages when a run
asks for it.

WHY A TRAINER SUBCLASS. ``postprocess_generator_output`` is the seam SkyRL itself uses for this
(``examples/train/algorithms/dapo/main_dapo.py``): it runs after generation and *before* rewards are
converted to per-token form, and ``self.cfg`` there is the whole config. The generator only ever receives
``cfg.generator`` and ``cfg.environment.skyrl_gym`` (see ``main_base.get_generator``), so a generator-side
implementation cannot read ``trainer.algorithm.*`` at all -- an earlier version of this penalty lived there
and died with ``AttributeError: 'SkyRLGymGenerator' object has no attribute 'cfg'`` on the first generate
call of both DAPO arms (the runs started and logged zero rows).

TWO DELIBERATE DEVIATIONS from that upstream example, both because ALFWorld is multi-turn:

1. BUDGET IS THE TRAJECTORY BUDGET. Upstream uses ``max_generate_length`` alone and says so in a comment:
   "NOTE: this is only valid for single turn generation". Here ``response_ids[i]`` is the whole
   trajectory -- every turn's generation concatenated, the quantity that reached 31,341 tokens in a
   Qwen3-4B-Thinking GRPO run -- not one turn's output. Penalising that against a single turn's cap would
   score essentially every multi-turn trajectory maximally overlong. So the onset is measured against
   ``max_input_length + max_generate_length``, which is also the cap the generator itself enforces when it
   stops a trajectory with ``stop_reason="length"``.

2. PER-TOKEN REWARD LISTS ARE HANDLED, NOT ASSERTED AWAY. Upstream asserts
   ``not isinstance(rewards[0], list)``. Multi-turn ALFWorld can carry a per-turn/per-token reward list, so
   the penalty lands on the LAST element, where the terminal env reward sits.

Kept from upstream: past the budget the reward is ZEROED rather than merely penalised, matching
docs.skyrl.ai/docs/algorithms/dapo. Inside the buffer it is a linear ramp, so there is a gradient toward
shorter trajectories rather than a cliff.

Interaction worth knowing: with ``generator.apply_overlong_filtering=true`` a truncated trajectory is
already masked out of the loss, so zeroing its reward is belt-and-braces -- but the ramp *inside* the
buffer applies to trajectories that stopped cleanly and are fully trained on, which is the whole point.
"""

import sys
from typing import List, Tuple

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from opsd.algorithms.overlong_buffer import apply_soft_overlong_punishment
from opsd.skyrl_ext.config import OPSDTrainConfig
from opsd.skyrl_ext.exp import OPSDExpMixin
from opsd.skyrl_ext.ray_runtime import initialize_ray
from opsd.skyrl_ext.trainer import OPSDRayPPOTrainer
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.generators.base import GeneratorOutput
from skyrl.train.utils import validate_cfg

# The config class main() parses with (looked up through this module global, so tests can swap it).
CONFIG_CLS = OPSDTrainConfig


class ALFWorldDAPOTrainer(OPSDRayPPOTrainer):
    """OPSDRayPPOTrainer + DAPO Soft Overlong Punishment on trajectory response length (in tokens)."""

    @torch.no_grad()
    def postprocess_generator_output(
        self, generator_output: GeneratorOutput, uids: List[str]
    ) -> Tuple[GeneratorOutput, List[str]]:
        ob = getattr(self.cfg.trainer.algorithm, "overlong_buffer", None)
        buffer_len = int(getattr(ob, "len", 0) or 0) if ob is not None else 0

        if buffer_len > 0:
            # The trajectory response budget -- see deviation (1) in the module docstring.
            budget = int(self.cfg.generator.max_input_length) + int(
                self.cfg.generator.sampling_params.max_generate_length
            )
            rewards, stats = apply_soft_overlong_punishment(
                rewards=generator_output["rewards"],
                response_lengths=[len(r) for r in generator_output["response_ids"]],
                budget=budget,
                buffer_len=buffer_len,
                penalty_factor=float(getattr(ob, "penalty_factor", 1.0)),
                reward_range=float(getattr(ob, "reward_range", 2.0)),
            )
            generator_output["rewards"] = rewards
            # Log what fired: a group where EVERY sample is penalised by the same amount contributes no
            # GRPO gradient at all (advantage is r - mean(group)), so "did it fire" is not the same
            # question as "did it do anything".
            #
            # Write to self.all_metrics, NOT generator_output["rollout_metrics"]: that key is consumed and
            # POPPED inside generate() (SkyRL trainer.py:740-742 @ ecdaa946), which runs earlier in the same
            # step than this override, so it is already gone here. self.all_metrics accumulates until the step
            # flushes it (trainer.py:341) and is cleared after (:348), so an update here lands in this step's log.
            self.all_metrics.update(stats)

        # Base impl does the per-token reward conversion, pass@N and step-wise merging.
        return super().postprocess_generator_output(generator_output, uids)


class ALFWorldDAPOExp(OPSDExpMixin, BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return ALFWorldDAPOTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: OPSDTrainConfig):
    exp = ALFWorldDAPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = CONFIG_CLS.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    # Pin the training loop to the driver's node. The generator and its env
    # workers run inside this task and reach the ALFWorld server pool at the driver host
    # (ALFWORLD_ADVERTISE_HOST, 127.0.0.1 on a single node), so on a multi-node cluster the task must not
    # be scheduled elsewhere. soft=False: fail to schedule rather than silently run on another node.
    driver_node = NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False)
    ray.get(skyrl_entrypoint.options(scheduling_strategy=driver_node).remote(cfg))


if __name__ == "__main__":
    main()

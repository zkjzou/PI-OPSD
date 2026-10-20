# Adapted from SkyRL (Apache-2.0), skyrl/train/entrypoints/main_generate.py @ ecdaa946; modified for OPSD: parses
# with OPSDTrainConfig, registers the OPSD envs in the eval task (OPSDExpMixin), runs opsd.skyrl_ext.evaluate's
# evaluate / evaluate_step_wise, logs through OPSDTracking, and pins the eval task to the driver's node.
"""Generation-only (eval-only) entrypoint for OPSD configs: ``python -m opsd.skyrl_ext.main_generate <overrides>``.

Upstream ``skyrl.train.entrypoints.main_generate`` cannot run an OPSD config: it parses with
``SkyRLTrainConfig`` (which rejects ``environment.skyrl_gym.alfworld.*`` / ``.appworld.*`` and
``trainer.algorithm.overlong_buffer.*``), never registers the OPSD envs (``aggregate_for_environment`` then raises
``No registered env with id: alfworld``), and imports ``evaluate`` from ``skyrl.train.evaluate``, so its dumps
carry no ``env_metrics`` and its metrics none of OPSD's per-dataset / env-success pass@n. This module is
upstream's flow with those three pieces swapped:

* ``main()`` parses with :data:`CONFIG_CLS` (``OPSDTrainConfig``), validates the generator config, and starts
  Ray through :func:`opsd.skyrl_ext.ray_runtime.initialize_ray` (project env-var forwarding).
* :func:`eval_entrypoint` (a Ray task, pinned to the driver's node: the env servers the launchers start listen
  on that node's loopback) builds :class:`OPSDEvalOnlyEntrypoint`. Its ``OPSDExpMixin.__init__`` registers the
  OPSD envs in this process before the generator exists.
* :meth:`OPSDEvalOnlyEntrypoint.run` is upstream ``EvalOnlyEntrypoint.run`` calling OPSD's
  :func:`~opsd.skyrl_ext.evaluate.evaluate` (or :func:`~opsd.skyrl_ext.evaluate.evaluate_step_wise`). Both
  already apply the ``SKYRL_EVAL_COMPACT`` / ``SKYRL_EVAL_NAMESPACE`` post-processors, so the result is not
  post-processed again here. ``OPSDTracking`` logs it at step 0 (or ``SKYRL_LOG_STEP_OVERRIDE``).
"""

import asyncio
import sys
from typing import Any

import ray
from loguru import logger
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from opsd.skyrl_ext import evaluate as opsd_evaluate
from opsd.skyrl_ext.config import OPSDTrainConfig
from opsd.skyrl_ext.exp import OPSDExpMixin
from opsd.skyrl_ext.ray_runtime import initialize_ray
from skyrl.backends.skyrl_train.inference_engines.base import InferenceEngineInterface
from skyrl.train.entrypoints.main_generate import EvalOnlyEntrypoint
from skyrl.train.utils.trainer_utils import build_dataloader
from skyrl.train.utils.utils import validate_generator_cfg

# The config class main() parses with (looked up through this module global, so tests can swap it).
CONFIG_CLS = OPSDTrainConfig


class OPSDEvalOnlyEntrypoint(OPSDExpMixin, EvalOnlyEntrypoint):
    async def run(self, inference_engine_client: InferenceEngineInterface) -> dict[str, Any]:
        assert self.eval_dataset is not None, "The evaluation only entrypoint requires an eval dataset is provided"

        await inference_engine_client.wake_up()
        generator = self.get_generator(self.cfg, self.tokenizer, inference_engine_client)

        eval_fn = (
            opsd_evaluate.evaluate_step_wise if self.cfg.generator.step_wise_trajectories else opsd_evaluate.evaluate
        )
        results: dict[str, Any] = await eval_fn(
            eval_dataloader=build_dataloader(self.cfg, self.eval_dataset, is_train=False),
            generator=generator,
            cfg=self.cfg,
            global_step=None,
            tokenizer=self.tokenizer,
        )

        tracker = self.get_tracker()
        tracker.log(results, step=0, commit=True)

        return results


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg: OPSDTrainConfig) -> dict:
    # OPSDExpMixin.__init__ registers the OPSD envs here, in the process that builds and scores them.
    exp = OPSDEvalOnlyEntrypoint(cfg)
    # Build the inference client from a sync context so _get_new_inference_client
    # can run its own asyncio.run() for the colocated-mode sleep step.
    inference_engine_client = exp.get_inference_client()
    return asyncio.run(exp.run(inference_engine_client))


def driver_node_strategy() -> NodeAffinitySchedulingStrategy:
    """Hard affinity to the driver's node, where the launchers start the env servers (loopback URLs) and stage
    the run's inputs. A no-op on a single node."""
    return NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False)


def main() -> None:
    cfg = CONFIG_CLS.from_cli_overrides(sys.argv[1:])
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    metrics = ray.get(eval_entrypoint.options(scheduling_strategy=driver_node_strategy()).remote(cfg))
    logger.info(f"Metrics from eval only run: {metrics}")


if __name__ == "__main__":
    main()

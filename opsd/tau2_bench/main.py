"""SkyRL GRPO entrypoint for the τ²-bench integration (Phase 2).

NOTE: deliberately NO ``from __future__ import annotations`` here. SkyRL's ``build_nested_dataclass``
resolves nested config classes from ``field.type``; under PEP-563 those become *strings*, resolution
fails, and ``cfg.tau2`` silently arrives as a plain dict (``AttributeError: 'dict' object has no
attribute ...`` deep inside the trainer). τ1's entrypoint omits the import for the same reason.
"""

import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from opsd.algorithms.overlong_buffer import apply_soft_overlong_punishment
from opsd.skyrl_ext.config import OPSDTrainConfig
from opsd.skyrl_ext.exp import OPSDExpMixin
from opsd.skyrl_ext.ray_runtime import initialize_ray
from opsd.skyrl_ext.trainer import OPSDRayPPOTrainer
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.train.config import BaseConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.generators.base import GeneratorOutput
from skyrl.train.utils import validate_cfg

from .constants import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_SIMULATOR_MODEL,
    DEFAULT_SOURCE_DIR,
    default_llm_base_url,
)
from .generator import Tau2Generator
from .manifest import Tau2ManifestDataset

# NOTE: the skyrl_gym metrics-env registration for "tau2_bench" lives in generator.py, NOT here —
# this module runs as __main__, so Ray cloudpickles the entrypoint BY VALUE and never imports it in
# the eval worker actor. generator.py IS imported by-reference there. (τ1 hit this in production.)


@dataclass
class Tau2Config(BaseConfig):
    source_dir: str = DEFAULT_SOURCE_DIR
    # --- user simulator (remote frontier model on any OpenAI-compatible endpoint; no GPUs) ---
    # Base-URL defaults resolve when the config is built (in the driver): $OPENAI_BASE_URL if set,
    # else https://api.openai.com/v1. The egress allowlist is built from the resolved values.
    simulator_model: str = DEFAULT_SIMULATOR_MODEL
    simulator_base_url: str = field(default_factory=default_llm_base_url)
    simulator_api_key_env: Optional[str] = DEFAULT_LLM_API_KEY_ENV
    simulator_max_tokens: int = 2048
    # Standing simulator config: gpt-5.6-luna, NON-THINKING, temperature 0.0. See UserSpec for the
    # probe evidence. "none" is HONOURED (only value with reasoning_tokens=0; invalid values 400).
    simulator_reasoning_effort: Optional[str] = "none"
    # 0.0 is τ²-bench's OFFICIAL user-leg temperature (verified in Sierra's shipped result files, not
    # just tau2/config.py). ⚠️ luna IGNORES temp 0 (3/4 distinct completions at 0.0), so this does NOT
    # make the environment reproducible -- an earlier version of this comment wrongly claimed it did.
    simulator_temperature: Optional[float] = 0.0
    simulator_top_p: Optional[float] = None
    simulator_top_k: Optional[int] = None
    simulator_min_p: Optional[float] = None
    simulator_presence_penalty: Optional[float] = None
    simulator_enable_thinking: Optional[bool] = False
    # --- EVAL user simulator (optional; empty model => eval reuses the TRAIN simulator) ---
    # The simulator IS the environment, so swapping the training sim for throughput or cost silently
    # moves the yardstick and invalidates every earlier eval number. Pinning eval to a fixed remote
    # sim decouples the two: training rollouts can run against a self-hosted Qwen3.5-35B-A3B while
    # the eval curve stays comparable step-to-step and run-to-run.
    # Defaults are the standing luna config and are DELIBERATELY NOT inherited from simulator_*:
    # UserSpec sends top_p/presence_penalty unconditionally, so a local-Qwen training profile would
    # otherwise be shipped verbatim to a hosted gateway.
    # ⚠️ This buys comparability of the environment DISTRIBUTION, not determinism -- luna ignores
    # temperature 0 (probed: 3/4 distinct completions at 0.0), and derive_seed rotates with
    # global_step so every eval re-seeds anyway.
    eval_simulator_model: Optional[str] = ""
    eval_simulator_base_url: str = field(default_factory=default_llm_base_url)
    eval_simulator_api_key_env: Optional[str] = DEFAULT_LLM_API_KEY_ENV
    eval_simulator_max_tokens: int = 2048
    eval_simulator_reasoning_effort: Optional[str] = "none"
    eval_simulator_temperature: Optional[float] = 0.0
    eval_simulator_top_p: Optional[float] = None
    eval_simulator_top_k: Optional[int] = None
    eval_simulator_min_p: Optional[float] = None
    eval_simulator_presence_penalty: Optional[float] = None
    eval_simulator_enable_thinking: Optional[bool] = False
    # --- NL-assertion judge (retail only; called INSIDE the reward function) ---
    judge_model: str = "openai/gpt-4.1"
    judge_base_url: str = field(default_factory=default_llm_base_url)
    # The judge used to borrow the SIMULATOR's key. That breaks the moment the simulator is
    # self-hosted and keyless: the judge would get "dummy" -> 401 -> judge_api_error, which makes
    # rewards WRONG rather than missing. Falls back to simulator_api_key_env when unset.
    judge_api_key_env: Optional[str] = DEFAULT_LLM_API_KEY_ENV
    # --- policy ---
    policy_api_key_env: Optional[str] = "VLLM_API_KEY"
    # THINKING MODE. True (default) = the Qwen3-4B-Thinking-2507 path, byte-identical to every run
    # to date. False targets a NON-thinking policy (Qwen/Qwen3-4B-Instruct-2507): it sends
    # chat_template_kwargs={"enable_thinking": false} on every policy call.
    # ⚠️ THIS IS LOAD-BEARING, NOT COSMETIC. vLLM builds Qwen3ReasoningParser PER REQUEST from that
    # flag. With it left true against a model that never emits </think>, a plain-text turn comes back
    # as (reasoning=<everything>, content=None) -> τ² raises "AssistantMessage must have either
    # content or tool_calls" and the episode DIES (the 88%-loss mode), while a tool-call turn has
    # everything before <tool_call> silently relocated into `reasoning` before hermes sees it.
    # The flag reaches the parser even though the Instruct chat template has no `enable_thinking`
    # variable: OpenAIServingChat hands the parser the RAW merged chat_template_kwargs, and only the
    # RENDERER filters unknown keys. Verified: Instruct-2507 renders identically true/false/absent.
    policy_enable_thinking: bool = True
    # Feed the agent its own prior <think> back during rollout (source.install_thinking_preservation).
    # None => FOLLOW policy_enable_thinking, i.e. True today and automatically off for a non-thinking
    # policy. That coupling matters: left installed with nothing to inline, the wrapper still counts
    # assistant turns while with_thinking stays 0, driving tau2/think_preserved_frac to a false 0.0 —
    # which its own docstring defines as "broken". Set True/False to override either way.
    preserve_thinking: Optional[bool] = None
    max_generate_length: int = 8192
    # Episodes longer than this are dropped from the objective (reward 0, mask 0) instead of being
    # trained on truncated. Keep < max_model_len - max_generate_length.
    max_response_tokens: int = 24_000
    # --- episode ---
    max_steps: int = 200
    episode_timeout: float = 1200.0
    evaluation_type: str = "ALL"
    max_concurrency: int = 32
    request_timeout: float = 600.0
    seed: int = 0
    rollout_batches: int = 200
    trajectory_dir: Optional[str] = None


@dataclass
class Tau2SkyRLConfig(OPSDTrainConfig):
    """OPSD's config tree (``trainer.algorithm.overlong_buffer`` etc.) plus the ``tau2.*`` block."""

    tau2: Tau2Config = field(default_factory=Tau2Config)


# The config class main() parses with (looked up through this module global, so tests can swap it).
CONFIG_CLS = Tau2SkyRLConfig


class Tau2DAPOTrainer(OPSDRayPPOTrainer):
    """OPSDRayPPOTrainer + DAPO Soft Overlong Punishment, priced against τ²'s OWN trajectory budget.

    No-op when ``trainer.algorithm.overlong_buffer.len <= 0`` (the default), so pointing every τ² arm
    at this trainer is safe -- the penalty engages only when a run asks for it.

    ⚠️ THE BUDGET IS ``tau2.max_response_tokens``, **NOT** ALFWorld's/AppWorld's
    ``max_input_length + max_generate_length``. Copying their formula here yields a SILENT NO-OP:
    on the live 332 arms that sum is 31,000 + 8,000 = 39,000, but ``generator.py:224`` already
    hard-drops anything past ``max_response_tokens`` = 33,000 by zeroing BOTH the reward and the
    whole loss_mask. An onset computed from 39,000 therefore sits 6k tokens ABOVE a cliff that has
    already discarded those trajectories, and ``overlong_buffer/frac_penalized`` would read 0.000
    forever while the config looked correct. τ² is the only one of the three envs with its own
    response cap, which is why it needs its own budget.

    SIZING, from the measured 332-arm rollouts: mean trajectory ~16.5k tokens, drifting +39/step off
    a 13.9k baseline, highest step-mean 18.7k -- i.e. nothing approaches 33,000. A buffer anchored at
    the budget would never fire. To price the observed drift the onset wants to be ~20k, which means
    ``OVERLONG_BUFFER_LEN ~= 13000`` (onset = 33,000 - 13,000).

    ``reward_range`` is 1.0 here: τ² rewards are binary 0/1 (AppWorld's scale), not ALFWorld's
    ``10.0 * won``. ``penalty_factor`` is in RAW reward units, so the published 1.0 against DAPO's
    ±1 span (2.0) saturates at half a success -- the pressure the paper intended.
    """

    @torch.no_grad()
    def postprocess_generator_output(
        self, generator_output: GeneratorOutput, uids: List[str]
    ) -> Tuple[GeneratorOutput, List[str]]:
        apply_tau2_overlong_penalty(self.cfg, generator_output, self.all_metrics)
        return super().postprocess_generator_output(generator_output, uids)


@torch.no_grad()
def apply_tau2_overlong_penalty(cfg, generator_output: GeneratorOutput, metrics: dict) -> None:
    """The Soft Overlong Punishment of ``Tau2DAPOTrainer`` (see its docstring), as a function so the
    OPSD λ-mix trainer applies the IDENTICAL penalty -- in that mode the env reward is the training
    reward again, and a copied block would be free to drift from this one. No-op at ``len <= 0``.

    Writes ``metrics`` (the trainer's ``self.all_metrics``), NOT the generator output's rollout-metrics
    dict: that one is consumed and POPPED inside generate(), which runs earlier in the same step than the
    postprocess hook, so an update there vanishes. See main_alfworld_dapo.py for the full note.
    """
    ob = getattr(cfg.trainer.algorithm, "overlong_buffer", None)
    buffer_len = int(getattr(ob, "len", 0) or 0) if ob is not None else 0
    if buffer_len <= 0:
        return
    rewards, stats = apply_soft_overlong_punishment(
        rewards=generator_output["rewards"],
        response_lengths=[len(r) for r in generator_output["response_ids"]],
        budget=int(cfg.tau2.max_response_tokens),  # NOT max_input+max_generate: see Tau2DAPOTrainer
        buffer_len=buffer_len,
        penalty_factor=float(getattr(ob, "penalty_factor", 1.0)),
        reward_range=float(getattr(ob, "reward_range", 1.0)),
    )
    generator_output["rewards"] = rewards
    metrics.update(stats)


class Tau2Exp(OPSDExpMixin, BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return Tau2DAPOTrainer(*args, **kwargs)

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return Tau2Generator(
            generator_cfg=cfg.generator,
            tau2_cfg=cfg.tau2,
            tokenizer=tokenizer,
            policy_model_name=resolve_policy_model_name(cfg),
            inference_engine_client=inference_engine_client,
        )

    def get_train_dataset(self):
        repeat_size = self.cfg.trainer.train_batch_size * self.cfg.tau2.rollout_batches
        return Tau2ManifestDataset(self.cfg.data.train_data, repeat_size=repeat_size)

    def get_eval_dataset(self):
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            return Tau2ManifestDataset(self.cfg.data.val_data)
        return None


def driver_node_strategy() -> NodeAffinitySchedulingStrategy:
    """Pin the entrypoint task to the node this driver runs on (hard affinity).

    The generator lives inside the entrypoint, so this is where τ² episodes run, trajectories are
    written, and loopback endpoints (a simulator served on 127.0.0.1, see serve_user_sim.sh) resolve.
    On a multi-node cluster an unpinned task may land on any worker node, where none of that holds.
    """

    return NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: Tau2SkyRLConfig) -> None:
    Tau2Exp(cfg).run()


def main() -> None:
    cfg = CONFIG_CLS.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.options(scheduling_strategy=driver_node_strategy()).remote(cfg))


if __name__ == "__main__":
    main()

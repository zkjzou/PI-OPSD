# Adapted from SkyRL (Apache-2.0), skyrl/train/evaluate.py @ ecdaa946; modified for OPSD: per-data-source env
#   classes, eval/all/pass_at_1, eval/all/* rollout metrics re-pooled over every eval batch, post-processors.
# Adapted from SkyRL (Apache-2.0), skyrl/train/utils/trainer_utils.py @ ecdaa946; modified for OPSD: per-data-source
#   pass_at_1 + generate/* + environment/* metrics, per-rollout env_metrics in every dumped eval row.
# Adapted from SkyRL (Apache-2.0), skyrl/train/generators/utils.py @ ecdaa946; modified for OPSD: pass@n counts
#   env_metrics["success"] when every rollout reports it, plus pass_at_1 (wraps the upstream function).
# Adapted from SkyRL (Apache-2.0), skyrl/train/trainer.py @ ecdaa946; modified for OPSD: RayPPOTrainer.eval routed
#   to the evaluate() below (OPSDEvalMixin.eval).
"""OPSD's eval: SkyRL's ``evaluate()`` with the metrics the paper's tables and checkpoint pickers read.

Differences from upstream ``skyrl.train.evaluate.evaluate`` (everything else is the upstream code path):

* pass@n / pass@1 count a rollout as solved by the env's own ``env_metrics[i]["success"]`` when every rollout
  reports one (AppWorld = all unit tests pass, ALFWorld-Passcode = key correct), else by ``reward > 0``.
* ``eval/<data_source>/pass_at_1`` (at N != 1), ``eval/<data_source>/generate/*`` and, with env_metrics,
  ``eval/<data_source>/environment/*`` per data source; ``eval/all/pass_at_1`` at N != 1.
* ``eval/all/environment/*`` is re-pooled over all eval batches. Upstream ``concatenate_generator_outputs``
  SUMS the per-batch environment/* means, so with two eval batches ``eval/all/environment/success`` can be > 1.
* Every ``dumped_evals/<...>/<data_source>.jsonl`` row carries that rollout's ``env_metrics``.
* ``SKYRL_EVAL_COMPACT`` / ``SKYRL_EVAL_NAMESPACE`` post-process the RETURNED dict (``aggregated_results.jsonl``
  keeps the full ``eval/*`` set).

``OPSDEvalMixin`` puts this behind ``RayPPOTrainer.eval``. Step-wise eval stays upstream (via ``super()``), with
only the post-processors applied. Import ``evaluate`` / ``evaluate_step_wise`` from here, never from
``skyrl.train.evaluate``, anywhere OPSD runs an eval (e.g. a generation-only entrypoint).
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_engines.utils import (
    get_sampling_params_for_backend,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.evaluate import evaluate_step_wise as _upstream_evaluate_step_wise
from skyrl.train.generators.base import (
    GeneratorInterface,
    GeneratorOutput,
    MetricsOutput,
)
from skyrl.train.generators.utils import (
    concatenate_generator_outputs,
    get_rollout_metrics,
    prepare_generator_input,
)
from skyrl.train.generators.utils import (
    get_metrics_from_generator_output as _upstream_get_metrics_from_generator_output,
)
from skyrl.train.utils import Timer
from skyrl.train.utils.logging_utils import log_example
from skyrl.train.utils.trainer_utils import (
    sanitize_data_source,
    validate_generator_output,
)

_TRUTHY = ("1", "true", "yes", "on")


class OPSDMetricsOutput(MetricsOutput):
    # pass@1 estimated from the SAME n samples: the mean of the per-rollout solved indicator. Reported next to
    # pass_at_n because pass_at_n alone is not comparable across runs with different n, and because the obvious
    # substitute (environment/success) uses a DIFFERENT solved definition when the env reward is fractional.
    pass_at_1: Optional[float]


@torch.no_grad()
def get_metrics_from_generator_output(generator_output: GeneratorOutput, uids: List[str]) -> OPSDMetricsOutput:
    """Upstream ``get_metrics_from_generator_output`` with OPSD's solved rule, plus ``pass_at_1``.

    ``avg_score`` and ``mean_positive_reward`` (and the empty-reward errors) come from upstream unchanged.
    ``pass_at_n`` counts an example as solved if any of its n rollouts is a trajectory-level success: the env's
    ``env_metrics[i]["success"]`` when EVERY rollout reports one, else ``reward > 0`` (upstream's rule; identical
    for binary-ish rewards such as ALFWorld's 10*won). This matters for a FRACTIONAL reward: AppWorld's
    partial-completion reward makes ``reward > 0`` mean "passed >= 1 subtest". ``pass_at_1`` is the unbiased
    single-attempt estimate from the same samples (the mean of the same solved indicator); at n == 1 it equals
    ``pass_at_n``.
    """
    metrics = _upstream_get_metrics_from_generator_output(generator_output, uids)

    env_metrics = generator_output.get("env_metrics")
    use_env_success = (
        isinstance(env_metrics, list)
        and len(env_metrics) == len(uids)
        and all(isinstance(m, dict) and "success" in m for m in env_metrics)
    )
    uid_to_pass = defaultdict(list)
    if use_env_success:
        for i, m in enumerate(env_metrics):
            uid_to_pass[uids[i]].append(float(m["success"]) > 0)
    else:
        rewards = generator_output["rewards"]
        # Token-level rewards: the last token's reward is the trajectory's, as in upstream.
        token_level = isinstance(rewards[0], list)
        for i, reward in enumerate(rewards):
            uid_to_pass[uids[i]].append((reward[-1] if token_level else reward) > 0)
    pass_at_n = sum(1 for passes in uid_to_pass.values() if any(passes)) / len(uid_to_pass)

    n_rollouts = sum(len(passes) for passes in uid_to_pass.values())
    pass_at_1 = (
        sum(1 for passes in uid_to_pass.values() for p in passes if p) / n_rollouts if n_rollouts else float("nan")
    )

    return OPSDMetricsOutput(
        avg_score=metrics["avg_score"],
        pass_at_n=pass_at_n,
        pass_at_1=pass_at_1,
        mean_positive_reward=metrics["mean_positive_reward"],
    )


def calculate_per_dataset_metrics(
    concat_generator_outputs: GeneratorOutput,
    concat_uids: List[str],
    concat_data_sources: List[str],
    n_samples_per_prompt: int,
    concat_env_classes: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Calculate metrics per data source.

    Emits ``eval/{data_source}/{avg_score,pass_at_N,mean_positive_reward}`` (plus ``pass_at_1`` when N != 1),
    ``eval/{data_source}/generate/*`` and, when the generator output carries per-row ``env_metrics`` and
    ``concat_env_classes`` is given, ``eval/{data_source}/environment/*``. In-domain (alfworld = valid_seen) and
    OOD (alfworld_valid_unseen) evals can then be read separately instead of only under ``eval/all/*``.
    """
    eval_metrics = {}

    # Group indices by data source
    data_source_indices = {}
    for i, data_source in enumerate(concat_data_sources):
        if data_source is None:
            data_source = "unknown"
        if data_source not in data_source_indices:
            data_source_indices[data_source] = []
        data_source_indices[data_source].append(i)

    # Calculate metrics for each data source
    for data_source, indices in data_source_indices.items():
        # Extract subset for this data source
        subset_generator_output = {
            key: [value[i] for i in indices]
            for key, value in concat_generator_outputs.items()
            if isinstance(value, list)
        }
        subset_uids = [concat_uids[i] for i in indices]

        # Calculate metrics for this subset
        overall_metrics = get_metrics_from_generator_output(subset_generator_output, subset_uids)

        # Add to eval metrics with proper naming
        sanitized_data_source = sanitize_data_source(data_source)
        eval_metrics[f"eval/{sanitized_data_source}/avg_score"] = overall_metrics["avg_score"]
        eval_metrics[f"eval/{sanitized_data_source}/pass_at_{n_samples_per_prompt}"] = overall_metrics["pass_at_n"]
        # pass@1 alongside pass@N per pool, so valid_seen and valid_unseen are each reported at both.
        if n_samples_per_prompt != 1:
            eval_metrics[f"eval/{sanitized_data_source}/pass_at_1"] = overall_metrics["pass_at_1"]
        eval_metrics[f"eval/{sanitized_data_source}/mean_positive_reward"] = overall_metrics["mean_positive_reward"]

        # Per-data_source rollout metrics (generate/* always; environment/* when env_metrics present).
        subset_responses = subset_generator_output.get("response_ids")
        subset_rewards = subset_generator_output.get("rewards")
        subset_loss_masks = subset_generator_output.get("loss_masks")
        subset_env_metrics = subset_generator_output.get("env_metrics")
        subset_env_classes = [concat_env_classes[i] for i in indices] if concat_env_classes is not None else None
        if subset_responses is not None and subset_rewards is not None:
            per_ds_rollout = get_rollout_metrics(
                subset_responses,
                subset_rewards,
                env_metrics=subset_env_metrics,
                env_classes=subset_env_classes,
                loss_masks=subset_loss_masks,
            )
            for key, value in per_ds_rollout.items():
                eval_metrics[f"eval/{sanitized_data_source}/{key}"] = value

    return eval_metrics


def dump_per_dataset_eval_results(
    dump_dir_path: Path,
    tokenizer: AutoTokenizer,
    concat_generator_outputs: GeneratorOutput,
    concat_data_sources: List[str],
    concat_all_envs: List[str],
    concat_env_extras: List[Dict[str, Any]],
    eval_metrics: Dict[str, float],
):
    """Dump evaluation results per dataset and overall aggregated results."""

    # Prepare common data
    input_prompts = [tokenizer.decode(prompt) for prompt in concat_generator_outputs["prompt_token_ids"]]
    output_responses = [tokenizer.decode(response) for response in concat_generator_outputs["response_ids"]]
    # Per-rollout env metrics (e.g. ALFWorld-Passcode's task_completed / submitted_decoy). The logged eval metrics
    # are only their MEANS, which cannot give joint rates or paired per-task statistics. None when the generator
    # emits no env_metrics. Numpy scalars -> python so json.dumps cannot fail on them.
    env_metrics = concat_generator_outputs.get("env_metrics") or [None] * len(input_prompts)
    env_metrics = [
        {k: (v.item() if hasattr(v, "item") else v) for k, v in m.items()} if isinstance(m, dict) else m
        for m in env_metrics
    ]

    # Group indices by data source
    data_source_indices = {}
    for i, data_source in enumerate(concat_data_sources):
        if data_source is None:
            data_source = "unknown"
        if data_source not in data_source_indices:
            data_source_indices[data_source] = []
        data_source_indices[data_source].append(i)

    # Dump per-dataset files
    for data_source, indices in data_source_indices.items():
        sanitized_data_source = sanitize_data_source(data_source)
        filename = dump_dir_path / f"{sanitized_data_source}.jsonl"

        with open(filename, "w") as f:
            for i in indices:
                entry = {
                    "input_prompt": input_prompts[i],
                    "output_response": output_responses[i],
                    "score": concat_generator_outputs["rewards"][i],
                    "stop_reason": concat_generator_outputs.get("stop_reasons", [None] * len(input_prompts))[i],
                    "env_class": concat_all_envs[i],
                    "env_extras": concat_env_extras[i],
                    "env_metrics": env_metrics[i] if i < len(env_metrics) else None,
                    "data_source": data_source,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(f"Dumped eval data for {data_source} to {filename}")

    # Dump aggregated results file
    aggregated_filename = dump_dir_path / "aggregated_results.jsonl"
    with open(aggregated_filename, "w") as f:
        f.write(json.dumps(eval_metrics, ensure_ascii=False) + "\n")

    logger.info(f"Dumped aggregated eval metrics to {aggregated_filename}")


def compact_eval_metrics(eval_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Optionally trim the eval metric set for a readable W&B eval panel.

    Keys look like ``eval/<bucket>/<rest...>`` (bucket = data_source or 'all'). When SKYRL_EVAL_COMPACT is truthy,
    keep only the headline score + pass@k + task metrics (``avg_score``, ``pass_at_*``, ``environment/*``) and
    ``generate/avg_num_tokens``, and drop the other per-bucket ``generate/*`` stats and ``mean_positive_reward``.
    Default off -> full set.
    """
    if os.environ.get("SKYRL_EVAL_COMPACT", "0").strip().lower() not in _TRUTHY:
        return eval_metrics

    def _keep(key: str) -> bool:
        parts = key.split("/", 2)
        rest = parts[2] if len(parts) == 3 else parts[-1]
        return (
            rest == "avg_score"
            or rest.startswith("pass_at_")
            or rest.startswith("environment/")
            or rest == "generate/avg_num_tokens"  # avg eval response length -- cheap + useful, keep in compact
        )

    return {k: v for k, v in eval_metrics.items() if _keep(k)}


def namespace_eval_metrics(eval_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Re-namespace an eval into a ``<ns>/*`` panel and add ``pass_at_1`` alongside ``pass_at_N``.

    Set ``SKYRL_EVAL_NAMESPACE=test`` on a FINAL-TEST run (the one that scores the selected checkpoint) so its
    numbers land in their own W&B section instead of being mixed into the ``eval/*`` panel that tracks the dev
    curve during training. Same metric names, different prefix.

    ``<bucket>/pass_at_1`` defaults to ``<bucket>/environment/success`` (the mean over all rollouts, i.e. the
    expected single-sample success rate) whether or not a namespace is set. ``evaluate()`` already emits
    ``pass_at_1`` for every bucket, so this only fills buckets that lack it (e.g. upstream step-wise eval).
    """
    ns = os.environ.get("SKYRL_EVAL_NAMESPACE", "").strip()

    out: Dict[str, Any] = {}
    if ns and ns != "eval":
        for key, val in eval_metrics.items():
            out[f"{ns}/{key[len('eval/'):]}" if key.startswith("eval/") else key] = val
    else:
        out = dict(eval_metrics)

    for key in list(out.keys()):
        if key.endswith("/environment/success"):
            bucket = key[: -len("/environment/success")]
            out.setdefault(f"{bucket}/pass_at_1", out[key])
    return out


def postprocess_eval_metrics(eval_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """SKYRL_EVAL_COMPACT, then SKYRL_EVAL_NAMESPACE. Applied to returned eval dicts, never to the dumps."""
    return namespace_eval_metrics(compact_eval_metrics(eval_metrics))


@torch.no_grad()
async def evaluate(
    eval_dataloader: StatefulDataLoader,
    generator: GeneratorInterface,
    cfg: SkyRLTrainConfig,
    global_step: int | None,
    tokenizer: AutoTokenizer,
) -> Dict[str, float]:
    """Runs generation and evaluation of trajectories.

    Args:
        eval_dataloader (StatefulDataLoader): dataloader of the eval dataset
        generator (GeneratorInterface): generator to use
        cfg (SkyRLTrainConfig): config
        global_step (int | None): current global step, or
            `None` to indicate a non-training context (e.g., eval-only)
        tokenizer (AutoTokenizer): tokenizer to use

    Returns:
        Dict[str, float]: evaluation metrics, post-processed by ``postprocess_eval_metrics``
    """

    # 1. Get all generator outputs
    generator_outputs: List[GeneratorOutput] = []
    concat_all_envs: List[str] = []
    concat_env_extras: List[Dict[str, Any]] = []
    concat_uids: List[str] = []
    sampling_params = cfg.generator.eval_sampling_params
    pbar = tqdm(total=len(eval_dataloader), initial=0, desc="Evaluation Progress")
    for _, prompts in enumerate(eval_dataloader):
        pbar.update(1)
        generator_input, uids = prepare_generator_input(
            prompts,
            cfg.generator.eval_n_samples_per_prompt,
            get_sampling_params_for_backend(cfg.generator.inference_engine.backend, sampling_params),
            cfg.environment.env_class,
            "eval",
            global_step,
        )
        generator_output: GeneratorOutput = await generator.generate(generator_input)
        validate_generator_output(len(generator_input["prompts"]), generator_output)
        generator_outputs.append(generator_output)
        concat_all_envs.extend(generator_input["env_classes"])
        concat_env_extras.extend(generator_input["env_extras"])
        concat_uids.extend(uids)
    concat_generator_outputs: GeneratorOutput = concatenate_generator_outputs(generator_outputs)

    # Extract data_sources from env_extras
    concat_data_sources = [env_extra.get("data_source") for env_extra in concat_env_extras]

    if cfg.trainer.log_example_interval > 0:
        vis = tokenizer.decode(generator_output["response_ids"][0])
        log_example(
            logger,
            prompt=generator_input["prompts"][0],
            response=vis,
            reward=generator_output["rewards"][0],
        )

    # 2. Group data by data source and calculate per-dataset metrics
    eval_metrics = calculate_per_dataset_metrics(
        concat_generator_outputs,
        concat_uids,
        concat_data_sources,
        cfg.generator.eval_n_samples_per_prompt,
        concat_env_classes=concat_all_envs,
    )

    # 3. Calculate overall metrics across all datasets
    overall_metrics = get_metrics_from_generator_output(concat_generator_outputs, concat_uids)
    eval_metrics.update(
        {
            "eval/all/avg_score": overall_metrics["avg_score"],
            f"eval/all/pass_at_{cfg.generator.eval_n_samples_per_prompt}": overall_metrics["pass_at_n"],
            "eval/all/mean_positive_reward": overall_metrics["mean_positive_reward"],
        }
    )
    # pass@1 for the pooled bucket too, matching the per-data-source rule in calculate_per_dataset_metrics.
    # At N=1 pass_at_N already IS pass_at_1.
    if cfg.generator.eval_n_samples_per_prompt != 1:
        eval_metrics["eval/all/pass_at_1"] = overall_metrics["pass_at_1"]

    # Re-pool rollout metrics over ALL eval trajectories. concatenate_generator_outputs combines the per-batch
    # rollout_metrics with a name-based heuristic that SUMS environment/* (per-trajectory means) across eval
    # batches -> inflated values (e.g. success > 1, num_turns > max_turns). Recomputing from the raw per-row
    # env_metrics pools environment/* + generate/* correctly (count-weighted). Custom generator keys not
    # recomputed by get_rollout_metrics are preserved from the concatenated dict.
    pooled = get_rollout_metrics(
        concat_generator_outputs["response_ids"],
        concat_generator_outputs["rewards"],
        env_metrics=concat_generator_outputs.get("env_metrics"),
        env_classes=concat_all_envs,
        loss_masks=concat_generator_outputs.get("loss_masks"),
    )
    combined = dict(concat_generator_outputs["rollout_metrics"])
    combined.update(pooled)
    for key, value in combined.items():
        eval_metrics[f"eval/all/{key}"] = value

    # 4. Prepare dumping data
    # TODO[Ben] update this to be cloud-compatible
    if cfg.trainer.dump_eval_results:
        with Timer("dump_eval_results"):
            data_save_dir = (
                Path(cfg.trainer.export_path)
                / "dumped_evals"
                / ("eval_only" if global_step is None else f"global_step_{global_step}_evals")
            )
            data_save_dir.mkdir(parents=True, exist_ok=True)
            dump_per_dataset_eval_results(
                data_save_dir,
                tokenizer,
                concat_generator_outputs,
                concat_data_sources,
                concat_all_envs,
                concat_env_extras,
                eval_metrics,
            )

    return postprocess_eval_metrics(eval_metrics)


async def evaluate_step_wise(
    eval_dataloader: StatefulDataLoader,
    generator: GeneratorInterface,
    cfg: SkyRLTrainConfig,
    global_step: int | None,
    tokenizer: AutoTokenizer,
) -> Dict[str, float]:
    """Upstream ``skyrl.train.evaluate.evaluate_step_wise`` with only the post-processors applied.

    No OPSD recipe enables ``generator.step_wise_trajectories``; this keeps SKYRL_EVAL_COMPACT /
    SKYRL_EVAL_NAMESPACE working on that path without copying it.
    """
    eval_metrics = await _upstream_evaluate_step_wise(
        eval_dataloader=eval_dataloader,
        generator=generator,
        cfg=cfg,
        global_step=global_step,
        tokenizer=tokenizer,
    )
    return postprocess_eval_metrics(eval_metrics)


class OPSDEvalMixin:
    """``RayPPOTrainer.eval`` through OPSD's ``evaluate()``. Stateless (no ``__init__``); list it before
    ``RayPPOTrainer`` in the bases. Step-wise eval is delegated to the next ``eval`` in the MRO (upstream
    ``evaluate_step_wise``), with the SKYRL_EVAL_* post-processors applied to its result."""

    @torch.no_grad()
    async def eval(self) -> Dict[str, float]:
        """
        Run generation and scoring on the evaluation dataset.

        The eval metrics are recorded after having finished training `self.global_step` steps.
        Metrics recorded in global_step 0 corresponds to evaluations before training.

        Returns:
            A dictionary of evaluation metrics.
        """
        if self.cfg.generator.step_wise_trajectories:
            return postprocess_eval_metrics(await super().eval())
        return await evaluate(
            eval_dataloader=self.eval_dataloader,
            generator=self.generator,
            cfg=self.cfg,
            global_step=self.global_step,
            tokenizer=self.tokenizer,
        )


__all__ = [
    "OPSDEvalMixin",
    "OPSDMetricsOutput",
    "calculate_per_dataset_metrics",
    "compact_eval_metrics",
    "dump_per_dataset_eval_results",
    "evaluate",
    "evaluate_step_wise",
    "get_metrics_from_generator_output",
    "namespace_eval_metrics",
    "postprocess_eval_metrics",
]

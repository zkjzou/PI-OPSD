"""AppWorld training entrypoint — mirrors AppWorld-specific metrics into a dedicated W&B "AppWorld" section.

Training behaviour is ``skyrl.train.entrypoints.main_base``'s (plain GRPO via ``AppBalancedTrainer``, an
``OPSDRayPPOTrainer`` + ``SkyRLGymGenerator`` + the ``appworld`` env), with the OPSD pieces every entrypoint
shares (``CONFIG_CLS`` = ``OPSDTrainConfig``, ``OPSDExpMixin``). The addition here is logging: W&B groups panels into
sections by the metric key's first ``/``-segment, so this entrypoint wraps the tracker to ADD
``AppWorld/…``-prefixed aliases of the AppWorld-relevant metrics. The originals (``environment/*``,
``eval/*``, ``reward/*``) are left untouched, so existing dashboards keep working while a single
"AppWorld" section collects the headline signals:

  AppWorld/train/{success,partial_completion,num_turns,completed_task_called,eval_unavailable,parse_errors}
  AppWorld/train/{avg_raw_reward,num_error_trajectories}
  AppWorld/eval/{test_normal,test_challenge,all}/{success,partial_completion,num_turns,...}
  AppWorld/eval/{test_normal,test_challenge,all}/{avg_score,pass_at_N,mean_positive_reward}

The eval sub-buckets come from the per-split ``data_source`` set in ``appworld_dataset.py``
(``appworld_test_normal`` / ``appworld_test_challenge`` -> SkyRL emits ``eval/<data_source>/*``).

Run with:  python -m opsd.appworld.main_appworld <hydra overrides...>
(normally invoked by ``scripts/appworld/run.sh``; upstream ``main_base`` cannot run these configs, it
rejects ``environment.skyrl_gym.appworld.*`` and ``trainer.algorithm.overlong_buffer.*``).

The training loop is pinned to the DRIVER's Ray node (see ``run_on_driver_node``): the AppWorld server
pool listens on 127.0.0.1 of the node that started it, and server recycling finds the server process by
scanning that node's /proc.
"""

import logging
import os
import sys

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from opsd.algorithms.overlong_buffer import apply_soft_overlong_punishment
from opsd.skyrl_ext.config import OPSDTrainConfig
from opsd.skyrl_ext.exp import OPSDExpMixin
from opsd.skyrl_ext.ray_runtime import initialize_ray
from opsd.skyrl_ext.tracking import OPSDTracking
from opsd.skyrl_ext.trainer import OPSDRayPPOTrainer
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.utils import validate_cfg

logger = logging.getLogger(__name__)

# The config class main() parses with (looked up through this module global, so tests can swap it).
CONFIG_CLS = OPSDTrainConfig

# AppWorld env per-trajectory metrics (from AppWorldEnv.get_metrics()); logged under environment/*.
_ENV_METRIC_KEYS = (
    "success",
    "partial_completion",
    "num_turns",
    "completed_task_called",
    "eval_unavailable",
    "parse_errors",
)
# Eval scalar scores emitted per data_source by opsd.skyrl_ext.evaluate.calculate_per_dataset_metrics().
_EVAL_SCALARS = ("avg_score", "mean_positive_reward")  # + pass_at_<N> handled by prefix


def _short_data_source(data_source: str) -> str:
    """appworld_test_normal -> test_normal; appworld -> all; all -> all (combined bucket)."""
    if data_source == "appworld":
        return "all"
    if data_source.startswith("appworld_"):
        return data_source[len("appworld_") :]
    return data_source


def appworld_section_metrics(data: dict) -> dict:
    """Return AppWorld/*-prefixed aliases for the AppWorld-relevant keys in a metrics dict.

    Pure function (no side effects) so it is unit-testable. Never overwrites existing keys.
    """
    out: dict = {}
    for key, value in data.items():
        if key.startswith("environment/"):
            # train per-turn env metrics: environment/<m> -> AppWorld/train/<m>
            out[f"AppWorld/train/{key[len('environment/'):]}"] = value
        elif key.startswith("eval/") and "/environment/" in key:
            # eval/<data_source>/environment/<m> -> AppWorld/eval/<short>/<m>
            ds, _, metric = key[len("eval/") :].partition("/environment/")
            out[f"AppWorld/eval/{_short_data_source(ds)}/{metric}"] = value
        elif key.startswith("eval/"):
            # eval/<data_source>/{avg_score,pass_at_N,mean_positive_reward} -> AppWorld/eval/<short>/<metric>
            ds, _, metric = key[len("eval/") :].rpartition("/")
            if ds and (metric in _EVAL_SCALARS or metric.startswith("pass_at_")):
                out[f"AppWorld/eval/{_short_data_source(ds)}/{metric}"] = value
    # Headline train scalars that live outside environment/*.
    for src, dst in (
        ("reward/avg_raw_reward", "AppWorld/train/avg_raw_reward"),
        ("generate/num_error_trajectories", "AppWorld/train/num_error_trajectories"),
    ):
        if src in data:
            out[dst] = data[src]
    return out


class AppWorldTracking(OPSDTracking):
    """OPSDTracking (SKYRL_LOG_STEP_OVERRIDE) that also emits AppWorld/*-prefixed aliases so W&B renders a
    dedicated section."""

    def log(self, data, step, commit=False):
        merged = {**data, **appworld_section_metrics(data)}
        super().log(merged, step=step, commit=commit)


class AppWorldTrainerMixin:
    """App-balanced TRAIN batches + DAPO Soft Overlong Punishment, as a MIXIN over any OPSDRayPPOTrainer.

    A mixin rather than a subclass because the OPSD path has a DIFFERENT base
    (``SelfDistillationTrainer``), and the overlong penalty is applied by a trainer override — so on
    that path ``trainer.algorithm.overlong_buffer.len`` parsed cleanly into the config, appeared in
    W&B, and did NOTHING. Mixing in here means a DAPO knob set either works on both entrypoints or
    fails loudly on neither. Do not re-inline these methods into one trainer.

    Only documented hooks are overridden — no framework edits. Eval is untouched. Both features are
    off by default, so the default path is bit-identical to before.

    The overlong punishment reuses ALFWorld's implementation verbatim
    (``opsd/algorithms/overlong_buffer.py``) rather than a second copy, so the two envs share
    one transform and one test suite. ALFWorld has a separate ``main_alfworld_dapo`` entrypoint for its
    DAPO arms; both AppWorld entrypoints mix this in instead, so the knob is honoured on each and cannot be
    silently dropped.
    """

    @torch.no_grad()
    def postprocess_generator_output(self, generator_output, uids):
        """Price trajectory length before advantages are computed. No-op when overlong_buffer.len <= 0."""
        ob = getattr(self.cfg.trainer.algorithm, "overlong_buffer", None)
        buffer_len = int(getattr(ob, "len", 0) or 0) if ob is not None else 0
        if buffer_len > 0:
            # Budget is the WHOLE trajectory's response budget, matching response_ids being all turns
            # concatenated -- max_input_length alone would put the onset 8k tokens too low.
            budget = int(self.cfg.generator.max_input_length) + int(
                self.cfg.generator.sampling_params.max_generate_length
            )
            rewards, stats = apply_soft_overlong_punishment(
                rewards=generator_output["rewards"],
                response_lengths=[len(r) for r in generator_output["response_ids"]],
                budget=budget,
                buffer_len=buffer_len,
                penalty_factor=float(getattr(ob, "penalty_factor", 1.0)),
                # AppWorld's reward spans 1.0 (binary 0/1, and partial TGC is also [0,1]), vs DAPO's
                # +/-1 span of 2.0 -- so the published factor 1.0 saturates at 0.5, half a success,
                # which is the pressure the paper intended. Absolute factors do not transfer between
                # envs: the same 1.0 is a tenth of a success on ALFWorld's 10.0*won scale.
                reward_range=float(getattr(ob, "reward_range", 2.0)),
            )
            generator_output["rewards"] = rewards
            # all_metrics, NOT rollout_metrics: the latter is popped inside generate() earlier in the
            # same step, so an update there would vanish. See main_alfworld_dapo.py for the full note.
            self.all_metrics.update(stats)
        return super().postprocess_generator_output(generator_output, uids)

    def _build_train_dataloader_and_compute_training_steps(self):
        on = os.environ.get("APPWORLD_APP_BALANCED_SAMPLING", "0") == "1"
        # Log the decision UNCONDITIONALLY. If the env var fails to reach this Ray actor the run would
        # otherwise fall back to plain shuffling in silence and burn its whole wall-clock training the
        # thing we were trying to change -- an invisible safe-failure. Absence of the "ON" line is not
        # evidence; this prints either way.
        logging.getLogger(__name__).info(
            "APP-BALANCED SAMPLING: %s (APPWORLD_APP_BALANCED_SAMPLING=%r, APPWORLD_APP_BALANCE_MODE=%r)",
            "ON" if on else "OFF -> plain shuffled sampling",
            os.environ.get("APPWORLD_APP_BALANCED_SAMPLING"),
            os.environ.get("APPWORLD_APP_BALANCE_MODE"),
        )
        if self.train_dataset is None or not on:
            return super()._build_train_dataloader_and_compute_training_steps()

        from torchdata.stateful_dataloader import StatefulDataLoader

        from opsd.appworld.app_balanced_sampler import (
            AppBalancedBatchSampler,
            infer_app_labels,
        )

        mode = os.environ.get("APPWORLD_APP_BALANCE_MODE", "cover")
        labels = infer_app_labels(self.train_dataset)
        sampler = AppBalancedBatchSampler(
            labels,
            batch_size=self.cfg.trainer.train_batch_size,
            mode=mode,
            seed=self.cfg.trainer.seed,
        )
        logging.getLogger(__name__).info("APP-BALANCED SAMPLING ON\n%s", sampler.exposure_summary())

        # batch_sampler owns batching, so batch_size/shuffle/drop_last must NOT be passed alongside it.
        self.train_dataloader = StatefulDataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=0 if self.cfg.generator.inference_engine.enable_http_endpoint else 8,
            multiprocessing_context=("spawn" if not self.cfg.generator.inference_engine.enable_http_endpoint else None),
        )
        # len(sampler) == len(dataset) // batch_size, i.e. the same steps/epoch as the shuffled path,
        # so eval/ckpt intervals and LR schedules are unaffected.
        self.total_training_steps = len(self.train_dataloader) * self.cfg.trainer.epochs


class AppBalancedTrainer(AppWorldTrainerMixin, OPSDRayPPOTrainer):
    """The pure-GRPO AppWorld trainer. Behaviour is unchanged — the body moved to the mixin so the OPSD
    trainer, whose base is SelfDistillationTrainer, gets the same two hooks from one implementation."""


class RolloutDumpingGymGenerator(SkyRLGymGenerator):
    """SkyRLGymGenerator that persists every decoded rollout and renders a W&B `rollouts/*` section.

    WHY: on the pure-GRPO path `DUMP_ROLLOUTS` was a no-op that advertised a capture which never
    happened -- the dump + wandb.Table live only in the OPSD generator, and `main_appworld` + the stock
    SkyRLGymGenerator have no such hook (verified: zero table keys in three GRPO runs that all set the
    var). When a 4B-Thinking Dr. GRPO run collapsed to exactly 0.0 success at ~step 97, the one artifact
    that would have shown WHAT the policy was emitting did not exist.

    NO TRUNCATION, unlike the OPSD dumper this mirrors. That one keeps full text for only the first
    OPSD_DUMP_FULL_N samples and stores `resp_text[-800:]` for the rest, and caps the HTML body at
    `response[:12000]`. Both caps are dropped here: every sample carries the full decoded prompt AND
    response, and the HTML renders the whole thing. A collapse is exactly when the interesting part is
    the middle of a 30k-token degenerate loop, which an 800-char tail throws away.

    Writes to $OPSD_ROLLOUT_DUMP_DIR (same variable as the OPSD dumper, so one setting covers both
    entrypoints); a no-op when it is unset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dump_dir = os.environ.get("OPSD_ROLLOUT_DUMP_DIR", "").strip()
        # rollouts rendered into the W&B section per generate(); 0 disables the panel (files still written)
        self._wandb_traj_n = int(os.environ.get("APPWORLD_WANDB_TRAJ_N", "4") or 0)
        self._dump_counter = 0
        if self._dump_dir:
            os.makedirs(self._dump_dir, exist_ok=True)
            logger.info(f"AppWorld rollout dump ON -> {self._dump_dir} (full text, no truncation)")

    async def generate(self, input_batch, disable_tqdm: bool = False):
        generator_output = await super().generate(input_batch, disable_tqdm)
        try:
            self._dump_and_log(generator_output, input_batch)
        except Exception as e:  # noqa: BLE001 -- capturing rollouts must never kill training
            logger.warning(f"AppWorld rollout dump failed (non-fatal): {e}")
        return generator_output

    def _dump_and_log(self, generator_output, input_batch) -> None:
        if not self._dump_dir:
            return
        import json

        responses = generator_output["response_ids"]
        prompts = generator_output["prompt_token_ids"]
        rewards = generator_output.get("rewards") or []
        stop_reasons = generator_output.get("stop_reasons") or []
        env_metrics = generator_output.get("env_metrics") or []
        env_extras = input_batch.get("env_extras") or []

        path = os.path.join(self._dump_dir, f"rollouts_{self._dump_counter:05d}.jsonl")
        self._dump_counter += 1
        recs = []
        with open(path, "w") as f:
            for i in range(len(responses)):
                rew = rewards[i] if i < len(rewards) else None
                rew_sum = (
                    float(sum(rew)) if isinstance(rew, (list, tuple)) else (float(rew) if rew is not None else None)
                )
                em = env_metrics[i] if i < len(env_metrics) else None
                ee = env_extras[i] if i < len(env_extras) else {}
                # AppWorld keys its task id off reward_spec.ground_truth (AppWorldEnv.__init__ in
                # opsd/envs/appworld/env.py), not a top-level task_id -- reading the wrong one yields None
                # for every row.
                task_id = None
                if isinstance(ee, dict):
                    rs = ee.get("reward_spec") or {}
                    task_id = (rs.get("ground_truth") if isinstance(rs, dict) else None) or ee.get("task_id")
                rec = {
                    "i": i,
                    "task_id": task_id,
                    "reward_sum": rew_sum,
                    "success": (em or {}).get("success") if isinstance(em, dict) else None,
                    "stop_reason": stop_reasons[i] if i < len(stop_reasons) else None,
                    "env_metrics": em,
                    "n_prompt_tokens": len(prompts[i]),
                    "n_response_tokens": len(responses[i]),
                    # FULL text for EVERY sample -- no head/tail cap.
                    "prompt_text": self.tokenizer.decode(prompts[i], skip_special_tokens=False),
                    "response_text": self.tokenizer.decode(responses[i], skip_special_tokens=False),
                }
                recs.append(rec)
                f.write(json.dumps(rec) + "\n")
        logger.info(f"AppWorld: dumped {len(recs)} full rollouts -> {path}")

        if self._wandb_traj_n > 0:
            self._log_rollouts_to_wandb(recs)

    def _log_rollouts_to_wandb(self, recs) -> None:
        """Render a sample of rollouts as a collapsible HTML panel under `rollouts/*`."""
        try:
            import wandb

            if wandb.run is None:
                return
        except Exception:  # noqa: BLE001
            return
        import html as _html

        # Prefer a mix of solved and unsolved so the panel is not all failures at a ~9% success rate.
        solved = [r for r in recs if r.get("success")]
        unsolved = [r for r in recs if not r.get("success")]
        n = self._wandb_traj_n
        picked = (solved[: max(1, n // 2)] + unsolved)[:n] if solved else unsolved[:n]

        parts = [
            "<html><body style='font-family:monospace;font-size:12px'>",
            f"<h3>AppWorld rollouts (step dump {self._dump_counter - 1}, " f"{len(solved)}/{len(recs)} solved)</h3>",
        ]
        for r in picked:
            head = (
                f"task={r['task_id']} success={r.get('success')} reward={r.get('reward_sum')} "
                f"stop={r.get('stop_reason')} resp_tok={r['n_response_tokens']}"
            )
            parts.append(f"<details><summary><b>{_html.escape(str(head))}</b></summary>")
            parts.append("<details><summary>prompt</summary>")
            parts.append(f"<pre style='white-space:pre-wrap'>{_html.escape(r['prompt_text'])}</pre></details>")
            # FULL response -- the OPSD renderer caps this at 12000 chars; a degenerate loop is only
            # legible in full.
            parts.append(f"<pre style='white-space:pre-wrap'>{_html.escape(r['response_text'])}</pre>")
            parts.append("</details><hr>")
        parts.append("</body></html>")
        try:
            wandb.log({"rollouts/train": wandb.Html("".join(parts))}, commit=False)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"AppWorld rollout wandb.Html log failed (non-fatal): {e}")


class AppWorldExp(OPSDExpMixin, BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Use the rollout-dumping generator unless explicitly disabled.

        Mirrors BasePPOExp.get_generator; the VLM branch is not reachable here (AppWorld models are
        text-only), so only the text generator is swapped.
        """
        if os.environ.get("APPWORLD_DUMP_ROLLOUTS", "1").strip().lower() in ("0", "false", "no"):
            return super().get_generator(cfg, tokenizer, inference_engine_client)
        logger.info("AppWorld generator = RolloutDumpingGymGenerator (full-text rollout capture)")
        return RolloutDumpingGymGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            policy_model_name=resolve_policy_model_name(cfg),
        )

    def get_tracker(self):
        return AppWorldTracking(
            project_name=self.cfg.trainer.project_name,
            experiment_name=self.cfg.trainer.run_name,
            backends=self.cfg.trainer.logger,
            config=self.cfg,
        )

    def get_trainer(
        self, cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator, colocate_pg
    ):
        return AppBalancedTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


def run_on_driver_node(remote_fn, cfg: OPSDTrainConfig):
    """Run the `skyrl_entrypoint` task on the same Ray node as this driver and wait for it.

    AppWorld needs this on multi-node clusters: the launcher starts the `appworld serve` pool on the node
    it runs on and passes 127.0.0.1 URLs, AppWorldEnv instances live in the training-loop process (the
    generator's env thread pool), and `AppWorldServerPool._recycle` finds the server pid by scanning the
    local /proc. A training loop scheduled onto another node would reach no server at all. On a single
    node this is a no-op (there is only one node). Hard affinity (``soft=False``), so the task is never
    silently placed elsewhere: if the driver node's CPUs are only temporarily busy the task waits for one,
    but if that node has no CPU resource at all (e.g. a Ray head started with ``--num-cpus=0``) the task
    fails with ``TaskUnschedulableError``. Run the launcher on a node that has CPUs registered with Ray.
    """
    driver_node_id = ray.get_runtime_context().get_node_id()
    strategy = NodeAffinitySchedulingStrategy(node_id=driver_node_id, soft=False)
    return ray.get(remote_fn.options(scheduling_strategy=strategy).remote(cfg))


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: OPSDTrainConfig):
    # A Ray task (not the driver process) so the training loop runs in a Ray worker; pinned to the
    # driver's node by run_on_driver_node (AppWorld servers are local to that node).
    AppWorldExp(cfg).run()


def main() -> None:
    cfg = CONFIG_CLS.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    run_on_driver_node(skyrl_entrypoint, cfg)


if __name__ == "__main__":
    main()

"""On-Policy Self-Distillation (OPSD) for ALFWorld (SkyRLGym multi-turn path).

ALFWorld trains on SkyRL's generic `BasePPOExp` flow with `SkyRLGymGenerator` + `AlfworldEnv`
(`opsd.envs.alfworld`), so this entrypoint pairs the reverse-KL trainer below (on `OPSDRayPPOTrainer`) with a
SkyRLGym generator that emits the with-hint ref prompt.

Mechanism (v2 — anchored + env-mixed; v1's pure-distill variant collapsed, see the trainer docstring):
    policy fwd : log π_θ (tok | NO-hint prompt)   → action_log_probs
    anchor fwd : log π_θ₀(tok | NO-hint prompt)   → base_action_log_probs  (use_kl_loss leash to θ₀)
    teacher fwd: log π_θ₀(tok | WITH-hint prompt) → hinted_log_probs       (OPSD_TEACHER=frozen default)
    reward     : env_reward − OPSD_DISTILL_COEF · clip(k1(student ‖ hinted-teacher))

The student rolls out ONCE, WITH NO HINT (standard ALFWorld ReAct rollout). Per rollout we additionally
build `ref_prompt_token_ids` = the SAME prompt with a `<Secret information>` hint block prepended (the exact
injection screened offline in pi/score_trajectories.py). Responses + loss_masks are shared, so the forwards
align token-for-token on the trailing response.

Teacher regularization (SDPO, arXiv:2601.20802 — see teacher_regularization.py): `frozen` and `live`
are the two endpoints of one trust-region constraint, and SDPO reports both are dominated by a teacher
held between them. OPSD_TEACHER=trust-region blends the two teachers' log-probs (no new weights, θ₀
stays the use_kl_loss anchor, one extra forward); OPSD_TEACHER=ema instead pulls the ref worker's
weights toward the student, θ' ← (1−a)θ' + aθ. Both reduce to `frozen` at a=0 and `live` at a=1.

Hints: OPSD_HINTS_PATH = {task_id: hint} JSON (keyed by extra_info.task_id); OPSD_STATIC_HINT fallback.
Env knobs: OPSD_TEACHER=frozen|live|ema|trust-region, OPSD_TEACHER_ALPHA (0.01), OPSD_EMA_INTERVAL (1),
OPSD_EMA_DIR (<export_path>/opsd_ema), OPSD_DISTILL_COEF (1.0), OPSD_TOKEN_CLIP (0.5), OPSD_LENGTH_NORM (0),
OPSD_ROLLOUT_DUMP_DIR (decoded train-rollout jsonl dumps), OPSD_DUMP_FULL_N (full-text samples per batch).
Recommended cfg (see the ALFWorld recipes, opsd/recipes/alfworld/): advantage_estimator=grpo +
policy_loss_type=regular + eps_clip_high=0.28 + loss_reduction=seq_mean_token_sum_norm + use_kl_loss=true
(the GRPO anti-collapse recipe), use_kl_in_reward=true (gates the distill shaping), ref.model.path=<student>.
"""

import json
import os
import sys

import ray
import torch
from loguru import logger
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from opsd.algorithms import pi_utility
from opsd.algorithms.reward_shaping import (
    apply_side_mask,
    kl_shaping_metrics,
    per_token_reverse_kl,
)
from opsd.pi.artifact import load_task_hints
from opsd.skyrl_ext.config import OPSDTrainConfig
from opsd.skyrl_ext.exp import OPSDExpMixin
from opsd.skyrl_ext.ray_runtime import initialize_ray
from opsd.skyrl_ext.trainer import OPSDRayPPOTrainer
from opsd.trainer.mix_schedule import mix_lambda, mix_lambda_configured
from opsd.trainer.rlsd import (
    rlsd_advantage,
    rlsd_clip,
    rlsd_enabled,
    rlsd_lambda,
    rlsd_teacher_is_inert,
)
from opsd.trainer.teacher_regularization import (
    VALID_MODES,
    EMATeacherSync,
    blend_teacher_log_probs,
    ema_half_life,
    require_ref_model,
    resolve_teacher_alpha,
    resolve_teacher_mode,
)
from opsd.trainer.token_advantage import (
    per_position_group_advantage,
    token_adv_coef,
    token_adv_enabled,
    token_adv_env_broadcast,
    token_adv_norm_std,
)
from skyrl.backends.skyrl_train.distributed.dispatch import loss_fn_outputs_to_tensor
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.ppo_utils import register_advantage_estimator
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.utils import validate_cfg

# The config class main() parses with (looked up through this module global, so tests can swap it).
CONFIG_CLS = OPSDTrainConfig

# The exact PI injection screened offline (pi/score_trajectories.py SECRET_TEMPLATE, system-prepend style):
# a `<|im_start|>system ... <|im_end|>` block prepended to the prompt token ids.
SECRET_TEMPLATE = (
    "<|im_start|>system\n"
    "<Secret information>\n"
    "This section contains privileged hints derived from prior successful trajectories. Use them to stay on\n"
    "track, but never mention that hints were provided.\n"
    "Hint Summary:\n{hint_lines}\n"
    "Your Task:\n"
    "1. Follow the hints to accomplish the task still reasoning before actions.\n"
    "2. Never acknowledge having access to secret hints.\n"
    "</Secret information><|im_end|>\n"
)

_DEFAULT_STATIC_HINT = (
    "Plan before acting: identify the target object(s) and receptacle, check which actions are admissible, "
    "take exactly one valid action per step, and avoid repeating actions that did not change the observation."
)


class SelfDistillationTrainer(OPSDRayPPOTrainer):
    """Two-prompt OPSD **v2** — anchored + env-mixed after the v1 collapse.

    v1 (pure distill) collapsed: reward REPLACED by −rev-KL(no-hint‖with-hint) with the LIVE policy as
    teacher has a trivial optimum — become hint-blind (degenerate never-closing-action loops reach KL≈0) —
    and nothing anchored task performance. v2 changes:
      * env reward is KEPT; the distill term is ADDED as shaping:
            reward = env_reward − OPSD_DISTILL_COEF · clip(k1(student‖hinted-teacher))
      * hinted teacher defaults to the FROZEN ref model θ₀ (OPSD_TEACHER=frozen) — hint-blindness no longer
        zeroes the KL, killing the trivial solution ("live" restores v1 behavior).
      * frozen-ref anchor: base_action_log_probs = ref(θ₀, no-hint) as in the standard trainer, so
        `use_kl_loss=true` (kl_loss_coef) tethers θ to θ₀ (the GRPO-proven leash).
      * OPSD_TOKEN_CLIP defaults ON (0.5) to bound per-token distill pressure.
    Pair with the GRPO anti-collapse recipe (clip-higher 0.28, seq_mean_token_sum_norm, grpo advantage);
    see the ALFWorld recipes (opsd/recipes/alfworld/).
    """

    KL_CLAMP = 20.0
    # Lazily built on the first EMA step (OPSD_TEACHER=ema only) so every other mode pays nothing.
    _ema_teacher = None

    def convert_to_training_input(self, generator_output, uids):
        training_input = super().convert_to_training_input(generator_output, uids)
        # Stash per-rollout env metrics for pi_utility. They have to be captured HERE: env_metrics is a
        # per-rollout list on generator_output and never reaches the TrainingInputBatch, but
        # apply_reward_kl_penalty (where the divergence attributes are computed) only sees the batch.
        #
        # num_turns feeds the `-nstep` reference. Without it `ndcg_at_4_neg_nstep` is silently absent --
        # and that is the one number that says whether `d` ranks success better than trajectory LENGTH,
        # which offline it does NOT (0.9432 for -nstep vs 0.65-0.73 for d-).
        #
        # success is the RANKING LABEL, and taking it from env_metrics rather than from the reward tensor
        # is deliberate. `sum(env_reward) > 0` is an exact success indicator on ALFWorld (10.0*won -
        # 0.1*invalid, so a win is >= 7.0 and a loss <= 0) but not on AppWorld: (a) by this point
        # AppWorldTrainerMixin.postprocess_generator_output has already subtracted the DAPO soft-overlong
        # penalty from data["rewards"], so the sign survives only because that penalty happens to cap at
        # 0.5 < 1.0, and (b) under REWARD_MODE=partial a partial-credit FAILURE scores > 0 and would be
        # labelled a win. Both envs report `success` from get_metrics(), so read it there.
        try:
            em = generator_output.get("env_metrics") or []
            self._last_env_num_turns = [float(m.get("num_turns", 0.0)) for m in em] if em else None
            self._last_env_success = (
                [float(m.get("success", 0.0)) for m in em]
                if em and isinstance(em[0], dict) and "success" in em[0]
                else None
            )
        except Exception:  # noqa: BLE001 — a diagnostic must never break the training step
            self._last_env_num_turns = None
            self._last_env_success = None
        ref_prompt = generator_output.get("ref_prompt_token_ids")
        if ref_prompt is not None:
            responses = generator_output["response_ids"]
            loss_masks = generator_output["loss_masks"]
            dummy_rewards = [[0.0] * len(r) for r in responses]
            tensors = convert_prompts_responses_to_batch_tensors(
                self.tokenizer,
                ref_prompt,
                responses,
                dummy_rewards,
                loss_masks,
                max_seq_len=self.cfg.trainer.algorithm.max_seq_len,
            )
            meta = dict(training_input.metadata or {})
            meta["ref_sequences"] = tensors[0]
            meta["ref_attention_mask"] = tensors[1]
            hints = generator_output.get("hints")
            if hints:
                meta["hints"] = list(hints)
            training_input.metadata = meta
        # W&B: sync a sampled subset of THIS gradient update's train rollouts as an HTML table.
        recs = list(getattr(self.generator, "_last_dump_recs", []) or [])
        self._log_trajectories_to_wandb(recs, key="train/trajectories", step=self.global_step)
        # Also sync ONE trajectory WITH its PI info: reconstruct the teacher's with-hint view
        # (SECRET_TEMPLATE(hint) prepended to the student prompt) so the privileged info the student
        # never sees is visible next to what it produced. Student rollout + hint are otherwise separate.
        pi_recs = [r for r in recs if r.get("hint") and r.get("prompt_text")]
        if pi_recs:
            r0 = dict(pi_recs[0])
            r0["prompt_text"] = SECRET_TEMPLATE.format(hint_lines=r0["hint"]) + r0["prompt_text"]
            self._log_trajectories_to_wandb([r0], key="train/trajectory_with_pi", step=self.global_step)
        return training_input

    # -- W&B trajectory sync (train per gradient update, eval per eval step) ------------------------
    def _log_trajectories_to_wandb(self, recs, key, step):
        """Sample OPSD_WANDB_TRAJ_N trajectories and log them as an HTML wandb.Table at `step`.

        No-op when OPSD_WANDB_TRAJ_N<=0, wandb isn't running, or there are no records. Never raises
        (trajectory viz must not kill training).
        """
        n = int(os.environ.get("OPSD_WANDB_TRAJ_N", "4") or 0)
        if n <= 0 or not recs:
            return
        try:
            import random as _random

            import wandb

            if wandb.run is None:
                return
            from opsd.alfworld.render_trajectories_html import _CSS, _render_traj

            picks = recs if len(recs) <= n else _random.Random(step).sample(recs, n)
            table = wandb.Table(columns=["step", "task_id", "success", "reward", "stop_reason", "trajectory"])
            for j, rec in enumerate(picks):
                em = rec.get("env_metrics") or {}
                succ = em.get("success") if isinstance(em, dict) else None
                html_doc = f"<style>{_CSS}</style>" + _render_traj(rec, j)
                table.add_data(
                    step,
                    str(rec.get("task_id")),
                    "" if succ is None else str(bool(succ)),
                    rec.get("reward_sum"),
                    str(rec.get("stop_reason")),
                    wandb.Html(html_doc),
                )
            wandb.log({key: table}, step=step, commit=False)
            logger.info(f"OPSD: logged {len(picks)} trajectories to W&B '{key}' @ step {step}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"OPSD W&B trajectory sync failed (non-fatal): {e}")

    async def eval(self):
        """Run eval, then sync a sampled subset of eval trajectories (per data_source) to W&B.

        Reads the per-dataset eval dumps SkyRL already writes (dump_eval_results=True) at
        <export_path>/dumped_evals/global_step_<step>_evals/<data_source>.jsonl, so in-domain
        (valid_seen) and OOD (valid_unseen) trajectories log under separate keys.
        """
        metrics = await super().eval()
        try:
            n = int(os.environ.get("OPSD_WANDB_TRAJ_N", "4") or 0)
            if n > 0 and self.cfg.trainer.dump_eval_results:
                d = os.path.join(self.cfg.trainer.export_path, "dumped_evals", f"global_step_{self.global_step}_evals")
                if os.path.isdir(d):
                    for fn in sorted(
                        f for f in os.listdir(d) if f.endswith(".jsonl") and f != "aggregated_results.jsonl"
                    ):
                        ds = fn[:-6]  # strip .jsonl -> data_source (e.g. alfworld_valid_unseen)
                        with open(os.path.join(d, fn)) as fh:
                            recs = [json.loads(line) for line in fh if line.strip()]
                        self._log_trajectories_to_wandb(recs, key=f"eval/{ds}/trajectories", step=self.global_step)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"OPSD eval W&B trajectory sync failed (non-fatal): {e}")
        return metrics

    @torch.no_grad()
    def fwd_logprobs_values_reward(self, training_input: TrainingInputBatch) -> TrainingInputBatch:
        fwd_keys = ["sequences", "attention_mask"]
        if training_input.get("rollout_expert_indices") is not None:
            fwd_keys.append("rollout_expert_indices")
        data_fwd_pass = training_input.select(keys=fwd_keys, metadata_keys=["response_length"])

        # Policy forward — no-hint prompt → student log-probs.
        policy_output = self.dispatch.forward("policy", data_fwd_pass)
        action_log_probs = loss_fn_outputs_to_tensor(policy_output.loss_fn_outputs, key="logprobs")

        # Frozen-ref forward — no-hint prompt → θ₀ log-probs = the ANCHOR consumed by use_kl_loss.
        # (v1 hijacked base_action_log_probs for the hinted teacher — the anchor was lost.)
        if self.ref_model is not None:
            ref_output = self.dispatch.forward("ref", data_fwd_pass)
            anchor_log_probs = loss_fn_outputs_to_tensor(ref_output.loss_fn_outputs, key="logprobs")
            self.dispatch.empty_cache("ref")
        else:
            anchor_log_probs = action_log_probs.clone()

        # Hinted-teacher forward — with-hint prompt over the SAME response tokens.
        #   frozen (default): θ₀+hint via the REF worker → hint-blindness does NOT zero the KL.
        #   live            : current θ+hint via the policy worker (v1 behavior — collapses; see docstring).
        teacher_mode = resolve_teacher_mode()
        teacher_alpha = resolve_teacher_alpha()
        require_ref_model(teacher_mode, self.ref_model)
        self.all_metrics["distill/teacher_mode"] = float(VALID_MODES.index(teacher_mode))
        self.all_metrics["distill/teacher_alpha"] = teacher_alpha if teacher_mode in ("ema", "trust-region") else -1.0
        meta = training_input.metadata or {}
        ref_sequences = meta.get("ref_sequences")
        if ref_sequences is not None:
            hinted_fwd = TrainingInputBatch({"sequences": ref_sequences, "attention_mask": meta["ref_attention_mask"]})
            hinted_fwd.metadata = {"response_length": data_fwd_pass.metadata["response_length"]}
            if teacher_mode == "trust-region":
                # SDPO's explicit trust region: BOTH teachers forward on the with-hint prompt, then
                # blend. theta_0 stays the anchor for use_kl_loss, so unlike the EMA mode this buys a
                # moving teacher without touching the ref weights -- at the cost of one extra forward
                # (the same trade SDPO reports: memory-free, one extra log-prob pass).
                frozen_lp = loss_fn_outputs_to_tensor(
                    self.dispatch.forward("ref", hinted_fwd).loss_fn_outputs, key="logprobs"
                )
                live_lp = loss_fn_outputs_to_tensor(
                    self.dispatch.forward("policy", hinted_fwd).loss_fn_outputs, key="logprobs"
                )
                hinted_log_probs = blend_teacher_log_probs(frozen_lp, live_lp, teacher_alpha)
                # keep the legacy gauge readable: 1.0 == fully frozen, 0.0 == fully live.
                self.all_metrics["distill/teacher_is_frozen"] = 1.0 - teacher_alpha
            else:
                # frozen / ema -> ref worker; live -> policy worker. Under `ema` the ref worker holds
                # theta' (see train_critic_and_policy), so the same forward yields the EMA teacher.
                worker = "ref" if (teacher_mode in ("frozen", "ema") and self.ref_model is not None) else "policy"
                teacher_output = self.dispatch.forward(worker, hinted_fwd)
                hinted_log_probs = loss_fn_outputs_to_tensor(teacher_output.loss_fn_outputs, key="logprobs")
                self.all_metrics["distill/teacher_is_frozen"] = float(worker == "ref")
        else:
            # No hints this batch → zero distill shaping; env reward still trains.
            hinted_log_probs = action_log_probs.clone()
            self.all_metrics["distill/teacher_is_frozen"] = -1.0

        self.dispatch.empty_cache()

        n = len(training_input["sequences"])
        training_input["action_log_probs"] = action_log_probs[:n]
        training_input["base_action_log_probs"] = anchor_log_probs[:n]
        # REWARD path + pi_utility: published UNCONDITIONALLY. apply_reward_kl_penalty consumes and pops it,
        # but the trainer only calls that when use_kl_in_reward=true (SkyRL trainer.py:281 @ ecdaa946). The
        # env-anchored arms (use_kl_in_reward=false) still need it: compute_advantages_and_returns computes
        # pi_utility/* from it for EVERY arm and then pops it right there, so on neither path does the extra key reach the policy
        # worker. (An earlier AppWorld variant gated this publish on use_kl_in_reward to keep the key out of
        # the worker batch; the pop in compute_advantages_and_returns does that without costing the
        # env-anchored arms their pi_utility panels -- test_opsd_env_anchor.py pins the unconditional publish.)
        training_input["hinted_log_probs"] = hinted_log_probs[:n]
        # LOSS path: the teacher-KL belongs here. KL(student||sg[teacher]) depends on theta through
        # log pi_student as well as through the sampling distribution, so folding it into the reward keeps
        # only the REINFORCE term and discards the exact per-token gradient -- and compute_grpo_outcome_
        # advantage then sums it to one scalar anyway. OPSDFSDPPolicyWorker (opsd/skyrl_ext/policy_worker.py)
        # splits this key per micro-batch and turns it into its own loss term.
        # Also carries SDAR's teacher: its gated auxiliary loss lives in the policy worker beside the beta
        # term and reads the same key, so either mechanism being on has to plumb the tensor through.
        if (
            float(os.environ.get("OPSD_DISTILL_LOSS_COEF", "0") or 0) != 0.0
            or float(os.environ.get("OPSD_SDAR_LAMBDA", "0") or 0) != 0.0
        ):
            training_input["distill_log_probs"] = hinted_log_probs[:n]
        # MIXED-ADVANTAGE path: A_SDPO needs the teacher log-probs at advantage time. Its own key because
        # `hinted_log_probs` is popped by apply_reward_kl_penalty (which runs BEFORE advantages whenever
        # use_kl_in_reward=true) and `distill_log_probs` must survive all the way to the policy worker.
        # Keeping them separate means the three paths never interfere.
        # Also carries RLSD's P_T: its evidence ratio w_t = (P_T/P_S)^sign(A) is built at advantage time
        # from the same tensor, so the two mechanisms share one transport key (they are mutually
        # exclusive at the consumer, which raises if both are on).
        # mix_lambda_configured() looks at the WHOLE schedule, not this step: a ramp that starts at lam=1
        # would otherwise never ship the tensor it needs later.
        if mix_lambda_configured() or rlsd_enabled() or token_adv_enabled():
            training_input["mix_teacher_log_probs"] = hinted_log_probs[:n]
        training_input["values"] = None
        return training_input

    def train_critic_and_policy(self, data: TrainingInputBatch):
        """Policy update, then (OPSD_TEACHER=ema) pull the teacher toward the new student.

        Placement mirrors SDPO, which fires its EMA at the tail of `update_policy` after the optimizer
        step, and mirrors SkyRL's own `update_ref_with_policy` (called at this point in the loop, just
        before the sampler weight sync). Skipped when the reported grad norm is non-finite so a blown-up
        step is never absorbed into the teacher -- SDPO guards the same way with its `did_update` flag.
        """
        status = super().train_critic_and_policy(data)
        if resolve_teacher_mode() != "ema":
            return status
        try:
            # `_execute_training_step` reduces the per-mini-batch norms under the key "grad_norm";
            # absent (optim_step returned None) means "unknown", and we proceed rather than skip.
            grad_norm = (status or {}).get("grad_norm")
            if grad_norm is not None and not bool(torch.isfinite(torch.as_tensor(float(grad_norm)))):
                logger.warning(f"OPSD EMA teacher: skipping sync at step {self.global_step} (grad_norm={grad_norm})")
                self.all_metrics["distill/ema_skipped_nonfinite"] = 1.0
                return status
            self.all_metrics["distill/ema_skipped_nonfinite"] = 0.0
            if self._ema_teacher is None:
                alpha = resolve_teacher_alpha()
                interval = int(os.environ.get("OPSD_EMA_INTERVAL", "1") or 1)
                work_dir = os.environ.get("OPSD_EMA_DIR") or os.path.join(self.cfg.trainer.export_path, "opsd_ema")
                self._ema_teacher = EMATeacherSync(
                    dispatch=self.dispatch,
                    tokenizer=self.tokenizer,
                    work_dir=work_dir,
                    alpha=alpha,
                    interval=interval,
                )
                logger.info(
                    f"OPSD EMA teacher: alpha={alpha} interval={interval} "
                    f"(alpha_eff={self._ema_teacher.alpha_eff:.4g}, half-life~{ema_half_life(alpha):.0f} steps) "
                    f"dir={work_dir}"
                )
            if self._ema_teacher.should_apply(self.global_step):
                # Skip the sync once the teacher can no longer affect anything (RLSD past its lambda
                # decay, with no other consumer reading the tensor). Each sync is a full policy export
                # plus a ref re-init, ~165 s measured, and under the paper's schedule two thirds of a
                # 150-step run sits in that regime.
                inert = rlsd_teacher_is_inert(
                    self.global_step, reward_path_active=self.cfg.trainer.algorithm.use_kl_in_reward
                )
                self.all_metrics["distill/ema_skipped_inert"] = float(inert)
                if inert:
                    logger.info(
                        f"OPSD EMA teacher: skipping sync at step {self.global_step} -- RLSD lambda has "
                        "decayed to 0 and no other consumer reads the teacher."
                    )
                else:
                    self.all_metrics.update(self._ema_teacher.apply(self.global_step))
        except Exception:
            # A silently-frozen teacher is a real (if benign) change of algorithm, so unlike the
            # monitoring hooks this must not be swallowed.
            logger.exception("OPSD EMA teacher sync failed")
            raise
        return status

    def _pi_utility_metrics(self, data, student_lp, teacher_lp, loss_mask) -> dict:
        """`pi_utility/*` — the offline PI study's d / d+ / d- / nDCG@4, computed live and for free.

        Mirrors the offline PI-format table's computation so the online numbers stay comparable to the
        published table; see opsd/algorithms/pi_utility.py for the definitions and the two facts
        (random baseline ~0.293, not 0; and `-nstep` out-ranks every divergence attribute) without which
        nDCG@4 is uninterpretable. Free because the OPSD trainer already holds both forward passes:
        `action_log_probs` is the no-hint student and `hinted_log_probs` the with-hint teacher, over
        identical tokens.

        SUCCESS LABEL comes from env_metrics["success"] (stashed in convert_to_training_input); the sign of
        the env reward is only a fallback. On ALFWorld the two agree (`10.0*won - 0.1*invalid` over at most
        30 turns, so a win is >= 7.0 and a loss <= 0; passcode mode reports the verifier's key_correct), but
        on AppWorld the reward sign is wrong under REWARD_MODE=partial and only survives the DAPO overlong
        penalty by luck -- see that stash.

        NOT TABLE-COMPARABLE ON THE ONLINE-ANALYZER ARMS. With OPSD_ONLINE_HINTS_PER_TRAJ=1 the hint store
        is keyed `{task_id}#rep{n}` and each rollout's hint is distilled from THAT SAME rollout, so here
        `d` measures how well a hindsight summary fits the trajectory it was written from -- not whether PI
        transfers to a fresh attempt. The offline table splices in a hint built from a DIFFERENT
        trajectory. On the online-analyzer arms read these as a within-arm time series; only the static-PI
        arms are comparable to the published numbers.

        Any failure here is swallowed: a diagnostic must never take down a 24-hour run.
        """
        try:
            uids = list((data.metadata or {}).get("uids") or [])
            attrs = pi_utility.divergence_attributes(student_lp, teacher_lp, loss_mask, kl_clamp=self.KL_CLAMP)
            n = int(attrs["d"].shape[0])
            successes = getattr(self, "_last_env_success", None)
            if not (isinstance(successes, (list, tuple)) and len(successes) >= n):
                successes = (data["rewards"].sum(dim=-1) > 0).float().cpu().tolist()
            successes = list(successes)[:n]
            if len(uids) != n:
                # No usable grouping -> report the pooled attributes only, and say so rather than
                # silently emitting an nDCG computed over the wrong groups.
                return {
                    "pi_utility/d": float(attrs["d"].mean().item()),
                    "pi_utility/d_plus": float(attrs["d_plus"].mean().item()),
                    "pi_utility/d_minus": float(attrs["d_minus"].mean().item()),
                    "pi_utility/d_signed": float(attrs["d_signed"].mean().item()),
                    "pi_utility/identity_err": pi_utility.check_identity(attrs),
                    "pi_utility/ndcg_both_class_tasks": 0.0,
                }
            n_turns = None
            em = getattr(self, "_last_env_num_turns", None)
            if isinstance(em, (list, tuple)) and len(em) >= n:
                n_turns = list(em)[:n]
            # Cumulative-AUC accumulator, created on first use so a bare/`__new__`-constructed trainer (the
            # unit tests) needs no extra setup. (sum, count) per attribute, so it costs nothing to carry.
            # It is NOT checkpointed: a resumed run restarts from empty, which is why auc_groups_cum is
            # logged beside it -- a drop in that count means "resumed", not "regressed".
            if not hasattr(self, "_pi_auc_cum"):
                self._pi_auc_cum = {}
            return pi_utility.pi_utility_metrics(
                attrs, uids[:n], successes, k=4, n_steps=n_turns, auc_cum_state=self._pi_auc_cum
            )
        except Exception as exc:  # noqa: BLE001 — diagnostics must not kill training
            logger.warning(f"pi_utility metrics skipped: {exc}")
            return {}

    def apply_reward_kl_penalty(self, data: TrainingInputBatch) -> TrainingInputBatch:
        """v2: ADD the distill shaping to the env reward (v1 REPLACED it — no task grounding)."""
        loss_masks_all: torch.Tensor = data["loss_mask"]
        student_lp: torch.Tensor = data["action_log_probs"]
        teacher_lp: torch.Tensor = data["hinted_log_probs"]

        # Per-token k1 rev-KL(student ‖ hinted-teacher): sanitized + clamped to +/-KL_CLAMP + zeroed
        # off-mask, with the OPSD extra per-token clip (v2 default 0.5, bounds the per-token pressure
        # that degraded action tokens first in v1). Shared helper. lp_detail="nan" preserves OPSD's
        # original (student/teacher)_lp_nan diagnostics (OPSD did not log inf/min/max).
        # NB THIS DEFAULT IS 0.5 BUT APPWORLD RUNS AT 0. Nothing here enforces that -- the only thing that
        # sets it is the AppWorld launcher/recipe default OPSD_TOKEN_CLIP=0 (unchanged since the AppWorld
        # run-file was created), against ALFWorld's default of 0.5. So an AppWorld run that does NOT go
        # through that recipe silently gets a clamp its sibling arms never had. Check `distill/token_clip`
        # (logged below).
        token_clip = float(os.environ.get("OPSD_TOKEN_CLIP", "0.5") or 0)
        per_token_kl, raw_diff = per_token_reverse_kl(
            student_lp, teacher_lp, loss_masks_all, kl_clamp=self.KL_CLAMP, token_clip=token_clip
        )
        # Log the EFFECTIVE clip. It was previously visible only in the run-file's stdout banner, which is
        # not collected, so "what clip did this arm actually use?" could not be answered from W&B at all --
        # and the answer changes whether pi_utility/d_plus and distill/side_mean_abs are comparable (they
        # coincide only at 0, since the attributes are deliberately unclipped).
        self.all_metrics["distill/token_clip"] = token_clip
        with torch.no_grad():
            self.all_metrics.update(
                kl_shaping_metrics(
                    raw_diff,
                    per_token_kl,
                    loss_masks_all,
                    student_lp=student_lp,
                    teacher_lp=teacher_lp,
                    lp_detail="nan",
                )
            )
            self.all_metrics.update(self._pi_utility_metrics(data, student_lp, teacher_lp, loss_masks_all))

        distill_coef = float(os.environ.get("OPSD_DISTILL_COEF", "1.0") or 0)
        length_norm = os.environ.get("OPSD_LENGTH_NORM", "0") == "1"
        self.all_metrics["distill/length_norm_on"] = float(length_norm)
        self.all_metrics["distill/distill_coef"] = distill_coef

        # --- d+/d- side ablation (OPSD_DISTILL_SIDE) --------------------------------------------------
        # Applied to the REWARD term only, AFTER every diagnostic above. That ordering is deliberate:
        # train/teacher_kl, the lp diagnostics, pi_utility/* and the per-turn/channel heatmap below all
        # keep measuring the FULL both-sided divergence, so a side arm stays directly comparable to the
        # baseline on every existing panel and `distill/term_sum_mean` is the one number that moves.
        side = (os.environ.get("OPSD_DISTILL_SIDE", "both") or "both").strip().lower()
        distill_term = per_token_kl
        if side != "both":
            distill_term, side_metrics = apply_side_mask(
                per_token_kl,
                loss_masks_all,
                side,
                match_magnitude=os.environ.get("OPSD_DISTILL_SIDE_MATCH", "0") == "1",
            )
            with torch.no_grad():
                self.all_metrics.update(side_metrics)
        else:
            self.all_metrics["distill/side_code"] = 0.0
        if length_norm:
            distill_term = distill_term / loss_masks_all.sum(dim=-1, keepdim=True).clamp(min=1)

        env_rewards: torch.Tensor = data["rewards"]
        with torch.no_grad():
            self.all_metrics["distill/env_reward_sum_mean"] = env_rewards.sum(dim=-1).mean().item()
        # reward = env_coef·env − distill_coef·k1. env_coef=1 (default) = v2 env-mixed shaping;
        # env_coef=0 = the canonical pure-distill reward (v1 / faithful-OPSD arms).
        env_coef = float(os.environ.get("OPSD_ENV_COEF", "1.0") or 0)
        self.all_metrics["distill/env_coef"] = env_coef
        # Log the two terms at the SCALE THEY ACTUALLY ENTER THE REWARD, so the env/distill balance is a
        # measured quantity rather than a guess. It is badly non-obvious: distill_term is DENSE (one k1 per
        # assistant token, ~15.5k of them on AppWorld) while env_rewards is a single end-of-episode spike,
        # so at coef 1/1 the distill side is ~680x larger and OPSD_ENV_COEF=1 shifts the reward by ~0.15%
        # -- i.e. looks like enabling env reward while changing nothing. Ratio >> 1 means distill-dominated.
        with torch.no_grad():
            _d = (distill_coef * distill_term).sum(dim=-1).abs().mean().item()
            _e = (env_coef * env_rewards).sum(dim=-1).abs().mean().item()
            self.all_metrics["distill/term_sum_mean"] = _d
            self.all_metrics["distill/env_term_sum_mean"] = _e
            self.all_metrics["distill/distill_over_env"] = (_d / _e) if _e > 1e-9 else float("inf")
        data["rewards"] = env_coef * env_rewards - distill_coef * distill_term

        # --- monitoring: per-turn x per-channel teacher-KL (line + heatmap), every eval-interval ---
        # best-effort; reuses the per_token_kl already computed above. Channels: agent / think / action.
        try:
            # KL-log cadence: OPSD_KL_LOG_INTERVAL overrides eval_interval (decoupled so a smoke
            # can log every step without also triggering a full eval every step).
            ei = int(os.environ.get("OPSD_KL_LOG_INTERVAL", "") or getattr(self.cfg.trainer, "eval_interval", 0) or 0)
            gs = int(getattr(self, "global_step", 0) or 0)
            if gs <= 0:  # fall back to a per-call step counter if the trainer exposes no global_step
                self._kl_ctr = getattr(self, "_kl_ctr", 0) + 1
                gs = self._kl_ctr
            # Always log the FIRST step (gs==1) as the PRE-TRAINING baseline: those rollouts + student_lp
            # come from the freshly-synced UNTRAINED policy (student==theta0), so step-1 teacher-KL is the
            # raw per-turn/channel PI signal (hint vs no-hint on the base model) before any gradient update.
            if ei > 0 and (gs % ei == 0 or gs == 1) and per_token_kl.shape[0] > 0:
                from opsd.trainer.pi.turn_channel_kl import (
                    channel_totals,
                    compute_turn_channel_kl,
                    log_wandb,
                )

                rl = (getattr(data, "metadata", None) or {}).get("response_length")
                # match the metric's turn range to the episode cap (OPSD_KL_MAX_TURNS -> generator.max_turns
                # -> 32) so a 50-turn budget doesn't silently drop turns 33..50 from the buckets/plot.
                mt = int(os.environ.get("OPSD_KL_MAX_TURNS", "") or getattr(self.cfg.generator, "max_turns", 32) or 32)
                pt, pt_norm = compute_turn_channel_kl(
                    data["sequences"], per_token_kl.detach(), loss_masks_all, rl, self.tokenizer, max_turns=mt
                )
                for _c, _v in channel_totals(pt).items():
                    self.all_metrics[f"train/teacher_kl_{_c}"] = _v  # native W&B lines over step
                if not hasattr(self, "_kl_turn_hist"):
                    self._kl_turn_hist = []
                if not hasattr(self, "_kl_turn_hist_norm"):
                    self._kl_turn_hist_norm = []
                # line + accumulated heatmap for absolute AND normalized turn position (wandb.Image).
                # Merge into all_metrics so the images ride the trainer's single canonical
                # tracker.log(step=global_step, commit=True) — a separate wandb.log(step=gs) races the
                # trainer's step pointer and gets silently dropped (that's why plots were missing).
                self.all_metrics.update(
                    log_wandb(pt, pt_norm, self._kl_turn_hist, self._kl_turn_hist_norm, gs, max_turns=mt)
                )
        except Exception as exc:  # monitoring must never break training
            logger.warning(f"OPSD per-turn/channel KL monitor skipped: {exc}")
        # consumed — drop so downstream batch ops / the policy worker never see the extra key.
        data.pop("hinted_log_probs")
        return data

    @torch.no_grad()
    def compute_advantages_and_returns(self, data: TrainingInputBatch) -> TrainingInputBatch:
        """Mixed advantage:  A_mix(i,t) = lam * A_GRPO(i,t) + (1 - lam) * A_SDPO(i,t).

            A_GRPO(i,t) = (sum_t r_env) - group_mean          -- from super(), constant in t
            A_SDPO(i,t) = log pi(y_t | s, I) - log pi(y_t | s) -- per token, = -(per-token k1)

        Why mix in ADVANTAGE space rather than in the reward: the two signals are then automatically on
        comparable scales, because both are per-token quantities compared token-by-token. Folding the
        teacher term into the reward instead made it a dense sum (~874/episode measured on AppWorld)
        against a sparse env spike -- an imbalance no coefficient search fixes cleanly, and which
        compute_grpo_outcome_advantage then flattens to one scalar anyway, destroying the per-token credit
        the teacher term exists to provide. ALFWorld's env reward (10.0*won) is far larger than AppWorld's
        (~0.014), so the imbalance here is ~87:1 rather than ~63000:1 -- milder, but the scalar-flattening
        objection is unchanged.

        lam=1.0 (default) reproduces the base behaviour exactly, so this is inert unless asked for.
        """
        data = super().compute_advantages_and_returns(data)  # -> data["advantages"] = A_GRPO

        # pi_utility/* FIRST, before any mechanism-specific early return. These are a property of the
        # (student, hinted-teacher) pair, not of whichever mechanism consumes it, so every arm with a
        # teacher forward should report them. Placing the call further down cost exactly that: it sat
        # after the RLSD branch's `return` and after the `lam >= 1.0` return, so of the env-anchored arms
        # only the lam<1 mix arm logged them -- the beta arms (lam=1.0, no mix key) and the RLSD arm
        # (returns in its own branch) silently logged nothing.
        #
        # Source the teacher from `hinted_log_probs`, which fwd_logprobs_values_reward publishes
        # unconditionally and only apply_reward_kl_penalty pops -- and that runs solely when
        # use_kl_in_reward=true, which every env-anchored arm sets false. So it is present here for all of
        # them. `mix_teacher_log_probs` is the fallback for the use_kl_in_reward=true arms, where the
        # unconditional key has already been consumed.
        if "pi_utility/d" not in self.all_metrics:
            _t = data.get("hinted_log_probs")
            if _t is None:
                _t = data.get("mix_teacher_log_probs")
            if _t is not None:
                self.all_metrics.update(self._pi_utility_metrics(data, data["action_log_probs"], _t, data["loss_mask"]))
        # pi_utility was its last consumer: drop it HERE so the key never reaches the policy worker on the
        # env-anchored arms (on the reward-path arms apply_reward_kl_penalty already popped it). Nothing
        # below reads it -- the mix / RLSD / token-adv teacher travels as `mix_teacher_log_probs`, and the
        # beta / SDAR loss teacher as `distill_log_probs`.
        data.pop("hinted_log_probs", None)

        # Per-step lam (constant unless OPSD_MIX_LAMBDA_DECAY_STEPS is set; see mix_schedule.py). The
        # mutual-exclusion checks below use mix_on, which is about the whole schedule, so a mix arm whose
        # lam has reached 1.0 still cannot be silently combined with RLSD / token-adv.
        lam = mix_lambda(int(getattr(self, "global_step", 0) or 0))
        mix_on = mix_lambda_configured()
        teacher_lp = data.get("mix_teacher_log_probs")

        # HARD GUARDS shared by the three advantage rewrites (token advantage, RLSD, lambda mix). Checked
        # once, at the mechanism level and before any branch or early return, so a scheduled mix arm whose
        # lam is 1.0 at this step still fails at step 1 rather than later in the run. Each is a recipe error
        # that would otherwise train for hours on a silently wrong signal.
        if token_adv_enabled():
            rewrite = "OPSD_TOKEN_ADV=1"
        elif rlsd_enabled():
            rewrite = "OPSD_RLSD=1"
        elif mix_on:
            rewrite = "OPSD_MIX_LAMBDA<1"
        else:
            rewrite = ""
        if rewrite:
            self._check_advantage_rewrite_recipe(rewrite, teacher_lp)

        # PER-POSITION GROUP-BASELINED TOKEN ADVANTAGE. Combines the reward first, then baselines it at
        # each POSITION from the rollout group:
        #     R[i,t] = env[i] - beta*k1[i,t];  A[i,t] = R[i,t] - mean_j Rtilde[j,t]
        # with Rtilde filling positions a short rollout never reached with ITS OUTCOME, so the baseline is
        # always a mean over all G rollouts instead of over the one or two still running.
        # This is the middle ground between `grpo` (sums the dense KL into one scalar and broadcasts it,
        # destroying the per-token structure) and `no_op` (keeps the structure but has no baseline at all,
        # so it carries the raw reward's full variance). At beta=0 it is EXACTLY grpo, so beta is a
        # one-variable dial off the reference estimator. Checked before RLSD/mix: all three rewrite
        # `advantages`, and this one derives its own from scratch rather than transforming A_GRPO.
        if token_adv_enabled() and teacher_lp is not None:
            if mix_on or rlsd_enabled():
                raise ValueError(
                    "OPSD_TOKEN_ADV=1 rewrites the advantage, as do OPSD_MIX_LAMBDA<1 and OPSD_RLSD=1. " "Pick one."
                )
            if self.cfg.trainer.algorithm.use_kl_in_reward:
                raise ValueError(
                    "OPSD_TOKEN_ADV=1 needs the PURE env reward: with use_kl_in_reward=true the teacher "
                    "is already folded into data['rewards'], so it would be counted twice. "
                    "Set USE_KL_IN_REWARD=false."
                )
            token_clip = float(os.environ.get("OPSD_TOKEN_CLIP", "0.5") or 0)
            per_token_kl, _ = per_token_reverse_kl(
                data["action_log_probs"],
                teacher_lp,
                data["loss_mask"],
                kl_clamp=self.KL_CLAMP,
                token_clip=token_clip,
            )
            adv, ta_metrics = per_position_group_advantage(
                env_rewards=data["rewards"],
                per_token_kl=per_token_kl,
                response_mask=data["response_mask"],
                uids=(data.metadata or {}).get("uids") or list(range(len(data["rewards"]))),
                beta=token_adv_coef(),
                norm_by_std=token_adv_norm_std(),
                env_broadcast=token_adv_env_broadcast(),
            )
            data["advantages"] = adv
            data["returns"] = adv
            self.all_metrics.update(ta_metrics)
            data.pop("mix_teacher_log_probs", None)
            return data

        # RLSD (arXiv:2604.03128) — MULTIPLICATIVE credit redistribution, checked first because it is a
        # different structure from the convex mix below, not a special case of it: A_hat = A * positive,
        # so the env reward keeps exclusive authority over the SIGN of every token's update and the
        # teacher only moves relative magnitudes within a trajectory. Our own sweep found the convex
        # lambda path (which displaces task signal) lost to the additive beta path at every dose; RLSD
        # displaces nothing. Mutually exclusive with OPSD_MIX_LAMBDA<1 — both rewrite `advantages`.
        if rlsd_enabled() and teacher_lp is not None:
            if mix_on:
                raise ValueError(
                    "OPSD_RLSD=1 and OPSD_MIX_LAMBDA<1 both rewrite the advantage; pick one. "
                    "Leave OPSD_MIX_LAMBDA at 1.0 to run RLSD."
                )
            rlsd_lam = rlsd_lambda(int(getattr(self, "global_step", 0) or 0))
            a_hat, rlsd_metrics = rlsd_advantage(
                a_grpo=data["advantages"],
                student_lp=data["action_log_probs"],
                teacher_lp=teacher_lp,
                loss_mask=data["loss_mask"],
                response_mask=data["response_mask"],
                lam=rlsd_lam,
                eps_w=rlsd_clip(),
            )
            data["advantages"] = a_hat
            data["returns"] = a_hat
            self.all_metrics.update(rlsd_metrics)
            data.pop("mix_teacher_log_probs", None)
            return data

        if lam >= 1.0 or teacher_lp is None:
            # A scheduled arm past its window is plain GRPO/DAPO; still log lam so the switch is verifiable.
            if mix_on:
                self.all_metrics["mix/lambda"] = lam
            data.pop("mix_teacher_log_probs", None)
            return data

        # A_GRPO comes from the PURE env reward here: _check_advantage_rewrite_recipe (above) refuses
        # use_kl_in_reward=true for every mix arm.
        a_grpo = data["advantages"]
        student_lp = data["action_log_probs"]
        loss_mask = data["loss_mask"]
        response_mask = data["response_mask"]

        # Reuse the reward-path helper for identical sanitisation (NaN -> 0, +/-inf -> +/-KL_CLAMP, clamp,
        # optional OPSD_TOKEN_CLIP, zeroed off-mask). It returns student - teacher, and A_SDPO is the
        # opposite sign: positive where the teacher likes the token MORE than the student.
        token_clip = float(os.environ.get("OPSD_TOKEN_CLIP", "0.5") or 0)
        per_token_kl, _ = per_token_reverse_kl(
            student_lp, teacher_lp, loss_mask, kl_clamp=self.KL_CLAMP, token_clip=token_clip
        )
        a_sdpo = -per_token_kl

        mixed = (lam * a_grpo + (1.0 - lam) * a_sdpo) * response_mask
        data["advantages"] = mixed
        data["returns"] = mixed

        # Measure the mix instead of trusting it: if one term dominates, these make it obvious at step 1.
        # NB lam weights UNNORMALISED advantages of different scales, so lam=0.5 is NOT a 50/50 effective
        # mix -- on AppWorld's lam=0.5 arm grpo_over_sdpo started at 0.30 (distill dominating 3.3:1) and
        # drifted to 0.83. Read grpo_over_sdpo, not lam, to know what the run actually optimised.
        rm = response_mask.bool()
        lm = loss_mask.bool()
        _m = lambda t, m: (t[m].abs().mean().item() if m.any() else 0.0)  # noqa: E731
        g_abs, s_abs = _m(a_grpo, rm), _m(a_sdpo, lm)
        self.all_metrics["mix/lambda"] = lam
        self.all_metrics["mix/a_grpo_abs"] = g_abs
        self.all_metrics["mix/a_sdpo_abs"] = s_abs
        self.all_metrics["mix/a_sdpo_max_abs"] = a_sdpo[lm].abs().max().item() if lm.any() else 0.0
        self.all_metrics["mix/contrib_grpo"] = lam * g_abs
        self.all_metrics["mix/contrib_sdpo"] = (1.0 - lam) * s_abs
        self.all_metrics["mix/grpo_over_sdpo"] = (
            (lam * g_abs) / ((1.0 - lam) * s_abs) if (1.0 - lam) * s_abs > 1e-12 else float("inf")
        )
        self.all_metrics["mix/advantage_abs"] = _m(mixed, rm)
        # Fraction of loss tokens where the two terms DISAGREE in sign -- where the teacher is actively
        # pushing against the outcome signal rather than just re-weighting it.
        self.all_metrics["mix/sign_disagree_frac"] = (
            ((a_grpo * a_sdpo) < 0)[lm].float().mean().item() if lm.any() else 0.0
        )
        data.pop("mix_teacher_log_probs", None)
        return data

    def _check_advantage_rewrite_recipe(self, what: str, teacher_lp) -> None:
        """Hard guards for the advantage rewrites (OPSD_MIX_LAMBDA<1 / OPSD_RLSD=1 / OPSD_TOKEN_ADV=1).

        All three rewrite `advantages` from the env reward plus the hinted teacher's log-probs, so they share
        the same forced recipe: grpo estimator, use_kl_in_reward=false, and a ref worker for the frozen
        teacher (use_kl_loss=true at kl_loss_coef=0 keeps it alive). These were warnings (or τ²-only errors)
        before; each one is a misconfiguration that trains on a wrong signal without crashing, so it now
        fails at the first step instead.

        The grpo requirement is load-bearing for the mix and RLSD, which transform A_GRPO. The token
        advantage builds its own group baseline from data["rewards"] and never reads A_GRPO, so for it the
        requirement is a policy choice: its beta=0 point is defined as the grpo reference, and a run
        configured with no_op would record an estimator that is not what its beta=0 point reproduces.
        """
        alg = self.cfg.trainer.algorithm
        if alg.advantage_estimator == "no_op":
            if what == "OPSD_TOKEN_ADV=1":
                raise ValueError(
                    f"{what}: run the token advantage with advantage_estimator=grpo, so its beta=0 point is "
                    "the grpo reference estimator (it builds its own group baseline, so no_op is a recipe "
                    "error rather than a different computation)."
                )
            raise ValueError(f"{what} needs a group-relative estimator (grpo); no_op has no A_GRPO to rewrite.")
        # The token advantage raises its own use_kl_in_reward error in its branch below.
        if alg.use_kl_in_reward and what != "OPSD_TOKEN_ADV=1":
            # apply_reward_kl_penalty has then already overwritten data["rewards"] with
            # env - distill_coef*k1, so A_GRPO would carry the teacher signal and this rewrite would add it
            # again: the teacher is counted twice (once flattened in the reward, once per-token here).
            raise ValueError(
                f"{what} needs use_kl_in_reward=false: A_GRPO must come from the PURE env reward, "
                "otherwise the teacher term is counted twice. Set USE_KL_IN_REWARD=false "
                "(and USE_KL_LOSS=true with KL_LOSS_COEF=0 to keep the frozen teacher's ref worker)."
            )
        if resolve_teacher_mode() == "frozen" and self.all_metrics.get("distill/teacher_is_frozen") == 0.0:
            # No ref worker -> fwd_logprobs_values_reward scored the teacher with the LIVE policy: the
            # hint-blind attractor that collapses. Fail at step 1 instead of training on it.
            raise RuntimeError(
                f"{what} with OPSD_TEACHER=frozen, but the teacher ran on the live policy: no ref worker. "
                "Set USE_KL_LOSS=true with KL_LOSS_COEF=0."
            )
        if teacher_lp is None:
            # fwd_logprobs_values_reward publishes the tensor under exactly these switches, so a missing
            # one is a wiring error; without this the rewrite would silently be skipped (plain GRPO).
            raise KeyError(
                f"{what} is on but the batch carries no 'mix_teacher_log_probs' (the teacher log-probs "
                "fwd_logprobs_values_reward publishes for the advantage rewrites)."
            )


@register_advantage_estimator("no_op")
def compute_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    return token_level_rewards, token_level_rewards


class SelfDistillSkyRLGymGenerator(SkyRLGymGenerator):
    """SkyRLGymGenerator that also emits the with-hint ref prompt per rollout (OPSD).

    The student rolls out with NO hint (base class behavior). After generate(), for each sample we build
    ``ref_prompt_token_ids`` = SECRET_TEMPLATE(hint) tokens + the sample's (no-hint) prompt token ids, keyed
    by ``env_extras[i]["task_id"]``. Responses/loss_masks are unchanged → the trainer's two forwards align.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hints_by_task: dict = {}
        hints_path = os.environ.get("OPSD_HINTS_PATH", "").strip()
        if hints_path and os.path.isfile(hints_path):
            self._hints_by_task = load_task_hints(hints_path)  # {task_id: hint str} (v1/v2 safe)
            logger.info(f"OPSD: loaded {len(self._hints_by_task)} per-task hints from {hints_path}")
        self._static_hint = os.environ.get("OPSD_STATIC_HINT", "").strip() or _DEFAULT_STATIC_HINT
        # OPSD_NO_STATIC_FALLBACK=1 -> a task that is absent from the hints file gets NO hint instead of the
        # generic static one. For PI files that deliberately cover only some tasks (e.g. failed self
        # trajectories: a task the student never failed has none), where the static text would be a
        # different PI. Pair with OPSD_SKIP_NO_HINT=1 so those rollouts leave the loss entirely.
        self._no_static_fallback = os.environ.get("OPSD_NO_STATIC_FALLBACK", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        # Cap on the ref (teacher-with-hint) sequence = hint + prompt + response. Default = the policy
        # budget (max_input_length + max_generate_length); at that default the hint is DROPPED on any
        # sample where hint+prompt+response would exceed it. Set OPSD_REF_SEQ_CAP higher to let the ref
        # forward exceed the policy budget so the FULL hint is always applied — the teacher forward runs
        # on the FSDP ref worker (NOT vLLM), so it is bounded by memory + the model's positional range
        # (Qwen3-4B-Thinking-2507 = 262144), not by max_model_len. Only a ref still exceeding THIS cap
        # drops the hint, as an OOM safety valve for pathological inputs.
        _default_seq_cap = int(self.generator_cfg.max_input_length) + int(
            self.generator_cfg.sampling_params.max_generate_length
        )
        self._seq_cap = int(os.environ.get("OPSD_REF_SEQ_CAP", "") or _default_seq_cap)
        # OPSD_SKIP_NO_HINT=1 -> a rollout that ends up with NO hint is SKIPPED (its loss mask is zeroed)
        # instead of being kept as a plain hint-less sample. Matters for the self/no-fallback arms: a task
        # with no successful student rollout yet gets hint="" -> ref == prompt -> per_token_kl == 0, so the
        # sample carries no distill signal, yet under loss_reduction=token_mean its tokens still sit in the
        # denominator and dilute the hinted samples. Zeroing the mask removes them from BOTH numerator and
        # denominator (masked_mean), i.e. a true skip. Same primitive as DAPO overlong filtering
        # (skyrl/train/generators/utils.py: apply_overlong_filtering).
        self._skip_no_hint = os.environ.get("OPSD_SKIP_NO_HINT", "0").strip().lower() in ("1", "true", "yes")
        # Decoded rollout dump (so humans can inspect what the policy generates during training).
        self._dump_dir = os.environ.get("OPSD_ROLLOUT_DUMP_DIR", "").strip()
        self._dump_full_n = int(os.environ.get("OPSD_DUMP_FULL_N", "8") or 0)
        # Save the FULL hint per rollout (hints are already capped to 24k tokens upstream — offline files
        # + online MAX_CHARS). 120000 chars comfortably holds a 24k-token hint (~3-4 ch/tok); it's a
        # defensive backstop, not a re-truncation. (Was hardcoded 2000, which gutted the saved hint.)
        self._dump_hint_max_chars = int(os.environ.get("OPSD_DUMP_HINT_MAX_CHARS", "120000") or 120000)
        # HTML render caps for the `rollouts/*` panels. **0 = unlimited.** Defaults preserve the previous
        # hardcoded values, so no existing run changes. A collapse is exactly when the interesting part is
        # the MIDDLE of a 30k-token degenerate loop, which a head-cap throws away -- set these to 0
        # alongside OPSD_DUMP_FULL_N when the run is one you intend to post-mortem.
        self._html_prompt_chars = int(os.environ.get("OPSD_DUMP_HTML_PROMPT_CHARS", "6000") or 0)
        self._html_resp_chars = int(os.environ.get("OPSD_DUMP_HTML_RESP_CHARS", "12000") or 0)
        self._dump_counter = 0
        # # rollouts to log as wandb.Html per generate() (train / train_with_hint / eval); 0 disables.
        self._wandb_traj_n = int(os.environ.get("OPSD_WANDB_TRAJ_N", "4") or 0)
        # W&B trajectory sync: the trainer reads self._last_dump_recs after each generate() and logs a
        # sampled subset as a wandb.Table (see SelfDistillationTrainer._log_trajectories_to_wandb).
        self._last_dump_recs: list = []
        # Per-sample hint provenance, populated by the ONLINE subclass's _resolve_hints (empty offline, where
        # the hint comes from a static file and the per-task meta already lives in that file).
        self._hint_meta: list = []
        if self._dump_dir:
            os.makedirs(self._dump_dir, exist_ok=True)
        logger.info(
            f"OPSD ALFWorld generator | per-task hints: {len(self._hints_by_task)} | "
            f"static-hint fallback: {'OFF' if self._no_static_fallback else f'{len(self._static_hint)} chars'} | "
            f"skip_no_hint={self._skip_no_hint} | seq_cap={self._seq_cap} | "
            f"rollout_dump={self._dump_dir or 'off'} (full_n={self._dump_full_n})"
        )

    @staticmethod
    def _task_id_of(ee):
        """Extract the per-task id used to key PI hints.

        The ALFWorld parquet nests it under ``extra_info`` (``env_extras[i]`` =
        ``{data_source, reward_spec, extra_info}``, with ``extra_info.task_id``); some older layouts
        keep it top-level. Read ``extra_info.task_id`` first, fall back to a flat ``task_id``. Reading
        only the top level returns ``None`` for every rollout → silent static-hint fallback (the PI
        hints never match), which is the bug this guards against.
        """
        if not isinstance(ee, dict):
            return None
        ei = ee.get("extra_info")
        if isinstance(ei, dict) and ei.get("task_id") is not None:
            return ei.get("task_id")
        return ee.get("task_id")

    def _hint_for_task_id(self, task_id):
        if task_id is not None:
            if task_id in self._hints_by_task:
                return self._hints_by_task[task_id]
            base = str(task_id).split("__")[0]
            if base in self._hints_by_task:
                return self._hints_by_task[base]
        return "" if self._no_static_fallback else self._static_hint

    def _ref_ids(self, prompt_ids, hint):
        block = self.tokenizer.encode(SECRET_TEMPLATE.format(hint_lines=hint), add_special_tokens=False)
        return block + list(prompt_ids)

    async def _resolve_hints(self, generator_output, env_extras, input_batch) -> list:
        """Per-rollout teacher hint. Base: static/offline hint keyed by ``extra_info.task_id``.

        Async so a subclass can await an analyzer LLM. Overridable seam — the online on-policy
        skill generator overrides this to build the hint from THIS batch's fresh rollouts.
        """
        n = len(generator_output["prompt_token_ids"])
        return [
            self._hint_for_task_id(self._task_id_of(env_extras[i] if i < len(env_extras) else {})) for i in range(n)
        ]

    async def generate(self, input_batch, disable_tqdm: bool = False):
        generator_output = await super().generate(input_batch, disable_tqdm)
        if self.generator_cfg.step_wise_trajectories:
            # OPSD ref-prompt is per-trajectory; step-wise splitting is out of scope for this baseline.
            logger.warning("OPSD: step_wise_trajectories=True — skipping ref-prompt (no OPSD signal).")
            return generator_output

        prompt_token_ids = generator_output["prompt_token_ids"]
        responses = generator_output["response_ids"]
        env_extras = input_batch["env_extras"]
        hints = await self._resolve_hints(generator_output, env_extras, input_batch)
        ref_prompt_token_ids, hints_used = [], []
        n_hinted, n_dropped = 0, 0
        for i, p_ids in enumerate(prompt_token_ids):
            hint = hints[i] if i < len(hints) else ""
            ref = self._ref_ids(p_ids, hint) if hint else list(p_ids)
            # Overflow safety: drop the hint (ref == prompt → zero distill shaping for THIS sample only —
            # env reward still trains it; v2 counts + logs this instead of failing silently like v1).
            if hint and (len(ref) + len(responses[i]) > self._seq_cap):
                ref, hint = list(p_ids), ""
                n_dropped += 1
            if hint:
                n_hinted += 1
            ref_prompt_token_ids.append(ref)
            hints_used.append(hint or "")
        generator_output["ref_prompt_token_ids"] = ref_prompt_token_ids
        generator_output["hints"] = hints_used
        # OPSD_SKIP_NO_HINT: drop hint-less rollouts from the loss entirely (see __init__ for why).
        n_skipped = 0
        if self._skip_no_hint:
            loss_masks = generator_output.get("loss_masks")
            if loss_masks:
                for i, hint in enumerate(hints_used):
                    if not hint and i < len(loss_masks):
                        loss_masks[i] = [0] * len(loss_masks[i])
                        n_skipped += 1
            else:
                logger.warning("OPSD_SKIP_NO_HINT set but generator_output has no loss_masks — not skipping.")
        rm = generator_output.get("rollout_metrics")
        if isinstance(rm, dict):
            rm["opsd/hints_attached"] = n_hinted
            rm["opsd/hints_dropped_overflow"] = n_dropped
            # Rollouts zeroed out of the loss because they had no hint (self arms w/o fallback). Should
            # start high and decay as the online hint store fills; a flat-high curve means the student
            # never succeeds and the arm is learning from almost nothing.
            rm["opsd/samples_skipped_no_hint"] = n_skipped
            # 0 here = OPSD_HINTS_PATH never reached this worker (env-propagation bug) → static fallback.
            rm["opsd/task_hints_loaded"] = len(self._hints_by_task)
            rm.update(getattr(self, "_last_online_metrics", {}) or {})  # online-skill metrics (if any)
        logger.info(
            f"OPSD: ref prompts for {len(ref_prompt_token_ids)} rollouts "
            f"({n_hinted} hinted, {n_dropped} dropped-overflow, {n_skipped} skipped-no-hint)"
        )
        self._dump_rollouts(generator_output, env_extras)
        self._log_rollouts_wandb(
            generator_output, env_extras, ref_prompt_token_ids, hints_used, input_batch.get("batch_metadata")
        )
        return generator_output

    def _n_samples_for_dump(self) -> int:
        """n_samples_per_prompt, for mapping a flat dump index to its repetition id. Defined on the base
        class (not just the online subclass) because _dump_rollouts runs for both paths."""
        return int(getattr(self.generator_cfg, "n_samples_per_prompt", 1) or 1)

    def _dump_rollouts(self, generator_output, env_extras) -> None:
        """Append one decoded jsonl per generate() call under OPSD_ROLLOUT_DUMP_DIR.

        First OPSD_DUMP_FULL_N samples carry the FULL decoded prompt+response; the rest carry a summary
        (tail of the response) to keep files small. Copy the dir off the training node to inspect it.

        Set OPSD_DUMP_FULL_N above the batch size (e.g. 100000) for untruncated capture of EVERY rollout.
        Cost is ~20 MB/step at 144 rollouts x ~30k chars, i.e. ~3 GB over a 150-step run -- negligible
        next to a single 45 GiB checkpoint, and the only artifact that says what the policy emitted as it
        died. The default stays 8 so unrelated runs are unaffected.
        """
        if not self._dump_dir:
            return
        try:
            responses = generator_output["response_ids"]
            prompts = generator_output["prompt_token_ids"]
            rewards = generator_output.get("rewards") or []
            stop_reasons = generator_output.get("stop_reasons") or []
            env_metrics = generator_output.get("env_metrics") or []
            hints = generator_output.get("hints") or []
            path = os.path.join(self._dump_dir, f"rollouts_{self._dump_counter:05d}.jsonl")
            self._dump_counter += 1
            recs = []
            with open(path, "w") as f:
                for i in range(len(responses)):
                    rew = rewards[i] if i < len(rewards) else None
                    rew_sum = (
                        float(sum(rew)) if isinstance(rew, (list, tuple)) else (float(rew) if rew is not None else None)
                    )
                    ee = env_extras[i] if i < len(env_extras) else {}
                    resp_text = self.tokenizer.decode(responses[i], skip_special_tokens=False)
                    hmeta = self._hint_meta[i] if i < len(self._hint_meta) else None
                    # Always keep FULL text for the rollout the analyzer actually read, otherwise the
                    # trajectory a hint was derived from is usually truncated: the analyzer picks one rep per
                    # GROUP, while `i < _dump_full_n` only covers the first _dump_full_n indices (= the first
                    # task's reps at n_samples=8). Costs one extra full record per group per step.
                    analyzed = bool(hmeta) and hmeta.get("rep") == (i % max(1, self._n_samples_for_dump()))
                    full = i < self._dump_full_n or analyzed
                    rec = {
                        "i": i,
                        "task_id": self._task_id_of(ee),
                        "reward_sum": rew_sum,
                        "stop_reason": stop_reasons[i] if i < len(stop_reasons) else None,
                        "env_metrics": env_metrics[i] if i < len(env_metrics) else None,
                        "n_prompt_tokens": len(prompts[i]),
                        "n_response_tokens": len(responses[i]),
                        "hint": (hints[i] or "")[: self._dump_hint_max_chars] if i < len(hints) else "",
                        # Provenance: {"source": online|fallback|none, "key", "pi_type", "outcome", "rep",
                        # "n_traj"}. `rep` is WHICH sibling rollout the analyzer read, so a hint can be traced
                        # to the trajectory it came from. None on the offline arms (static hints file).
                        "hint_meta": hmeta,
                        "is_analyzed_rollout": analyzed,
                        "prompt_text": self.tokenizer.decode(prompts[i], skip_special_tokens=False) if full else None,
                        "response_text": resp_text if full else None,
                        "response_tail": None if full else resp_text[-800:],
                    }
                    recs.append(rec)
                    f.write(json.dumps(rec) + "\n")
            # Stash the full-text recs (first OPSD_DUMP_FULL_N) for the trainer's W&B trajectory sync.
            self._last_dump_recs = [r for r in recs if r.get("response_text")]
            logger.info(f"OPSD: dumped {len(responses)} decoded rollouts -> {path}")
        except Exception as e:  # noqa: BLE001 — dumping must never kill training
            logger.warning(f"OPSD rollout dump failed (non-fatal): {e}")

    def _rollouts_to_html(self, title, recs, show_hint: bool) -> str:
        """Render a few sampled rollouts as an HTML doc for wandb.Html (collapsible prompt/response).

        Instance method (not static) so the hint block honours OPSD_DUMP_HINT_MAX_CHARS — a hardcoded 4000
        hid ~95% of a full-trajectory hint (online trajectory-hint runs set OPSD_ONLINE_HINTS_MAX_CHARS=72000),
        which is exactly the thing you open this panel to inspect.
        """
        import html as _html

        parts = [f"<h3>{_html.escape(title)}</h3>"]
        for j, r in enumerate(recs):
            rew = r["reward"]
            rew_s = f"{rew:.3f}" if isinstance(rew, float) and rew == rew else str(rew)  # nan-safe
            parts.append(
                f"<hr><b>#{j} &middot; task_id={_html.escape(str(r['task_id']))} &middot; "
                f"reward={rew_s} &middot; stop={_html.escape(str(r['stop_reason']))}</b>"
            )
            if show_hint and r["hint"]:
                parts.append(
                    "<details open><summary><b>hint</b></summary>"
                    f"<pre style='white-space:pre-wrap'>{_html.escape(r['hint'][:self._dump_hint_max_chars])}</pre></details>"
                )
            _p = r["prompt"][-self._html_prompt_chars :] if self._html_prompt_chars > 0 else r["prompt"]
            _r = r["response"][: self._html_resp_chars] if self._html_resp_chars > 0 else r["response"]
            parts.append(
                f"<details><summary><b>prompt</b> ({len(r['prompt'])} chars)</summary>"
                f"<pre style='white-space:pre-wrap'>{_html.escape(_p)}</pre></details>"
            )
            parts.append(
                f"<details open><summary><b>response</b> ({len(r['response'])} chars)</summary>"
                f"<pre style='white-space:pre-wrap'>{_html.escape(_r)}</pre></details>"
            )
        return "<div style='font-family:monospace;font-size:12px'>" + "".join(parts) + "</div>"

    def _log_rollouts_wandb(
        self, generator_output, env_extras, ref_prompt_token_ids, hints, batch_metadata, k: int = 0
    ) -> None:
        """Log a few sampled rollouts (+ rewards) to wandb as HTML: train / train_with_hint / eval.

        Keyed by ``batch_metadata.training_phase`` (train vs eval) and logged at ``global_step``.
        ``k`` defaults to ``OPSD_WANDB_TRAJ_N`` (0 disables). Non-fatal: never kill a rollout batch.
        """
        try:
            import wandb

            k = k or self._wandb_traj_n
            if k <= 0 or wandb.run is None:
                return
            phase = str(getattr(batch_metadata, "training_phase", "") or "").lower()
            step = int(getattr(batch_metadata, "global_step", 0) or 0)
            is_eval = "eval" in phase
            responses = generator_output["response_ids"]
            prompts = generator_output["prompt_token_ids"]
            rewards = generator_output.get("rewards") or []
            stop_reasons = generator_output.get("stop_reasons") or []
            if not responses:
                return

            def _rew(i):
                r = rewards[i] if i < len(rewards) else None
                return float(sum(r)) if isinstance(r, (list, tuple)) else (float(r) if r is not None else float("nan"))

            def _rec(i, prompt_ids):
                ee = env_extras[i] if i < len(env_extras) else {}
                return {
                    "task_id": self._task_id_of(ee),
                    "reward": _rew(i),
                    "stop_reason": stop_reasons[i] if i < len(stop_reasons) else None,
                    "hint": (hints[i] if i < len(hints) else "") or "",
                    "prompt": self.tokenizer.decode(prompt_ids, skip_special_tokens=False),
                    "response": self.tokenizer.decode(responses[i], skip_special_tokens=False),
                }

            n = min(k, len(responses))
            if is_eval:
                recs = [_rec(i, prompts[i]) for i in range(n)]
                wandb.log(
                    {"rollouts/eval": wandb.Html(self._rollouts_to_html("Eval rollouts", recs, show_hint=False))},
                    step=step,
                    commit=False,
                )
            else:
                recs = [_rec(i, prompts[i]) for i in range(n)]
                wandb.log(
                    {
                        "rollouts/train": wandb.Html(
                            self._rollouts_to_html("Train rollouts (no-hint prompt)", recs, show_hint=False)
                        )
                    },
                    step=step,
                    commit=False,
                )
                # with-hint view: decode the ref (hint-prepended) prompt, only for samples that got a hint.
                hinted = [i for i in range(len(responses)) if i < len(hints) and hints[i]][:k]
                if hinted:
                    hrecs = [_rec(i, ref_prompt_token_ids[i]) for i in hinted]
                    wandb.log(
                        {
                            "rollouts/train_with_hint": wandb.Html(
                                self._rollouts_to_html("Train rollouts (WITH-hint prompt)", hrecs, show_hint=True)
                            )
                        },
                        step=step,
                        commit=False,
                    )
        except Exception as e:  # noqa: BLE001 — logging must never kill training
            logger.warning(f"OPSD wandb rollout HTML log failed (non-fatal): {e}")


def _online_pi_needs_analyzer(pi_type: str, constructor) -> bool:
    """True if the online hint constructor calls the analyzer LLM.

    Only ``static`` and ``reference`` in a mode other than ``abstracted`` (``actions`` = the literal action
    list, ``trajectory`` = the full transcript) build hints without one. The constructor's resolved mode
    is read rather than the config string, so this follows whatever ReferenceHintConstructor picked.
    """
    pt = (pi_type or "").strip().lower()
    if pt == "static":
        return False
    if pt == "reference":
        return str(getattr(constructor, "mode", "") or "").lower() == "abstracted"
    return True


def _require_online_analyzer_model(mgr) -> None:
    """Fail at generator construction, not after the first rollout batch, when the online pi_type needs an
    analyzer and none is configured. ``make_analyzer`` already raises for the policy/endpoint backends;
    the frontier backend defers the check to the first call, which is a full rollout batch too late."""
    from opsd.pi.analyzer import AnalyzerConfigError

    if _online_pi_needs_analyzer(mgr.cfg.pi_type, mgr._constructor) and not getattr(mgr._analyzer, "model", None):
        raise AnalyzerConfigError(
            f"OPSD online hints: OPSD_ONLINE_HINTS_PI_TYPE={mgr.cfg.pi_type!r} calls an analyzer, but no "
            "analyzer model is configured. Set OPSD_ONLINE_HINTS_ANALYZER_MODEL (or OPSD_PI_ANALYZER_MODEL), "
            "or use OPSD_ONLINE_HINTS_PI_TYPE=reference with OPSD_ONLINE_HINTS_REFERENCE_MODE=actions|trajectory."
        )


class SelfDistillOnlineSkillGenerator(SelfDistillSkyRLGymGenerator):
    """OPSD teacher hint = an ON-POLICY SKILL rebuilt each step from the student's OWN fresh
    rollouts by an analyzer LLM (e.g. Qwen3.7), via ``pi/online.OnlineHintManager``.

    Per generate(): the student rolls out no-hint (base) → reconstruct TrajViews from the batch
    tokens → the analyzer writes a per-task skill (golden workflow) → that skill becomes the TEACHER
    ref-prompt hint for the SAME batch's distillation. The student never sees the hint, so there is
    no prompt recurrence. Configured via ``OPSD_ONLINE_HINTS_*`` env (see pi/online.py): analyzer
    backend/model/url, ``pi_type`` (onpolicy_skill), ``key_mode`` (task_id).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from opsd.pi.online import OnlineHintManager, online_hint_cfg_from_env

        self._online_mgr = OnlineHintManager(online_hint_cfg_from_env(), self.tokenizer)
        _require_online_analyzer_model(self._online_mgr)
        self._last_online_metrics: dict = {}
        self._n_samples = int(getattr(self.generator_cfg, "n_samples_per_prompt", 1) or 1)

    async def _resolve_hints(self, generator_output, env_extras, input_batch) -> list:
        n = len(generator_output["prompt_token_ids"])
        bm = input_batch.get("batch_metadata")
        gstep = int(getattr(bm, "global_step", 0) or 0)
        uids = list(input_batch.get("uids") or [None] * n)
        rep_ids = [i % self._n_samples for i in range(n)]
        rewards = generator_output.get("rewards") or [0.0] * n
        # EVAL/TEST generates must NEVER touch the analyzer. batch_metadata.training_phase is "train"|"eval"
        # (BatchMetadata, built by SkyRL's generators/utils.py prepare_generator_input). Both lines of this
        # trainer found the same leak independently:
        #   * ALFWorld: every dev eval ran the FULL online path -- 200 analyzer calls per eval x 5 evals =
        #     1000 wasted analyzer calls, a 200-call burst at step 0 that is prime rate-limit bait, and a
        #     store polluted with hints keyed to DEV task_ids that training can never use (hint_calls=200 at
        #     step 0 is exactly this).
        #   * AppWorld: store_size reached 755 on the online-skill runs, above the 720 ceiling that 90 train
        #     tasks x 8 reps allows; at eval n=4, 5 dev evals of 57x4 plus a final test of 307x4 over two checkpoints
        #     is ~3.6k extra analyzer calls against ~14.4k real ones, each up to 16384 tokens.
        # Hints are useless at eval anyway: they only feed the teacher/ref forward, which eval never runs.
        # cfg.inject_eval (OPSD_ONLINE_HINTS_INJECT_EVAL, default False) is the opt-in: serve the hints
        # ALREADY in the store (for the dumps / W&B panels), but still never analyze an eval rollout.
        phase = str((getattr(bm, "training_phase", "train") if bm is not None else "train") or "train").lower()
        if phase != "train":
            self._last_online_metrics = {}
            if not self._online_mgr.cfg.inject_eval:
                self._hint_meta = [{"source": "none", "phase": phase} for _ in range(n)]
                return [""] * n
            self._hint_meta = [
                self._online_mgr.meta_for(
                    env_extras[i] if i < len(env_extras) else {}, uids[i] if i < len(uids) else None, rep_ids[i]
                )
                for i in range(n)
            ]
            return [
                self._online_mgr.hint_for(
                    env_extras[i] if i < len(env_extras) else {}, uids[i] if i < len(uids) else None, rep_ids[i]
                )
                for i in range(n)
            ]
        from opsd.pi.analyzer import AnalyzerConfigError

        try:
            self._last_online_metrics = (
                await self._online_mgr.update(
                    generator_output["prompt_token_ids"],
                    generator_output["response_ids"],
                    generator_output["loss_masks"],
                    rewards,
                    env_extras,
                    uids,
                    rep_ids,
                    gstep,
                )
                or {}
            )
        except AnalyzerConfigError:
            # A misconfigured analyzer (no model, missing base URL, ...) fails every call the same way, so
            # swallowing it would train the whole run on fallback/no hints. Re-raised whatever STRICT says.
            raise
        except Exception as exc:  # never kill a training step on an analyzer hiccup...
            # ...UNLESS strict is on. _refine_one already re-raises under OPSD_ONLINE_HINTS_STRICT=1, but
            # this handler swallowed it again, so a totally dead analyzer still exited 0 having produced no
            # hints -- three separate smokes "passed" that way. Strict must reach the job's exit status.
            if os.environ.get("OPSD_ONLINE_HINTS_STRICT", "0").strip().lower() in ("1", "true", "yes"):
                raise
            logger.warning(f"OPSD online-skill: update failed (keeping prior hints): {exc}")
            self._last_online_metrics = {}
        # Per-sample hint provenance for the rollout dump (which sibling rollout the analyzer read, whether
        # the static fallback was used). Kept parallel to `hints` so _dump_rollouts can zip them by index.
        self._hint_meta = [
            self._online_mgr.meta_for(
                env_extras[i] if i < len(env_extras) else {}, uids[i] if i < len(uids) else None, rep_ids[i]
            )
            for i in range(n)
        ]
        return [
            self._online_mgr.hint_for(
                env_extras[i] if i < len(env_extras) else {}, uids[i] if i < len(uids) else None, rep_ids[i]
            )
            for i in range(n)
        ]


class SelfDistillExp(OPSDExpMixin, BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        online = os.environ.get("OPSD_ONLINE_SKILL", "0").strip().lower() in ("1", "true", "yes")
        gen_cls = SelfDistillOnlineSkillGenerator if online else SelfDistillSkyRLGymGenerator
        logger.info(f"OPSD generator = {gen_cls.__name__} (OPSD_ONLINE_SKILL={online})")
        return gen_cls(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            policy_model_name=resolve_policy_model_name(cfg),
        )

    def get_trainer(
        self, cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator, colocate_pg
    ):
        # Parse the teacher knobs here so a typo dies at startup rather than after the first
        # (multi-minute) rollout batch. `require_ref_model` still has to wait for the trainer, whose
        # ref_model is only resolved once the actor groups exist.
        _mode, _alpha = resolve_teacher_mode(), resolve_teacher_alpha()
        logger.info(
            f"OPSD teacher = {_mode}"
            + (
                f" (alpha={_alpha}, half-life~{ema_half_life(_alpha):.0f} steps)"
                if _mode in ("ema", "trust-region")
                else ""
            )
        )
        return SelfDistillationTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: OPSDTrainConfig):
    SelfDistillExp(cfg).run()


def driver_node_strategy() -> NodeAffinitySchedulingStrategy:
    """Pin the entrypoint task to the node this driver runs on (hard affinity).

    The generator (and so every env client) lives inside the entrypoint task. The launcher starts the
    ALFWorld env servers on its own node and passes their URLs (loopback by default), and the per-run
    inputs it stages (OPSD_HINTS_PATH, task-id lists, rollout dump dirs) are local paths on that node. On a
    multi-node Ray cluster an unpinned task may land on a worker node where none of that resolves -- a
    missing hints file there silently degrades to the static fallback hint. On a single node this is a
    no-op (there is only one node). soft=False: if the driver node cannot host the task, Ray keeps it
    pending rather than placing it elsewhere.
    """
    return NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False)


def main() -> None:
    cfg = CONFIG_CLS.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.options(scheduling_strategy=driver_node_strategy()).remote(cfg))


if __name__ == "__main__":
    main()

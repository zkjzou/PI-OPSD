# Adapted from SkyRL (Apache-2.0), skyrl/backends/skyrl_train/workers/worker.py @ ecdaa946; modified for OPSD: the RL
# branch of PolicyWorkerBase._forward_backward_micro is copied into OPSDFSDPPolicyWorkerBase and adds the beta
# teacher-KL and SDAR loss terms (opsd.algorithms.distill_loss); forward_backward feeds it the teacher log-probs.
"""FSDP policy worker with OPSD's loss-side teacher terms (beta teacher-KL and SDAR).

Swapped in for SkyRL's ``fsdp_worker.PolicyWorker`` by ``OPSDWorkerMixin.build_models``
(opsd.skyrl_ext.worker_selection) only when OPSD_DISTILL_LOSS_COEF or OPSD_SDAR_LAMBDA is non-zero; every other
run uses the upstream actor classes unchanged.

Teacher transport: the OPSD trainer ships ``training_input["distill_log_probs"]`` (B, A). Upstream's
``BatchIterator`` drops keys it does not know, so ``forward_backward`` splits the tensor exactly the way
``TensorBatch.chunk`` splits the batch and ``_forward_backward_micro`` takes one chunk per call, BEFORE any
fall-through, so the chunks stay aligned with the micro-batches whatever path a micro-batch takes.

Every policy actor process imports this module when it unpickles the actor class (by reference), so it imports
only torch, the SkyRL FSDP worker stack (which the actor loads anyway) and opsd.algorithms.distill_loss.
"""

from typing import Any, Dict, Iterator, Optional

import ray
import torch
from omegaconf import OmegaConf

from opsd.algorithms.distill_loss import LossKnobs, opsd_loss_terms
from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    PolicyLossRegistry,
    compute_approx_kl,
)
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase
from skyrl.train.dataset.replay_buffer import Experience

# The batch key the OPSD trainers fill with the teacher's log-probs of the sampled tokens.
DISTILL_LOG_PROBS_KEY = "distill_log_probs"


class OPSDFSDPPolicyWorkerBase(FSDPPolicyWorkerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Pushed by OPSDWorkerMixin.build_models right after the actors are built. None until then, and
        # _forward_backward_micro refuses to run without it: this class is only ever swapped in because a knob
        # is set, so a missing push is a wiring bug that would otherwise train as plain GRPO.
        self._opsd_loss_knobs: Optional[LossKnobs] = None
        self._opsd_teacher_chunks: Optional[Iterator[torch.Tensor]] = None

    def set_opsd_loss_knobs(self, knobs: LossKnobs) -> None:
        """Receive the driver-resolved loss knobs (called through the ``pass_through`` dispatch)."""
        if not isinstance(knobs, LossKnobs):
            raise TypeError(f"expected opsd.algorithms.distill_loss.LossKnobs, got {type(knobs).__name__}")
        self._opsd_loss_knobs = knobs

    def forward_backward(
        self,
        data: TrainingInputBatch,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
    ) -> WorkerOutput:
        teacher = data.get(DISTILL_LOG_PROBS_KEY)
        if teacher is None:
            self._opsd_teacher_chunks = None
        else:
            assert (
                teacher.shape[0] == data.batch_size
            ), f"{DISTILL_LOG_PROBS_KEY} has {teacher.shape[0]} rows for a batch of {data.batch_size}"
            # Same slices, in the same order, as TensorBatch.chunk (training_batch.py), which is what
            # upstream's BatchIterator walks; PolicyWorkerBase.forward_backward calls _forward_backward_micro
            # exactly once per chunk, in order.
            chunk = self.cfg.micro_train_batch_size_per_gpu
            self._opsd_teacher_chunks = iter([teacher[i : i + chunk] for i in range(0, data.batch_size, chunk)])
        try:
            output = super().forward_backward(data, loss_fn=loss_fn, loss_fn_config=loss_fn_config)
            if self._opsd_teacher_chunks is not None:
                assert (
                    next(self._opsd_teacher_chunks, None) is None
                ), f"{DISTILL_LOG_PROBS_KEY} chunks left over after the last micro-batch: teacher/student misaligned"
            return output
        finally:
            self._opsd_teacher_chunks = None

    def _next_teacher_chunk(self, experience: Experience) -> Optional[torch.Tensor]:
        if self._opsd_teacher_chunks is None:
            return None
        teacher = next(self._opsd_teacher_chunks, None)
        if teacher is None:
            raise RuntimeError(f"ran out of {DISTILL_LOG_PROBS_KEY} chunks before the micro-batches")
        assert teacher.shape[0] == experience.sequences.shape[0], (
            f"{DISTILL_LOG_PROBS_KEY} chunk has {teacher.shape[0]} rows, micro-batch has "
            f"{experience.sequences.shape[0]}"
        )
        assert teacher.shape[-1] == experience.num_actions, (
            f"{DISTILL_LOG_PROBS_KEY} chunk is {teacher.shape[-1]} tokens wide, micro-batch has "
            f"{experience.num_actions} actions"
        )
        return teacher

    def _forward_backward_micro(
        self,
        experience: Experience,
        microbatch_weight: float,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        # Consume the teacher chunk first, on every path, so the side iterator never drifts.
        teacher = self._next_teacher_chunk(experience)
        knobs = self._opsd_loss_knobs
        if knobs is None:
            raise RuntimeError(
                "OPSDFSDPPolicyWorkerBase: set_opsd_loss_knobs was never called; "
                "OPSDWorkerMixin.build_models pushes the knobs right after building the policy actors"
            )
        resolved_loss_name = loss_fn if loss_fn is not None else self.cfg.algorithm.policy_loss_type
        if not knobs.active or resolved_loss_name == "cross_entropy":
            # Knobs off: the fork added exact zeros here, so upstream's method is bit-identical. The SFT branch
            # never carried the OPSD terms.
            return super()._forward_backward_micro(
                experience, microbatch_weight, loss_fn=loss_fn, loss_fn_config=loss_fn_config
            )
        return self._opsd_forward_backward_micro(
            experience, microbatch_weight, teacher, knobs, loss_fn=loss_fn, loss_fn_config=loss_fn_config
        )

    def _opsd_forward_backward_micro(
        self,
        experience: Experience,
        microbatch_weight: float,
        teacher_log_probs: Optional[torch.Tensor],
        knobs: LossKnobs,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Upstream ``PolicyWorkerBase._forward_backward_micro`` (RL branch) plus the OPSD loss terms.

        COPIED from SkyRL @ ecdaa946 (the SFT branch is dropped: cross_entropy falls through to upstream).
        Everything outside the ``# >>> OPSD`` / ``# <<< OPSD`` blocks is upstream's text verbatim;
        opsd/skyrl_ext/tests/test_policy_worker_drift.py pins upstream's source hash and checks this copy against
        it, so a SkyRL bump that touches the method fails loudly instead of silently diverging.
        """
        self.model.train()

        experience.to_device(torch.cuda.current_device())

        sequences = experience.sequences
        old_action_log_probs = experience.action_log_probs
        base_action_log_probs = (
            experience.base_action_log_probs if experience.base_action_log_probs is not None else None
        )
        advantages = experience.advantages
        num_actions = experience.num_actions
        attention_mask = experience.attention_mask
        loss_mask = experience.loss_mask
        action_mask = experience.action_mask
        rollout_action_logprobs = experience.rollout_logprobs

        # Determine which loss function to use
        resolved_loss_name = loss_fn if loss_fn is not None else self.cfg.algorithm.policy_loss_type
        if loss_fn is not None:
            # Use the provided loss function (Tinker API style)
            current_loss_fn = PolicyLossRegistry.get(loss_fn)
        else:
            # Fall back to config default
            current_loss_fn = self.policy_loss_fn

        # Build config for loss function, applying any overrides
        loss_config = self.cfg.algorithm
        if loss_fn_config is not None:
            # Create a copy of the config and apply overrides
            # TODO: Fix nested overrides
            from dataclasses import asdict

            new_loss_config = OmegaConf.merge(OmegaConf.create(asdict(loss_config)), OmegaConf.create(loss_fn_config))
            # NOTE: users can provide a custom loss config class, so we need to use the same class after applying overrides
            loss_config = type(loss_config).from_dict_config(new_loss_config)

        # TODO (sumanthrh): don't think this does anything for fsdp rn because autocast happens internally
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            # actor loss
            action_log_probs, output = self.model(
                sequences,
                num_actions,
                attention_mask=attention_mask,
                temperature=self.cfg.algorithm.temperature,
                return_output=True,
                compute_entropy=True,
                entropy_requires_grad=self.cfg.algorithm.use_entropy_loss,
                pixel_values=experience.pixel_values,
                image_grid_thw=experience.image_grid_thw,
            )
            # loss function
            # TODO: recompute advantages
            policy_loss, loss_metrics = current_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantages,
                config=loss_config,
                loss_mask=loss_mask,
                rollout_logprobs=rollout_action_logprobs,
            )

        # >>> OPSD: the SFT branch is upstream's (the caller falls through for cross_entropy)
        assert resolved_loss_name != "cross_entropy"
        # <<< OPSD

        # RL path: add optional KL/entropy terms
        # entropy loss
        with torch.set_grad_enabled(self.cfg.algorithm.use_entropy_loss):
            # batch_size, seqlen
            entropy_BS = output["entropy"]
            entropy_BS = entropy_BS[:, -num_actions - 1 : -1]
            entropy = masked_mean(entropy_BS, loss_mask)

        if self.cfg.algorithm.use_entropy_loss:
            entropy_loss_term = entropy * self.cfg.algorithm.entropy_loss_coef
        else:
            entropy_loss_term = torch.tensor(0.0)

        # kl loss
        if self.cfg.algorithm.use_kl_loss:
            kl_loss = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                loss_mask=loss_mask,
                kl_estimator_type=self.cfg.algorithm.kl_estimator_type,
            )
            kl_loss = masked_mean(kl_loss, loss_mask, dim=-1).mean()
        else:
            kl_loss = torch.tensor(0.0)
        kl_loss_term = kl_loss * self.cfg.algorithm.kl_loss_coef

        # >>> OPSD: beta teacher-KL + SDAR, a SEPARATE loss group from the theta0 leash above (independent of
        # use_kl_loss); maths in opsd.algorithms.distill_loss
        distill_loss_term, sdar_loss_term, opsd_status = opsd_loss_terms(
            knobs,
            action_log_probs,
            teacher_log_probs,
            loss_mask,
            entropy_BS,
            self.cfg.algorithm.kl_estimator_type,
        )
        # <<< OPSD

        # DP all-reduce averages gradients, but policy losses are pre-scaled sums
        # (see `apply_loss_reduction_to_advantages_minibatch`), so we multiply by
        # dp_size to recover the correct sum reduction across workers.
        grad_sum_correction_factor = self.mesh_rank.dp_size

        # NOTE: The KL and entropy loss terms are not pre-scaled,
        # so we just average them across microbatches and DP workers.
        # >>> OPSD: replaces upstream's
        #   loss = policy_loss * grad_sum_correction_factor + (kl_loss_term - entropy_loss_term) * microbatch_weight
        # (the teacher terms are per-sequence token means like the KL term, so they join its group)
        loss = (
            policy_loss * grad_sum_correction_factor
            + (kl_loss_term + distill_loss_term + sdar_loss_term - entropy_loss_term) * microbatch_weight
        )
        # <<< OPSD
        unscaled_loss = loss / grad_sum_correction_factor
        self.strategy.backward(loss, self.model, self.optimizer)

        # Build per-sequence loss_fn_outputs with logprobs.
        batch_size = action_log_probs.shape[0]
        seq_len = action_log_probs.shape[1]

        if action_mask is not None:
            valid_lens = action_mask.sum(dim=1).int().tolist()
        elif loss_mask is not None:
            valid_lens = loss_mask.sum(dim=1).int().tolist()
        else:
            valid_lens = [seq_len] * batch_size

        detached_log_probs = action_log_probs.detach().cpu()
        loss_fn_outputs = []
        for i, valid_len in enumerate(valid_lens):
            loss_fn_outputs.append(
                {
                    "logprobs": detached_log_probs[i, -valid_len:].tolist() if valid_len > 0 else [],
                }
            )

        status = {
            "final_loss": unscaled_loss.item(),
            "policy_loss": policy_loss.item(),
            "policy_entropy": entropy.item(),
            "response_length": num_actions,
            "policy_lr": self.scheduler.get_last_lr()[0],
            "loss_fn_outputs": loss_fn_outputs,
        }
        for k, v in loss_metrics.items():
            status["loss_metrics/" + k] = v
        if self.cfg.algorithm.use_kl_loss:
            status["policy_kl"] = kl_loss.item()

        # >>> OPSD: distill_loss_* (beta) and sdar_* keys, only for a REQUESTED term (see opsd_loss_terms)
        status.update(opsd_status)
        # <<< OPSD

        return status


# Module-level, like fsdp_worker.PolicyWorker: Ray pickles the actor class by reference.
OPSDFSDPPolicyWorker = ray.remote(num_gpus=1)(OPSDFSDPPolicyWorkerBase)

# Adapted from SkyRL (Apache-2.0), skyrl/train/trainer.py @ ecdaa946; modified for OPSD: the grpo/* gradient-signal
# metrics re-derive RayPPOTrainer.compute_advantages_and_returns's unpadded / last-step row selection, and
# load_checkpoints drains a resumed epoch whose restored iterator is already exhausted.
"""``OPSDRayPPOTrainer``: the one ``RayPPOTrainer`` subclass every OPSD trainer derives from.

SkyRL's ``RayPPOTrainer.train()`` runs unmodified. Everything OPSD adds at the trainer level is an override that
calls ``super()`` first:

* ``compute_advantages_and_returns`` adds the ``grpo/*`` gradient-signal breakdown
  (``opsd.algorithms.grpo_signal``) for the batch the advantage estimator just consumed.
* ``load_checkpoints`` fixes the epoch-boundary resume: a checkpoint written right after an epoch's last batch
  restores an iterator with nothing left, so upstream ``train()`` would spend its first resumed epoch on 0
  batches and finish one epoch short.
* ``eval`` (``OPSDEvalMixin``) and ``build_models`` (``OPSDWorkerMixin``) come from their own modules.
"""

from typing import Any, List, Optional, Tuple

import torch
from loguru import logger

from opsd.algorithms.grpo_signal import gradient_signal_metrics
from opsd.skyrl_ext.evaluate import OPSDEvalMixin
from opsd.skyrl_ext.worker_selection import OPSDWorkerMixin
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.trainer import RayPPOTrainer


def gradient_signal_inputs(data: TrainingInputBatch, step_wise: bool) -> Tuple[List[float], List[Any], List[float]]:
    """(per-sequence reward sums, group keys, trainable-token counts) for ``gradient_signal_metrics``.

    The rows ``RayPPOTrainer.compute_advantages_and_returns`` scored: padding rows dropped, and on the step-wise
    path only each trajectory's last step, because groups are formed per trajectory there (counting every step
    would inflate the group sizes). Reads ``rewards`` / ``uids`` / ``is_last_step``, which ``train()`` pops only
    after ``compute_advantages_and_returns`` returns.
    """
    n = len(data["rewards"]) - data.metadata.get("pad_size", 0)
    scores = data["rewards"].sum(dim=-1)[:n]
    uids = list(data.metadata["uids"])[:n]
    tokens = data["response_mask"][:n].sum(dim=-1)
    if step_wise:
        keep = torch.tensor(data.metadata["is_last_step"], dtype=torch.bool)[:n]
        scores, tokens = scores[keep], tokens[keep]
        uids = [u for u, k in zip(uids, keep.tolist()) if k]
    return scores.tolist(), uids, tokens.tolist()


def batches_consumed(state: dict) -> int:
    """Batches a saved ``StatefulDataLoader`` iterator had yielded (torchdata 0.11 state-dict keys).

    Single-process iterators count them in ``_num_yielded``; multi-process ones in their last snapshot step plus
    the steps since. Raises ``KeyError`` on an unknown layout.
    """
    if "_num_yielded" in state:
        return int(state["_num_yielded"])
    return int(state["_snapshot"]["_snapshot_step"]) + int(state["_steps_since_snapshot"])


class OPSDRayPPOTrainer(OPSDEvalMixin, OPSDWorkerMixin, RayPPOTrainer):
    @torch.no_grad()
    def compute_advantages_and_returns(self, data: TrainingInputBatch) -> TrainingInputBatch:
        data = super().compute_advantages_and_returns(data)
        # How much of this batch carried a usable gradient, on exactly the (score, group-key) pairs the estimator
        # consumed. Returns {} for non-group estimators (every group has one member). Instrumentation must never
        # fail a step.
        try:
            inputs = gradient_signal_inputs(data, bool(self.cfg.generator.step_wise_trajectories))
            self.all_metrics.update(gradient_signal_metrics(*inputs))
        except Exception:  # noqa: BLE001
            logger.exception("gradient_signal_metrics failed; continuing without it")
        return data

    def load_checkpoints(self) -> Tuple[int, Optional[str]]:
        global_step, checkpoint_path = super().load_checkpoints()
        if checkpoint_path is not None:
            self._drain_exhausted_resume_epoch(global_step)
        return global_step, checkpoint_path

    def _drain_exhausted_resume_epoch(self, global_step: int) -> None:
        """Spend the empty pass of an exhausted restored iterator now, so ``train()``'s first epoch is a full one.

        A checkpoint saved after an epoch's last batch but before ``StopIteration`` (every in-loop epoch-end
        save, and the fork's ``global_step_{T+1}`` safety net) restores an iterator that yields nothing:
        ``train()`` would then log 0 batches for epoch ``global_step // len`` and end one epoch short. Iterating
        it once here runs the same iterator creations and generator draws as that empty pass, so the data order
        is unchanged; the next ``iter()`` reshuffles a fresh epoch.

        Skipped when no state was restored (``data.pt`` missing or unreadable), when the saved iterator already
        hit ``StopIteration`` (``StatefulDataLoader`` starts a fresh epoch by itself; draining would eat a
        batch), when it is mid-epoch, and when ``train()`` has no epoch left to run (``trainer.epochs=0`` eval
        runs).
        """
        dataloader = self.train_dataloader
        state = getattr(dataloader, "next_iter_state", None)
        if not state or state.get("_iterator_finished", False):
            return
        try:
            consumed = batches_consumed(state)
        except (KeyError, TypeError) as e:
            logger.warning(f"resume: unrecognised dataloader state ({e!r}); not draining the resumed epoch")
            return
        steps_per_epoch = len(dataloader)
        if consumed < steps_per_epoch or global_step // steps_per_epoch >= self.cfg.trainer.epochs:
            return
        if next(iter(dataloader), None) is not None:
            # Not actually exhausted (e.g. the training pool changed size): put the restored state back.
            logger.warning(
                f"resume: dataloader state at global_step {global_step} reports {consumed}/{steps_per_epoch} batches "
                "consumed but still yields; restoring it unchanged"
            )
            dataloader.load_state_dict(state)
            return
        logger.info(
            f"resume: the restored epoch at global_step {global_step} was exhausted ({consumed}/{steps_per_epoch} "
            f"batches); drained it so epoch {global_step // steps_per_epoch} starts fresh"
        )

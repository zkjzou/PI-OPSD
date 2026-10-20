"""Drift guard for the upstream code opsd.skyrl_ext.trainer, opsd.skyrl_ext.tracking and the OPSD trainer adapt.

``tracking.get_tracker`` is a copy of ``BasePPOExp.get_tracker`` building ``OPSDTracking``.
``trainer.gradient_signal_inputs`` re-derives the rows ``RayPPOTrainer.compute_advantages_and_returns`` scores, and
``OPSDRayPPOTrainer.load_checkpoints`` calls upstream's and then drains an exhausted restored epoch using
``train()``'s own epoch arithmetic (``start_epoch = global_step // len(train_dataloader)``).
``SelfDistillationTrainer`` (opsd/trainer/main_opsd_alfworld.py; the AppWorld and tau2 OPSD trainers inherit
it) replaces ``RayPPOTrainer.fwd_logprobs_values_reward`` (an adapted copy with the extra teacher forward) and
``apply_reward_kl_penalty`` without calling ``super()``. A SkyRL pin bump that touches any of these fails
here: re-check the OPSD code against the new upstream source (the header of each file lists what it adapts),
re-run the trainer / tracking fork-parity tests, then update the hash.
"""

import hashlib
import importlib
import inspect

import pytest

# "<module>:<qualname>" -> sha256 of inspect.getsource at SkyRL ecdaa946
PINNED = {
    # copied into opsd.skyrl_ext.tracking.get_tracker
    "skyrl.train.entrypoints.main_base:BasePPOExp.get_tracker": (
        "97966c0254c0f13eb6dcbdc06ebfbc09a0ce039e1d294cf935ee88ce59989b27"
    ),
    # row selection re-derived in opsd.skyrl_ext.trainer.gradient_signal_inputs; overridden via super()
    "skyrl.train.trainer:RayPPOTrainer.compute_advantages_and_returns": (
        "0204a1c19181158a65f352fdbfbec9f3a75c6c72fdc0cbe8ed7013747d8e7413"
    ),
    # overridden via super(); the drain depends on what it restores (train_dataloader.next_iter_state)
    "skyrl.train.trainer:RayPPOTrainer.load_checkpoints": (
        "91557ea2587471e8eaad8f13b886a64d63e2d0fcf137d3af288478f959fe8df0"
    ),
    # runs unmodified; the drain mirrors its start_epoch / epoch-loop arithmetic
    "skyrl.train.trainer:RayPPOTrainer.train": "34689cb5965a55d1f5195efb791ea399aa479de976d2bc2b811a1c6b17b83b5f",
    # replaced by SelfDistillationTrainer (opsd/trainer/main_opsd_alfworld.py), no super(); an upstream
    # change would be skipped silently
    "skyrl.train.trainer:RayPPOTrainer.fwd_logprobs_values_reward": (
        "aa9de644c0cc226b3575951dbade5ef7d1505fb882bcca6eb8069b63bb6f3d34"
    ),
    "skyrl.train.trainer:RayPPOTrainer.apply_reward_kl_penalty": (
        "54dbf73d74b8bade832b8babb319f5fcfe0dad6fbf59df63b049528f7d1bb1af"
    ),
}


def _resolve(path):
    module, qualname = path.split(":")
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize("path", sorted(PINNED))
def test_upstream_source_unchanged(path):
    digest = hashlib.sha256(inspect.getsource(_resolve(path)).encode()).hexdigest()
    assert digest == PINNED[path], f"{path} changed upstream (sha256 {digest}); re-check the opsd.skyrl_ext code"


def test_the_drain_still_matches_trains_epoch_arithmetic():
    """The drain's skip rule (``global_step // len >= epochs``) is train()'s ``range(start_epoch, epochs)``."""
    from skyrl.train.trainer import RayPPOTrainer

    src = inspect.getsource(RayPPOTrainer.train)
    assert "start_epoch = self.global_step // len(self.train_dataloader)" in src
    assert "for epoch in range(start_epoch, self.cfg.trainer.epochs):" in src

"""OPSD's config tree: SkyRL's ``SkyRLTrainConfig`` plus the fields the OPSD recipes set.

SkyRL's config is a tree of plain dataclasses. ``SkyRLTrainConfig.from_cli_overrides`` is a classmethod that
builds ``cls`` level by level from each field's declared type, and rejects every key a level does not declare.
So OPSD extends it the way upstream asks ("subclass the relevant config dataclass"): a subclass at every level
on the path to a new field, each parent redeclaring the child's field with the subclass as its type and
default. ``OPSDTrainConfig.from_cli_overrides(argv)`` then builds the whole subclass tree, and parses and
validates exactly like upstream otherwise. Added fields:

* ``trainer.algorithm.overlong_buffer`` (:class:`OverlongBufferConfig`): DAPO Soft Overlong Punishment. Read by
  the DAPO trainers' ``postprocess_generator_output``; SkyRL itself never reads it.
* ``environment.skyrl_gym.alfworld`` / ``.appworld``: the env configs. The stock ``SkyRLGymGenerator`` hands
  ``getattr(skyrl_gym_cfg, env_class, {})`` to ``skyrl_gym.make``, so the ids double as attribute names.

Redeclaring a field keeps its position. New fields are appended to their class, so ``overlong_buffer`` comes
after ``trainer.algorithm.max_seq_len`` (the fork had it after ``dynamic_sampling``); that only reorders
``asdict`` output. The fork's ``trainer.update_ref_every_n_steps`` is gone: no arm set it to anything but 0.

``cfg.trainer`` is pickled by reference into every policy / ref / critic Ray actor, so this module is imported
there. Keep its imports to SkyRL's config module and the env config dataclasses (no skyrl_gym registration, no
env code); register the envs with ``opsd.envs.registry.register_opsd_envs()`` where they are built. No
``from __future__ import annotations`` either: ``build_nested_dataclass`` resolves nested configs from real
class annotations, and a string one silently leaves the nested config a plain dict (every
``getattr(overlong_buffer, "len", 0)`` reader then reads 0, i.e. DAPO becomes a no-op).
"""

from dataclasses import dataclass, field

from opsd.envs.alfworld.config import AlfworldEnvConfig
from opsd.envs.appworld.config import AppWorldEnvConfig
from skyrl.train.config import (
    AlgorithmConfig,
    BaseConfig,
    EnvironmentConfig,
    SkyRLGymConfig,
    SkyRLTrainConfig,
    TrainerConfig,
)


@dataclass
class OverlongBufferConfig(BaseConfig):
    """DAPO Soft Overlong Punishment — field names and defaults per docs.skyrl.ai/docs/algorithms/dapo.

    A length-aware penalty ADDED to the reward, ramping linearly across the last ``len`` tokens of the
    response budget. DAPO pairs this WITH ``generator.apply_overlong_filtering`` rather than replacing it:
    filtering MASKS truncated trajectories out of the loss (length ignored), this PRICES length while the
    trajectory is still learnable.

    Motivating measurement (ALFWorld GRPO, Qwen3-4B-Thinking-2507): with length unpriced, response_length ran
    16,294 -> 31,341 against a 32,768 budget while entropy went 0.88 -> 8.72 and success -> 0.000.

    NB the penalty only yields gradient where trajectories WITHIN a GRPO group differ in length --
    ``r - mean(group)`` cancels a penalty every sample takes equally. Prophylactic, not curative.
    """

    len: int = 0
    """Buffer width in TOKENS; 0 disables. DAPO publishes 512. Onset = response_budget - len, so at a
    32,000-token budget len=7000 starts penalising at 25,000."""
    penalty_factor: float = 1.0
    """Penalty magnitude at/beyond the budget, expressed on DAPO's own reward scale. DAPO publishes 1.0.
    Scaled to the env by ``reward_range`` below, so this keeps its published meaning across envs."""
    reward_range: float = 2.0
    """Span of the env's reward, used to keep ``penalty_factor`` PROPORTIONAL rather than absolute.

    The saturating penalty is ``penalty_factor * reward_range / 2.0``. The default 2.0 is DAPO's own
    span (a rule-based correctness reward of +/-1), so the default reproduces the paper exactly and is a
    no-op for anyone already on a +/-1 env.

    WHY THIS EXISTS. ``penalty_factor`` alone is in RAW reward units, so copying DAPO's published 1.0
    onto an env with a different scale silently changes how hard it bites: on DAPO's +/-1 it is half the
    reward range (enough to sink a correct answer to neutral), but on ALFWorld's ``10.0 * won`` it is
    only 10% of a success -- ~5x weaker in proportional terms. Set this to the env's span (ALFWorld:
    10.0) and the published factor lands with the pressure the paper intended.
    """


@dataclass
class OPSDAlgorithmConfig(AlgorithmConfig):
    overlong_buffer: OverlongBufferConfig = field(default_factory=OverlongBufferConfig)


@dataclass
class OPSDTrainerConfig(TrainerConfig):
    algorithm: OPSDAlgorithmConfig = field(default_factory=OPSDAlgorithmConfig)


@dataclass
class OPSDSkyRLGymConfig(SkyRLGymConfig):
    alfworld: AlfworldEnvConfig = field(default_factory=AlfworldEnvConfig)
    appworld: AppWorldEnvConfig = field(default_factory=AppWorldEnvConfig)


@dataclass
class OPSDEnvironmentConfig(EnvironmentConfig):
    skyrl_gym: OPSDSkyRLGymConfig = field(default_factory=OPSDSkyRLGymConfig)


@dataclass
class OPSDTrainConfig(SkyRLTrainConfig):
    """``SkyRLTrainConfig`` with OPSD's fields. Parse with ``OPSDTrainConfig.from_cli_overrides(argv)``."""

    trainer: OPSDTrainerConfig = field(default_factory=OPSDTrainerConfig)
    environment: OPSDEnvironmentConfig = field(default_factory=OPSDEnvironmentConfig)

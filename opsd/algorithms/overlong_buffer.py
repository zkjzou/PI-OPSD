"""DAPO Soft Overlong Punishment -- the reward transform, as a pure function.

Deliberately free of ray / vllm / skyrl imports so the arithmetic can be unit-tested in milliseconds.
The trainer-side wiring lives in ``opsd/alfworld/main_alfworld_dapo.py``.

Shape of the penalty, all lengths in TOKENS::

    onset = budget - buffer_len
    len <= onset            ->  untouched
    onset < len <= budget   ->  reward -= penalty_factor * (len - onset) / buffer_len     (linear ramp)
    len  > budget           ->  reward -= penalty_factor                                  (saturates)

The ramp rather than a cliff is the point: it gives a gradient toward shorter trajectories instead of a
step function that is flat on both sides of the threshold.

FAITHFUL TO THE PAPER AND TO verl, NOT to SkyRL's bundled example. DAPO (arXiv:2503.14476 eq. 13) is a
three-case function whose third case is a FIXED -1, *added* to the correctness reward; verl's reference
implementation (the DAPO authors' own framework) is::

    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
    reward += overlong_reward

-- additive, with `min(..., 0)` clamping only the upper side so the term can never become a bonus, and no
zeroing branch at all. SkyRL's `examples/train/algorithms/dapo/main_dapo.py:72-74` instead does
`rewards[i] = 0.0` past the budget, which contradicts both the paper and SkyRL's own docs page. That
creates a discontinuity the scheme exists to avoid: with ALFWorld's 10.0*won, a trajectory at exactly the
budget is worth 9.0 and one token longer is worth 0.0. We saturate at -penalty_factor instead.

SCALE WARNING. `penalty_factor` is in RAW reward units, so its meaning depends on the env's reward scale.
DAPO's correctness reward is +/-1, so factor 1.0 is HALF the total reward range -- enough to sink a
correct answer. ALFWorld pays 10.0*won, so factor 1.0 is only 10% of a success. Matching DAPO's actual
pressure on this env needs a factor nearer 5-10, not 1.0.
"""

from typing import Dict, List, Sequence, Tuple, Union

Reward = Union[float, List[float]]


DAPO_REWARD_RANGE = 2.0
"""Span of DAPO's own rule-based correctness reward (+/-1). The unit `reward_range` is measured in."""


def apply_soft_overlong_punishment(
    rewards: List[Reward],
    response_lengths: Sequence[int],
    budget: int,
    buffer_len: int,
    penalty_factor: float = 1.0,
    reward_range: float = DAPO_REWARD_RANGE,
) -> Tuple[List[Reward], Dict[str, float]]:
    """Penalise long responses in place-ish, returning (rewards, stats).

    Args:
        rewards: one entry per response; either a scalar or a per-token/per-turn list. For a list the
            penalty lands on the LAST element, where the terminal env reward sits.
        response_lengths: token count per response. For multi-turn rollouts this is the WHOLE
            trajectory's response ids (all turns concatenated), so `budget` must be the trajectory budget.
        budget: length at which the penalty saturates at `penalty_factor`.
        buffer_len: width of the ramp, in tokens. <= 0 disables the transform entirely.
        penalty_factor: penalty at `budget` on DAPO's own reward scale; DAPO publishes 1.0.
        reward_range: span of THIS env's reward, so the factor stays proportional instead of absolute.
            Saturating penalty = `penalty_factor * reward_range / 2.0`. Default 2.0 = DAPO's own +/-1
            span, which reproduces the paper exactly. ALFWorld pays `10.0 * won`, so pass 10.0 there.

    Returns:
        (rewards, stats) where stats reports what actually fired -- a penalty nothing triggers is
        indistinguishable from a penalty that is not wired up.
    """
    n = len(rewards)
    stats: Dict[str, float] = {
        "overlong_buffer/frac_penalized": 0.0,
        "overlong_buffer/frac_saturated": 0.0,
        "overlong_buffer/onset_tokens": 0.0,
        "overlong_buffer/effective_penalty": 0.0,
    }
    if buffer_len <= 0 or n == 0:
        return rewards, stats

    onset = max(0, budget - buffer_len)
    # Proportional, not absolute: the published factor is defined against DAPO's +/-1 span, so rescale it
    # to this env's. reward_range == DAPO_REWARD_RANGE leaves it untouched.
    effective_factor = penalty_factor * (reward_range / DAPO_REWARD_RANGE)
    n_ramp = 0
    n_saturated = 0
    for i, length in enumerate(response_lengths):
        if length <= onset:
            continue
        # Linear ramp, capped at effective_factor -- the `min(..., 0)` in verl, expressed on the
        # magnitude. Additive, never a wholesale zeroing.
        penalty = min(1.0, (length - onset) / buffer_len) * effective_factor
        if isinstance(rewards[i], list):
            if rewards[i]:
                rewards[i][-1] -= penalty
        else:
            rewards[i] -= penalty
        n_ramp += 1
        if length > budget:
            n_saturated += 1

    stats["overlong_buffer/frac_penalized"] = n_ramp / n
    stats["overlong_buffer/frac_saturated"] = n_saturated / n
    stats["overlong_buffer/effective_penalty"] = effective_factor
    stats["overlong_buffer/onset_tokens"] = float(onset)
    return rewards, stats

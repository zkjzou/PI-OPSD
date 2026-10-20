"""How much of a GRPO batch actually carries gradient.

WHY THIS EXISTS. Group-relative estimators compute ``A_i = (r_i - group_mean) / (group_std + eps)``
(SkyRL's ``ppo_utils.compute_grpo_outcome_advantage``). When every sample in a prompt's group scores the
SAME — all solved or all failed — each ``r_i`` equals the mean, so the advantage is **exactly 0** and
that group teaches nothing. It still costs a full rollout: N multi-turn episodes, every environment
step, every token, every paid simulator call.

On agentic pools this is not a corner case, it is most of the batch. Measured on this repo:
  * τ1 `retail_train`, 500 tasks x8 on the untrained policy: **227 always-solved + 31 never-solved =
    52% of groups dead**, which showed up live as ``grad_norm`` EXACTLY 0 with ``entropy=nan`` on 6%
    of steps — with nothing in the reward curve to explain it.
  * A τ² arm once ran a step with only **4 of 16 usable groups** after the simulator throttled;
    the update was computed on a quarter of the intended batch and no metric said so.
  * On the retail overfit, zero-variance groups held **24-61% of trainable tokens**.

Every one of those came from offline profiling or post-hoc checkpoint dumps. This module makes it a
live per-step signal for **every** group-based run, in the trainer rather than per environment:
``opsd.skyrl_ext.trainer.OPSDRayPPOTrainer.compute_advantages_and_returns``, the base of every OPSD
trainer, logs it under ``grpo/*``, so τ², ALFWorld, AppWorld and anything added later get it without
wiring.

⚠️ THE TOKEN METRIC IS THE NON-OBVIOUS ONE. ``zero_variance_filter`` cannot rescue dead groups —
their advantage is already 0 — it only decides whether their tokens stay in ``token_mean``'s
denominator. With it off they inflate the normaliser while contributing nothing, diluting the real
gradient a measured 1.3-2.6x. ``signal_token_frac`` reports that directly.

⚠️ ``signal_frac`` FALLING IS NOT AUTOMATICALLY BAD. As a policy masters a pool, groups saturate
all-solved and the fraction drops — that is success, and the fix is a harder pool, not a knob. Read
it with ``all_solved_frac`` vs ``all_failed_frac``: the two saturation modes have opposite remedies,
which a single "dead groups" count cannot distinguish.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence


def gradient_signal_metrics(
    scores: Sequence[float],
    index: Sequence[Any],
    token_counts: Sequence[float] | None = None,
    *,
    prefix: str = "grpo/",
    eps: float = 1e-12,
) -> dict[str, float]:
    """Per-batch breakdown of which groups carry a usable gradient.

    Args:
        scores: per-sequence scalar reward, exactly as the advantage estimator sees it — i.e. after
            any transform that zeroed infra failures or over-long traces. Reporting the
            pre-transform reward would claim signal the optimiser never receives.
        index: the group key per sequence (SkyRL's ``uids``); groups are formed by equality.
        token_counts: optional per-sequence trainable-token count, for the dilution figures.
        prefix: metric namespace. Algorithm-level (``grpo/``) rather than per-env, so the number is
            directly comparable across benchmarks.

    ⚠️ ``all_solved_frac`` / ``all_failed_frac`` assume a reward where >0 means success (true for
    every env here). Under a general continuous reward a dead group is merely all-same and a
    constant non-zero value counts as "solved"; ``signal_frac`` stays correct either way.

    Returns a flat ``{metric: float}`` dict, empty when there is nothing to report — notably when
    every group has one member, which is how a non-group estimator (GAE/PPO) shows up. Emitting
    ``signal_frac=0`` there would read as a broken run rather than an inapplicable metric.
    """

    n = len(scores)
    if not n or len(index) != n:
        return {}

    by_group: dict[Any, list[int]] = defaultdict(list)
    for i, key in enumerate(index):
        by_group[key].append(i)

    # Single-member groups mean this is not a group-relative estimator; the metric is meaningless.
    if max(len(v) for v in by_group.values()) < 2:
        return {}

    n_groups = len(by_group)
    n_signal = n_all_solved = n_all_failed = 0
    std_sum = 0.0
    tokens_total = tokens_signal = 0.0

    for indices in by_group.values():
        group = [float(scores[i]) for i in indices]
        mean = sum(group) / len(group)
        var = sum((r - mean) ** 2 for r in group) / len(group)
        has_signal = var > eps
        if has_signal:
            n_signal += 1
            std_sum += var**0.5
        elif mean > 0:
            # Mastered: the pool is too easy for the current policy -> harder tasks, not a new knob.
            n_all_solved += 1
        else:
            # Out of reach, or the reward/environment is broken for these tasks.
            n_all_failed += 1

        if token_counts is not None:
            group_tokens = sum(float(token_counts[i]) for i in indices)
            tokens_total += group_tokens
            if has_signal:
                tokens_signal += group_tokens

    out = {
        f"{prefix}groups": float(n_groups),
        f"{prefix}groups_with_signal": float(n_signal),
        # THE HEADLINE: the share of this batch's rollout cost that produced any gradient at all.
        f"{prefix}signal_frac": n_signal / n_groups,
        f"{prefix}all_solved_frac": n_all_solved / n_groups,
        f"{prefix}all_failed_frac": n_all_failed / n_groups,
        # Mean spread over the groups that HAVE spread. Averaging dead groups in would just
        # re-encode signal_frac and hide whether live groups are borderline (one sample apart) or
        # genuinely split.
        f"{prefix}reward_std_mean": (std_sum / n_signal) if n_signal else 0.0,
        f"{prefix}group_size_mean": n / n_groups,
    }

    if token_counts is not None and tokens_total:
        out[f"{prefix}signal_token_frac"] = tokens_signal / tokens_total
        out[f"{prefix}dead_tokens"] = tokens_total - tokens_signal

    return out

"""Per-position group-baselined token advantage for OPSD.

    R[i,t]   = env[i] - beta * k1[i,t]                         combined per-token reward
    A[i,t]   = R[i,t] - mean_{j in group(i)} Rtilde[j,t]       per-POSITION group baseline
    Rtilde[j,t] = R[j,t]            if j has a response token at t
                = env[j]            otherwise   (that rollout's outcome; it pays no more KL once done)

Motivation. GRPO's estimator does `scores = token_level_rewards.sum(dim=-1)` and broadcasts one scalar
per sequence, which flattens a dense per-token KL into a single number -- the exact structure the
distillation signal exists to provide. `no_op` keeps the density but has no baseline at all, so it
carries the full variance of the raw reward. This sits between them: keep the per-token reward, and
subtract a baseline estimated per POSITION from the rollout group.

BETA=0 IS EXACTLY GRPO. With the teacher term off, R[i,t] = env[i] at every position and the baseline
collapses to the group mean outcome, constant in t, so A[i,t] = env[i] - mean_j env[j] -- vanilla GRPO
(norm_by_std=False; set it True to match a grpo_norm_by_std=true recipe). That makes beta a genuine
one-variable dial away from the reference estimator rather than a wholesale replacement, which is the
only reason an arm on this path is interpretable against the GRPO/DAPO numbers.

WHY env[i] AND NOT env[i,t]. SkyRL's reward tensor is [B,T] with the outcome sitting on the LAST
response token and zeros everywhere else. Using it literally would (a) leave the task signal touching
exactly one token per rollout while the teacher term drives all the others -- near-pure distillation
wearing an env-anchored label -- and (b) put R (~beta*k1, order 0.1) and the imputed value (the outcome,
order 10 on ALFWorld) on wildly different scales, so at deep positions the baseline is dominated by the
finished rollouts' outcomes and every tail token of a long rollout collects an advantage of roughly
-mean(outcome). On ALFWorld at 90% success that is about -9 per tail token: a length penalty an order of
magnitude larger than any teacher signal, which is what the arm would actually be measuring. Broadcasting
the outcome (the reward-to-go reading, gamma=1 with a terminal reward) puts R and the imputation on one
scale and gives the task signal at every position. OPSD_TOKEN_ADV_ENV=terminal keeps the literal
per-token tensor as an ablation; it is not a sane default.

Is a per-position group mean meaningful? Position t in two different rollouts holds unrelated tokens,
so this is NOT a semantic average. It is a crude non-parametric estimate of V(t) -- the typical
reward-to-date at depth t for this task -- which is a legitimate, standard time-dependent baseline. For
multi-turn agents t tracks episode stage (turn 1 looks like turn 1 across rollouts), so the estimate is
better than it sounds. Bias-wise it is no worse than GRPO, which likewise includes the sample in its own
group mean.

The imputation is load-bearing. Without it the group at position t shrinks to only the rollouts still
running, so the deep-tail baseline would be a mean over one or two long sequences -- maximum noise at
exactly the positions where trajectories diverge most. Filling short rollouts with their outcome keeps
the denominator at G for every position and encodes the right prior: a rollout that already finished
contributes what it actually earned and stops accruing teacher cost.

Knobs
-----
OPSD_TOKEN_ADV        1 to enable (replaces the advantage; mutually exclusive with MIX_LAMBDA/RLSD)
OPSD_TOKEN_ADV_COEF   beta on the k1 term (default 1.0; 0 = exactly GRPO)
OPSD_TOKEN_ADV_STD    1 to divide by the per-position group std (default 0 = mean-only baseline)
OPSD_TOKEN_ADV_ENV    broadcast (default) | terminal -- where the env reward sits along t
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import torch

STD_EPS = 1e-6


def token_adv_enabled() -> bool:
    return os.environ.get("OPSD_TOKEN_ADV", "0").strip().lower() in ("1", "true", "yes")


def token_adv_coef() -> float:
    return float(os.environ.get("OPSD_TOKEN_ADV_COEF", "1.0") or 0.0)


def token_adv_norm_std() -> bool:
    return os.environ.get("OPSD_TOKEN_ADV_STD", "0").strip().lower() in ("1", "true", "yes")


def token_adv_env_broadcast() -> bool:
    """False only for OPSD_TOKEN_ADV_ENV=terminal (the ablation). See the module docstring."""
    mode = os.environ.get("OPSD_TOKEN_ADV_ENV", "broadcast").strip().lower()
    if mode not in ("broadcast", "terminal"):
        raise ValueError(f"OPSD_TOKEN_ADV_ENV must be broadcast|terminal, got {mode!r}")
    return mode == "broadcast"


def group_index(uids: Sequence) -> Tuple[List[int], int]:
    """Map rollout uids -> dense group ids, preserving first-seen order."""
    seen: Dict[object, int] = {}
    out: List[int] = []
    for u in uids:
        if u not in seen:
            seen[u] = len(seen)
        out.append(seen[u])
    return out, len(seen)


def per_position_group_advantage(
    env_rewards: torch.Tensor,
    per_token_kl: torch.Tensor,
    response_mask: torch.Tensor,
    uids: Sequence,
    *,
    beta: float = 1.0,
    norm_by_std: bool = False,
    env_broadcast: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """A[i,t] = R[i,t] - baseline_group(i)[t], with R = env - beta*k1 and the outcome imputation.

    Args:
        env_rewards:   [B, T] per-token env reward (sparse; the outcome sits on the last response token)
        per_token_kl:  [B, T] sanitized k1(student || hinted-teacher), already clipped/masked
        response_mask: [B, T] 1 where the rollout actually has a response token
        uids:          length-B sequence identifying which rollouts share a prompt
        env_broadcast: put the OUTCOME at every position (default) rather than only where the reward
                       tensor holds it. See the module docstring -- `terminal` is an ablation, not a
                       second reasonable default.
    """
    mask = response_mask.to(env_rewards.dtype)
    # outcome[i] = that rollout's env return. Also what fills positions it never reached, so under
    # broadcast the imputed value and R live on ONE scale (both are the outcome, +/- the teacher term).
    outcome = (env_rewards * mask).sum(dim=-1, keepdim=True)  # [B, 1]

    env_t = outcome.expand_as(env_rewards) if env_broadcast else env_rewards
    R = env_t - float(beta) * per_token_kl
    R_tilde = torch.where(response_mask.bool(), R, outcome.expand_as(R))  # [B, T]

    gidx_list, n_groups = group_index(uids)
    gidx = torch.as_tensor(gidx_list, device=R.device, dtype=torch.long)  # [B]

    B, T = R.shape
    ones = torch.ones(B, 1, device=R.device, dtype=R.dtype)
    counts = torch.zeros(n_groups, 1, device=R.device, dtype=R.dtype).index_add_(0, gidx, ones)
    counts = counts.clamp(min=1.0)

    sums = torch.zeros(n_groups, T, device=R.device, dtype=R.dtype).index_add_(0, gidx, R_tilde)
    baseline = sums / counts  # [n_groups, T]

    adv = R - baseline[gidx]
    if norm_by_std:
        sq = torch.zeros(n_groups, T, device=R.device, dtype=R.dtype).index_add_(0, gidx, R_tilde * R_tilde)
        var = (sq / counts) - baseline * baseline
        std = var.clamp(min=0.0).sqrt()
        adv = adv / (std[gidx] + STD_EPS)
    adv = adv * mask

    with torch.no_grad():
        rm = response_mask.bool()
        _absm = lambda t: (t[rm].abs().mean().item() if rm.any() else 0.0)  # noqa: E731
        # How much of R is task vs teacher, and how much variance the baseline actually removes --
        # a baseline that explains nothing means the positional estimate is not carrying information.
        raw_var = float(R[rm].var().item()) if int(rm.sum()) > 1 else 0.0
        adv_var = float(adv[rm].var().item()) if int(rm.sum()) > 1 else 0.0
        metrics = {
            "tokenadv/beta": float(beta),
            "tokenadv/env_broadcast": 1.0 if env_broadcast else 0.0,
            "tokenadv/groups": float(n_groups),
            "tokenadv/rollouts_per_group": float(B) / max(1, n_groups),
            "tokenadv/env_abs": _absm(env_t),
            "tokenadv/kl_abs": _absm(float(beta) * per_token_kl),
            "tokenadv/baseline_abs": _absm(baseline[gidx]),
            "tokenadv/advantage_abs": _absm(adv),
            # fraction of positions that were IMPUTED rather than real (rises with length spread)
            "tokenadv/imputed_frac": float((~rm).to(R.dtype).mean().item()),
            "tokenadv/variance_explained": (1.0 - adv_var / raw_var) if raw_var > 0 else 0.0,
        }
    return adv, metrics

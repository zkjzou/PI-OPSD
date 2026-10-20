"""Per-token reverse-KL reward shaping for on-policy distillation (OPD / OPSD / Skill-SD).

The k1 estimator ``student_logp - teacher_logp`` with the exact NaN/inf sanitize + clamp + mask guards
was copy-pasted verbatim in ``OnPolicyDistillationTrainer.apply_reward_kl_penalty`` and
``SelfDistillationTrainer.apply_reward_kl_penalty`` (Skill-SD inherits OPD's). This module is the single
source of truth; the trainers keep only the parts that genuinely differ (OPD: reward = -k1; OPSD:
optional per-token clip + env-mixed reward + turn/channel monitoring).

The functions are pure (torch tensors in/out), so they are numerically parity-tested on CPU against the
original inline computation (see opsd/algorithms/tests/test_reward_shaping.py) — no GPU needed
to prove the extraction changed nothing.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

# Which half of the per-token reverse-KL term is allowed to reach the reward. See `apply_side_mask`.
DISTILL_SIDES = ("both", "plus", "minus")

# Clamp range for the per-token reverse-KL reward. A teacher log-prob of -inf (teacher assigns ~0
# probability to a token the student sampled) gives an unbounded KL; clamping bounds the per-token
# penalty, mirroring the numerical guards skyrl uses in compute_approx_kl.
KL_CLAMP = 20.0


def per_token_reverse_kl(
    student_lp: torch.Tensor,
    teacher_lp: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    kl_clamp: float = KL_CLAMP,
    token_clip: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token k1 reverse-KL(student || teacher) reward term.

    ``per_token_kl = where(valid, clamp(nan_to_num(student_lp - teacher_lp), ±kl_clamp)[, ±token_clip] * loss_mask, 0)``

    Sanitize semantics (unchanged from the trainers):
      - NaN (broken forward)          -> 0.0    (neutral, no gradient from those tokens)
      - +inf (teacher hates the token) -> +kl_clamp (bounded penalty); -inf -> -kl_clamp
      - masked positions               -> 0.0 via torch.where (avoids NaN*0 leakage)
      - token_clip (OPSD, default 0.5) -> an extra symmetric per-token clamp, applied only when > 0.

    Returns ``(per_token_kl, raw_diff)`` — ``raw_diff`` (pre-sanitize) feeds the bad-value diagnostic.
    """
    valid = loss_mask.bool()
    raw_diff = student_lp - teacher_lp
    diff = torch.nan_to_num(raw_diff, nan=0.0, posinf=kl_clamp, neginf=-kl_clamp)
    diff = diff.clamp(-kl_clamp, kl_clamp)
    if token_clip:
        tc = float(token_clip)
        if tc > 0:
            diff = diff.clamp(-tc, tc)
    per_token_kl = torch.where(valid, diff * loss_mask, torch.zeros_like(diff))
    return per_token_kl, raw_diff


def apply_side_mask(
    per_token_kl: torch.Tensor,
    loss_mask: torch.Tensor,
    side: str,
    *,
    match_magnitude: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Keep only the ``d+`` or the ``d-`` half of the per-token reverse-KL term (`OPSD_DISTILL_SIDE`).

    THE SIGN CONVENTION, which is the whole trap. ``per_token_kl = student_lp - teacher_lp`` and the OPSD
    reward is ``-distill_coef * per_token_kl``, so with the `no_op` estimator passing reward straight
    through to a per-token advantage and ratio==1 on the first inner step, the update is
    ``theta <- theta - lr * distill_coef * per_token_kl_t * grad log pi(y_t)``:

      per_token_kl_t < 0  teacher likes the emitted token MORE  -> `d+` -> reward > 0 -> token pushed UP
      per_token_kl_t > 0  teacher likes it LESS                 -> `d-` -> reward < 0 -> token pushed DOWN

    So ``side="plus"`` keeps the NEGATIVE entries (up-only reinforcement, advantage >= 0 everywhere) and
    ``side="minus"`` keeps the POSITIVE ones (down-only suppression, advantage <= 0 everywhere). Naming the
    halves after the direction of REINFORCEMENT rather than the sign of the difference matches
    `pi_utility.divergence_attributes`, so ``mean|plus-only term| == pi_utility/d_plus`` exactly (at
    token_clip=0) and the ablation is readable off the panels already being logged.

    MAGNITUDE MATCHING. ``d = d+ + d-``, so masking a side also removes its share of the gradient NORM --
    and on AppWorld that share is wildly lopsided: `d-` measured 79-92% of `d` on all six Instruct arms
    (d+ 8-23%). A raw plus-only arm therefore runs at ~1/8 the effective learning rate of the baseline and
    would test dose, not direction. ``match_magnitude=True`` rescales the surviving side so its mean
    |value| over masked tokens equals the full both-sided term's, per batch (the share drifts over
    training -- 8.8% -> 12.2% on ssa -- so a static coefficient would only match at one step).

    Returns ``(term, metrics)``. Off-mask positions stay zero; ``side="both"`` returns the input unchanged.
    """
    if side not in DISTILL_SIDES:
        raise ValueError(f"OPSD_DISTILL_SIDE must be one of {DISTILL_SIDES}, got {side!r}")
    valid = loss_mask.bool()
    n_valid = valid.sum().clamp(min=1)
    full_mean_abs = (per_token_kl.abs() * loss_mask).sum() / n_valid
    metrics: Dict[str, float] = {
        # numeric so it plots: 0 = both, +1 = plus-only, -1 = minus-only
        "distill/side_code": {"both": 0.0, "plus": 1.0, "minus": -1.0}[side],
        "distill/side_match_on": float(bool(match_magnitude)),
        "distill/full_mean_abs": full_mean_abs.item(),
    }
    if side == "both":
        metrics.update(
            {"distill/side_scale": 1.0, "distill/side_mean_abs": full_mean_abs.item(), "distill/side_frac_tokens": 1.0}
        )
        return per_token_kl, metrics

    # plus -> keep per_token_kl < 0 ; minus -> keep per_token_kl > 0
    term = per_token_kl.clamp(max=0.0) if side == "plus" else per_token_kl.clamp(min=0.0)
    term = torch.where(valid, term, torch.zeros_like(term))
    side_mean_abs = (term.abs() * loss_mask).sum() / n_valid
    # Fraction of masked tokens that survive -- separates "few tokens, large values" from the converse,
    # which mean|.| alone cannot, and is what says whether a side arm still has dense per-token credit.
    metrics["distill/side_frac_tokens"] = ((term != 0) & valid).sum().item() / float(n_valid.item())
    metrics["distill/side_mean_abs"] = side_mean_abs.item()

    scale = 1.0
    if match_magnitude:
        # An exactly-matched teacher (or an all-one-side batch) gives side_mean_abs == 0; leaving scale at
        # 1.0 there means "no pressure this step", which is correct, rather than inf.
        if float(side_mean_abs.item()) > 1e-12:
            scale = float((full_mean_abs / side_mean_abs).item())
            term = term * scale
    metrics["distill/side_scale"] = scale
    return term, metrics


def kl_shaping_metrics(
    raw_diff: torch.Tensor,
    per_token_kl: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    student_lp: Optional[torch.Tensor] = None,
    teacher_lp: Optional[torch.Tensor] = None,
    lp_detail: str = "full",
) -> Dict[str, float]:
    """Diagnostics for the k1 reward. Always: ``train/teacher_kl`` (mean per-token; -> 0 as the student
    matches the teacher) and ``train/teacher_kl_bad_valid`` (count of NaN/inf raw diffs in the response).

    Per-tensor log-prob diagnostics (``distill/{student,teacher}_lp_*``), selected by ``lp_detail``:
      - ``"full"`` (OPD): ``_lp_nan`` + ``_lp_inf`` + finite ``_lp_min`` / ``_lp_max``
      - ``"nan"``  (OPSD): ``_lp_nan`` only
      - ``None``:   none
    """
    valid = loss_mask.bool()
    mask_sum = loss_mask.sum().clamp(min=1)
    metrics: Dict[str, float] = {
        "train/teacher_kl": (per_token_kl.sum() / mask_sum).item(),
        "train/teacher_kl_bad_valid": float(((torch.isnan(raw_diff) | torch.isinf(raw_diff)) & valid).sum().item()),
    }
    if lp_detail is None:
        return metrics
    for nm, t in (("student", student_lp), ("teacher", teacher_lp)):
        if t is None:
            continue
        tv = t[valid]
        metrics[f"distill/{nm}_lp_nan"] = float(torch.isnan(tv).sum().item())
        if lp_detail == "full":
            metrics[f"distill/{nm}_lp_inf"] = float(torch.isinf(tv).sum().item())
            fin = tv[torch.isfinite(tv)]
            if fin.numel():
                metrics[f"distill/{nm}_lp_min"] = float(fin.min().item())
                metrics[f"distill/{nm}_lp_max"] = float(fin.max().item())
    return metrics


__all__ = ["KL_CLAMP", "DISTILL_SIDES", "per_token_reverse_kl", "apply_side_mask", "kl_shaping_metrics"]

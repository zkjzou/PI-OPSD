"""RLSD — RLVR with Self-Distillation (arXiv:2604.03128, "Self-Distilled RLVR").

The paper's diagnosis is the one our own AppWorld sweep arrived at empirically. OPSD asks a student that
CANNOT see the privileged information r to match a teacher that CAN, which is ill-posed: their Theorem 1
gives L_OPSD = L* + I(Y_t; R | X, Y_<t), an irreducible mutual-information floor that is independent of
theta. At the gradient level (their Prop. 1) the per-sample gradient splits into a benign marginal-matching
term plus an r-specific deviation with zero mean but variance proportional to that MI; a path-dependent
optimizer accumulates the deviation, so the model learns an x -> r mapping and starts appealing to a
reference solution it never sees at inference. Hence early gains then decay -- which is exactly the shape
of our own arms, and of SDPO's live-teacher collapse.

RLSD's fix is to stop using the teacher as a GENERATIVE TARGET and use it only as a MAGNITUDE EVALUATOR:

    Delta_t = sg(log P_T(y_t) - log P_S(y_t))                       privileged information gain   (13)
    w_t     = exp(sign(A) * Delta_t) = (P_T/P_S)^sign(A)            evidence ratio                (14)
    A_hat_t = A * ((1 - lam) + lam * clip(w_t, 1-eps_w, 1+eps_w))   Algorithm 1, line 17

with P_S = pi_theta(y_t | x, y_<t) and P_T = pi_theta(y_t | x, r, y_<t) -- the SAME weights, re-prompted,
i.e. precisely the two tensors OPSD already computes (`action_log_probs` and `hinted_log_probs`).

Why this matters for us specifically. Our experiments found that the beta (additive loss) path beat
the lambda (convex advantage mix) path at EVERY dose, "consistent with beta being additive (task gradient
intact) while lambda is convex (task signal displaced)". RLSD is neither: it is MULTIPLICATIVE. Because
exp(.) > 0 and lam in [0,1], the factor ((1-lam) + lam*w) is strictly positive, so

    sign(A_hat_t) == sign(A)  for every token, always.

The environment reward keeps exclusive authority over WHICH tokens are reinforced or penalised; the
teacher only redistributes credit WITHIN a trajectory. That is a strictly weaker intrusion on the task
signal than either of the two paths we have already run, which is the reason to expect it to behave
differently rather than land on the same null.

Knobs (all off by default -- OPSD_RLSD=0 leaves every existing arm byte-identical):
  OPSD_RLSD               1 to enable (drop-in replacement for the uniform GRPO advantage)
  OPSD_RLSD_LAMBDA        initial lambda (paper: 0.5)
  OPSD_RLSD_LAMBDA_FINAL  lambda after the decay window (paper: 0.0)
  OPSD_RLSD_DECAY_STEPS   linear decay length in steps (paper: 50; <=0 pins lambda constant)
  OPSD_RLSD_CLIP          eps_w (paper: 0.2)

Teacher choice: the paper syncs the teacher to the student every 10 steps and freezes it in between,
which our teacher_regularization.EMATeacherSync reproduces exactly at OPSD_TEACHER=ema,
OPSD_TEACHER_ALPHA=1.0, OPSD_EMA_INTERVAL=10 (a hard copy every 10 steps). `live` is also safe here in a
way it is NOT for OPSD/SDPO: RLSD has no term pushing pi(.|s) toward pi(.|s,r), so the hint-blind
attractor that collapsed our live arms has nothing to optimise.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Tuple

import torch

# Bound for a raw log-prob difference before it is exponentiated. eps_w clamping makes this nearly
# irrelevant numerically; it exists so a broken forward (inf) cannot poison the batch.
DELTA_CLAMP = 20.0


def rlsd_enabled() -> bool:
    return os.environ.get("OPSD_RLSD", "0").strip().lower() in ("1", "true", "yes")


def rlsd_clip() -> float:
    eps = float(os.environ.get("OPSD_RLSD_CLIP", "0.2") or 0.2)
    if not 0.0 < eps < 1.0:
        raise ValueError(f"OPSD_RLSD_CLIP must be in (0,1), got {eps}")
    return eps


def rlsd_lambda(global_step: int) -> float:
    """Linear lambda schedule.

    The paper is internally inconsistent about the direction: 4.1 says the interpolation exists "to avoid
    an abrupt transition at the start of training" (which reads like a ramp UP), while 5.1 states lambda
    "is initialized at 0.5 and linearly decayed to 0 over the first 50 training steps" and 4.1 also ends
    on "gradually shifting to the uniform advantage". Two of the three statements agree on decay-to-
    uniform, so that is the default here -- but the endpoints are separate knobs, so setting
    OPSD_RLSD_LAMBDA=0.0 / OPSD_RLSD_LAMBDA_FINAL=0.5 gives the ramp-up reading, and
    OPSD_RLSD_DECAY_STEPS=0 pins lambda constant for an ablation that removes the schedule entirely.
    """
    lam0 = float(os.environ.get("OPSD_RLSD_LAMBDA", "0.5") or 0.0)
    lam1 = float(os.environ.get("OPSD_RLSD_LAMBDA_FINAL", "0.0") or 0.0)
    steps = int(os.environ.get("OPSD_RLSD_DECAY_STEPS", "50") or 0)
    for name, v in (("OPSD_RLSD_LAMBDA", lam0), ("OPSD_RLSD_LAMBDA_FINAL", lam1)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be in [0,1], got {v}")
    if steps <= 0:
        return lam0
    frac = min(max(float(global_step) / float(steps), 0.0), 1.0)
    return lam0 + (lam1 - lam0) * frac


def rlsd_teacher_is_inert(global_step: int, *, reward_path_active: bool = False) -> bool:
    """True when the teacher can no longer affect ANY future update, so syncing it is pure waste.

    Motivation: under the paper's default schedule lambda hits 0 at step 50, after which
    `A_hat = A * ((1-0) + 0*w) = A` exactly -- the teacher is irrelevant for the rest of the run. But
    the EMA sync keeps firing on its own cadence, and each one costs a full policy export plus a ref
    re-init (~165 s measured). On a 150-step run at OPSD_EMA_INTERVAL=10 that is ~10 syncs, ~28 min,
    for a tensor nothing reads.

    Two traps this deliberately avoids, both of which a naive `lambda == 0` check would fall into:

      1. **lambda == 0 is not the same as lambda is permanently 0.** The endpoints are configurable, and
         the ramp-UP reading of the paper (OPSD_RLSD_LAMBDA=0 -> LAMBDA_FINAL=0.5) starts at exactly 0.
         Skipping there would freeze the teacher at theta_0 for the whole warm-up and then hand the
         mechanism a stale teacher precisely when it switches on. So we require the schedule's TERMINAL
         value to be 0 as well.
      2. **RLSD is not the only consumer of the teacher tensor.** beta (OPSD_DISTILL_LOSS_COEF), SDAR
         (OPSD_SDAR_LAMBDA), the advantage mix (OPSD_MIX_LAMBDA<1) and the reward path
         (use_kl_in_reward) all read the same forward. Only the mix is mutually exclusive with RLSD;
         the others can legitimately be co-enabled, and freezing the teacher under them would silently
         change THEIR algorithm. Hence the explicit all-consumers check.
    """
    if not rlsd_enabled():
        return False
    if reward_path_active:
        return False
    if float(os.environ.get("OPSD_DISTILL_LOSS_COEF", "0") or 0) != 0.0:
        return False
    if float(os.environ.get("OPSD_SDAR_LAMBDA", "0") or 0) != 0.0:
        return False
    if float(os.environ.get("OPSD_MIX_LAMBDA", "1.0") or 1.0) < 1.0:
        return False
    steps = int(os.environ.get("OPSD_RLSD_DECAY_STEPS", "50") or 0)
    lam0 = float(os.environ.get("OPSD_RLSD_LAMBDA", "0.5") or 0.0)
    lam1 = float(os.environ.get("OPSD_RLSD_LAMBDA_FINAL", "0.0") or 0.0)
    terminal = lam0 if steps <= 0 else lam1
    return terminal == 0.0 and rlsd_lambda(global_step) == 0.0


def rlsd_advantage(
    a_grpo: torch.Tensor,
    student_lp: torch.Tensor,
    teacher_lp: torch.Tensor,
    loss_mask: torch.Tensor,
    response_mask: torch.Tensor,
    lam: float,
    eps_w: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """A_hat_t = A * ((1-lam) + lam * clip(exp(sign(A)*Delta_t), 1-eps_w, 1+eps_w)).

    `a_grpo` is the sequence-level GRPO advantage already broadcast per token by the base trainer, so the
    multiplication is elementwise and the per-sequence sign is inherited unchanged.

    Numerics: we clamp the EXPONENT to [log(1-eps_w), log(1+eps_w)] rather than clipping after exp().
    exp() is monotone, so the two are identical mathematically, but clamping first means a large Delta
    from a broken forward can never produce an inf that then has to be repaired.
    """
    lo, hi = math.log(1.0 - eps_w), math.log(1.0 + eps_w)

    delta = teacher_lp.to(student_lp.dtype) - student_lp  # Delta_t, eq. (13)
    delta = torch.nan_to_num(delta, nan=0.0, posinf=DELTA_CLAMP, neginf=-DELTA_CLAMP)
    delta = delta.clamp(-DELTA_CLAMP, DELTA_CLAMP).detach()  # sg[.]

    exponent = torch.sign(a_grpo) * delta  # eq. (14)
    clipped_low = (exponent < lo) & loss_mask.bool()
    clipped_high = (exponent > hi) & loss_mask.bool()
    w = torch.exp(exponent.clamp(lo, hi))  # eq. (15) clip

    # Off-mask positions carry no credit signal; force w=1 there so they fall back to the uniform
    # advantage rather than inheriting a meaningless log-prob difference.
    w = torch.where(loss_mask.bool(), w, torch.ones_like(w))

    a_hat = (a_grpo * ((1.0 - lam) + lam * w)) * response_mask

    with torch.no_grad():
        rm, lm = response_mask.bool(), loss_mask.bool()
        n_lm = int(lm.sum().item())
        _mean = lambda t, m: (t[m].mean().item() if m.any() else 0.0)  # noqa: E731
        _absm = lambda t, m: (t[m].abs().mean().item() if m.any() else 0.0)  # noqa: E731
        metrics = {
            "rlsd/lambda": float(lam),
            "rlsd/eps_w": float(eps_w),
            "rlsd/w_mean": _mean(w, lm),
            "rlsd/w_min": (w[lm].min().item() if lm.any() else 1.0),
            "rlsd/w_max": (w[lm].max().item() if lm.any() else 1.0),
            "rlsd/delta_abs_mean": _absm(delta, lm),
            # The paper's Figure 5(c) health check: clipping should be ACTIVE but rare (they report the
            # ratios stabilising around 3-6%). ~0 means the teacher is saying nothing; very high means
            # eps_w is throwing most of the signal away.
            "rlsd/clip_low_ratio": (int(clipped_low.sum().item()) / n_lm if n_lm else 0.0),
            "rlsd/clip_high_ratio": (int(clipped_high.sum().item()) / n_lm if n_lm else 0.0),
            "rlsd/a_grpo_abs": _absm(a_grpo, rm),
            "rlsd/advantage_abs": _absm(a_hat, rm),
            # Structural invariant, not a diagnostic: ((1-lam) + lam*w) > 0 for lam in [0,1], w > 0, so
            # the environment reward keeps exclusive authority over direction. If this is ever < 1.0 the
            # implementation is wrong, not the run.
            "rlsd/sign_preserved": (
                float((torch.sign(a_hat[rm]) == torch.sign(a_grpo[rm])).float().mean().item()) if rm.any() else 1.0
            ),
        }
    return a_hat, metrics

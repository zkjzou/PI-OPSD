"""Loss-side teacher terms of OPSD: beta (teacher-KL) and SDAR (the gated variant).

Both terms read the frozen/live teacher's log-probs of the sampled tokens (``distill_log_probs``, computed
under ``torch.no_grad`` by the OPSD trainer) and are added to the policy loss in the FSDP policy worker
(``opsd.skyrl_ext.policy_worker``), beside SkyRL's theta0 KL leash and entropy bonus:

    loss = policy_loss * dp_size + (kl_loss_term + distill_loss_term + sdar_loss_term - entropy_loss_term) * mbw

beta -- J_OPSD = E[R_env] - beta * KL(pi_student || sg[pi_teacher(.|s,I)]). The KL MUST live in the loss, not the
    reward: it depends on theta both through the sampling distribution and directly through log pi_student, so
    folding it into the reward keeps only the REINFORCE term and throws away the exact pathwise per-token
    gradient -- which is the entire mechanism of distillation. It is also destroyed downstream:
    compute_grpo_outcome_advantage sums token rewards to one scalar and broadcasts a constant advantage, erasing
    the per-token structure. Deliberately independent of ``use_kl_loss`` so the teacher term toggles via
    OPSD_DISTILL_LOSS_COEF alone while the theta0 anti-collapse leash stays put. Because masked_mean normalises
    by token count, the coefficient is on a ~0.04-scale quantity, so it is directly comparable to the beta swept
    in arXiv:2602.04942 ({0.1, 0.25, 0.5}).

SDAR (arXiv:2605.15155, "Self-Distilled Agentic Reinforcement Learning") -- the beta term, but GATED per token:

    Delta_t = sg(log pi_teacher(y_t | s_t^+) - log pi_student(y_t | s_t))
    g_t     = sigmoid(beta * score_t)            (score = Delta_t for the default `gap` gate)
    l_t     = g_t * (log pi_teacher(y_t) - log pi_student(y_t))
    L       = L_GRPO + lam * Agg(l_t)

    The gate is DETACHED, so gradients flow only through the student log-prob. On ALFWorld the paper measures
    the mean teacher-student gap as consistently NEGATIVE (their Fig. 5): the privileged teacher assigns LOWER
    probability than the student to sampled tokens on average, so ungated distillation spends most of its
    pressure pushing the student toward a teacher that disagrees. sigmoid(beta*Delta) attenuates exactly those
    tokens and amplifies the minority the teacher endorses. beta=0 recovers a uniform gate of 0.5, which is
    their no-gate ablation.

The maths is pure (torch in, torch out) and byte-for-byte the expressions the fork computed inline in
``PolicyWorkerBase._forward_backward_micro``, so it is CPU-tested here and parity-tested against fork goldens
(opsd/tests/fixtures/fork_parity/worker). The knobs are resolved ONCE on the driver (``LossKnobs.from_env``) so
a typo fails at startup, then pushed to the policy actors; the worker itself reads no environment.

Imported inside every policy actor (the OPSD worker module imports it), so keep it import-light.
"""

import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch

from skyrl.backends.skyrl_train.utils.ppo_utils import compute_approx_kl
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean

# Bound a raw log-prob gap before it reaches the sigmoid. The gate is bounded anyway; this exists so a broken
# teacher forward cannot produce a non-finite score.
SDAR_DELTA_CLAMP = 20.0

SDAR_GATES = ("gap", "entropy", "soft_or")
_SDAR_GATE_ALIASES = {
    "gap": "gap",
    "entropy": "entropy",
    "soft_or": "soft_or",
    "softor": "soft_or",
    "soft-or": "soft_or",
}


def normalise_sdar_gate(kind: str) -> str:
    """``OPSD_SDAR_GATE`` -> one of ``SDAR_GATES``. Case/whitespace-insensitive; ``softor``/``soft-or`` alias
    ``soft_or``. Anything else raises: an unknown gate must never silently default."""
    canon = _SDAR_GATE_ALIASES.get(kind.strip().lower())
    if canon is None:
        raise ValueError(f"OPSD_SDAR_GATE must be gap|entropy|soft_or, got {kind.strip().lower()!r}")
    return canon


@dataclass(frozen=True)
class LossKnobs:
    """The loss-side teacher knobs, resolved on the driver and pushed to the policy actors.

    ``distill_coef`` is beta (OPSD_DISTILL_LOSS_COEF); ``sdar_lambda`` / ``sdar_beta`` / ``sdar_gate`` are
    OPSD_SDAR_LAMBDA / OPSD_SDAR_BETA / OPSD_SDAR_GATE. Both terms default OFF.
    """

    distill_coef: float = 0.0
    sdar_lambda: float = 0.0
    sdar_beta: float = 5.0
    sdar_gate: str = "gap"

    def __post_init__(self):
        object.__setattr__(self, "sdar_gate", normalise_sdar_gate(self.sdar_gate))

    @property
    def active(self) -> bool:
        """True when either loss-side term is requested; False means the run is plain upstream SkyRL."""
        return self.distill_coef != 0.0 or self.sdar_lambda != 0.0

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "LossKnobs":
        """Parse the knobs exactly as the fork's policy worker did (``float(v or 0)``, so an empty string is 0).

        OPSD_SDAR_BETA and OPSD_SDAR_GATE are read only when OPSD_SDAR_LAMBDA != 0, as in the fork, where a
        bogus gate with lambda=0 was harmless; with lambda != 0 a bogus gate now raises here, at startup, instead
        of at the first micro-batch.
        """
        env = os.environ if environ is None else environ
        distill_coef = float(env.get("OPSD_DISTILL_LOSS_COEF", "0") or 0)
        sdar_lambda = float(env.get("OPSD_SDAR_LAMBDA", "0") or 0)
        if sdar_lambda == 0.0:
            return cls(distill_coef=distill_coef)
        return cls(
            distill_coef=distill_coef,
            sdar_lambda=sdar_lambda,
            sdar_beta=float(env.get("OPSD_SDAR_BETA", "5.0") or 0),
            sdar_gate=env.get("OPSD_SDAR_GATE", "gap"),
        )


def distill_kl_term(
    student_lp: torch.Tensor, teacher_lp: torch.Tensor, loss_mask: torch.Tensor, kl_estimator_type: str
) -> torch.Tensor:
    """beta's KL: per-sequence token-mean of ``compute_approx_kl(student, teacher)``, averaged over the batch.

    Same estimator (``algorithm.kl_estimator_type``) and reduction as SkyRL's theta0 KL loss.
    """
    distill_kl = compute_approx_kl(
        student_lp,
        teacher_lp,
        loss_mask=loss_mask,
        kl_estimator_type=kl_estimator_type,
    )
    return masked_mean(distill_kl, loss_mask, dim=-1).mean()


def sdar_gate(
    student_lp: torch.Tensor,
    teacher_lp: torch.Tensor,
    beta: float,
    gate: str = "gap",
    entropy: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SDAR's per-token pieces -> ``(teacher, delta, g)``: the sanitised teacher log-probs, the detached and
    clamped gap Delta_t, and the detached gate g_t = sigmoid(beta * score_t).

    ``entropy`` (B, A) is required by the ``entropy`` / ``soft_or`` gates and only ever read detached.
    """
    gate = normalise_sdar_gate(gate)
    # A broken teacher forward must not take the run down; NaN -> treat as "teacher agrees".
    teacher = torch.nan_to_num(teacher_lp, nan=0.0, posinf=SDAR_DELTA_CLAMP, neginf=-SDAR_DELTA_CLAMP)
    teacher = torch.where(torch.isfinite(teacher_lp), teacher, student_lp.detach())
    delta = (teacher - student_lp).detach().clamp(-SDAR_DELTA_CLAMP, SDAR_DELTA_CLAMP)
    if gate == "gap":
        score = delta
    elif gate == "entropy":
        score = entropy.detach()
    else:  # soft_or
        score = 1.0 - (1.0 - entropy.detach()) * (1.0 - delta)
    return teacher, delta, torch.sigmoid(beta * score)


def sdar_term(
    student_lp: torch.Tensor,
    teacher_lp: torch.Tensor,
    loss_mask: torch.Tensor,
    beta: float,
    gate: str = "gap",
    entropy: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SDAR's gated distillation loss -> ``(sdar_loss, gate_active, delta_mean)``.

    ``sdar_loss`` = per-sequence token-mean of g_t * (log pi_T - log pi_S), averaged over the batch; gradients
    flow only through the student log-prob. ``gate_active`` is the fraction of tokens with g_t > 0.5 (the
    paper's Fig. 5, right; starts below 0.5 and rises as the student improves) and ``delta_mean`` the mean
    teacher-student gap (Fig. 5, left; expected NEGATIVE early and to converge toward 0).
    """
    teacher, delta, g = sdar_gate(student_lp, teacher_lp, beta, gate, entropy)
    sdar_tok = g * (teacher - student_lp)
    sdar_loss = masked_mean(sdar_tok, loss_mask, dim=-1).mean()
    gate_active = masked_mean((g > 0.5).float(), loss_mask, dim=-1).mean()
    delta_mean = masked_mean(delta, loss_mask, dim=-1).mean()
    return sdar_loss, gate_active, delta_mean


def opsd_loss_terms(
    knobs: LossKnobs,
    action_log_probs: torch.Tensor,
    teacher_lp: Optional[torch.Tensor],
    loss_mask: torch.Tensor,
    entropy_BS: torch.Tensor,
    kl_estimator_type: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Both loss-side terms for one micro-batch -> ``(distill_loss_term, sdar_loss_term, status)``.

    ``teacher_lp`` is None when the batch carried no ``distill_log_probs``; a requested term then degrades to
    an exact 0 instead of crashing the run. ``status`` holds the worker's diagnostic keys (they reach W&B as
    ``policy/<key>``), emitted only for a REQUESTED term -- gated on the request, not on the tensor arriving,
    so ``distill_loss_attached`` / ``sdar_attached`` = 0 still separates "coef set but the teacher tensor never
    reached the worker" from a working arm, while an unrelated run gets no constant-zero panels. Named
    ``distill_loss_*`` to stay distinct from the trainer's ``distill/*`` series (the REWARD-path teacher term).
    """
    # Move it here rather than relying on Experience.to_device, which does not know about the tensor.
    if teacher_lp is not None:
        teacher_lp = teacher_lp.to(action_log_probs.device)

    if knobs.distill_coef != 0.0 and teacher_lp is not None:
        distill_kl = distill_kl_term(action_log_probs, teacher_lp, loss_mask, kl_estimator_type)
    else:
        distill_kl = torch.tensor(0.0, device=action_log_probs.device)
    distill_loss_term = distill_kl * knobs.distill_coef

    if knobs.sdar_lambda != 0.0 and teacher_lp is not None:
        sdar_loss, sdar_gate_active, sdar_delta_mean = sdar_term(
            action_log_probs, teacher_lp, loss_mask, knobs.sdar_beta, knobs.sdar_gate, entropy_BS
        )
    else:
        sdar_loss = torch.tensor(0.0, device=action_log_probs.device)
        sdar_gate_active = torch.tensor(0.0, device=action_log_probs.device)
        sdar_delta_mean = torch.tensor(0.0, device=action_log_probs.device)
    sdar_loss_term = sdar_loss * knobs.sdar_lambda

    status: Dict[str, float] = {}
    if knobs.distill_coef != 0.0:
        status["distill_loss_coef"] = knobs.distill_coef
        status["distill_loss_attached"] = float(teacher_lp is not None)
        status["distill_loss_kl"] = distill_kl.item()
    if knobs.sdar_lambda != 0.0:
        status["sdar_lambda"] = knobs.sdar_lambda
        status["sdar_attached"] = float(teacher_lp is not None)
        # ends in `_loss`, so reduce_metrics SUMS it over micro-batches and DP ranks (like policy_loss)
        status["sdar_loss"] = sdar_loss.item()
        status["sdar_gate_active"] = sdar_gate_active.item()
        status["sdar_delta_mean"] = sdar_delta_mean.item()
    return distill_loss_term, sdar_loss_term, status

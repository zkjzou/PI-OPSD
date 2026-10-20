"""Tests for SDAR (arXiv:2605.15155) — the gated auxiliary self-distillation loss.

SDAR is the beta loss path with a per-token sigmoid gate:

    Delta_t = sg(log pi_T(y_t|s_t^+) - log pi_S(y_t|s_t))
    g_t     = sigmoid(beta * score_t)          score = Delta_t for the default `gap` gate
    l_t     = g_t * (log pi_T(y_t) - log pi_S(y_t))
    L       = L_GRPO + lam_SDAR * Agg(l_t)

The gate is the method. On ALFWorld the paper measures the mean teacher-student gap as consistently
NEGATIVE (their Fig. 5) — the privileged teacher assigns LOWER probability than the student to sampled
tokens on average — so ungated distillation spends most of its pressure pulling the student toward a
teacher that disagrees. That is the multi-turn instability that collapses naive OPSD, and it is also why
our own live-teacher arms died. These tests pin the asymmetry that fixes it, plus the two wiring facts
whose absence would leave a silently ungated (or silently teacher-free) run that still trains.

The maths is the real implementation (opsd.algorithms.distill_loss); the wiring is checked by running the
OPSD policy worker (opsd.skyrl_ext.policy_worker) on the CPU stub harness of
opsd/skyrl_ext/tests/test_policy_worker.py.
"""

import math
import re
from pathlib import Path

import pytest
import torch

import opsd
from opsd.algorithms.distill_loss import LossKnobs, normalise_sdar_gate, sdar_gate, sdar_term
from opsd.skyrl_ext.policy_worker import OPSDFSDPPolicyWorkerBase
from opsd.skyrl_ext.tests.test_policy_worker import (
    load_inputs,
    make_batch,
    make_cfg,
    make_worker,
    run_forward_backward,
)

TRAINER = Path(opsd.__file__).parent / "trainer/main_opsd_alfworld.py"
SDAR_KEYS = ("sdar_lambda", "sdar_attached", "sdar_loss", "sdar_gate_active", "sdar_delta_mean")

pytestmark = pytest.mark.filterwarnings("ignore:CUDA is not available or torch_xla is imported")


def sdar_terms(teacher_lp, student_lp, mask, beta, gate="gap", entropy=None):
    """(delta, g, loss) from the real implementation."""
    _, delta, g = sdar_gate(student_lp, teacher_lp, beta, gate, entropy)
    loss, _, _ = sdar_term(student_lp, teacher_lp, mask, beta, gate, entropy)
    return delta, g, loss


def run_worker(knobs, teacher="teacher", **algo):
    inputs = load_inputs()
    cfg = make_cfg({"use_kl_loss": True, "use_entropy_loss": True, **algo})
    worker = make_worker(OPSDFSDPPolicyWorkerBase, cfg, inputs["W0"], knobs=knobs)
    return run_forward_backward(worker, make_batch(inputs, teacher))


@pytest.fixture
def toks():
    """Token 0: teacher STRONGLY endorses. Token 1: teacher strongly rejects. Token 2: agreement."""
    student = torch.tensor([[math.log(0.10), math.log(0.60), math.log(0.30)]], requires_grad=False)
    teacher = torch.tensor([[math.log(0.60), math.log(0.10), math.log(0.30)]])
    mask = torch.ones(1, 3)
    return teacher, student, mask


# --------------------------------------------------------------------------- wiring (behavioural)
def test_worker_adds_sdar_to_the_loss():
    """The SDAR term is ADDED to the loss beside the KL and entropy terms: with everything else equal, the loss
    handed to backward grows by exactly lam * sdar_loss * microbatch_weight on every micro-batch."""
    base = run_worker(LossKnobs())
    sdar = run_worker(LossKnobs(sdar_lambda=0.5))
    for b, s, m in zip(base["backward_losses"], sdar["backward_losses"], sdar["micro"]):
        expected = 0.5 * m["status"]["sdar_loss"] * m["microbatch_weight"]
        assert s - b == pytest.approx(expected, rel=1e-4, abs=1e-6)
    # the existing terms are still there, unchanged
    for mb, ms in zip(base["micro"], sdar["micro"]):
        assert ms["status"]["policy_kl"] == mb["status"]["policy_kl"]
        assert ms["status"]["policy_entropy"] == mb["status"]["policy_entropy"]
    assert not torch.equal(base["grad_W"], sdar["grad_W"]), "SDAR must change the gradient"


def test_worker_detaches_the_gate_but_not_the_student(toks):
    """Gradient of Agg(g_t*(logT - logS)) w.r.t. logS is exactly -g_t/n per token: the gate is a constant (sg),
    while the student log-prob keeps its gradient."""
    teacher, student, mask = toks
    s = student.clone().requires_grad_(True)
    loss, _, _ = sdar_term(s, teacher, mask, 5.0)
    loss.backward()
    _, _, g = sdar_gate(student, teacher, 5.0)
    assert torch.allclose(s.grad, -g * mask / mask.sum(), atol=1e-7)
    ent = torch.rand(1, 3).requires_grad_(True)
    s2 = student.clone().requires_grad_(True)
    sdar_term(s2, teacher, mask, 5.0, "soft_or", ent)[0].backward()
    assert ent.grad is None, "the entropy gates read entropy detached"


def test_worker_logs_the_attachment_gate():
    """Logged when SDAR is CONFIGURED, and gated on lambda -- never on the attachment itself.

    These were unconditional, which put five constant-zero `policy/sdar_*` series on every non-SDAR run.
    Gating on lambda != 0 removes that noise while keeping the one ambiguity the logging exists to resolve:
    with lambda set but the teacher tensor missing, the block still emits `sdar_attached=0`.

    Gating on `sdar_attached` (or directly on the tensor) instead would be the silent-no-op shape that hid the
    overlong buffer and the detached distill_log_probs on this project -- an arm would train as plain GRPO
    with no metric saying so. Pin that it does not happen.
    """
    missing = run_worker(LossKnobs(sdar_lambda=0.5), teacher=None)
    for m in missing["micro"]:
        assert all(k in m["status"] for k in SDAR_KEYS)
        assert m["status"]["sdar_attached"] == 0.0 and m["status"]["sdar_loss"] == 0.0
    present = run_worker(LossKnobs(sdar_lambda=0.5))
    assert all(m["status"]["sdar_attached"] == 1.0 for m in present["micro"])
    off = run_worker(LossKnobs(distill_coef=0.1))  # teacher present, SDAR off
    assert not any(k in m["status"] for m in off["micro"] for k in SDAR_KEYS)


def test_trainer_transports_teacher_logprobs_for_sdar():
    src = TRAINER.read_text()
    assert (
        "OPSD_SDAR_LAMBDA" in src and 'training_input["distill_log_probs"]' in src
    ), "SDAR needs distill_log_probs plumbed to the worker"


def test_sdar_is_off_by_default():
    assert LossKnobs.from_env({}).sdar_lambda == 0.0
    base = run_worker(LossKnobs.from_env({}))
    assert not any(k in m["status"] for m in base["micro"] for k in SDAR_KEYS)


def test_transport_attaches_without_a_coefficient_gate():
    """`policy/sdar_attached` is the gate we tell people to trust, so the thing it measures has to be
    unconditional. If the transport were ever gated on OPSD_DISTILL_LOSS_COEF (the beta knob), an SDAR arm
    would run with lam set, attached=0, and train as plain GRPO on panels that look completely normal. That
    failure has happened on this path before, so pin it: SDAR alone (beta = 0) receives the tensor.
    """
    result = run_worker(LossKnobs(distill_coef=0.0, sdar_lambda=0.5))
    assert [m["status"]["sdar_attached"] for m in result["micro"]] == [1.0, 1.0, 1.0]
    assert not any("distill_loss_attached" in m["status"] for m in result["micro"])


def test_trainer_gate_covers_both_consumers():
    """The trainer only ships distill_log_probs when a consumer wants it; SDAR is a second consumer, so
    the gate must be an OR rather than the original beta-only condition."""
    src = TRAINER.read_text()
    i = src.index('training_input["distill_log_probs"]')
    block = src[max(0, i - 400) : i]
    assert (
        "OPSD_DISTILL_LOSS_COEF" in block and "OPSD_SDAR_LAMBDA" in block
    ), "both beta and SDAR must be able to request the teacher tensor"
    assert " or " in block, "the two conditions must be OR'd, not one replacing the other"


# --------------------------------------------------------------------------- the gate
def test_gate_is_asymmetric_between_endorsement_and_rejection(toks):
    teacher, student, mask = toks
    delta, g, _ = sdar_terms(teacher, student, mask, beta=5.0)
    assert delta[0, 0] > 0 and delta[0, 1] < 0 and abs(delta[0, 2]) < 1e-6
    # endorsed token amplified, rejected token attenuated, agreement sits at the neutral 0.5
    assert g[0, 0] > 0.9, f"teacher-endorsed token should be strongly gated on, got {g[0,0]:.3f}"
    assert g[0, 1] < 0.1, f"teacher-rejected token should be attenuated, got {g[0,1]:.3f}"
    assert g[0, 2] == pytest.approx(0.5, abs=1e-6)


def test_beta_zero_is_the_no_gate_ablation(toks):
    """beta=0 => g_t == 0.5 everywhere, i.e. uniform (ungated) distillation at half weight."""
    teacher, student, mask = toks
    _, g, _ = sdar_terms(teacher, student, mask, beta=0.0)
    assert torch.allclose(g, torch.full_like(g, 0.5))


def test_larger_beta_sharpens_toward_a_hard_gate(toks):
    teacher, student, mask = toks
    _, g5, _ = sdar_terms(teacher, student, mask, beta=5.0)
    _, g50, _ = sdar_terms(teacher, student, mask, beta=50.0)
    assert g50[0, 0] > g5[0, 0] and g50[0, 1] < g5[0, 1]


def test_gradient_raises_likelihood_of_endorsed_tokens_more_than_rejected(toks):
    """The mechanism, checked on the actual gradient rather than the formula.

    Minimizing lam*Agg(g_t*(logT - logS)) moves logS UP weighted by g_t, so the endorsed token must
    receive substantially more upward pressure than the rejected one.
    """
    teacher, student, mask = toks
    s = student.clone().requires_grad_(True)
    _, _, loss = sdar_terms(teacher, s, mask, beta=5.0)
    loss.backward()
    grad = s.grad[0]
    assert grad[0] < 0 and grad[1] < 0  # both push logS up (loss decreases)
    assert abs(grad[0]) > 5 * abs(
        grad[1]
    ), f"endorsed token must dominate the update, got {grad[0]:.4f} vs {grad[1]:.4f}"


def test_negative_mean_gap_is_attenuated_overall():
    """The paper's ALFWorld regime: most tokens have Delta<0. Total gate mass must stay well under
    the ungated 0.5/token, which is exactly what 'softly attenuating negative rejections' means."""
    torch.manual_seed(0)
    student = torch.log(torch.rand(4, 32) * 0.4 + 0.3)
    teacher = student - torch.rand(4, 32) * 0.5  # teacher systematically LOWER => Delta<0
    mask = torch.ones(4, 32)
    delta, g, _ = sdar_terms(teacher, student, mask, beta=5.0)
    assert delta.mean() < 0
    assert g.mean() < 0.5, "a systematically disagreeing teacher must be attenuated, not followed"


def test_entropy_and_soft_or_gates_are_available(toks):
    teacher, student, mask = toks
    ent = torch.rand(1, 3)
    for kind in ("entropy", "soft_or"):
        _, g, loss = sdar_terms(teacher, student, mask, beta=5.0, gate=kind, entropy=ent)
        assert torch.all((g > 0) & (g < 1)) and torch.isfinite(loss)


def test_worker_supports_all_three_gates_and_rejects_typos():
    losses = {}
    for kind in ("gap", "entropy", "soft_or"):
        result = run_worker(LossKnobs(sdar_lambda=0.5, sdar_gate=kind))
        losses[kind] = result["metrics"]["sdar_loss"]
        assert math.isfinite(losses[kind])
    assert len(set(losses.values())) == 3, f"each gate must change the loss: {losses}"
    for alias in ("softor", "soft-or", " SOFT_OR "):
        assert normalise_sdar_gate(alias) == "soft_or"
    with pytest.raises(ValueError, match=re.escape("OPSD_SDAR_GATE must be gap|entropy|soft_or")):
        LossKnobs.from_env({"OPSD_SDAR_LAMBDA": "0.5", "OPSD_SDAR_GATE": "gapp"})


# --------------------------------------------------------------------------- robustness
def test_nan_teacher_degrades_to_agreement(toks):
    teacher, student, mask = toks
    teacher = teacher.clone()
    teacher[0, 0] = float("nan")
    teacher[0, 1] = float("inf")
    delta, g, loss = sdar_terms(teacher, student, mask, beta=5.0)
    assert torch.isfinite(delta).all() and torch.isfinite(g).all() and torch.isfinite(loss)
    assert delta[0, 0] == pytest.approx(0.0, abs=1e-6), "NaN teacher must read as agreement, not a huge gap"

"""Tests for opsd.algorithms.distill_loss: the knob parse and the pure beta / SDAR maths.

The end-to-end worker behaviour (where the terms enter the loss, chunk alignment, fork parity) is tested in
opsd/skyrl_ext/tests/; this file pins the pieces those tests build on.
"""

import pickle
import re

import pytest
import torch

from opsd.algorithms.distill_loss import (
    SDAR_GATES,
    LossKnobs,
    distill_kl_term,
    normalise_sdar_gate,
    opsd_loss_terms,
    sdar_gate,
    sdar_term,
)
from skyrl.backends.skyrl_train.utils.ppo_utils import compute_approx_kl
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(7)
    student = torch.log(torch.rand(3, 5, generator=g) * 0.8 + 0.1)
    teacher = student + 0.4 * torch.randn(3, 5, generator=g)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float32)
    entropy = torch.rand(3, 5, generator=g)
    return student, teacher, mask, entropy


# --------------------------------------------------------------------------- knobs
def test_knobs_default_off():
    assert LossKnobs() == LossKnobs.from_env({})
    assert not LossKnobs().active
    assert (LossKnobs().distill_coef, LossKnobs().sdar_lambda, LossKnobs().sdar_beta, LossKnobs().sdar_gate) == (
        0.0,
        0.0,
        5.0,
        "gap",
    )


@pytest.mark.parametrize(
    "env,expected",
    [
        ({"OPSD_DISTILL_LOSS_COEF": "0.1"}, LossKnobs(distill_coef=0.1)),
        ({"OPSD_DISTILL_LOSS_COEF": ""}, LossKnobs()),  # the fork's `float(v or 0)`
        ({"OPSD_DISTILL_LOSS_COEF": "1e-4"}, LossKnobs(distill_coef=1e-4)),
        ({"OPSD_SDAR_LAMBDA": "0.5"}, LossKnobs(sdar_lambda=0.5)),
        ({"OPSD_SDAR_LAMBDA": "0.5", "OPSD_SDAR_BETA": ""}, LossKnobs(sdar_lambda=0.5, sdar_beta=0.0)),
        ({"OPSD_SDAR_LAMBDA": "0.5", "OPSD_SDAR_GATE": " Soft-Or "}, LossKnobs(sdar_lambda=0.5, sdar_gate="soft_or")),
        # with lambda == 0 the fork never read BETA or GATE, so junk there is harmless
        ({"OPSD_SDAR_BETA": "junk", "OPSD_SDAR_GATE": "bogus"}, LossKnobs()),
        ({"OPSD_SDAR_LAMBDA": "0", "OPSD_SDAR_GATE": "bogus", "OPSD_DISTILL_LOSS_COEF": "0.1"}, LossKnobs(0.1)),
    ],
)
def test_from_env_parses_like_the_fork(env, expected):
    assert LossKnobs.from_env(env) == expected


@pytest.mark.parametrize(
    "env",
    [
        {"OPSD_DISTILL_LOSS_COEF": "abc"},
        {"OPSD_SDAR_LAMBDA": "x"},
        {"OPSD_SDAR_LAMBDA": "0.5", "OPSD_SDAR_BETA": "five"},
        {"OPSD_SDAR_LAMBDA": "0.5", "OPSD_SDAR_GATE": "gapp"},
    ],
)
def test_from_env_rejects_typos_at_startup(env):
    with pytest.raises(ValueError):
        LossKnobs.from_env(env)


def test_from_env_reads_os_environ_by_default(monkeypatch):
    monkeypatch.setenv("OPSD_DISTILL_LOSS_COEF", "0.25")
    monkeypatch.delenv("OPSD_SDAR_LAMBDA", raising=False)
    assert LossKnobs.from_env() == LossKnobs(distill_coef=0.25)


@pytest.mark.parametrize("knobs", [LossKnobs(distill_coef=0.1), LossKnobs(sdar_lambda=0.5, sdar_gate="entropy")])
def test_active(knobs):
    assert knobs.active


def test_knobs_survive_pickling():
    """They travel to the actors through Ray's pass_through dispatch."""
    k = LossKnobs(distill_coef=0.1, sdar_lambda=0.5, sdar_beta=2.0, sdar_gate="soft_or")
    assert pickle.loads(pickle.dumps(k)) == k


def test_knobs_are_frozen():
    with pytest.raises(Exception):
        LossKnobs().distill_coef = 1.0


@pytest.mark.parametrize(
    "raw,canon",
    [("gap", "gap"), ("ENTROPY", "entropy"), ("soft_or", "soft_or"), ("softor", "soft_or"), (" soft-or ", "soft_or")],
)
def test_gate_aliases(raw, canon):
    assert normalise_sdar_gate(raw) == canon and canon in SDAR_GATES


def test_unknown_gate_raises_the_forks_message():
    with pytest.raises(ValueError, match=re.escape("OPSD_SDAR_GATE must be gap|entropy|soft_or, got 'nope'")):
        normalise_sdar_gate(" NOPE ")


# --------------------------------------------------------------------------- beta
@pytest.mark.parametrize("estimator", ["k1", "k2", "k3", "abs"])
def test_distill_kl_is_skyrls_kl_reduction(batch, estimator):
    student, teacher, mask, _ = batch
    expected = masked_mean(
        compute_approx_kl(student, teacher, loss_mask=mask, kl_estimator_type=estimator), mask, dim=-1
    )
    assert torch.equal(distill_kl_term(student, teacher, mask, estimator), expected.mean())


def test_distill_kl_gradient_is_pathwise_through_the_student(batch):
    student, teacher, mask, _ = batch
    s = student.clone().requires_grad_(True)
    distill_kl_term(s, teacher, mask, "k3").backward()
    assert s.grad[0].abs().sum() > 0 and s.grad[1, :2].abs().sum() > 0
    assert torch.all(s.grad[mask == 0] == 0), "masked tokens must get no gradient"


def test_distill_kl_is_zero_when_teacher_equals_student(batch):
    student, _, mask, _ = batch
    assert distill_kl_term(student, student.clone(), mask, "k3").item() == 0.0


# --------------------------------------------------------------------------- SDAR
def test_sdar_gate_is_sigmoid_of_the_detached_score(batch):
    student, teacher, _, entropy = batch
    _, delta, g = sdar_gate(student, teacher, 5.0, "gap")
    assert torch.equal(delta, (teacher - student).clamp(-20, 20))
    assert torch.equal(g, torch.sigmoid(5.0 * delta))
    _, _, g_ent = sdar_gate(student, teacher, 5.0, "entropy", entropy)
    assert torch.equal(g_ent, torch.sigmoid(5.0 * entropy))
    _, _, g_or = sdar_gate(student, teacher, 5.0, "soft_or", entropy)
    assert torch.equal(g_or, torch.sigmoid(5.0 * (1.0 - (1.0 - entropy) * (1.0 - delta))))


def test_sdar_term_reduction(batch):
    student, teacher, mask, _ = batch
    loss, gate_active, delta_mean = sdar_term(student, teacher, mask, 5.0)
    g = torch.sigmoid(5.0 * (teacher - student))
    assert torch.allclose(loss, masked_mean(g * (teacher - student), mask, dim=-1).mean())
    assert torch.allclose(gate_active, masked_mean((g > 0.5).float(), mask, dim=-1).mean())
    assert torch.allclose(delta_mean, masked_mean(teacher - student, mask, dim=-1).mean())


def test_sdar_gap_is_clamped(batch):
    student, _, mask, _ = batch
    _, delta, _ = sdar_gate(student, student + 100.0, 1.0)
    assert torch.all(delta == 20.0)


# --------------------------------------------------------------------------- the per-micro-batch assembly
def test_terms_off_are_exact_zeros_and_log_nothing(batch):
    student, teacher, mask, entropy = batch
    d, s, status = opsd_loss_terms(LossKnobs(), student, teacher, mask, entropy, "k3")
    assert d.item() == 0.0 and s.item() == 0.0 and status == {}


def test_beta_only_logs_beta_keys(batch):
    student, teacher, mask, entropy = batch
    d, s, status = opsd_loss_terms(LossKnobs(distill_coef=0.1), student, teacher, mask, entropy, "k3")
    kl = distill_kl_term(student, teacher, mask, "k3")
    assert torch.equal(d, kl * 0.1) and s.item() == 0.0
    assert status == {"distill_loss_coef": 0.1, "distill_loss_attached": 1.0, "distill_loss_kl": kl.item()}


def test_sdar_only_logs_sdar_keys(batch):
    student, teacher, mask, entropy = batch
    knobs = LossKnobs(sdar_lambda=0.5, sdar_gate="soft_or")
    d, s, status = opsd_loss_terms(knobs, student, teacher, mask, entropy, "k3")
    loss, active, dm = sdar_term(student, teacher, mask, 5.0, "soft_or", entropy)
    assert d.item() == 0.0 and torch.equal(s, loss * 0.5)
    assert status == {
        "sdar_lambda": 0.5,
        "sdar_attached": 1.0,
        "sdar_loss": loss.item(),
        "sdar_gate_active": active.item(),
        "sdar_delta_mean": dm.item(),
    }


def test_missing_teacher_degrades_to_zero_but_still_reports(batch):
    """Gated on the REQUEST, not on the tensor: attached=0 is what tells a broken arm from a working one."""
    student, _, mask, entropy = batch
    d, s, status = opsd_loss_terms(LossKnobs(0.1, 0.5), student, None, mask, entropy, "k3")
    assert d.item() == 0.0 and s.item() == 0.0
    assert status["distill_loss_attached"] == 0.0 and status["sdar_attached"] == 0.0
    assert status["distill_loss_kl"] == 0.0 and status["sdar_loss"] == 0.0

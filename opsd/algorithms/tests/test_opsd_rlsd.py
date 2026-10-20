"""Tests for RLSD (arXiv:2604.03128) as ported into the OPSD advantage path.

RLSD is a MULTIPLICATIVE credit redistribution: A_hat_t = A * ((1-lam) + lam*clip(w_t)), with
w_t = exp(sign(A) * (log P_T - log P_S)). The properties worth pinning are structural, because they are
the whole argument for preferring it to the two injection paths we have already run:

  1. SIGN PRESERVATION. ((1-lam) + lam*w) > 0 always, so the environment reward keeps exclusive authority
     over which tokens are reinforced vs penalised. Our convex OPSD_MIX_LAMBDA path can and does flip
     signs; that is what "the teacher REPLACES task signal" means.
  2. DIRECTION AWARENESS. For A>0 the teacher's favoured tokens get MORE credit; for A<0 they get LESS
     blame. Getting the sign(A) exponent backwards would silently invert credit on every failed
     trajectory -- a bug that would still train, just wrongly, so it needs a test rather than a reading.
  3. lam=0 is EXACTLY plain GRPO, so an RLSD arm after its decay window is a GRPO arm and any residual
     difference is path dependence, not mechanism.
"""

import importlib.util
import math
import os

import pytest
import torch

from opsd._repo import repo_root

REPO = repo_root(__file__)
MODULE = REPO / "opsd/trainer/rlsd.py"
TRAINER = REPO / "opsd/trainer/main_opsd_alfworld.py"


def _load():
    spec = importlib.util.spec_from_file_location("opsd_rlsd", MODULE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rlsd = _load()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("OPSD_RLSD", "OPSD_RLSD_LAMBDA", "OPSD_RLSD_LAMBDA_FINAL", "OPSD_RLSD_DECAY_STEPS", "OPSD_RLSD_CLIP"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def batch():
    """2 sequences x 6 tokens: seq 0 has a POSITIVE advantage, seq 1 NEGATIVE."""
    torch.manual_seed(0)
    a = torch.tensor([[1.5] * 6, [-2.0] * 6])
    student = torch.log(torch.rand(2, 6) * 0.5 + 1e-2)
    teacher = torch.log(torch.rand(2, 6) * 0.5 + 1e-2)
    mask = torch.ones(2, 6)
    return a, student, teacher, mask


# --------------------------------------------------------------------------- defaults are OFF
def test_disabled_by_default():
    assert rlsd.rlsd_enabled() is False


def test_trainer_gate_is_mutually_exclusive_with_mix_lambda():
    src = TRAINER.read_text()
    assert "if rlsd_enabled() and teacher_lp is not None:" in src
    assert "both rewrite the advantage; pick one" in src, "RLSD + OPSD_MIX_LAMBDA<1 must raise"


def test_trainer_transports_teacher_logprobs_for_rlsd():
    """The invariant is that RLSD can REQUEST the teacher tensor, not the exact spelling of the
    condition: OPSD_TOKEN_ADV later joined the same OR and broke an order-sensitive substring match
    while changing nothing about RLSD. Match the guard expression instead."""
    src = TRAINER.read_text()
    i = src.index('training_input["mix_teacher_log_probs"]')
    guard = src[max(0, i - 400) : i]
    assert "rlsd_enabled()" in guard, "RLSD needs mix_teacher_log_probs plumbed through"
    # the lambda path requests it through the schedule-aware helper (mix_schedule.py), not a raw env read
    assert "mix_lambda_configured()" in guard, "the lambda path must still request it too"


# --------------------------------------------------------------------------- lambda schedule
def test_lambda_defaults_match_the_paper():
    assert rlsd.rlsd_lambda(0) == pytest.approx(0.5)  # initialized at 0.5
    assert rlsd.rlsd_lambda(50) == pytest.approx(0.0)  # linearly decayed to 0 over 50 steps
    assert rlsd.rlsd_lambda(25) == pytest.approx(0.25)
    assert rlsd.rlsd_lambda(500) == pytest.approx(0.0)  # clamped, never negative


def test_lambda_constant_when_decay_disabled():
    os.environ["OPSD_RLSD_DECAY_STEPS"] = "0"
    os.environ["OPSD_RLSD_LAMBDA"] = "0.3"
    assert rlsd.rlsd_lambda(0) == pytest.approx(0.3)
    assert rlsd.rlsd_lambda(999) == pytest.approx(0.3)


def test_lambda_can_ramp_up_for_the_other_reading():
    os.environ["OPSD_RLSD_LAMBDA"] = "0.0"
    os.environ["OPSD_RLSD_LAMBDA_FINAL"] = "0.5"
    assert rlsd.rlsd_lambda(0) == pytest.approx(0.0)
    assert rlsd.rlsd_lambda(50) == pytest.approx(0.5)


@pytest.mark.parametrize("bad", ["1.5", "-0.2"])
def test_lambda_range_validated(bad):
    os.environ["OPSD_RLSD_LAMBDA"] = bad
    with pytest.raises(ValueError, match="OPSD_RLSD_LAMBDA"):
        rlsd.rlsd_lambda(0)


@pytest.mark.parametrize("bad", ["0", "1.0", "-0.1"])
def test_clip_range_validated(bad):
    os.environ["OPSD_RLSD_CLIP"] = bad
    with pytest.raises(ValueError, match="OPSD_RLSD_CLIP"):
        rlsd.rlsd_clip()


# --------------------------------------------------------------------------- inert-teacher guard
def test_inert_guard_off_when_rlsd_is_off():
    """Every other mechanism owns the teacher's cadence; RLSD must not speak for them."""
    assert rlsd.rlsd_teacher_is_inert(999) is False


def test_inert_after_lambda_decays_to_zero():
    os.environ["OPSD_RLSD"] = "1"
    assert rlsd.rlsd_teacher_is_inert(49) is False, "still live one step before the decay completes"
    assert rlsd.rlsd_teacher_is_inert(50) is True
    assert rlsd.rlsd_teacher_is_inert(150) is True


def test_ramp_up_schedule_is_never_treated_as_inert():
    """TRAP 1: a ramp-up starts at exactly lambda=0. Skipping there would freeze the teacher at
    theta_0 through the warm-up and hand the mechanism a stale teacher the moment it switches on."""
    os.environ["OPSD_RLSD"] = "1"
    os.environ["OPSD_RLSD_LAMBDA"] = "0.0"
    os.environ["OPSD_RLSD_LAMBDA_FINAL"] = "0.5"
    assert rlsd.rlsd_lambda(0) == 0.0, "precondition: this schedule really does start at 0"
    for step in (0, 10, 50, 150):
        assert rlsd.rlsd_teacher_is_inert(step) is False


def test_constant_nonzero_lambda_is_never_inert():
    os.environ["OPSD_RLSD"] = "1"
    os.environ["OPSD_RLSD_DECAY_STEPS"] = "0"
    os.environ["OPSD_RLSD_LAMBDA"] = "0.5"
    assert rlsd.rlsd_teacher_is_inert(10_000) is False


@pytest.mark.parametrize(
    "var,val",
    [
        ("OPSD_DISTILL_LOSS_COEF", "0.25"),  # beta loss path
        ("OPSD_SDAR_LAMBDA", "0.01"),  # SDAR gated loss
        ("OPSD_MIX_LAMBDA", "0.9"),  # convex advantage mix
    ],
)
def test_other_consumers_keep_the_teacher_alive(var, val):
    """TRAP 2: RLSD is not the only reader of the teacher forward. Freezing it under a co-enabled
    beta/SDAR/mix arm would silently change THAT mechanism's algorithm."""
    os.environ["OPSD_RLSD"] = "1"
    os.environ[var] = val
    try:
        assert rlsd.rlsd_teacher_is_inert(150) is False
    finally:
        os.environ.pop(var, None)


def test_reward_path_keeps_the_teacher_alive():
    os.environ["OPSD_RLSD"] = "1"
    assert rlsd.rlsd_teacher_is_inert(150, reward_path_active=True) is False


def test_trainer_wires_the_guard_with_the_reward_path():
    src = TRAINER.read_text()
    assert "rlsd_teacher_is_inert(" in src, "the guard must actually be called"
    assert (
        "reward_path_active=self.cfg.trainer.algorithm.use_kl_in_reward" in src
    ), "the reward path is a cfg flag the helper cannot see; it must be passed in"
    assert 'self.all_metrics["distill/ema_skipped_inert"]' in src, "a skipped sync must be observable"


# --------------------------------------------------------------------------- core invariants
def test_lambda_zero_is_exactly_plain_grpo(batch):
    a, s, t, m = batch
    out, _ = rlsd.rlsd_advantage(a, s, t, m, m, lam=0.0, eps_w=0.2)
    assert torch.allclose(out, a * m, atol=0, rtol=0)


def test_sign_is_never_flipped(batch):
    """The whole point: the verifier owns direction, the teacher owns magnitude only."""
    a, s, t, m = batch
    for lam in (0.1, 0.5, 1.0):
        out, met = rlsd.rlsd_advantage(a, s, t, m, m, lam=lam, eps_w=0.2)
        assert torch.all(torch.sign(out) == torch.sign(a))
        assert met["rlsd/sign_preserved"] == pytest.approx(1.0)


def test_magnitude_stays_within_the_clip_band(batch):
    a, s, t, m = batch
    eps, lam = 0.2, 1.0
    out, _ = rlsd.rlsd_advantage(a, s, t, m, m, lam=lam, eps_w=eps)
    ratio = (out / a).abs()
    assert torch.all(ratio <= 1.0 + eps + 1e-6) and torch.all(ratio >= 1.0 - eps - 1e-6)


def test_direction_aware_reweighting():
    """A>0: teacher-favoured tokens gain credit. A<0: teacher-favoured tokens lose BLAME.

    Token 0 is strongly teacher-favoured (Delta>0), token 1 strongly disfavoured (Delta<0).
    """
    student = torch.tensor([[math.log(0.10), math.log(0.50)]])
    teacher = torch.tensor([[math.log(0.50), math.log(0.10)]])
    mask = torch.ones(1, 2)

    pos, _ = rlsd.rlsd_advantage(torch.full((1, 2), 1.0), student, teacher, mask, mask, 1.0, 0.2)
    assert pos[0, 0] > pos[0, 1], "A>0: favoured token must receive MORE positive credit"
    assert pos[0, 0] == pytest.approx(1.2, abs=1e-5)  # clipped to 1+eps
    assert pos[0, 1] == pytest.approx(0.8, abs=1e-5)  # clipped to 1-eps

    neg, _ = rlsd.rlsd_advantage(torch.full((1, 2), -1.0), student, teacher, mask, mask, 1.0, 0.2)
    # magnitudes invert: the favoured token now carries the SMALLER penalty
    assert neg[0, 0].abs() < neg[0, 1].abs(), "A<0: favoured token must bear LESS blame"
    assert neg[0, 0] == pytest.approx(-0.8, abs=1e-5)
    assert neg[0, 1] == pytest.approx(-1.2, abs=1e-5)


def test_zero_advantage_stays_zero(batch):
    _, s, t, m = batch
    a = torch.zeros(2, 6)
    out, _ = rlsd.rlsd_advantage(a, s, t, m, m, lam=1.0, eps_w=0.2)
    assert torch.all(out == 0.0)


def test_identical_teacher_and_student_is_a_no_op(batch):
    """Delta=0 => w=1 => A_hat=A. A hint that changes nothing must change nothing."""
    a, s, _, m = batch
    out, met = rlsd.rlsd_advantage(a, s, s.clone(), m, m, lam=1.0, eps_w=0.2)
    assert torch.allclose(out, a * m, atol=1e-6)
    assert met["rlsd/w_mean"] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- robustness
def test_nan_and_inf_logprobs_do_not_poison_the_batch(batch):
    a, s, t, m = batch
    t = t.clone()
    t[0, 0], t[0, 1], t[1, 0] = float("nan"), float("inf"), float("-inf")
    out, met = rlsd.rlsd_advantage(a, s, t, m, m, lam=1.0, eps_w=0.2)
    assert torch.isfinite(out).all()
    assert torch.isfinite(torch.tensor(list(met.values()))).all()
    assert out[0, 0] == pytest.approx(a[0, 0].item(), abs=1e-6)  # NaN -> Delta 0 -> w=1


def test_masked_positions_fall_back_to_uniform(batch):
    a, s, t, m = batch
    loss_mask = m.clone()
    loss_mask[:, 3:] = 0.0
    out, _ = rlsd.rlsd_advantage(a, s, t, loss_mask, m, lam=1.0, eps_w=0.2)
    assert torch.allclose(out[:, 3:], a[:, 3:], atol=1e-6), "off-mask tokens must keep the uniform A"


def test_response_mask_zeroes_padding(batch):
    a, s, t, m = batch
    response_mask = m.clone()
    response_mask[:, 4:] = 0.0
    out, _ = rlsd.rlsd_advantage(a, s, t, m, response_mask, lam=1.0, eps_w=0.2)
    assert torch.all(out[:, 4:] == 0.0)


def test_clip_ratio_metrics_track_the_papers_health_check(batch):
    """Figure 5(c): clipping should be ACTIVE but bounded (they report ~3-6%)."""
    a, s, t, m = batch
    _, met = rlsd.rlsd_advantage(a, s, t, m, m, lam=1.0, eps_w=0.2)
    total = met["rlsd/clip_low_ratio"] + met["rlsd/clip_high_ratio"]
    assert 0.0 <= total <= 1.0
    # a tiny eps_w must clip nearly everything; a huge one nearly nothing
    _, tight = rlsd.rlsd_advantage(a, s, t, m, m, lam=1.0, eps_w=1e-6)
    _, loose = rlsd.rlsd_advantage(a, s, t, m, m, lam=1.0, eps_w=0.99)
    assert tight["rlsd/clip_low_ratio"] + tight["rlsd/clip_high_ratio"] > total
    assert loose["rlsd/clip_low_ratio"] + loose["rlsd/clip_high_ratio"] < total

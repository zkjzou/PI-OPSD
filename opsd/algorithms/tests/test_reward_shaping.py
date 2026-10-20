"""Numerical parity tests for opsd.algorithms.reward_shaping.

Reproduces the ORIGINAL inline k1 reward computation from the OPD and OPSD trainers and asserts the
shared helper produces bit-identical per_token_kl and identical diagnostics on random tensors that
include NaN / +-inf / masked positions — the guarantee that the Phase-4 extraction is a pure refactor
(so no GPU training run is needed to prove correctness).
"""

import pytest
import torch

from opsd.algorithms import pi_utility
from opsd.algorithms.reward_shaping import (
    KL_CLAMP,
    apply_side_mask,
    kl_shaping_metrics,
    per_token_reverse_kl,
)


# --- ORIGINAL inline computations (verbatim from the pre-refactor trainers) ---
def _orig_opd_per_token_kl(action_lp, teacher_lp, loss_mask, kl_clamp):
    valid = loss_mask.bool()
    raw_diff = action_lp - teacher_lp
    diff = torch.nan_to_num(raw_diff, nan=0.0, posinf=kl_clamp, neginf=-kl_clamp)
    diff = diff.clamp(-kl_clamp, kl_clamp)
    return torch.where(valid, diff * loss_mask, torch.zeros_like(diff))


def _orig_opsd_per_token_kl(student_lp, teacher_lp, loss_mask, kl_clamp, token_clip):
    valid = loss_mask.bool()
    raw_diff = student_lp - teacher_lp
    diff = torch.nan_to_num(raw_diff, nan=0.0, posinf=kl_clamp, neginf=-kl_clamp)
    diff = diff.clamp(-kl_clamp, kl_clamp)
    if token_clip > 0:
        diff = diff.clamp(-token_clip, token_clip)
    return torch.where(valid, diff * loss_mask, torch.zeros_like(diff))


def _make_batch(seed, B=4, L=7):
    g = torch.Generator().manual_seed(seed)
    student = torch.randn(B, L, generator=g) * 3.0
    teacher = torch.randn(B, L, generator=g) * 3.0
    # inject NaN / +-inf into a few positions to exercise the sanitize path
    student[0, 1] = float("nan")
    teacher[1, 2] = float("-inf")  # student - teacher -> +inf
    teacher[2, 0] = float("inf")  # student - teacher -> -inf
    loss_mask = (torch.rand(B, L, generator=g) > 0.4).float()
    loss_mask[3] = 0.0  # a fully-masked rollout
    return student, teacher, loss_mask


def test_per_token_kl_parity_opd():
    for seed in range(20):
        s, t, lm = _make_batch(seed)
        ours, _ = per_token_reverse_kl(s, t, lm, kl_clamp=KL_CLAMP)
        ref = _orig_opd_per_token_kl(s, t, lm, KL_CLAMP)
        assert torch.equal(ours, ref), seed


def test_per_token_kl_parity_opsd_token_clip():
    for seed in range(20):
        s, t, lm = _make_batch(seed)
        for token_clip in (0.5, 1.0):
            ours, _ = per_token_reverse_kl(s, t, lm, kl_clamp=KL_CLAMP, token_clip=token_clip)
            ref = _orig_opsd_per_token_kl(s, t, lm, KL_CLAMP, token_clip)
            assert torch.equal(ours, ref), (seed, token_clip)


def test_token_clip_zero_or_none_is_noop():
    s, t, lm = _make_batch(0)
    base, _ = per_token_reverse_kl(s, t, lm, kl_clamp=KL_CLAMP)
    for tc in (None, 0, 0.0):
        out, _ = per_token_reverse_kl(s, t, lm, kl_clamp=KL_CLAMP, token_clip=tc)
        assert torch.equal(out, base)


def test_masked_positions_are_zero_and_finite():
    s, t, lm = _make_batch(1)
    ptk, _ = per_token_reverse_kl(s, t, lm, kl_clamp=KL_CLAMP)
    assert torch.isfinite(ptk).all()  # no NaN/inf leaks into the reward
    assert torch.equal(ptk[lm == 0], torch.zeros_like(ptk[lm == 0]))


def test_metrics_parity():
    s, t, lm = _make_batch(5)
    kl_clamp = KL_CLAMP
    ptk, raw = per_token_reverse_kl(s, t, lm, kl_clamp=kl_clamp)
    m = kl_shaping_metrics(raw, ptk, lm, student_lp=s, teacher_lp=t, lp_detail="full")

    valid = lm.bool()
    # teacher_kl = mean per-token over the loss mask
    assert m["train/teacher_kl"] == (ptk.sum() / lm.sum().clamp(min=1)).item()
    # bad_valid = count of NaN/inf raw diffs inside the response
    exp_bad = float(((torch.isnan(raw) | torch.isinf(raw)) & valid).sum().item())
    assert m["train/teacher_kl_bad_valid"] == exp_bad
    # full per-tensor diagnostics
    for nm, tensor in (("student", s), ("teacher", t)):
        tv = tensor[valid]
        assert m[f"distill/{nm}_lp_nan"] == float(torch.isnan(tv).sum().item())
        assert m[f"distill/{nm}_lp_inf"] == float(torch.isinf(tv).sum().item())


def test_metrics_lp_detail_modes():
    s, t, lm = _make_batch(2)
    ptk, raw = per_token_reverse_kl(s, t, lm)
    nan_only = kl_shaping_metrics(raw, ptk, lm, student_lp=s, teacher_lp=t, lp_detail="nan")
    assert "distill/student_lp_nan" in nan_only
    assert "distill/student_lp_inf" not in nan_only  # OPSD mode: nan only
    none_mode = kl_shaping_metrics(raw, ptk, lm, lp_detail=None)
    assert set(none_mode) == {"train/teacher_kl", "train/teacher_kl_bad_valid"}


# --- d+/d- side ablation (OPSD_DISTILL_SIDE) -----------------------------------------------------------
def test_side_both_is_a_noop():
    """`both` must be bit-identical to no ablation, so the default arm is provably unchanged."""
    for seed in range(10):
        s, t, lm = _make_batch(seed)
        ptk, _ = per_token_reverse_kl(s, t, lm)
        out, m = apply_side_mask(ptk, lm, "both")
        assert out is ptk or torch.equal(out, ptk)
        assert m["distill/side_code"] == 0.0
        assert m["distill/side_scale"] == 1.0


def test_side_signs_match_the_reward_direction():
    """The trap: `plus` keeps NEGATIVE per_token_kl (teacher likes the token more -> reward > 0 -> up)."""
    ptk = torch.tensor([[-2.0, 3.0, 0.0, -0.5]])
    lm = torch.ones_like(ptk)
    plus, _ = apply_side_mask(ptk, lm, "plus")
    minus, _ = apply_side_mask(ptk, lm, "minus")
    assert torch.equal(plus, torch.tensor([[-2.0, 0.0, 0.0, -0.5]]))
    assert torch.equal(minus, torch.tensor([[0.0, 3.0, 0.0, 0.0]]))
    # reward = -coef * term, so plus-only rewards are >= 0 and minus-only <= 0 everywhere
    assert (-plus >= 0).all() and (-minus <= 0).all()


def test_side_halves_partition_the_full_term():
    """plus + minus == both, elementwise. The one check that catches a swapped clamp bound."""
    for seed in range(10):
        s, t, lm = _make_batch(seed)
        ptk, _ = per_token_reverse_kl(s, t, lm)
        plus, _ = apply_side_mask(ptk, lm, "plus")
        minus, _ = apply_side_mask(ptk, lm, "minus")
        assert torch.allclose(plus + minus, ptk, atol=1e-6), seed


def test_side_mean_abs_equals_pi_utility_d_plus_d_minus():
    """mean|plus-only term| must equal pi_utility/d_plus (and minus -> d_minus) at token_clip=0.

    This is what lets the ablation be SIZED off panels already logged: the d+ share measured on the live
    arms (8-23%) is exactly the factor by which a raw plus-only arm loses gradient norm.
    """
    for seed in range(8):
        s, t, lm = _make_batch(seed)
        ptk, _ = per_token_reverse_kl(s, t, lm)
        _, mp = apply_side_mask(ptk, lm, "plus")
        _, mm = apply_side_mask(ptk, lm, "minus")
        attrs = pi_utility.divergence_attributes(s, t, lm)
        # pi_utility averages per trajectory then over trajectories; here weight by each row's valid count
        n = lm.sum(dim=-1)
        tot = n.sum().clamp(min=1)
        d_plus = float((attrs["d_plus"] * n).sum() / tot)
        d_minus = float((attrs["d_minus"] * n).sum() / tot)
        assert mp["distill/side_mean_abs"] == pytest.approx(d_plus, abs=1e-6), seed
        assert mm["distill/side_mean_abs"] == pytest.approx(d_minus, abs=1e-6), seed


def test_magnitude_match_restores_the_both_sided_norm():
    for seed in range(10):
        s, t, lm = _make_batch(seed)
        ptk, _ = per_token_reverse_kl(s, t, lm)
        for side in ("plus", "minus"):
            out, m = apply_side_mask(ptk, lm, side, match_magnitude=True)
            got = float((out.abs() * lm).sum() / lm.sum().clamp(min=1))
            assert got == pytest.approx(m["distill/full_mean_abs"], rel=1e-5), (seed, side)
            assert m["distill/side_scale"] > 1.0  # a side is always <= the whole
            assert m["distill/side_match_on"] == 1.0


def test_magnitude_match_does_not_change_which_tokens_survive():
    """Rescaling is dose, not direction: the support and the signs must be untouched."""
    s, t, lm = _make_batch(3)
    ptk, _ = per_token_reverse_kl(s, t, lm)
    for side in ("plus", "minus"):
        raw, _ = apply_side_mask(ptk, lm, side)
        scaled, m = apply_side_mask(ptk, lm, side, match_magnitude=True)
        assert torch.equal((raw != 0), (scaled != 0))
        assert torch.equal(torch.sign(raw), torch.sign(scaled))
        assert torch.allclose(scaled, raw * m["distill/side_scale"], atol=1e-6)


def test_side_mask_keeps_masked_positions_zero():
    s, t, lm = _make_batch(1)
    ptk, _ = per_token_reverse_kl(s, t, lm)
    for side in ("plus", "minus"):
        for match in (False, True):
            out, _ = apply_side_mask(ptk, lm, side, match_magnitude=match)
            assert torch.isfinite(out).all()
            assert torch.equal(out[lm == 0], torch.zeros_like(out[lm == 0]))


def test_all_one_side_batch_does_not_divide_by_zero():
    """A batch with NO surviving tokens gives scale 1.0 (= no pressure this step), never inf/NaN."""
    ptk = torch.tensor([[1.0, 2.0, 3.0]])  # all positive -> plus side is empty
    lm = torch.ones_like(ptk)
    out, m = apply_side_mask(ptk, lm, "plus", match_magnitude=True)
    assert torch.equal(out, torch.zeros_like(ptk))
    assert m["distill/side_scale"] == 1.0
    assert m["distill/side_frac_tokens"] == 0.0


def test_side_frac_tokens_counts_surviving_masked_tokens():
    ptk = torch.tensor([[-1.0, 2.0, -3.0, 4.0]])
    lm = torch.tensor([[1.0, 1.0, 1.0, 0.0]])  # the last (positive) token is masked out
    _, mp = apply_side_mask(ptk, lm, "plus")
    _, mm = apply_side_mask(ptk, lm, "minus")
    assert mp["distill/side_frac_tokens"] == pytest.approx(2 / 3)
    assert mm["distill/side_frac_tokens"] == pytest.approx(1 / 3)


def test_unknown_side_raises():
    s, t, lm = _make_batch(0)
    ptk, _ = per_token_reverse_kl(s, t, lm)
    with pytest.raises(ValueError, match="OPSD_DISTILL_SIDE"):
        apply_side_mask(ptk, lm, "positive")

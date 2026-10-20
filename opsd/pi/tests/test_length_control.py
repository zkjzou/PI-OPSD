"""Tests for opsd/pi/signals/length_control.py.

Two are load-bearing. ``test_monotone_..._annihilated`` pins the claim that motivates the whole design
-- the double rank transform already removes ALL monotone length dependence, so degree-1 was never the
weak point it looks like -- and ``test_non_monotone_...`` pins the one shape it cannot remove, an
asymmetric U, which is the only reason the spline is the default. If the second ever fails, the basis
is broken and the upgrade buys nothing over the straight line it replaced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opsd.pi.signals.length_control import (
    CALIPERS,
    LADDER,
    matched_auc,
    ndtri,
    ns_basis,
    parse_method,
    residualise,
    restrict_common,
)
from opsd.pi.signals.metrics import roc_auc

POOLED = [m for m in LADDER if m.endswith("-pool")]
ALL_FITS = [m for m in LADDER if m != "none"]


def _corpus(n_tasks, n_roll, attr_fn, *, win_fn=None, p_win=0.4, seed=0):
    """attr_fn(length_rank_pct, rng) -> attribute. Lengths are lognormal (~10x spread within task).

    ``win_fn(length_rank_pct, rng)`` defaults to an outcome INDEPENDENT of length; pass one to make
    success length-driven, which is what turns "did the control remove the confound" into a test.
    """
    rng = np.random.default_rng(seed)
    rows, toks = [], []
    for t in range(n_tasks):
        ln = np.exp(rng.normal(8.0, 0.8, n_roll))
        lr = pd.Series(ln).rank(pct=True).to_numpy()
        v = np.broadcast_to(np.asarray(attr_fn(lr, rng), float), (n_roll,))
        y = (rng.random(n_roll) < p_win).astype(int) if win_fn is None else np.asarray(win_fn(lr, rng), int)
        for j in range(n_roll):
            uid = f"t{t}_r{j}"
            rows.append(dict(uid=uid, task_id=f"t{t}", success=int(y[j]), v=float(v[j])))
            toks.append(dict(uid=uid, task_id=f"t{t}", success=int(y[j]), ntok=float(ln[j])))
    return pd.DataFrame(rows), pd.DataFrame(toks)


def _macro_auc(r):
    a = [roc_auc(list(g.v), list(g.success)) for _, g in r.groupby("task_id") if g.success.nunique() > 1]
    a = [v for v in a if v == v]
    return float(np.mean(a)) if a else float("nan")


# --------------------------------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------------------------------
def test_parse_method_round_trip_and_rejections():
    assert parse_method("ns12-task") == ("ns", 12, "task")
    assert parse_method("lin-pool") == ("lin", 2, "pool")
    for bad in ("ns2-pool", "spline", "lin", "ns5", "isotonic-pool", "lin-global"):
        with pytest.raises(ValueError):
            parse_method(bad)


def test_ndtri_matches_known_quantiles():
    got = ndtri([0.001, 0.025, 0.5, 0.975, 0.999])
    want = [-3.0902323062, -1.9599639845, 0.0, 1.9599639845, 3.0902323062]
    assert np.allclose(got, want, atol=2e-9)


def test_ns_basis_df_and_linear_fallback():
    x = np.linspace(0, 1, 50)
    assert ns_basis(x, np.linspace(0, 1, 7)).shape == (50, 7)
    assert ns_basis(x, np.array([0.3, 0.7])).shape == (50, 2)  # <3 knots -> linear, no crash


# --------------------------------------------------------------------------------------------------
# (1) monotone length dependence -- the rank transform alone annihilates it
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("g", [np.log, np.sqrt, lambda L: L**3, lambda L: 1.0 - 1.0 / L])
def test_monotone_attribute_of_length_is_annihilated_by_every_method(g):
    """A = g(length) with g strictly monotone -- saturating, exploding, whatever -- leaves NOTHING to
    rank, because rank(A) == rank(L) exactly and every method here fits x as a basis column.

    The assertion is on the residual MAGNITUDE, not on AUC: once the residual is numerically zero its
    AUC is decided by 1e-17 rounding noise, which is meaningless in either direction.
    """
    rows, toks = _corpus(24, 64, lambda lr, _r: g(np.exp(8 + lr)), seed=1)
    for m in ALL_FITS:
        res = residualise(rows, toks, m)
        assert not res.empty, m
        assert np.abs(res["v"]).max() < 1e-9, f"{m}: residual survives a monotone length restatement"


def test_monotone_attribute_is_also_chance_under_caliper_matching():
    """Same corpus, no fit at all: among rollouts of near-identical length, a pure length restatement
    cannot order wins above losses."""
    rows, toks = _corpus(40, 64, lambda lr, _r: np.log1p(lr), seed=11)
    _, means = restrict_common({h: matched_auc(rows, toks, h) for h in CALIPERS})
    for h, v in means.items():
        assert abs(v - 0.5) < 0.03, f"caliper {h}: {v:.4f}"


# --------------------------------------------------------------------------------------------------
# (2) independence -- the control must be nearly free
# --------------------------------------------------------------------------------------------------
def test_independent_attribute_survives_the_control():
    rows, toks = _corpus(40, 64, lambda lr, rng: rng.normal(size=len(lr)), seed=2)
    raw = _macro_auc(rows)
    for m in POOLED:
        rep: dict = {}
        auc = _macro_auc(residualise(rows, toks, m, report=rep))
        assert abs(rep["r2"]) < 0.02, f"{m}: fitted R^2 {rep['r2']:.4f} where length is irrelevant"
        assert abs(auc - raw) < 0.02, f"{m}: moved AUC {raw:.4f} -> {auc:.4f} with no confound present"
        assert abs(rep["resid_len_rho"]) < 0.02, f"{m}: residual still correlates with length"


# --------------------------------------------------------------------------------------------------
# (3) the asymmetric U -- the ONLY case where the spline beats the straight line
# --------------------------------------------------------------------------------------------------
def test_non_monotone_length_relation_defeats_the_line_but_not_the_spline():
    """A = (rank(L) - 0.3)^2, success = (rank(L) > 0.5): the outcome is 100% length-driven, so EVERY
    scrap of residual signal is uncontrolled confound and the correct answer is exactly chance.

    rank(A) is not monotone in rank(L), so this is the shape the double rank transform cannot absorb.
    Measured: raw 0.985 -> lin-pool 0.685 (fails), vdw-pool 0.702 (fails), ns5 0.546, ns8 0.484,
    ns20 0.500. Flexibility is what closes the gap, and it closes it monotonically in df.
    """
    rows, toks = _corpus(30, 64, lambda lr, _r: (lr - 0.30) ** 2, win_fn=lambda lr, _r: lr > 0.5, seed=3)
    assert _macro_auc(rows) > 0.95, "the fixture itself should be an almost-pure length confound"

    auc = {
        m: _macro_auc(residualise(rows, toks, m)) for m in ("lin-pool", "vdw-pool", "ns5-pool", "ns8-pool", "ns20-pool")
    }
    assert auc["lin-pool"] > 0.65, f"a straight line should FAIL to remove a U (got {auc['lin-pool']:.4f})"
    assert auc["vdw-pool"] > 0.65, f"normal scores are still degree 1 (got {auc['vdw-pool']:.4f})"
    assert abs(auc["ns5-pool"] - 0.5) < 0.06, f"ns5 should mostly remove it (got {auc['ns5-pool']:.4f})"
    assert abs(auc["ns20-pool"] - 0.5) <= abs(auc["ns5-pool"] - 0.5), "more df should not do worse here"


# --------------------------------------------------------------------------------------------------
# (4) degenerate inputs: no crash, no NaN leak, no invented signal
# --------------------------------------------------------------------------------------------------
def test_no_length_spread_leaves_the_ordering_untouched():
    """All rollouts the same length => no confound CAN exist => the pooled residual must preserve the
    within-task ordering exactly. The per-task fits are undefined there and correctly drop the task."""
    rows, toks = _corpus(6, 32, lambda lr, rng: rng.normal(size=len(lr)), seed=4)
    toks = toks.assign(ntok=1000.0)
    raw = _macro_auc(rows)
    for m in POOLED:
        assert abs(_macro_auc(residualise(rows, toks, m)) - raw) < 1e-9, m
    assert residualise(rows, toks, "lin-task").empty


def test_constant_attribute_does_not_become_a_length_signal():
    """THE GUARD THAT THIS TEST FOUND. Subtracting f(length) from a constant yields -f(length), so
    without a zero-variance check the control converts a completely uninformative attribute into a pure
    length predictor -- measured at residual AUC 0.25 (i.e. 0.75 flipped) before the fix."""
    rows, toks = _corpus(20, 32, lambda lr, _r: 7.0, win_fn=lambda lr, _r: lr > 0.5, seed=5)
    assert abs(_macro_auc(rows) - 0.5) < 1e-12
    for m in ALL_FITS:
        out = residualise(rows, toks, m)
        if out.empty:  # per-task fits legitimately drop these tasks
            continue
        assert not out["v"].isna().any(), m
        assert out.groupby("task_id")["v"].nunique().max() == 1, f"{m} invented within-task variation"


def test_tiny_tasks_are_safe():
    rows, toks = _corpus(5, 2, lambda lr, rng: rng.normal(size=len(lr)), seed=6)
    for m in ALL_FITS:
        out = residualise(rows, toks, m)
        assert out.empty or not out["v"].isna().any(), m


def test_missing_length_rows_are_dropped_not_imputed():
    rows, toks = _corpus(4, 16, lambda lr, rng: rng.normal(size=len(lr)), seed=7)
    toks.loc[toks.index[:5], "ntok"] = np.nan
    out = residualise(rows, toks, "ns5-pool")
    assert len(out) == len(rows) - 5 and not out["v"].isna().any()


def test_missing_ntok_column_fails_loudly():
    rows, toks = _corpus(3, 8, lambda lr, rng: rng.normal(size=len(lr)), seed=8)
    with pytest.raises(SystemExit):
        residualise(rows, toks.drop(columns=["ntok"]), "ns5-pool")


def test_report_records_df_and_attrition():
    rows, toks = _corpus(10, 64, lambda lr, rng: rng.normal(size=len(lr)), seed=9)
    rep: dict = {}
    residualise(rows, toks, "ns8-pool", report=rep)
    assert rep["df"] == 8 and rep["n"] == 640 and rep["tasks_in"] == rep["tasks_out"] == 10
    assert rep["df_over_n"] == pytest.approx(8 / 640)


# --------------------------------------------------------------------------------------------------
# caliper matching identities
# --------------------------------------------------------------------------------------------------
def test_caliper_one_reproduces_plain_within_task_auc():
    """h=1.0 admits every pair, so it must equal metrics.roc_auc EXACTLY -- the identity that makes the
    caliper column readable on the same scale as the AUC column beside it."""
    rows, toks = _corpus(20, 48, lambda lr, rng: rng.normal(size=len(lr)), seed=10)
    got = matched_auc(rows, toks, 1.0)
    for t, g in rows.groupby("task_id"):
        if g.success.nunique() < 2:
            continue
        assert got[t] == pytest.approx(roc_auc(list(g.v), list(g.success)), abs=1e-12), t


def test_restrict_common_uses_one_population_across_calipers():
    rows, toks = _corpus(12, 40, lambda lr, rng: rng.normal(size=len(lr)), seed=12)
    per = {h: matched_auc(rows, toks, h) for h in (1.0, *CALIPERS)}
    common, means = restrict_common(per)
    assert common and all(common <= set(d) for d in per.values())
    assert all(v == v for v in means.values())


def test_df_over_n_uses_actual_task_pool_size():
    """df/n for a per-TASK fit must track the real pool size, not a hardcoded 128.

    This is the diagnostic the whole pool-vs-task argument rests on, and it was constant-128, so it
    read 0.016 on a 16-rollout subsample whose true ratio is 0.125. Measured consequence of that 8x
    understatement: on ALFWorld-30bi `lin-task` falls 0.7223 -> 0.6666 between the n=128 pool and a
    min(8,wins)+fill-16 subsample while `lin-pool` moves only 0.7765 -> 0.7562, i.e. the per-task fit
    over-corrects exactly where df/n says it should and the stale denominator hid it.
    """
    rng = np.random.default_rng(0)
    for n in (16, 32, 128):
        tid = np.repeat(np.arange(40), n)
        tok = pd.DataFrame({"uid": np.arange(len(tid)), "task_id": tid, "ntok": rng.uniform(100, 9000, len(tid))})
        r = pd.DataFrame(
            {
                "uid": np.arange(len(tid)),
                "task_id": tid,
                "success": rng.integers(0, 2, len(tid)).astype(float),
                "v": rng.normal(size=len(tid)),
            }
        )
        rep: dict = {}
        residualise(r, tok, "lin-task", report=rep)
        assert rep["df_over_n"] == pytest.approx(2.0 / n), (n, rep["df_over_n"])
    # pooled scope stays df / TOTAL rows -- unchanged by this fix
    rep = {}
    residualise(r, tok, "ns5-pool", report=rep)
    assert rep["df_over_n"] == pytest.approx(rep["df"] / rep["n"])

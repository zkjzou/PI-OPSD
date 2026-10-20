"""Length controls for within-task attribute ranking: residualisation, and caliper matching.

WHY THIS MODULE EXISTS. Trajectory length is the strongest single predictor of success in 5 of the 9
pass@128 cells, with a benchmark-dependent SIGN, so any attribute correlated with length inherits a
spurious ranking signal. Every table in the PI offline screen therefore needs a length control, and
until now the same 15-line degree-1 routine was pasted into three analysis scripts.

TWO CONTROLS, BECAUSE THEY ANSWER DIFFERENT QUESTIONS AND DISAGREE.

  RESIDUALISATION  regress the attribute's within-task rank on the within-task LENGTH rank, keep the
                   residual, then rank the residual. Produces a SCORE, so it feeds every downstream
                   metric (nDCG@k, best@k, hit@k, BoN@N) -- but it is only as good as the fitted f.
  CALIPER MATCHING compare wins against losses ONLY among rollouts of near-identical length
                   (|dx| <= h on the length-rank scale). Assumption-free: no f is fitted at all. But
                   it yields a pairwise concordance, not a score, so it can only replace AUC.

Report both. Where they disagree the residualisation is doing something the matching is not.

WHAT THE DOUBLE RANK TRANSFORM ALREADY BUYS -- AND WHY THE FUNCTIONAL FORM BARELY MATTERS.
``rank(A)`` vs ``rank(L)`` is invariant to ANY monotone reparametrisation of either variable, so a
saturating relation ``A = g(L)`` with g monotone maps to an exactly STRAIGHT rank-rank line. Degree-1
is thus not the restriction it looks like: measured over all 8 (cell x attribute) combinations of the
pass@128 grid, going from degree 1 to a 20-df natural cubic spline buys +0.002 to +0.014 pooled R^2
and moves residual AUC by <= 0.015, and the format ordering raw > raw_no_think > actions is IDENTICAL
for lin-pool / ns5 / ns8 / ns20 in 8 of 8 cells. What survives the rank transform is the mild S-curve
of a Gaussian copula -- confirmed by ``vdw-pool``, a 2-parameter fit in van-der-Waerden normal scores,
which matched or beat the 20-df spline on R^2 in 4 of 8 cells.

The spline is still the default because it makes that a MEASURED statement rather than an assumption,
and because it also removes the one shape the rank transform cannot: a NON-monotone (U / inverted-U)
length relation, which is exactly what SS19's inverted-U rank curve looked like.

POOLED, NOT PER-TASK -- this is the bigger lever, worth 3-5x the functional form. A per-task fit spends
2 (or df) parameters on ~128 points and absorbs task-specific noise along with the length effect; it
moved residual AUC by up to 0.054 versus the pooled fit, e.g. ALFWorld-30bi `d` 0.7223 -> 0.7765. The
pooled fit costs ZERO per-task degrees of freedom: because x and y are already within-task ranks, the
pooled sample is the empirical COPULA of (length, attribute) and f is its conditional-mean curve, which
is a property of the (cell, PI) and is estimated from ~19,000 pairs instead of 128.

OVER-CORRECTION IS THE REAL HAZARD. Every added degree of freedom strictly removes signal (ns20 <= ns8
<= ns5 <= lin in residual AUC on 6 of 8 cells), and if length is GENUINELY predictive of success then
residualising deletes real predictive power. A residualised column is only readable beside the
un-residualised one and beside the ``ntok`` control row.

numpy is used freely here. ``metrics.py`` next door is deliberately pure-Python so the minimal
HF-scorer path can import it -- it must NOT import this module.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from opsd.pi.signals.metrics import spearman

# --------------------------------------------------------------------------------------------------
# method names
# --------------------------------------------------------------------------------------------------
#   none                 no control (raw attribute)
#   lin-task             degree 1, refitted INSIDE each task -- the historical routine, kept as the
#                        bit-for-bit regression anchor
#   lin-pool             degree 1, one fit per (cell, PI) over all within-task rank pairs
#   vdw-pool             degree 1 in van-der-Waerden normal scores; exact under a Gaussian copula
#   ns<df>-pool          natural cubic spline, <df> df, pooled          <-- DEFAULT is ns5-pool
#   ns<df>-task          natural cubic spline, <df> df, refitted per task (diagnostic; needs n >= 3*df)
_NS_RE = re.compile(r"^ns(\d+)-(pool|task)$")
DEFAULT_METHOD = "ns5-pool"
LADDER = ("none", "lin-task", "lin-pool", "vdw-pool", "ns5-pool", "ns8-pool", "ns20-pool")
CALIPERS = (0.20, 0.10, 0.05)


def parse_method(method: str) -> Tuple[str, int, str]:
    """('none'|'lin'|'vdw'|'ns', df, 'pool'|'task'). Raises on anything unrecognised."""
    if method == "none":
        return "none", 0, "pool"
    if method in ("lin-pool", "lin-task"):
        return "lin", 2, method.split("-")[1]
    if method == "vdw-pool":
        return "vdw", 2, "pool"
    m = _NS_RE.match(method)
    if m:
        df = int(m.group(1))
        if df < 3:
            raise ValueError(f"{method}: a natural spline needs df >= 3 (df=2 is 'lin-{m.group(2)}')")
        return "ns", df, m.group(2)
    raise ValueError(f"unknown --length-control {method!r}; try one of {LADDER} or ns<df>-{{pool,task}}")


# --------------------------------------------------------------------------------------------------
# bases
# --------------------------------------------------------------------------------------------------
def ns_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis (Hastie/Tibshirani/Friedman eq. 5.4-5.5). df == len(knots).

    Natural (linear beyond the boundary knots) rather than an unconstrained cubic spline: the tails of
    the length-rank distribution are exactly where an unconstrained cubic swings hardest, and a wild
    tail fit would be subtracted straight out of the residual of the longest and shortest trajectories
    -- the rollouts the ranking metrics care most about.
    """
    xi = np.asarray(knots, float)
    K = len(xi)
    if K < 3:  # not enough distinct knots for a spline term
        return np.column_stack([np.ones_like(x), x])

    def d(j: int) -> np.ndarray:
        return ((np.clip(x - xi[j], 0, None) ** 3) - (np.clip(x - xi[-1], 0, None) ** 3)) / (xi[-1] - xi[j])

    return np.column_stack([np.ones_like(x), x] + [d(j) - d(K - 2) for j in range(K - 2)])


def ndtri(p) -> np.ndarray:
    """Inverse standard-normal CDF (Acklam's rational approximation, |err| < 1.2e-9).

    Hand-rolled because this venv has numpy + pandas only -- no scipy, no statsmodels, no sklearn.
    """
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    p = np.asarray(p, float)
    q = np.empty_like(p)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mi = ~(lo | hi)
    if lo.any():
        r = np.sqrt(-2 * np.log(p[lo]))
        q[lo] = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / (
            (((dd[0] * r + dd[1]) * r + dd[2]) * r + dd[3]) * r + 1
        )
    if hi.any():
        r = np.sqrt(-2 * np.log(1 - p[hi]))
        q[hi] = -(
            (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5])
            / ((((dd[0] * r + dd[1]) * r + dd[2]) * r + dd[3]) * r + 1)
        )
    if mi.any():
        r = p[mi] - 0.5
        s = r * r
        q[mi] = (
            (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5])
            * r
            / (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1)
        )
    return q


def _lstsq(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yh = X @ beta
    den = ((y - y.mean()) ** 2).sum()
    return yh, (1.0 - ((y - yh) ** 2).sum() / den if den > 0 else float("nan"))


# --------------------------------------------------------------------------------------------------
# length attachment + within-task ranks
# --------------------------------------------------------------------------------------------------
def attach_length(r: pd.DataFrame, tok: pd.DataFrame, col: str = "ntok") -> pd.DataFrame:
    """Map uid -> trajectory length. ``ntok`` is PI-INVARIANT (a hint changes the conditioning, never
    the response), so the same length is attached under all six PIs by construction."""
    if col not in tok.columns:
        raise SystemExit(f"length control needs a `{col}` column in the rollout-attrs table")
    ln = tok.drop_duplicates("uid").set_index("uid")[col].astype(float)
    out = r.copy()
    out["_len"] = out["uid"].map(ln)
    return out


def add_rank_pairs(r: pd.DataFrame) -> pd.DataFrame:
    """x = within-task rank_pct of length, y = within-task rank_pct of the attribute.

    A task with NO length spread gets a constant x, so f(x) is constant and the residual is y shifted
    -- ordering preserved, residual == raw. That is the correct answer, not a degenerate one: a task
    whose rollouts are all the same length cannot carry a length confound. Such tasks are therefore
    kept under the pooled methods (the per-task methods must drop them: their fit is undefined).
    """
    # RESET THE INDEX FIRST. Callers pool several cells with pd.concat, so the incoming index has
    # DUPLICATE labels; any label-based alignment or label->position map then silently collapses them.
    # That bug dropped 4,680 of 18,688 AppWorld rows before it was caught by comparing against the
    # routine this module replaced. Everything below is positional from here on.
    out = r.dropna(subset=["_len", "v"]).reset_index(drop=True)
    g = out.groupby("task_id")
    out["x"] = g["_len"].rank(pct=True).to_numpy()
    out["y"] = g["v"].rank(pct=True).to_numpy()
    n = g["v"].transform("size").to_numpy()
    out["xn"] = ndtri((g["_len"].rank().to_numpy() / (n + 1)))
    out["yn"] = ndtri((g["v"].rank().to_numpy() / (n + 1)))
    return out


# --------------------------------------------------------------------------------------------------
# the control itself
# --------------------------------------------------------------------------------------------------
def residualise(
    r: pd.DataFrame, tok: pd.DataFrame, method: str = DEFAULT_METHOD, report: dict | None = None
) -> pd.DataFrame:
    """Replace ``v`` with the length-controlled attribute. Returns a frame with the same schema.

    Rows/tasks that cannot be controlled are DROPPED, never faked with a zero residual.
    ``report`` (if given) is filled in place with the fit diagnostics -- df, df/n, R^2, the
    residual-vs-length Spearman that proves the control worked, and the task attrition.
    """
    kind, df, scope = parse_method(method)
    if kind == "none":
        if report is not None:
            report.update(
                method=method,
                df=0,
                n=len(r),
                tasks_in=r.task_id.nunique(),
                tasks_out=r.task_id.nunique(),
                r2=float("nan"),
                resid_len_rho=float("nan"),
            )
        return r

    rr = add_rank_pairs(attach_length(r, tok))
    tasks_in = rr.task_id.nunique()
    if rr.empty:
        if report is not None:
            report.update(
                method=method, df=df, n=0, tasks_in=tasks_in, tasks_out=0, r2=float("nan"), resid_len_rho=float("nan")
            )
        return rr.drop(columns=[c for c in ("_len", "x", "y", "xn", "yn") if c in rr])

    # A task whose attribute is CONSTANT carries zero ranking information, and subtracting f(length)
    # from a constant turns it into -f(length): the control would MANUFACTURE a pure length signal out
    # of an attribute that had none (measured: residual AUC 0.25 on a constant attribute before this
    # guard). Such tasks keep a constant residual, i.e. chance.
    flat = rr.groupby("task_id")["v"].transform("nunique").to_numpy() < 2

    eff_df, r2 = df, float("nan")
    if scope == "pool":
        x, y = rr["x"].to_numpy(), rr["y"].to_numpy()
        if kind == "lin":
            yh, r2 = _lstsq(np.column_stack([np.ones_like(x), x]), y)
            rr["v"] = y - yh
        elif kind == "vdw":
            xn, yn = rr["xn"].to_numpy(), rr["yn"].to_numpy()
            yh, r2 = _lstsq(np.column_stack([np.ones_like(xn), xn]), yn)
            rr["v"] = yn - yh
        else:
            knots = np.unique(np.quantile(x, np.linspace(0, 1, df)))
            eff_df = max(len(knots), 2)  # dedup can collapse knots on tied `ntok`
            yh, r2 = _lstsq(ns_basis(x, knots), y)
            rr["v"] = y - yh
        rr.loc[flat, "v"] = 0.0
    else:  # per-task fit
        min_n = 4 if kind == "lin" else max(4, 3 * df)
        res = np.full(len(rr), np.nan)
        r2s = []
        for _, g in rr.groupby("task_id"):
            idx = g.index.to_numpy()  # == positions: add_rank_pairs reset the index
            gx, gy = g["x"].to_numpy(), g["y"].to_numpy()
            if len(g) < min_n or gx.std() == 0 or gy.std() == 0:
                continue  # undefined fit -> drop the task
            if kind == "lin":
                yh = np.polyval(np.polyfit(gx, gy, 1), gx)
            else:
                knots = np.unique(np.quantile(gx, np.linspace(0, 1, df)))
                yh, _ = _lstsq(ns_basis(gx, knots), gy)
            res[idx] = gy - yh
            den = ((gy - gy.mean()) ** 2).sum()
            if den > 0:
                r2s.append(1.0 - ((gy - yh) ** 2).sum() / den)
        rr["v"] = res
        r2 = float(np.nanmean(r2s)) if r2s else float("nan")
        rr = rr.dropna(subset=["v"])

    if report is not None:
        # df/n for a per-TASK fit must use the ACTUAL per-task pool size, not a constant. This was
        # hardcoded to 128 -- correct for the full pass@128 pools and silently 8x optimistic the moment
        # a caller subsamples (min(8,wins)+fill-16 makes it 2/16=0.125, not 2/128=0.016). df/n is the
        # entire argument for pooling f, so a stale denominator understates the one risk it measures.
        median_task_n = float(np.median(rr.groupby("task_id").size().to_numpy())) or 1.0
        # Both correlations are reported against the WITHIN-TASK LENGTH RANK -- the variable f was
        # actually fitted on -- not against raw `ntok` (against raw length the figure is dominated by
        # BETWEEN-task variation, which no within-task control targets, and reads as a failed control).
        #
        # THEY SAY DIFFERENT THINGS, AND THE GAP IS THE HONEST LIMIT OF RESIDUALISATION.
        # Pearson is ~1e-16 for every least-squares rung -- exact, by the normal equations, since x is a
        # basis column. Spearman is NOT: measured 0.06-0.10 pooled (0.06-0.13 within task) on
        # AppWorld-4bi dtopk, and it does not shrink with df (ns5 0.0998, ns20 0.0994). Least squares
        # removes the conditional MEAN in the fitted basis; a monotone RANK association survives it, and
        # every downstream metric here is rank-based. No residualisation rung fixes this -- which is
        # precisely why the caliper columns are first-class rather than a diagnostic: matching to within
        # h on the length rank bounds the residual length association by construction instead of fitting
        # it away.
        rho = spearman(list(rr["v"]), list(rr["x"])) if len(rr) > 1 else float("nan")
        pe = float(np.corrcoef(rr["v"].to_numpy(float), rr["x"].to_numpy(float))[0, 1]) if len(rr) > 1 else float("nan")
        report.update(
            method=method,
            df=eff_df,
            n=len(rr),
            tasks_in=tasks_in,
            tasks_out=rr.task_id.nunique(),
            r2=r2,
            resid_len_rho=rho,
            resid_len_pearson=pe,
            df_over_n=(eff_df / len(rr) if scope == "pool" else eff_df / float(median_task_n)),
        )
    return rr.drop(columns=[c for c in ("_len", "x", "y", "xn", "yn") if c in rr])


# --------------------------------------------------------------------------------------------------
# caliper matching -- the assumption-free control
# --------------------------------------------------------------------------------------------------
def matched_auc(r: pd.DataFrame, tok: pd.DataFrame, caliper: float) -> Dict[object, float]:
    """Per-task concordance over (win, loss) pairs whose LENGTH RANKS differ by <= ``caliper``.

    ``caliper=1.0`` admits every pair and reproduces the plain within-task AUC exactly (the pairwise
    Mann-Whitney definition ``metrics.roc_auc`` computes), which is the identity the tests pin.
    Ties in the attribute count 0.5, matching ``roc_auc`` and unlike the top-k family.

    Returns {task_id: auc} rather than a mean: the caller must intersect the task sets across calipers
    before averaging, or a tighter caliper silently reports a different (easier) population.
    """
    rr = add_rank_pairs(attach_length(r, tok))
    out: Dict[object, float] = {}
    for t, g in rr.groupby("task_id"):
        y = g["success"].to_numpy(float)
        if len(np.unique(y)) < 2:
            continue
        s, x = g["v"].to_numpy(float), g["x"].to_numpy(float)
        P, N = np.where(y == 1)[0], np.where(y == 0)[0]
        ok = np.abs(x[P][:, None] - x[N][None, :]) <= caliper
        if not ok.any():
            continue  # no length-comparable pair: task is unscoreable
        ds = s[P][:, None] - s[N][None, :]
        out[t] = float((((ds > 0).astype(float) + 0.5 * (ds == 0)))[ok].mean())
    return out


def restrict_common(per_caliper: Dict[float, Dict[object, float]]) -> Tuple[set, Dict[float, float]]:
    """Intersect the scoreable task sets across calipers, then macro-average on that one population."""
    if not per_caliper:
        return set(), {}
    common = set.intersection(*[set(d) for d in per_caliper.values()])
    means = {h: (float(np.mean([d[t] for t in common])) if common else float("nan")) for h, d in per_caliper.items()}
    return common, means

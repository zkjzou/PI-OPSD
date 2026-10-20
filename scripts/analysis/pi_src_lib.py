"""
src_lib.py -- canonical, benchmark-agnostic implementation of SRC (Step-Ranking Concentration).

Implements SPEC.md sections "The metric" and "Controls" exactly:

  1. per-step value   x[pi,u,p] = sum_ch(a * n_tokens) / sum_ch(n_tokens)      (token-wtd mean over channels)
  2. LOO baseline     b[pi,t,p] = mean over siblings u'!=u of task t that reach step p of x[pi,u',p]
                                  require >= min_sibs (5) contributing siblings, else DROP that step
  3. residual         r[pi,u,p] = x[pi,u,p] - b[pi,t,p]
  4. labels           E_u = set of error steps of rollout u (1-indexed, arm-invariant)
  5. within-rollout   A[pi,u] = P(score at error step > score at non-error step), ties = 0.5
  6. arm score        SRC[pi,a] = mean over TASKS of (mean over that task's labelled rollouts of A[pi,u])

Data facts assumed (verified in SPEC.md, not re-derived here):
  * rows.parquet key is (pi, uid, step, channel); channel in {other, action} (+ stray 'think' on ALFWorld)
  * kl / d / dsigned are per-segment token-weighted MEANS -> multiply by n_tokens for mass
  * dplus  = (d - dsigned)/2   (teacher-over-student mass)
    dminus = (d + dsigned)/2   (student-over-teacher mass)
  * step is 1-indexed; labels are 1-indexed
  * the rollout pool is bit-identical across arms in a cell (checked by check_token_identity / T5)

Dependencies: numpy, pandas, pyarrow (tested with pandas 2.3.3, pyarrow 23.0.1); scipy is NOT needed.

Data locations (public dataset layout): a short cell name resolves to
  $PI_GRID_ROOT/<cell>/rows.parquet                       if PI_GRID_ROOT is set, else
  ./outputs/pi_cells/<bench>/<cell>/rows.parquet          (cells you scored with opsd.alfworld.analyze_pi_positions)
with <bench> read off the cell-name prefix (alfworld_* / appworld_* / tau2_*). Error labels live in
$PI_LABELS_ROOT (default $OPSD_DATA_ROOT/<bench>/pi_metrics/error_labels).

    python pi_src_lib.py                                  # synthetic self-tests (no data needed)
    python pi_src_lib.py --data-tests                     # + T7/T8 on the bundle's appworld_chk_train90_4bi
    python pi_src_lib.py --data-tests --labels-dir DIR    # + T5/T13 (step-ranking study labels; see below)
"""

from __future__ import annotations

import itertools
import json
import math
import os
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Optional single directory holding every cell (overrides the per-benchmark default, see _cell_path).
GRID_ROOT = os.environ.get("PI_GRID_ROOT") or None

# attributes carried through per_step (token-weighted means).  'ntok' and 'pos' are the C3/C4 controls.
BASE_ATTRS = ("kl", "klf", "d", "dsigned", "dH")
DERIVED_ATTRS = ("dplus", "dminus")
CONTROL_ATTRS = ("ntok", "pos")

# minimal column set needed by per_step()
REQUIRED_COLUMNS = ["pi", "task_id", "uid", "step", "channel", "n_tokens"]


# --------------------------------------------------------------------------------------
# data locations
# --------------------------------------------------------------------------------------
def repo_root() -> str:
    """The repository root: the nearest ancestor of this file holding pyproject.toml, opsd/__init__.py and
    opsd/skyrl_ext/__init__.py (the predicate of opsd._repo)."""
    markers = ("pyproject.toml", "opsd/__init__.py", "opsd/skyrl_ext/__init__.py")
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if all(os.path.isfile(os.path.join(d, m)) for m in markers):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(f"cannot locate the repository root (pyproject.toml + opsd/skyrl_ext/) above {__file__}")
        d = parent


def data_root() -> str:
    """$OPSD_DATA_ROOT (the unpacked dataset bundle), default ${XDG_CACHE_HOME:-~/.cache}/opsd/data."""
    if os.environ.get("OPSD_DATA_ROOT"):
        return os.environ["OPSD_DATA_ROOT"]
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(cache, "opsd", "data")


def bench_dir(bench: str) -> str:
    """Per-benchmark directory of the dataset bundle.

    AppWorld ships encrypted and is unpacked under appworld/plain/; the bundle's own top-level folder
    (opsd_appworld_pi/) is used when present.
    """
    if bench != "appworld":
        return os.path.join(data_root(), bench)
    plain = os.path.join(data_root(), "appworld", "plain")
    nested = os.path.join(plain, "opsd_appworld_pi")
    if os.path.isdir(nested) or not os.path.isdir(os.path.join(plain, "pi_metrics")):
        return nested
    return plain


def appworld_corpus_root() -> str:
    """Root of the regenerated AppWorld no-hint corpora (not in the dataset bundle: AppWorld's license
    allows only encrypted redistribution). $APPWORLD_CORPUS_ROOT, default $OPSD_DATA_ROOT/appworld/corpus --
    the one convention every AppWorld corpus producer and consumer script uses."""
    return os.environ.get("APPWORLD_CORPUS_ROOT") or os.path.join(data_root(), "appworld", "corpus")


def bench_of(cell: str) -> str:
    """Benchmark of a cell name (its prefix); unprefixed names are ALFWorld's."""
    name = os.path.basename(str(cell).rstrip("/"))
    for b in ("alfworld", "appworld", "tau2"):
        if name.startswith(b):
            return b
    return "alfworld"


def grid_root(bench: str) -> str:
    """Directory of the scored cells: $PI_GRID_ROOT, else ./outputs/pi_cells/<bench> (your scoring output)."""
    return os.environ.get("PI_GRID_ROOT") or os.path.join("outputs", "pi_cells", bench)


def labels_root(bench: str) -> str:
    """Directory of the error-step label JSONs: $PI_LABELS_ROOT, else <bench>/pi_metrics/error_labels."""
    return os.environ.get("PI_LABELS_ROOT") or os.path.join(bench_dir(bench), "pi_metrics", "error_labels")


def arm_of_pi(bench: str, pi: str) -> str:
    """Paper PI id (glm_a, qwen_ao, self_skill, glm_chk, ...) -> the arm name used in the scored cells.

    Trajectory densities: a = actions, ao = raw_no_think (Action + Obs.), aot = raw (Action + Obs. +
    Thou.); the GLM-5.2 prefix is glm52_ on AppWorld and glm_ elsewhere, Qwen3.7-Plus is qwen37_.
    Analyzer arms: skill -> anlz_<src>, summ -> summ_<src>, chk -> chk_<src> with src glm|qwn|self.
    """
    src, form = pi.split("_", 1)
    if form in ("a", "ao", "aot"):
        fmt = {"a": "actions", "ao": "raw_no_think", "aot": "raw"}[form]
        if src == "glm":
            return ("glm52_" if bench == "appworld" else "glm_") + fmt
        return "qwen37_" + fmt
    kind = {"skill": "anlz", "summ": "summ", "chk": "chk"}[form]
    short = {"glm": "glm", "qwen": "qwn", "self": "self"}[src]
    return f"{kind}_{short}"


def opsd_best_window(bench: str, path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """{student: {arm: OPSD best-window pass@1}} for `bench`: the target every PI-metric correlation uses.

    Read from YOUR vanilla-OPSD runs, summarised in a CSV with columns ``bench, student, pi, pass1``
    (one row per trained arm; ``pi`` is the public PI id, e.g. glm_ao; ``pass1`` the best-window pass@1
    from ``python -m opsd.eval.window_metric``, in percent). The path is ``path`` or $OPSD_TARGETS_CSV.
    """
    import csv

    path = path or os.environ.get("OPSD_TARGETS_CSV")
    if not path:
        raise SystemExit(
            "correlations need the OPSD targets: pass --targets <csv> (or set OPSD_TARGETS_CSV), a CSV with "
            "columns bench,student,pi,pass1 built from python -m opsd.eval.window_metric over your OPSD runs"
        )
    out: Dict[str, Dict[str, float]] = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["bench"] == bench:
                out.setdefault(r["student"], {})[arm_of_pi(bench, r["pi"])] = float(r["pass1"])
    return out


# --------------------------------------------------------------------------------------
# D aggregation over rollouts
# --------------------------------------------------------------------------------------
AGG_CHOICES = ("token_pooled", "traj_mean")


def agg_divergence(
    df: pd.DataFrame, col: str, agg: str = "token_pooled", weight: str = "n_tokens", by: str = "uid"
) -> float:
    """Aggregate a per-(rollout, step, channel) mean column (d, kl, dsigned, ...) over rollouts.

    token_pooled  sum(col * n_tokens) / sum(n_tokens) over ALL rows of all rollouts -- every token
                  weighs the same, so long rollouts weigh more. This is what produced the paper's tables.
    traj_mean     the paper's equation for D: first the token-weighted mean within each rollout (its
                  per-trajectory mean over sampled tokens), then the unweighted mean over rollouts.
    """
    if agg == "token_pooled":
        return float(np.average(df[col], weights=df[weight]))
    if agg == "traj_mean":
        w = df[weight].astype(float)
        num = (df[col].astype(float) * w).groupby(df[by], observed=True).sum()
        den = w.groupby(df[by], observed=True).sum()
        per_traj = num[den > 0] / den[den > 0]
        return float(per_traj.mean())
    raise ValueError(f"agg must be one of {AGG_CHOICES}, got {agg!r}")


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def _cell_path(cell: str) -> str:
    if os.path.sep in cell and cell.endswith(".parquet"):
        return cell
    return os.path.join(GRID_ROOT or grid_root(bench_of(cell)), cell, "rows.parquet")


def load_cells(
    cells: Sequence[str],
    columns: Sequence[str],
    task_ids: Optional[Iterable[str]] = None,
    pis: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Read `columns` from each cell's rows.parquet and concatenate.

    `cells` may be short cell names (resolved by _cell_path) or explicit .parquet paths.
    A `cell` column is added.  Optional `task_ids` / `pis` push a row filter down to pyarrow,
    which is the big speedup for the 1.4 GB ALFWorld cell (only labelled tasks + their
    siblings are ever needed).
    """
    columns = list(dict.fromkeys(list(columns)))
    task_ids = None if task_ids is None else sorted(set(task_ids))
    pis = None if pis is None else sorted(set(pis))
    filt = []
    if task_ids is not None:
        filt.append(("task_id", "in", task_ids))
    if pis is not None:
        filt.append(("pi", "in", pis))
    out = []
    for cell in cells:
        path = _cell_path(cell)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path}: no scored cell there; score the PI first: python -m opsd.alfworld.analyze_pi_positions "
                f"... --out-dir {os.path.dirname(path)} (then --merge), or set PI_GRID_ROOT"
            )
        tbl = pq.read_table(path, columns=columns, filters=filt or None)
        df = tbl.to_pandas()
        df["cell"] = os.path.basename(os.path.dirname(path))
        out.append(df)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------------------
# step 1 -- per-step token-weighted aggregation over channels
# --------------------------------------------------------------------------------------
def per_step(df: pd.DataFrame, attrs: Sequence[str] = None) -> pd.DataFrame:
    """Collapse (pi, task_id, uid, step, channel) -> (pi, task_id, uid, step).

    n_tokens is SUMMED; every attribute is a token-weighted mean over the channels of that step:
        x = sum_ch(a_ch * n_tokens_ch) / sum_ch(n_tokens_ch)

    dplus/dminus are formed at the CHANNEL level (before aggregation).  They are affine in
    (d, dsigned), so the token-weighted mean commutes with the transform -- assert_dplus_commutes()
    proves this numerically on real data; forming them first is the definitionally safe order.

    Adds:
      ntok  = summed n_tokens  (C3 length control)
      pos   = step as float    (C4 positional control)
    """
    have = [a for a in (attrs or BASE_ATTRS) if a in df.columns]
    d = df.copy()
    if "d" in d.columns and "dsigned" in d.columns:
        d["dplus"] = (d["d"] - d["dsigned"]) / 2.0
        d["dminus"] = (d["d"] + d["dsigned"]) / 2.0
        have = have + ["dplus", "dminus"]

    w = d["n_tokens"].astype(float)
    mass_cols = {}
    for a in have:
        mass_cols["_m_" + a] = d[a].astype(float) * w
    d = d.assign(**mass_cols)

    keys = ["pi", "task_id", "uid", "step"]
    agg = {"n_tokens": "sum"}
    agg.update({"_m_" + a: "sum" for a in have})
    if "success" in d.columns:
        agg["success"] = "max"
    ps = d.groupby(keys, sort=False, observed=True).agg(agg).reset_index()

    tot = ps["n_tokens"].astype(float)
    for a in have:
        ps[a] = np.where(tot > 0, ps["_m_" + a] / tot.where(tot > 0, 1.0), np.nan)
    ps = ps.drop(columns=["_m_" + a for a in have])

    ps["ntok"] = ps["n_tokens"].astype(float)
    ps["pos"] = ps["step"].astype(float)
    return ps


def assert_dplus_commutes(df: pd.DataFrame, tol: float = 1e-12) -> float:
    """Proof that the token-weighted mean commutes with dplus/dminus = (d -/+ dsigned)/2.

    Computes both orders (transform-then-aggregate vs aggregate-then-transform) and returns
    the max abs deviation.
    """
    a = per_step(df)
    b = per_step(df.drop(columns=[c for c in ("dplus", "dminus") if c in df.columns]))
    b = b.copy()
    b["dplus_after"] = (b["d"] - b["dsigned"]) / 2.0
    b["dminus_after"] = (b["d"] + b["dsigned"]) / 2.0
    keys = ["pi", "task_id", "uid", "step"]
    m = a[keys + ["dplus", "dminus"]].merge(b[keys + ["dplus_after", "dminus_after"]], on=keys)
    dev = max(
        np.nanmax(np.abs(m["dplus"] - m["dplus_after"])),
        np.nanmax(np.abs(m["dminus"] - m["dminus_after"])),
    )
    assert dev <= tol, f"dplus/dminus do NOT commute with the token-weighted mean: dev={dev}"
    return float(dev)


# --------------------------------------------------------------------------------------
# step 2/3 -- leave-one-out sibling baseline and residual
# --------------------------------------------------------------------------------------
def loo_baseline(ps: pd.DataFrame, attr: str, min_sibs: int = 5) -> pd.DataFrame:
    """Add `b` (leave-one-out sibling mean) and `r` (= attr - b) within (pi, task_id, step).

    b[pi,t,p,u] = mean over siblings u' != u of task t that REACH step p, of x[pi,u',p].
    Rows with fewer than `min_sibs` *other* contributing siblings are DROPPED (SPEC step 2).
    A rollout never contributes to its own baseline (that is what the -x / -1 does).
    NaN values of `attr` neither contribute to nor receive a baseline.
    """
    d = ps.copy()
    x = d[attr].astype(float)
    ok = x.notna()
    g = d.assign(_x=x.where(ok, 0.0), _ok=ok.astype(float)).groupby(
        ["pi", "task_id", "step"], sort=False, observed=True
    )
    tot = g["_x"].transform("sum")
    cnt = g["_ok"].transform("sum")
    n_sib = cnt - ok.astype(float)  # siblings excluding self
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (tot - x.where(ok, 0.0)) / n_sib.replace(0.0, np.nan)
    d["b"] = np.where(ok & (n_sib > 0), b, np.nan)
    d["r"] = x - d["b"]
    keep = ok.to_numpy() & (n_sib.to_numpy() >= float(min_sibs))
    return d.loc[keep].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# step 5 -- within-rollout ranking AUC
# --------------------------------------------------------------------------------------
def _avg_rank(v: np.ndarray) -> np.ndarray:
    """Average (fractional) ranks, 1-based -- numpy-only stand-in for scipy.stats.rankdata."""
    n = v.size
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sv[j] == sv[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0  # mean of 1-based ranks i+1 .. j
        i = j
    return ranks


def rollout_auc(vals, is_err) -> Optional[float]:
    """P(score at an error step > score at a non-error step), ties counted 0.5.

    Exactly the Mann-Whitney U statistic normalised by n_pos*n_neg, over all
    |E| x (P-|E|) pairs of the rollout.  Returns None if there is no valid pair
    (no error step, no non-error step, or fewer than 2 finite values).
    """
    v = np.asarray(vals, dtype=float)
    e = np.asarray(is_err, dtype=bool)
    if v.size != e.size:
        raise ValueError("vals and is_err length mismatch")
    fin = np.isfinite(v)
    v, e = v[fin], e[fin]
    npos, nneg = int(e.sum()), int((~e).sum())
    if npos == 0 or nneg == 0:
        return None
    ranks = _avg_rank(v)
    u = ranks[e].sum() - npos * (npos + 1) / 2.0
    return float(u / (npos * nneg))


def _rollout_auc_bruteforce(vals, is_err) -> Optional[float]:
    """O(n^2) reference used only by the self-tests."""
    v = np.asarray(vals, dtype=float)
    e = np.asarray(is_err, dtype=bool)
    fin = np.isfinite(v)
    v, e = v[fin], e[fin]
    pos, neg = v[e], v[~e]
    if pos.size == 0 or neg.size == 0:
        return None
    s = 0.0
    for a in pos:
        for b in neg:
            if a > b:
                s += 1.0
            elif a == b:
                s += 0.5
    return float(s / (pos.size * neg.size))


# --------------------------------------------------------------------------------------
# step 6 -- arm score
# --------------------------------------------------------------------------------------
def src(
    ps: pd.DataFrame,
    labels: Mapping[str, Sequence[int]],
    attr: str,
    use_residual: bool,
    min_sibs: int = 5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """SRC for every arm present in `ps`.

    `ps` is a per_step() frame (RAW -- loo_baseline is applied internally when use_residual).
    `labels` maps uid -> list of 1-indexed error steps; it is arm-invariant.

    Returns
      per_rollout : DataFrame [pi, task_id, uid, auc, n_steps, n_err]  (auc NaN if no valid pair)
      scores      : Series indexed by pi -- mean over TASKS of the within-task mean AUC (SPEC step 6)
    """
    if use_residual:
        work = loo_baseline(ps, attr, min_sibs=min_sibs)
        col = "r"
    else:
        work = ps
        col = attr
    lab_uids = set(labels)
    work = work[work["uid"].isin(lab_uids)]

    rows = []
    for (pi, uid), g in work.groupby(["pi", "uid"], sort=False, observed=True):
        err = set(int(s) for s in labels[uid])
        steps = g["step"].to_numpy()
        is_err = np.isin(steps, list(err)) if err else np.zeros(steps.size, dtype=bool)
        a = rollout_auc(g[col].to_numpy(), is_err)
        rows.append(
            {
                "pi": pi,
                "task_id": g["task_id"].iloc[0],
                "uid": uid,
                "auc": np.nan if a is None else a,
                "n_steps": int(steps.size),
                "n_err": int(is_err.sum()),
            }
        )
    per_rollout = pd.DataFrame(rows, columns=["pi", "task_id", "uid", "auc", "n_steps", "n_err"])
    ok = per_rollout.dropna(subset=["auc"])
    scores = (
        ok.groupby(["pi", "task_id"], observed=True)["auc"].mean().groupby("pi", observed=True).mean()
        if len(ok)
        else pd.Series(dtype=float)
    )
    return per_rollout, scores


def auc_series(per_rollout: pd.DataFrame, pi: str) -> pd.Series:
    """Per-rollout AUC of one arm as a Series indexed by uid (input to paired_boot)."""
    s = per_rollout[per_rollout["pi"] == pi].set_index("uid")["auc"]
    return s[~s.index.duplicated()]


def task_of_uid_map(per_rollout: pd.DataFrame) -> Dict[str, str]:
    return dict(zip(per_rollout["uid"], per_rollout["task_id"]))


# --------------------------------------------------------------------------------------
# C2 -- placebo floor over null-null pairs
# --------------------------------------------------------------------------------------
def placebo(scores_by_arm: Mapping[str, float], null_arms: Sequence[str]) -> Dict[str, float]:
    """Noise floor: |SRC[null_A] - SRC[null_B]| over ALL unordered pairs of null arms."""
    nulls = [a for a in null_arms if a in scores_by_arm and np.isfinite(scores_by_arm[a])]
    diffs = [
        abs(float(scores_by_arm[a]) - float(scores_by_arm[b])) for a, b in itertools.combinations(sorted(nulls), 2)
    ]
    if not diffs:
        return {"n_arms": len(nulls), "n_pairs": 0, "mean": np.nan, "p95": np.nan, "max": np.nan}
    d = np.asarray(diffs, dtype=float)
    return {
        "n_arms": len(nulls),
        "n_pairs": int(d.size),
        "mean": float(d.mean()),
        "p95": float(np.percentile(d, 95)),
        "max": float(d.max()),
    }


# --------------------------------------------------------------------------------------
# paired inference -- cluster bootstrap over tasks
# --------------------------------------------------------------------------------------
def paired_boot(
    a_series: pd.Series,
    b_series: pd.Series,
    task_of_uid: Mapping[str, str],
    n: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired difference SRC[a] - SRC[b] with a task-clustered bootstrap.

    Arms share rollouts AND labels, so the per-rollout difference A[a,u] - A[b,u] is the unit.
    Statistic = mean over TASKS of the within-task mean difference (matches SPEC step 6).
    Resamples TASKS with replacement (cluster bootstrap), `n` times.
    """
    a = a_series.dropna()
    b = b_series.dropna()
    common = a.index.intersection(b.index)
    diff = (a.loc[common] - b.loc[common]).astype(float)
    tasks = pd.Series([task_of_uid[u] for u in common], index=common, name="task_id")
    tmean = diff.groupby(tasks).mean()
    if tmean.empty:
        return {"n_uids": 0, "n_tasks": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan, "p_two_sided": np.nan}
    vals = tmean.to_numpy(dtype=float)
    k = vals.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n, k))
    boot = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided bootstrap p (fraction of resamples on the other side of 0, doubled)
    frac = float(np.mean(boot <= 0.0))
    p = 2 * min(frac, 1 - frac)
    return {
        "n_uids": int(len(common)),
        "n_tasks": int(k),
        "mean": float(vals.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "p_two_sided": float(min(1.0, p)),
    }


# --------------------------------------------------------------------------------------
# T5 -- rollout-pool identity across arms
# --------------------------------------------------------------------------------------
def check_token_identity(ps: pd.DataFrame) -> Dict[str, float]:
    """Verify n_tokens per (uid, step) is identical across every arm of a cell.

    Returns max abs deviation from the per-(uid,step) mean, plus shape/coverage counts.
    Nonzero max_abs_dev falsifies the SPEC's "rollout pool is bit-identical" claim.
    """
    g = ps.groupby(["uid", "step"], sort=False, observed=True)["n_tokens"]
    dev = (ps["n_tokens"] - g.transform("mean")).abs()
    n_arms = ps["pi"].nunique()
    cover = g.size()
    return {
        "n_arms": int(n_arms),
        "n_uid_step": int(cover.size),
        "max_abs_dev": float(dev.max()) if len(dev) else 0.0,
        "min_arms_per_uid_step": int(cover.min()) if cover.size else 0,
        "max_arms_per_uid_step": int(cover.max()) if cover.size else 0,
        "all_arms_present_everywhere": bool(cover.size and cover.min() == cover.max() == n_arms),
    }


def restrict_to_common_pool(ps: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Keep only (uid, step) cells that are present in EVERY arm of the frame.

    Needed for `appworld_orig6null_normal_4bi_base32`, which contains TWO slightly different
    rollout pools (a glm52 family and a qwen37 family); within each family the 6 arms are
    bit-identical, but the families differ by ~1-2% of (uid, step) cells.  Cross-family arm
    scores are only comparable on the intersection.
    """
    n_arms = ps["pi"].nunique()
    g = ps.groupby(["uid", "step"], sort=False, observed=True)["pi"]
    keep = g.transform("size").to_numpy() == n_arms
    stats = {
        "n_arms": int(n_arms),
        "rows_in": int(len(ps)),
        "rows_out": int(keep.sum()),
        "rows_dropped": int((~keep).sum()),
        "uids_in": int(ps["uid"].nunique()),
        "uids_out": int(ps.loc[keep, "uid"].nunique()),
    }
    return ps.loc[keep].reset_index(drop=True), stats


def load_labels(paths: Sequence[str]) -> Dict[str, list]:
    """Union of several label files: {uid: [1-indexed error steps]}.  Later files win on collision."""
    out: Dict[str, list] = {}
    for p in paths:
        with open(p) as f:
            out.update(json.load(f))
    return {k: [int(s) for s in v] for k, v in out.items()}


# --------------------------------------------------------------------------------------
# rank correlation across arms (numpy-only; there is no scipy in this venv)
# --------------------------------------------------------------------------------------
def spearman(x: Sequence[float], y: Sequence[float], n_perm: int = 20000, seed: int = 0) -> Dict[str, float]:
    """Spearman rho with a permutation p-value (exact when n! <= n_perm).

    Ties get average ranks (via _avg_rank), so this is the tie-corrected
    Pearson-on-ranks definition, identical to scipy.stats.spearmanr's rho.
    """
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    ok = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[ok], yv[ok]
    n = xv.size
    if n < 3:
        return {"n": int(n), "rho": np.nan, "p": np.nan}
    rx, ry = _avg_rank(xv), _avg_rank(yv)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    rho = float((rx * ry).sum() / den) if den > 0 else np.nan
    # permutation null on the ranks
    if not np.isfinite(rho):
        return {"n": int(n), "rho": np.nan, "p": np.nan}
    if math.factorial(n) <= n_perm:
        perms = np.array(list(itertools.permutations(range(n))))
    else:
        rng = np.random.default_rng(seed)
        perms = np.argsort(rng.random((n_perm, n)), axis=1)
    stats = (rx[perms] * ry).sum(axis=1) / den
    p = float((np.abs(stats) >= abs(rho) - 1e-12).mean())
    return {"n": int(n), "rho": rho, "p": p}


# ======================================================================================
# MANDATORY SELF-TESTS
# ======================================================================================
# T5/T7/T8/T13 read real scored cells and run only with --data-tests. Every other test is synthetic.
#   T7/T8  use the dataset bundle: the AppWorld cell --aw-cell (default appworld_chk_train90_4bi, the
#          smallest released cell; 4 arms incl. chk_glm / chk_glm_null) and, for T8, the released error
#          labels --aw-labels (default <appworld labels root>/labels_appworld_train90_4bi_k2_medium.json).
#          The step-ranking study ran them on appworld_chk_normal_4bi with esc_labels_m{1,2,3,4,5,8}.json;
#          neither is in the bundle (pass both flags to rerun that exact configuration).
#   T5/T13 additionally need the step-ranking study's label files (esc_labels_m*.json /
#          esc_alf_labels_m*.json) and its test_normal / full train200 cells, none of which are in the
#          bundle; they run only with --labels-dir DIR and are skipped otherwise.
_SELFTEST_LABELS_DIR: Optional[str] = None
_SELFTEST_AW_CELL = "appworld_chk_train90_4bi"
_SELFTEST_AW_LABELS: Optional[str] = None  # comma list of label JSONs; None -> the bundle's train90 4bi labels


def _selftest_aw_labels() -> list:
    if _SELFTEST_AW_LABELS:
        return [p for p in _SELFTEST_AW_LABELS.split(",") if p]
    return [os.path.join(labels_root("appworld"), "labels_appworld_train90_4bi_k2_medium.json")]


def _selftest_labels_dir() -> str:
    if not _SELFTEST_LABELS_DIR:
        raise RuntimeError("data self-tests need --labels-dir (directory with esc_*labels_m*.json)")
    return _SELFTEST_LABELS_DIR


def _t1_auc_vs_bruteforce(trials: int = 1000, seed: int = 12345) -> str:
    rng = np.random.default_rng(seed)
    worst = 0.0
    exact = 0
    checked = 0
    for _ in range(trials):
        n = int(rng.integers(2, 40))
        # coarse quantisation on purpose -> lots of ties
        v = rng.integers(0, int(rng.integers(2, 12)), size=n).astype(float)
        if rng.random() < 0.4:
            v = v + rng.normal(0, 1e-9, size=n)  # near-ties
        e = rng.random(n) < rng.uniform(0.05, 0.95)
        fast = rollout_auc(v, e)
        slow = _rollout_auc_bruteforce(v, e)
        if fast is None or slow is None:
            assert fast is None and slow is None, "None-disagreement between fast and brute force"
            continue
        checked += 1
        worst = max(worst, abs(fast - slow))
        exact += int(fast == slow)
    assert worst == 0.0, f"T1 FAILED: max |fast-brute| = {worst}"
    return (
        f"T1 rollout_auc vs O(n^2) brute force: {trials} trials, {checked} with a valid pair, "
        f"bitwise-exact matches {exact}/{checked}, max abs diff {worst:.1e}  -> PASS"
    )


def _t2_ties() -> str:
    outs = []
    for n, k in [(2, 1), (5, 2), (10, 5), (31, 3)]:
        v = np.full(n, 7.0)
        e = np.zeros(n, dtype=bool)
        e[:k] = True
        a = rollout_auc(v, e)
        assert a == 0.5, f"T2 FAILED: n={n} k={k} -> {a}"
        outs.append(f"n={n},n_err={k} -> {a}")
    return "T2 all-equal values give exactly 0.5: " + "; ".join(outs) + "  -> PASS"


def _t3_loo() -> str:
    # one task, one step, 7 siblings with hand-checkable values
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    ps = pd.DataFrame(
        {
            "pi": ["A"] * 7,
            "task_id": ["t"] * 7,
            "uid": [f"t#{i}" for i in range(7)],
            "step": [1] * 7,
            "n_tokens": [10.0] * 7,
            "kl": vals,
        }
    )
    out = loo_baseline(ps, "kl", min_sibs=5)
    tot = sum(vals)
    exp_b = np.array([(tot - v) / 6.0 for v in vals])
    got_b = out.sort_values("uid")["b"].to_numpy()
    assert np.allclose(got_b, exp_b, atol=0, rtol=0), f"T3 FAILED b: {got_b} vs {exp_b}"
    assert np.allclose(out.sort_values("uid")["r"].to_numpy(), np.array(vals) - exp_b, atol=0, rtol=0)
    # hand check of the first one: mean(2..7) = 4.5
    assert got_b[0] == 4.5, got_b[0]
    # self-exclusion: perturbing ONLY rollout 0 must not move rollout 0's own baseline
    ps2 = ps.copy()
    ps2.loc[0, "kl"] = 1000.0
    b0_pert = loo_baseline(ps2, "kl", min_sibs=5).sort_values("uid")["b"].to_numpy()[0]
    assert b0_pert == got_b[0], f"T3 FAILED self-contamination: {b0_pert} != {got_b[0]}"
    # but it MUST move every sibling's baseline
    b_rest = loo_baseline(ps2, "kl", min_sibs=5).sort_values("uid")["b"].to_numpy()[1:]
    assert np.all(b_rest > exp_b[1:] + 100), "T3 FAILED: siblings did not see the perturbation"
    # min_sibs gate: with 6 rows there are 5 other siblings (kept); with 5 rows only 4 (dropped)
    keep6 = len(loo_baseline(ps.iloc[:6], "kl", min_sibs=5))
    keep5 = len(loo_baseline(ps.iloc[:5], "kl", min_sibs=5))
    assert (keep6, keep5) == (6, 0), f"T3 FAILED min_sibs gate: {(keep6, keep5)}"
    return (
        f"T3 LOO: b[t#0]={got_b[0]} == mean(2..7)=4.5 (hand-checked); all 7 b == "
        f"(sum-self)/6 exactly; perturbing u's own value leaves b[u] IDENTICAL "
        f"({b0_pert}) while all 6 siblings move; min_sibs=5 keeps 6-sibling group "
        f"({keep6} rows) and drops 5-sibling group ({keep5} rows)  -> PASS"
    )


def _t4_scale_invariance(seed: int = 7) -> str:
    rng = np.random.default_rng(seed)
    n_tasks, n_sib, n_step = 6, 10, 12
    recs = []
    for t in range(n_tasks):
        for s in range(n_sib):
            P = int(rng.integers(6, n_step + 1))
            for p in range(1, P + 1):
                for pi in ("A", "B"):
                    recs.append(
                        dict(
                            pi=pi,
                            task_id=f"t{t}",
                            uid=f"t{t}#{s}",
                            step=p,
                            n_tokens=float(rng.integers(20, 200)),
                            kl=float(rng.gamma(2, 1)),
                        )
                    )
    ps = pd.DataFrame(recs)
    labels = {}
    for uid, g in ps[ps.pi == "A"].groupby("uid"):
        P = int(g.step.max())
        labels[uid] = sorted(rng.choice(np.arange(1, P + 1), size=2, replace=False).tolist())
    lines = []
    for use_res in (False, True):
        _, s0 = src(ps, labels, "kl", use_residual=use_res)
        for c in (3.0, 1e-6, 1e6):
            ps2 = ps.copy()
            m = ps2["pi"] == "A"
            ps2.loc[m, "kl"] = ps2.loc[m, "kl"] * c
            _, s1 = src(ps2, labels, "kl", use_residual=use_res)
            dA = abs(s1["A"] - s0["A"])
            dB = abs(s1["B"] - s0["B"])
            assert dA < 1e-12, f"T4 FAILED (residual={use_res}, c={c}): SRC[A] moved by {dA}"
            assert dB == 0.0, f"T4 FAILED: untouched arm B moved by {dB}"
            lines.append(f"residual={int(use_res)} c={c:g}: dSRC[A]={dA:.1e} dSRC[B]={dB:.1e}")
        lines[-1] += f"  (SRC[A]={s0['A']:.4f})"
    return "T4 positive-scale invariance:\n     " + "\n     ".join(lines) + "\n     -> PASS"


def _t5_token_identity() -> str:
    cells = [
        ("appworld_chk_normal_4bi", None),
        ("appworld_new12_normal_4bi", None),
        ("appworld_orig6null_normal_4bi_base32", None),
        ("alfworld_train200_4bi", "ALF_TASKS"),
        ("alfworld_new8_train200_4bi", "ALF_TASKS"),
    ]
    lab_dir = _selftest_labels_dir()
    alf_lab = load_labels([f"{lab_dir}/esc_alf_labels_m{k}.json" for k in (1, 2, 4, 8)])
    alf_tasks = sorted({u.split("#")[0] for u in alf_lab})
    lines = []
    worst = 0.0
    for cell, sub in cells:
        tids = alf_tasks if sub == "ALF_TASKS" else None
        df = load_cells([cell], REQUIRED_COLUMNS, task_ids=tids)
        ps = per_step(df, attrs=())
        chk = check_token_identity(ps)
        worst = max(worst, chk["max_abs_dev"])
        _, st = restrict_to_common_pool(ps)
        note = ""
        if not chk["all_arms_present_everywhere"]:
            note = (
                f"  <-- NOT all arms cover all cells: {st['rows_dropped']} / {st['rows_in']} rows "
                f"({100*st['rows_dropped']/st['rows_in']:.2f}%) missing from >=1 arm"
            )
        lines.append(
            f"{cell:<40s} arms={chk['n_arms']:>2d} (uid,step)={chk['n_uid_step']:>7d} "
            f"max|dev|={chk['max_abs_dev']:.1f} all_arms_everywhere={chk['all_arms_present_everywhere']}"
            + ("" if tids is None else f"  [subset: {len(tids)} labelled tasks]")
            + note
        )
    assert worst == 0.0, f"T5 FAILED: n_tokens is NOT identical across arms, max abs dev = {worst}"
    return (
        "T5 token identity across arms (n_tokens per uid,step):\n     "
        + "\n     ".join(lines)
        + f"\n     GLOBAL max abs deviation = {worst:.1f}  -> PASS (rollout pool is identical)"
    )


def _t6_shuffled_labels(seed: int = 3, reps: int = 200) -> str:
    rng = np.random.default_rng(seed)
    n_tasks, n_sib = 8, 12
    recs = []
    for t in range(n_tasks):
        for s in range(n_sib):
            P = int(rng.integers(10, 20))
            for p in range(1, P + 1):
                recs.append(
                    dict(
                        pi="A",
                        task_id=f"t{t}",
                        uid=f"t{t}#{s}",
                        step=p,
                        n_tokens=float(rng.integers(20, 200)),
                        kl=float(rng.gamma(2, 1)),
                    )
                )
    ps = pd.DataFrame(recs)
    lens = ps.groupby("uid")["step"].max().to_dict()
    outs = []
    for use_res in (False, True):
        vals = []
        for _ in range(reps):
            labels = {}
            for uid, P in lens.items():
                k = int(rng.integers(1, 4))
                labels[uid] = sorted(rng.choice(np.arange(1, P + 1), size=k, replace=False).tolist())
            _, s = src(ps, labels, "kl", use_residual=use_res)
            vals.append(float(s["A"]))
        v = np.asarray(vals)
        se = v.std(ddof=1) / np.sqrt(reps)
        assert abs(v.mean() - 0.5) < 4 * se + 1e-3, f"T6 FAILED (residual={use_res}): {v.mean()}"
        outs.append(
            f"residual={int(use_res)}: mean SRC over {reps} random label draws = "
            f"{v.mean():.4f} (se {se:.4f}, |dev from 0.5| = {abs(v.mean()-0.5):.4f})"
        )
    return "T6 shuffled/random labels -> chance:\n     " + "\n     ".join(outs) + "\n     -> PASS"


def _t7_dplus_commutes() -> str:
    # kl/d/dsigned only: the released cells prune klf/dH, and the commute check needs d and dsigned alone
    df = load_cells([_SELFTEST_AW_CELL], REQUIRED_COLUMNS + ["kl", "d", "dsigned", "success"])
    dev = assert_dplus_commutes(df)
    return (
        f"T7 (extra) dplus/dminus BEFORE vs AFTER token-weighted aggregation on real data "
        f"({_SELFTEST_AW_CELL}, {len(df):,} rows): max abs deviation = {dev:.1e}  -> PASS "
        f"(the transform is affine, so the order does not matter)"
    )


def _t8_end_to_end_smoke() -> str:
    """API smoke on the smallest real cell (4 arms).  NOT a study result -- 2 real arms only,
    so the placebo floor here has a single null pair and is uninformative."""
    labels = load_labels(_selftest_aw_labels())
    tasks = sorted({u.split("#")[0] for u in labels})
    df = load_cells([_SELFTEST_AW_CELL], REQUIRED_COLUMNS + ["kl", "d", "dsigned", "success"], task_ids=tasks)
    assert len(df), f"T8: none of the {len(tasks)} labelled tasks is in {_SELFTEST_AW_CELL}"
    ps = per_step(df)
    lines = []
    for attr, use_res in [("kl", True), ("dminus", True), ("dplus", True), ("ntok", False), ("pos", False)]:
        pr, sc = src(ps, labels, attr, use_residual=use_res)
        n_ok = int(pr["auc"].notna().sum()) // max(1, ps["pi"].nunique())
        lines.append(
            f"attr={attr:<7s} residual={int(use_res)}  "
            + "  ".join(f"{a}={sc[a]:.4f}" for a in sorted(sc.index))
            + f"   (labelled rollouts scored/arm = {n_ok})"
        )
    pr, sc = src(ps, labels, "kl", use_residual=True)
    pl = placebo(sc.to_dict(), [a for a in sc.index if a.endswith("_null")])
    bt = paired_boot(auc_series(pr, "chk_glm"), auc_series(pr, "chk_glm_null"), task_of_uid_map(pr), n=2000, seed=0)
    lines.append(f"placebo(kl,residual) over {pl['n_pairs']} null pair(s): mean={pl['mean']:.4f} p95={pl['p95']:.4f}")
    lines.append(
        f"paired_boot chk_glm - chk_glm_null (kl,residual): mean={bt['mean']:+.4f} "
        f"95% CI [{bt['lo']:+.4f}, {bt['hi']:+.4f}] p={bt['p_two_sided']:.3f} "
        f"n_uids={bt['n_uids']} n_tasks={bt['n_tasks']}"
    )
    return (
        f"T8 (extra) end-to-end API smoke on {_SELFTEST_AW_CELL}:\n     "
        + "\n     ".join(lines)
        + "\n     -> PASS (all seams execute; numbers are a smoke check, NOT the study)"
    )


def _t9_planted_signal(seed: int = 11) -> str:
    """POSITIVE control through the FULL src() path (T6 only proves the negative direction).

    Arm 'HOT' gets a large bump planted at exactly the labelled error steps; arm 'COLD' gets the
    same bump planted at exactly the NON-error steps; arm 'FLAT' gets nothing.  SRC must read
    ~1.0 / ~0.0 / ~0.5.  This pins the SIGN convention end to end (higher score at error steps
    -> AUC > 0.5), which nothing else in the suite does.
    """
    rng = np.random.default_rng(seed)
    n_tasks, n_sib = 6, 12
    lens = {}
    for t in range(n_tasks):
        for s in range(n_sib):
            lens[f"t{t}#{s}"] = int(rng.integers(10, 20))
    labels = {
        u: sorted(rng.choice(np.arange(1, P + 1), size=int(rng.integers(1, 4)), replace=False).tolist())
        for u, P in lens.items()
    }
    recs = []
    for u, P in lens.items():
        t = u.split("#")[0]
        for p in range(1, P + 1):
            base = float(rng.gamma(2, 1))
            hit = p in labels[u]
            for pi, val in (("HOT", base + 50.0 * hit), ("COLD", base + 50.0 * (not hit)), ("FLAT", base)):
                recs.append(dict(pi=pi, task_id=t, uid=u, step=p, n_tokens=float(rng.integers(20, 200)), kl=val))
    ps = pd.DataFrame(recs)
    out = []
    for use_res in (False, True):
        _, sc = src(ps, labels, "kl", use_residual=use_res)
        assert sc["HOT"] > 0.99, f"T9 FAILED (residual={use_res}): HOT={sc['HOT']}"
        assert sc["COLD"] < 0.01, f"T9 FAILED (residual={use_res}): COLD={sc['COLD']}"
        assert abs(sc["FLAT"] - 0.5) < 0.10, f"T9 FAILED (residual={use_res}): FLAT={sc['FLAT']}"
        out.append(f"residual={int(use_res)}: HOT={sc['HOT']:.4f} COLD={sc['COLD']:.4f} FLAT={sc['FLAT']:.4f}")
    return (
        "T9 (ADDED) planted-signal positive control through src() -- pins the SIGN:\n     "
        + "\n     ".join(out)
        + "\n     -> PASS (bump AT error steps -> ~1, bump OFF them -> ~0)"
    )


def _t10_paired_boot() -> str:
    """paired_boot must (a) return exactly the constant when b = a - c, (b) return 0 when a == b,
    (c) weight by TASK not by rollout (SPEC step 6)."""
    task_of = {}
    a, b = {}, {}
    # task t0: 3 rollouts, diff 1.0 each   |  task t1: 1 rollout, diff 5.0
    for i in range(3):
        u = f"t0#{i}"
        task_of[u] = "t0"
        a[u] = 0.7
        b[u] = 0.7 - 1.0
    u = "t1#0"
    task_of[u] = "t1"
    a[u] = 0.9
    b[u] = 0.9 - 5.0
    A, B = pd.Series(a), pd.Series(b)
    r = paired_boot(A, B, task_of, n=2000, seed=0)
    rollout_mean = float(np.mean([a[u] - b[u] for u in a]))  # = 2.0  (WRONG statistic)
    assert abs(r["mean"] - 3.0) < 1e-12, f"T10 FAILED task-weighting: {r['mean']} (rollout mean {rollout_mean})"
    assert abs(rollout_mean - 2.0) < 1e-12
    # identical arms -> exactly zero, zero-width CI
    r0 = paired_boot(A, A, task_of, n=500, seed=0)
    assert r0["mean"] == 0.0 and r0["lo"] == 0.0 and r0["hi"] == 0.0, f"T10 FAILED identity: {r0}"
    # constant shift on every rollout -> mean == c, degenerate CI
    C = A - 0.25
    rc = paired_boot(A, C, task_of, n=500, seed=0)
    assert abs(rc["mean"] - 0.25) < 1e-12 and abs(rc["lo"] - 0.25) < 1e-12 and abs(rc["hi"] - 0.25) < 1e-12
    # NaNs are dropped, not propagated
    An = A.copy()
    An["t0#0"] = np.nan
    rn = paired_boot(An, B, task_of, n=500, seed=0)
    assert rn["n_uids"] == 3 and np.isfinite(rn["mean"]), f"T10 FAILED nan handling: {rn}"
    # coverage: 200 synthetic replicates of a known mean-zero effect -> ~95% CI coverage
    rng = np.random.default_rng(4)
    cov = 0
    for _ in range(200):
        tof, X, Y = {}, {}, {}
        for t in range(40):
            for k in range(2):
                u = f"s{t}#{k}"
                tof[u] = f"s{t}"
                X[u] = float(rng.normal(0.5, 0.15))
                Y[u] = X[u] - float(rng.normal(0.0, 0.15))
        rr = paired_boot(pd.Series(X), pd.Series(Y), tof, n=400, seed=int(rng.integers(1e6)))
        cov += int(rr["lo"] <= 0.0 <= rr["hi"])
    assert cov >= 175, f"T10 FAILED coverage: {cov}/200"
    return (
        f"T10 (ADDED) paired_boot: task-weighted statistic = {r['mean']:.4f} (NOT the rollout mean "
        f"{rollout_mean:.4f}); identical arms -> {r0['mean']:.1f} CI[{r0['lo']:.1f},{r0['hi']:.1f}]; "
        f"constant shift 0.25 -> {rc['mean']:.4f}; NaN dropped (n_uids={rn['n_uids']}); "
        f"null coverage {cov}/200 = {cov/2:.1f}%  -> PASS"
    )


def _t11_placebo() -> str:
    sc = {"a": 0.50, "a_null": 0.40, "b": 0.60, "b_null": 0.55, "c_null": 0.70, "d_null": np.nan}
    nulls = [k for k in sc if k.endswith("_null")]
    p = placebo(sc, nulls)
    # pairs among {0.40, 0.55, 0.70}: |.15|, |.30|, |.15|  (d_null is NaN -> excluded)
    assert p["n_arms"] == 3 and p["n_pairs"] == 3, p
    assert abs(p["mean"] - 0.20) < 1e-12 and abs(p["max"] - 0.30) < 1e-12, p
    assert abs(p["p95"] - np.percentile([0.15, 0.15, 0.30], 95)) < 1e-12
    # real arms must NOT leak into the floor
    assert placebo(sc, nulls)["max"] != abs(0.60 - 0.40)
    e = placebo({"x_null": 0.5}, ["x_null"])
    assert e["n_pairs"] == 0 and np.isnan(e["mean"]), e
    return (
        f"T11 (ADDED) placebo floor: 3 finite nulls -> {p['n_pairs']} pairs, mean={p['mean']:.4f} "
        f"p95={p['p95']:.4f} max={p['max']:.4f} (hand-checked |.40-.55|,|.40-.70|,|.55-.70|); "
        f"NaN null excluded; real arms excluded; single null -> n_pairs=0  -> PASS"
    )


def _t12_per_step_weighted_mean() -> str:
    """Hand-checked token-weighted mean over channels, n_tokens SUM, and dplus/dminus algebra."""
    df = pd.DataFrame(
        {
            "pi": ["A"] * 3,
            "task_id": ["t"] * 3,
            "uid": ["t#0"] * 3,
            "step": [1, 1, 2],
            "channel": ["other", "action", "action"],
            "n_tokens": [10.0, 30.0, 5.0],
            "kl": [1.0, 5.0, 2.0],
            "d": [2.0, 6.0, 1.0],
            "dsigned": [1.0, -2.0, 1.0],
        }
    )
    ps = per_step(df).sort_values("step").reset_index(drop=True)
    # step 1: kl = (10*1 + 30*5)/40 = 4.0 ; ntok = 40 ; pos = 1
    assert ps.loc[0, "kl"] == 4.0, ps.loc[0, "kl"]
    assert ps.loc[0, "ntok"] == 40.0 and ps.loc[0, "pos"] == 1.0
    # dplus  = (d-dsigned)/2 per channel -> other 0.5, action 4.0 -> (10*.5+30*4)/40 = 3.125
    # dminus = (d+dsigned)/2 per channel -> other 1.5, action 2.0 -> (10*1.5+30*2)/40 = 1.875
    assert abs(ps.loc[0, "dplus"] - 3.125) < 1e-12, ps.loc[0, "dplus"]
    assert abs(ps.loc[0, "dminus"] - 1.875) < 1e-12, ps.loc[0, "dminus"]
    assert abs((ps.loc[0, "dplus"] + ps.loc[0, "dminus"]) - ps.loc[0, "d"]) < 1e-12
    assert ps.loc[1, "kl"] == 2.0 and ps.loc[1, "ntok"] == 5.0 and ps.loc[1, "pos"] == 2.0
    assert len(ps) == 2, len(ps)
    return (
        "T12 (ADDED) per_step hand-check: 2 channels (10 tok @ kl 1, 30 tok @ kl 5) -> kl=4.0, "
        "ntok=40 (SUM), pos=step; dplus=3.125 dminus=1.875 and dplus+dminus==d exactly; "
        "3 channel rows -> 2 step rows  -> PASS"
    )


def _t13_alfworld_label_integrity() -> str:
    """SPEC claims 0/152 ALFWorld labels exceed their rollout's max step and all 80 uids are in the
    parquet.  Re-verified here because THIS study depends on it (a label past the end would be
    silently dropped by np.isin and would deflate n_err without any error)."""
    lab_dir = _selftest_labels_dir()
    lines = []
    for k in (1, 2, 4, 8):
        lab = load_labels([f"{lab_dir}/esc_alf_labels_m{k}.json"])
        tasks = sorted({u.split("#")[0] for u in lab})
        df = load_cells(
            ["alfworld_train200_4bi"], ["pi", "task_id", "uid", "step"], task_ids=tasks, pis=["glm_raw"]
        )  # NB: this cell's arms are the TRAJECTORY
        # family; 'anlz_glm' lives in the new8 cell
        mx = df.groupby("uid")["step"].max()
        missing = [u for u in lab if u not in mx.index]
        over = [(u, s, int(mx[u])) for u, ss in lab.items() if u in mx.index for s in ss if s > mx[u]]
        under = [(u, s) for u, ss in lab.items() for s in ss if s < 1]
        assert not missing, f"T13 FAILED: {len(missing)} label uids absent from parquet: {missing[:3]}"
        assert not over, f"T13 FAILED: labels past end of rollout: {over[:5]}"
        assert not under, f"T13 FAILED: non-positive label steps: {under[:5]}"
        lines.append(
            f"m{k}: {len(lab)} rollouts / {len(tasks)} tasks / {sum(len(v) for v in lab.values())} "
            f"labels -- 0 missing uids, 0 labels > max step, 0 labels < 1"
        )
    return "T13 (ADDED) ALFWorld label integrity:\n     " + "\n     ".join(lines) + "\n     -> PASS"


def _t14_task_aggregation() -> str:
    """SRC step 6 must be mean-over-TASKS of the within-task mean AUC, NOT a pooled mean over
    rollouts, and the AUC must be oriented so that HIGH score on error steps -> SRC -> 1.

    Construction: task tA has 1 labelled rollout with a PERFECT separator (AUC 1.0);
    task tB has 3 labelled rollouts, all PERFECTLY ANTI-separating (AUC 0.0).
      mean over tasks   = (1.0 + 0.0) / 2 = 0.5
      pooled over rolls = (1.0 + 0 + 0 + 0) / 4 = 0.25     <- what a pooled bug would give
    """
    recs, labels = [], {}
    plan = [("tA", 1, +1.0), ("tB", 3, -1.0)]
    for t, n_lab, sign in plan:
        for s in range(8):  # 8 siblings so the min_sibs=5 gate passes
            uid = f"{t}#{s}"
            P = 6
            err = {2, 5}
            for p in range(1, P + 1):
                val = (10.0 if p in err else 1.0) * sign
                recs.append(dict(pi="A", task_id=t, uid=uid, step=p, n_tokens=100.0, kl=val))
            if s < n_lab:
                labels[uid] = sorted(err)
    ps = pd.DataFrame(recs)
    pr, sc = src(ps, labels, "kl", use_residual=False)
    got = float(sc["A"])
    per = pr.set_index("uid")["auc"]
    assert per["tA#0"] == 1.0, f"T14 FAILED orientation: perfect separator gave {per['tA#0']}, expected 1.0"
    assert per["tB#0"] == 0.0, f"T14 FAILED orientation: perfect anti-separator gave {per['tB#0']}, expected 0.0"
    assert abs(got - 0.5) < 1e-12, f"T14 FAILED task aggregation: SRC={got}, expected 0.5 (pooled bug -> 0.25)"
    pooled = float(pr["auc"].mean())
    assert abs(pooled - 0.25) < 1e-12, pooled
    return (
        f"T14 (added) step-6 aggregation + orientation: perfect separator AUC={per['tA#0']}, perfect "
        f"anti-separator AUC={per['tB#0']}; 1 labelled rollout on task tA (AUC 1.0) + 3 on task tB "
        f"(AUC 0.0) -> SRC={got:.4f} == mean-over-TASKS 0.5 (a pooled-over-rollouts bug would give "
        f"{pooled:.4f})  -> PASS"
    )


def _t15_paired_boot_uid_join() -> str:
    """paired_boot must (a) pair by uid label, not by row order, (b) reduce to the exact task-mean
    difference, (c) give a degenerate zero-width CI when every rollout differs by the same constant,
    and (d) weight by TASK, not by rollout."""
    # task t0: 3 rollouts, task t1: 1 rollout.  Differences chosen so task-mean != rollout-mean.
    diffs = {"t0#0": 0.30, "t0#1": 0.30, "t0#2": 0.30, "t1#0": -0.10}
    a = pd.Series({u: 0.5 + d for u, d in diffs.items()})
    b = pd.Series({u: 0.5 for u in diffs})
    t_of_u = {u: u.split("#")[0] for u in diffs}
    exp = (0.30 + -0.10) / 2.0  # mean over TASKS
    pooled = float(np.mean(list(diffs.values())))  # 0.20 -- rollout-weighted, the wrong answer
    r = paired_boot(a, b, t_of_u, n=2000, seed=0)
    assert abs(r["mean"] - exp) < 1e-12, f"T15 FAILED task weighting: {r['mean']} != {exp} (pooled={pooled})"
    # (a) pairing by uid, not row order: shuffle b's index order -> identical answer
    b_shuf = b.iloc[[3, 1, 0, 2]]
    r2 = paired_boot(a, b_shuf, t_of_u, n=2000, seed=0)
    assert r2 == r, "T15 FAILED: paired_boot is order-dependent (it must join on uid)"
    # (c) constant shift -> zero-width CI at that constant
    c = pd.Series({u: 0.5 + 0.07 for u in diffs})
    r3 = paired_boot(c, b, t_of_u, n=2000, seed=0)
    assert abs(r3["mean"] - 0.07) < 1e-12 and abs(r3["hi"] - r3["lo"]) < 1e-12, r3
    # (d) a genuinely null difference must straddle 0
    rng = np.random.default_rng(11)
    uids = [f"t{t}#{s}" for t in range(30) for s in range(2)]
    x = pd.Series({u: float(rng.random()) for u in uids})
    y = pd.Series({u: float(rng.random()) for u in uids})
    r4 = paired_boot(x, y, {u: u.split("#")[0] for u in uids}, n=2000, seed=1)
    assert r4["lo"] < 0.0 < r4["hi"], f"T15 FAILED: pure-noise pair excluded 0: {r4}"
    return (
        f"T15 (added) paired_boot uid-join: task-weighted mean={r['mean']:+.4f} (== (0.30+(-0.10))/2; a "
        f"rollout-weighted bug gives {pooled:+.4f}); index-shuffled b gives a bitwise-identical "
        f"result (joins on uid); constant +0.07 shift -> mean {r3['mean']:+.4f} CI width "
        f"{r3['hi']-r3['lo']:.1e}; 60 pure-noise rollouts -> CI [{r4['lo']:+.4f}, {r4['hi']:+.4f}] "
        f"straddles 0  -> PASS"
    )


def _t16_spearman() -> str:
    """spearman() vs hand-computed values."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert spearman(x, x)["rho"] == 1.0
    assert spearman(x, x[::-1])["rho"] == -1.0
    # tie handling: y has a tied pair -> average ranks
    y = [1.0, 2.0, 2.0, 4.0, 5.0]
    rho_t = spearman(x, y)["rho"]
    rx = np.array([1, 2, 3, 4, 5], float)
    ry = np.array([1, 2.5, 2.5, 4, 5], float)
    exp = float(np.corrcoef(rx, ry)[0, 1])
    assert abs(rho_t - exp) < 1e-12, f"T16 FAILED ties: {rho_t} vs {exp}"
    # known case: monotone-but-not-linear is still 1.0 (this is what distinguishes it from Pearson)
    assert spearman([1.0, 2, 3, 4, 5], [1.0, 4, 9, 16, 250])["rho"] == 1.0
    # permutation p: n=5 -> 120 exact perms; perfect correlation p = 2/120
    p_perf = spearman(x, x)["p"]
    assert abs(p_perf - 2 / 120) < 1e-12, p_perf
    # null-ish
    r = spearman([3.0, 1, 4, 1, 5, 9, 2, 6], [2.0, 7, 1, 8, 2, 8, 1, 8])
    return (
        f"T16 (added) spearman: rho(x,x)=1.0 rho(x,rev)=-1.0 exact-perm p(perfect,n=5)="
        f"{p_perf:.4f}==2/120; tie-corrected rho={rho_t:.6f} matches Pearson-on-average-ranks "
        f"{exp:.6f}; monotone-nonlinear rho=1.0; scrambled n=8 -> rho={r['rho']:+.3f} p={r['p']:.3f}"
        f"  -> PASS"
    )


_DATA_TESTS = ("_t5_token_identity", "_t7_dplus_commutes", "_t8_end_to_end_smoke", "_t13_alfworld_label_integrity")
_STUDY_LABEL_TESTS = ("_t5_token_identity", "_t13_alfworld_label_integrity")  # need --labels-dir


def run_self_tests(data_tests: bool = False) -> None:
    print("=" * 96)
    print("src_lib.py SELF-TESTS" + ("" if data_tests else "  (synthetic only; --data-tests adds T5/T7/T8/T13)"))
    print("=" * 96)
    for fn in (
        _t1_auc_vs_bruteforce,
        _t2_ties,
        _t3_loo,
        _t4_scale_invariance,
        _t5_token_identity,
        _t6_shuffled_labels,
        _t7_dplus_commutes,
        _t8_end_to_end_smoke,
        _t9_planted_signal,
        _t10_paired_boot,
        _t11_placebo,
        _t12_per_step_weighted_mean,
        _t13_alfworld_label_integrity,
        _t14_task_aggregation,
        _t15_paired_boot_uid_join,
        _t16_spearman,
    ):
        if fn.__name__ in _DATA_TESTS and not data_tests:
            continue
        if fn.__name__ in _STUDY_LABEL_TESTS and not _SELFTEST_LABELS_DIR:
            print(f"{fn.__name__}: SKIPPED (needs the step-ranking study labels and cells, --labels-dir)")
            print("-" * 96)
            continue
        print(fn())
        print("-" * 96)
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description="Self-tests of the PI step-metric library.")
    _ap.add_argument(
        "--data-tests",
        action="store_true",
        help="also run the tests that read real scored cells: T7/T8 on the dataset bundle "
        "(--aw-cell / --aw-labels), plus T5/T13 when --labels-dir is given",
    )
    _ap.add_argument(
        "--labels-dir",
        default=None,
        help="directory holding the step-ranking labels esc_labels_m*.json / esc_alf_labels_m*.json "
        "(not in the bundle; enables T5/T13)",
    )
    _ap.add_argument(
        "--aw-cell",
        default=_SELFTEST_AW_CELL,
        help="AppWorld scored cell for T7/T8; must carry chk_glm and chk_glm_null arms "
        "(default %(default)s, in the bundle)",
    )
    _ap.add_argument(
        "--aw-labels",
        default=None,
        help="comma list of label JSONs for T8 (default: the bundle's " "labels_appworld_train90_4bi_k2_medium.json)",
    )
    _a = _ap.parse_args()
    _SELFTEST_LABELS_DIR = _a.labels_dir
    _SELFTEST_AW_CELL = _a.aw_cell
    _SELFTEST_AW_LABELS = _a.aw_labels
    run_self_tests(data_tests=_a.data_tests)

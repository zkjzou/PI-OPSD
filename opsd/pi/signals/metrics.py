"""Canonical selector-quality metrics for PI ranking / offline analysis.

Single source of truth for the metrics that were copy-pasted (3-4 times, with subtly different
signatures) across the PI analysis code:

- ``opsd/trainer/pi/score_trajectories.py`` — ``roc_auc``, ``spearman`` (pure-Python)
- several offline PI-ranking analysis scripts (not released) — ``roc_auc``, ``spearman``, ``cohens_d``,
  ``best_at_k``, ``ndcg_at_k`` (pandas) and ``best/random/oracle_at_k``, ``ndcg_at_k`` (dict-of-lists)

Design:
- **Pure Python, no numpy/pandas dependency** — so every caller (including the minimal HF-scorer
  path) can import it. Inputs may be lists, numpy arrays, or pandas Series (anything iterable).
- **Scalar metrics** (``roc_auc``/``spearman``/``cohens_d``) take ``(scores, labels)`` and return a
  ``float`` (``nan`` for degenerate cases: one class only, <2 samples, zero variance). Non-finite
  scores are dropped pairwise before computing (matches ``pandas.rank`` + ``dropna`` behaviour on the
  finite subset; the recorded results were computed on finite data, so numbers are unchanged).
- **Curation metrics** (``best_at_k``/``random_at_k``/``oracle_at_k``/``ndcg_at_k``) operate on a
  generic ``groups`` iterable: each group is a ``(scores, labels)`` pair for one task. Callers adapt
  their own container (DataFrame ``groupby`` / ``by_task`` dict) into this shape once.

Numerical notes:
- ``roc_auc`` is the rank-based Mann-Whitney U estimator, identical to the pairwise definition
  ``P(score(pos) > score(neg))`` with ties counted 0.5.
- ``spearman`` uses average ranks (tie-corrected) + Pearson correlation of the ranks.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

Number = float
Group = Tuple[Sequence[float], Sequence[float]]


def _average_ranks(xs: Sequence[float]) -> List[float]:
    """1-based average (tie-corrected) ranks, matching ``pandas.Series.rank(method='average')``."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # mean of the 1-based positions i+1..j+1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def _finite_pairs(scores: Iterable[float], labels: Iterable[float]) -> Tuple[List[float], List[float]]:
    """Drop pairs whose score is non-finite (NaN/inf). Labels are cast to float."""
    s_out: List[float] = []
    y_out: List[float] = []
    for s, y in zip(scores, labels):
        s = float(s)
        if math.isfinite(s):
            s_out.append(s)
            y_out.append(float(y))
    return s_out, y_out


def roc_auc(scores: Iterable[float], labels: Iterable[float]) -> float:
    """AUC = P(score(pos) > score(neg)), ties = 0.5.

    ``labels`` are truthy for the positive class (success=1). Returns ``nan`` if either class is
    empty. Rank-based (Mann-Whitney U); non-finite scores are dropped.
    """
    s, y = _finite_pairs(scores, labels)
    npos = sum(1 for v in y if v == 1.0)
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = _average_ranks(s)
    sum_pos = sum(r for r, yy in zip(ranks, y) if yy == 1.0)
    return (sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg)


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    """Spearman rank correlation (tie-corrected average ranks). ``nan`` if <2 points or zero variance."""
    xs, ys = _finite_pairs(x, y)  # drop non-finite x; keep paired y
    if len(xs) < 2:
        return float("nan")
    rx, ry = _average_ranks(xs), _average_ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        return float("nan")
    return num / (dx * dy)


def cohens_d(values: Iterable[float], labels: Iterable[float]) -> float:
    """Cohen's d = (mean(pos) - mean(neg)) / pooled_sd. ``nan`` if either class has <2 points."""
    vals, ys = _finite_pairs(values, labels)
    a = [v for v, yy in zip(vals, ys) if yy == 1.0]
    b = [v for v, yy in zip(vals, ys) if yy == 0.0]
    if len(a) < 2 or len(b) < 2:
        return float("nan")

    def _var(z: List[float]) -> float:
        m = sum(z) / len(z)
        return sum((v - m) ** 2 for v in z) / (len(z) - 1)  # ddof=1

    sp2 = ((len(a) - 1) * _var(a) + (len(b) - 1) * _var(b)) / (len(a) + len(b) - 2)
    sp = math.sqrt(sp2)
    if sp <= 0.0:
        return float("nan")
    return (sum(a) / len(a) - sum(b) / len(b)) / sp


def dcg(rels: Sequence[float]) -> float:
    """Discounted cumulative gain of a relevance sequence (position 0 = top)."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


# --------------------------------------------------------------------------------------------------
# Curation@k over per-task groups. ``groups`` = iterable of ``(scores, labels)`` per task.
# Callers materialize a list of groups once and pass it to each metric.
# --------------------------------------------------------------------------------------------------
def _topk_labels_by_score(scores: Sequence[float], labels: Sequence[float], k: int) -> List[float]:
    """Labels of the top-k items ranked by descending score (stable ties = original order).

    TIE HANDLING DIFFERS FROM ``roc_auc``, AND IT MATTERS FOR INTERPRETATION. ``roc_auc`` scores a tie as
    half a pair, so a CONSTANT attribute returns exactly 0.5 -- chance, as it should. Everything built on
    this helper (``best_at_k``/``recall_at_k``/``ndcg_at_k``) instead breaks ties by original rollout
    order, so a constant attribute does NOT return chance: it returns whatever bias the rollout ordering
    happens to carry. Not a bug -- there is no better tie rule for a top-k selection, and a real selector
    must pick something -- but it means a curation@k number near the ceiling is only meaningful once you
    know the attribute actually varies. Read it beside the attribute's own mean/spread, and treat AUC as
    the tie-robust cross-check.
    """
    order = sorted(range(len(labels)), key=lambda i: scores[i], reverse=True)[:k]
    return [float(labels[i]) for i in order]


def best_at_k(groups: Iterable[Group], k: int) -> float:
    """Curation precision@k: per task, take the top-k by score; fraction of that set that succeeded."""
    won = tot = 0.0
    for scores, labels in groups:
        top = _topk_labels_by_score(scores, labels, k)
        won += sum(top)
        tot += len(top)
    return won / tot if tot else float("nan")


def random_at_k(groups: Iterable[Group]) -> float:
    """Baseline for best@k: mean per-task success rate (expected success of a random pick, any k)."""
    fr = []
    for _scores, labels in groups:
        if len(labels):
            fr.append(sum(float(v) for v in labels) / len(labels))
    return sum(fr) / len(fr) if fr else float("nan")


def oracle_at_k(groups: Iterable[Group], k: int) -> float:
    """Ceiling for best@k: an oracle that picks true successes first. best@k <= oracle@k."""
    won = tot = 0.0
    for _scores, labels in groups:
        n_won = sum(1 for v in labels if float(v) == 1.0)
        won += min(k, n_won)
        tot += min(k, len(labels))
    return won / tot if tot else float("nan")


def recall_at_k(groups: Iterable[Group], k: int) -> float:
    """Curation RECALL@k: per task, the fraction of that task's wins that land in the top-k by score.

    Macro-averaged, and over tasks with >=1 win only -- recall is 0/0 with no wins, so an all-loss task
    is SKIPPED rather than scored 0. Matches ``ndcg_at_k``, which skips on ``idcg == 0`` for the same
    reason, so the two are computed over the same tasks.

    WHY THIS IS NOT ``best_at_k`` WITH THE DENOMINATOR MOVED. ``best_at_k`` is precision@k -- of the k
    picked, how many won -- so it is capped by the task's win COUNT: a task with 1 win in 8 can never
    score above 1/k no matter how good the ranker is. ``recall_at_k`` asks the complementary question --
    of the wins that exist, how many did we surface -- and reaches 1.0 on that same task. Precision
    answers "is the top-k clean", recall answers "did we find everything", and they diverge exactly
    where win counts are unbalanced across tasks. On AppWorld they are: n=8 reps with per-task wins
    observed anywhere in 1..7, so reporting only one of the pair systematically flatters or penalises
    the ranker depending on the win-rate mix of that step's both-class tasks.

    CHANCE LEVEL IS k/n, NOT 0.5. Unlike AUC there is no antisymmetry here, so the number cannot be read
    on its own: recall@1 = 0.125 at n=8 is exactly random, 0.25 is 2x chance. Emit it beside
    ``random_recall_at_k`` or it is uninterpretable -- the same trap ``ndcg_at_k`` has with its
    win-rate-dependent baseline, which is why ``random_ndcg_at_k`` exists.

    ``nan`` if no group has a win -- W&B drops nan, so an absent panel honestly means "no winnable task
    this step".
    """
    vals = []
    for scores, labels in groups:
        total_wins = sum(1 for v in labels if float(v) > 0)
        if total_wins <= 0:
            continue
        top = _topk_labels_by_score(scores, labels, k)
        vals.append(sum(1 for v in top if float(v) > 0) / total_wins)
    return sum(vals) / len(vals) if vals else float("nan")


def random_recall_at_k(groups: Iterable[Group], k: int) -> float:
    """EXACT expected recall@k of a random ordering, macro-averaged over the same groups.

    Closed form rather than a simulation: by linearity each of a task's wins independently has
    probability ``min(k,n)/n`` of landing in the top-k, so E[recall@k] = min(k,n)/n for ANY win count w.
    Averaging over groups therefore averages min(k,n)/n, which departs from a flat k/n only for tasks
    with fewer than k rollouts. Exact, free and zero-variance -- a sampled baseline would add noise to
    the very number meant to anchor the metric (same argument as ``random_ndcg_at_k``).
    """
    vals = []
    for _scores, labels in groups:
        n = len(labels)
        if n == 0 or sum(1 for v in labels if float(v) > 0) <= 0:
            continue
        vals.append(min(k, n) / n)
    return sum(vals) / len(vals) if vals else float("nan")


def ndcg_at_k(groups: Iterable[Group], k: int) -> float:
    """Mean NDCG@k of the score ordering vs the binary success label, over tasks with >=1 win."""
    vals = []
    for scores, labels in groups:
        ranked = _topk_labels_by_score(scores, labels, k)
        ideal = sorted((float(v) for v in labels), reverse=True)[:k]
        idcg = dcg(ideal)
        if idcg > 0:
            vals.append(dcg(ranked) / idcg)
    return sum(vals) / len(vals) if vals else float("nan")


def random_ndcg_at_k(groups: Iterable[Group], k: int) -> float:
    """EXACT expected nDCG@k of a random ordering, macro-averaged over the same groups as ``ndcg_at_k``.

    WHY THIS IS MANDATORY, NOT OPTIONAL. nDCG@k is mechanically coupled to the win rate: a task solved
    n/n scores 1.0 under ANY ordering, and a task solved 1/n is capped near chance because the single win
    usually misses the top k. So a bare nDCG@k -- or a correlation between nDCG@k and pass@1 -- partly
    measures arithmetic rather than the ranker. Only the excess over this baseline is selector signal.

    Closed form, not a simulation. IDCG@k is deterministic given (w, n, k), so E[nDCG@k] = E[DCG@k]/IDCG@k
    exactly. By symmetry every position of a random permutation holds a win with probability w/n, so

        E[DCG@k] = (w/n) * S(min(k, n)),    IDCG@k = S(min(k, w)),    S(m) = sum_{i<m} 1/log2(i+2)

    giving ``E[nDCG@k] = (w/n) * S(min(k,n)) / S(min(k,w))``. Exact, free and zero-variance -- the 2000-draw
    Monte Carlo it replaces added noise to the very number meant to anchor the metric.

    Skips zero-win tasks, exactly as ``ndcg_at_k`` does, so the two average over the same task set.
    """

    def _s(m: int) -> float:
        return sum(1.0 / math.log2(i + 2) for i in range(m))

    vals = []
    for _scores, labels in groups:
        n = len(labels)
        w = sum(1 for v in labels if float(v) > 0)
        if n == 0 or w <= 0:
            continue
        idcg = _s(min(k, w))
        vals.append((w / n) * _s(min(k, n)) / idcg if idcg > 0 else float("nan"))
    return sum(vals) / len(vals) if vals else float("nan")


__all__ = [
    "roc_auc",
    "spearman",
    "cohens_d",
    "dcg",
    "best_at_k",
    "random_at_k",
    "oracle_at_k",
    "recall_at_k",
    "random_recall_at_k",
    "ndcg_at_k",
    "random_ndcg_at_k",
]

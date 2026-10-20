"""Unit + parity tests for opsd.pi.signals.metrics.

Parity tests reproduce the ORIGINAL (pandas / dict-of-lists) implementations inline and assert the
canonical pure-Python versions match on finite data — this is the guarantee that the Phase-0 dedup
did not change any recorded numbers.
"""

import math
import random

import numpy as np
import pandas as pd
import pytest

from opsd.pi.signals import metrics as M


# --------------------------------------------------------------------------------------------------
# Reference implementations copied verbatim from the pre-refactor scripts (the "ground truth").
# --------------------------------------------------------------------------------------------------
def _ref_roc_auc_pandas(s, y):  # from pi_ranking_allkl.py / pi_meankl_analysis.py
    s = pd.Series(np.asarray(s, float))
    y = np.asarray(y, int)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0 or s.isna().all():
        return float("nan")
    r = s.rank(method="average").to_numpy()
    return (r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def _ref_spearman_pandas(x, y):  # from pi_ranking_allkl.py
    x = pd.Series(np.asarray(x, float))
    y = pd.Series(np.asarray(y, float))
    rx, ry = x.rank(), y.rank()
    return float(np.corrcoef(rx, ry)[0, 1])


def _ref_cohens_d(x, y):  # from pi_ranking_allkl.py
    x = np.asarray(x, float)
    y = np.asarray(y, int)
    a, b = x[y == 1], x[y == 0]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = math.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def _ref_best_at_k_bytask(by_task, k):  # from run_pi_ranking_experiment.py
    won = tot = 0
    for rs in by_task.values():
        top = sorted(rs, key=lambda r: r["score"], reverse=True)[:k]
        won += sum(r["success"] for r in top)
        tot += len(top)
    return won / tot if tot else float("nan")


def _ref_ndcg_at_k_bytask(by_task, k):  # from run_pi_ranking_experiment.py
    def dcg(rels):
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))

    vals = []
    for rs in by_task.values():
        ranked = [r["success"] for r in sorted(rs, key=lambda r: r["score"], reverse=True)][:k]
        ideal = sorted((r["success"] for r in rs), reverse=True)[:k]
        idcg = dcg(ideal)
        if idcg > 0:
            vals.append(dcg(ranked) / idcg)
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------------------------------
# roc_auc
# --------------------------------------------------------------------------------------------------
def test_roc_auc_perfect_and_reversed():
    assert M.roc_auc([3, 2, 1], [1, 1, 0]) == 1.0  # both positives outrank the negative
    assert M.roc_auc([1, 2, 3], [1, 1, 0]) == 0.0  # the negative (3) outranks both positives
    assert M.roc_auc([0, 0, 1], [0, 0, 1]) == 1.0
    assert M.roc_auc([1, 0, 0], [1, 0, 0]) == 1.0


def test_roc_auc_ties_are_half():
    # positive and negative tied -> 0.5
    assert M.roc_auc([1.0, 1.0], [1, 0]) == pytest.approx(0.5)


def test_roc_auc_single_class_is_nan():
    assert math.isnan(M.roc_auc([1, 2, 3], [1, 1, 1]))
    assert math.isnan(M.roc_auc([1, 2, 3], [0, 0, 0]))


def test_roc_auc_parity_with_pandas():
    rng = random.Random(0)
    for _ in range(50):
        n = rng.randint(3, 60)
        scores = [rng.gauss(0, 1) for _ in range(n)]
        labels = [rng.randint(0, 1) for _ in range(n)]
        if len(set(labels)) < 2:
            continue
        ours = M.roc_auc(scores, labels)
        ref = _ref_roc_auc_pandas(scores, labels)
        assert ours == pytest.approx(ref, abs=1e-9), (scores, labels)


def test_roc_auc_drops_non_finite():
    # NaN score is dropped; the remaining [3(+),1(-)] -> AUC 1.0
    assert M.roc_auc([float("nan"), 3.0, 1.0], [1, 1, 0]) == 1.0


# --------------------------------------------------------------------------------------------------
# spearman
# --------------------------------------------------------------------------------------------------
def test_spearman_monotone():
    assert M.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert M.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_parity_with_pandas():
    rng = random.Random(1)
    for _ in range(50):
        n = rng.randint(3, 60)
        x = [rng.gauss(0, 1) for _ in range(n)]
        y = [float(rng.randint(0, 1)) for _ in range(n)]
        ours = M.spearman(x, y)
        ref = _ref_spearman_pandas(x, y)
        if math.isnan(ref):
            assert math.isnan(ours)
        else:
            assert ours == pytest.approx(ref, abs=1e-9)


def test_spearman_too_few_points_is_nan():
    assert math.isnan(M.spearman([1.0], [1.0]))


# --------------------------------------------------------------------------------------------------
# cohens_d
# --------------------------------------------------------------------------------------------------
def test_cohens_d_parity():
    rng = random.Random(2)
    for _ in range(50):
        n = rng.randint(4, 60)
        vals = [rng.gauss(0, 1) for _ in range(n)]
        labels = [rng.randint(0, 1) for _ in range(n)]
        ours = M.cohens_d(vals, labels)
        ref = _ref_cohens_d(vals, labels)
        if math.isnan(ref):
            assert math.isnan(ours)
        else:
            assert ours == pytest.approx(ref, abs=1e-9)


def test_cohens_d_degenerate_is_nan():
    assert math.isnan(M.cohens_d([1.0, 2.0, 3.0], [1, 1, 1]))  # no negatives


# --------------------------------------------------------------------------------------------------
# curation@k
# --------------------------------------------------------------------------------------------------
def test_best_at_k_hand_computed():
    # task A: scores/labels -> top-2 by score = labels [1,0]; task B: top-2 = [1,1]
    groups = [([9, 8, 1], [1, 0, 1]), ([5, 4, 3], [1, 1, 0])]
    assert M.best_at_k(groups, 2) == pytest.approx((1 + 1 + 1) / 4)  # 3 wins / 4 picks
    assert M.best_at_k(groups, 1) == pytest.approx(1.0)  # both top-1 succeeded


def test_oracle_and_random_at_k():
    groups = [([1, 2, 3], [1, 0, 0]), ([1, 2, 3], [1, 1, 0])]
    # oracle@1 = per task min(1, n_won)/min(1,n) = (1+1)/(1+1) = 1.0
    assert M.oracle_at_k(groups, 1) == pytest.approx(1.0)
    # random@k = mean per-task success rate = mean(1/3, 2/3) = 0.5
    assert M.random_at_k(groups) == pytest.approx(0.5)


def test_curation_parity_with_bytask_reference():
    rng = random.Random(3)
    by_task = {}
    for t in range(20):
        rs = [{"score": rng.gauss(0, 1), "success": rng.randint(0, 1)} for _ in range(rng.randint(2, 8))]
        by_task[t] = rs
    groups = [([r["score"] for r in rs], [r["success"] for r in rs]) for rs in by_task.values()]
    for k in (1, 4):
        assert M.best_at_k(groups, k) == pytest.approx(_ref_best_at_k_bytask(by_task, k), abs=1e-12)
        assert M.ndcg_at_k(groups, k) == pytest.approx(_ref_ndcg_at_k_bytask(by_task, k), abs=1e-12)


# --------------------------------------------------------------------------------------------------
# container-agnostic inputs
# --------------------------------------------------------------------------------------------------
def test_accepts_numpy_and_pandas_inputs():
    scores = np.array([3.0, 2.0, 1.0])
    labels = pd.Series([1, 1, 0])
    assert M.roc_auc(scores, labels) == 1.0
    assert M.spearman(scores, np.array([3.0, 2.0, 1.0])) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------------------
# recall@k -- the complement of best@k (precision@k). Added alongside the 30B PI arms.
# --------------------------------------------------------------------------------------------------
def test_recall_at_k_perfect_and_worst_ranker():
    labels = [1, 1, 0, 0, 0, 0, 0, 0]
    good = [([8, 7, 6, 5, 4, 3, 2, 1], labels)]  # wins ranked first
    bad = [([1, 2, 3, 4, 5, 6, 7, 8], labels)]  # wins ranked last
    assert M.recall_at_k(good, 1) == pytest.approx(0.5)  # 1 of the 2 wins surfaced
    assert M.recall_at_k(good, 2) == pytest.approx(1.0)
    assert M.recall_at_k(good, 4) == pytest.approx(1.0)
    assert M.recall_at_k(bad, 4) == pytest.approx(0.0)


def test_recall_and_precision_diverge_when_wins_are_scarce():
    """The reason BOTH are logged: same data, same perfect ranker, opposite readings.

    precision@4 is capped at 1/k by the single win; recall@4 is 1.0 because that win was found.
    """
    one_win = [([8, 7, 6, 5, 4, 3, 2, 1], [1, 0, 0, 0, 0, 0, 0, 0])]
    assert M.recall_at_k(one_win, 4) == pytest.approx(1.0)
    assert M.best_at_k(one_win, 4) == pytest.approx(0.25)


def test_random_recall_at_k_is_k_over_n_regardless_of_win_count():
    for k, want in ((1, 0.125), (2, 0.25), (4, 0.5)):
        two_wins = [([0.0] * 8, [1, 1, 0, 0, 0, 0, 0, 0])]
        one_win = [([0.0] * 8, [1, 0, 0, 0, 0, 0, 0, 0])]
        assert M.random_recall_at_k(two_wins, k) == pytest.approx(want)
        assert M.random_recall_at_k(one_win, k) == pytest.approx(want)


def test_recall_at_k_skips_all_loss_groups_like_ndcg():
    """0/0 is undefined, so an all-loss task is skipped -- not scored 0, which would drag the mean."""
    all_loss = [([5, 4, 3, 2, 1, 0, 0, 0], [0] * 8)]
    assert math.isnan(M.recall_at_k(all_loss, 4))
    assert math.isnan(M.ndcg_at_k(all_loss, 4))  # same skip rule => same task population
    mixed = all_loss + [([8, 7, 6, 5, 4, 3, 2, 1], [1, 1, 0, 0, 0, 0, 0, 0])]
    assert M.recall_at_k(mixed, 2) == pytest.approx(1.0)  # the all-loss group contributes nothing


def test_curation_at_k_is_not_tie_robust_unlike_auc():
    """Documents a real asymmetry: a CONSTANT attribute gives chance under AUC but not under top-k.

    roc_auc counts ties as half a pair -> exactly 0.5. _topk_labels_by_score breaks ties by original
    order, so a constant score returns whatever bias the rollout order carries (here: wins listed first
    => a perfect-looking 1.0). Guards the interpretation note in metrics.py.
    """
    wins_first = [([0.0] * 8, [1, 1, 0, 0, 0, 0, 0, 0])]
    wins_last = [([0.0] * 8, [0, 0, 0, 0, 0, 0, 1, 1])]
    assert M.roc_auc([0.0] * 8, [1, 1, 0, 0, 0, 0, 0, 0]) == pytest.approx(0.5)
    assert M.recall_at_k(wins_first, 2) == pytest.approx(1.0)
    assert M.recall_at_k(wins_last, 2) == pytest.approx(0.0)

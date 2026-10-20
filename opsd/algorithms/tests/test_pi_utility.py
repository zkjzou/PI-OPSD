"""Tests for the online PI-utility attributes (d / d+ / d- / nDCG@k / ROC-AUC / curation@k).

These mirror the OFFLINE definitions behind the published PI-format table. If they drift, the online
numbers stop being comparable to that table, which is the only reason to log them.

Union of the AppWorld-scaffold suite and the ALFWorld-line suite. The 12 tests both lines carried verbatim
appear once; the ALFWorld copy of `test_no_both_class_tasks_reports_zero_not_a_fake_ndcg` is subsumed by the
AppWorld one (identical plus an `auc_d`-absent assertion), so it also appears once.
"""

import math
import random

import pytest
import torch

from opsd.algorithms import pi_utility as U
from opsd.pi.signals.metrics import ndcg_at_k


def _attrs(s, t, m):
    return U.divergence_attributes(torch.tensor(s), torch.tensor(t), torch.tensor(m))


# --------------------------------------------------------------------------- the identity
def test_identity_d_equals_dplus_plus_dminus():
    """The one check that catches a d+/d- sign flip: both stay positive and both move plausibly,
    only their MEANING swaps. The offline table asserts this too."""
    g = torch.Generator().manual_seed(0)
    s = torch.randn(8, 32, generator=g)
    t = torch.randn(8, 32, generator=g)
    m = torch.ones(8, 32)
    a = U.divergence_attributes(s, t, m)
    assert U.check_identity(a) < 1e-5
    assert torch.allclose(a["d_signed"], a["d_minus"] - a["d_plus"], atol=1e-5)


def test_dplus_is_teacher_agreement():
    """d+ must count tokens the TEACHER likes MORE than the student (dsigned < 0). Getting this
    backwards is the whole reason the +/- naming is documented."""
    # student -1, teacher 0  => dsigned = -1 => teacher likes it more => all mass in d+
    a = _attrs([[-1.0, -1.0]], [[0.0, 0.0]], [[1.0, 1.0]])
    assert a["d_plus"].item() == pytest.approx(1.0)
    assert a["d_minus"].item() == pytest.approx(0.0)
    # reversed => all mass in d-
    b = _attrs([[0.0, 0.0]], [[-1.0, -1.0]], [[1.0, 1.0]])
    assert b["d_minus"].item() == pytest.approx(1.0)
    assert b["d_plus"].item() == pytest.approx(0.0)


# --------------------------------------------------------------------------- masking / aggregation
def test_only_masked_tokens_count():
    """True per-token mean over UNMASKED tokens -- not a mean over the padded width."""
    a = _attrs([[1.0, 1.0, 99.0]], [[0.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]])
    assert a["d"].item() == pytest.approx(1.0)  # the 99 is masked out entirely


def test_empty_mask_is_zero_not_nan():
    a = _attrs([[1.0, 2.0]], [[0.0, 0.0]], [[0.0, 0.0]])
    assert a["d"].item() == 0.0 and not math.isnan(a["d"].item())


def test_per_trajectory_then_mean_not_mean_of_bucket_means():
    """Trajectories of different length must weight by their own token count, not equally per bucket."""
    s = torch.tensor([[2.0, 2.0, 2.0, 2.0], [4.0, 0.0, 0.0, 0.0]])
    t = torch.zeros(2, 4)
    m = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    a = U.divergence_attributes(s, t, m)
    assert a["d"][0].item() == pytest.approx(2.0)  # 4 tokens of 2
    assert a["d"][1].item() == pytest.approx(4.0)  # 1 token of 4


def test_nan_and_inf_are_sanitised():
    a = _attrs([[float("nan"), float("inf")]], [[0.0, 0.0]], [[1.0, 1.0]])
    assert not math.isnan(a["d"].item())
    assert a["d"].item() == pytest.approx(U.KL_CLAMP / 2)  # nan->0, inf->clamp


def test_token_clip_is_not_applied():
    """OPSD_TOKEN_CLIP=0.5 bounds training PRESSURE; applying it here would saturate every attribute
    at 0.5 and destroy the magnitude the metric is made of."""
    a = _attrs([[5.0]], [[0.0]], [[1.0]])
    assert a["d"].item() == pytest.approx(5.0)


# --------------------------------------------------------------------------- grouping / nDCG
def test_both_class_filter_drops_degenerate_tasks():
    uids = ["t1", "t1", "t2", "t2", "t3", "t3"]
    succ = [1, 0, 1, 1, 0, 0]  # t1 both-class; t2 all-win; t3 all-loss
    g = U.group_by_task(uids, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], succ)
    assert len(g) == 1
    assert sorted(g[0][1]) == [0.0, 1.0]


def test_ndcg_perfect_and_worst_ordering():
    uids = ["t"] * 4
    succ = [1, 1, 0, 0]
    perfect = U.group_by_task(uids, [4, 3, 2, 1], succ)
    worst = U.group_by_task(uids, [1, 2, 3, 4], succ)
    assert ndcg_at_k(perfect, 4) == pytest.approx(1.0)
    assert ndcg_at_k(worst, 4) < ndcg_at_k(perfect, 4)


def test_random_baseline_is_exact_not_simulated():
    """Closed form must match a large Monte-Carlo average of ndcg_at_k over shuffles."""
    import random

    labels = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    groups = [([0.0] * len(labels), labels)]
    closed = U.ndcg_random_expectation(groups, 4)
    rng = random.Random(0)
    sim = []
    for _ in range(4000):
        sc = [rng.random() for _ in labels]
        sim.append(ndcg_at_k([(sc, labels)], 4))
    assert closed == pytest.approx(sum(sim) / len(sim), abs=0.02)


def test_random_baseline_is_well_above_zero():
    """~0.29 on an ALFWorld-shaped group. A bare nDCG@4 near 0.3 means NOTHING; without this anchor an
    inert attribute reads as marginally positive."""
    labels = [1.0] * 2 + [0.0] * 6
    val = U.ndcg_random_expectation([([0.0] * 8, labels)], 4)
    assert 0.2 < val < 0.45


# --------------------------------------------------------------------------- the full step payload
def test_metrics_payload_shape_and_keys():
    g = torch.Generator().manual_seed(1)
    a = U.divergence_attributes(torch.randn(8, 16, generator=g), torch.randn(8, 16, generator=g), torch.ones(8, 16))
    uids = ["t1"] * 4 + ["t2"] * 4
    succ = [1, 0, 0, 0, 1, 1, 0, 0]
    out = U.pi_utility_metrics(a, uids, succ, k=4, n_steps=[5, 9, 12, 30, 4, 6, 22, 28])
    for key in (
        "pi_utility/d",
        "pi_utility/d_plus",
        "pi_utility/d_minus",
        "pi_utility/d_signed",
        "pi_utility/ndcg_at_4_d",
        "pi_utility/ndcg_at_4_random",
        "pi_utility/ndcg_at_4_neg_nstep",
        "pi_utility/ndcg_both_class_tasks",
    ):
        assert key in out, key
    assert out["pi_utility/ndcg_both_class_tasks"] == 2.0
    assert out["pi_utility/identity_err"] < 1e-5


def test_no_both_class_tasks_reports_zero_not_a_fake_ndcg():
    """If every task is all-win or all-loss the step carries no ranking information; it must say so
    rather than emit a number computed from nothing."""
    a = U.divergence_attributes(torch.randn(4, 8), torch.randn(4, 8), torch.ones(4, 8))
    out = U.pi_utility_metrics(a, ["t1"] * 4, [1, 1, 1, 1], k=4)
    assert out["pi_utility/ndcg_both_class_tasks"] == 0.0
    assert "pi_utility/ndcg_at_4_d" not in out
    assert "pi_utility/auc_d" not in out, "an AUC over single-class groups is undefined, not 0.5"


# --------------------------------------------------------------------------- ROC-AUC ranking view
def test_within_task_auc_perfect_and_inverted():
    """1.0 for a perfect ranker, 0.0 for its negation. The antisymmetry is why AUC was added."""
    groups = [([3.0, 2.0, 1.0, 0.0], [1, 1, 0, 0]), ([9.0, 5.0, 1.0], [1, 0, 0])]
    assert U.within_task_auc(groups) == pytest.approx(1.0)
    flipped = [([-s for s in sc], lb) for sc, lb in groups]
    assert U.within_task_auc(flipped) == pytest.approx(0.0)


def test_within_task_auc_antisymmetry_on_random_data():
    """AUC(-x) == 1 - AUC(x) exactly. This identity is the entire reason AUC reads a sign flip that nDCG
    cannot: it makes 0.5 a fixed, win-rate-independent chance level."""
    rng = random.Random(0)
    for _ in range(20):
        groups = []
        for _g in range(4):
            n = rng.randint(2, 8)
            lb = [rng.randint(0, 1) for _ in range(n)]
            if 0 < sum(lb) < n:  # both-class only, as group_by_task guarantees
                groups.append(([rng.gauss(0, 1) for _ in range(n)], lb))
        if not groups:
            continue
        a = U.within_task_auc(groups)
        b = U.within_task_auc([([-s for s in sc], lb) for sc, lb in groups])
        assert a + b == pytest.approx(1.0, abs=1e-9)


def test_within_task_auc_ties_are_half():
    """A constant attribute carries no information and must score exactly chance, not 0 or 1."""
    assert U.within_task_auc([([1.0, 1.0, 1.0, 1.0], [1, 1, 0, 0])]) == pytest.approx(0.5)


def test_within_task_auc_is_nan_when_no_group_is_scorable():
    assert math.isnan(U.within_task_auc([]))
    assert math.isnan(U.within_task_auc([([1.0, 2.0], [1, 1])]))  # single-class -> nan, dropped


def test_within_task_auc_differs_from_pooled_when_tasks_sit_at_different_levels():
    """The reason `auc_*` is within-task: task difficulty alone can INVERT a pooled AUC.

    Task A is easy (wins, LOW attribute); task B is hard (losses, HIGH attribute). Within each task the
    attribute ranks wins ABOVE losses, but pooled the level difference dominates and reports the opposite.
    """
    uids = ["A"] * 4 + ["B"] * 4
    succ = [1, 1, 1, 0, 1, 0, 0, 0]  # A wins 3/4 (easy), B wins 1/4 (hard)
    scores = [0.3, 0.2, 0.1, 0.0, 5.3, 5.2, 5.1, 5.0]  # A's level ~0, B's ~5
    groups = U.group_by_task(uids, scores, succ)
    assert U.within_task_auc(groups) == pytest.approx(1.0)  # perfect inside every task
    from opsd.pi.signals.metrics import roc_auc

    pooled = roc_auc(scores, succ)
    assert pooled < 0.5, f"pooled must be misled here, which is the point (got {pooled})"


def test_metrics_payload_includes_auc_and_share_keys():
    g = torch.Generator().manual_seed(1)
    a = U.divergence_attributes(torch.randn(8, 16, generator=g), torch.randn(8, 16, generator=g), torch.ones(8, 16))
    out = U.pi_utility_metrics(
        a, ["t1"] * 4 + ["t2"] * 4, [1, 0, 0, 0, 1, 1, 0, 0], k=4, n_steps=[5, 9, 12, 30, 4, 6, 22, 28]
    )
    keys = (
        "auc_d",
        "auc_d_plus",
        "auc_d_minus",
        "auc_d_signed",
        "auc_neg_nstep",
        "auc_pooled_d",
        "auc_pooled_d_plus",
        "auc_pooled_d_minus",
        "auc_pooled_d_signed",
        "auc_pooled_neg_nstep",
    )
    for key in keys + ("d_minus_share", "frac_tokens_d_minus"):
        assert f"pi_utility/{key}" in out, key
    for key in keys:
        assert 0.0 <= out[f"pi_utility/{key}"] <= 1.0, key


def test_pooled_auc_ignores_task_grouping_and_within_does_not():
    """Both views must be present PER ATTRIBUTE, and they must genuinely differ where difficulty confounds.

    Built so the two disagree in OPPOSITE directions for d_minus vs d_plus, which is the case a single
    pooled gauge on `d` cannot see: task B is hard (high d_minus, mostly losses) while d_plus is held
    task-INVARIANT, mirroring the measured 0.035-0.044 flatness of d_plus across all six live arms.
    """
    from opsd.pi.signals.metrics import roc_auc

    # 2 tasks x 2 rollouts. Within each task the win has the LOWER d_minus; across tasks the hard task's
    # level dominates. d_plus is identical in both tasks, so pooling cannot confound it.
    student = torch.tensor(
        [
            [0.10, 0.10],
            [0.20, 0.20],  # task A (easy): d_minus level ~0.15
            [0.90, 0.90],
            [1.00, 1.00],  # task B (hard): d_minus level ~0.95
        ]
    )
    teacher = torch.zeros(4, 2)
    a = U.divergence_attributes(student, teacher, torch.ones(4, 2))
    uids, succ = ["A", "A", "B", "B"], [1, 0, 1, 0]
    out = U.pi_utility_metrics(a, uids, succ, k=4)

    # within-task: the win has the lower d_minus in BOTH tasks -> a perfect INVERSE ranker -> 0.0
    assert out["pi_utility/auc_d_minus"] == pytest.approx(0.0)
    # pooled: task B's losses (0.90/1.00) outrank task A's win (0.10), so pooling pulls it UP off 0.0
    assert out["pi_utility/auc_pooled_d_minus"] > out["pi_utility/auc_d_minus"]
    # sanity: the pooled key really is the ungrouped Mann-Whitney over every rollout
    assert out["pi_utility/auc_pooled_d_minus"] == pytest.approx(roc_auc(a["d_minus"].tolist(), succ))
    # d_plus is 0 everywhere here (teacher never prefers the token), so it is degenerate/tied at chance in
    # BOTH views -- no task-level structure to confound, which is the contrast being demonstrated.
    assert out["pi_utility/auc_pooled_d_plus"] == pytest.approx(0.5)
    assert 0.0 <= out["pi_utility/d_minus_share"] <= 1.0
    assert 0.0 <= out["pi_utility/frac_tokens_d_minus"] <= 1.0


def test_auc_d_signed_is_the_complement_of_auc_of_its_negation():
    """d_signed = d- - d+, so its AUC and the AUC of -d_signed must sum to 1 -- the check that catches a
    sign error in how d_signed is assembled, which no magnitude-only assertion can see."""
    uids = ["t1"] * 4
    succ = [1, 1, 0, 0]
    g = torch.Generator().manual_seed(7)
    a = U.divergence_attributes(torch.randn(4, 12, generator=g), torch.randn(4, 12, generator=g), torch.ones(4, 12))
    ds = a["d_signed"].tolist()
    fwd = U.within_task_auc(U.group_by_task(uids, ds, succ))
    rev = U.within_task_auc(U.group_by_task(uids, [-v for v in ds], succ))
    assert fwd + rev == pytest.approx(1.0, abs=1e-9)


def test_frac_tokens_d_minus_counts_tokens_not_magnitude():
    """The token share and the magnitude share are different numbers and must not be conflated.

    Row 0: 3 of 4 tokens on the d- side but each tiny. Row 1: 1 of 4 tokens, but huge. Magnitude share
    says row 1 is more d--dominant; token share says row 0 is. Both are true and both are logged.
    """
    student = torch.tensor([[0.01, 0.01, 0.01, -1.0], [5.0, -0.01, -0.01, -0.01]])
    teacher = torch.zeros(2, 4)
    a = U.divergence_attributes(student, teacher, torch.ones(2, 4))
    assert a["frac_minus"].tolist() == pytest.approx([0.75, 0.25])
    share = (a["d_minus"] / a["d"]).tolist()
    assert share[0] < share[1], "row 1 carries more d- MAGNITUDE despite fewer d- tokens"


# --------------------------------------------------------------------------- cumulative AUC
def test_cumulative_auc_is_a_macro_average_over_all_groups_ever_seen():
    """Over GROUPS, not over steps: a step with 4 scorable groups must outweigh one with 1.

    Averaging the per-step macro-averages instead would give (1.0 + 0.0)/2 = 0.5 here; the correct
    group-weighted answer is (4*1.0 + 1*0.0)/5 = 0.8.
    """
    state = {}
    U.update_cumulative_auc(state, {"d": [1.0, 1.0, 1.0, 1.0]})
    out = U.update_cumulative_auc(state, {"d": [0.0]})
    assert out["pi_utility/auc_d_cum"] == pytest.approx(0.8)
    assert out["pi_utility/auc_groups_cum"] == 5.0


def test_cumulative_auc_tracks_each_attribute_independently():
    state = {}
    U.update_cumulative_auc(state, {"d": [0.2], "d_minus": [0.9]})
    out = U.update_cumulative_auc(state, {"d": [0.4], "d_minus": [0.7]})
    assert out["pi_utility/auc_d_cum"] == pytest.approx(0.3)
    assert out["pi_utility/auc_d_minus_cum"] == pytest.approx(0.8)


def test_cumulative_auc_skips_steps_with_no_scorable_group():
    """A step where every task was all-win or all-loss must not move the average or the count."""
    state = {}
    U.update_cumulative_auc(state, {"d": [0.25, 0.25]})
    out = U.update_cumulative_auc(state, {"d": []})
    assert out["pi_utility/auc_d_cum"] == pytest.approx(0.25)
    assert out["pi_utility/auc_groups_cum"] == 2.0


def test_cumulative_auc_absent_before_any_group():
    assert U.update_cumulative_auc({}, {"d": []}) == {}


def test_cumulative_auc_accumulates_across_pi_utility_metrics_calls():
    """End-to-end through the public entrypoint: the count must grow, and the cumulative value must equal
    the group-weighted mean of the per-step values."""
    g = torch.Generator().manual_seed(3)
    state = {}
    per_step, weights = [], []
    for _ in range(5):
        a = U.divergence_attributes(torch.randn(8, 16, generator=g), torch.randn(8, 16, generator=g), torch.ones(8, 16))
        out = U.pi_utility_metrics(a, ["t1"] * 4 + ["t2"] * 4, [1, 0, 0, 0, 1, 1, 0, 0], k=4, auc_cum_state=state)
        n = out["pi_utility/ndcg_both_class_tasks"]
        per_step.append(out["pi_utility/auc_d"] * n)
        weights.append(n)
    assert state["d"][1] == sum(weights)
    assert out["pi_utility/auc_d_cum"] == pytest.approx(sum(per_step) / sum(weights))
    assert out["pi_utility/auc_groups_cum"] == sum(weights)


def test_cumulative_auc_is_opt_in():
    """No state passed -> no cumulative keys, so an unmodified caller is unaffected."""
    a = U.divergence_attributes(torch.randn(8, 16), torch.randn(8, 16), torch.ones(8, 16))
    out = U.pi_utility_metrics(a, ["t1"] * 4 + ["t2"] * 4, [1, 0, 0, 0, 1, 1, 0, 0], k=4)
    assert not any(key.endswith("_cum") for key in out)


def test_cumulative_auc_does_not_pool_a_task_across_steps():
    """The confound the design exists to avoid, made concrete.

    Two steps of the SAME task. `d` falls between steps (as it does in every real run) while success
    rises. Within each step the attribute ranks wins BELOW losses (AUC 0.0 both times), so the honest
    cumulative answer is 0.0. Pooling the task's rollouts across steps instead would let the step-level
    trend dominate and report the opposite.
    """
    state = {}
    # step 1: d high, one win which has the LOWEST d in its step -> per-group AUC 0.0
    s1 = ([0.9, 0.8, 0.7], [0, 0, 1])
    # step 2: d lower overall, same shape
    s2 = ([0.5, 0.4, 0.3], [0, 0, 1])
    U.update_cumulative_auc(state, {"d": U.group_aucs([s1])})
    out = U.update_cumulative_auc(state, {"d": U.group_aucs([s2])})
    assert out["pi_utility/auc_d_cum"] == pytest.approx(0.0)
    # the pooled-across-steps alternative is misled: step 2's win (0.3) now outranks step 1's losses
    from opsd.pi.signals.metrics import roc_auc

    pooled = roc_auc(s1[0] + s2[0], list(s1[1]) + list(s2[1]))
    assert pooled > 0.0, f"pooling across steps must disagree, which is why we do not (got {pooled})"


def test_frac_tokens_d_minus_ignores_masked_positions():
    student = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    teacher = torch.zeros(1, 4)
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])  # keep one d- and one d+ token
    a = U.divergence_attributes(student, teacher, mask)
    assert a["frac_minus"].item() == pytest.approx(0.5)


# --------------------------------------------------------------------------- AUC: the trendable metric
# (ALFWorld line. `auc_task` is the ALFWorld-line name for `within_task_auc`; both are kept.)
def test_auc_is_invariant_to_win_rate_but_ndcg_is_not():
    """THE reason auc_task exists. Attribute quality held fixed; only the win rate varies.

    Measured: nDCG@4 climbs 0.62 -> 0.96 as wins go 1/8 -> 7/8 while the attribute is unchanged, and its
    margin over random simultaneously COMPRESSES (+0.30 -> +0.09) as headroom vanishes. Neither can be
    trended over training, where the win rate rises by design. AUC stays flat.
    """
    import random

    n, mu, tasks = 8, 1.0, 1500
    aucs, ndcgs = {}, {}
    for w in (1, 4, 7):
        rng = random.Random(1000 + w)  # per-w seed: no shared-stream confound between conditions
        groups = []
        for _ in range(tasks):
            labels = [1.0] * w + [0.0] * (n - w)
            scores = [rng.gauss(mu, 1) if y else rng.gauss(0, 1) for y in labels]
            groups.append((scores, labels))
        aucs[w] = U.auc_task(groups)
        ndcgs[w] = ndcg_at_k(groups, 4)

    auc_spread = max(aucs.values()) - min(aucs.values())
    ndcg_spread = ndcgs[7] - ndcgs[1]
    # The claim is RELATIVE: the win rate barely moves AUC while it moves nDCG a lot. Asserting the
    # relationship rather than an absolute tolerance keeps the test honest about its own sample size --
    # per-task AUC noise at 8 reps is ~0.02, so a tight absolute bound would just be flaky.
    assert auc_spread < 0.03, f"AUC drifted with win rate: {aucs}"
    assert ndcg_spread > 0.25, f"nDCG should inflate with win rate: {ndcgs}"
    assert ndcg_spread > 8 * auc_spread, f"auc_spread={auc_spread:.4f} ndcg_spread={ndcg_spread:.4f}"


def test_auc_anchor_is_one_half_for_an_inert_attribute():
    import random

    rng = random.Random(1)
    groups = []
    for _ in range(400):
        labels = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        groups.append(([rng.random() for _ in labels], labels))  # score independent of label
    assert U.auc_task(groups) == pytest.approx(0.5, abs=0.02)


def test_auc_is_one_for_a_perfect_attribute():
    groups = [([9.0, 8.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0])]
    assert U.auc_task(groups) == pytest.approx(1.0)


def test_win_rate_keys_explain_the_anchor():
    g = torch.Generator().manual_seed(3)
    a = U.divergence_attributes(torch.randn(8, 16, generator=g), torch.randn(8, 16, generator=g), torch.ones(8, 16))
    uids = ["t1"] * 4 + ["t2"] * 4
    succ = [1, 0, 0, 0, 1, 1, 1, 0]  # overall 4/8; both tasks both-class
    out = U.pi_utility_metrics(a, uids, succ, k=4)
    assert out["pi_utility/win_rate"] == pytest.approx(0.5)
    # t1 = 1/4, t2 = 3/4 -> macro mean 0.5
    assert out["pi_utility/win_rate_both_class"] == pytest.approx(0.5)
    assert "pi_utility/auc_task_d" in out and "pi_utility/auc_task_d_minus" in out


def test_win_rate_is_logged_even_when_no_task_is_both_class():
    """The win rate is exactly what explains an absent nDCG, so it must survive the filter."""
    a = U.divergence_attributes(torch.randn(4, 8), torch.randn(4, 8), torch.ones(4, 8))
    out = U.pi_utility_metrics(a, ["t1"] * 4, [1, 1, 1, 1], k=4)
    assert out["pi_utility/win_rate"] == pytest.approx(1.0)
    assert out["pi_utility/ndcg_both_class_tasks"] == 0.0
    assert "pi_utility/auc_task_d" not in out


# --------------------------------------------------------------------------- merged-module contract
def test_auc_task_keys_alias_the_auc_keys_with_identical_values():
    """Both dashboards' key sets exist after the ALFWorld/AppWorld merge, and they cannot drift: each
    `auc_task_*` (ALFWorld line) is a COPY of the matching `auc_*` (AppWorld scaffold), not a recompute.
    Only the three keys the ALFWorld line ever emitted are aliased."""
    g = torch.Generator().manual_seed(5)
    a = U.divergence_attributes(torch.randn(8, 16, generator=g), torch.randn(8, 16, generator=g), torch.ones(8, 16))
    out = U.pi_utility_metrics(
        a, ["t1"] * 4 + ["t2"] * 4, [1, 0, 0, 0, 1, 1, 0, 0], k=4, n_steps=[5, 9, 12, 30, 4, 6, 22, 28]
    )
    for name in ("d", "d_minus", "neg_nstep"):
        assert out[f"pi_utility/auc_task_{name}"] == out[f"pi_utility/auc_{name}"], name
    assert "pi_utility/auc_task_d_plus" not in out and "pi_utility/auc_task_d_signed" not in out
    # the ALFWorld line's selection trio at k=4 is still present under its old names
    for name in ("d", "d_minus", "neg_nstep"):
        for key in (f"recall_at_4_{name}", f"hit_at_4_{name}", f"best_at_1_{name}"):
            assert f"pi_utility/{key}" in out, key
    # the function-level alias agrees with the AppWorld function on arbitrary groups
    groups = U.group_by_task(["t1"] * 4 + ["t2"] * 4, a["d"].tolist(), [1, 0, 0, 0, 1, 1, 0, 0])
    assert U.auc_task(groups) == U.within_task_auc(groups)
    assert U.best_at_1(groups) == pytest.approx(
        sum(1.0 if max(zip(sc, lb))[1] > 0 else 0.0 for sc, lb in groups) / len(groups)
    )


def test_hit_at_k_any_win_in_top_k():
    # task 1: the win is ranked 3rd -> miss at k=2, hit at k=3; task 2: the win is ranked 1st -> hit
    groups = [([0.9, 0.8, 0.7, 0.1], [0, 0, 1, 0]), ([0.9, 0.1, 0.2], [1, 0, 0])]
    assert U.hit_at_k(groups, 2) == pytest.approx(0.5)
    assert U.hit_at_k(groups, 3) == pytest.approx(1.0)
    assert math.isnan(U.hit_at_k([([0.1, 0.2], [0, 0])], 2))  # all-loss groups are skipped, not scored 0


def test_residualize_within_task_removes_a_linear_covariate_and_counts_skips():
    """Retired for n_steps (it regresses out the outcome) but kept for an exogenous covariate."""
    uids = ["A"] * 4 + ["B"] * 2 + ["C"] * 3
    cov = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 5.0, 5.0, 5.0]
    scores = [2 * x + 1 for x in cov[:4]] + [7.0, 8.0] + [0.1, 0.2, 0.3]
    resid, n_fit, n_skip = U.residualize_within_task(uids, scores, cov)
    assert (n_fit, n_skip) == (1, 2)  # B has < 3 rollouts; C has zero covariate variance
    assert resid[:4] == pytest.approx([0.0] * 4, abs=1e-9)  # exact linear fit
    assert resid[4:] == pytest.approx(scores[4:])  # skipped tasks pass through unchanged

"""Online PI-utility attributes: `d`, `d+`, `d-`, `dsigned`, and their within-task nDCG@k and ROC-AUC.

These are the metrics from the OFFLINE PI study (the per-format PI table built from rows.parquet), computed
live in the trainer instead of by a separate re-scoring job. The definitions here mirror that study exactly
-- if they drift, the online numbers stop being comparable to the published table, which is the whole point
of logging them.

WHY THIS IS FREE. The offline pipeline re-runs two forward passes over dumped trajectories: the student's
own tokens scored WITHOUT the hint, and the same tokens scored WITH the hint spliced into the prompt
(analyze_pi_positions.py:439-446 -- it never generates a hinted rollout). The OPSD trainer already has both:
`action_log_probs` is the no-hint student and `hinted_log_probs` is the with-hint teacher, over identical
tokens. So this is a few masked reductions, not a second forward.

    dsigned = log pi_S(y_t) - log pi_T(y_t)            per token, over the student's emitted tokens
    d+      = mean max(-dsigned, 0)   teacher likes the emitted token MORE -> agreement / reinforcement
    d-      = mean max(+dsigned, 0)   teacher likes it LESS                -> disagreement
    d       = mean |dsigned|          identity: d = d+ + d- ; dsigned = d- - d+

THE +/- NAME THE DIRECTION OF REINFORCEMENT, NOT THE SIGN OF dsigned. Easy to get backwards, which is why
the offline analysis asserts the identity and so does `divergence_attributes`.

AGGREGATION. Per trajectory, the TRUE per-token mean over its masked tokens -- not a mean of per-bucket
means. The (step, channel) bucket means in the offline scorer's rows.parquet are "averages of averages" and
are NOT what matched-pair tests should use; the offline reference recomputes the true per-trajectory means
(`d_true`, `dplus_true`, `dminus_true`) from the raw tokens. The arm-level number is then the plain mean over
trajectories.

TWO RANKING VIEWS, AND WHY BOTH. `ndcg_at_k` is TOP-HEAVY: it answers "would taking the best k rollouts by
this attribute get me wins", the question a PI SELECTOR asks. `within_task_auc` is RANK-UNIFORM and
ANTISYMMETRIC about 0.5 (``AUC(-x) == 1 - AUC(x)``): it answers "does this attribute separate wins from
losses at all, and in which direction", the question a DIAGNOSTIC asks. The antisymmetry is why AUC was
added -- on AppWorld the attributes are INVERTED, and nDCG can only report "below a baseline that itself
depends on the win rate" while AUC reports 0.25-vs-0.75 directly. Read AUC first to get the direction, then
nDCG for whether the top of the ranking is usable.

nDCG IS WITHIN-TASK, ALWAYS. So is every `auc_<attr>`. The `auc_pooled_<attr>` keys are the marked
exception and are CONFOUND GAUGES, not results: read `auc_pooled_X - auc_X` as "how much of X's apparent
signal is task difficulty", and never quote a pooled number on its own -- see `within_task_auc`. Rank one
task's rollouts by the attribute, take the top k, normalise by that task's ideal ordering, relevance =
binary success; macro-average over BOTH-CLASS tasks only (>=1 win and >=1 loss). Pooling across tasks
answers a different question and is systematically misleading -- three attributes FLIP SIGN between pooled
and within-task in our ALFWorld analysis, because tasks sit at different absolute attribute levels.

READ THE NUMBERS WITH THESE THREE FACTS OR NOT AT ALL:
  * The random baseline on ALFWorld is ~0.293, not 0. `ndcg_random_expectation` gives the exact value.
    Using the base rate instead understates it (0.247 vs 0.293) and makes an inert attribute look
    marginally positive.
  * `-nstep` (shorter trajectory) ranks success far better than any divergence attribute -- nDCG@4 0.9432
    vs d- 0.65-0.73. A good `ndcg_at_4_d` is not evidence the hint is informative until it beats length.
  * **ON APPWORLD THE SIGN IS FLIPPED: `d` ANTI-correlates with success, so nDCG@4 lands BELOW random and
    the informative ranker is `-d`.** Measured 2026-08-12 on the six Instruct arms: `ndcg_at_4_d` beat its
    own random baseline in **0 of 65** logged steps on opsd_awi_onskill_qwen37_v2 and **14 of 58** on
    opsd_awi_self125_nothink -- both under the 50% a sign-neutral attribute would give, with
    `ndcg_both_class_tasks` reaching 14/18, so this is not a small-sample artifact. nDCG ranks by
    DESCENDING attribute, so an attribute that marks failures puts them first and scores below chance. The
    plausible reading is that a failing AppWorld rollout is the one that least resembles what the
    hinted teacher would have done. Consequence: **do not use `ndcg_at_4_*` to select PI on AppWorld
    without negating the attribute** -- the magnitude is the signal, the direction is env-specific, and
    the ALFWorld numbers quoted above do NOT transfer. `auc_d` is the panel that shows this at a glance:
    expect ~0.25 on AppWorld (the offline corpus measured 0.21-0.29) against >0.5 on ALFWorld.
  * On ALFWorld `OPSD_TOKEN_CLIP` defaults to **0.5** while AppWorld runs it at 0. This module never
    clips, so on ALFWorld `pi_utility/d_plus` is the UNCLIPPED attribute while `distill/side_mean_abs`
    (reward path) is the clipped one -- they will not match there, and that is correct, not a bug.
And note rho(utility, OPSD) = -0.543: these are DIAGNOSTICS, not an objective to optimise.

WHAT `d` DOES TRACK RELIABLY: distillation progress. It falls monotonically as the student converges on the
PI teacher -- measured -50% (0.4567 -> 0.2293 over 58 steps) and -57% (0.2721 -> 0.1182 over 65 steps) on
the two arms above, with `d_plus` falling alongside it. It plateaus rather than reaching zero (~0.22 / ~0.11),
which is expected: the PI block shifts the teacher's distribution in ways the student cannot match without
seeing the hint. A `d` that is FLAT across steps on an `ENV_COEF=0` arm means the distillation term is not
reaching the policy -- that is the useful alarm this metric supports, rather than ranking.

TWO KEY VOCABULARIES, ONE VALUE. This module is the union of two independently grown copies (the ALFWorld
line and the AppWorld scaffold), and existing W&B dashboards read both key sets. Where they name the SAME
quantity, both keys are emitted with the identical value:
  * `auc_task_d`, `auc_task_d_minus`, `auc_task_neg_nstep` (ALFWorld-line names) are ALIASES of
    `auc_d`, `auc_d_minus`, `auc_neg_nstep` -- the same within-task, both-class, macro-averaged ROC-AUC.
  * `recall_at_4_*` and `best_at_1_*` already coincide (same name, same definition, same tie rule).
ALFWorld-line keys with no AppWorld counterpart are emitted as before: `win_rate` (policy success rate,
logged even when no task is both-class), `win_rate_both_class` (the mean w/n that SETS
`ndcg_at_k_random`), and `hit_at_<k>_{d,d_minus,neg_nstep}`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch

from opsd.pi.signals.metrics import (
    best_at_k,
    dcg,
    ndcg_at_k,
    oracle_at_k,
    random_at_k,
    random_recall_at_k,
    recall_at_k,
    roc_auc,
)

# Matches the trainer's own sanitisation bound (SelfDistillationTrainer.KL_CLAMP).
KL_CLAMP = 20.0


def divergence_attributes(
    student_lp: torch.Tensor,
    teacher_lp: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    kl_clamp: float = KL_CLAMP,
) -> Dict[str, torch.Tensor]:
    """Per-trajectory `d`, `d_plus`, `d_minus`, `dsigned` as the true per-token mean over masked tokens.

    NB the OPSD per-token clip (``OPSD_TOKEN_CLIP``, default 0.5) is deliberately NOT applied. That clip
    exists to bound the training PRESSURE; applying it here would saturate every attribute at 0.5 and
    destroy the magnitude information the metric is made of. Only the NaN/inf sanitisation and the
    ``kl_clamp`` bound are shared with the reward path.

    Returns tensors of shape [batch]; trajectories with no unmasked tokens yield 0.0 rather than NaN.
    """
    valid = loss_mask.bool()
    diff = torch.nan_to_num(student_lp - teacher_lp, nan=0.0, posinf=kl_clamp, neginf=-kl_clamp)
    diff = diff.clamp(-kl_clamp, kl_clamp)
    diff = torch.where(valid, diff, torch.zeros_like(diff))

    n_tok = valid.sum(dim=-1).clamp(min=1).to(diff.dtype)
    d_signed = diff.sum(dim=-1) / n_tok
    d_abs = diff.abs().sum(dim=-1) / n_tok
    d_plus = torch.clamp(-diff, min=0).sum(dim=-1) / n_tok
    d_minus = torch.clamp(diff, min=0).sum(dim=-1) / n_tok
    # TOKEN share on the d- side, as opposed to the MAGNITUDE share d-/d. The two answer different
    # questions and on AppWorld they disagree: d- carries 79-92% of the magnitude, and whether that is
    # "most tokens, mildly disliked" or "a few tokens, strongly disliked" decides what a one-sided
    # objective (OPSD_DISTILL_SIDE) actually does to the policy. Not part of the offline table.
    frac_minus = ((diff > 0) & valid).sum(dim=-1).to(diff.dtype) / n_tok
    return {"d": d_abs, "d_plus": d_plus, "d_minus": d_minus, "d_signed": d_signed, "frac_minus": frac_minus}


def check_identity(attrs: Dict[str, torch.Tensor], tol: float = 1e-4) -> float:
    """Max violation of ``d == d+ + d-``. The offline table asserts this; so should we.

    It is the one check that catches a sign flip in d+/d-, which is otherwise invisible: both remain
    positive, both move plausibly, and only their MEANING swaps.
    """
    return float((attrs["d_plus"] + attrs["d_minus"] - attrs["d"]).abs().max().item())


def group_by_task(
    uids: Sequence[str], scores: Sequence[float], successes: Sequence[float]
) -> List[Tuple[List[float], List[float]]]:
    """`(scores, labels)` per task, keeping ONLY both-class tasks (>=1 win AND >=1 loss).

    A task where every rollout won (or every one lost) has a degenerate ideal ordering and contributes no
    ranking information; the offline study filters them the same way.
    With 16 tasks per training step this filter can be aggressive -- log how many survive, or a number
    computed from 2 tasks reads exactly like one computed from 16.
    """
    by: Dict[str, Tuple[List[float], List[float]]] = {}
    for uid, s, y in zip(uids, scores, successes):
        sc, lb = by.setdefault(str(uid), ([], []))
        sc.append(float(s))
        lb.append(float(y))
    out = []
    for sc, lb in by.values():
        w = sum(1 for v in lb if v > 0)
        if 0 < w < len(lb):
            out.append((sc, lb))
    return out


def ndcg_random_expectation(groups: Sequence[Tuple[List[float], List[float]]], k: int) -> float:
    """EXACT expected nDCG@k of a random ordering, macro-averaged over the same groups.

    Closed form rather than a simulation: by linearity each of the k positions
    holds a win with probability w/n, so E[DCG] = (w/n) * sum(discounts) and IDCG is deterministic. Exact,
    free, and zero-variance -- a sampled baseline would add noise to the very number it is meant to anchor.
    """
    vals = []
    for _scores, labels in groups:
        n = len(labels)
        if n == 0:
            continue
        w = sum(1 for v in labels if float(v) > 0)
        kk = min(k, n)
        e_dcg = (w / n) * sum(1.0 / math.log2(i + 2) for i in range(kk))
        idcg = dcg(sorted((float(v) for v in labels), reverse=True)[:k])
        if idcg > 0:
            vals.append(e_dcg / idcg)
    return sum(vals) / len(vals) if vals else float("nan")


def within_task_auc(groups: Sequence[Tuple[List[float], List[float]]]) -> float:
    """Mean ROC-AUC of the attribute against success, computed WITHIN each task then macro-averaged.

    WHY AUC EARNS ITS PLACE NEXT TO nDCG, rather than duplicating it. AUC is antisymmetric about 0.5 --
    ``AUC(-x) == 1 - AUC(x)`` exactly -- so it reads a sign flip off the number itself: 0.5 is chance,
    0.75 means the attribute marks success, 0.25 means it marks FAILURE just as strongly. nDCG has no such
    property: its chance level depends on each task's win rate (~0.293 on ALFWorld, and `ndcg_at_4_random`
    exists only to recover it), and an inverted attribute merely lands "somewhere below" that baseline
    with no way to read off how strong the inversion is. That is exactly the situation on AppWorld, where
    `d` ANTI-correlates with success (offline AUC 0.21-0.29) and `ndcg_at_4_d` could only say "below
    random in 0 of 65 steps". AUC turns that into a magnitude.

    Complementary, not redundant: nDCG@k is top-heavy and answers "would picking the top k by this
    attribute get me wins", which is the question a PI SELECTOR asks; AUC is rank-uniform and answers "does
    this attribute separate wins from losses at all", which is the question a DIAGNOSTIC asks.

    WITHIN-TASK, for the same reason nDCG is: ALFWorld difficulty spreads the per-task attribute level by
    17x, and pooling was DIRECTIONALLY WRONG for 2 of 3 analyzers in our runs. `groups` must already
    be both-class-filtered (`group_by_task` does it), since AUC is undefined on a single class.

    ``nan`` if no group yields a defined AUC -- W&B drops nan, so an absent panel means "no both-class
    task this step", which is the honest reading.
    """
    vals = group_aucs(groups)
    return sum(vals) / len(vals) if vals else float("nan")


def group_aucs(groups: Sequence[Tuple[List[float], List[float]]]) -> List[float]:
    """Per-group AUCs with undefined ones dropped. Exposed so the cumulative accumulator can sum the
    SAME per-group values the per-step macro-average is built from, instead of averaging averages."""
    return [v for v in (roc_auc(sc, lb) for sc, lb in groups) if v == v]


def auc_task(groups: Sequence[Tuple[List[float], List[float]]]) -> float:
    """Macro-averaged WITHIN-TASK AUC = P(attribute(win) > attribute(loss)), ties 0.5.

    The ALFWorld-line name for `within_task_auc` -- the same quantity, kept so existing callers and the
    `auc_task_*` W&B keys keep resolving. Identical value by construction (it delegates).

    THIS IS THE ONE SAFE TO TREND OVER TRAINING. nDCG@k is inflated by the win rate -- measured, at fixed
    attribute quality, 0.622 -> 0.960 as wins go 1/8 -> 7/8 -- and its margin over random is simultaneously
    COMPRESSED by the shrinking headroom (+0.302 -> +0.085). So both the raw nDCG and its margin move
    substantially with the policy's success rate while the attribute itself is unchanged, and neither can be
    read as a trend. AUC is invariant by construction (a pairwise probability, indifferent to class
    balance): the same simulation holds it flat at 0.760 across every win rate.

    The offline study reaches the same conclusion and calls this the honest per-task number
    (per-task vs pooled AUC). 0.5 is the fixed no-signal anchor.
    """
    return within_task_auc(groups)


def _rank_desc(scores: Sequence[float], labels: Sequence[float]) -> List[float]:
    """Labels reordered by DESCENDING score.

    TIE POLICY: `sorted` is stable, so exact ties keep rollout order -- the same rule as
    `metrics._topk_labels_by_score`, so `hit_at_k` and the imported `recall_at_k`/`best_at_k` agree on
    ties. The divergence attributes are continuous per-token means so exact ties are effectively
    measure-zero; the case that DOES tie is a degenerate attribute (all-zero because something upstream
    broke) -- there the ordering is rollout order, the metrics read like random, and reading them as random
    is the correct conclusion.
    """
    order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
    return [float(labels[i]) for i in order]


def hit_at_k(groups: Sequence[Tuple[List[float], List[float]]], k: int) -> float:
    """Macro-averaged within-task hit@k: 1 if ANY win is in the top k.

    NOTE hit@1 IS IDENTICAL TO best_at_1 -- both ask "is rank 1 a win". Only logged for the step's `k`
    (4 in every caller) so the two do not appear as independent evidence for the same fact.
    """
    vals = []
    for sc, lb in groups:
        if not any(float(v) > 0 for v in lb):
            continue
        top = _rank_desc(sc, lb)[:k]
        vals.append(1.0 if any(v > 0 for v in top) else 0.0)
    return sum(vals) / len(vals) if vals else float("nan")


def best_at_1(groups: Sequence[Tuple[List[float], List[float]]]) -> float:
    """Top-1 selection accuracy: is the single highest-attribute rollout a win?

    Kept for API compatibility with the ALFWorld line; delegates to `best_at_k(groups, 1)`, which for k=1 is
    exactly the macro mean over groups (every group contributes one pick). Its no-signal anchor is the base
    rate (`win_rate_both_class` / `best_at_k_random`), NOT 0.5.
    """
    return best_at_k(groups, 1)


def residualize_within_task(
    uids: Sequence[str], scores: Sequence[float], covariate: Sequence[float]
) -> Tuple[List[float], int, int]:
    """Per-task OLS of `scores` on `covariate`; returns (residuals, n_tasks_fit, n_tasks_skipped).

    NOT USED WITH n_steps ANY MORE -- see the retired lin-task block in ``pi_utility_metrics``:
    within a task, trajectory length is a near-perfect proxy for the OUTCOME (best_at_1_neg_nstep
    measured exactly 1.0 in six arms), so residualizing on it removes the label. Kept for a
    genuinely exogenous covariate (and for API compatibility).

    ORIGINAL RATIONALE, now known to be misapplied: `-nstep` ranks success far better
    than any divergence attribute (nDCG@4 0.9432 vs d- 0.65-0.73), so an attribute that appears to rank
    wins may be tracking TRAJECTORY LENGTH rather than anything the hint contributed. Regressing the
    attribute on length WITHIN each task and ranking the residual asks the sharper question: does the
    attribute rank wins beyond what length alone already explains? If `auc_task_d_lintask` collapses toward
    0.5 while `auc_task_d` looks healthy, the apparent PI signal WAS length.

    Within-task is the only correct level. Tasks sit at different absolute attribute levels AND different
    typical lengths, so a pooled regression would remove between-task variation that is not the confound --
    the same reason nDCG is within-task here (three attributes flip sign between pooled and within-task).

    A task with fewer than 3 rollouts, or zero variance in the covariate, cannot support a fit; those pass
    through UNCHANGED and are counted, because silently leaving them raw would mix residualized and
    non-residualized scores in one ranking.
    """
    by: Dict[str, List[int]] = {}
    for i, u in enumerate(uids):
        by.setdefault(str(u), []).append(i)
    resid = [float(s) for s in scores]
    n_fit = n_skip = 0
    for idxs in by.values():
        if len(idxs) < 3:
            n_skip += 1
            continue
        xs = [float(covariate[i]) for i in idxs]
        ys = [float(scores[i]) for i in idxs]
        mx = sum(xs) / len(xs)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx <= 1e-12:
            n_skip += 1
            continue
        my = sum(ys) / len(ys)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        a = my - b * mx
        for i in idxs:
            resid[i] = float(scores[i]) - (a + b * float(covariate[i]))
        n_fit += 1
    return resid, n_fit, n_skip


def update_cumulative_auc(
    state: Dict[str, List[float]], group_aucs_by_attr: Dict[str, List[float]]
) -> Dict[str, float]:
    """Running macro-average AUC over EVERY both-class group seen since the run started.

    WHY THIS IS THE NUMBER TO READ, and the per-step one is not. A step contributes 18 tasks, of which only
    the both-class ones can be scored -- observed 3-7 of 18 on these arms. So `auc_d` at any single step is
    a macro-average over a handful of groups and swings wildly; a run's direction is not decidable from it.
    The cumulative version tightens monotonically and answers "over this run so far, does the attribute
    separate wins from losses" with one number.

    GROUPED PER (task, step), NEVER PER TASK ACROSS STEPS. Accumulating one task's rollouts across steps
    would pool policies from different points in training, and since `d` falls monotonically while success
    rises, that confound alone manufactures a spurious negative association -- the same pooling trap as
    `auc_pooled_d`, one level up. Keeping each step's groups intact and averaging over groups makes this a
    strictly larger sample of the identical quantity, not a different one.

    Averaged over GROUPS, not over steps: a step contributing 7 scorable groups should weigh more than one
    contributing 3, and averaging the per-step macro-averages would weight them equally.

    O(1) memory -- only (sum, count) per attribute, no stored rollouts.

    NOT CHECKPOINTED: a resumed run restarts its accumulator from empty. `auc_groups_cum` is the honest
    denominator to read next to it, and a sudden drop in that count means "resumed", not "regressed".
    """
    out: Dict[str, float] = {}
    for name, vals in group_aucs_by_attr.items():
        acc = state.setdefault(name, [0.0, 0.0])
        acc[0] += float(sum(vals))
        acc[1] += float(len(vals))
        if acc[1] > 0:
            out[f"pi_utility/auc_{name}_cum"] = acc[0] / acc[1]
    if "d" in state and state["d"][1] > 0:
        out["pi_utility/auc_groups_cum"] = state["d"][1]
    return out


def pi_utility_metrics(
    attrs: Dict[str, torch.Tensor],
    uids: Sequence[str],
    successes: Sequence[float],
    *,
    k: int = 4,
    n_steps: Sequence[float] | None = None,
    auc_cum_state: Dict[str, List[float]] | None = None,
) -> Dict[str, float]:
    """The `pi_utility/*` series for one training step.

    ``n_steps`` (turns per trajectory) is optional but strongly recommended: `-nstep` is a far better
    success-ranker than any divergence attribute (nDCG@4 0.9432 vs 0.65-0.73), so logging it alongside is
    what keeps `ndcg_at_4_d` honest.
    """
    d_mean = float(attrs["d"].mean().item())
    out: Dict[str, float] = {
        "pi_utility/d": d_mean,
        "pi_utility/d_plus": float(attrs["d_plus"].mean().item()),
        "pi_utility/d_minus": float(attrs["d_minus"].mean().item()),
        "pi_utility/d_signed": float(attrs["d_signed"].mean().item()),
        "pi_utility/identity_err": check_identity(attrs),
    }
    # How one-sided the objective is, as a ratio rather than something to eyeball off two panels. This is
    # the number that sizes an OPSD_DISTILL_SIDE ablation: it measured 0.79-0.92 on all six AppWorld
    # Instruct arms, i.e. vanilla OPSD there is already ~85% SUPPRESSION, so masking d- barely changes the
    # dose while masking d+ cuts it ~8x.
    if d_mean > 1e-12:
        out["pi_utility/d_minus_share"] = float(attrs["d_minus"].mean().item()) / d_mean
    if "frac_minus" in attrs:
        out["pi_utility/frac_tokens_d_minus"] = float(attrs["frac_minus"].mean().item())

    # Win rate, logged so the movement of the nDCG anchor is explicit rather than inferred. `win_rate` is
    # the policy's actual success rate this step -- emitted BEFORE the both-class filter, because it is
    # exactly what explains an absent nDCG/AUC. `win_rate_both_class` (below) is the mean w/n over the tasks
    # that survive the filter, and it is the quantity that actually SETS ndcg_at_k_random. When it rises,
    # expect ndcg_at_4_* to rise and its margin over random to shrink, both for free.
    if len(successes) > 0:
        out["pi_utility/win_rate"] = float(sum(1 for y in successes if float(y) > 0) / len(successes))

    lists = {name: attrs[name].detach().float().cpu().tolist() for name in ("d", "d_plus", "d_minus", "d_signed")}

    g_d = group_by_task(uids, lists["d"], successes)
    out["pi_utility/ndcg_both_class_tasks"] = float(len(g_d))
    if g_d:
        out[f"pi_utility/ndcg_at_{k}_d"] = float(ndcg_at_k(g_d, k))
        out[f"pi_utility/ndcg_at_{k}_d_minus"] = float(ndcg_at_k(group_by_task(uids, lists["d_minus"], successes), k))
        out[f"pi_utility/ndcg_at_{k}_random"] = float(ndcg_random_expectation(g_d, k))
        # AUC on the SAME both-class groups, so it and the nDCG describe one population. Unmarked name =
        # WITHIN-TASK, matching the nDCG convention in this module; `auc_pooled_d` is the marked exception.
        # 0.5 is chance for all of these -- no companion baseline key, which is the point of using AUC.
        # Each attribute gets BOTH views, from the same values, so the pooled-minus-within gap is
        # computable per attribute. `auc_pooled_*` uses NO task grouping at all: one Mann-Whitney over
        # every rollout in the step (144 here) against the binary success labels.
        #
        # POOLED IS A CONFOUND GAUGE, NEVER A RESULT. The gap `auc_pooled_X - auc_X` is how much of X's
        # apparent signal is task DIFFICULTY rather than within-task discrimination; pooling reversed the
        # direction outright for 2 of 3 ALFWorld analyzers, so a pooled number quoted alone is worse than
        # no number.
        #
        # WHY PER-ATTRIBUTE AND NOT JUST FOR `d` (which is what this first shipped as). The gap is NOT
        # shared across the attributes, because they carry task difficulty differently: `d_minus` is what
        # grows on a hard task, while `d_plus` is nearly task-INVARIANT -- measured at 0.035-0.044 across
        # all six AppWorld Instruct arms regardless of PI source or density. And since `d` is ~92% `d_minus`
        # on AppWorld, a gauge on `d` is very nearly a gauge on `d_minus` alone and says nothing about the
        # d+ side. Expect a LARGE pooled-vs-within gap on `d`/`d_minus` and a SMALL one on `d_plus`; a
        # large gap on `d_plus` would mean d+ has picked up task-level structure it did not have before,
        # which is worth knowing.
        per_group: Dict[str, List[float]] = {}
        for name in ("d", "d_plus", "d_minus", "d_signed"):
            per_group[name] = group_aucs(group_by_task(uids, lists[name], successes))
            vals = per_group[name]
            out[f"pi_utility/auc_{name}"] = (sum(vals) / len(vals)) if vals else float("nan")
            out[f"pi_utility/auc_pooled_{name}"] = float(roc_auc(lists[name], list(successes)))
        if n_steps is not None:
            # negated: FEWER turns should rank first
            neg = [-float(v) for v in n_steps]
            g_ns = group_by_task(uids, neg, successes)
            out[f"pi_utility/ndcg_at_{k}_neg_nstep"] = float(ndcg_at_k(g_ns, k))
            # The reference that keeps every divergence AUC honest: on ALFWorld `-nstep` ranks success far
            # better than any divergence attribute, so a divergence AUC only means something once it beats
            # THIS one, not once it beats 0.5. Its POOLED counterpart is the textbook confound case --
            # trajectory length is task-determined far more than it is success-determined.
            per_group["neg_nstep"] = group_aucs(g_ns)
            vals = per_group["neg_nstep"]
            out["pi_utility/auc_neg_nstep"] = (sum(vals) / len(vals)) if vals else float("nan")
            out["pi_utility/auc_pooled_neg_nstep"] = float(roc_auc(neg, list(successes)))
        # ALFWorld-line key names for the SAME within-task AUC (module docstring, "TWO KEY VOCABULARIES"):
        # copied, never recomputed, so the two dashboards cannot drift apart. Only the three keys that line
        # ever emitted are aliased; d_plus / d_signed never had an `auc_task_*` panel.
        for name in ("d", "d_minus", "neg_nstep"):
            if f"pi_utility/auc_{name}" in out:
                out[f"pi_utility/auc_task_{name}"] = out[f"pi_utility/auc_{name}"]
        out["pi_utility/win_rate_both_class"] = float(
            sum(sum(1 for v in lb if v > 0) / len(lb) for _sc, lb in g_d) / len(g_d)
        )
        # ---- CURATION@k: recall@k and best@k (precision@k) at k = 1, 2, 4 -----------------------------
        # THE PAIR IS THE POINT, not either number. best@k is precision -- of the k picked, how many won --
        # so it is CAPPED BY THE WIN COUNT: a task with 1 win in 8 cannot exceed 1/k however good the
        # ranker is. recall@k asks the complement -- of the wins that exist, how many surfaced -- and hits
        # 1.0 on that same task. They diverge exactly where per-task win counts are unbalanced, which here
        # they are (n=8 reps, per-task wins run 1..7), so quoting one alone flatters or penalises the
        # ranker according to that step's win-rate mix.
        #
        # NEITHER HAS A FIXED CHANCE LEVEL, unlike AUC's 0.5 -- and that is why the baselines below are
        # emitted rather than left to be recomputed by eye. recall@k's chance is k/n (0.125/0.25/0.5 at
        # n=8) and best@k's is the task win RATE, so `best_at_1 = 0.5` is excellent at a 12% win rate and
        # poor at 70%. `random_at_k` (mean per-task win rate) and `oracle_at_k` (the achievable ceiling)
        # bracket it: read best@k as its position between them, never as an absolute.
        #
        # SAME both-class groups as the AUC/nDCG above, deliberately: the module's convention is that
        # every ranking metric describes ONE population per step, so a difference between them is about
        # the metric and never about which tasks it happened to see. NB recall/ndcg additionally skip
        # all-loss groups internally (0/0 is undefined), which for both-class groups is a no-op.
        #
        # ALL FIVE ATTRIBUTES get the full set, matching how AUC is emitted. `d_plus` is nearly
        # task-invariant (0.021-0.044 measured) so its curation numbers may be uninformative -- that is
        # itself worth seeing rather than assuming, and `neg_nstep` stays the honest reference: it ranks
        # success far better than any divergence attribute (nDCG@4 0.9432 vs 0.65-0.73), so a divergence
        # curation number only means something once it beats THAT, not once it beats chance.
        #
        # The step's own `k` is folded in so `recall_at_<k>_*` exists for any k a caller passes (the
        # ALFWorld line emitted it at `k`); with the default k=4 this is exactly (1, 2, 4).
        _CURATION_KS = tuple(sorted({1, 2, 4, int(k)}))
        _groups_for: Dict[str, List[Tuple[List[float], List[float]]]] = {
            name: group_by_task(uids, lists[name], successes) for name in ("d", "d_plus", "d_minus", "d_signed")
        }
        if n_steps is not None:
            _groups_for["neg_nstep"] = g_ns
        for name, grps in _groups_for.items():
            for kk in _CURATION_KS:
                out[f"pi_utility/recall_at_{kk}_{name}"] = float(recall_at_k(grps, kk))
                out[f"pi_utility/best_at_{kk}_{name}"] = float(best_at_k(grps, kk))
        # Label-only baselines: identical for every attribute, so emitted ONCE off the `d` groups (all
        # attributes share the same groups, hence the same labels). Duplicating them per attribute would
        # add 20 keys that are guaranteed equal.
        for kk in _CURATION_KS:
            out[f"pi_utility/recall_at_{kk}_random"] = float(random_recall_at_k(g_d, kk))
            out[f"pi_utility/best_at_{kk}_oracle"] = float(oracle_at_k(g_d, kk))
        out["pi_utility/best_at_k_random"] = float(random_at_k(g_d))
        # hit@k ("any win in the top k"), ALFWorld-line keys, at the step's `k` only and only for the
        # attributes that line logged it for. hit@1 is omitted: it is best_at_1 by another name.
        for name in ("d", "d_minus", "neg_nstep"):
            if name in _groups_for:
                out[f"pi_utility/hit_at_{k}_{name}"] = float(hit_at_k(_groups_for[name], k))
        # ---- lin-task: RETIRED, and the reason is worth keeping ----------------------------------------
        # The intent was: auc_task_d may look healthy only because d tracks trajectory length, which by
        # itself ranks success extremely well, so residualize d on n_steps within task and re-rank.
        # MEASURED, AND IT DOES NOT WORK, because the premise is wrong. Within a task, trajectory length is
        # not a CONFOUNDER of success -- it is very nearly the OUTCOME. Across the six summariser arms,
        # auc_task_neg_nstep ran 0.948-1.000 and best_at_1_neg_nstep was EXACTLY 1.0 in every arm: picking
        # the shortest rollout picks a winner every single time. The mechanism is structural, not
        # statistical -- a solved ALFWorld episode terminates when the goal is met (~12-15 turns) while an
        # unsolved one runs to the 30-turn cap, so length and success are nearly the same variable.
        #
        # Regressing d on length within task therefore regresses out a near-perfect proxy for the LABEL.
        # Whatever variance in d predicts success is by construction the variance correlated with length,
        # so removing it must drive AUC to chance -- and it did: all six arms collapsed into 0.473-0.601
        # while their RAW auc_task_d spanned 0.487-0.924. That is an artifact of controlling for the
        # outcome, not a finding about the hints, and it destroys exactly the between-arm discrimination
        # the metric was meant to protect.
        #
        # WHAT TO USE INSTEAD. Raw auc_task_d (== auc_d) is the valid cross-arm comparator here: length's
        # ranking power is essentially CONSTANT across arms (0.948-1.000), and a confounder that does not
        # vary between conditions cannot produce between-condition differences. Keep auc_neg_nstep logged
        # as the sanity check that length dominates (it does, everywhere).
        #
        # The length confound that IS real is HINT length, which varies systematically by arm (25-114
        # median words) and is fixed BEFORE the rollout, so it is a genuine covariate rather than an
        # outcome. Controlling for that requires a cross-arm model, not a within-task one, and so cannot
        # live in this per-step function. `residualize_within_task` is kept: it is correct code for a
        # legitimately exogenous covariate (unit-tested in test_pi_utility.py); it is simply not applicable
        # to n_steps.
        # CUMULATIVE AUC -- the stable read. With only 3-7 both-class tasks per step, the per-step numbers
        # above swing too much to decide a direction from; these tighten monotonically over the run.
        if auc_cum_state is not None:
            out.update(update_cumulative_auc(auc_cum_state, per_group))
    return out

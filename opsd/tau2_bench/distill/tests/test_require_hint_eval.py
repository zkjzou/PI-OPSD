"""OPSD_REQUIRE_HINT must fire on training batches and must NOT fire on eval batches.

``OPSD_REQUIRE_HINT=1`` turns a MISSING per-task hint (not only an overflow drop) into a hard failure, so
an arm cannot silently train on the generic static hint. ``generate()`` builds the with-hint ref prompts
for every batch, including the step-0 eval that ``EVAL_BEFORE_TRAIN=true`` runs first. But the τ² eval
split (``retail_base``, task ids '0', '1', ...) is DISJOINT from every training pool (``retail_<n>`` ids)
by design, so no eval task has a per-task hint. Enforcing the flag there raises on the very first eval
prompt and ends the run before step 1, and such a run can be mistaken for one that finished with zero
training steps.

The ref prompts are not consumed at eval time, so the whole hint path is inert there; enforcing it is pure
downside. These tests pin both halves (train batches stay strict, eval batches are lenient) and that the
strict behaviour is the default.
"""

import types

import pytest

from opsd.tau2_bench.distill.generator import Tau2OPSDGenerator


def _gen(hints):
    """A generator with __init__ bypassed -- we exercise _build_ref_prompts only."""
    g = object.__new__(Tau2OPSDGenerator)
    g._hints_by_task = dict(hints)
    g._static_hint = "STATIC"
    g._require_hint = True
    g._seq_cap = 10_000
    g._dropped_total = 0
    g._rollouts_total = 0
    g.tokenizer = types.SimpleNamespace(encode=lambda s, add_special_tokens=False: [0] * 5)
    return g


TRAIN_EXTRAS = [{"task_id": "retail_10", "domain": "retail"}]
EVAL_EXTRAS = [{"task_id": "0", "domain": "retail"}]  # retail_base id namespace
HINTS = {"retail_10": "a real hint"}


def test_train_batch_still_raises_on_missing_hint():
    g = _gen(HINTS)
    with pytest.raises(RuntimeError, match="no per-task hint"):
        g._build_ref_prompts([[1, 2]], [[3]], [{"task_id": "retail_999"}], enforce=True)


def test_eval_batch_does_not_raise():
    g = _gen(HINTS)
    ref, hints, m = g._build_ref_prompts([[1, 2]], [[3]], EVAL_EXTRAS, enforce=False)
    assert len(ref) == 1 and m["opsd/hint_static_fallback"] == 1.0


def test_train_batch_with_hint_is_unaffected():
    g = _gen(HINTS)
    ref, hints, m = g._build_ref_prompts([[1, 2]], [[3]], TRAIN_EXTRAS, enforce=True)
    assert m["opsd/hinted"] == 1.0 and m["opsd/hint_static_fallback"] == 0.0
    assert len(ref[0]) == 5 + 2, "ref must be the encoded hint block PREPENDED to prompt_ids"


def test_enforce_defaults_to_true():
    """A caller that forgets the kwarg must get the strict behaviour, not the lenient one."""
    g = _gen(HINTS)
    with pytest.raises(RuntimeError, match="no per-task hint"):
        g._build_ref_prompts([[1, 2]], [[3]], [{"task_id": "nope"}])

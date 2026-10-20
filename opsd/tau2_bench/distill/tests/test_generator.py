"""Unit tests for Tau2OPSDGenerator's with-hint ref-prompt construction (no τ² infra / rollouts).

Uses ``object.__new__`` to bypass ``Tau2Generator.__init__`` (which builds litellm specs and resolves a
policy URL) plus a fake tokenizer, so these exercise the pure ref-prompt logic — hint lookup, prepend,
overflow behaviour — with no policy or simulator running.

The load-bearing invariant is ``test_response_ids_are_unchanged``: OPSD's reverse-KL is only defined
because the teacher scores the SAME response tokens under a hint-prefixed context. If anything here
ever re-tokenized instead of prepending, k1 would be silently comparing different token streams.
"""

import pytest

from opsd.tau2_bench.distill.generator import Tau2OPSDGenerator


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [901, 902]  # fixed 2-token stand-in for the encoded SECRET_TEMPLATE(hint) block


def _mk(hints, static="STATIC", cap=100, require=False):
    g = object.__new__(Tau2OPSDGenerator)
    g.tokenizer = FakeTok()
    g._hints_by_task = hints
    g._static_hint = static
    g._seq_cap = cap
    g._require_hint = require
    # run-level counters __init__ would set (opsd/hint_dropped_total, opsd/rollouts_total)
    g._dropped_total = 0
    g._rollouts_total = 0
    return g


def test_hint_for_prefers_plain_task_id():
    # τ² keys hints on the PLAIN task_id (build_tau2_pi_hints.py:141-143); the τ1-style
    # "domain:task_id" is still honoured so a τ1-built artifact would not silently miss.
    g = _mk({"retail:retail_0": "H0", "retail_5": "P5"})
    assert g._hint_for("retail", "retail_0") == "H0"
    assert g._hint_for("retail", "retail_5") == "P5"
    assert g._hint_for("retail", "retail_9") == "STATIC"


def test_ref_ids_prepends_block():
    assert _mk({})._ref_ids([1, 2, 3], "hint") == [901, 902, 1, 2, 3]


def test_build_ref_prompts_hinted_and_static_fallback():
    g = _mk({"retail_0": "HINT0"})
    ref, hints, m = g._build_ref_prompts(
        [[1, 2, 3], [4, 5]],
        [[6, 7], [8]],
        [{"domain": "retail", "task_id": "retail_0"}, {"domain": "retail", "task_id": "retail_1"}],
    )
    assert ref[0] == [901, 902, 1, 2, 3] and hints[0] == "HINT0"
    assert ref[1] == [901, 902, 4, 5] and hints[1] == "STATIC"  # still hints the teacher, weakly
    assert m["opsd/hinted"] == 1.0
    assert m["opsd/hint_dropped_overflow"] == 0.0
    assert m["opsd/hint_static_fallback"] == 0.5  # the signal that hints did not load
    assert m["opsd/hint_tokens_mean"] == 2.0


def test_overflow_drops_hint_by_default():
    g = _mk({"retail_0": "HINT0"}, cap=4)  # 2 (block) + 3 (prompt) + 2 (response) = 7 > 4
    ref, hints, m = g._build_ref_prompts([[1, 2, 3]], [[6, 7]], [{"domain": "retail", "task_id": "retail_0"}])
    assert ref[0] == [1, 2, 3] and hints[0] == ""  # ref == prompt => k1 collapses to 0 for this sample
    assert m["opsd/hinted"] == 0.0 and m["opsd/hint_dropped_overflow"] == 1.0


def test_overflow_raises_under_require_hint():
    # The whole point of OPSD_REQUIRE_HINT: a dropped hint is otherwise invisible, and an arm that
    # drops many looks like a weak result rather than a broken one.
    g = _mk({"retail_0": "HINT0"}, cap=4, require=True)
    with pytest.raises(RuntimeError, match="OPSD_REF_SEQ_CAP"):
        g._build_ref_prompts([[1, 2, 3]], [[6, 7]], [{"domain": "retail", "task_id": "retail_0"}])


def test_response_ids_are_unchanged():
    g = _mk({"retail_0": "HINT0"})
    responses = [[6, 7], [8, 9, 10]]
    before = [list(r) for r in responses]
    g._build_ref_prompts(
        [[1], [2]],
        responses,
        [{"domain": "retail", "task_id": "retail_0"}, {"domain": "retail", "task_id": "retail_1"}],
    )
    assert responses == before


def test_missing_env_extras_falls_back_rather_than_indexing_out_of_range():
    g = _mk({"retail_0": "HINT0"})
    ref, hints, m = g._build_ref_prompts([[1], [2]], [[3], [4]], [{"task_id": "retail_0"}])
    assert hints[0] == "HINT0" and hints[1] == "STATIC"
    assert len(ref) == 2

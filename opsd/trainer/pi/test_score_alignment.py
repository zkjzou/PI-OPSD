"""Unit tests for the offline PI scorer's pure bookkeeping (no vLLM server needed).

Covers the alignment-critical pieces: hint splicing, assistant-position mapping under the hint
shift, and the dependency-free AUC / Spearman. Run: pytest this file, or `python -m ...test_score_alignment`.
"""

from __future__ import annotations

import math

from opsd.trainer.pi.score_trajectories import (
    assistant_positions,
    build_hint_prompt,
    first_token_index,
    last_token_index,
    roc_auc,
    spearman,
    splice_hint,
)

IM_END = 999


class _StubTok:
    """encode() returns a fixed 3-token block regardless of text (for injection-shape tests)."""

    def encode(self, s, add_special_tokens=False):
        return [900, 901, 902]


def test_inject_styles():
    tok = _StubTok()
    prompt = [10, 11, IM_END, 12]  # user content ... <im_end> ... assistant-hdr
    # system-prepend: the block goes at the very front
    hp_sys = build_hint_prompt(tok, prompt, "hint", "system-prepend", IM_END)
    assert hp_sys == [900, 901, 902, 10, 11, IM_END, 12]
    # splice: the block goes before the last <|im_end|> (inside the user turn)
    hp_spl = build_hint_prompt(tok, prompt, "hint", "splice", IM_END)
    assert hp_spl == [10, 11, 900, 901, 902, IM_END, 12]
    # position-shift invariant (system-prepend): student positions shift by exactly the block length
    lm = [0, 1, 1]
    base_pos = assistant_positions(len(prompt), lm)
    hint_pos = assistant_positions(len(hp_sys), lm)
    assert all(h - b == len(hp_sys) - len(prompt) for b, h in zip(base_pos, hint_pos))


def test_splice_inserts_before_last_im_end():
    # single im_end: prompt = [user content a,b] <im_end> [\n, <im_start>assistant header]
    prompt = [10, 11, IM_END, 12, 13]
    hint = [77, 78]
    assert splice_hint(prompt, hint, IM_END) == [10, 11, 77, 78, IM_END, 12, 13]
    # with a leading system turn: splice before the LAST im_end (user close), not the system close.
    sys_prompt = [1, 2, IM_END, 3, 10, 11, IM_END, 12]  # system...<end> user...<end> assistant-hdr
    assert splice_hint(sys_prompt, hint, IM_END) == [1, 2, IM_END, 3, 10, 11, 77, 78, IM_END, 12]
    # absent im_end -> None
    assert splice_hint([1, 2, 3], hint, IM_END) is None
    assert first_token_index(prompt, IM_END) == 2
    assert last_token_index(sys_prompt, IM_END) == 6


def test_assistant_positions_shift_by_hint_len():
    prompt = [10, 11, IM_END, 12, 13]  # len 5
    response = [20, 21, 22, 23]
    loss_mask = [0, 1, 1, 0]  # tokens 21,22 are assistant
    hint = [77, 78]
    hinted = splice_hint(prompt, hint, IM_END)  # len 7

    base_pos = assistant_positions(len(prompt), loss_mask)
    hint_pos = assistant_positions(len(hinted), loss_mask)
    assert base_pos == [5 + 1, 5 + 2] == [6, 7]
    assert hint_pos == [7 + 1, 7 + 2] == [8, 9]
    # the alignment invariant: same response token, teacher position shifted by exactly len(hint)
    assert all(h - b == len(hint) for b, h in zip(base_pos, hint_pos))
    # and the spliced response tokens are byte-identical & contiguous trailing in both
    assert (list(prompt) + response)[base_pos[0] :] == [21, 22, 23]
    assert (hinted + response)[hint_pos[0] :] == [21, 22, 23]


def test_roc_auc():
    assert roc_auc([1, 2, 3], [0, 0, 1]) == 1.0
    assert roc_auc([1, 2], [1, 0]) == 0.0  # higher score on the negative -> 0
    assert roc_auc([1, 3, 2, 4], [0, 1, 0, 1]) == 1.0
    assert roc_auc([5, 5], [1, 0]) == 0.5  # tie
    assert math.isnan(roc_auc([1, 2, 3], [1, 1, 1]))  # one class only -> nan (canonical sentinel)


def test_spearman():
    # perfect separation with tied binary labels -> strong positive (< 1.0 because of label ties)
    s_pos = spearman([1, 2, 3, 4], [0, 0, 1, 1])
    assert s_pos is not None and s_pos > 0.8
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9  # no ties -> exactly 1.0
    s_neg = spearman([4, 3, 2, 1], [0, 0, 1, 1])
    assert s_neg is not None and s_neg < 0  # anti-correlated


def _run_all():
    test_splice_inserts_before_last_im_end()
    test_inject_styles()
    test_assistant_positions_shift_by_hint_len()
    test_roc_auc()
    test_spearman()
    print("all score-alignment tests passed")


if __name__ == "__main__":
    _run_all()

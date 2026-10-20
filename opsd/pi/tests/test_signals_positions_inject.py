"""Unit tests for opsd.pi.signals.positions and .inject (Phase 1 extraction).

These cover the same alignment-critical bookkeeping that opsd/trainer/pi/test_score_alignment.py exercises, but
against the new canonical modules directly (no transformers needed — a tiny fake tokenizer suffices).
"""

from opsd.pi.signals import inject as I
from opsd.pi.signals import positions as P

IM_END = 999


class FakeTok:
    """Whitespace 'tokenizer': encode -> list of char codes of the words, deterministic + reversible enough."""

    def encode(self, text, add_special_tokens=False):
        # Represent each word by a distinct negative id so spliced hints are identifiable.
        return [-(i + 1) for i, _ in enumerate(text.split())]


# --------------------------------------------------------------------------------------------------
# positions
# --------------------------------------------------------------------------------------------------
def test_first_last_token_index():
    ids = [1, 999, 2, 999, 3]
    assert P.first_token_index(ids, 999) == 1
    assert P.last_token_index(ids, 999) == 3
    assert P.first_token_index(ids, 42) == -1
    assert P.last_token_index(ids, 42) == -1


def test_splice_hint_inserts_before_last_im_end():
    prompt = [10, 11, IM_END, 12]  # ...user content <im_end> assistant-hdr
    out = P.splice_hint(prompt, [77, 78], IM_END)
    assert out == [10, 11, 77, 78, IM_END, 12]


def test_splice_hint_none_when_no_im_end():
    assert P.splice_hint([1, 2, 3], [9], IM_END) is None


def test_splice_hint_uses_last_im_end():
    prompt = [IM_END, 5, IM_END, 6]  # leading system <im_end>, then user <im_end>
    out = P.splice_hint(prompt, [7], IM_END)
    assert out == [IM_END, 5, 7, IM_END, 6]  # spliced before the LAST im_end


def test_assistant_positions_maps_loss_mask():
    # prompt_len=3, response loss_mask picks response tokens 0 and 2 -> abs positions 3 and 5
    assert P.assistant_positions(3, [1, 0, 1, 0]) == [3, 5]
    assert P.assistant_positions(0, []) == []


# --------------------------------------------------------------------------------------------------
# inject
# --------------------------------------------------------------------------------------------------
def test_build_hint_prompt_system_prepend():
    tok = FakeTok()
    prompt = [10, 11, IM_END, 12]
    out = I.build_hint_prompt(tok, prompt, "the hint", "system-prepend", IM_END)
    block = I.build_system_block_ids(tok, "the hint")
    assert out == block + prompt  # prepended, prompt unchanged
    assert out[len(block) :] == prompt


def test_build_hint_prompt_splice_inserts_hint_tokens_before_im_end():
    tok = FakeTok()
    prompt = [10, 11, IM_END, 12]
    out = I.build_hint_prompt(tok, prompt, "hint text here", "splice", IM_END)
    # the spliced hint tokens land immediately before the last IM_END
    assert out is not None
    k = out.index(IM_END)
    assert out[:2] == [10, 11]
    assert out[k:] == [IM_END, 12]
    assert len(out) > len(prompt)  # hint tokens were inserted


def test_build_hint_prompt_splice_none_without_im_end():
    tok = FakeTok()
    assert I.build_hint_prompt(tok, [1, 2, 3], "h", "splice", IM_END) is None


def test_templates_match_env_wording():
    # guard against accidental drift from AlfworldEnv._HINT_TEMPLATE
    assert "<hint>" in I.HINT_TEMPLATE and "{hint}" in I.HINT_TEMPLATE
    assert "<Secret information>" in I.SECRET_TEMPLATE and "{hint_lines}" in I.SECRET_TEMPLATE


def test_secret_system_content_renders_back_to_the_teacher_block():
    """hint_placement=system must show the model EXACTLY the OPSD teacher's prompt.

    The Qwen template wraps a system message as `<|im_start|>system\\n{content}<|im_end|>\\n`, so the body
    returned here, re-wrapped, must equal SECRET_TEMPLATE -- the block main_opsd_alfworld.py prepends.
    """
    hint = "[OBSERVATION] Task complete! Your private key is abcd-efgh.\n[AGENT] <action>submit abcd-efgh</action>"
    body = I.secret_system_content(hint)
    assert "<|im_start|>" not in body and "<|im_end|>" not in body
    assert body.startswith("<Secret information>") and body.endswith("</Secret information>")
    assert "<|im_start|>system\n" + body + "<|im_end|>\n" == I.SECRET_TEMPLATE.format(hint_lines=hint)

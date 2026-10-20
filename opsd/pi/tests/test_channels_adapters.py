"""Tests for opsd.pi.signals.channels and opsd.pi.adapters (Phase 2).

The channel test asserts the new spec-driven ``token_channels`` matches the ORIGINAL regex machine
(reproduced inline from opsd/trainer/pi/turn_channel_kl.py) on the tricky cases — normal spans and Qwen-Thinking's
frequently-omitted ``</think>`` — so the published per-channel numbers are unchanged.
"""

import re

import pytest

from opsd.pi.adapters import get_adapter, get_token_spec, resolve_im_end_id
from opsd.pi.adapters.alfworld import (
    AlfworldAdapter,
    alfworld_is_success,
    alfworld_task_key,
)
from opsd.pi.signals import channels as C

# --- ORIGINAL implementation (verbatim from opsd/trainer/pi/turn_channel_kl.py pre-refactor) ---
_THINK = re.compile(r"<think>(.*?)(?:</think>|(?=<action>)|(?=<think>)|\Z)", re.DOTALL)
_ACTION = re.compile(r"<action>(.*?)(?:</action>|(?=<think>)|(?=<action>)|\Z)", re.DOTALL)


def _orig_token_channels(ids, tokenizer):
    pieces = [tokenizer.decode([int(t)], skip_special_tokens=False) for t in ids]
    text = "".join(pieces)
    think = [(m.start(1), m.end(1)) for m in _THINK.finditer(text)]
    action = [(m.start(1), m.end(1)) for m in _ACTION.finditer(text)]

    def _inside(spans, a, b):
        return any(s <= a and b <= e for s, e in spans)

    labels, pos = [], 0
    for p in pieces:
        a, b = pos, pos + len(p)
        pos = b
        if _inside(think, a, b):
            labels.append("think")
        elif _inside(action, a, b):
            labels.append("action")
        else:
            labels.append("other")
    return labels


class MapTok:
    """decode([id]) -> a fixed string per id, so we can craft token streams that decode to tagged text."""

    def __init__(self, id2str):
        self.id2str = id2str

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.id2str[i] for i in ids)


# id -> decoded piece. Build several streams out of these.
PIECES = {
    1: "<think>",
    2: " plan",
    3: " more",
    4: "</think>",
    5: "<action>",
    6: " go north",
    7: "</action>",
    8: " obs",
    9: "<think>",
}
TOK = MapTok(PIECES)


@pytest.mark.parametrize(
    "stream",
    [
        [1, 2, 3, 4, 5, 6, 7, 8],  # normal: think span, action span
        [1, 2, 3, 5, 6, 7],  # OMITTED </think>: think then action (the important case)
        [5, 6, 7, 1, 2, 4],  # action first, then think
        [2, 3, 8],  # no tags at all -> all "other"
        [1, 2, 9, 3, 4],  # a second <think> opens before </think>
    ],
)
def test_token_channels_matches_original(stream):
    assert C.token_channels(stream, TOK, C.QWEN_CHANNELS) == _orig_token_channels(stream, TOK)


def test_channel_spec_names_and_glm():
    assert C.QWEN_CHANNELS.names == ("think", "action")
    assert C.GLM_CHANNELS.names == ("action",)
    # GLM spec never labels a token "think"
    labels = C.token_channels([1, 2, 3, 4, 5, 6, 7], TOK, C.GLM_CHANNELS)
    assert "think" not in labels and "action" in labels


def test_token_channels_tag_tokens_are_other():
    # the <think>/</think>/<action>/</action> tag tokens themselves fall outside the (.*?) span
    labels = C.token_channels([1, 2, 4], TOK, C.QWEN_CHANNELS)
    assert labels == ["other", "think", "other"]


# --------------------------------------------------------------------------------------------------
# adapters / tokenization
# --------------------------------------------------------------------------------------------------
def test_get_token_spec():
    assert get_token_spec("qwen").im_end_token == "<|im_end|>"
    assert get_token_spec("qwen").channels is C.QWEN_CHANNELS
    assert get_token_spec("glm").channels is C.GLM_CHANNELS
    with pytest.raises(KeyError):
        get_token_spec("not-a-family")


class SpecialsTok:
    def __init__(self, mapping, unk=-1, eos=2):
        self.mapping = mapping
        self.unk_token_id = unk
        self.eos_token_id = eos

    def convert_tokens_to_ids(self, t):
        return self.mapping.get(t, self.unk_token_id)


def test_resolve_im_end_id_falls_back_to_eos():
    tok = SpecialsTok({"<|im_end|>": 42})
    assert resolve_im_end_id(tok, "<|im_end|>") == 42
    # unknown token -> unk -> falls back to eos
    assert resolve_im_end_id(tok, "<|nope|>") == 2


def test_alfworld_task_key_and_success():
    assert alfworld_task_key({"task_id": "t1"}) == "t1"
    assert alfworld_task_key({"extra_info": {"task_id": "t2"}}) == "t2"
    assert alfworld_task_key({"task_id": "top", "extra_info": {"task_id": "nested"}}) == "top"
    assert alfworld_task_key({}) is None
    # success rule: env_metrics.success OR reward >= 10
    assert alfworld_is_success({"env_metrics": {"success": 1.0}}) == 1
    assert alfworld_is_success({"reward": 10.0}) == 1
    assert alfworld_is_success({"reward": 5.0}) == 0
    assert alfworld_is_success({}) == 0


def test_get_adapter_registry():
    a = get_adapter("alfworld")
    assert isinstance(a, AlfworldAdapter)
    assert a.is_success({"reward": 10.0}) == 1
    assert a.task_key({"extra_info": {"task_id": "z"}}) == "z"
    assert a.token_spec().family == "qwen"
    with pytest.raises(KeyError):
        get_adapter("nope")

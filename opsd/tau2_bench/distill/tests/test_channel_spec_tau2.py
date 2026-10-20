"""τ² channel labelling for the per-turn/channel teacher-KL monitor.

The monitor logs `train/teacher_kl_{agent,think,action}` and the turn-position heatmaps. Its channel
labeller was written for ALFWorld, whose agent emits `<action>...</action>`. τ² acts by TOOL CALL, so
the ALFWorld spec matches nothing and the `action` series is a FLAT ZERO -- which does not raise, does
not warn, and reads on a dashboard as "the teacher exerts no pressure on action tokens".

These tests pin both halves: the τ² spec labels a real Thinking turn, and the old spec demonstrably
does not (so the regression that motivated the fix stays visible).
"""

from __future__ import annotations

import pytest

from opsd.pi.signals.channels import (
    QWEN_CHANNELS,
    TAU2_CHANNELS,
    spec_from_env,
    token_channels,
)

MODEL = "Qwen/Qwen3-4B-Thinking-2507"

# Exactly what qwen3thinking_preserve.jinja2 renders for an assistant turn that reasons, speaks, and
# then calls a tool -- <think> re-rendered for EVERY assistant turn, tool call in <tool_call> tags.
TURN = (
    "<|im_start|>assistant\n"
    "<think>\n"
    "The user wants to exchange two mice. I must authenticate them first, so I need their email.\n"
    "</think>\n\n"
    "Happy to help with that exchange! Could you confirm the email on the account?\n"
    "<tool_call>\n"
    '{"name": "find_user_id_by_email", "arguments": {"email": "wayne.rogers@example.com"}}\n'
    "</tool_call><|im_end|>\n"
)


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(MODEL)


def _counts(ids, tokenizer, spec):
    labels = token_channels(ids, tokenizer, spec)
    return {c: labels.count(c) for c in ("think", "action", "other")}


def test_tau2_spec_labels_both_channels(tok):
    ids = tok(TURN, add_special_tokens=False)["input_ids"]
    n = _counts(ids, tok, TAU2_CHANNELS)
    assert n["think"] > 0, "τ² spec found no think tokens in a turn that reasons"
    assert n["action"] > 0, "τ² spec found no action tokens in a turn that calls a tool"
    # the prose sentence must land in neither channel -- it is the agent speaking to the user
    assert n["other"] > 0


def test_alfworld_spec_silently_loses_every_action_token(tok):
    """The regression this fix exists for: same turn, ALFWorld spec, zero actions and no error."""
    ids = tok(TURN, add_special_tokens=False)["input_ids"]
    n = _counts(ids, tok, QWEN_CHANNELS)
    assert n["think"] > 0, "the <think> tags are shared, so thinking should still be found"
    assert n["action"] == 0, (
        "expected the ALFWorld <action> spec to match nothing on a τ² tool call -- if this now "
        "finds actions, the flat-zero failure mode changed and the τ² spec may be unnecessary"
    )


def test_unclosed_think_does_not_swallow_the_tool_call(tok):
    """Qwen3-Thinking frequently omits </think>. The span must still end at the next opening tag,
    or the tool call is mislabelled as reasoning and the action channel silently empties again."""
    turn = TURN.replace("</think>\n\n", "\n")
    ids = tok(turn, add_special_tokens=False)["input_ids"]
    n = _counts(ids, tok, TAU2_CHANNELS)
    assert n["action"] > 0, "an unclosed <think> swallowed the <tool_call> span"


def test_spec_from_env_selects_tau2_and_rejects_typos(monkeypatch):
    monkeypatch.setenv("OPSD_KL_CHANNEL_SPEC", "tau2")
    assert spec_from_env() is TAU2_CHANNELS
    # default must stay ALFWorld so existing runs are unchanged by this commit
    monkeypatch.delenv("OPSD_KL_CHANNEL_SPEC", raising=False)
    assert spec_from_env() is QWEN_CHANNELS
    # a typo must RAISE, not fall back to the spec that produces flat zeros
    monkeypatch.setenv("OPSD_KL_CHANNEL_SPEC", "tau-2")
    with pytest.raises(ValueError):
        spec_from_env()


def test_env_selection_is_what_the_monitor_actually_calls(tok, monkeypatch):
    """Guards the wiring, not just the spec: turn_channel_kl._token_channels must honour the env."""
    from opsd.trainer.pi import turn_channel_kl

    ids = tok(TURN, add_special_tokens=False)["input_ids"]
    monkeypatch.setenv("OPSD_KL_CHANNEL_SPEC", "tau2")
    assert turn_channel_kl._token_channels(ids, tok).count("action") > 0
    monkeypatch.setenv("OPSD_KL_CHANNEL_SPEC", "qwen")
    assert turn_channel_kl._token_channels(ids, tok).count("action") == 0

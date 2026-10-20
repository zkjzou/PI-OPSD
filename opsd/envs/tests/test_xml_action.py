"""Tests for the `<action>...</action>` parser (opsd.envs.common.xml_action).

Focus: an <action> the model REHEARSES inside its <think>...</think> reasoning must NOT be parsed
as the real action (regression for the think-block leak). The real command is emitted after </think>.
"""

from opsd.envs.common.xml_action import parse_think_action


def test_rehearsed_action_in_think_is_ignored():
    # The model writes an <action> inside its reasoning, then the REAL action after </think>.
    text = "<think>I could <action>go to cabinet 1</action> first, but let me check</think>\n<action>examine desk 1</action>"
    action, valid = parse_think_action(text)
    assert action == "examine desk 1"
    assert valid is True


def test_multiple_rehearsed_actions_in_think():
    text = (
        "<think>maybe <action>open fridge 1</action> or <action>go to fridge 1</action></think>"
        "<action>go to fridge 1</action>"
    )
    action, valid = parse_think_action(text)
    assert action == "go to fridge 1"
    assert valid is True


def test_normal_think_then_action():
    action, valid = parse_think_action("<think>reasoning</think>\n<action>go to cabinet 1</action>")
    assert action == "go to cabinet 1"
    assert valid is True


def test_action_without_think():
    action, valid = parse_think_action("<action>take mug 1 from desk 1</action>")
    assert action == "take mug 1 from desk 1"
    assert valid is True


def test_bare_line_fallback_enable_thinking_false():
    # enable_thinking=false: no tags, use the last non-empty line.
    action, valid = parse_think_action("examine cabinet 1")
    assert action == "examine cabinet 1"
    assert valid is True


def test_cjk_is_invalid_but_action_still_the_real_one():
    text = "<think>只是想想 <action>go to cabinet 1</action></think><action>look</action>"
    action, valid = parse_think_action(text)
    assert action == "look"  # real action, not the rehearsed one
    assert valid is False  # CJK anywhere -> invalid (unchanged behavior)


def test_unclosed_think_best_effort():
    # No </think> and only the reasoning-embedded action -> best-effort takes it.
    action, valid = parse_think_action("<think>thinking without close <action>go to cabinet 1</action>")
    assert action == "go to cabinet 1"
    assert valid is True


def test_empty_is_none_invalid():
    action, valid = parse_think_action("")
    assert action is None
    assert valid is False


# ---- thinking models (require_think_close=True): action only counts AFTER a closed </think> ----


def test_thinking_no_close_is_invalid():
    # Model ran out mid-reasoning: no </think>, and it quoted the instruction's empty <action> </action>.
    # Must be INVALID — not the quoted empty tag, not a reasoning line.
    text = (
        "Okay, let me figure out how to clean the mug. First find a mug... let me think.\n"
        'The problem says "give exactly one command inside <action> </action><|im_end|>'
    )
    action, valid = parse_think_action(text, require_think_close=True)
    assert action is None
    assert valid is False


def test_thinking_action_in_unclosed_think_is_valid():
    # This model acts WITHOUT closing </think> (emits <action> inside the still-open think). That real,
    # non-empty command must still count — requiring a closed </think> here wrongly rejected ~every action.
    text = "Let me find the tomato. I should take it now. <action>take tomato 2 from countertop 1</action>"
    action, valid = parse_think_action(text, require_think_close=True)
    assert action == "take tomato 2 from countertop 1"
    assert valid is True


def test_thinking_multiple_actions_in_open_think_takes_last():
    text = "maybe <action>go to fridge 1</action> or actually <action>open fridge 1</action>"
    action, valid = parse_think_action(text, require_think_close=True)
    assert action == "open fridge 1"
    assert valid is True


def test_thinking_proper_action_after_close():
    text = "reasoning about the task</think>\n<action>go to cabinet 1</action>"
    action, valid = parse_think_action(text, require_think_close=True)
    assert action == "go to cabinet 1"
    assert valid is True


def test_thinking_rehearsed_before_close_ignored():
    text = "I might <action>go to fridge 1</action></think>\n<action>examine desk 1</action>"
    action, valid = parse_think_action(text, require_think_close=True)
    assert action == "examine desk 1"
    assert valid is True


def test_thinking_bare_command_after_close():
    # No <action> tag but a bare command after </think> -> use the answer region's last line.
    action, valid = parse_think_action("thinking...</think>\nexamine cabinet 1", require_think_close=True)
    assert action == "examine cabinet 1"
    assert valid is True


def test_empty_action_block_skipped_relaxed():
    # A quoted empty <action> </action> is skipped; fall back to the real bare command (relaxed mode).
    action, valid = parse_think_action("see <action> </action> then\nexamine cabinet 1")
    assert action == "examine cabinet 1"
    assert valid is True


# ---- STRICT mode (strict_outside_think=True): action ONLY as an <action> tag AFTER a closed </think> ----


def test_strict_action_after_close_is_valid():
    text = "reasoning about the task</think>\n<action>go to cabinet 1</action>"
    action, valid = parse_think_action(text, require_think_close=True, strict_outside_think=True)
    assert action == "go to cabinet 1"
    assert valid is True


def test_strict_action_in_unclosed_think_is_invalid():
    # The relaxed thinking mode accepts this (see test_thinking_action_in_unclosed_think_is_valid); STRICT
    # must REJECT it — no </think> closed -> unfinished -> empty/invalid action.
    text = "Let me find the tomato. <action>take tomato 2 from countertop 1</action>"
    action, valid = parse_think_action(text, require_think_close=True, strict_outside_think=True)
    assert action is None
    assert valid is False


def test_strict_no_close_at_all_is_invalid():
    action, valid = parse_think_action(
        "thinking and thinking, never closing", require_think_close=True, strict_outside_think=True
    )
    assert action is None
    assert valid is False


def test_strict_closed_but_no_action_after_is_invalid():
    # Closed </think> but only a bare line after it (no <action> tag) -> STRICT has no bare-line fallback.
    action, valid = parse_think_action(
        "thinking...</think>\nexamine cabinet 1", require_think_close=True, strict_outside_think=True
    )
    assert action is None
    assert valid is False


def test_strict_empty_action_tag_after_close_is_invalid():
    action, valid = parse_think_action(
        "done thinking</think>\n<action></action>", require_think_close=True, strict_outside_think=True
    )
    assert action is None
    assert valid is False


def test_strict_rehearsed_before_close_ignored():
    # <action> rehearsed inside the think (before </think>) does not count; real one after does.
    text = "I might <action>go to fridge 1</action></think>\n<action>examine desk 1</action>"
    action, valid = parse_think_action(text, require_think_close=True, strict_outside_think=True)
    assert action == "examine desk 1"
    assert valid is True


def test_strict_takes_precedence_over_require_think_close():
    # strict wins even if require_think_close is False.
    text = "reasoning<action>rehearsed</action>"
    action, valid = parse_think_action(text, require_think_close=False, strict_outside_think=True)
    assert action is None
    assert valid is False

"""Prompt/reward unit tests for the ALFWorld env — no server, no GPU, no network.

These cover the knobs added while trying to get a NON-reasoning checkpoint
(Qwen3-4B-Instruct-2507) to emit reasoning, each of which was an empirical response to a
measured failure:

  think_tags_in_prompt   name the reasoning tags in the init turn (a reasoning-native model
                         gets a <think> block free from its chat template; an Instruct one
                         does not, and the stock ReAct prompt names no tags).
  oneshot_example        a worked two-turn demo, since an instruction alone measured 0%.
  reasoning_tag/_style   WHICH delimiter to ask for. <think> measured 0% across three prompt
                         variants while 72-95% of turns emitted <tool_call> instead; asking
                         for <tool_call> got a well-formed block only 69% of the time (30%
                         closed with <tool_call>, using it as a separator). "prefix" drops
                         delimiters entirely for a plain "THOUGHTS:" label.
  format_reward_coef     pay per turn for a well-formed reasoning block, because nothing else
                         does: reward = 10*won - 0.1*invalid, and GRPO duly deleted reasoning
                         within 3 steps (1171 -> 225 assistant tokens).

The invariant that matters most here is that EVERY one of these defaults to off/"think"/0.0,
so arms that never set them keep the original prompt byte-for-byte.
"""

import re

import pytest

from opsd.envs.alfworld import env as E
from opsd.envs.alfworld.config import AlfworldEnvConfig
from opsd.envs.common.xml_action import parse_think_action

OBS = "You are in a room. You see a drawer 1 and a shelf 1."
TASK = "put a keychain in a drawer"
PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}")


def make_env(*, style="tag", tag="think", prefix="THOUGHTS:", oneshot=False, tags=False, coef=0.0, min_chars=30):
    """Build an AlfworldEnv without __init__ (which would acquire a live server from the pool).

    Mirrors exactly what __init__ derives, so the prompt/reward paths can be exercised offline.
    """
    o = object.__new__(E.AlfworldEnv)
    o.prompt_style, o.show_admissible = "react", False
    o.think_tags_in_prompt, o.oneshot_example = tags, oneshot
    o.history_length, o._task, o.hint = 0, TASK, ""
    o._admissible_commands = ["go to drawer 1"]
    o.reasoning_tag, o.reasoning_style, o.reasoning_prefix = tag, style, prefix
    if style == "prefix":
        o._r_open, o._r_close = f"{prefix} ", "\n"
        o._r_instr = f"on a line beginning with {prefix}"
        o._r_rule = f"one {prefix} line, then one <action>...</action> block."
        o._reasoning_re = re.compile(rf"{re.escape(prefix)}\s*(.*?)(?=<action>|$)", re.DOTALL | re.IGNORECASE)
    else:
        o._r_open, o._r_close = f"<{tag}>", f"</{tag}>"
        o._r_instr = f"inside <{tag}> </{tag}> tags"
        o._r_rule = (
            f"one <{tag}>...</{tag}> block, then one <action>...</action> block. "
            f"Always close <{tag}> with </{tag}>."
        )
        o._reasoning_re = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
    o.format_reward_coef, o.format_reward_min_think_chars = coef, min_chars
    return o


def render(env, init):
    return env._build_message(OBS, init)["content"]


def format_bonus(env, generation):
    """The reward-side check, mirroring step()."""
    m = env._reasoning_re.search(generation or "")
    return env.format_reward_coef if (m and len(m.group(1).strip()) >= env.format_reward_min_think_chars) else 0.0


def normalise_for_parse(env, text):
    """The parse-side normalisation, mirroring step()."""
    if text and env.reasoning_style == "prefix":
        m = re.match(rf"\s*{re.escape(env.reasoning_prefix)}\s*(.*?)(?=<action>|$)", text, re.DOTALL | re.IGNORECASE)
        if m:
            return f"<think>{m.group(1)}</think>" + text[m.end() :]
    elif text and env.reasoning_tag != "think":
        return text.replace(env._r_open, "<think>").replace(env._r_close, "</think>")
    return text


# --------------------------------------------------------------------------------------------
# defaults: the finished arms must stay byte-identical
# --------------------------------------------------------------------------------------------


def test_config_defaults_are_all_off():
    c = AlfworldEnvConfig()
    assert c.think_tags_in_prompt is False
    assert c.oneshot_example is False
    assert c.format_reward_coef == 0.0
    assert c.reasoning_tag == "think"
    assert c.reasoning_style == "tag"


def test_default_prompt_is_byte_identical():
    env = make_env()
    expected = E.ALFWORLD_REACT_INIT.format(
        task_description=TASK,
        current_observation=OBS,
        admissible_block="",
        action_guide=E.ALFWORLD_ACTION_GUIDE,
        r_open="<think>",
        r_close="</think>",
        r_instr="",
        r_rule="",
    )
    assert render(env, init=True) == expected
    assert "<think>" not in render(env, init=True)


@pytest.mark.parametrize("style,tag", [("tag", "think"), ("tag", "tool_call"), ("prefix", "think")])
@pytest.mark.parametrize("oneshot", [False, True])
@pytest.mark.parametrize("tags", [False, True])
def test_step_turn_carries_no_instructions(style, tag, oneshot, tags):
    """A per-turn reminder was tried and removed; STEP is a bare observation in every config."""
    env = make_env(style=style, tag=tag, oneshot=oneshot, tags=tags)
    assert render(env, init=False) == f"Observation: {OBS}"


@pytest.mark.parametrize("style,tag", [("tag", "think"), ("tag", "tool_call"), ("prefix", "think")])
@pytest.mark.parametrize("oneshot", [False, True])
@pytest.mark.parametrize("tags", [False, True])
@pytest.mark.parametrize("init", [True, False])
def test_no_unrendered_placeholders(style, tag, oneshot, tags, init):
    """str.format does not recursively format substituted VALUES: the action guide and the demo
    block carry their own {r_open}/{r_close}/{r_rule} and must be pre-rendered. They were not,
    and a live run's prompt read 'reason inside {r_open} {r_close} tags'."""
    text = render(make_env(style=style, tag=tag, oneshot=oneshot, tags=tags), init=init)
    assert not PLACEHOLDER_RE.findall(text), f"unrendered: {PLACEHOLDER_RE.findall(text)}"


# --------------------------------------------------------------------------------------------
# the reasoning delimiter
# --------------------------------------------------------------------------------------------


def test_tag_style_names_the_configured_tag_everywhere():
    text = render(make_env(tag="tool_call", oneshot=True, tags=True), init=True)
    assert "<tool_call>" in text and "</tool_call>" in text
    assert "<think>" not in text
    # instruction line, action guide, and both demo turns
    assert text.count("<tool_call>") >= 4
    # opens and closes are balanced, so the prompt does not over-represent the opener
    assert text.count("<tool_call>") == text.count("</tool_call>")


def test_prefix_style_contains_no_xml_reasoning_tags():
    """The whole point: a plain label cannot collide with a special token or be mis-closed."""
    text = render(make_env(style="prefix", oneshot=True, tags=True), init=True)
    assert "on a line beginning with THOUGHTS:" in text
    assert "THOUGHTS: I need a mug" in text
    assert "<think>" not in text and "<tool_call>" not in text
    assert "<action>" in text  # the ACTION grammar is unchanged


def test_oneshot_demo_shows_two_complete_turns_and_leaks_nothing():
    pairs = re.findall(
        r"<think>.+?</think><action>.+?</action>",
        E.ALFWORLD_REACT_ONESHOT_BLOCK.format(r_open="<think>", r_close="</think>", r_rule=""),
    )
    assert len(pairs) == 2
    assert "keychain" not in E.ALFWORLD_REACT_ONESHOT_BLOCK  # demo task differs from the test task


# --------------------------------------------------------------------------------------------
# the format reward
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "style,tag,good,bad_empty,bad_short",
    [
        (
            "tag",
            "think",
            "<think>The keychain is probably inside the drawer so I will open it.</think><action>open drawer 1</action>",
            "<think></think><action>open drawer 1</action>",
            "<think>ok</think><action>open drawer 1</action>",
        ),
        (
            "tag",
            "tool_call",
            "<tool_call>The keychain is probably inside the drawer so I will open it.</tool_call><action>open drawer 1</action>",
            "<tool_call></tool_call><action>open drawer 1</action>",
            "<tool_call>ok</tool_call><action>open drawer 1</action>",
        ),
        (
            "prefix",
            "think",
            "THOUGHTS: The keychain is probably inside the drawer so I will open it.\n<action>open drawer 1</action>",
            "THOUGHTS:\n<action>open drawer 1</action>",
            "THOUGHTS: ok\n<action>open drawer 1</action>",
        ),
    ],
)
def test_format_reward_resists_the_cheap_hacks(style, tag, good, bad_empty, bad_short):
    env = make_env(style=style, tag=tag, coef=0.05)
    assert format_bonus(env, good) == 0.05
    assert format_bonus(env, bad_empty) == 0.0, "an empty block is well-formed and free — must not pay"
    assert format_bonus(env, bad_short) == 0.0
    assert format_bonus(env, "<action>open drawer 1</action>") == 0.0


def test_format_reward_is_small_relative_to_task_success():
    """Success is 10.0. A fully compliant ~18-turn episode must stay well under it, or the policy
    would rather format tidily forever than finish the task."""
    assert 18 * 0.05 < 1.0


def test_tag_style_does_not_pay_for_the_wrong_tag():
    env = make_env(tag="tool_call", coef=0.05)
    assert format_bonus(env, "<think>a" * 40 + "</think><action>open drawer 1</action>") == 0.0


# --------------------------------------------------------------------------------------------
# parsing: normalisation must keep the committed action correct
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "style,tag,rehearsed",
    [
        (
            "tag",
            "tool_call",
            "<tool_call>I could <action>go to shelf 1</action> but the drawer is better.</tool_call><action>open drawer 1</action>",
        ),
        (
            "prefix",
            "think",
            "THOUGHTS: I could <action>go to shelf 1</action> but the drawer is better.\n<action>open drawer 1</action>",
        ),
    ],
)
def test_rehearsed_action_inside_reasoning_is_not_committed(style, tag, rehearsed):
    """The reason parsing normalises rather than ignoring the block: an <action> the model merely
    considered must not become the committed one."""
    env = make_env(style=style, tag=tag)
    assert parse_think_action(normalise_for_parse(env, rehearsed))[0] == "open drawer 1"


def test_unclosed_block_still_yields_the_action():
    """30% of observed turns closed a <tool_call> block with <tool_call>; under relaxed parsing the
    action must still be committed (measured <action> rate was 99%)."""
    env = make_env(tag="tool_call")
    malformed = "<tool_call>The keychain is not here, I will try the drawer.<tool_call><action>open drawer 1</action>"
    assert parse_think_action(normalise_for_parse(env, malformed))[0] == "open drawer 1"


# --------------------------------------------------------------------------------------------
# invalid-action feedback
# --------------------------------------------------------------------------------------------


def _feedback(env):
    """Reproduce the invalid-action observation text built in step()."""
    note = "Your last action was not a valid command."
    if env.invalid_action_feedback_guide:
        note += " " + E.ALFWORLD_ACTION_GUIDE
    if not env.show_admissible and env.invalid_action_show_admissible:
        note += "\nAdmissible actions: [go to drawer 1]."
    return f"Nothing happens.\n[{note}]"


def test_invalid_action_feedback_omits_the_guide_by_default():
    """The guide is already in the init turn; re-injecting ~700 chars on every invalid action added
    several thousand characters per rollout early in training, when invalid actions are frequent."""
    assert AlfworldEnvConfig().invalid_action_feedback_guide is False
    env = make_env()
    env.invalid_action_feedback_guide = False
    env.invalid_action_show_admissible = False
    text = _feedback(env)
    assert text == "Nothing happens.\n[Your last action was not a valid command.]"
    assert "- look" not in text and "You may use" not in text
    assert len(text) < 100


def test_invalid_action_feedback_guide_can_be_restored():
    """True reproduces the prompt the six trajectory-PI 4bt main-table arms trained with."""
    env = make_env()
    env.invalid_action_feedback_guide = True
    env.invalid_action_show_admissible = False
    text = _feedback(env)
    assert E.ALFWORLD_ACTION_GUIDE in text
    assert len(text) > 700


def test_reasoning_prefix_survives_the_hydra_mapping_misparse():
    """`reasoning_prefix=THOUGHTS:` unquoted is valid YAML mapping syntax, so OmegaConf hands the
    env {'THOUGHTS': None}. That shipped once: str(dict) became the label in the prompt AND the
    reward regex, so it was self-consistent and scored 100% compliance while the prompt read
    "{'THOUGHTS': None}". No metric could catch it -- only reading a rollout did."""
    from omegaconf import OmegaConf

    parsed = OmegaConf.from_dotlist(["reasoning_prefix=THOUGHTS:"]).reasoning_prefix
    assert not isinstance(parsed, str), "precondition: hydra really does mis-parse this"
    keys = list(parsed.keys())
    assert len(keys) == 1 and parsed[keys[0]] is None
    assert f"{keys[0]}:" == "THOUGHTS:"  # the inversion the env applies
    # quoting is the actual fix
    assert OmegaConf.from_dotlist(["reasoning_prefix='THOUGHTS:'"]).reasoning_prefix == "THOUGHTS:"

"""Parsing / observation contracts of the AppWorld ReAct scaffold (``opsd/envs/appworld/scaffold.py``).

Pure-function tests: no AppWorld install, no ``appworld serve`` and no network. The env-level tests at
the bottom monkeypatch ``AppWorldEnv._post`` so nothing leaves the process.

    pytest opsd/envs/tests/test_appworld_scaffold.py

What is pinned here is what training rollouts are scored under, so a change to any of these is a change
to the training contract:
  * ``official`` -- only a ```python fence counts, the FIRST complete block wins, an unterminated
    trailing fence is recovered only when no complete block exists;
  * ``loop``     -- ```python and ```py, ALL blocks concatenated, trailing partial fence appended;
  * ``union``    -- official + ```py + bare ``` fences;
  * ``legacy``   -- pre-scaffold behaviour (LAST block, no partial recovery, raw observations);
  * ``strict_parse`` -- no fence at all -> ``None`` (rejected action) instead of executing raw text;
  * ``<think>`` -- reasoning (closed or truncated) is stripped before parsing, so code drafted inside
    the thinking block is never executed.
"""

import dataclasses

import pytest

from opsd.envs.appworld.env import AppWorldEnv, AppWorldEnvConfig
from opsd.envs.appworld.scaffold import (
    MULTI_BLOCK_CHOICES,
    REMINDER_MESSAGE,
    SCAFFOLDS,
    ScaffoldSpec,
    extract_code,
    format_observation,
    strip_reasoning,
)


def spec(name: str, **overrides) -> ScaffoldSpec:
    return dataclasses.replace(SCAFFOLDS[name], **overrides)


# =============================================================================
# presets
# =============================================================================


def test_presets_exist_and_are_well_formed():
    assert set(SCAFFOLDS) == {"official", "loop", "union", "legacy"}
    for s in SCAFFOLDS.values():
        assert s.multi_block in MULTI_BLOCK_CHOICES
        assert s.strip_think is True  # every preset strips <think> before parsing
        assert s.strict_parse is False  # strictness is an env-level knob, not a preset property
    assert SCAFFOLDS["official"].fence_langs == ("python",)
    assert SCAFFOLDS["official"].multi_block == "first"
    assert SCAFFOLDS["loop"].multi_block == "concat"
    assert SCAFFOLDS["legacy"].multi_block == "last"
    assert SCAFFOLDS["legacy"].partial_fence_recovery is False
    assert SCAFFOLDS["legacy"].obs_format == "raw"
    assert SCAFFOLDS["official"].obs_format == "official"


# =============================================================================
# official
# =============================================================================


@pytest.mark.parametrize(
    "text, expected",
    [
        ("```python\nx = 1\n```", "x = 1"),
        ("Let me look up the apps first.\n```python\nprint(1)\n```\nthen more prose", "print(1)"),
        # FIRST complete block only; everything after it is ignored
        ("```python\na = 1\n```\n```python\nb = 2\n```", "a = 1"),
        # missing newline after the opening fence is tolerated (`\n?`)
        ("```python print(2)```", "print(2)"),
        # unterminated trailing fence (token cap / forgotten closer) is a real action
        ("I'll paginate.\n```python\nfor p in range(3):\n    print(p)", "for p in range(3):\n    print(p)"),
        # a complete block wins over a later partial one (first-block short-circuit)
        ("```python\na = 1\n```\n```python\nb = 2", "a = 1"),
        # empty fence -> empty action (the env nudges rather than rejects)
        ("```python\n```", ""),
    ],
)
def test_official_extraction(text, expected):
    assert extract_code(text, SCAFFOLDS["official"]) == expected


@pytest.mark.parametrize("text", ["```py\nx = 1\n```", "```\nx = 1\n```"])
def test_official_ignores_non_python_fences(text):
    # ```py / bare ``` are NOT an official action: non-strict falls back to the raw text, strict rejects
    assert extract_code(text, SCAFFOLDS["official"]) == text.strip()
    assert extract_code(text, spec("official", strict_parse=True)) is None


def test_no_fence_falls_back_to_raw_text_unless_strict():
    text = "  print(apis.api_docs.show_app_descriptions())  \n"
    assert extract_code(text, SCAFFOLDS["official"]) == text.strip()
    for name in SCAFFOLDS:
        assert extract_code(text, spec(name, strict_parse=True)) is None
        assert extract_code("", spec(name, strict_parse=True)) is None


# =============================================================================
# loop / union / legacy
# =============================================================================


@pytest.mark.parametrize(
    "text, expected",
    [
        ("```py\nx = 1\n```", "x = 1"),
        # all complete blocks concatenated, in order
        ("```python\na = 1\n```\nprose\n```py\nb = 2\n```", "a = 1\nb = 2"),
        # trailing partial fence appended after the complete blocks
        ("```python\na = 1\n```\nnow:\n```python\nb = 2", "a = 1\nb = 2"),
        # bare fences are NOT loop actions -> raw fallback
        ("```\nx = 1\n```", "```\nx = 1\n```"),
    ],
)
def test_loop_extraction(text, expected):
    assert extract_code(text, SCAFFOLDS["loop"]) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("```\ny = 2\n```", "y = 2"),
        ("```py\ny = 2\n```", "y = 2"),
        ("```python\na = 1\n```\n```\nb = 2\n```", "a = 1"),  # first block
        ("prose\n```\nz = 3", "z = 3"),  # partial bare fence recovered
    ],
)
def test_union_extraction(text, expected):
    assert extract_code(text, SCAFFOLDS["union"]) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("```python\na = 1\n```\n```python\nb = 2\n```", "b = 2"),  # LAST block
        ("```\ny = 2\n```", "y = 2"),  # bare fence accepted
        # no partial-fence recovery: an unterminated fence falls back to the raw text
        ("```python\nfoo()", "```python\nfoo()"),
    ],
)
def test_legacy_extraction(text, expected):
    assert extract_code(text, SCAFFOLDS["legacy"]) == expected


# =============================================================================
# <think> stripping
# =============================================================================


def test_strip_reasoning_closed_unclosed_and_noop():
    assert strip_reasoning("<think>plan</think>\ncode") == "\ncode"
    assert strip_reasoning("<THINK>Plan</Think>code") == "code"  # case-insensitive
    assert strip_reasoning("code<think>truncated mid-thought") == "code"  # unclosed -> rest is reasoning
    assert strip_reasoning("no reasoning here") == "no reasoning here"


@pytest.mark.parametrize("name", sorted(SCAFFOLDS))
def test_code_drafted_inside_think_is_never_executed(name):
    text = "<think>maybe\n```python\ndraft()\n```\n</think>\n```python\nreal()\n```"
    assert extract_code(text, SCAFFOLDS[name]) == "real()"


def test_strip_think_can_be_disabled():
    text = "<think>\n```python\ndraft()\n```\n</think>\n```python\nreal()\n```"
    assert extract_code(text, spec("official", strip_think=False)) == "draft()"


def test_truncated_think_yields_empty_or_rejected_action():
    text = "<think>still reasoning ```python\ndraft("
    assert extract_code(text, SCAFFOLDS["official"]) == ""  # empty action -> env nudges
    assert extract_code(text, spec("official", strict_parse=True)) is None  # strict -> parse error


# =============================================================================
# observation rendering
# =============================================================================


def test_format_observation_official_and_raw():
    assert format_observation("hello", SCAFFOLDS["official"], 100) == "Output:\n```\nhello\n```\n"
    assert format_observation("hello\n", SCAFFOLDS["official"], 100) == "Output:\n```\nhello\n```\n"
    assert format_observation("hello", SCAFFOLDS["legacy"], 100) == "hello"
    assert format_observation(["a", 1], SCAFFOLDS["legacy"], 100) == "['a', 1]"  # non-str coerced


def test_format_observation_head_tail_truncation():
    out = "".join(chr(ord("a") + i % 26) for i in range(1000))
    rendered = format_observation(out, SCAFFOLDS["legacy"], 100)
    head, tail = 75, 25  # head = 3/4 of the budget
    assert rendered.startswith(out[:head])
    assert rendered.endswith(out[-tail:])
    assert "[900 chars truncated]" in rendered
    # exactly at the budget: untouched
    assert format_observation(out[:100], SCAFFOLDS["legacy"], 100) == out[:100]


# =============================================================================
# env wiring (no network: _post is monkeypatched)
# =============================================================================


def _env(**cfg_kwargs) -> AppWorldEnv:
    cfg = AppWorldEnvConfig(server_urls=["http://x"], **cfg_kwargs)
    return AppWorldEnv(cfg, extras={"reward_spec": {"ground_truth": "t1"}, "max_turns": 10})


def test_env_resolves_scaffold_and_strictness():
    env = _env(scaffold="loop", strict_parse=True)
    assert env.scaffold_spec.multi_block == "concat"
    assert env.scaffold_spec.strict_parse is True
    assert _env().scaffold_name == "official"  # default
    with pytest.raises(ValueError, match="unknown scaffold"):
        _env(scaffold="nope")


def test_env_executes_only_the_parsed_code_and_formats_the_observation():
    env = _env(scaffold="official", strict_parse=True)
    posted = []

    def fake_post(path, payload):
        posted.append((path, payload))
        if path == "/execute":
            return {"output": "42"}
        if path == "/task_completed":
            return {"output": False}
        raise AssertionError(path)

    env._post = fake_post
    out = env.step("<think>```python\ndraft()\n```</think>Plan.\n```python\nprint(6*7)\n```\n```python\nnope()\n```")
    assert posted[0] == ("/execute", {"task_id": "t1", "code": "print(6*7)"})
    assert out["done"] is False
    assert out["observations"] == [{"role": "user", "content": "Output:\n```\n42\n```\n"}]
    assert env.get_metrics()["parse_errors"] == 0.0


def test_env_reminder_every_appends_reminder():
    env = _env(scaffold="official", reminder_every=2)
    env._post = lambda path, payload: {"output": False if path == "/task_completed" else "ok"}
    first = env.step("```python\nx = 1\n```")["observations"][0]["content"]
    second = env.step("```python\nx = 2\n```")["observations"][0]["content"]
    assert REMINDER_MESSAGE not in first
    assert second.endswith(REMINDER_MESSAGE)

"""Tests for the `checklist` PI type (pi/checklist.py + prompts.CHECKLIST_*) and the env parameterisation.

Two things are under test, and the first matters more than the second:

  1. THE ENV PARAMETERISATION IS A NO-OP FOR EXISTING CALLERS. Making the analyzer's environment a
     parameter meant rewriting three live system messages as templates. Those exact strings produced
     every PI artifact currently published, so a stray character silently makes previously-built hints
     unreproducible. The sha1 pins below fail loudly if a template changes; updating a pin is a
     deliberate act that means "the analyzer prompt changed and old artifacts no longer reproduce".

  2. THE CHECKLIST CONSTRUCTOR PICKS THE RIGHT PAIR. A frontier win (supplied as text) against the
     FIRST losing student rep by repetition id. ContrastiveConstructor's best-vs-worst-reward rule
     would silently compare two failures on an all-loss task; this must return None instead.
"""

import asyncio
import hashlib

from opsd.pi.analyzer import strip_reasoning
from opsd.pi.base import TrajView, task_description_from_messages
from opsd.pi.checklist import ChecklistConstructor
from opsd.pi.prompts import (
    CONTRASTIVE_SYSTEM,
    ENV_LONG,
    ENV_SHORT,
    SKILL_SYSTEM,
    SKILL_UPDATE_SYSTEM,
    checklist_item_count,
    checklist_system,
    checklist_to_hint,
    contrastive_system,
    env_desc,
    skill_system,
    skill_update_system,
)

# sha1 of the ALFWorld-rendered system messages, pinned from the pre-parameterisation literals.
PINS = {
    "SKILL_SYSTEM": "572869a980640c17f02a3b6f70c90a81f0794d1a",
    "CONTRASTIVE_SYSTEM": "bee126f2abbc3c3d266db6a27f6dc0680ddb785a",
    "SKILL_UPDATE_SYSTEM": "5fd1a9ca92da01cb68216cda9cb317be1bf9bcc7",
}

RESP = (
    "<think>the weaker agent guessed an API name {with braces}</think>\n"
    '{"before_acting": ["Inspect the API docs for every app before the first call",'
    ' "Identify which app owns the entity named in the task"],'
    ' "while_acting": ["Print the result of each operation before continuing"],'
    ' "before_finishing": ["Re-read the task and enumerate every clause"],'
    ' "pitfalls": ["Submitting a scalar where a list was requested"]}'
)

APPWORLD_MSGS = [
    {
        "role": "user",
        "content": "I am your supervisor, and you are an AI Assistant...\n"
        "You will interact with app(s) using their APIs.\n"
        "Task: How many playlists do I have in my Spotify library?",
    },
    {"role": "assistant", "content": "<think>guessing the endpoint</think>\nprint(apis.spotify.playlists())"},
]


class FakeAnalyzer:
    """Mirrors the real Analyzer contract, which strips <think> before returning (analyzer.py:128).

    Replicating the strip matters: RESP's reasoning contains braces, and parse_analyzer_json's brace
    fallback is greedy from the FIRST `{`. A fake that skipped the strip would fail for a reason the
    production path does not have — the same trap test_episode_skill.py:30 sidesteps.
    """

    model = "fake"

    def __init__(self, payload=RESP):
        self.payload = payload
        self.calls = []

    async def analyze(self, prompt, system=None, strip_think=True, **kwargs):
        self.calls.append({"prompt": prompt, "system": system, **kwargs})
        return strip_reasoning(self.payload) if strip_think else self.payload


# --------------------------------------------------------------------------------------------------
# 1. env parameterisation is backward compatible
# --------------------------------------------------------------------------------------------------
def test_alfworld_system_messages_unchanged():
    """The bare constants must still be the ALFWorld renderings, byte for byte."""
    assert SKILL_SYSTEM == skill_system("alfworld") == skill_system(None) == skill_system()
    assert CONTRASTIVE_SYSTEM == contrastive_system("alfworld")
    assert SKILL_UPDATE_SYSTEM == skill_update_system("alfworld")
    # An unknown benchmark falls back to ALFWorld rather than emitting an empty blurb.
    assert skill_system("nonesuch") == SKILL_SYSTEM


def test_system_message_pins():
    """Fail loudly if a live analyzer template drifts. See the module docstring before updating a pin."""
    actual = {
        "SKILL_SYSTEM": hashlib.sha1(SKILL_SYSTEM.encode()).hexdigest(),
        "CONTRASTIVE_SYSTEM": hashlib.sha1(CONTRASTIVE_SYSTEM.encode()).hexdigest(),
        "SKILL_UPDATE_SYSTEM": hashlib.sha1(SKILL_UPDATE_SYSTEM.encode()).hexdigest(),
    }
    assert actual == PINS, f"analyzer prompt drift: {actual}"


def test_blurb_length_variants():
    """SKILL/SKILL_UPDATE carried a parenthetical gloss; CONTRASTIVE carried only the bare name."""
    assert ENV_LONG["alfworld"].startswith(ENV_SHORT["alfworld"])
    assert "(" in ENV_LONG["alfworld"] and "(" not in ENV_SHORT["alfworld"]
    assert ENV_SHORT["alfworld"] in CONTRASTIVE_SYSTEM
    assert ENV_LONG["alfworld"] in SKILL_SYSTEM
    for bench in ("alfworld", "appworld", "tau2"):
        assert env_desc(bench, long=True) == ENV_LONG[bench]
        assert bench.replace("2", "") in env_desc(bench, long=False).lower().replace("-", "")


def test_checklist_system_names_the_right_environment():
    assert "AppWorld" in checklist_system("appworld") and "ALFWorld" not in checklist_system("appworld")
    assert "ALFWorld" in checklist_system("alfworld") and "AppWorld" not in checklist_system("alfworld")


# --------------------------------------------------------------------------------------------------
# 2. task descriptor
# --------------------------------------------------------------------------------------------------
def test_task_description_appworld_reaches_past_the_preamble():
    assert task_description_from_messages(APPWORLD_MSGS, bench="appworld") == (
        "How many playlists do I have in my Spotify library?"
    )
    # Without the appworld marker every task collapses to the identical scaffold line.
    assert task_description_from_messages(APPWORLD_MSGS, bench="alfworld").startswith("I am your supervisor")


def test_task_description_alfworld_keeps_first_occurrence():
    """Existing artifacts were built with first-occurrence; few-shot examples also carry the marker."""
    msgs = [
        {
            "role": "user",
            "content": "Example. Your task is to: put a mug in the sink.\n" "Now solve. Your task is to: heat an egg.",
        }
    ]
    assert task_description_from_messages(msgs, bench="alfworld") == "put a mug in the sink"
    assert task_description_from_messages(msgs) == "put a mug in the sink"


# --------------------------------------------------------------------------------------------------
# 3. renderer
# --------------------------------------------------------------------------------------------------
def test_checklist_to_hint_drops_empty_sections_and_dedups_globally():
    parsed = {
        "before_acting": ["Check the docs first", "- [ ] Identify the owning app"],
        "while_acting": "Print every result",  # collapsed to a bare string
        "before_finishing": ["check the docs first."],  # dup of an earlier section, case/period
        "pitfalls": [],
    }
    hint = checklist_to_hint(parsed)
    assert hint.count("## ") == 2, hint  # pitfalls empty; before_finishing fully deduped away
    assert "Watch out for" not in hint
    assert checklist_item_count(hint) == 3
    assert hint.count("Check the docs first") == 1
    assert "[ ] Identify the owning app" in hint  # analyzer's own bullet/checkbox stripped
    assert "[ ] Print every result" in hint


def test_checklist_to_hint_respects_max_items():
    parsed = {"before_acting": [f"item {i}" for i in range(10)]}
    assert checklist_item_count(checklist_to_hint(parsed, max_items=3)) == 3


def test_checklist_to_hint_empty_on_no_content():
    assert checklist_to_hint({}) == ""
    assert checklist_to_hint({"before_acting": [""], "pitfalls": None}) == ""


# --------------------------------------------------------------------------------------------------
# 4. constructor
# --------------------------------------------------------------------------------------------------
def _ctor(analyzer, **kw):
    kw.setdefault("win_texts", {"t1": "FRONTIER WIN TRANSCRIPT " * 50})
    kw.setdefault("bench", "appworld")
    return ChecklistConstructor(analyzer, **kw)


def test_picks_first_losing_rep_by_id():
    fake = FakeAnalyzer()
    tvs = [
        TrajView("t1", APPWORLD_MSGS, 1.0, True, repetition_id=0),
        TrajView("t1", APPWORLD_MSGS, 0.0, False, repetition_id=7),
        TrajView("t1", APPWORLD_MSGS, 0.0, False, repetition_id=3),
    ]
    rec = asyncio.run(_ctor(fake).build_offline("t1", tvs))
    assert rec.meta["loss_rep"] == 3, rec.meta
    assert rec.meta["n_items"] == 5 and rec.meta["pi_type"] == "checklist"
    prompt = fake.calls[0]["prompt"]
    assert prompt.startswith("Task: How many playlists do I have in my Spotify library?")
    assert "STRONGER REFERENCE AGENT, SUCCEEDED" in prompt
    assert "FRONTIER WIN TRANSCRIPT" in prompt
    # The student's reasoning is kept: the checklist should correct how it thought, not only what it did.
    assert "<think>guessing the endpoint</think>" in prompt
    assert "AppWorld" in fake.calls[0]["system"]


def test_skips_when_no_loss_or_no_win():
    fake = FakeAnalyzer()
    all_win = [TrajView("t1", APPWORLD_MSGS, 1.0, True, repetition_id=0)]
    assert asyncio.run(_ctor(fake).build_offline("t1", all_win)) is None
    tvs = [TrajView("t9", APPWORLD_MSGS, 0.0, False, repetition_id=0)]
    assert asyncio.run(_ctor(fake).build_offline("t9", tvs)) is None
    assert fake.calls == [], "must not spend an analyzer call on an ineligible task"


def test_unparseable_or_empty_analyzer_output_yields_none():
    tvs = [TrajView("t1", APPWORLD_MSGS, 0.0, False, repetition_id=0)]
    assert asyncio.run(_ctor(FakeAnalyzer("not json at all")).build_offline("t1", tvs)) is None
    assert asyncio.run(_ctor(FakeAnalyzer('{"before_acting": []}')).build_offline("t1", tvs)) is None


def test_requires_analyzer_and_win_texts():
    for kwargs in ({"win_texts": {"a": "x"}}, {}):
        try:
            ChecklistConstructor(None, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError without an analyzer")
    try:
        ChecklistConstructor(FakeAnalyzer(), win_texts={})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty win_texts")


def test_default_output_budget_is_not_the_zero_hint_trap():
    """1024 output tokens yields ZERO hints from a thinking analyzer; the default must exceed it."""
    assert _ctor(FakeAnalyzer()).max_tokens >= 8192

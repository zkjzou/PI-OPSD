"""Tests for the episode_skill PI type (pi/episode_skill.py + prompts.EPISODE_SKILL_USER).

Analyze ONE episode -> {episode_summary, episode_skill}; the hint is the policy-facing episode_skill.
The analyzer runs with model-default sampling (temperature=None) + thinking, and its <think> reasoning
is stripped before parsing (critical: a <think> block with braces would otherwise break JSON extraction).
"""

import asyncio

from opsd.pi.analyzer import strip_reasoning
from opsd.pi.base import TrajView
from opsd.pi.episode_skill import EpisodeSkillConstructor

# Analyzer response: a <think> block (with braces, to test strip-before-parse) then the strict JSON.
RESP = (
    "<think>let me reason about {this} episode</think>\n"
    '{"episode_summary": "found the mug then placed it on the desk", '
    '"episode_skill": "First locate the target object, then move it directly to the goal receptacle."}'
)


class FakeAnalyzer:
    def __init__(self, payload):
        self.payload = payload
        self.last = None

    async def analyze(
        self,
        prompt,
        *,
        system=None,
        max_tokens=1024,
        temperature=None,
        top_p=None,
        top_k=None,
        reasoning_effort=None,
        strip_think=True,
    ):
        self.last = {"prompt": prompt, "reasoning_effort": reasoning_effort, "temperature": temperature}
        return strip_reasoning(self.payload) if strip_think else self.payload


def _traj(success=True):
    return TrajView(
        task_id="t1",
        messages=[
            {"role": "user", "content": "Task: put a mug on the desk.\nObservation: you see a mug 1."},
            {"role": "assistant", "content": "<think>x</think><action>take mug 1 from counter 1</action>"},
        ],
        reward=1.0 if success else 0.0,
        success=success,
        num_turns=1,
    )


def test_strip_reasoning_drops_think_and_dangling():
    assert strip_reasoning("<think>a{b}</think>ANSWER") == "ANSWER"
    assert strip_reasoning("reasoning...</think>\nANSWER") == "ANSWER"  # dangling close tag
    assert strip_reasoning("no think here") == "no think here"


def test_episode_skill_success_uses_skill_and_strips_think():
    fa = FakeAnalyzer(RESP)
    rec = asyncio.run(EpisodeSkillConstructor(analyzer=fa).build_offline("t1", [_traj(True)]))
    assert rec is not None
    assert "First locate the target object" in rec.task  # hint == episode_skill
    assert "found the mug" not in rec.task  # summary NOT in the hint (kept in meta)
    assert "reason about" not in rec.task  # analyzer <think> stripped
    assert rec.meta["outcome"] == "success"
    assert rec.meta["episode_summary"] == "found the mug then placed it on the desk"
    # analyzer invoked with thinking on (reasoning_effort) + model-default sampling (temperature omitted)
    assert fa.last["reasoning_effort"] == "high" and fa.last["temperature"] is None
    assert "episode_success: success" in fa.last["prompt"]


def test_episode_skill_failure_outcome_routing():
    fa = FakeAnalyzer(RESP)
    rec = asyncio.run(EpisodeSkillConstructor(analyzer=fa).build_offline("t1", [_traj(False)]))
    assert rec is not None and rec.meta["outcome"] == "failure"
    assert "episode_success: failure" in fa.last["prompt"]


def test_include_summary_prepends_summary():
    fa = FakeAnalyzer(RESP)
    rec = asyncio.run(EpisodeSkillConstructor(analyzer=fa, include_summary=True).build_offline("t1", [_traj(True)]))
    assert "Episode summary:" in rec.task and "Skill:" in rec.task


def test_requires_analyzer():
    try:
        EpisodeSkillConstructor(analyzer=None)
    except ValueError:
        return
    raise AssertionError("EpisodeSkillConstructor should require an analyzer")

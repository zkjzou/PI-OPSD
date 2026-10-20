"""Episode-skill hint (PI source: ONE episode -> summary + a single policy-facing skill).

Variant of on-policy skill using the EPISODE_SKILL prompt: analyze ONE trajectory (success or failure)
and emit {episode_summary, episode_skill}. The injected hint is the ``episode_skill`` — a short,
policy-facing rule (success -> workflow / decision-rule + ordering; failure -> avoidance rule). Analyzes
BOTH successes and failures (no require_success), one analysis per trajectory.

The analyzer runs with the model's DEFAULT sampling params (temperature=None) and thinking ENABLED; the
analyzer's ``<think>`` reasoning is stripped from the response before parsing so it never enters the hint.
"""

from __future__ import annotations

from typing import List, Optional

from opsd.pi.base import (
    HintConstructor,
    HintRecord,
    TrajView,
    task_description_from_messages,
)
from opsd.pi.prompts import (
    EPISODE_SKILL_USER,
    episode_to_hint,
    parse_analyzer_json,
    scrub_hint,
)


class EpisodeSkillConstructor(HintConstructor):
    pi_type = "episode_skill"

    def __init__(
        self,
        analyzer=None,
        *,
        max_tokens: int = 32000,  # output budget INCLUDING thinking (out-of-band reasoning_content)
        temperature: Optional[float] = None,  # None -> endpoint uses the model's default sampling
        reasoning_effort: Optional[str] = "high",  # enable analyzer thinking (Fireworks-native; None=off)
        strip_traj_think: bool = True,  # strip the STUDENT's <think> from the shown trajectory
        include_summary: bool = False,  # hint = episode_skill only (summary kept in meta)
        **kwargs,
    ):
        super().__init__(analyzer, **kwargs)
        if analyzer is None:
            raise ValueError("EpisodeSkillConstructor requires an analyzer.")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.strip_traj_think = strip_traj_think
        self.include_summary = include_summary

    @staticmethod
    def _pick(traj_views: List[TrajView]) -> TrajView:
        """One analysis per trajectory. With per_traj=1 there is exactly one view per key; if a group
        carries several, analyze the most informative (success first, then highest reward / most turns)."""
        if len(traj_views) == 1:
            return traj_views[0]
        return max(traj_views, key=lambda t: (t.success, t.reward, t.num_turns))

    async def build_offline(self, task_id: str, traj_views: List[TrajView]) -> Optional[HintRecord]:
        if not traj_views:
            return None
        tv = self._pick(traj_views)
        outcome = "success" if tv.success else "failure"
        task = task_description_from_messages(tv.messages) or task_id
        prompt = EPISODE_SKILL_USER.format(
            task=task, outcome=outcome, trajectory=self._render_for_analyzer(tv, strip_think=self.strip_traj_think)
        )
        raw = await self.analyzer.analyze(
            prompt,
            system=None,  # the prompt is self-contained (no separate system message)
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            strip_think=True,  # safety net; Fireworks returns reasoning out-of-band so content is already clean
        )
        parsed = parse_analyzer_json(raw)
        if not parsed:
            return None
        hint = scrub_hint(episode_to_hint(parsed, include_summary=self.include_summary), max_chars=self.max_chars)
        if not hint:
            return None
        return HintRecord(
            task=hint,
            meta={
                "pi_type": self.pi_type,
                "outcome": outcome,
                "episode_summary": (parsed.get("episode_summary") or "").strip(),
                "rep": tv.repetition_id,
                "n_traj": len(traj_views),
            },
        )

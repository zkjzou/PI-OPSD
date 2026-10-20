"""On-policy skill summary (PI source 2, SDAR/OPID/Skill-SD style).

Analyze the student's OWN completed trajectory + its outcome → a transferable skill hint
(success / mistake / golden_workflow). Offline: cheapest source — only needs the logged rollouts
already produced for a task + an analyzer.
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
    parse_analyzer_json,
    scrub_hint,
    skill_prompts,
    skill_to_hint,
)


class OnPolicySkillConstructor(HintConstructor):
    pi_type = "onpolicy_skill"

    def __init__(
        self,
        analyzer=None,
        *,
        max_tokens: int = 1024,
        temperature: Optional[float] = 0.3,
        prompt_env: Optional[str] = None,
        strip_think: bool = True,
        reasoning_effort: Optional[str] = None,
        max_input_tokens: Optional[int] = 0,
        **kwargs,
    ):
        # DEFAULTS are the legacy AppWorld-line ones (1024 / 0.3 / no thinking / legacy max_chars input render):
        # that constructor produced the published ALFWorld + AppWorld skill/summary hints. Every training
        # path passes explicit values -- OnlineHintConfig.constructor_kwargs sends max_tokens,
        # max_input_tokens, temperature and reasoning_effort -- so these only govern bare offline callers.
        #
        # max_input_tokens: 0/None = LEGACY input render (``tv.render(max_chars)``, i.e. max_chars caps
        # BOTH the analyzer input and the hint, as on the legacy AppWorld line). A positive value enables the
        # token-exact analyzer INPUT cap from HintConstructor._render_for_analyzer (32000 on the
        # ALFWorld line, which the online path passes explicitly).
        super().__init__(analyzer, max_input_tokens=max_input_tokens, **kwargs)
        if analyzer is None:
            raise ValueError("OnPolicySkillConstructor requires an analyzer.")
        # Output budget INCLUDING thinking (out-of-band reasoning_content).
        self.max_tokens = max_tokens
        # temperature=None means "omit the param" -> the served model applies its OWN default sampling.
        # The 0.3 kept here is only for backward compatibility with the offline CLI callers that never
        # passed it; the online path passes None explicitly (OnlineHintConfig.analyzer_temperature).
        self.temperature = temperature
        # 'low'|'medium'|'high' turns the analyzer's thinking on the Fireworks/litellm-native way
        # (chat_template_kwargs is REJECTED by Fireworks). None omits it. Reasoning comes back
        # out-of-band in message.reasoning_content, so it never reaches the hint text -- and
        # Analyzer.analyze still strips any inline <think> as a safety net for models that inline it.
        self.reasoning_effort = reasoning_effort
        # TrajView.render defaults strip_think=True and this class called it positionally, so the
        # analyzer never saw the agent's reasoning. The ALFWorld PI generation that produced the first
        # offline skill results KEPT think, so expose the knob and let the caller match that. Note the
        # prompt's trajectory label must agree with this.
        self.strip_think = strip_think
        # Per-env analyzer wording ($OPSD_PI_PROMPT_ENV). The default pair is ALFWorld-specific, so an
        # AppWorld run left on the default gets an analyst told it is reading a household simulator
        # while shown `apis.spotify.login(...)` -- wrong skills, no error. Log the choice so a
        # mismatch is visible in the run rather than silent.
        self._skill_system, self._skill_user = skill_prompts(prompt_env)
        import logging

        logging.getLogger(__name__).info(
            "OnPolicySkillConstructor prompt env=%s",
            prompt_env or __import__("os").environ.get("OPSD_PI_PROMPT_ENV", "alfworld"),
        )

    @staticmethod
    def _pick(traj_views: List[TrajView]) -> TrajView:
        """Prefer a success (extract what worked); else the failure with the most turns (most signal)."""
        wins = [t for t in traj_views if t.success]
        if wins:
            return max(wins, key=lambda t: (t.reward, t.num_turns))
        return max(traj_views, key=lambda t: t.num_turns)

    async def build_offline(self, task_id: str, traj_views: List[TrajView]) -> Optional[HintRecord]:
        if not traj_views:
            return None
        tv = self._pick(traj_views)
        outcome = "SUCCESS" if tv.success else "FAILURE"
        task = task_description_from_messages(tv.messages) or task_id
        prompt = self._skill_user.format(
            task=task, outcome=outcome, trajectory=self._render_for_analyzer(tv, strip_think=self.strip_think)
        )
        raw = await self.analyzer.analyze(
            prompt,
            system=self._skill_system,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
        )
        parsed = parse_analyzer_json(raw)
        if not parsed:
            return None
        hint = scrub_hint(skill_to_hint(parsed), max_chars=self.max_chars)
        if not hint:
            return None
        return HintRecord(
            task=hint,
            meta={"pi_type": self.pi_type, "outcome": outcome, "rep": tv.repetition_id, "n_traj": len(traj_views)},
        )

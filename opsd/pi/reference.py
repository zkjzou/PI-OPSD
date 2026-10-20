"""Reference-trajectory hint (PI source 1, frontier-trace / oracle).

Given a VERIFIED successful trajectory for a task (a frontier-model or ALFWorld-oracle rollout, or the
best successful student rollout), produce a reference hint. Three densities:
  - ``actions``    (default w/o analyzer): the literal ordered action list — dense reference PI.
  - ``abstracted`` (default w/ analyzer) : an abstracted plan (analyzer summarizes success -> workflow),
    the lower-leakage density. Reuses the skill prompt with outcome=SUCCESS.
  - ``trajectory`` (no analyzer)         : the full reference transcript (obs + assistant turns) rendered
    verbatim, with ``<think>`` reasoning kept or stripped via ``strip_think`` — the densest self-reference
    PI (mirrors the offline ``hints_raw`` / ``hints_raw_no_think`` formats for the online self path).

Records are read through their chat ``messages``; codecontest-style ``reasoning_steps`` records are not
handled in this release.
"""

from __future__ import annotations

from typing import List, Optional

from opsd.pi.base import (
    HintConstructor,
    HintRecord,
    TrajView,
    extract_actions,
    task_description_from_messages,
)
from opsd.pi.prompts import (
    SKILL_SYSTEM,
    SKILL_USER,
    parse_analyzer_json,
    scrub_hint,
    skill_to_hint,
)


class ReferenceHintConstructor(HintConstructor):
    pi_type = "reference"

    def __init__(
        self,
        analyzer=None,
        *,
        mode: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        require_success: bool = True,
        strip_think: bool = True,
        **kwargs,
    ):
        super().__init__(analyzer, **kwargs)
        # default density: abstracted if we have an analyzer to summarize, else literal actions.
        self.mode = (mode or ("abstracted" if analyzer is not None else "actions")).lower()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.require_success = require_success
        # For mode="trajectory": keep (False) or strip (True) the reference's <think> reasoning.
        self.strip_think = strip_think

    def _pick_reference(self, traj_views: List[TrajView]) -> Optional[TrajView]:
        wins = [t for t in traj_views if t.success]
        if wins:
            return max(wins, key=lambda t: (t.reward, -t.num_turns))  # success, fewest turns = cleanest
        if self.require_success:
            return None
        return max(traj_views, key=lambda t: t.reward) if traj_views else None

    async def build_offline(self, task_id: str, traj_views: List[TrajView]) -> Optional[HintRecord]:
        ref = self._pick_reference(traj_views)
        if ref is None:
            return None  # no verified reference for this task

        if self.mode == "abstracted":
            if self.analyzer is None:
                raise ValueError("reference mode='abstracted' requires an analyzer.")
            task = task_description_from_messages(ref.messages) or task_id
            prompt = SKILL_USER.format(task=task, outcome="SUCCESS", trajectory=self._render_for_analyzer(ref))
            raw = await self.analyzer.analyze(
                prompt, system=SKILL_SYSTEM, max_tokens=self.max_tokens, temperature=self.temperature
            )
            parsed = parse_analyzer_json(raw)
            if not parsed:
                return None
            hint = scrub_hint(skill_to_hint(parsed), max_chars=self.max_chars)
        elif self.mode == "trajectory":  # full reference transcript (± <think>), no analyzer
            hint = scrub_hint(ref.render(self.max_chars, strip_think=self.strip_think), max_chars=self.max_chars)
        else:  # literal action list — dense reference PI
            actions = extract_actions(ref.messages)
            if not actions:
                return None
            body = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(actions))
            hint = scrub_hint(
                f"A reference solution reaches the goal with these actions:\n{body}", max_chars=self.max_chars
            )

        if not hint:
            return None
        return HintRecord(
            task=hint,
            meta={
                "pi_type": self.pi_type,
                "mode": self.mode,
                "strip_think": self.strip_think,
                "ref_rep": ref.repetition_id,
                "ref_reward": ref.reward,
            },
        )

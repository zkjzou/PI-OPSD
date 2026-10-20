"""Prompt-level running skill (online PI source), folded ONE trajectory at a time.

Unlike the offline on-policy-skill constructor (which analyzes a single picked trajectory in one pass),
this online source maintains a *running* hint per prompt/task-type and updates it INCREMENTALLY: start
from the prior hint (empty on the first visit) and fold in the group's trajectories one by one, each via
a small ``prior_hint + one new trajectory`` analyzer call. This keeps every call cheap (no stuffing all N
trajectories into one prompt) and matches the "refine-in-place" update semantics.

Folds are sequential (each depends on the previous running hint); different prompts/task-types run
concurrently under the caller's semaphore. ``fold_per_step`` caps how many of the group's trajectories
are folded per visit (default: the highest-reward then the lowest-reward — the most informative pair).
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
    SKILL_UPDATE_SYSTEM,
    SKILL_UPDATE_USER,
    parse_analyzer_json,
    scrub_hint,
    skill_update_to_hint,
)


class PromptSkillConstructor(HintConstructor):
    pi_type = "prompt_skill"

    def __init__(
        self,
        analyzer=None,
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
        fold_per_step: int = 2,
        **kwargs,
    ):
        super().__init__(analyzer, **kwargs)
        if analyzer is None:
            raise ValueError("PromptSkillConstructor requires an analyzer.")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.fold_per_step = max(1, int(fold_per_step))

    def _select(self, traj_views: List[TrajView]) -> List[TrajView]:
        """Pick up to ``fold_per_step`` most-informative trajectories: highest reward, then lowest."""
        if len(traj_views) <= self.fold_per_step:
            return list(traj_views)
        ordered = sorted(traj_views, key=lambda t: t.reward, reverse=True)
        # interleave from the extremes (best, worst, 2nd-best, 2nd-worst, ...) so a small budget still
        # sees both a success-flavored and a failure-flavored trajectory.
        picked: List[TrajView] = []
        lo, hi = 0, len(ordered) - 1
        while len(picked) < self.fold_per_step and lo <= hi:
            picked.append(ordered[lo])
            lo += 1
            if len(picked) < self.fold_per_step and lo <= hi:
                picked.append(ordered[hi])
                hi -= 1
        return picked

    async def _fold_one(self, task: str, tv: TrajView, prior: str) -> Optional[str]:
        outcome = "SUCCESS" if tv.success else "FAILURE"
        prompt = SKILL_UPDATE_USER.format(
            task=task,
            outcome=outcome,
            prior_hint=prior or "(none yet)",
            trajectory=self._render_for_analyzer(tv),
        )
        raw = await self.analyzer.analyze(
            prompt, system=SKILL_UPDATE_SYSTEM, max_tokens=self.max_tokens, temperature=self.temperature
        )
        parsed = parse_analyzer_json(raw)
        if not parsed:
            return None
        return scrub_hint(skill_update_to_hint(parsed, prior), max_chars=self.max_chars)

    async def _build(self, task_id: str, traj_views: List[TrajView], prior: Optional[str]) -> Optional[HintRecord]:
        if not traj_views:
            return None
        task = task_description_from_messages(traj_views[0].messages) or task_id
        running = (prior or "").strip()
        folded = 0
        for tv in self._select(traj_views):
            updated = await self._fold_one(task, tv, running)
            if updated:
                running = updated
                folded += 1
        if not running:
            return None
        return HintRecord(
            task=running,
            meta={"pi_type": self.pi_type, "folded": folded, "n_traj": len(traj_views), "refined": bool(prior)},
        )

    async def build_offline(self, task_id: str, traj_views: List[TrajView]) -> Optional[HintRecord]:
        return await self._build(task_id, traj_views, prior=None)

    async def build_online(
        self, task_id: str, traj_views: List[TrajView], prior: Optional[str] = None
    ) -> Optional[HintRecord]:
        return await self._build(task_id, traj_views, prior=prior)

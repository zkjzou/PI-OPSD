"""Contrastive hint (PI source 4, MOPD style).

Pair a winning and a losing trajectory for the SAME task → analyzer produces a forward-looking
corrective hint. ALFWorld with n_samples_per_prompt>=2 supplies both classes for free; degenerate
(all-win / all-fail) groups return None and are skipped (logged by the driver).
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
    CONTRASTIVE_REFINE_USER,
    CONTRASTIVE_SYSTEM,
    CONTRASTIVE_USER,
    contrastive_to_hint,
    parse_analyzer_json,
    scrub_hint,
)


class ContrastiveConstructor(HintConstructor):
    pi_type = "contrastive"

    def __init__(self, analyzer=None, *, max_tokens: int = 1024, temperature: float = 0.3, **kwargs):
        super().__init__(analyzer, **kwargs)
        if analyzer is None:
            raise ValueError("ContrastiveConstructor requires an analyzer.")
        self.max_tokens = max_tokens
        self.temperature = temperature

    @staticmethod
    def _outcome(tv: TrajView) -> str:
        return "SUCCESS" if tv.success else "FAILURE"

    async def _build(self, task_id: str, traj_views: List[TrajView], prior: Optional[str]) -> Optional[HintRecord]:
        # Need two DISTINCT trajectories to compare. The pair may be success-vs-failure, but also
        # success-vs-success or failure-vs-failure — pick the most informative spread (best vs worst by
        # reward); ties still yield two different rollouts. Only a singleton group has no pair.
        if len(traj_views) < 2:
            return None  # no pair (online: keep the prior hint)
        ordered = sorted(traj_views, key=lambda t: t.reward, reverse=True)
        a, b = ordered[0], ordered[-1]
        task = task_description_from_messages(a.messages) or task_id
        # Two trajectories share the analyzer input budget, so cap EACH at half of max_input_tokens.
        # Passed as an explicit override rather than mutating self: _refine_one runs up to
        # max_concurrency groups concurrently against ONE shared constructor instance, so touching
        # self here would race across groups.
        _half = max(1, self.max_input_tokens // 2)
        # LEGACY mode (max_input_tokens <= 0) reproduces the legacy AppWorld-line per-side chars budget.
        _legacy_half = max(self.max_chars // 2, 1500)
        prior = (prior or "").strip()
        template = CONTRASTIVE_REFINE_USER if prior else CONTRASTIVE_USER
        fields = dict(
            task=task,
            a_outcome=self._outcome(a),
            b_outcome=self._outcome(b),
            a_reward=round(a.reward, 3),
            b_reward=round(b.reward, 3),
            a_trajectory=self._render_for_analyzer(a, max_tokens=_half, legacy_max_chars=_legacy_half),
            b_trajectory=self._render_for_analyzer(b, max_tokens=_half, legacy_max_chars=_legacy_half),
        )
        if prior:
            fields["prior_hint"] = prior
        prompt = template.format(**fields)
        raw = await self.analyzer.analyze(
            prompt, system=CONTRASTIVE_SYSTEM, max_tokens=self.max_tokens, temperature=self.temperature
        )
        parsed = parse_analyzer_json(raw)
        if not parsed:
            return None
        hint = scrub_hint(contrastive_to_hint(parsed), max_chars=self.max_chars)
        if not hint:
            return None
        return HintRecord(
            task=hint,
            meta={
                "pi_type": self.pi_type,
                "a_rep": a.repetition_id,
                "b_rep": b.repetition_id,
                "a_reward": a.reward,
                "b_reward": b.reward,
                "same_outcome": a.success == b.success,
                "refined": bool(prior),
            },
        )

    async def build_offline(self, task_id: str, traj_views: List[TrajView]) -> Optional[HintRecord]:
        return await self._build(task_id, traj_views, prior=None)

    async def build_online(
        self, task_id: str, traj_views: List[TrajView], prior: Optional[str] = None
    ) -> Optional[HintRecord]:
        return await self._build(task_id, traj_views, prior=prior)

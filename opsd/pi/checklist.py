"""Checklist hint (PI source 5): a frontier WIN vs the student's own LOSS -> a pre-action checklist.

How this differs from ``contrastive.py``, which it is otherwise modelled on:

  1. THE PAIR IS CROSS-SOURCE AND ASYMMETRIC. The win comes from a frontier reference (GLM-5.2), the
     loss from the student's own pool. ``ContrastiveConstructor`` picks best-vs-worst reward from ONE
     group, so on an all-loss task it silently compares two failures and emits a hint built from no
     success at all. Here a real student loss AND a frontier win are both required; anything else
     returns None and is counted by the driver.

  2. THE WIN SIDE ARRIVES AS TEXT, NOT AS A TrajView. It is the exact string that the paired
     demonstration arm (``glm52_raw``) injects, read from that arm's hint artifact. Taking it from the
     artifact rather than re-deriving it from a generation dump is what makes the two arms carry
     BYTE-IDENTICAL privileged information, so a difference between them is a difference of FORM and
     nothing else. Re-deriving could land on a different frontier rollout -- the exact failure mode an
     earlier pinned frontier-hint builder was written to prevent.

  3. THE OUTPUT IS A CHECKLIST OF IMPERATIVES, not a retrospective sentence. The PI is injected before
     the agent acts, so "the successful attempt checked the API docs first" is unusable where
     "[ ] Check the API docs before the first call" is actionable.

The win side needs no holdout: frontier trajectories are never in the scored student pool. The LOSS rep
is in-pool and IS held out by the driver -- it cannot leak the answer (it failed), but a checklist shaped
as "don't do what that rollout did" would otherwise hand that specific rollout an unfairly negative
prediction on the ranking axis.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from opsd.pi.base import (
    HintConstructor,
    HintRecord,
    TrajView,
    task_description_from_messages,
)
from opsd.pi.prompts import (
    CHECKLIST_USER,
    CHECKLIST_WINWIN_USER,
    checklist_item_count,
    checklist_system,
    checklist_to_hint,
    checklist_winwin_system,
    parse_analyzer_json,
    scrub_hint,
)


class ChecklistConstructor(HintConstructor):
    pi_type = "checklist"

    def __init__(
        self,
        analyzer=None,
        *,
        win_texts: Optional[Dict[str, str]] = None,
        bench: str = "alfworld",
        win_source: str = "",
        max_items: int = 5,
        allow_win_vs_win: bool = False,
        # NOT 1024. A thinking analyzer spends its budget on reasoning and the JSON gets cut mid-object,
        # which parse_analyzer_json then rejects -- measured as ZERO hints produced on long trajectories.
        max_tokens: int = 8192,
        temperature: float = 0.3,
        **kwargs,
    ):
        super().__init__(analyzer, **kwargs)
        if analyzer is None:
            raise ValueError("ChecklistConstructor requires an analyzer.")
        self.win_texts = win_texts or {}
        if not self.win_texts:
            raise ValueError(
                "ChecklistConstructor requires win_texts={task_id: frontier transcript}; without it "
                "there is no success side and every task would be skipped."
            )
        self.bench = bench
        self.win_source = win_source
        self.max_items = max_items
        self.allow_win_vs_win = allow_win_vs_win
        self.max_tokens = max_tokens
        self.temperature = temperature

    @staticmethod
    def _pick_loss(traj_views: List[TrajView]) -> Optional[TrajView]:
        """First LOSING rep by repetition id — deterministic, matching the existing builders' `wins[0]`
        rule (the AppWorld pool self-hint builder). Reward order would tie arbitrarily on binary
        reward and so vary with dict order between runs."""
        losses = sorted((t for t in traj_views if not t.success), key=lambda t: t.repetition_id)
        return losses[0] if losses else None

    @staticmethod
    def _pick_win(traj_views: List[TrajView]) -> Optional[TrajView]:
        """First WINNING rep by repetition id — same determinism rule as ``_pick_loss``."""
        wins = sorted((t for t in traj_views if t.success), key=lambda t: t.repetition_id)
        return wins[0] if wins else None

    async def build_offline(self, task_id: str, traj_views: List[TrajView]) -> Optional[HintRecord]:
        win_text = (self.win_texts.get(task_id) or "").strip()
        if not win_text:
            return None  # no frontier win for this task
        loss = self._pick_loss(traj_views)
        derivation = "win_vs_loss"
        if loss is None:
            # Every student rep succeeded. Without a fallback these tasks get NO hint and train hint-less
            # inside an arm meant to cover the pool (14 of ALFWorld train1600's 1600). Fall back to a
            # WIN-vs-WIN pair — the student's own first winning rep — which keeps the prompt two-sided and
            # shifts the question to what makes success reliable rather than incidental. Tagged in meta so
            # these tasks stay separable: they are NOT the same contrast as the rest of the arm.
            if not self.allow_win_vs_win:
                return None
            loss = self._pick_win(traj_views)
            if loss is None:
                return None  # no student rollout at all
            derivation = "win_vs_win"
        if not loss.messages:
            # A loader may parse transcripts LAZILY for only the rep it expects to be picked (the SkyRL
            # dump loader does: 32-128 reps/task at ~13k chars each). If the selection rule here and the
            # loader's expectation ever diverge, we would silently send the analyzer an empty trajectory
            # and get a checklist built from the win alone. Fail loudly instead.
            raise ValueError(
                f"{task_id}: selected loss rep {loss.repetition_id} has NO messages — the pool loader's "
                "lazy-parse assumption no longer matches _pick_loss's rule."
            )

        # Both sides share the analyzer's input budget, so cap EACH at half. Passed explicitly rather
        # than mutated onto self: the driver runs many tasks concurrently against ONE instance.
        half = max(1, self.max_input_tokens // 2)
        task = task_description_from_messages(loss.messages, bench=self.bench) or task_id
        template = CHECKLIST_WINWIN_USER if derivation == "win_vs_win" else CHECKLIST_USER
        prompt = template.format(
            task=task,
            max_items=self.max_items,
            # The win side is already a rendered transcript (with <think> inline) — cap only.
            a_trajectory=self._cap_to_tokens(win_text, half),
            # Keep the student's reasoning: the checklist should correct how it THOUGHT, not just what
            # it did, and `actions`-only demonstrations are the worst-performing arm on every benchmark.
            b_trajectory=self._render_for_analyzer(loss, strip_think=False, max_tokens=half),
        )
        raw = await self.analyzer.analyze(
            prompt,
            system=(checklist_winwin_system if derivation == "win_vs_win" else checklist_system)(self.bench),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        parsed = parse_analyzer_json(raw)
        if not parsed:
            return None
        hint = scrub_hint(checklist_to_hint(parsed, max_items=self.max_items), max_chars=self.max_chars)
        if not hint:
            return None
        return HintRecord(
            task=hint,
            meta={
                "pi_type": self.pi_type,
                "bench": self.bench,
                "derivation": derivation,
                "win_source": self.win_source or "frontier_artifact",
                "win_chars": len(win_text),
                "student_rep": loss.repetition_id,
                "student_reward": loss.reward,
                "loss_rep": loss.repetition_id if derivation == "win_vs_loss" else None,
                "n_items": checklist_item_count(hint),
                "hint_chars": len(hint),
            },
        )

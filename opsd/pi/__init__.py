"""Shared PI (prior-information / hint) library for SkyRL.

Reusable across benchmarks/models (grown out of ``opsd/trainer/pi/``):

- ``opsd.pi.signals``  — model/benchmark-agnostic scoring signals + selector-quality metrics
- ``opsd.pi.adapters`` — benchmark schema adapters + model tokenization specs
- construction (this module) — the ``HintConstructor`` registry + hint-artifact I/O; the analyzer and
  the online-hint manager live in ``opsd.pi.analyzer`` / ``opsd.pi.online``.

A backward-compatible shim remains at ``opsd.trainer.pi`` (re-exports ``build_constructor``
and the core construction API), so existing importers are unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

from opsd.pi.artifact import load_hint_artifact, load_task_hints, save_hint_artifact
from opsd.pi.base import HintConstructor, HintRecord, StaticHintConstructor, TrajView

__all__ = [
    "HintConstructor",
    "HintRecord",
    "TrajView",
    "StaticHintConstructor",
    "load_hint_artifact",
    "load_task_hints",
    "save_hint_artifact",
    "build_constructor",
]


def build_constructor(pi_type: str, analyzer: Optional[Any] = None, **cfg: Any) -> HintConstructor:
    """Registry: map a ``pi_type`` string to a concrete constructor (lazy imports)."""
    pt = (pi_type or "").lower()
    if pt == "static":
        return StaticHintConstructor(analyzer, **cfg)
    if pt == "reference":
        from opsd.pi.reference import ReferenceHintConstructor

        return ReferenceHintConstructor(analyzer, **cfg)
    if pt in ("onpolicy_skill", "skill"):
        from opsd.pi.onpolicy_skill import OnPolicySkillConstructor

        return OnPolicySkillConstructor(analyzer, **cfg)
    if pt in ("episode_skill", "episode"):
        from opsd.pi.episode_skill import EpisodeSkillConstructor

        return EpisodeSkillConstructor(analyzer, **cfg)
    if pt == "contrastive":
        from opsd.pi.contrastive import ContrastiveConstructor

        return ContrastiveConstructor(analyzer, **cfg)
    if pt in ("prompt_skill", "group_skill"):
        from opsd.pi.group_skill import PromptSkillConstructor

        return PromptSkillConstructor(analyzer, **cfg)
    if pt == "checklist":
        from opsd.pi.checklist import ChecklistConstructor

        return ChecklistConstructor(analyzer, **cfg)
    raise ValueError(
        f"unknown pi_type {pi_type!r}; expected one of: static, reference, onpolicy_skill, "
        f"episode_skill, contrastive, prompt_skill, checklist"
    )

"""ALFWorld schema adapter for PI scoring/analysis.

Free functions ``alfworld_task_key`` / ``alfworld_is_success`` capture the ALFWorld record schema
(top-level or ``extra_info.task_id``; success = env win reward >= 10) that was inlined in
``score_trajectories`` and an earlier offline PI-ranking script. ``AlfworldAdapter`` bundles them with the
Qwen ``ModelTokenSpec``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from opsd.pi.adapters.tokenization import ModelTokenSpec, get_token_spec


def alfworld_task_key(rec: Dict[str, Any]) -> Optional[str]:
    """Per-task id: top-level ``task_id`` or nested ``extra_info.task_id`` (parquet env_extras nesting)."""
    return rec.get("task_id") or (rec.get("extra_info") or {}).get("task_id")


def alfworld_is_success(rec: Dict[str, Any]) -> int:
    """1 if the env reports success or the reward reached the win threshold (10.0)."""
    em = rec.get("env_metrics") or {}
    return int(bool(em.get("success", 0.0)) or float(rec.get("reward", 0.0) or 0.0) >= 10.0)


class AlfworldAdapter:
    name = "alfworld"

    def task_key(self, rec: Dict[str, Any]) -> Optional[str]:
        return alfworld_task_key(rec)

    def is_success(self, rec: Dict[str, Any]) -> int:
        return alfworld_is_success(rec)

    def token_spec(self) -> ModelTokenSpec:
        return get_token_spec("qwen")


__all__ = ["alfworld_task_key", "alfworld_is_success", "AlfworldAdapter"]

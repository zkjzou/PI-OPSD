"""Benchmark schema-adapter protocol for PI scoring/analysis.

A ``SchemaAdapter`` isolates the per-benchmark trajectory-record schema (how to read the task id and
the success label, and which ``ModelTokenSpec`` the traces use) so the scoring/analysis code can be
written once against ``List[record]`` instead of hardcoding ALFWorld field names + Qwen tokens.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

from opsd.pi.adapters.tokenization import ModelTokenSpec


@runtime_checkable
class SchemaAdapter(Protocol):
    name: str

    def task_key(self, rec: Dict[str, Any]) -> Optional[str]:
        """Per-task id used to key PI hints (e.g. ALFWorld: top-level or extra_info.task_id)."""
        ...

    def is_success(self, rec: Dict[str, Any]) -> int:
        """1 if the trajectory succeeded, else 0 (benchmark-specific reward/metric rule)."""
        ...

    def token_spec(self) -> ModelTokenSpec:
        """The ModelTokenSpec (im_end token + reasoning channels) for this benchmark's traces."""
        ...

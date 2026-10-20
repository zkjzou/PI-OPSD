"""Load/save the PI hint artifact (v2 schema, keyed by task_id; v1 plain-string backward-compat)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from opsd.pi.base import HintRecord

SCHEMA = "opsd_hints/v2"


def load_hint_artifact(path: str) -> Dict[str, HintRecord]:
    """Load ``{task_id: HintRecord}`` from a hints JSON.

    Accepts both the v2 schema (dict values with task/steps/meta) and the original v1 schema
    (plain-string values), so existing artifacts load unchanged. The ``__schema__`` marker key,
    if present, is ignored.
    """
    raw = json.loads(Path(path).expanduser().read_text())
    out: Dict[str, HintRecord] = {}
    for key, val in raw.items():
        if key == "__schema__":
            continue
        out[key] = HintRecord.from_json(val)
    return out


def load_task_hints(path: Optional[str]) -> Dict[str, str]:
    """Convenience: ``{task_id: task-level hint string}`` (drops step-level hints)."""
    if not path:
        return {}
    return {k: r.task for k, r in load_hint_artifact(path).items() if r.task}


def save_hint_artifact(path: str, records: Dict[str, HintRecord]) -> None:
    """Write a v2 artifact. Pure task-level records still serialize as plain strings (v1-readable)."""
    payload: Dict[str, object] = {"__schema__": SCHEMA}
    for task_id, rec in records.items():
        payload[task_id] = rec.to_json()
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

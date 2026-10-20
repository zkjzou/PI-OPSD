"""Turn pinned τ² task splits into public JSONL manifests and a SkyRL dataset.

Only *public* metadata is written: the manifest is safe to commit and to read in a prompt-adjacent
context.  Everything that reveals the objective (instructions, persona, evaluation criteria, initial
state, ticket, …) stays in the pinned checkout and is loaded at rollout time by ``task_id``.

**Difference from τ1.** τ1 asserted that task ids were exactly ``range(count)``.  τ² ids are opaque
strings (telecom's encode the whole scenario and contain ``[ ] : |``), so the ordinal contract is
replaced by an *identity* contract: ids must be unique, non-empty strings that round-trip against
the pinned split loader, and a dense ``task_index`` is carried separately for ergonomics.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .constants import HIDDEN_TASK_KEYS, MANIFEST_SPECS, TAU2_REVISION, ManifestSpec
from .source import load_tasks


class ManifestValidationError(RuntimeError):
    """Raised when a generated manifest does not match the pinned source."""


def task_digest(task_id: str) -> str:
    """Stable short digest of a τ² task id, safe for use inside a SkyRL ``uid``."""

    return hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:10]


def _iter_keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _iter_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_keys(item)


def build_manifest_rows(spec: ManifestSpec, source_dir: str | Path) -> list[dict[str, Any]]:
    """Build the public manifest rows for one split from the pinned checkout.

    ``uid`` deliberately avoids the raw task id: SkyRL appends ``f"{uid}:repeat-{n}"`` to form GRPO
    groups, and telecom ids already contain ``:``.  The digest keeps ids short, printable and
    delimiter-safe while remaining traceable back to the original.
    """

    tasks = load_tasks(spec.domain, spec.split_name)
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        task_id = task.id
        if not isinstance(task_id, str) or not task_id:
            raise ManifestValidationError(f"{spec.name} task at index {index} has a non-string id: {task_id!r}")
        rows.append(
            {
                "uid": f"tau2-{spec.name}-{index:04d}-{task_digest(task_id)}",
                "prompt": [
                    {
                        "role": "user",
                        "content": f"Start the tau2 {spec.domain} customer-support interaction.",
                    }
                ],
                "env_class": "tau2_bench",
                "domain": spec.domain,
                # Drives eval bucketing: eval/airline/*, eval/retail/*, eval/telecom/*, eval/all/*.
                "data_source": spec.domain,
                "task_split": spec.split_name,
                "dataset_split": spec.name,
                "dataset_role": spec.role,
                "task_id": task_id,
                "task_index": index,
                "task_digest": task_digest(task_id),
                "source_revision": TAU2_REVISION,
            }
        )
    return rows


def validate_rows(rows: Sequence[dict[str, Any]], spec: ManifestSpec) -> None:
    """Validate a manifest against the pinned split, without assuming integer ids."""

    if len(rows) != spec.count:
        raise ManifestValidationError(f"{spec.name} has {len(rows)} rows, expected {spec.count}")

    if [row["task_index"] for row in rows] != list(range(len(rows))):
        raise ManifestValidationError(f"{spec.name} task_index values are not dense/ordered")

    task_ids = [row["task_id"] for row in rows]
    if len(set(task_ids)) != len(task_ids):
        raise ManifestValidationError(f"{spec.name} contains duplicate task ids")
    if not all(isinstance(t, str) and t for t in task_ids):
        raise ManifestValidationError(f"{spec.name} contains a non-string or empty task id")
    if any(row["task_digest"] != task_digest(row["task_id"]) for row in rows):
        raise ManifestValidationError(f"{spec.name} has a task_digest that does not match its task_id")
    if len({row["uid"] for row in rows}) != len(rows):
        raise ManifestValidationError(f"{spec.name} contains duplicate uids")
    if any(":" in row["uid"] for row in rows):
        # A ':' would collide with SkyRL's f"{uid}:repeat-{n}" GRPO grouping suffix.
        raise ManifestValidationError(f"{spec.name} has a uid containing ':'")

    for index, row in enumerate(rows):
        hidden = HIDDEN_TASK_KEYS.intersection(_iter_keys(row))
        if hidden:
            raise ManifestValidationError(f"{spec.name} row {index} exposes hidden keys: {sorted(hidden)}")
        if row["domain"] != spec.domain or row["task_split"] != spec.split_name:
            raise ManifestValidationError(f"{spec.name} row {index} has mismatched domain/split metadata")
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or len(prompt) != 1 or prompt[0].get("role") != "user":
            raise ManifestValidationError(f"{spec.name} row {index} has an invalid SkyRL prompt")


def cross_check_against_source(rows: Sequence[dict[str, Any]], spec: ManifestSpec) -> None:
    """Assert the manifest's task-id *set* still equals the pinned loader's set."""

    live = {task.id for task in load_tasks(spec.domain, spec.split_name)}
    stored = {row["task_id"] for row in rows}
    if live != stored:
        missing, extra = sorted(live - stored)[:3], sorted(stored - live)[:3]
        raise ManifestValidationError(
            f"{spec.name} task ids drifted from the pinned source (missing≈{missing}, extra≈{extra})"
        )


def write_manifests(source_dir: str | Path, output_dir: str | Path) -> dict[str, int]:
    """Write every split's manifest to ``output_dir`` and return ``{name: count}``."""

    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for spec in MANIFEST_SPECS:
        rows = build_manifest_rows(spec, source_dir)
        validate_rows(rows, spec)
        path = out / spec.filename
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        counts[spec.name] = len(rows)
    return counts


def read_manifest(path: str | Path, expected_name: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if expected_name and any(row.get("dataset_split") != expected_name for row in rows):
        raise ManifestValidationError(f"{path} contains rows from another split")
    return rows


def validate_manifest_directory(path: str | Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for spec in MANIFEST_SPECS:
        rows = read_manifest(Path(path) / spec.filename, spec.name)
        validate_rows(rows, spec)
        cross_check_against_source(rows, spec)
        counts[spec.name] = len(rows)
    return counts


class Tau2ManifestDataset:
    """Small dataset adapter for SkyRL's stateful data loader (mirrors τ1's)."""

    def __init__(self, data_files: Iterable[str], repeat_size: int | None = None):
        self.rows: list[dict[str, Any]] = []
        for data_file in data_files:
            self.rows.extend(read_manifest(data_file))
        if not self.rows:
            raise ManifestValidationError("No tau2 manifest rows were loaded")
        self.repeat_size = repeat_size

    def __len__(self) -> int:
        return self.repeat_size if self.repeat_size is not None else len(self.rows)

    def __getitem__(self, index: int) -> tuple[list[dict[str, str]], str, dict[str, Any], str]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        row = copy.deepcopy(self.rows[index % len(self.rows)])
        prompt = row.pop("prompt")
        env_class = row.pop("env_class")
        base_uid = row.pop("uid")
        repeat = index // len(self.rows)
        uid = base_uid if self.repeat_size is None else f"{base_uid}:repeat-{repeat}"
        return prompt, env_class, row, uid

    @staticmethod
    def collate_fn(items: Iterable[tuple[list[dict[str, str]], str, dict[str, Any], str]]) -> list[dict[str, Any]]:
        return [
            {"prompt": prompt, "env_class": env_class, "env_extras": extras, "uid": uid}
            for prompt, env_class, extras, uid in items
        ]

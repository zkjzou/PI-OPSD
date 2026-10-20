#!/usr/bin/env python
"""App-balanced batch sampling for AppWorld training.

WHY. `train.parquet` is 90 tasks / 30 scenarios, and **16 of the 30 scenarios are Spotify** (53%),
with venmo 7, phone 3, file-system 2, note 1, other 1 — and amazon/gmail/todoist/splitwise at ZERO.
Uniform shuffling therefore hands GRPO a batch that is ~half one app, and the measured consequence
(2026-08-06, step-106 OPSD checkpoint vs base, per-scenario) is app-shaped:

    test_normal   spotify +0.256 (14 scen)   venmo -0.059 (23)   note -0.157 (9)   file-system -0.167 (2)
    dev           spotify +0.639 (6 scen)    venmo -0.007 (4)
    challenge     spotify +0.375 (2 scen)    amazon -0.026 (57)

i.e. the gain lands on the over-represented app and the under-represented ones REGRESS, which nets to
"no transfer" on test_normal only because spotify is 25% there vs 53% in train/dev. This sampler fixes
the batch composition instead of leaving it to the pool's natural skew.

The app label is inferred by keyword match on the task instruction (the same rule used for the split
analysis above) — a proxy for the APIs a task actually calls, not ground truth. Rows whose text matches
nothing get `other`, which is a real class here (a photo-organising task).

MODES — one batch of 18 on the current pool, and the resulting per-task exposure spread:

    cover (default)  1 slot per app GUARANTEED + the other 12 drawn at random from the whole pool
                     -> expected  spotify 7.4  venmo 3.8  phone 2.2  file-system 1.8  note 1.4  other 1.4
                     exposure ratio 3.0x
    proportional     spotify 9  venmo 4  phone 2  file-system 1  note 1  other 1     ratio 2.0x (≈ status quo)
    sqrt             spotify 6  venmo 4  phone 3  file-system 2  note 2  other 1     ratio 5.3x
    uniform          spotify 3  venmo 3  phone 3  file-system 3  note 3  other 3     ratio 16.0x

`cover` is the light touch: no app can ever be absent from a batch, but the remaining 12 slots keep the
pool's natural proportions, so spotify still leads (41% of the batch vs its 53% pool share) and nothing
is force-fed. `uniform` is the strongest de-biasing and the bluntest — the `note` class holds 3 tasks
total, so a quota of 3 puts ALL of them in EVERY batch while each of the 48 spotify tasks is seen once
per 16 batches. At n_samples_per_prompt=8 that is ~848 rollouts per note task over a 106-step run, which
memorises rather than teaches; worse for GRPO, a task that is reliably solved has near-zero reward
variance across its group, so its advantage collapses and the slot stops producing gradient while still
costing rollout budget. `sqrt` sits between. The real fix is a less skewed pool — amazon/gmail/todoist/
splitwise have ZERO train scenarios and no apportionment rule can sample what is not there.

Each app draws from its own independently-shuffled cyclic queue, so small classes repeat *within* an
epoch by construction. `__len__` is kept at `len(dataset) // batch_size` so the step count per epoch —
and therefore every schedule, eval interval and checkpoint interval — is unchanged.

DETERMINISM. The per-app and whole-pool shuffle streams are seeded from Python's built-in ``hash()`` of a
tuple that contains strings (``(seed, app, cycle)`` / ``(seed, "__global__", cycle)``). String hashing is
salted per interpreter unless ``PYTHONHASHSEED`` is fixed, so with it unset WHICH tasks land in which batch
differs from run to run even at the same ``trainer.seed`` (the per-app quotas do not). The paper runs used
exactly this seeding, so it is kept as-is for parity. The launcher bootstrap
(``scripts/common/bootstrap.sh``) exports ``PYTHONHASHSEED=0`` and SkyRL forwards it into the Ray
runtime env, so every interpreter that builds this sampler starts with the same hash seed and the order is
reproducible. Set it yourself when calling the sampler outside the launcher. (The within-batch shuffle
hashes an all-int tuple and is deterministic regardless.)
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterator, List, Optional, Sequence

# Ordered: first match wins, so a task naming both spotify and venmo is filed under spotify.
APP_KEYWORDS: tuple[str, ...] = (
    "spotify",
    "venmo",
    "gmail",
    "phone",
    "amazon",
    "todoist",
    "splitwise",
    "note",
    "file system",
    "email",
)
OTHER = "other"


def infer_app_label(task_text: str) -> str:
    """Primary app of a task instruction; `other` when nothing matches."""
    low = task_text.lower()
    for app in APP_KEYWORDS:
        if app in low:
            return app
    return OTHER


def _task_text(prompt: Sequence[Dict[str, Any]]) -> str:
    """The actual task lives in the LAST user message of the few-shot ReAct prompt, after 'Task:'."""
    if not len(prompt):
        return ""
    content = str(prompt[-1].get("content", ""))
    _, sep, tail = content.partition("Task:")
    return tail if sep else content


def infer_app_labels(dataset) -> List[str]:
    """One label per dataset row, in dataset order.

    Works off `PromptDataset.dataframe` (a `datasets.Dataset`) when present so the whole prompt column
    is read in one pass; falls back to per-row indexing for anything list-like.
    """
    frame = getattr(dataset, "dataframe", None)
    if frame is not None:
        prompt_key = getattr(dataset, "prompt_key", "prompt")
        return [infer_app_label(_task_text(p)) for p in frame[prompt_key]]
    labels = []
    for i in range(len(dataset)):
        row = dataset[i]
        prompt = row[0] if isinstance(row, tuple) else row.get("prompt", [])
        labels.append(infer_app_label(_task_text(prompt)))
    return labels


def compute_quotas(counts: Dict[str, int], batch_size: int, mode: str) -> Dict[str, int]:
    """Per-app slots in one batch. Sums exactly to batch_size; every app gets >=1 unless proportional.

    Largest-remainder apportionment, ties broken by pool size then name so the result is deterministic.
    """
    apps = sorted(counts)
    if mode == "cover":
        # Guarantee ONE slot per app; the caller fills the rest from the pool at random, so the
        # remainder keeps the pool's natural proportions and no app can be absent from a batch.
        if batch_size < len(apps):
            raise ValueError(
                f"batch_size={batch_size} < {len(apps)} apps: cannot give every app a slot. "
                "Raise train_batch_size or use mode=proportional."
            )
        return {a: 1 for a in apps}
    if mode == "uniform":
        weights = {a: 1.0 for a in apps}
    elif mode == "sqrt":
        weights = {a: counts[a] ** 0.5 for a in apps}
    elif mode == "proportional":
        weights = {a: float(counts[a]) for a in apps}
    else:
        raise ValueError(f"unknown mode {mode!r}; want cover|uniform|sqrt|proportional")

    if batch_size < len(apps) and mode != "proportional":
        raise ValueError(
            f"batch_size={batch_size} < {len(apps)} apps: cannot give every app a slot. "
            "Raise train_batch_size or use mode=proportional."
        )

    total = sum(weights.values())
    exact = {a: batch_size * weights[a] / total for a in apps}
    # Floor, with a floor of 1 for the coverage-guaranteeing modes, then hand out the remainder.
    quota = {a: max(1, int(exact[a])) if mode != "proportional" else int(exact[a]) for a in apps}
    while sum(quota.values()) > batch_size:  # only possible when the >=1 floors overshoot
        victim = max(apps, key=lambda a: (quota[a] - exact[a], counts[a], a))
        if quota[victim] <= 1:
            break
        quota[victim] -= 1
    remainder = sorted(apps, key=lambda a: (-(exact[a] - int(exact[a])), -counts[a], a))
    i = 0
    while sum(quota.values()) < batch_size:
        quota[remainder[i % len(remainder)]] += 1
        i += 1
    return quota


class AppBalancedBatchSampler:
    """Yields lists of dataset indices whose per-app composition is fixed by `mode`.

    Pass as `batch_sampler=` to a DataLoader (which then ignores batch_size/shuffle/drop_last).
    Implements `state_dict`/`load_state_dict` so `StatefulDataLoader` can resume mid-epoch — these runs
    resume from checkpoints, and a non-stateful sampler would silently restart the sequence on resume.
    """

    def __init__(
        self,
        labels: Sequence[str],
        batch_size: int,
        mode: str = "cover",
        seed: int = 42,
        num_batches: Optional[int] = None,
    ):
        self.labels = list(labels)
        self.batch_size = int(batch_size)
        self.mode = mode
        self.seed = int(seed)
        self.by_app: Dict[str, List[int]] = {}
        for idx, app in enumerate(self.labels):
            self.by_app.setdefault(app, []).append(idx)
        self.counts = {a: len(v) for a, v in self.by_app.items()}
        self.quotas = compute_quotas(self.counts, self.batch_size, mode)
        # Slots not pinned to an app -> drawn at random from the whole pool (mode="cover" only).
        self.remainder = self.batch_size - sum(self.quotas.values())
        self.num_batches = int(num_batches) if num_batches is not None else max(1, len(self.labels) // self.batch_size)
        self._epoch = 0
        self._batch_in_epoch = 0
        self._queues: Dict[str, List[int]] = {}
        self._reset_queues()

    # --- exposure accounting, so the trade-off is visible in the logs rather than implicit ---
    def exposure_summary(self) -> str:
        lines = [
            f"AppBalancedBatchSampler(mode={self.mode}, batch={self.batch_size}, batches/epoch={self.num_batches})"
        ]
        pool_total = sum(self.counts.values())
        if self.remainder:
            lines.append(
                f"    {self.remainder} of {self.batch_size} slots are pinned to no app "
                f"(drawn at random from all {pool_total} tasks); effective quota is the expectation"
            )
        eff = {a: self.quotas[a] + self.remainder * self.counts[a] / pool_total for a in self.counts}
        for app in sorted(self.counts, key=lambda a: -self.counts[a]):
            n = self.counts[app]
            lines.append(
                f"    {app:12s} pool={n:3d} quota={self.quotas[app]:2d}+rand -> {eff[app]:4.1f}/batch"
                f"  ({100 * eff[app] / self.batch_size:4.1f}% of batch, pool share {100 * n / pool_total:4.1f}%)"
                f"  each task {eff[app] / n:.2f}x per batch"
            )
        best = max(eff[a] / self.counts[a] for a in self.counts)
        worst = min(eff[a] / self.counts[a] for a in self.counts)
        lines.append(f"    exposure ratio (most:least sampled task) = {best / worst:.1f}x")
        return "\n".join(lines)

    def _global_rng(self, cycle: int) -> random.Random:
        # hash() of a str-containing tuple: reproducible only under a fixed PYTHONHASHSEED (module doc).
        return random.Random((self.seed, "__global__", cycle).__hash__() & 0xFFFFFFFF)

    def _take_random(self, k: int, exclude: set) -> List[int]:
        """k indices from a shuffled cycle over the WHOLE pool, skipping ones already in the batch.

        A cyclic queue rather than IID draws: the remainder still follows the pool's proportions in
        expectation, but every task is guaranteed to come round rather than being left to luck.
        """
        out: List[int] = []
        guard = 0
        while len(out) < k:
            if not self._global_queue:
                self._global_cycle += 1
                q = list(range(len(self.labels)))
                self._global_rng(self._global_cycle).shuffle(q)
                self._global_queue = q
                guard += 1
                if guard > 2 and len(exclude) + len(out) >= len(self.labels):
                    break  # pool smaller than the batch; cannot fill without duplicates
            idx = self._global_queue.pop(0)
            if idx in exclude:
                continue
            out.append(idx)
            exclude.add(idx)
        return out

    def _rng(self, app: str, cycle: int) -> random.Random:
        # Per-(app, cycle) stream, independent of how the other apps drain. Reproducible across processes
        # only under a fixed PYTHONHASHSEED (str hashing is salted otherwise; see the module docstring).
        return random.Random((self.seed, app, cycle).__hash__() & 0xFFFFFFFF)

    def _reset_queues(self) -> None:
        self._cycles = {a: 0 for a in self.by_app}
        self._queues = {}
        for app, idxs in self.by_app.items():
            q = list(idxs)
            self._rng(app, 0).shuffle(q)
            self._queues[app] = q
        self._global_cycle = 0
        self._global_queue = list(range(len(self.labels)))
        self._global_rng(0).shuffle(self._global_queue)

    def _take(self, app: str, k: int) -> List[int]:
        out: List[int] = []
        while len(out) < k:
            if not self._queues[app]:
                self._cycles[app] += 1
                q = list(self.by_app[app])
                self._rng(app, self._cycles[app]).shuffle(q)
                self._queues[app] = q
            take = min(k - len(out), len(self._queues[app]))
            out.extend(self._queues[app][:take])
            self._queues[app] = self._queues[app][take:]
        return out

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        while self._batch_in_epoch < self.num_batches:
            batch: List[int] = []
            for app in sorted(self.quotas):
                batch.extend(self._take(app, self.quotas[app]))
            if self.remainder:
                batch.extend(self._take_random(self.remainder, set(batch)))
            # Shuffle within the batch so app order never correlates with position in the batch.
            random.Random((self.seed, self._epoch, self._batch_in_epoch).__hash__() & 0xFFFFFFFF).shuffle(batch)
            self._batch_in_epoch += 1
            yield batch
        self._epoch += 1
        self._batch_in_epoch = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self._epoch,
            "batch_in_epoch": self._batch_in_epoch,
            "queues": {a: list(v) for a, v in self._queues.items()},
            "cycles": dict(self._cycles),
            "global_queue": list(self._global_queue),
            "global_cycle": self._global_cycle,
            "seed": self.seed,
            "mode": self.mode,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._epoch = state.get("epoch", 0)
        self._batch_in_epoch = state.get("batch_in_epoch", 0)
        self._cycles = dict(state.get("cycles", {a: 0 for a in self.by_app}))
        queues = state.get("queues") or {}
        # Only trust saved queues for apps we still have; anything else is rebuilt.
        for app in self.by_app:
            self._queues[app] = [i for i in queues.get(app, []) if 0 <= i < len(self.labels)]
        self._global_cycle = state.get("global_cycle", 0)
        self._global_queue = [i for i in (state.get("global_queue") or []) if 0 <= i < len(self.labels)]

"""Offline reward-path validation — **no GPU, no LLM calls** (for the deterministic domains).

τ² ships full result sets under ``data/tau2/results/final/*_4trials.json``: every simulation's recorded
``messages`` plus the ``reward_info.reward`` the authors computed.  This replays those recorded messages
through the *exact* evaluator our episode runner uses (``evaluate_simulation``) and checks that we
recompute the published reward.

It is the cheapest possible proof that our reward wiring is faithful: if the numbers match on airline and
telecom — whose reward is fully deterministic (DB / action / env-assertion / communicate checks, **zero
LLM-judge calls**) — then the reward a training run would see is τ²'s own reward, not an artefact of our
integration.  Retail is reported separately and only on its deterministic ``DB`` component, because its
NL-assertion component is a nondeterministic ``gpt-4.1`` call and cannot be bit-reproduced offline.

    python -m opsd.tau2_bench.reward_replay --domain airline --limit 200
    python -m opsd.tau2_bench.reward_replay --domain telecom
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .constants import DEFAULT_SOURCE_DIR
from .source import bootstrap_tau2

# The shipped result file to replay per domain (base-task, gpt-4.1 policy + gpt-4.1 judge, 4 trials).
RESULT_FILES = {
    "airline": "gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json",
    "telecom": "gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json",
    "retail": "gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json",
}
DETERMINISTIC_DOMAINS = ("airline", "telecom")


def _result_path(source_dir: Path, domain: str) -> Path:
    return source_dir / "data" / "tau2" / "results" / "final" / RESULT_FILES[domain]


def replay_domain(source_dir: Path, domain: str, *, limit: int | None, tol: float) -> dict[str, Any]:
    """Recompute reward for each recorded simulation and diff against the published value."""

    from tau2.data_model.simulation import Results
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    results = Results.model_validate(json.loads(_result_path(source_dir, domain).read_text()))
    tasks_by_id = {task.id: task for task in results.tasks}
    # airline/telecom reward is fully deterministic -> compare the whole reward with ALL.
    # retail mixes in a nondeterministic NL judge -> compare only its deterministic DB component,
    # recomputed via the ENV evaluator and read out of reward_breakdown["DB"].
    deterministic = domain in DETERMINISTIC_DOMAINS
    eval_type = EvaluationType.ALL if deterministic else EvaluationType.ENV

    def _value(reward_info: Any) -> float | None:
        if deterministic:
            return float(getattr(reward_info, "reward", 0.0) or 0.0)
        breakdown = getattr(reward_info, "reward_breakdown", None) or {}
        # keys may be RewardType enums or their string values depending on load path
        for key, val in breakdown.items():
            if str(getattr(key, "value", key)).rsplit(".", 1)[-1] == "DB":
                return float(val)
        return None

    sims = results.simulations if limit is None else results.simulations[:limit]
    matches = 0
    skipped = 0
    mismatches: list[dict[str, Any]] = []
    for sim in sims:
        task = tasks_by_id.get(sim.task_id)
        if task is None:
            mismatches.append({"task_id": sim.task_id, "reason": "task not found in results.tasks"})
            continue
        published = _value(sim.reward_info)
        try:
            recomputed_info = evaluate_simulation(
                simulation=sim,
                task=task,
                evaluation_type=eval_type,
                solo_mode=False,
                domain=domain,
            )
        except Exception as exc:  # noqa: BLE001
            mismatches.append(
                {"task_id": sim.task_id, "trial": sim.trial, "reason": f"{type(exc).__name__}: {exc}"[:160]}
            )
            continue
        recomputed = _value(recomputed_info)
        if published is None or recomputed is None:
            skipped += 1
            continue
        if abs(recomputed - published) <= tol:
            matches += 1
        else:
            mismatches.append(
                {"task_id": sim.task_id, "trial": sim.trial, "published": published, "recomputed": recomputed}
            )

    compared = matches + len(mismatches)
    return {
        "domain": domain,
        "evaluation_type": eval_type.name,
        "simulations": len(sims),
        "compared": compared,
        "skipped": skipped,
        "matches": matches,
        "mismatch_count": len(mismatches),
        "match_rate": matches / compared if compared else None,
        "mismatches": mismatches[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline τ² reward-path replay")
    parser.add_argument("--domain", choices=("airline", "telecom", "retail", "all"), default="all")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tol", type=float, default=1e-9)
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser()
    bootstrap_tau2(source_dir)

    domains = ("airline", "telecom", "retail") if args.domain == "all" else (args.domain,)
    reports = [replay_domain(source_dir, d, limit=args.limit, tol=args.tol) for d in domains]
    print(json.dumps(reports, indent=2, default=str))

    # A deterministic domain must reproduce exactly; retail is informational (DB-only).
    hard_fail = [
        r
        for r in reports
        if r["domain"] in DETERMINISTIC_DOMAINS and (r["match_rate"] is None or r["match_rate"] < 1.0)
    ]
    if hard_fail:
        raise SystemExit(f"REWARD REPLAY FAILED for {[r['domain'] for r in hard_fail]} — reward path is not faithful")
    print("\nOK: deterministic domains reproduce τ²'s published rewards exactly.")


if __name__ == "__main__":
    main()

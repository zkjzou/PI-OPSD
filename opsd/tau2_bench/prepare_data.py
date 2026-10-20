"""CLI: materialise / validate / audit the τ² manifests.

    python -m opsd.tau2_bench.prepare_data prepare
    python -m opsd.tau2_bench.prepare_data validate
    python -m opsd.tau2_bench.prepare_data audit

``audit`` is the safety gate for the reward path: it reports, per split, how many tasks will actually
invoke τ²'s NL-assertion **LLM judge from inside the reward function**.  Those calls default to
``gpt-4.1`` at OpenAI; ``bootstrap_tau2`` redirects them to the configured judge endpoint, and on a
node without outbound access an un-redirected call would fail *during reward computation*.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .constants import DEFAULT_SOURCE_DIR, MANIFEST_SPECS
from .manifest import validate_manifest_directory, write_manifests
from .source import bootstrap_tau2, ensure_checkout, load_tasks

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"


def _reward_basis(task) -> tuple[str, ...]:
    criteria = getattr(task, "evaluation_criteria", None)
    basis = getattr(criteria, "reward_basis", None) or []
    return tuple(sorted(str(getattr(b, "name", b)).rsplit(".", 1)[-1] for b in basis))


def audit_judge_usage(source_dir: str) -> dict[str, dict]:
    """Per split: reward_basis histogram and the number of *actual* judge invocations.

    A task only costs a judge call when NL_ASSERTION is in its ``reward_basis`` **and** it carries
    non-empty ``nl_assertions`` — the evaluator short-circuits to 1.0 otherwise.
    """

    report: dict[str, dict] = {}
    for spec in MANIFEST_SPECS:
        tasks = load_tasks(spec.domain, spec.split_name)
        hist: collections.Counter = collections.Counter()
        nl_in_basis = 0
        judge_calls = 0
        for task in tasks:
            basis = _reward_basis(task)
            hist[basis] += 1
            if any("NL" in b.upper() for b in basis):
                nl_in_basis += 1
                criteria = getattr(task, "evaluation_criteria", None)
                if getattr(criteria, "nl_assertions", None):
                    judge_calls += 1
        report[spec.name] = {
            "tasks": len(tasks),
            "reward_basis": {"+".join(k) or "(none)": v for k, v in sorted(hist.items())},
            "nl_in_basis": nl_in_basis,
            "judge_calls": judge_calls,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare τ² manifests from the pinned checkout")
    parser.add_argument("command", choices=("prepare", "validate", "audit"))
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--allow-judge",
        action="store_true",
        help="permit `prepare` to succeed even when tasks would invoke the NL-assertion LLM judge",
    )
    args = parser.parse_args()

    if args.command == "prepare":
        source_dir = ensure_checkout(args.source_dir)
    else:
        source_dir = Path(args.source_dir).expanduser()
    bootstrap_tau2(source_dir)

    if args.command == "audit":
        print(json.dumps(audit_judge_usage(str(source_dir)), indent=2, sort_keys=True))
        return

    if args.command == "validate":
        print(json.dumps(validate_manifest_directory(args.output_dir), indent=2, sort_keys=True))
        return

    report = audit_judge_usage(str(source_dir))
    judged = {name: r["judge_calls"] for name, r in report.items() if r["judge_calls"]}
    counts = write_manifests(source_dir, args.output_dir)
    print(json.dumps({"source_revision_dir": str(source_dir), **counts}, indent=2, sort_keys=True))
    if judged:
        message = (
            f"NL-assertion judge will be invoked for {judged} task(s). Configure a judge redirect "
            "(bootstrap_tau2(judge_model=..., judge_llm_args=...)) or these reward computations will "
            "attempt an outbound OpenAI call."
        )
        if not args.allow_judge:
            raise SystemExit(f"ERROR: {message}\nRe-run with --allow-judge once a redirect is configured.")
        print(f"WARNING: {message}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Merge the two frontier teachers' τ² retail PI hints into ONE pool-scoped artifact, and report
where a hint can actually teach the student.

WHY MERGE. The 2026-08-02 PI pass produced hints from two teachers separately — `glm-5p2` (solve
0.6039) and `qwen3p7-plus` (0.5755) — each split by the OLD `tau2_train500` / `tau2_dev62` manifests.
Neither alone covers the pool we now train on, and the union is materially larger than either:
glm 340 + qwen 325 -> **union 410** of 563. That union headroom (+11 pts over the best single teacher)
was the strongest finding of the original PI run and it is thrown away by using one teacher's file.

⚠️ `actions` IS FORCED ONLY FOR STUDENT-SIDE INJECTION — NOT FOR OPSD. Which budget applies depends
entirely on WHERE the hint goes, and the two differ by ~6x:

  * STUDENT prompt (prepending the hint to what the policy sees): bounded by `MAX_PROMPT_LENGTH=6144`,
    against which τ²'s system prompt — domain policy + 16 tool schemas — already spends ~4.7k. Measured
    medians: raw 4,346 tok (glm) / 6,051 (qwen) and raw_no_think 2,909 / 3,640 cannot coexist with it;
    actions at 263 / 274 fits comfortably. Here `actions` really is the only option.
  * TEACHER ref prompt (OPSD): the student rolls out UNHINTED and the hint is prepended only to
    `ref_prompt_token_ids`, bounded by `OPSD_REF_SEQ_CAP` — default `max_input_length +
    max_generate_length` = 39,000, and set to 52,000 on the PI arms. **All three formats fit**, which
    is what makes a 3-way format comparison possible at all. Sized in
    `scripts/analysis/build_tau2_pi_pool.py`, which re-checks the headroom at build time.

Note the ordering INVERTS τ1's ranking finding (there, `raw` ranked trajectories best, AUC 0.740) —
ranking and guiding are different jobs and τ1's own frontier hints were net-NEGATIVE for guiding, so
do not assume a hint helps just because it is available.

⚠️ **HINT AVAILABILITY IS ANTI-CORRELATED WITH NEED**, which is the whole reason this script reports
buckets instead of just writing a file. A hint exists only where a FRONTIER model solved the task, and
difficulty correlates across models — so coverage is thinnest exactly where the student has headroom.
Measured against the Instruct 8-sample profile on the 478 pool:

    bucket   tasks   has hint   coverage
    0/8        136         51      37.5%   <- maximum headroom, worst coverage
    1/8         65         37      56.9%
    2/8         40         24      60.0%
    3/8         34         27      79.4%
    4/8         40         35      87.5%
    5-6/8       89         80      89.9%
    7/8         74         72      97.3%   <- teaches nothing, near-perfect coverage

Attaching every available hint would put 72 of them on tasks the student already solves 7/8 of the
time — pure prompt-budget cost — and only 51 on the 136 it never solves. `--target-band` therefore
keeps hints only where the student is neither saturated nor hopeless.

THE τ² HINT-INJECTION PATH EXISTS: `opsd/tau2_bench/distill/`
consumes these artifacts via `OPSD_HINTS_PATH`, keyed on the plain `task_id` this builder emits. It
injects hints into the TEACHER's ref prompt only, never the student's — so the format budget above is
`OPSD_REF_SEQ_CAP`, not `MAX_PROMPT_LENGTH`. `generator.py` / `main.py` / `agent.py` still contain zero
hint references, and that is correct: the student must roll out unhinted for OPSD's KL to be on-policy.

Output is `opsd_hints/v2`, the same schema the OPSD loader consumes.

Usage:
    .venv/bin/python scripts/analysis/build_tau2_retail_pi.py \\
        --pi-root <run_tau2_pi_all.sh output root> \\
        --profile <run_tau2_instruct_profile563.sh output>/trajectories.jsonl \\
        --pool opsd/tau2_bench/data/retail_areal_instruct_drop8of8.jsonl \\
        --out-dir ./outputs/pi_tau2_retail_merged
    ... --target-band 0 6      # keep hints only for students scoring 0-6 of 8 (drop the saturated)
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

# glm-5p2 outscored qwen3p7-plus on retail (0.6039 vs 0.5755), so it wins ties. Order matters:
# whichever teacher is listed first supplies the hint when both solved a task.
TEACHERS = ("glm_retail", "qwen_retail")
SPLITS = ("hints_train", "hints_dev")  # the OLD manifest split; both are needed for full coverage


def load_teacher(pi_root: Path, teacher: str, fmt: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for split in SPLITS:
        path = pi_root / teacher / split / f"hints_{fmt}.json"
        if not path.is_file():
            continue
        blob = json.loads(path.read_text())
        blob.pop("__schema__", None)
        out.update(blob)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pi-root", required=True, type=Path)
    ap.add_argument(
        "--profile",
        type=Path,
        help="student trajectories.jsonl (563 tasks x8) — REQUIRED for bucket reporting "
        "and for --target-band; without it the script cannot tell a hint that "
        "teaches from one that is dead weight",
    )
    ap.add_argument("--pool", type=Path, help="training manifest to scope to (default: no scoping)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument(
        "--fmt",
        default="actions",
        choices=("actions", "raw", "raw_no_think"),
        help="actions is the only format that fits the prompt budget; see docstring",
    )
    ap.add_argument(
        "--target-band",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        help="keep hints only for tasks whose student score is in [MIN,MAX] of 8",
    )
    args = ap.parse_args()

    merged: dict[str, object] = {}
    provenance: dict[str, str] = {}
    per_teacher: dict[str, int] = {}
    for teacher in TEACHERS:
        got = load_teacher(args.pi_root, teacher, args.fmt)
        per_teacher[teacher] = len(got)
        for task, hint in got.items():
            if task not in merged:  # first teacher listed wins
                merged[task] = hint
                provenance[task] = teacher
    print(f"format={args.fmt}  " + "  ".join(f"{t}={n}" for t, n in per_teacher.items()) + f"  union={len(merged)}")

    buckets: dict[str, int] = {}
    if args.profile:
        acc = collections.defaultdict(list)
        for line in open(args.profile):
            rec = json.loads(line)
            # tau2 records carry `reward`; `raw_reward` is a tau1 field name and is absent here --
            # reading it silently scores every task 0.
            acc[str(rec["task_id"])].append(1 if float(rec.get("reward") or 0.0) > 0.5 else 0)
        buckets = {t: sum(v) for t, v in acc.items()}

    pool = None
    if args.pool:
        pool = {json.loads(l)["task_id"] for l in args.pool.read_text().splitlines() if l.strip()}
        print(f"pool {args.pool.name}: {len(pool)} tasks")

    scope = pool if pool is not None else set(buckets) or set(merged)
    if buckets:
        print(f"\n{'bucket':>7} {'tasks':>6} {'hint':>6} {'coverage':>9}")
        for b in range(9):
            ts = {t for t in scope if buckets.get(t) == b}
            if not ts:
                continue
            h = len(ts & set(merged))
            print(f"  {b}/8 {len(ts):>7} {h:>6} {h/len(ts):>8.1%}")

    keep = set(merged) & scope if pool is not None else set(merged)
    dropped_saturated = 0
    if args.target_band:
        lo, hi = args.target_band
        if not buckets:
            raise SystemExit("--target-band needs --profile")
        before = len(keep)
        keep = {t for t in keep if lo <= buckets.get(t, -1) <= hi}
        dropped_saturated = before - len(keep)
        print(
            f"\n--target-band {lo}-{hi}: kept {len(keep)}, dropped {dropped_saturated} "
            f"whose student score is outside the band (a hint there teaches nothing)"
        )

    out = {"__schema__": "opsd_hints/v2"}
    out.update({t: merged[t] for t in sorted(keep)})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dest = args.out_dir / f"hints_{args.fmt}.json"
    dest.write_text(json.dumps(out, indent=1, sort_keys=True))

    tc = collections.Counter(provenance[t] for t in keep)
    report = {
        "schema": "opsd_hints/v2",
        "format": args.fmt,
        "n_hints": len(keep),
        "per_teacher_available": per_teacher,
        "union_available": len(merged),
        "teacher_of_kept": dict(tc),
        "pool": str(args.pool) if args.pool else None,
        "pool_size": len(pool) if pool else None,
        "target_band": args.target_band,
        "dropped_outside_band": dropped_saturated,
        "bucket_coverage": (
            {
                f"{b}/8": {
                    "tasks": len({t for t in scope if buckets.get(t) == b}),
                    "with_hint": len({t for t in scope if buckets.get(t) == b} & set(merged)),
                }
                for b in range(9)
                if any(buckets.get(t) == b for t in scope)
            }
            if buckets
            else None
        ),
        "uncoverable": sorted(t for t in scope if buckets.get(t) == 0 and t not in merged) if buckets else None,
    }
    (args.out_dir / "pi_report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nwrote {len(keep)} hints -> {dest}")
    print(f"  teacher split of kept: {dict(tc)}")
    if report["uncoverable"]:
        print(
            f"  ⚠️ {len(report['uncoverable'])} tasks are 0/8 for the student AND have no hint — "
            f"neither GRPO nor PI can teach these; consider excluding them from the pool"
        )
    print(f"  report -> {args.out_dir/'pi_report.json'}")


if __name__ == "__main__":
    main()

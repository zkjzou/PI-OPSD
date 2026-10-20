"""Aggregate PI-guided-generation conditions into a utility-Δ table.

utility Δ = success(with-hint) − success(no-hint), the PI-guides-generation (utility) metric of a
privileged-information screen. Reports both the pooled success rate and the paired per-task Δ (same
task_ids across conditions), plus turn / invalid-action / stop-reason deltas so a success change can be
read against behaviour change.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_rows(cond_dir: Path, schema: str = "alfworld") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if schema == "appworld":
        # AppWorld's driver writes FLAT rows ({task_id, reward, success, num_turns, stop_reason}) with no
        # env_metrics nesting, and TWO jsonl files per condition: trajectories.jsonl (one row per rollout)
        # and transcripts.jsonl (full conversations, NO success field). Globbing *.jsonl the ALFWorld way
        # would fold the transcripts in as success-0 rows and roughly halve every success rate.
        for f in sorted(glob.glob(str(cond_dir / "**" / "trajectories.jsonl"), recursive=True)):
            with open(f) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    r.setdefault(
                        "env_metrics",
                        {
                            "success": float(r.get("success", 0.0) or 0.0),
                            "num_turns": float(r.get("num_turns", 0.0) or 0.0),
                            # AppWorld has no invalid-action counter: under STRICT_PARSE a fence-less turn is
                            # a WASTED turn, so it surfaces in num_turns rather than here.
                            "num_invalid_actions": 0.0,
                        },
                    )
                    rows.append(r)
        return rows
    for f in glob.glob(str(cond_dir / "**" / "*.jsonl"), recursive=True):
        if "alfworld_servers" in f:
            continue
        with open(f) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def paired_stats(paired: List[float], seed: int = 0, iters: int = 10000) -> Dict[str, Any]:
    """Dispersion for the WITHIN-TASK (paired) Δ. The mean alone is not actionable.

    The paired mean is the metric to read -- pooled contrasts were directionally WRONG for 2 of 3
    analyzers on ALFWorld, and AppWorld task difficulty spans a comparable range. But a bare mean hides
    whether a +0.03 is a real effect or four lucky tasks, and these tables get compared across 6 arms
    where the between-arm spread is often smaller than the within-arm noise.

    Three numbers, none needing scipy:
      se        normal-approx standard error of the mean over tasks (sample sd / sqrt(n)).
      ci95      BOOTSTRAP percentile CI, resampling TASKS. Preferred over mean+-1.96se because per-task
                deltas are bounded in [-1, 1], heavily tied at 0, and visibly non-normal at n=90.
      sign_p    exact two-sided sign test on improved vs hurt, ties DROPPED. This is the assumption-free
                check: it asks only whether more tasks got better than worse, so it survives the fact
                that the baseline is n=16 and the with-hint arms are n=4 (very different per-task
                variance, which a t-style test would not handle honestly).
    Deterministic seed so re-running the aggregator on the same corpus reproduces the CI exactly.
    """
    n = len(paired)
    if n == 0:
        return {"paired_n": 0, "paired_se": None, "paired_ci95": None, "sign_p": None}
    mean = sum(paired) / n
    if n > 1:
        var = sum((d - mean) ** 2 for d in paired) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        means.append(sum(paired[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[min(int(0.975 * iters), iters - 1)]
    # exact two-sided sign test, ties excluded
    up = sum(1 for d in paired if d > 0)
    dn = sum(1 for d in paired if d < 0)
    m = up + dn
    if m == 0:
        p = 1.0
    else:
        k = min(up, dn)
        tail = sum(math.comb(m, i) for i in range(0, k + 1)) / (2**m)
        p = min(1.0, 2 * tail)
    return {
        "paired_n": n,
        "paired_se": round(se, 4),
        "paired_ci95": [round(lo, 4), round(hi, 4)],
        "sign_p": round(p, 4),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    succ = [float(r["env_metrics"].get("success", 0.0)) for r in rows]
    turns = [float(r["env_metrics"].get("num_turns", 0.0)) for r in rows]
    inval = [float(r["env_metrics"].get("num_invalid_actions", 0.0)) for r in rows]
    by_task: Dict[str, List[float]] = defaultdict(list)
    for r, s in zip(rows, succ):
        by_task[r.get("task_id")].append(s)
    per_task_rate = {t: sum(v) / len(v) for t, v in by_task.items()}
    solved_any = sum(1 for v in by_task.values() if any(x > 0 for x in v))
    return {
        "n": n,
        "n_tasks": len(by_task),
        "success_rate": round(sum(succ) / n, 4),
        "solve_at_1_pooled": round(solved_any / len(by_task), 4),  # tasks with >=1 win / tasks
        "avg_turns": round(sum(turns) / n, 2),
        "avg_invalid": round(sum(inval) / n, 2),
        "stop_reasons": dict(Counter(r.get("stop_reason") for r in rows)),
        "_per_task_rate": per_task_rate,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument(
        "--baseline",
        required=True,
        help="condition name to difference against; MUST also appear in --conditions "
        "(omitting it is a KeyError that discards the whole table).",
    )
    ap.add_argument(
        "--schema",
        choices=("alfworld", "appworld"),
        default="alfworld",
        help="row schema: `appworld` reads only trajectories.jsonl and lifts the flat "
        "success/num_turns fields into env_metrics.",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="markdown caption (benchmark / split / model / sampling). Default: derived "
        "from --schema and the --root directory name.",
    )
    args = ap.parse_args()
    root = Path(args.root)

    summ: Dict[str, Dict[str, Any]] = {}
    for c in args.conditions:
        summ[c] = summarize(load_rows(root / c, args.schema))

    base = summ[args.baseline]
    base_rate = base.get("_per_task_rate", {})

    table = []
    for c in args.conditions:
        s = summ[c]
        if not s.get("n"):
            table.append({"condition": c, "n": 0})
            continue
        # paired per-task Δ over the tasks both conditions cover
        common = set(s["_per_task_rate"]) & set(base_rate)
        paired = [s["_per_task_rate"][t] - base_rate[t] for t in common]
        improved = sum(1 for d in paired if d > 0)
        hurt = sum(1 for d in paired if d < 0)
        ps = paired_stats(paired)
        row = {
            "condition": c,
            "n": s["n"],
            "n_tasks": s["n_tasks"],
            "success_rate": s["success_rate"],
            "utility_delta_pooled": round(s["success_rate"] - base["success_rate"], 4),
            "utility_delta_paired_mean": round(sum(paired) / len(paired), 4) if paired else None,
            "tasks_improved": improved,
            "tasks_hurt": hurt,
            "tasks_tied": len(paired) - improved - hurt,
            "solve_at_1": s["solve_at_1_pooled"],
            "avg_turns": s["avg_turns"],
            "avg_invalid": s["avg_invalid"],
            "stop_reasons": s["stop_reasons"],
        }
        row.update(ps)
        # Rendered form of the CI, so the markdown table stays readable at a glance. `sig` is the
        # bootstrap CI excluding 0 -- the honest "is this arm's within-task effect distinguishable from
        # nothing" flag. Report it per arm; do NOT use it to rank arms against each other (that needs a
        # paired test BETWEEN arms, which this table does not compute).
        ci = ps.get("paired_ci95")
        row["paired_ci95_str"] = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else ""
        row["sig"] = "yes" if ci and (ci[0] > 0 or ci[1] < 0) else "no"
        table.append(row)

    out = {"root": str(root), "baseline": args.baseline, "table": table}
    (root / "utility_delta.json").write_text(json.dumps(out, indent=2))

    # WITHIN-TASK columns sit immediately after the paired mean, so the mean is never read without its
    # uncertainty. pooled stays in the table for continuity but is the column NOT to draw conclusions from.
    cols = [
        "condition",
        "n",
        "success_rate",
        "utility_delta_pooled",
        "utility_delta_paired_mean",
        "paired_ci95_str",
        "paired_se",
        "sign_p",
        "sig",
        "paired_n",
        "tasks_improved",
        "tasks_hurt",
        "tasks_tied",
        "solve_at_1",
        "avg_turns",
        "avg_invalid",
    ]
    # Title must name the actual run. It used to be hardcoded to "ALFWorld valid_unseen fullset,
    # Qwen3.5-2B", so every table claimed that model and split no matter what it scored -- a
    # Qwen3-4B-Instruct-2507 seen-split run and AppWorld tables came out mislabelled, which is exactly the
    # kind of caption that gets a number quoted against the wrong benchmark later. Pass --title for the
    # full description; otherwise fall back to the schema + root dir name rather than a stale constant.
    title = args.title if args.title else f"`{args.schema}`, {root.name}"
    lines = [
        f"# PI-guided generation — utility Δ ({title})",
        "",
        f"utility Δ = success(with-hint) − success(no-hint). Baseline = **{args.baseline}**. "
        "paired_mean = mean over per-task success-rate deltas (same task_ids).",
        "",
        "**Read `utility_delta_paired_mean` with its CI, not `utility_delta_pooled`.** Pooled "
        "contrasts were directionally WRONG for 2 of 3 analyzers on ALFWorld because task "
        "difficulty varies far more than the hint effect. `paired_ci95_str` is a 10k bootstrap over "
        "TASKS (seeded, reproducible); `sign_p` is an exact two-sided sign test with ties dropped; "
        "`sig` = the CI excludes 0. `sig` judges ONE arm against no-effect — it does not rank arms "
        "against each other, which needs a paired between-arm test this table does not compute.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for r in table:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    md = "\n".join(lines) + "\n"
    (root / "utility_delta.md").write_text(md)
    print(md)
    print(f"wrote {root}/utility_delta.{{md,json}}")


if __name__ == "__main__":
    main()

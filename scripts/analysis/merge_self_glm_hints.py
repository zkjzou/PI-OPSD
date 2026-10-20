#!/usr/bin/env python
"""Build full-coverage PI hint artifacts by backfilling self-hints with frontier (GLM) hints.

WHY. A `self` PI source can only cover the tasks the policy actually solved. For the Instruct DAPO
checkpoint at step 200 that is 1432/1600 train tasks -- the remaining 168 have no successful
trajectory to distil, and no amount of resampling fixes it (a second independent N=8 on them flipped
only 4). Training needs a hint for every prompt, so those 168 fall back to the GLM-5.2 reference.

The output is therefore a HYBRID source. Every record carries `meta["pi_source"]` = "self" | "glm" so
downstream analysis can split by it -- important, because a hybrid arm is NOT a clean "self vs
frontier" contrast and any per-arm number computed over all 1600 mixes the two.

Only `raw` and `actions` are built. `raw_no_think` is deliberately omitted for the self source: this
checkpoint emits pure <action> turns with zero prose and zero <think> tags, so the no-think transform
is a no-op and the artifact came out byte-identical to `raw` on 1432/1432 tasks.

THE GLM BACKFILL OF `raw` READS GLM'S `raw_no_think` ARTIFACT, NOT ITS `raw`. Otherwise the hybrid is
internally inconsistent: the 1432 self tasks would get observation+action transcripts (the policy emits
nothing else) while the 168 backfilled tasks would get transcripts that ALSO carry GLM's reasoning
prose -- a different kind of hint on a non-random subset, namely the hardest tasks. Measured on those
168, GLM's `raw` has prose in 55.4% of agent turns (92k chars) vs 0.5% (3k) after the strip, so the
difference is large, not cosmetic. Using no_think also cuts cap-truncated hints from 64 to 51.

  python scripts/analysis/merge_self_glm_hints.py \
      --self-dir  .../hints_self --glm-dir .../glmhints \
      --task-ids  .../task_ids_train1600.json --out-dir .../hints_self_glm
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

from opsd.pi.artifact import load_hint_artifact, save_hint_artifact
from opsd.pi.base import HintRecord

# output format -> (self artifact, glm artifact). The asymmetry on `raw` is deliberate; see module doc.
FORMATS = {
    "raw": ("hints_raw.json", "hints_raw_no_think.json"),
    "actions": ("hints_actions.json", "hints_actions.json"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-dir", required=True)
    ap.add_argument("--glm-dir", required=True)
    ap.add_argument("--task-ids", required=True, help="the full task-id list the artifact must cover")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    task_ids = json.loads(pathlib.Path(args.task_ids).read_text())
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    for fmt, (self_file, glm_file) in FORMATS.items():
        selfh = load_hint_artifact(f"{args.self_dir}/{self_file}")
        glmh = load_hint_artifact(f"{args.glm_dir}/{glm_file}")
        if self_file != glm_file:
            print(f"[{fmt}] self <- {self_file}   glm <- {glm_file}  (asymmetric by design)")

        merged: dict[str, HintRecord] = {}
        n_self = n_glm = n_missing = 0
        for tid in task_ids:
            if tid in selfh and selfh[tid].task:
                rec = selfh[tid]
                meta = {**(rec.meta or {}), "pi_source": "self"}
                merged[tid] = HintRecord(task=rec.task, steps=rec.steps, meta=meta)
                n_self += 1
            elif tid in glmh and glmh[tid].task:
                rec = glmh[tid]
                meta = {**(rec.meta or {}), "pi_source": "glm"}
                merged[tid] = HintRecord(task=rec.task, steps=rec.steps, meta=meta)
                n_glm += 1
            else:
                n_missing += 1

        save_hint_artifact(str(out_dir / f"hints_{fmt}.json"), merged)
        cs = lambda src: [len(r.task) for r in merged.values() if (r.meta or {}).get("pi_source") == src]
        s_chars, g_chars = cs("self"), cs("glm")
        report[fmt] = {
            "n_total": len(merged),
            "n_self": n_self,
            "n_glm": n_glm,
            "n_missing": n_missing,
            "mean_chars_self": round(statistics.mean(s_chars)) if s_chars else None,
            "mean_chars_glm": round(statistics.mean(g_chars)) if g_chars else None,
        }
        print(
            f"{fmt:8s} total {len(merged):4d}  self {n_self:4d}  glm {n_glm:3d}  missing {n_missing:3d}"
            f"   mean chars: self {report[fmt]['mean_chars_self']}  glm {report[fmt]['mean_chars_glm']}"
        )
        if n_missing:
            print(f"  WARNING: {n_missing} task(s) have NO hint from either source")

    (out_dir / "merge_summary.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

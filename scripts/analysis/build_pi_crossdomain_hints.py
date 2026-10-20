#!/usr/bin/env python
"""Build a CROSS-DOMAIN mismatched-PI null: same PI format, hints drawn from a DIFFERENT benchmark.

The n1/n2/n3 ladder destroys task correspondence but keeps the hint inside ALFWorld -- even n3 hands
over a competent ALFWorld household trajectory with the right verbs, receptacles and action grammar.
This tier goes further: the hint is a real trajectory-with-reasoning hint, in the same PI format, but
from tau2 (customer-service dialogue) or AppWorld (code/API agent). It asks whether the teacher's
conditioning needs to be about THIS DOMAIN at all, or merely competent-agent-shaped.

TWO PROPERTIES OF THE n-LADDER THIS CANNOT PRESERVE, and both are recorded rather than hidden:

1. POOL SIZE -> SAMPLING WITH REPLACEMENT. tau2 has 424 hints and AppWorld 160, against 1600 ALFWorld
   train tasks, so donors are necessarily REUSED (~3.8x and ~10x). n1-n3 are permutations where every
   hint appears exactly once. Reuse is spread as evenly as possible via a per-donor usage cap of
   ceil(n_tasks / n_donors) rather than left to chance, because unbounded nearest-length matching
   concentrates hard on the few shortest donors.

2. LENGTH. Raw cross-domain hints are 1.9-3.1x longer than glt (medians 11,986 / 19,618 vs 6,355
   chars), and hint length has a measured +0.080-per-log-word slope on outcome. So each task gets the
   NEAREST-LENGTH available donor, minimising |len(donor) - len(target)| subject to the usage cap. The
   length distribution still cannot match glt's -- tau2's p10 alone exceeds glt's median -- so the
   residual offset is reported and must be carried into any interpretation.

REASONING SURFACE IS DELIBERATELY NOT NORMALISED. The donor artifacts carry `<think>` while glt carries
`THOUGHTS:` (glt exists precisely because Instruct students never emit `<think>`). Keeping the donor
byte-for-byte means this arm differs from glt in surface AS WELL AS domain, so an effect cannot be
attributed to domain alone. That was an explicit choice -- the alternative (`<think>` -> `THOUGHTS:`,
the same substitution that built glt) is one flag away via --normalise-surface.

  python scripts/analysis/build_pi_crossdomain_hints.py \
      --target <glt.json> --task-ids <train1600.json> --donor <tau2 hints_raw.json> \
      --label tau2 --out <hints_null_xd_tau2.json>

POOLED DONORS. Pass --donor more than once as LABEL=PATH to pool several benchmarks; donor ids become
LABEL::ID. The paper's cross-domain null (the ALFWorld `null_xdomain` arm,
glt_xdmix_tau2_appworld.json) is

  python scripts/analysis/build_pi_crossdomain_hints.py \
      --target $OPSD_DATA_ROOT/alfworld/hints/glm-5.2/hints_traj_thoughts_cap24k.json \
      --task-ids $OPSD_DATA_ROOT/alfworld/splits/task_ids_train1600.json \
      --donor tau2=$OPSD_DATA_ROOT/tau2/hints/glm-5.2/pool424/hints_raw.json \
      --donor appworld=<GLM-5.2 AppWorld test_normal hints_raw.json> \
      --label xdmix --source-artifact xd_pooled_donors.json --out glt_xdmix_tau2_appworld.json

The AppWorld donors are test_normal reference solutions (appworld_opsd_test_normal_hints.bundle): a model
trained with this null must never be evaluated on AppWorld test_normal.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from pathlib import Path

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def body(v):
    return v["task"] if isinstance(v, dict) and "task" in v else str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="the arm being nulled, e.g. glt (for lengths+keys)")
    ap.add_argument("--task-ids", required=True)
    ap.add_argument("--donor", required=True, action="append",
                    help="other-benchmark hint artifact, same PI format; repeat as LABEL=PATH to pool")
    ap.add_argument("--label", required=True, help="tau2 | appworld")
    ap.add_argument("--out", required=True)
    ap.add_argument("--normalise-surface", action="store_true",
                    help="<think>X</think> -> THOUGHTS: X, matching glt's surface")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source-artifact", default=None, help="name recorded in meta.source_artifact")
    a = ap.parse_args()

    ids = set(json.loads(Path(a.task_ids).read_text()))
    tgt_raw = json.loads(Path(a.target).read_text()); schema = tgt_raw.pop("__schema__", None)
    tgt = {k: body(v) for k, v in tgt_raw.items() if k in ids}
    dnr = {}
    pooled = len(a.donor) > 1 or "=" in a.donor[0]
    for spec in a.donor:
        lab, _, path = spec.rpartition("=") if "=" in spec else ("", "", spec)
        raw = json.loads(Path(path).read_text()); raw.pop("__schema__", None)
        for k, v in raw.items():
            dnr[f"{lab}::{k}" if pooled else k] = body(v)
    src_name = a.source_artifact or (Path(a.donor[0]).name if not pooled else "pooled")
    if a.normalise_surface:
        dnr = {k: _THINK.sub(lambda m: "THOUGHTS: " + m.group(1).strip(), v) for k, v in dnr.items()}

    rng = random.Random(a.seed)
    cap = math.ceil(len(tgt) / len(dnr))
    dkeys = sorted(dnr, key=lambda k: len(dnr[k]))
    dlen = [len(dnr[k]) for k in dkeys]
    used = {k: 0 for k in dkeys}

    # Longest targets first: they have the fewest length-compatible donors, so serving them while the
    # pool is still unconstrained gives a better overall match than a random order.
    order = sorted(tgt, key=lambda k: -len(tgt[k]))
    out, pairs = {}, []
    for t in order:
        want = len(tgt[t])
        avail = [k for k in dkeys if used[k] < cap]
        best = min(avail, key=lambda k: (abs(len(dnr[k]) - want), rng.random()))
        used[best] += 1
        rec = {"task": dnr[best],
               "meta": {"null_tier": f"xd_{a.label}", "donor_benchmark": a.label,
                        "donor_task_id": best, "donor_chars": len(dnr[best]),
                        "target_chars": want, "surface_normalised": bool(a.normalise_surface),
                        "source_artifact": src_name}}
        out[t] = rec
        pairs.append((want, len(dnr[best])))

    assert set(out) == ids, f"coverage {len(out)}/{len(ids)}"
    reuse = sorted(used.values())
    tl = sorted(p[0] for p in pairs); dl = sorted(p[1] for p in pairs)
    ratio = statistics.median(dl) / statistics.median(tl)
    absdiff = statistics.median(abs(d - t) for t, d in pairs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(({"__schema__": schema} if schema else {}) | out))
    print(f"  {a.label}: {len(out)}/{len(ids)} tasks | donors {len(dnr)} used, reuse min {reuse[0]} "
          f"max {reuse[-1]} (cap {cap})")
    print(f"     chars: target med {statistics.median(tl):.0f} -> donor med {statistics.median(dl):.0f} "
          f"({ratio:.2f}x), median |diff| {absdiff:.0f}")
    print(f"     surface: <think> kept" if not a.normalise_surface else "     surface: normalised to THOUGHTS:")
    print(f"     -> {a.out}")


if __name__ == "__main__":
    main()

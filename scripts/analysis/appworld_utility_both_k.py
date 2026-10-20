#!/usr/bin/env python
"""AppWorld PI utility at BOTH k=1 and k=4, optionally gated against published dp@4 values.

AUXILIARY REPORT -- it does not reproduce any published table. The paper's AppWorld utility column (U)
comes from build_appworld_traindev_pi_table.py. This script prints dp@1 / dp@4 per (student, arm) for
whatever utility tree it is pointed at. On the released bundle that tree is train90 only, so its numbers
cover train90 only. The original dp@4 values it was checked against pooled other splits (train90 +
test_normal for the 4bi and 30bi hint arms) and are not released; --published accepts such a file if you
have one.

BASELINE IS PER STUDENT AND MUST BE NAMED EXPLICITLY. Each student's no-hint arm is its own n=128 pass@k
corpus (same sampler and context limits as the hinted n=4 arms); differencing one student's hint arm
against another student's corpus would be a cross-model contrast, so the student is part of every path.

LAYOUT (your utility runs, arranged as):
    <utility-root>/<student>/<arm>/trajectories.jsonl            hinted arms, n=4 per task
    <utility-root>/<student>/nohint_passk128/trajectories.jsonl  the no-hint baseline
with <utility-root> defaulting to ./outputs/utility/appworld. A per-task paired
delta is taken over the tasks an arm shares with its baseline; when a file holds several splits (train90
+ dev) the pooled result equals concatenating the per-split deltas, because task ids are split-specific.

Rows whose dp@4 does not reproduce a value in --published (JSON {"<student>|<arm>": dp4}) are flagged
DIFF; without --published the gate column reads '-'.

    python appworld_utility_both_k.py [--utility-root DIR] [--published pub.json] [--out out.json]
"""

import argparse
import collections
import json
import os
import sys
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pi_src_lib as S  # noqa: E402


def pk(s, k):
    n, c = len(s), sum(s)
    if k == 1:
        return c / n if n else None
    if n < k:
        return None
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def load(p):
    if not p or not os.path.exists(p):
        return None
    per = collections.defaultdict(list)
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line)
        per[str(r["task_id"])].append(float(r.get("success", r.get("reward", 0)) or 0) > 0)
    return per


# output key -> arm directory in the utility tree
TRAJ = {
    "glm_actions": "glm52_actions",
    "glm_rnt": "glm52_raw_no_think",
    "glm_raw": "glm52_raw",
    "qwn_actions": "qwen37_actions",
    "qwn_rnt": "qwen37_raw_no_think",
    "qwn_raw": "qwen37_raw",
}
HINT = ["anlz_glm", "summ_glm", "chk_glm", "anlz_qwn", "summ_qwn", "chk_qwn", "anlz_self", "summ_self"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--utility-root", default=None, help="default: ./outputs/utility/appworld")
    ap.add_argument("--baseline-arm", default="nohint_passk128")
    ap.add_argument("--students", default="4bi,4bt,30bi")
    ap.add_argument("--published", default=None, help='optional JSON {"<student>|<arm>": published dp@4}')
    ap.add_argument("--out", default="./outputs/appworld_utility_both_k/aw_util_both_k.json")
    a = ap.parse_args()
    root = a.utility_root or os.path.join("outputs", "utility", "appworld")
    if not os.path.isdir(root):
        sys.exit(f"{root}: no utility rollouts; run the utility launchers first (scripts/appworld/pi_utility.sh)")
    pub_all = json.load(open(a.published)) if a.published else {}

    out = {}
    for stu in a.students.split(","):
        base = load(f"{root}/{stu}/{a.baseline_arm}/trajectories.jsonl")
        print(f"\n=== {stu} ===")
        if base is None:
            print(f"  (no baseline at {root}/{stu}/{a.baseline_arm})")
            continue
        print(f"  {'arm':<12}{'n':>5}{'dp@1':>10}{'dp@4':>10}{'pub dp@4':>10}{'gate':>7}")
        for arm in list(TRAJ) + HINT:
            per = load(f"{root}/{stu}/{TRAJ.get(arm, arm)}/trajectories.jsonl")
            if not per:
                print(f"  {arm:<12}(no source)")
                continue
            rec = {}
            for k in (1, 4):
                bp = {t: pk(v, k) for t, v in base.items() if pk(v, k) is not None}
                dd = [pk(v, k) - bp[t] for t, v in per.items() if t in bp and pk(v, k) is not None]
                rec[f"d{k}"] = float(np.mean(dd)) if dd else float("nan")
                rec["n"] = len(dd)
            pub = pub_all.get(f"{stu}|{arm}")
            gate = "-" if pub is None else ("OK" if abs(rec["d4"] - pub) < 1.5e-3 else "DIFF")
            out[f"{stu}|{arm}"] = dict(rec, gate=gate)
            ps = f"{pub:+.4f}" if pub is not None else "     --"
            print(f"  {arm:<12}{rec['n']:>5}{rec['d1']:>+10.4f}{rec['d4']:>+10.4f}{ps:>10}{gate:>7}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

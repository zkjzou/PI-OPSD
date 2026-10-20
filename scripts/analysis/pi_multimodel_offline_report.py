#!/usr/bin/env python
"""d and Δd⁻_err of raw PI from different authors on the 200 offline tasks (pi_multimodel_offline_prepare.py).

Arms: the four OPSD-contract authors (deepseek-v4p1-flash, minimax-m3, qwen3p8-max, kimi-k3) plus the
published R1-contract GLM-5.2 / Qwen3.7-Plus arms of the cap24k cell as references. Every arm is scored on the
same student rollouts (all 128 per task). Estimators are the published ones (pi_glm8_report); `level` and
`ratio` are pi_selfwrong_explain's (step-mean d⁻ over all scored steps, and Δd⁻_err / level).

Two views: the tasks EVERY new author solved (common set, like-for-like) and each author's own coverage.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alf_study_common as C  # noqa: E402
import pi_src_lib as S  # noqa: E402
from pi_glm8_report import CAP24K, COLS, LABELS, d_by, derr_frame  # noqa: E402
from pi_self4_report import load_chunks  # noqa: E402

PUBLISHED = ["glm_actions", "glm_raw_no_think", "glm_raw", "qwen37_raw"]


def metrics(rows: pd.DataFrame, labels: dict) -> pd.DataFrame:
    ps = S.per_step(rows)
    level = ps.groupby("pi")["dminus"].mean()
    fe = derr_frame(rows, labels)
    t = pd.DataFrame(
        {
            "d": d_by(rows, ["pi"]),
            "derr": fe.groupby("pi").dev.mean(),
            "level_dminus": level,
            "tasks": rows.groupby("pi").task_id.nunique(),
            "flagged_steps": fe.groupby("pi").size(),
        }
    )
    t["ratio"] = t["derr"] / t["level_dminus"]
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--w",
        action="append",
        default=None,
        help="scoring dir(s) of pi_multimodel_offline_prepare.py (repeatable; authors are pooled)",
    )
    ap.add_argument(
        "--partial",
        action="store_true",
        help="report on the chunks finished so far (preliminary; writes report_partial/)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="default ./outputs/pi_multimodel_offline_report/<last --w name>/{report,report_combined,report_partial}",
    )
    args = ap.parse_args()
    ws = [Path(w) for w in (args.w or [str(C.PI_STUDY_ROOT / "pi_mm_offline200")])]
    out = Path(
        args.out
        or C.default_out("pi_multimodel_offline_report")
        / ws[-1].name
        / ("report_partial" if args.partial else ("report" if len(ws) == 1 else "report_combined"))
    )
    out.mkdir(parents=True, exist_ok=True)
    man = {}
    for w in ws:  # an author may be split across dirs (e.g. top-up tasks): union its task list
        for m, v in json.load(open(w / "hints/manifest.json")).items():
            if m in man:
                man[m]["tasks"] = sorted(set(man[m]["tasks"]) | set(v["tasks"]))
                man[m]["hinted_tasks"] = len(man[m]["tasks"])
            else:
                man[m] = dict(v)
    labels = json.load(open(LABELS))

    parts = [load_chunks(w) for w in ws]
    if not args.partial:  # the scorer writes rows only for tasks some author in that dir has a hint for
        for w, r in zip(ws, parts):
            hinted = set().union(*(set(v["tasks"]) for v in json.load(open(w / "hints/manifest.json")).values()))
            assert r.uid.nunique() == 128 * len(hinted), (w, r.uid.nunique(), len(hinted))
            print(f"{w.name}: {r.uid.nunique()} rollouts = 128 x {len(hinted)} tasks with a hint")
    rows = pd.concat(parts, ignore_index=True)
    pub = pd.read_parquet(CAP24K, columns=COLS, filters=[("pi", "in", PUBLISHED)])
    allrows = pd.concat([rows, pub[pub.uid.isin(set(rows.uid))]], ignore_index=True)
    scored = set(rows.task_id)
    common = sorted(set.intersection(*(set(v["tasks"]) for v in man.values())) & scored)
    print(f"scored tasks {len(scored)}/200 ({rows.uid.nunique()} rollouts)")

    views = {}
    for name, sel in (("common", allrows.task_id.isin(common)), ("own_coverage", slice(None))):
        t = metrics(allrows[sel], labels)
        views[name] = t
        print(
            f"\n=== {name}: "
            + (
                f"{len(common)} tasks every new author solved"
                if name == "common"
                else "each arm on the tasks it has a hint for"
            )
            + " ==="
        )
        print(t.sort_values("d").round(4).to_string())

    d_task = d_by(allrows[allrows.task_id.isin(common)], ["pi", "task_id"]).unstack(0)
    e_task = derr_frame(allrows[allrows.task_id.isin(common)], labels).groupby(["task_id", "pi"]).dev.mean().unstack(1)
    paired = {}
    for p in man:
        pd_ = (d_task[p] - d_task["glm_raw"]).dropna()
        pe_ = (e_task[p] - e_task["glm_raw"]).dropna()
        paired[p] = {
            "d_minus_glm_raw": float(pd_.mean()),
            "d_higher_on": f"{int((pd_ > 0).sum())}/{len(pd_)}",
            "derr_minus_glm_raw": float(pe_.mean()),
            "derr_higher_on": f"{int((pe_ > 0).sum())}/{len(pe_)}",
        }
    hint_desc = {m: {k: v for k, v in s.items() if k != "tasks"} for m, s in man.items()}
    (out / "summary.json").write_text(
        json.dumps(
            {
                "common_tasks": len(common),
                "views": {k: json.loads(v.round(4).to_json(orient="index")) for k, v in views.items()},
                "paired_vs_glm_raw_on_common": paired,
                "hints": hint_desc,
            },
            indent=1,
        )
    )
    print("\npaired per task vs glm_raw (common tasks):", json.dumps(paired, indent=1))
    print("hint descriptors:", json.dumps(hint_desc, indent=1))


if __name__ == "__main__":
    main()

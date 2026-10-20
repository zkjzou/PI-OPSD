#!/usr/bin/env python
"""Re-render the ALFWorld frontier reference trajectories into the three uncapped hint formats.

Per source (e.g. glm = GLM-5.2, qwen37 = Qwen3.7-Plus), the FIRST won trajectory per task across the given
trajectory files (in order) becomes that task's hint, in three opsd_hints/v2 artifacts:

  REBUILT_<src>__traj_withthink.json   role-tagged transcript with each turn's out-of-band reasoning folded
                                       back in as <think>...</think> before the visible text (glw / glr arms)
  REBUILT_<src>__traj_nothink.json     the same transcript without the reasoning (TrajView.render)
  REBUILT_<src>__actions.json          "Reference actions (in order):" numbered action list

Uncapped: alfworld_cap_many_24k.py / alfworld_cap_withthink_24k.py then cap them at 24k student tokens
(the published hints_traj_{withthink,nothink}_cap24k.json). The rendering logic is the one that produced the
published artifacts, unchanged; only the input/output paths are arguments.

The published glm / qwen37 sources are TWO trajectory files each, in this order: the pi_train2400 round
(alfworld/reference_trajectories/<author>.jsonl in the data bundle) and the earlier pi_train1k round.

  python scripts/analysis/alfworld_rebuild_hints.py --out-dir outputs/alf_hint_render \\
      --source qwen37=trajectories_qwen37.jsonl,1k_qwen37.jsonl \\
      --source glm=trajectories_glm.jsonl,1k_glm.jsonl \\
      [--check-md5 glm__traj_withthink=<expected REBUILT_glm__traj_withthink.json> ...]

--check-md5 NAME=FILE compares the written REBUILT_<NAME>.json byte for byte with FILE (exit 1 on a
mismatch); --check-content NAME=FILE compares the parsed records (a re-serialised copy of the same hints,
e.g. the data bundle's, passes this but not the md5 check).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    """The repository root: the nearest ancestor of this file holding pyproject.toml, opsd/__init__.py and
    opsd/skyrl_ext/__init__.py (the predicate of opsd._repo)."""
    for d in Path(__file__).resolve().parents:
        if all((d / m).is_file() for m in ("pyproject.toml", "opsd/__init__.py", "opsd/skyrl_ext/__init__.py")):
            return d
    raise RuntimeError("no repository root (pyproject.toml + opsd/skyrl_ext/) above this script")


sys.path.insert(0, str(_repo_root()))
from opsd.pi.base import TrajView, extract_actions  # noqa: E402

D = "."
BIG = 10**9


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def s(xs):
    return f"n={len(xs)} avg={sum(xs)/len(xs):.0f} p50={pct(xs,.5)} p95={pct(xs,.95)} p99={pct(xs,.99)} max={max(xs)}"


# ---- rendering: unchanged from the build that produced the published hints --------------------------
def build_for(src, files):
    seen = {}
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("won"):
                continue
            tid = r.get("task_id")
            if tid and tid not in seen:
                seen[tid] = r
    wt = {"__schema__": "opsd_hints/v2"}
    nt = {"__schema__": "opsd_hints/v2"}
    ac = {"__schema__": "opsd_hints/v2"}
    wtc = []
    ntc = []
    acc = []
    align_bad = 0
    for tid, r in seen.items():
        msgs = r.get("messages") or []
        turns = r.get("turns") or []
        tv = TrajView(task_id=tid, messages=msgs, reward=float(r.get("reward", 0) or 0), success=True)
        nt_txt = tv.render(max_chars=BIG, strip_think=False)
        wt_msgs = []
        a = 0
        for m in msgs:
            if m.get("role") == "assistant":
                reasoning = (turns[a].get("reasoning") if a < len(turns) else "") or ""
                if a < len(turns) and str(turns[a].get("assistant", "")).strip() != str(m.get("content", "")).strip():
                    align_bad += 1
                c = str(m.get("content", ""))
                nc = f"<think>{reasoning.strip()}</think>\n{c}" if reasoning.strip() else c
                wt_msgs.append({**m, "content": nc})
                a += 1
            else:
                wt_msgs.append(m)
        wt_txt = TrajView(task_id=tid, messages=wt_msgs, reward=0, success=True).render(
            max_chars=BIG, strip_think=False
        )
        acts = extract_actions(msgs)
        ac_txt = "Reference actions (in order):\n" + "\n".join(f"{i+1}. {x}" for i, x in enumerate(acts))
        wt[tid] = {"task": wt_txt, "meta": {"pi_format": "traj_withthink"}}
        nt[tid] = {"task": nt_txt, "meta": {"pi_format": "traj_nothink"}}
        ac[tid] = {"task": ac_txt, "meta": {"pi_format": "actions", "n_actions": len(acts)}}
        wtc.append(len(wt_txt))
        ntc.append(len(nt_txt))
        acc.append(len(ac_txt))
    for name, obj in [("traj_withthink", wt), ("traj_nothink", nt), ("actions", ac)]:
        json.dump(obj, open(f"{D}/REBUILT_{src}__{name}.json", "w"))
    print(f"\n=== {src}: {len(seen)} unique won tasks (align mismatches: {align_bad}) ===")
    print(f"  withthink chars: {s(wtc)}")
    print(f"  nothink   chars: {s(ntc)}")
    print(f"  actions   chars: {s(acc)}")
    print(
        f"  withthink/nothink avg char ratio: {sum(wtc)/sum(ntc):.2f}x  (~token est /2.95: withthink avg {sum(wtc)/len(wtc)/2.95:.0f} tok, nothink {sum(ntc)/len(ntc)/2.95:.0f} tok)"
    )


# ---- parity checks ----------------------------------------------------------------------------------
def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(out_dir: Path, specs, content: bool) -> bool:
    ok = True
    for spec in specs:
        name, expected = spec.split("=", 1)
        got = out_dir / f"REBUILT_{name}.json"
        if content:
            a, b = json.load(open(got)), json.load(open(expected))
            same = a == b
            detail = f"{sum(a.get(k) == b.get(k) for k in set(a) | set(b))}/{len(set(a) | set(b))} keys equal"
        else:
            same = md5(got) == md5(expected)
            detail = f"md5 {md5(got)} vs {md5(expected)}"
        print(f"  [{'OK  ' if same else 'FAIL'}] {got.name} vs {expected}: {detail}")
        ok &= same
    return ok


def main() -> int:
    global D
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=FILE[,FILE...]",
        help="trajectory files of one source, in priority order (first won trajectory per task wins)",
    )
    ap.add_argument("--out-dir", default="outputs/alf_hint_render")
    ap.add_argument(
        "--check-md5",
        action="append",
        default=[],
        metavar="NAME=FILE",
        help="byte-compare REBUILT_<NAME>.json (e.g. glm__traj_withthink) with FILE",
    )
    ap.add_argument(
        "--check-content",
        action="append",
        default=[],
        metavar="NAME=FILE",
        help="compare the parsed records of REBUILT_<NAME>.json with FILE",
    )
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    D = str(out)
    for spec in a.source:
        src, files = spec.split("=", 1)
        build_for(src, [f for f in files.split(",") if f])
    print("\nDONE")
    ok_md5 = check(out, a.check_md5, content=False)
    ok_content = check(out, a.check_content, content=True)
    return 0 if ok_md5 and ok_content else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Strip the leaked ALFWorld system preamble from a trajectory-hint artifact.

WHY. The ALFWorld traj_* hints carry the scaffold's own instruction preamble -- the "You are an expert
agent solving tasks in the ALFRED household environment" header plus the full action vocabulary -- in
100% of entries, before the task goal. It is present in the base hints AND in the cap24k re-render, so
no arm run so far varies it, and it reaches OPSD's frozen teacher through OPSD_HINTS_PATH.

THE RULE. Cut to the first "Task: ". Verified to reproduce the existing hints_glm_stripped/ artifact
byte-for-byte on 2356/2356 entries, so *_strip arms built here are directly comparable to the base ones.

TWO DIFFERENT THINGS GET CONFLATED HERE, AND ONLY THE FIRST IS A LEAK.
  A. the SYSTEM PREAMBLE -- exactly one header block per hint, prepended at hint-construction time.
     That is scaffolding the student should not be handed. This script removes it.
  B. the ENV's invalid-action reply -- "[Your last action was not a valid command. At each turn, give
     exactly one command ... - look ...]" -- which re-prints the action vocabulary INSIDE the
     trajectory, 0-12 times, scaling with the reference agent's invalid actions. That is a real
     observation the reference agent received; deleting it would falsify the trajectory. This script
     LEAVES IT and reports the residual count, so the arm is honestly labelled.
  An earlier reading treated B's residue as evidence that the strip was only partial. It is not: the
  header-block count goes 1 -> 0 on every entry. B is trajectory content, not leaked scaffolding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HDR = "You are an expert agent solving tasks in the ALFRED household environment (TextWorld)."
ENVMSG = "give exactly one command"
CUT = "Task: "


def strip(text: str) -> str:
    i = text.find(CUT)
    return text[i:] if i >= 0 else text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allowlist", default=None, help="JSON list of task_ids to report coverage over")
    args = ap.parse_args()
    raw = json.loads(Path(args.src).read_text())
    out: dict = {"__schema__": raw.get("__schema__", "opsd_hints/v2")}
    n = kept = hdr_before = hdr_after = goal_after = envmsg_after = 0
    for k, v in raw.items():
        if str(k).startswith("__"):
            continue
        t = v.get("task") if isinstance(v, dict) else v
        if not isinstance(t, str) or not t.strip():
            continue
        n += 1
        hdr_before += t.count(HDR)
        s = strip(t)
        if not s.strip():
            continue  # never emit a blank hint: `if hint:` in the env skips it,
        kept += 1  # so a blank entry runs UNHINTED while passing --require-hint
        hdr_after += s.count(HDR)
        goal_after += 1 if "Your task is to:" in s or s.startswith(CUT) else 0
        envmsg_after += s.count(ENVMSG)
        meta = dict(v.get("meta") or {}) if isinstance(v, dict) else {}
        meta.update(
            {
                "stripped": "alfworld_system_preamble",
                "strip_rule": f"cut to first {CUT!r}",
                "source_artifact": str(args.src),
            }
        )
        out[str(k)] = {"task": s, "meta": meta}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    cov = ""
    if args.allowlist:
        ids = set(map(str, json.loads(Path(args.allowlist).read_text())))
        cov = f"  allowlist coverage {len(ids & set(out))}/{len(ids)}"
    print(
        f"{Path(args.src).name:38s} n={n:5d} kept={kept:5d}  HEADER {hdr_before}->{hdr_after}"
        f"  goal-present={goal_after}/{kept}  residual env-vocab msgs={envmsg_after}{cov}"
    )
    if hdr_after:
        print("FATAL: preamble survived the strip", file=sys.stderr)
        raise SystemExit(1)
    if kept != n:
        print(f"WARNING: {n-kept} entries became blank and were DROPPED", file=sys.stderr)


if __name__ == "__main__":
    main()

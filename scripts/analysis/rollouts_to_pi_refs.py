#!/usr/bin/env python
"""Turn SkyRL dumped eval rollouts into a `refs.jsonl` that `build_pi_hint_formats.py` can consume.

WHY. The PI hint builder expects won reference trajectories with a `messages` list -- the shape the
frontier (GLM / Qwen3.7) generation pipeline emits. A SkyRL rollout dump is a different shape: a flat
`input_prompt` string plus an `output_response` string, with the chat structure only present as
`<|im_start|>role ... <|im_end|>` markers. This bridges the two so a TRAINED POLICY'S OWN successful
trajectories can be used as a `self` PI source.

TRANSCRIPT RECONSTRUCTION. `input_prompt` ends with a dangling `<|im_start|>assistant\\n` header and
`output_response` begins with that turn's content, so the full conversation is exactly the
concatenation -- verified to parse with zero leftover text. We do NOT trust the split and re-derive
every message from the concatenated string.

WHICH TRAJECTORY. The builder keeps the FIRST won trajectory per task, so ordering here is the
selection policy. We emit the SHORTEST successful trajectory (fewest assistant turns), which is the
cleanest demonstration and avoids handing the student a hint padded with the wandering that a longer
success happened to contain. Pass --select longest/first to override.

  python scripts/analysis/rollouts_to_pi_refs.py \
      --dumps a.jsonl b.jsonl --out refs_self.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

MSG_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.S)


def parse_messages(row: dict) -> list[dict]:
    full = str(row.get("input_prompt") or "") + str(row.get("output_response") or "")
    return [{"role": r, "content": c} for r, c in MSG_RE.findall(full)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True, help="one or more dumped_evals *.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--select", choices=["shortest", "longest", "first"], default="shortest")
    args = ap.parse_args()

    best: dict[str, tuple] = {}
    n_rows = n_won = 0
    for d in args.dumps:
        for line in pathlib.Path(d).open():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1
            score = row.get("score")
            reward = float(sum(score)) if isinstance(score, list) else float(score or 0.0)
            if reward <= 0:
                continue
            n_won += 1
            msgs = parse_messages(row)
            if not msgs:
                continue
            tid = row["env_extras"]["extra_info"]["task_id"]
            n_turns = sum(1 for m in msgs if m["role"] == "assistant")
            key = n_turns if args.select == "shortest" else (-n_turns if args.select == "longest" else 0)
            if tid not in best or key < best[tid][0]:
                best[tid] = (
                    key,
                    {"task_id": tid, "messages": msgs, "reward": reward, "won": True, "num_turns": n_turns},
                )

    out = pathlib.Path(args.out)
    with out.open("w") as f:
        for tid in sorted(best):
            f.write(json.dumps(best[tid][1]) + "\n")

    turns = [v[1]["num_turns"] for v in best.values()]
    hist = collections.Counter(turns)
    print(f"rows read      : {n_rows:,}")
    print(f"won rollouts   : {n_won:,}")
    print(f"tasks with a win: {len(best):,}   (select={args.select})")
    print(f"turns: min {min(turns)}  median {sorted(turns)[len(turns)//2]}  max {max(turns)}")
    print(f"  most common turn counts: {hist.most_common(5)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Emit the summary.json that run_appworld_pi_utility.sh requires beside a no-hint baseline.

WHY THIS IS MANDATORY, NOT DECORATION. The utility driver refuses a MODEL_TAG'd baseline that has no
summary.json, and its own comment records why: a laxer check once let a 4B baseline pair with 30B
with-hint arms, because both are "Instruct" at the same sampling triple. That turns every Delta into a
cross-MODEL contrast (30B with-hint minus 4B no-hint), which reads as a large PI effect and is pure
model-size difference. The guard compares model_name exactly (modulo a trailing "-base") plus
temperature/top_p/top_k.

The values here are not asserted from memory. Each baseline was derived by
build_appworld_nohint_baselines_from_passk.sh from
    aw_passk128_transcripts/<split>_<model>.jsonl
the n=128 pass@k corpus generated with SAMPLING_PROFILE=onpolicy for THAT model, at max_model_len
49152 / MAX_TURNS 30 / repl_output_max_chars 20000 -- the same contract the with-hint arms run under, at
128 samples/task instead of 4.

    python write_appworld_baseline_provenance.py --root <OUTROOT of build_appworld_nohint_baselines_from_passk.sh>
"""

from __future__ import annotations

import argparse
import json
import os

# served-model-name as the utility driver will pass it via MODEL_NAME
MODEL_NAME = {
    "4bi": "Qwen3-4B-Instruct-2507",
    "4bt": "Qwen3-4B-Thinking-2507",
    "30bi": "Qwen3-30B-A3B-Instruct-2507",
}
# (temperature, top_p, top_k) per sampling profile, matching the pass@k corpus generation
# (scripts/appworld/passk.sh SAMPLING_PROFILE)
PROFILE = {"onpolicy": (1.0, 1.0, -1), "official": (0.7, 0.8, 20)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="baseline root (<profile>_<split>_<model>/trajectories.jsonl dirs)")
    ap.add_argument(
        "--corpus-name",
        default="aw_passk128_transcripts",
        help="name of the pass@k corpus dir recorded in derived_from",
    )
    ap.add_argument("--force", action="store_true", help="overwrite an existing summary.json")
    args = ap.parse_args()

    wrote = skipped = bad = 0
    for name in sorted(os.listdir(args.root)):
        d = os.path.join(args.root, name)
        traj = os.path.join(d, "trajectories.jsonl")
        if not os.path.isdir(d) or not os.path.isfile(traj):
            continue
        # dir layout is <profile>_<split>_<model>; model is the LAST segment so a split like
        # "train90" containing no underscore is irrelevant either way
        parts = name.split("_")
        if len(parts) < 3:
            print(f"  SKIP {name}: cannot parse <profile>_<split>_<model>")
            bad += 1
            continue
        profile, split, model = parts[0], "_".join(parts[1:-1]), parts[-1]
        if profile not in PROFILE or model not in MODEL_NAME:
            print(f"  SKIP {name}: unknown profile {profile!r} or model {model!r}")
            bad += 1
            continue
        out = os.path.join(d, "summary.json")
        if os.path.exists(out) and not args.force:
            skipped += 1
            continue
        t, p, k = PROFILE[profile]
        n = sum(1 for line in open(traj) if line.strip())
        json.dump(
            {
                "model_name": MODEL_NAME[model],
                "temperature": t,
                "top_p": p,
                "top_k": k,
                "sampling_profile": profile,
                "split": split,
                "rollouts": n,
                "n_samples_per_task": 128,
                "derived_from": f"{args.corpus_name}/{split}_{model}.jsonl",
                "note": (
                    "no-hint arm derived from the passk128 corpus for THIS model, not regenerated; "
                    "same sampler and context limits as the with-hint arms (max_model_len 49152, "
                    "MAX_TURNS 30, repl_output_max_chars 20000) at 128 samples/task instead of 4"
                ),
            },
            open(out, "w"),
            indent=2,
        )
        print(f"  {name:28s} -> {MODEL_NAME[model]} @ {t}/{p}/{k}  ({n} rollouts)")
        wrote += 1
    print(f"\nwrote {wrote}, already present {skipped}, unparsed {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

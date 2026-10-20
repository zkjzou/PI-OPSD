#!/usr/bin/env python
"""Inputs for the SELF-hint variance study: the GLM-8 question, with the student's own wins as the PI.

For every offline-analysis task where Qwen3-4B-Instruct won MORE than 4 of its 128 rollouts, 4 of those
winning rollouts become 4 raw self-hints (slots s0..s3), uncapped, rendered like every other raw hint
(role-tagged [OBSERVATION]/[AGENT] lines of the full transcript).

LEAKAGE. A hint rollout is dropped from the scored base for ALL 4 slots, so no slot scores a rollout
against its own copy and the slots stay paired on the same 124 rollouts. >4 wins guarantees at least one
won rollout is still scored.

Hint rollouts are drawn (seeded) from UNLABELLED winners first, so the published error-step labels, which
Δd⁻_err reads, keep their full coverage.

Writes, under --out:
  task_ids.json, hints/hints_self_s{0..3}.json, hints/manifest.json
  chunks/chunk_<k>.jsonl + chunk_<k>_index.json   scored rollouts by task chunk, with full-dump line index
"""

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alf_study_common as C  # noqa: E402
from alf_study_common import ACT_RE, parse_chat  # noqa: E402

DUMP = str(C.DUMP200_4BI)
LABELS = str(C.LABELS_4BI)
SCHEMA = "opsd_hints/v2"


def render_uncapped(messages: list) -> str:
    tag = {"user": "OBSERVATION", "assistant": "AGENT", "system": "SYSTEM"}
    return "\n".join(f"[{tag.get(m['role'], '?')}] {m['content']}" for m in messages if m["content"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-wins", type=int, default=5, help="keep tasks with at least this many wins")
    ap.add_argument("--max-wins", type=int, default=None, help="...and at most this many (default: no cap)")
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--nest-from",
        default=None,
        help="hints/manifest.json of a smaller design: its picks become slots 0..n-1 here, so the "
        "larger design's first n hints ARE the smaller design's hints",
    )
    ap.add_argument("--dump", default=DUMP, help="the N=128 no-hint student corpus on the 200 tasks")
    ap.add_argument("--labels", default=LABELS, help="published judge error-step labels")
    args = ap.parse_args()
    assert args.min_wins > args.slots, "need a won rollout left to score"
    prev = json.load(open(args.nest_from)) if args.nest_from else None

    lines = open(args.dump).read().splitlines()
    assert len(lines) == 25600, len(lines)
    labelled = set(json.load(open(args.labels)))
    recs = collections.defaultdict(list)  # task -> [(global k, rec)]
    for k, line in enumerate(lines):
        r = json.loads(line)
        recs[r["env_extras"]["extra_info"]["task_id"]].append((k, r))
    won = {t: [k for k, r in v if max(r["score"]) >= 10] for t, v in recs.items()}
    tasks = sorted(
        t for t in recs if len(won[t]) >= args.min_wins and (args.max_wins is None or len(won[t]) <= args.max_wins)
    )

    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "hints").mkdir(parents=True, exist_ok=True)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    files = [{"__schema__": SCHEMA} for _ in range(args.slots)]
    manifest, hint_ks = {}, {}
    for t in tasks:
        if prev is not None:
            base = [int(h["uid"].rsplit("#", 1)[1]) for h in prev[t]["hints"]]
            # per-task stream, so a pick does not depend on which other tasks are in the design
            rt = random.Random(f"{args.seed}:{t}")
            rest = sorted(
                (k for k in won[t] if k not in set(base)),
                key=lambda k: (f"{t}#{k}" in labelled, rt.random()),
            )
            pick = (base + rest)[: args.slots]
        else:
            keyed = sorted(won[t], key=lambda k: (f"{t}#{k}" in labelled, rng.random()))
            pick = keyed[: args.slots]
        assert len(pick) == args.slots, (t, len(pick))
        hint_ks[t] = set(pick)
        manifest[t] = {"wins": len(won[t]), "hints": []}
        for j, k in enumerate(pick):
            r = lines[k]
            rec = json.loads(r)
            msgs = parse_chat(rec["input_prompt"] + rec["output_response"])
            text = render_uncapped(msgs)
            acts = [a.strip() for m in msgs if m["role"] == "assistant" for a in ACT_RE.findall(m["content"])]
            files[j][t] = {"task": text, "meta": {"pi_format": "self_raw", "src_uid": f"{t}#{k}"}}
            manifest[t]["hints"].append(
                {
                    "slot": j,
                    "uid": f"{t}#{k}",
                    "labelled": f"{t}#{k}" in labelled,
                    "chars": len(text),
                    "turns": sum(m["role"] == "assistant" for m in msgs),
                    "has_reasoning": any(
                        re.sub(ACT_RE, "", m["content"]).strip() for m in msgs if m["role"] == "assistant"
                    ),
                    "actions": acts,
                }
            )
    for j, f in enumerate(files):
        (out / "hints" / f"hints_self_s{j}.json").write_text(json.dumps(f))
    (out / "hints" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "task_ids.json").write_text(json.dumps(tasks, indent=1))

    per_chunk = [tasks[i :: args.chunks] for i in range(args.chunks)]
    n_scored = 0
    for c, ts in enumerate(per_chunk):
        idx, ts = [], set(ts)
        with open(out / "chunks" / f"chunk_{c}.jsonl", "w") as f:
            for k, line in enumerate(lines):
                t = json.loads(line)["env_extras"]["extra_info"]["task_id"] if ts else None
                if t in ts and k not in hint_ks[t]:
                    f.write(line + "\n")
                    idx.append(k)
        (out / "chunks" / f"chunk_{c}_index.json").write_text(json.dumps(idx))
        n_scored += len(idx)
    assert n_scored == sum(128 - args.slots for _ in tasks), n_scored
    print(
        f"{len(tasks)} tasks with >= {args.min_wins} wins; {args.slots} self hints each; "
        f"{n_scored} rollouts to score in {args.chunks} chunks"
    )


if __name__ == "__main__":
    main()

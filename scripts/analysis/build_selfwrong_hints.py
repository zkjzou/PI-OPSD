#!/usr/bin/env python
"""Wrong-self PI for ALFWorld train1600: each task's hint is ONE of the student's own FAILED rollouts.

Student = untrained Qwen3-4B-Instruct-2507; the rollouts are the pass@k corpora (original env, plain prompt,
T 1.0 / top-p 1.0 / top-k -1, 30 turns): N=128 on the 200 offline-analysis tasks, N=32 on the other 1,400.
Format = raw, uncapped: the full transcript as role-tagged [OBSERVATION]/[AGENT] lines, the same render as
the self-hint variance study (pi_self4_prepare.render_uncapped).

Pick per task: a seeded draw among its failed rollouts, with an rng stream seeded per task so a pick does
not depend on which other tasks are present. On the 200 offline tasks, rollouts WITHOUT a published
error-step label are drawn first, so scoring this PI there (excluding the hint rollout) keeps full label
coverage for Δd⁻_err.

Tasks the student never failed have no wrong trajectory and are left OUT of the file. Train with
OPSD_NO_STATIC_FALLBACK=1 + OPSD_SKIP_NO_HINT=1 so they get no hint and leave the loss, instead of the
generic static hint.

    python scripts/analysis/build_selfwrong_hints.py --out outputs/pi_selfwrong

Inputs (defaults under $PI_CORPUS_ROOT / $OPSD_DATA_ROOT, see alf_study_common.py): the N=128 train200 corpus
(source "dump200"), the seven rem1400 N=32 shards sh00..sh06 (sources "sh00".."sh06", the shard files the
pass@k corpus generation wrote as exports/dumped_evals/eval_only/alfworld.jsonl), the train1600 split and
the published 4bi error-step labels.
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alf_study_common as C  # noqa: E402
from alf_study_common import parse_chat  # noqa: E402
from pi_self4_prepare import render_uncapped  # noqa: E402

DUMP200 = str(C.DUMP200_4BI)
SHARD = str(C.REM1400_SHARDS_4BI).replace("sh*", "sh{:02d}")
TRAIN1600 = str(C.TRAIN1600)
LABELS = str(C.LABELS_4BI)
SCHEMA = "opsd_hints/v2"


def lines_of(src: str):
    yield from open(src)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument(
        "--outcome",
        choices=("lost", "won"),
        default="lost",
        help="lost = wrong-self PI (hints_selfwrong_raw.json); won = correct-self PI " "(hints_selfcorrect_raw.json)",
    )
    ap.add_argument("--dump200", default=DUMP200, help="N=128 train200 no-hint corpus")
    ap.add_argument("--shard", default=SHARD, help="rem1400 N=32 shard path template ({:02d} = 00..06)")
    ap.add_argument("--task-ids", default=TRAIN1600, help="train1600 task ids")
    ap.add_argument("--labels", default=LABELS, help="published 4bi error-step labels")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "wrong" if args.outcome == "lost" else "correct"

    pool = json.load(open(args.task_ids))
    labelled = set(json.load(open(args.labels)))
    cands = collections.defaultdict(list)  # task -> [(source, k, rec)] with the wanted outcome
    n = collections.Counter()
    sources = [("dump200", args.dump200)] + [(f"sh{i:02d}", args.shard.format(i)) for i in range(7)]
    for name, src in sources:
        for k, line in enumerate(lines_of(src)):
            r = json.loads(line)
            t = r["env_extras"]["extra_info"]["task_id"]
            n[t] += 1
            if (max(r["score"]) >= 10) == (args.outcome == "won"):
                cands[t].append((name, k, r))
        print(f"{name}: {sum(1 for t in n)} tasks seen so far", flush=True)
    assert set(n) == set(pool), (len(set(n)), len(set(pool)))

    hints, manifest, none = {"__schema__": SCHEMA}, {}, []
    for t in sorted(pool):
        if not cands[t]:
            none.append(t)
            continue
        rt = random.Random(f"{args.seed}:{t}")
        name, k, r = sorted(cands[t], key=lambda x: (x[0] == "dump200" and f"{t}#{x[1]}" in labelled, rt.random()))[0]
        text = render_uncapped(parse_chat(r["input_prompt"] + r["output_response"]))
        hints[t] = {
            "task": text,
            "meta": {"pi_format": f"self_{tag}_raw", "source": name, "line": k},
        }
        manifest[t] = {
            "source": name,
            "line": k,
            "uid": f"{t}#{k}" if name == "dump200" else None,
            "n": n[t],
            "candidates": len(cands[t]),
            "labelled_pick": f"{t}#{k}" in labelled and name == "dump200",
            "chars": len(text),
        }
    no_key = "no_failed_rollout" if args.outcome == "lost" else "no_won_rollout"

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    lens = sorted(
        len(tok(v["task"], add_special_tokens=False).input_ids) for k, v in hints.items() if k != "__schema__"
    )
    stats = {
        "tasks": len(pool),
        "hinted": len(lens),
        no_key: len(none),
        "tokens_p50": lens[len(lens) // 2],
        "tokens_p95": lens[int(0.95 * len(lens))],
        "tokens_max": lens[-1],
        "labelled_picks_on_200": sum(m["labelled_pick"] for m in manifest.values()),
    }
    (out / f"hints_self{tag}_raw.json").write_text(json.dumps(hints))
    (out / "manifest.json").write_text(json.dumps({"stats": stats, no_key: none, "tasks": manifest}, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------------
# Derive the AppWorld no-hint utility baselines from the EXISTING passk128 corpora, for every
# (model, split) -- no GPU, no env rollouts, no baseline generation run.
#
# WHY THIS IS LEGITIMATE AND NOT A SHORTCUT. THE BASELINE IS NOT REGENERATED: the test_normal/train90/dev
# pass@k runs already are the no-hint arm on these tasks at this sampling profile, at higher n than 4 --
# a MORE precise avg@1 than an n=4 utility run gives. Verified rather than assumed: the passk128 runs
# used max_model_len 49152, MAX_TURNS=30, repl_output_max_chars=20000 and SAMPLING_PROFILE=onpolicy,
# which is exactly what the utility runs use (MAX_MODEL_LEN 49152 / MAX_TURNS 30 / onpolicy 1.0/1.0/-1).
# Same sampler, same context limits, 128 samples/task instead of 4.
#
# ⚠️ THE FIELD IS `reward`, NOT `success`. The baseline check in run_appworld_pi_utility_new6.sh does
# `float(r.get("success", 0.0) or 0.0)`, so handing it a raw passk128 transcript reports
# **avg@1 = 0.0000 and does NOT error** -- a silently-zero baseline against which every hint arm looks
# like a massive win. That is the whole reason this converter exists instead of a symlink.
#
# Output layout is per (profile, split, model) because the baseline IS model-specific -- differencing
# one student's hint arm against another student's no-hint corpus is meaningless.
#
# SRC holds the n=128 pass@k transcripts as <split>_<model>.jsonl, by default
# $APPWORLD_CORPUS_ROOT/aw_passk128_transcripts (APPWORLD_CORPUS_ROOT defaults to
# $OPSD_DATA_ROOT/appworld/corpus, the convention every AppWorld corpus script shares). Regenerate them with
# the pass@k launcher, whose defaults ARE the original corpora's settings (N=128, SAMPLING_PROFILE onpolicy
# 1.0/1.0/top-k -1, official ReAct prompt + scaffold, --strict-parse, --reward-mode binary, 30 turns, 50
# AppWorld interactions, 8192 tokens per turn, max_model_len 49152, observations cut to 20000 chars; 4bt
# with --preserve-thinking --enable-thinking; 30bi at TP2):
#   bash scripts/appworld/passk.sh <4bi|4bt|30bi> SPLITS="train90 dev normal"
# then place each $OUT_DIR/<split>/transcripts.jsonl at $SRC/<split>_<student>.jsonl. The baselines land in $OUTROOT/<profile>_<split>_<model>
# (default OUTROOT = $APPWORLD_CORPUS_ROOT, where run_appworld_utility_stage6.sh looks for them). Then run
# write_appworld_baseline_provenance.py --root $OUTROOT.
# ---------------------------------------------------------------------------------------------------
set -o pipefail
OPSD_DATA_ROOT=${OPSD_DATA_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data}
APPWORLD_CORPUS_ROOT=${APPWORLD_CORPUS_ROOT:-$OPSD_DATA_ROOT/appworld/corpus}   # regenerated no-hint corpora
SRC=${SRC:-$APPWORLD_CORPUS_ROOT/aw_passk128_transcripts}
OUTROOT=${OUTROOT:-$APPWORLD_CORPUS_ROOT}
PROFILE=${PROFILE:-onpolicy}
PY=${PY:-python}
MODELS=${MODELS:-"4bi 4bt 30bi"}
SPLITS=${SPLITS:-"train90 dev normal"}

for m in $MODELS; do
  for sp in $SPLITS; do
    in="$SRC/${sp}_${m}.jsonl"
    [ -s "$in" ] || in="$SRC/${sp}_${m}.jsonl.hold"     # may be parked by a stage hold
    if [ ! -s "$in" ]; then echo "  SKIP ${sp}/${m}: no corpus" >&2; continue; fi
    out="$OUTROOT/${PROFILE}_${sp}_${m}"
    mkdir -p "$out"
    "$PY" - "$in" "$out/trajectories.jsonl" <<'PY'
import json, sys, collections, math
src, dst = sys.argv[1], sys.argv[2]
per = collections.defaultdict(list)
n = 0
with open(dst, "w") as o:
    with open(src) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            tid = str(r.get("task_id"))
            rew = float(r.get("reward", 0.0) or 0.0)
            # `success` is the field the utility aggregator reads; keep reward too so the mapping is
            # auditable rather than a lossy rename.
            o.write(json.dumps({"task_id": tid, "success": rew, "reward": rew,
                                "sample": r.get("sample")}) + "\n")
            per[tid].append(rew)
            n += 1
def passk(v, k):
    N = len(v); c = sum(1 for x in v if x > 0)
    return 1.0 if N - c < k else 1.0 - math.comb(N - c, k) / math.comb(N, k)
avg1 = sum(sum(v)/len(v) for v in per.values())/len(per)
p4 = sum(passk(v, 4) for v in per.values())/len(per)
ns = sorted({len(v) for v in per.values()})
print(f"  {'/'.join(dst.split('/')[-2:]):32s} {len(per):4d} tasks  n/task {ns}  avg@1 {avg1:.4f}  pass@4 {p4:.4f}  ({n} rows)")
PY
  done
done

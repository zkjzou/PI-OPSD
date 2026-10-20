#!/usr/bin/env bash
set -uo pipefail
# =====================================================================================================
# AppWorld PI ATTRIBUTE analysis (#2): per-token divergence between the student's own distribution and
# its hint-conditioned one, at every student position, labelled by agentic STEP and CHANNEL.
#
# For each no-hint base rollout and each PI, two forwards over the SAME student tokens give
#   kl / klf   reverse+forward KL      d / dsigned  |Δlogp| and signed     dH  entropy change
#   o{k}       top-k overlap           tkkl{k}      top-k(union) KL
# then pi_attr_rollup.py token-weight-averages those to per-(PI, rollout) scalars and reports, per PI,
# each attribute's mean plus its won/lost split and its AUC / spearman / cohen_d / best@k AGAINST
# SUCCESS -- which is the attribute<->success correlation this analysis is for.
#
# INJECT_STYLE=system-prepend, matching AppWorld OPSD training (SelfDistillSkyRLGymGenerator's
# SECRET_TEMPLATE), so an attribute AUC, a ranking AUC and a utility Δ all describe the SAME
# intervention and can legitimately be correlated.
#
# CHANNEL SPLIT IS DEGENERATE ON APPWORLD. analyze_pi_positions._TAGS is <think>/<action> (ALFWorld
# Qwen grammar); AppWorld uses ```python fences and Instruct-2507 emits no <think>, so every token
# lands in channel=other unless --benchmark appworld is passed (it is, below). Pooling the channels
# keeps the attribute MEANS right either way.
#
# ALL SIX PI variants are looped INSIDE each worker, because hidden_base is cached per trajectory and
# reused across PIs -- running one PI per invocation would recompute the expensive base forward 6x.
#
# MAX_SEQ_LEN matters here: raw hints average ~13.7k chars (~3.5k tokens) on top of a ~3.1k-token prompt
# and ~3.1k-token response. The default 40000 is fine, but a trajectory whose hinted sequence exceeds it
# is SKIPPED, and skips that differ per PI would silently compare formats on different populations --
# so the skip count is logged per shard and must be checked before reading the table.
#
# DUMP is the student's no-hint corpus (transcripts.jsonl) for the split, by default
# $APPWORLD_CORPUS_ROOT/<profile>_<split>[_<MODEL_TAG>]/ (APPWORLD_CORPUS_ROOT defaults to
# $OPSD_DATA_ROOT/appworld/corpus). The paper's trajectory cells (appworld_<split>_<student>[_thinkfix]) used
# the n=128 pass@k corpus: pass DUMP=<scripts/appworld/passk.sh OUT_DIR>/<split>/transcripts.jsonl, or the
# shipped corpus $OPSD_DATA_ROOT/appworld/plain/opsd_appworld_corpora_<split>/<student>/transcripts.jsonl.
#
#   PROFILE=onpolicy SPLIT=train90 GPUS=4,5,6 DUMP=<transcripts.jsonl> bash scripts/analysis/run_appworld_pi_positions.sh
# =====================================================================================================
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$WT" != / ] && ! { [ -f "$WT/pyproject.toml" ] && [ -f "$WT/opsd/__init__.py" ] && [ -f "$WT/opsd/skyrl_ext/__init__.py" ]; }; do WT="$(dirname "$WT")"; done
[ -f "$WT/opsd/skyrl_ext/__init__.py" ] || { echo "FATAL: cannot locate the repo root above $0" >&2; exit 1; }
cd "$WT"
export PYTHONPATH="$WT"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
PY=${PY:-python}
WORK=${WORK:-$WT/outputs/appworld_pi}
OPSD_DATA_ROOT=${OPSD_DATA_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data}
AW_DIR=${AW_DIR:-$OPSD_DATA_ROOT/appworld/plain/opsd_appworld_pi}   # unpacked AppWorld bundle
APPWORLD_CORPUS_ROOT=${APPWORLD_CORPUS_ROOT:-$OPSD_DATA_ROOT/appworld/corpus}   # regenerated no-hint corpora

PROFILE=${PROFILE:-onpolicy}
SPLIT=${SPLIT:-normal}
SOURCES=${SOURCES:-"glm52 qwen37"}
FORMATS=${FORMATS:-"raw raw_no_think actions"}
GPUS=${GPUS:-4,5,6}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}   # HF id or a local snapshot dir
# Suffix for every corpus/output root. Empty default => existing Instruct paths are byte-identical.
MODEL_TAG=${MODEL_TAG:-}
SUF=${MODEL_TAG:+_$MODEL_TAG}
case "$(basename "$MODEL")" in
  # EXACT basename, NOT *Instruct-2507*. That glob also matches Qwen3-30B-A3B-Instruct-2507, so a 30B run
  # with an empty MODEL_TAG sailed straight past this guard and overwrote the 4B positions root in place.
  # Only the 4B owns the untagged roots.
  Qwen3-4B-Instruct-2507) : ;;
  *) [ -n "$MODEL_TAG" ] || { echo "FATAL: MODEL=$MODEL is not the default Instruct model and MODEL_TAG is empty -- this would overwrite the Instruct positions root. Set MODEL_TAG." >&2; exit 1; } ;;
esac
DUMP=${DUMP:-$APPWORLD_CORPUS_ROOT/${PROFILE}_${SPLIT}${SUF}/transcripts.jsonl}
OUT=${OUT:-$WORK/positions/${PROFILE}_${SPLIT}${SUF}}
# system-prepend is what AppWorld OPSD training does: main_opsd_appworld -> SelfDistillExp ->
# SelfDistillSkyRLGymGenerator, which prepends SECRET_TEMPLATE.
INJECT_STYLE=${INJECT_STYLE:-system-prepend}
KS=${KS:-"10,20,50,100"}
MAX_STEPS=${MAX_STEPS:-50}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-65536}      # no-truncation rule: hints built uncapped, scored at 64k,
                                       # scorer SKIPS+logs >64k
LM_HEAD_CHUNK=${LM_HEAD_CHUNK:-128}
LIMIT=${LIMIT:-}
SAVE_TOKENS=${SAVE_TOKENS:-1}   # runbook: raw per-token dumps -> tokens.parquet, GPU-free re-aggregation
# DEFAULT appworld, and it must be FORWARDED -- this is an AppWorld-only driver, but the scorer's own
# --benchmark default is `alfworld` (kept so ALFWorld output stays bit-identical to pre---benchmark
# runs). Until now this driver never passed the flag at all, so every AppWorld positions cell it ever
# produced was labelled with ALFWorld's <think>/<action> grammar. AppWorld emits fenced Python, so
# 100% of tokens fell in channel="other" and the channel dimension silently collapsed to one bucket --
# measured on the 2026-08-25 smoke: `other 5019` of 5019 rows. Not a scorer bug; a missing argument.
BENCHMARK=${BENCHMARK:-appworld}

[ -f "$DUMP" ] || { echo "FATAL: no dump at $DUMP" >&2; exit 1; }
mkdir -p "$OUT"

# --pis takes a comma list of name=hints_path; name it <source>_<format> so the roll-up separates the
# SOURCE axis from the FORMAT axis. Those two axes behaved OPPOSITELY on ALFWorld (utility
# anti-correlated with OPSD across formats, r=-0.85, but positively across analyzer quality, r=+0.998),
# so they must never be pooled into one "PI" factor.
#
# HINT LAYOUT DIFFERS BY SPLIT. The eval splits live under hints225/<source>_<split>/ (built for dev-57 +
# test_normal-168 by build_appworld_pi_dev57.sh), but train-90 -- the split the OPSD runs actually
# distilled on -- is the bundle's hints/trajectory/<author>/. Getting this wrong yields "injected into
# 0/N", not an error.
_author() { case "$1" in glm52) echo glm-5.2 ;; qwen37) echo qwen3.7-plus ;; *) echo "$1" ;; esac; }
hint_root() {  # source -> dir holding hints_<format>.json
  case "$SPLIT" in
    train90) echo "${HINTS_TRAIN90_ROOT:-$AW_DIR/hints/trajectory}/$(_author "$1")" ;;
    *)       echo "${HINTS225:-$WORK/hints225}/${1}_${SPLIT}" ;;
  esac
}
PIS=""
for src in $SOURCES; do
  for fmt in $FORMATS; do
    hp="$(hint_root "$src")/hints_${fmt}.json"
    if [ -f "$hp" ]; then PIS="${PIS:+$PIS,}${src}_${fmt}=$hp"
    else echo "  WARN: missing $hp -- ${src}_${fmt} excluded"; fi
  done
done
[ -n "$PIS" ] || { echo "FATAL: no hint artifacts found for split=$SPLIT" >&2; exit 1; }
echo "=== positions $PROFILE/$SPLIT | $(wc -l < "$DUMP") rollouts | inject=$INJECT_STYLE"
echo "    PIs: $(echo "$PIS" | tr ',' '\n' | cut -d= -f1 | tr '\n' ' ')"

IFS=',' read -ra GA <<< "$GPUS"
NSH=${#GA[@]}
pids=()
for i in "${!GA[@]}"; do
  g=${GA[$i]}
  CUDA_VISIBLE_DEVICES="$g" "$PY" \
    opsd/alfworld/analyze_pi_positions.py \
      --appworld-transcripts "$DUMP" --pis "$PIS" --benchmark "$BENCHMARK" \
      --model-path "$MODEL" --device cuda:0 \
      --inject-style "$INJECT_STYLE" --ks "$KS" --max-steps "$MAX_STEPS" \
      --max-seq-len "$MAX_SEQ_LEN" --lm-head-chunk "$LM_HEAD_CHUNK" \
      --num-shards "$NSH" --shard-index "$i" ${LIMIT:+--limit $LIMIT} \
      ${SAVE_TOKENS:+--save-token-logprobs} \
      --out-dir "$OUT" > "$OUT/shard${i}_gpu${g}.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ "$rc" = 0 ] || echo "WARN: a shard exited non-zero (see $OUT/shard*.log)"
grep -h "skip" "$OUT"/shard*.log 2>/dev/null | tail -3

"$PY" opsd/alfworld/analyze_pi_positions.py \
  --merge --out-dir "$OUT" 2>&1 | tail -8

# ---- optional: roll up to per-PI attribute means + AUC/spearman/cohen vs success (attribute study;
# pi_attr_rollup.py is not released, so this is skipped in a release checkout) ----
if [ -f scripts/analysis/pi_attr_rollup.py ]; then
  "$PY" scripts/analysis/pi_attr_rollup.py \
    --split "${SPLIT}=$OUT/rows.parquet" --channel all \
    --out "$OUT/pi_attr_rollup.json" 2>&1 | tail -15 || \
    echo "NOTE: check pi_attr_rollup.py's exact flags if this failed"
fi
echo "=== done -> $OUT"

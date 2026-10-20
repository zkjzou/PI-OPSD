#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------------
# Regenerate ALL AppWorld analyzer (onpolicy_skill) hint cells with the FIXED task extraction.
# 5 summarisers x 3 splits = 15 cells, 1530 hints. Input is glm-5.2 RAW PI trajectories throughout,
# so the summariser is the only thing that varies within a split.
#
# THE BUG THIS FIXES. AppWorld's official scaffold puts a 10-exchange worked DEMO in conv[0], and the
# only "Task:" in it is the demo's ("How many playlists..."). Taking the FIRST "Task:" therefore
# injected one constant task into every episode's analyzer prompt -- 1 distinct text for 90 tasks --
# and the analyzer dutifully wrote hints for a task the trajectory was not solving. The extraction now
# takes the LAST "Task:" in conv[:generated_span], and the generator refuses to run if any single task
# text covers >25% of the map.
#
# THE FIRST GUARD WAS A RATIO TEST (>=90% distinct) AND IT FALSE-POSITIVED. AppWorld ships task
# VARIANTS -- genuinely repeated prompts across task ids -- so normal-168 has 148 distinct texts
# (88.1%) when perfectly healthy, and all five `normal` cells died on it. Mode share is the right test:
# healthy is 1.8% (normal) / 3.3% (train90) / 5.3% (dev), nowhere near 25%.
#
# INPUTS per split (a split whose input is absent is skipped):
#   IN_<SPLIT>  glm-5.2 raw trajectory hints; train90 defaults to the dataset bundle
#               (hints/trajectory/glm-5.2/hints_raw.json), dev/normal come from build_appworld_pi_dev57.sh
#               (or your own test_normal build).
#   CM_<SPLIT>  a student rollout corpus (transcripts.jsonl) of that split, used only to recover each
#               task's statement (--task-map-corpus); default the 4bi n=128 pass@k transcripts under
#               CORPORA = $APPWORLD_CORPUS_ROOT/aw_passk128_transcripts (APPWORLD_CORPUS_ROOT defaults to
#               $OPSD_DATA_ROOT/appworld/corpus).
# Remote cells use ANALYZER_BASE_URL / ANALYZER_KEY_ENV / GLM_MODEL / QWEN_MODEL as in
# gen_alfworld_analyzer_hints.sh; local cells need the three students served on PORT_4BI/4BT/30BI.
#
# Cells resume: a cell already holding its hints reports `skipped_existing` and costs nothing, so this
# is safe to re-run after a partial failure.
#
#   bash scripts/analysis/regen_appworld_analyzer_hints.sh    # needs the 3 local vLLM servers up
# ---------------------------------------------------------------------------------------------------
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$REPO" != / ] && ! { [ -f "$REPO/pyproject.toml" ] && [ -f "$REPO/opsd/__init__.py" ] && [ -f "$REPO/opsd/skyrl_ext/__init__.py" ]; }; do REPO="$(dirname "$REPO")"; done
[ -f "$REPO/opsd/skyrl_ext/__init__.py" ] || { echo "FATAL: cannot locate the repo root above $0" >&2; exit 1; }
cd "$REPO" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}
OPSD_DATA_ROOT=${OPSD_DATA_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data}
AW_DIR=${AW_DIR:-$OPSD_DATA_ROOT/appworld/plain/opsd_appworld_pi}   # unpacked AppWorld bundle
APPWORLD_CORPUS_ROOT=${APPWORLD_CORPUS_ROOT:-$OPSD_DATA_ROOT/appworld/corpus}   # regenerated no-hint corpora
CORPORA=${CORPORA:-$APPWORLD_CORPUS_ROOT/aw_passk128_transcripts}
O=${O:-./outputs/aw_hints/hints_onpolicy}
SPLITS=${SPLITS:-"normal train90 dev"}
ANALYZER_BASE_URL=${ANALYZER_BASE_URL:-${OPSD_PI_ANALYZER_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}}
ANALYZER_KEY_ENV=${ANALYZER_KEY_ENV:-${OPSD_PI_ANALYZER_API_KEY_ENV:-OPENAI_API_KEY}}
GLM_MODEL=${GLM_MODEL:-fireworks_ai/glm-5p2}
QWEN_MODEL=${QWEN_MODEL:-fireworks_ai/qwen3p7-plus}
PORT_4BI=${PORT_4BI:-8630}; PORT_4BT=${PORT_4BT:-8631}; PORT_30BI=${PORT_30BI:-8632}
[ -n "${!ANALYZER_KEY_ENV:-}" ] || { echo "FATAL: set $ANALYZER_KEY_ENV (API key for $ANALYZER_BASE_URL)" >&2; exit 1; }
mkdir -p "$O"
# split -> (input artifact, task-map corpus)
declare -A IN=( [normal]="${IN_NORMAL:-}" \
                [train90]="${IN_TRAIN90:-$AW_DIR/hints/trajectory/glm-5.2/hints_raw.json}" \
                [dev]="${IN_DEV:-./outputs/appworld_pi/hints225/glm52_dev/hints_raw.json}" )
declare -A CM=( [normal]="${CM_NORMAL:-}" \
                [train90]="${CM_TRAIN90:-$CORPORA/train90_4bi.jsonl}" \
                [dev]="${CM_DEV:-$CORPORA/dev_4bi.jsonl}" )
run() {  # tag where model max_tokens split   (where = remote | local server root URL)
  local tag=$1 where=$2 model=$3 mt=$4 split=$5
  local out="$O/${tag}_from_glm52"; [ "$split" = normal ] || out="${out}_${split}"
  local endpoint=(--url "$where")
  [ "$where" = remote ] && endpoint=(--base-url "$ANALYZER_BASE_URL" --api-key-env "$ANALYZER_KEY_ENV")
  echo "######## OSK $tag [$split]"
  "$PY" scripts/analysis/gen_hints_local_models.py \
    --input-hints "${IN[$split]}" --task-map-corpus "${CM[$split]}" \
    --prompt-style onpolicy_skill "${endpoint[@]}" --model "$model" \
    --out "${out}.json" \
    --max-chars 140000 --max-tokens "$mt" --concurrency 8 --timeout 900 \
    || echo "CELL FAILED $tag/$split"
}
for split in $SPLITS; do
  if [ ! -s "${IN[$split]}" ] || [ ! -s "${CM[$split]}" ]; then
    echo "SKIP $split: missing input '${IN[$split]}' or task-map corpus '${CM[$split]}' (set IN_/CM_${split^^})" >&2
    continue
  fi
  # remote first: no GPU, runs while the local servers finish warming
  run glm52  remote                       "$GLM_MODEL"                8192  "$split"
  run qwen37 remote                       "$QWEN_MODEL"               8192  "$split"
  run 4bi    "http://127.0.0.1:$PORT_4BI"  Qwen3-4B-Instruct-2507      3072  "$split"
  run 4bt    "http://127.0.0.1:$PORT_4BT"  Qwen3-4B-Thinking-2507      16384 "$split"
  run 30bi   "http://127.0.0.1:$PORT_30BI" Qwen3-30B-A3B-Instruct-2507 3072  "$split"
done
echo "REGEN_ANALYZER_DONE"

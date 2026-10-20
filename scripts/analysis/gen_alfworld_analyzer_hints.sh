#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------------
# Build the analyzer/summariser-hint grid on ALFWORLD: 5 summarisers x 2 prompt styles = 10 cells.
#
# Input is the glm-5.2 RAW reference trajectories (hints/glm-5.2/hints_raw.json), so the summariser is
# the only thing that varies within a style -- the same design as the AppWorld set in
# regen_appworld_analyzer_hints.sh. The published cells were built over a 1,623-task allowlist (the
# 1600-task train list UNIONED with an earlier 200-task ranking pool); the self cells were then closed to
# train1600 by fill_self_hints.sh. The default allowlist here is the released train1600 split.
#
# ALFWORLD IS THE EASY DIRECTION. Its goal is IN the trajectory ("Your task is to: <goal>" in 2301 of
# 2356, "Task: <goal>" in all of them, modal goal 0.9% of tasks), so --task-map-corpus is not needed
# and the AppWorld scaffold-demo failure cannot occur. Two things still MUST be passed:
#
#   --env alfworld     picks the ALFWorld analyzer prompt AND the in-trajectory goal extraction.
#   --outcome success  the reference set carries NO terminal marker (0/2356 match any of 7 candidate
#                      markers), so the default `auto` heuristic -- which looks for AppWorld's
#                      complete_task -- would label every one of a corpus of WINS a FAILURE and send
#                      the analyzer hunting mistakes that are not there. Silent: coverage still 100%.
#
# Remote cells (glm52, qwen37) call an OpenAI-compatible endpoint synchronously:
#   ANALYZER_BASE_URL  default $OPSD_PI_ANALYZER_BASE_URL, else $OPENAI_BASE_URL, else api.openai.com
#   ANALYZER_KEY_ENV   NAME of the key variable, default $OPSD_PI_ANALYZER_API_KEY_ENV, else OPENAI_API_KEY
#   GLM_MODEL / QWEN_MODEL  the model ids at that endpoint (defaults are the LiteLLM provider strings
#                      the published cells used; GLM-5.2 / Qwen3.7-Plus may no longer be served).
# Local cells (4bi, 4bt, 30bi) need the student served by vLLM on PORT_4BI / PORT_4BT / PORT_30BI.
#
# Cells resume (`skipped_existing`), so re-running after a partial failure is free.
#
#   bash scripts/analysis/gen_alfworld_analyzer_hints.sh              # all 10 cells
#   MODELS=4bi STYLES=plain bash .../gen_alfworld_analyzer_hints.sh      # one cell
#   LIMIT=8 bash .../gen_alfworld_analyzer_hints.sh                      # smoke
# ---------------------------------------------------------------------------------------------------
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$REPO" != / ] && ! { [ -f "$REPO/pyproject.toml" ] && [ -f "$REPO/opsd/__init__.py" ] && [ -f "$REPO/opsd/skyrl_ext/__init__.py" ]; }; do REPO="$(dirname "$REPO")"; done
[ -f "$REPO/opsd/skyrl_ext/__init__.py" ] || { echo "FATAL: cannot locate the repo root above $0" >&2; exit 1; }
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}
OPSD_DATA_ROOT=${OPSD_DATA_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data}
ART=${ART:-$OPSD_DATA_ROOT/alfworld/hints/glm-5.2/hints_raw.json}
ALLOW=${ALLOW:-$OPSD_DATA_ROOT/alfworld/splits/task_ids_train1600.json}
OUT=${OUT:-./outputs/alf_hints1600}
MODELS=${MODELS:-"glm52 qwen37 4bi 4bt 30bi"}
STYLES=${STYLES:-"onpolicy_skill plain"}
LIMIT=${LIMIT:-0}
ANALYZER_BASE_URL=${ANALYZER_BASE_URL:-${OPSD_PI_ANALYZER_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}}
ANALYZER_KEY_ENV=${ANALYZER_KEY_ENV:-${OPSD_PI_ANALYZER_API_KEY_ENV:-OPENAI_API_KEY}}
GLM_MODEL=${GLM_MODEL:-fireworks_ai/glm-5p2}
QWEN_MODEL=${QWEN_MODEL:-fireworks_ai/qwen3p7-plus}
PORT_4BI=${PORT_4BI:-8630}; PORT_4BT=${PORT_4BT:-8631}; PORT_30BI=${PORT_30BI:-8632}

[ -f "$ART" ]   || { echo "FATAL: input artifact $ART missing" >&2; exit 1; }
[ -f "$ALLOW" ] || { echo "FATAL: allowlist $ALLOW missing" >&2; exit 1; }
case " $MODELS " in
  *" glm52 "*|*" qwen37 "*)
    [ -n "${!ANALYZER_KEY_ENV:-}" ] || { echo "FATAL: set $ANALYZER_KEY_ENV (API key for $ANALYZER_BASE_URL)" >&2; exit 1; } ;;
esac

# REMOTE CELLS NEED MORE RETRIES THAN THE DEFAULT 4. At concurrency 24 the remote endpoint returned
# occasional non-200s: qwen37 lost 50 hints on one cell (96.9%) and 92 on another (94.3%, below the 95%
# coverage guard, so that cell reported CELL FAILED). Every one of them succeeded on a resume -- they are
# transient, not content-related. --retries 8 plus a gentler concurrency keeps a cell from tripping
# the guard; and because cells resume, a short cell only ever costs the missing hints.

# tag -> where model max_tokens concurrency   (where = remote | a local server root URL)
# max_tokens: 4bt needs a large budget because it spends ~90% of it on reasoning that is then
# stripped (AppWorld-measured: 1465 completion tokens per hint vs 171 for the non-thinking 4B).
# concurrency: the local servers run --max-num-seqs 8, so >8 only queues; the remote endpoint takes more.
spec() {
  case "$1" in
    glm52)  echo "remote $GLM_MODEL  8192 12" ;;
    qwen37) echo "remote $QWEN_MODEL 8192 12" ;;
    4bi)    echo "http://127.0.0.1:$PORT_4BI Qwen3-4B-Instruct-2507      3072  8" ;;
    4bt)    echo "http://127.0.0.1:$PORT_4BT Qwen3-4B-Thinking-2507      16384 8" ;;
    30bi)   echo "http://127.0.0.1:$PORT_30BI Qwen3-30B-A3B-Instruct-2507 3072  8" ;;
    *) echo "unknown summariser tag $1" >&2; exit 1 ;;
  esac
}

for style in $STYLES; do
  sub=$([ "$style" = plain ] && echo hints_summarized || echo hints_onpolicy)
  mkdir -p "$OUT/$sub"
  for tag in $MODELS; do
    read -r where model mt conc <<<"$(spec "$tag")"
    if [ "$where" = remote ]; then
      endpoint=(--base-url "$ANALYZER_BASE_URL" --api-key-env "$ANALYZER_KEY_ENV")
    else
      endpoint=(--url "$where")
    fi
    echo "######## $tag [$style]  ($model, max_tokens=$mt, conc=$conc)"
    # shellcheck disable=SC2086
    $PY scripts/analysis/gen_hints_local_models.py \
      --env alfworld --outcome success --out-schema opsd_v2 \
      --input-hints "$ART" --task-allowlist "$ALLOW" --prompt-style "$style" \
      "${endpoint[@]}" --model "$model" \
      --out "$OUT/$sub/${tag}_from_glm.json" \
      --max-chars 24000 --max-tokens "$mt" --concurrency "$conc" --timeout 900 --retries 8 \
      ${LIMIT:+--limit $LIMIT} \
      || echo "CELL FAILED $tag/$style"
  done
done
echo "GEN_ALFWORLD_HINTS_DONE"

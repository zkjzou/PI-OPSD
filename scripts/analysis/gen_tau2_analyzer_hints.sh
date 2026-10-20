#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------------
# Generate the tau2 hint arms: anlz_* (onpolicy_skill, STRICT JSON) and summ_* (plain free text),
# for each summariser tag. tau2 previously had ONLY the 6 trajectory PIs.
#
# NO GPU for the remote tags. That matters on a contended box: CPU-only work survives there even when
# every GPU-touching process is being SIGKILLed, so the glm52/qwen37 arms run locally and only the
# `self` arm needs a served student.
#
# tau2 SPECIFICS
#   --env tau2      selects the tau2 analyzer prompt AND the task extractor (first [OBSERVATION];
#                   the agent opens with a greeting, so conversation[0] is NOT the request). This is
#                   now strict: an unknown --env raises instead of silently using ALFWorld wording.
#   --outcome success   the refs are VERIFIED WINS (measured reward 1.0 / won 1.0). `auto` is the
#                   AppWorld "complete_task present" heuristic and is meaningless here -- it would
#                   label every win a FAILURE and send the analyzer hunting absent mistakes.
#   no --task-map-corpus   tau2 is self-contained, like ALFWorld.
#   --max-chars 140000     tau2 `raw` refs are p50 ~19.6k chars, 5-7x ALFWorld's; do not elide them.
#   --max-tokens 8192      a small budget yields ZERO hints on long inputs (reasoning eats it and the
#                   JSON is cut mid-object).
#
# ENDPOINTS. The remote authors go through any OpenAI-compatible chat endpoint: LLM_BASE_URL (default
# $OPENAI_BASE_URL, else https://api.openai.com/v1) with the key in the env var named by LLM_KEY_ENV
# (default OPENAI_API_KEY; passed to the generator by NAME, never on the command line). GLM_MODEL /
# QWN_MODEL are the two authors as that endpoint names them (both are no longer served by their
# provider; the built artifacts are published under hints/derived and hints/t211/analyzer_rb).
# Input default: the published glm-5.2 pool424 hints_raw.json.
# ---------------------------------------------------------------------------------------------------
set -o pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_tau2_env.sh"
cd "$REPO_ROOT" || exit 1
IN=${IN:-$T2/hints/glm-5.2/pool424/hints_raw.json}
O=${O:-$OUT_ROOT/pi_tau2_hints}
LIMIT=${LIMIT:-0}
CONC=${CONC:-8}
GLM_MODEL=${GLM_MODEL:-fireworks_ai/glm-5p2}
QWN_MODEL=${QWN_MODEL:-fireworks_ai/qwen3p7-plus}
require_llm_key
[ -s "$IN" ] || { echo "FATAL: missing input artifact $IN" >&2; exit 1; }
mkdir -p "$O"
# The generator appends /v1/chat/completions to --url, so it takes the endpoint WITHOUT the /v1 suffix.
LL=${LLM_BASE_URL%/}; LL=${LL%/v1}

run() {  # style tag url model max_tokens [api-key env NAME]
  local style=$1 tag=$2 url=$3 model=$4 mt=$5 keyenv=$6
  local pre; [ "$style" = onpolicy_skill ] && pre=anlz || pre=summ
  local out="$O/${pre}_${tag}.json"
  echo "######## tau2 $pre/$tag -> $out"
  "$PY" scripts/analysis/gen_hints_local_models.py \
    --input-hints "$IN" --env tau2 --outcome success --out-schema opsd_v2 \
    --prompt-style "$style" --url "$url" --model "$model" \
    ${keyenv:+--api-key-env "$keyenv"} --out "$out" \
    --max-chars 140000 --max-tokens "$mt" --concurrency "$CONC" --timeout 900 \
    ${LIMIT:+--limit $LIMIT} \
    || echo "ARM FAILED $pre/$tag"
}

# remote first: no GPU needed, so these complete even on a box where GPU procs are being killed
run onpolicy_skill glm  "$LL" "$GLM_MODEL" 8192 "$LLM_KEY_ENV"
run plain          glm  "$LL" "$GLM_MODEL" 8192 "$LLM_KEY_ENV"
run onpolicy_skill qwn  "$LL" "$QWN_MODEL" 8192 "$LLM_KEY_ENV"
run plain          qwn  "$LL" "$QWN_MODEL" 8192 "$LLM_KEY_ENV"
# the `self` arms need a served student model (GPU): set SELF_URL (e.g. http://127.0.0.1:8350) + SELF_MODEL.
[ -n "$SELF_URL" ] && {
  run onpolicy_skill self "$SELF_URL" "${SELF_MODEL:?set SELF_MODEL}" 3072
  run plain          self "$SELF_URL" "${SELF_MODEL:?set SELF_MODEL}" 3072
}
echo "TAU2_ANALYZER_HINTS_DONE"

#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------------
# Frontier PI trajectories for ALFWorld train1600: ONE trajectory per task per model, under the OPSD
# TRAINING contract, default reasoning effort, every token kept.
#
# Contract = the resolved config of the 4B-Instruct OPSD arms (e.g. alfworld/passcode/4bi/pc_rawkey with
# passcode off):
#   env      react prompt, reasoning_style=prefix "THOUGHTS:", think_tags_in_prompt, no admissible list,
#            invalid-action feedback WITHOUT the admissible list, history 0, 30 turns, reward_scale 0.1
#   sampling T 1.0, top-p 1.0, top-k 20, min-p 0, 8000 tokens/turn, NO stop strings, 32000 input tokens
# Kept per turn: content, reasoning_content, usage (prompt/cached/completion/reasoning), finish_reason.
#
# Models run in parallel; each model's chunks run in sequence, each on a FRESH ALFWorld server pool
# (every /reset pins a 35 MB deleted .so until the server exits). Spend is metered from the returned token
# usage x the per-model prices below: each model has a cap carried across its chunks (the driver's
# --budget-usd stops STARTING trajectories at it), so the run as a whole stays under the sum of the caps.
# `touch $OUT/STOP` stops every model before its next chunk. A model the endpoint does not serve (GLM-5.2
# and Qwen3.7-Plus are no longer served; their trajectories are published) is re-probed every 30 min for
# WAIT_H hours before it starts (each probe's HTTP code is logged; a 401/403 skips the model at once, and a
# provider-routed model id with no BASE_URL / OPENAI_BASE_URL is a fatal error before anything starts).
#
# ENDPOINT. Any OpenAI-compatible /chat/completions endpoint: BASE_URL (default $OPENAI_BASE_URL, else
# https://api.openai.com/v1) and the key in the env var named by API_KEY_ENV (default OPENAI_API_KEY). The
# model column is sent verbatim as the request's "model":
#   * the litellm provider strings below (fireworks_ai/..., azure_ai/...) are what a LiteLLM proxy
#     (`litellm --config`, open source) routes to Fireworks / Azure AI Foundry; point BASE_URL at it;
#   * direct Fireworks: BASE_URL=https://api.fireworks.ai/inference/v1 API_KEY_ENV=FIREWORKS_API_KEY and
#     MODEL_SPECS with accounts/fireworks/models/<id> names (Fireworks-specific request fields, e.g.
#     --reasoning-history, then work as-is).
#
#   TASKS=<dir with train1600_c0..c7.parquet (make_train_chunks.py)> OUT=outputs/pi_multimodel \
#     bash scripts/analysis/run_alf_pi_multimodel.sh
# Resumable: rerunning skips every trajectory already in the per-chunk output files.
# ---------------------------------------------------------------------------------------------------
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ "$PWD" != / ] && ! { [ -f pyproject.toml ] && [ -f opsd/__init__.py ] && [ -f opsd/skyrl_ext/__init__.py ]; }; do cd ..; done; pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"  # `opsd` = the package at <repo root>/opsd
OUT=${OUT:-outputs/pi_multimodel}
TASKS=${TASKS:-$OUT/tasks}
CONC=${CONC:-12}
WAIT_H=${WAIT_H:-12}
PY=${PY:-$REPO/.venv/bin/python}
BASE=${BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}
API_KEY_ENV=${API_KEY_ENV:-OPENAI_API_KEY}
: "${ALFWORLD_VENV:?set ALFWORLD_VENV (the ALFWorld server venv)}"
: "${ALFWORLD_DATA:?set ALFWORLD_DATA (the ALFWorld game data root)}"
[ -n "${!API_KEY_ENV:-}" ] || { echo "FATAL: set $API_KEY_ENV (the endpoint key; see API_KEY_ENV)" >&2; exit 1; }
mkdir -p "$OUT"
export TMPDIR=${TMPDIR_OVERRIDE:-/var/tmp/$USER-pimm}; mkdir -p "$TMPDIR"
rm -f "$OUT/STOP"

# name  model (litellm provider string)  $/M-in  $/M-out  $/M-cached  cap-USD  base-port
MODELS=(
  "dsv41flash fireworks_ai/deepseek-v4p1-flash 0.22 0.66 0.01 30 10100"
  "minimaxm3  fireworks_ai/minimax-m3          0.30 1.20 0.06 30 10130"
  "qwen38max  fireworks_ai/qwen3p8-max         2.00 6.00 0.25 120 10160"
  "kimik3     fireworks_ai/kimi-k3             3.00 15.00 0.30 200 10190"
  "glm52      fireworks_ai/glm-5p2             1.40 4.40 0.14 100 10220"
)
# EXTRA_BODY='{"chat_template_kwargs":{"thinking":true}}' is passed to every driver as --extra-body.
# MAX_COMPLETION_TOKENS raises the 8000-token per-turn output cap (the OPSD default), e.g. for models whose hidden
# reasoning counts against it.
# N_SAMPLES samples several trajectories per task (default 1), e.g. 7 more for an 8-slot same-source PI.
# MODEL_SPECS="name model pin pout pcache cap port;..." replaces the built-in list (e.g. models served under
# another route, like azure_ai/DeepSeek-V4-Pro or fireworks_ai/glm-5p3).
if [ -n "${MODEL_SPECS:-}" ]; then
  IFS=';' read -ra MODELS <<< "$MODEL_SPECS"
fi
if [ -n "${ONLY:-}" ]; then   # ONLY="dsv41flash kimik3" runs a subset
  keep=()
  for spec in "${MODELS[@]}"; do
    for o in $ONLY; do [ "${spec%% *}" = "$o" ] && keep+=("$spec"); done
  done
  MODELS=("${keep[@]}")
fi

# A litellm provider string (fireworks_ai/..., azure_ai/...) only means something to a LiteLLM proxy (or the
# provider's own endpoint); the OpenAI default rejects it. Fail fast instead of re-probing an endpoint that
# can never serve it for WAIT_H hours.
if [ -z "${BASE_URL:-}${OPENAI_BASE_URL:-}" ]; then
  for spec in "${MODELS[@]}"; do
    read -r _ m _ <<< "$spec"
    case "$m" in openai/*) ;; */*)
      echo "FATAL: $m is a provider-routed model id; set BASE_URL (or OPENAI_BASE_URL) to the endpoint that serves it" >&2
      exit 1 ;;
    esac
  done
fi

PROBE_CODE=
probe() {  # the key goes in via stdin (-H @-), never on the curl command line; sets PROBE_CODE
  PROBE_CODE=$(printf 'Authorization: Bearer %s\n' "${!API_KEY_ENV}" |
    curl -s -o /dev/null -w '%{http_code}' "$BASE/chat/completions" -H @- \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply ok.\"}],\"max_tokens\":64}")
  [ "$PROBE_CODE" = 200 ]
}

OPSD_FLAGS=(--prompt-style react --show-admissible false --invalid-action-feedback true
  --invalid-action-show-admissible false --history-length 0 --max-turns 30 --max-input-tokens 32000
  --reasoning-style prefix --reasoning-prefix "THOUGHTS:" --reasoning-tag think --think-tags-in-prompt true
  --oneshot-example false --enable-thinking false --strict-action-outside-think false
  --invalid-action-feedback-guide false --reward-scale 0.1
  --temperature 1.0 --top-p 1.0 --top-k 20 --min-p 0.0 --max-completion-tokens "${MAX_COMPLETION_TOKENS:-8000}" --no-stop
  --n-samples "${N_SAMPLES:-1}" --max-retries 8)

run_model() {
  local name=$1 model=$2 pin=$3 pout=$4 pcache=$5 cap=$6 port=$7
  local d="$OUT/$name"; mkdir -p "$d"
  local waited=0
  until probe "$model"; do
    case "$PROBE_CODE" in 401|403)  # a key / permission problem: waiting cannot fix it
      echo "$(date -u +%H:%M) $name: probe HTTP $PROBE_CODE from $BASE (check $API_KEY_ENV), skipped"; return ;;
    esac
    if [ "$waited" -ge $((WAIT_H * 60)) ]; then echo "$(date -u +%H:%M) $name: UNAVAILABLE after ${WAIT_H}h (last probe HTTP $PROBE_CODE), skipped"; return; fi
    [ "$waited" = 0 ] && echo "$(date -u +%H:%M) $name: probe HTTP $PROBE_CODE (not available); re-probing every 30 min for ${WAIT_H}h"
    sleep 1800; waited=$((waited + 30))
    [ -f "$OUT/STOP" ] && return
  done
  local spent=0
  for f in $(ls "$TASKS"/train1600_c*.parquet | sort -V); do
    [ -f "$OUT/STOP" ] && { echo "$(date -u +%H:%M) $name: STOP ($OUT/STOP)"; return; }
    local c; c=$(basename "$f" .parquet); c=${c#train1600_}
    local left; left=$(awk -v a="$cap" -v b="$spent" 'BEGIN{printf "%.4f", a-b}')
    if awk -v l="$left" 'BEGIN{exit !(l <= 0)}'; then echo "$(date -u +%H:%M) $name: model cap \$$cap reached"; return; fi
    local pool="$d/serve_$c"; mkdir -p "$pool"
    local urls
    urls=$(ALFWORLD_VENV="$ALFWORLD_VENV" ALFWORLD_DATA="$ALFWORLD_DATA" \
      NUM_ALFWORLD_SERVERS="$CONC" ALFWORLD_BASE_PORT="$port" ALFWORLD_HOST=0.0.0.0 \
      ALFWORLD_ADVERTISE_HOST=127.0.0.1 ALFWORLD_MAX_STEPS=30 ALFWORLD_LOG_DIR="$pool" \
      bash opsd/alfworld/server/launch_alfworld_servers.sh 2>>"$pool/launch.log" | tail -1)
    sleep "${ALFWORLD_SERVER_WARMUP_SEC:-20}"
    echo "$(date -u +%H:%M) $name: $c (cap left \$$left)"
    PYTHONPATH="$REPO" "$PY" -m opsd.alfworld.generate_alfworld_litellm \
      --data-file "$f" --output "$d/$c.jsonl" --base-url "$BASE" --model "$model" --api-key-env "$API_KEY_ENV" \
      --server-urls "$urls" --max-concurrency "$CONC" "${OPSD_FLAGS[@]}" --budget-usd "$left" \
      --price-in-per-1m "$pin" --price-out-per-1m "$pout" --price-cache-read-per-1m "$pcache" \
      ${EXTRA_BODY:+--extra-body "$EXTRA_BODY"} \
      > "$d/$c.log" 2>&1 &
    echo $! > "$d/driver.pid"
    wait "$(cat "$d/driver.pid")"
    [ -f "$pool/alfworld_serve.pids" ] && kill $(cat "$pool/alfworld_serve.pids") 2>/dev/null
    local cost; cost=$(tr '\r' '\n' < "$d/$c.log" | grep -oE '^COST: \$[0-9.]+' | tail -1 | sed -E 's/^COST: \$//')
    spent=$(awk -v a="$spent" -v b="${cost:-0}" 'BEGIN{printf "%.4f", a+b}')
    echo "$(date -u +%H:%M) $name: $c done, chunk \$${cost:-?}, model total \$$spent"
  done
  echo "$(date -u +%H:%M) $name: ALL CHUNKS DONE, \$$spent"
}

pids=()
for spec in "${MODELS[@]}"; do
  # shellcheck disable=SC2086
  run_model $spec & pids+=($!)
  sleep 5
done
for p in "${pids[@]}"; do wait "$p"; done
echo "$(date -u +%H:%M) ALL MODELS DONE (per-chunk metered cost: the COST lines of $OUT/*/c*.log)"

#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------------
# Top up each PI author until EVERY train1600 task has at least one WON trajectory from that author.
#
# Round r (1..ROUNDS): collect the tasks the author has not solved yet in any of its outputs
# (c*.jsonl from run_alf_pi_multimodel.sh + earlier topup_r*.jsonl), and sample ONE more trajectory for
# each into <author>/topup_r<r>.jsonl. Every attempt is kept (all tokens, like the main run); a solved task
# is never re-sampled, so each task has at most one won trajectory per author. Same OPSD training contract,
# per-round fresh ALFWorld pool, and a per-author cap carried across rounds, metered from token usage x the
# per-model prices (the driver's --budget-usd). `touch $OUT/STOP_TOPUP` stops every author before its next
# round. Endpoint and key: BASE_URL / API_KEY_ENV exactly as in run_alf_pi_multimodel.sh.
#
#   OUT=outputs/pi_multimodel bash scripts/analysis/run_alf_pi_topup.sh
# EXTRA_BODY='{"chat_template_kwargs":{"thinking":true}}' is passed to every driver as --extra-body.
# MAX_COMPLETION_TOKENS raises the 8000-token per-turn output cap (the OPSD default), e.g. for models whose hidden
# reasoning counts against it.
# MODEL_SPECS as in run_alf_pi_multimodel.sh ("name model pin pout pcache cap port;...").
# WINS_TARGET=K tops up until each task has K won trajectories in the author dir (default 1).
# START_ROUND continues after an earlier top-up (e.g. START_ROUND=8 ROUNDS=27 after rounds 1-7): each round
# writes NEW topup_r<r>.* files, and the per-author cap counts only this invocation's rounds.
# ---------------------------------------------------------------------------------------------------
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ "$PWD" != / ] && ! { [ -f pyproject.toml ] && [ -f opsd/__init__.py ] && [ -f opsd/skyrl_ext/__init__.py ]; }; do cd ..; done; pwd)"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"  # `opsd` = the package at <repo root>/opsd
OUT=${OUT:-outputs/pi_multimodel}
ROUNDS=${ROUNDS:-7}
START_ROUND=${START_ROUND:-1}
WINS_TARGET=${WINS_TARGET:-1}   # top up until every task has this many WON trajectories in the author dir
if [ "$WINS_TARGET" = 1 ]; then UNIT="unsolved tasks"; else UNIT="attempts (tasks under $WINS_TARGET wins)"; fi
CONC=${CONC:-12}
PY=${PY:-$REPO/.venv/bin/python}
BASE=${BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}
API_KEY_ENV=${API_KEY_ENV:-OPENAI_API_KEY}
: "${ALFWORLD_VENV:?set ALFWORLD_VENV (the ALFWorld server venv)}"
: "${ALFWORLD_DATA:?set ALFWORLD_DATA (the ALFWorld game data root)}"
[ -n "${!API_KEY_ENV:-}" ] || { echo "FATAL: set $API_KEY_ENV (the endpoint key; see API_KEY_ENV)" >&2; exit 1; }
export TMPDIR=${TMPDIR_OVERRIDE:-/var/tmp/$USER-pimm}; mkdir -p "$TMPDIR"
rm -f "$OUT/STOP_TOPUP"

MODEL_SPECS=${MODEL_SPECS:-"dsv41flash fireworks_ai/deepseek-v4p1-flash 0.22 0.66 0.01 20 10100;\
minimaxm3 fireworks_ai/minimax-m3 0.30 1.20 0.06 20 10130;\
qwen38max fireworks_ai/qwen3p8-max 2.00 6.00 0.25 40 10160;\
kimik3 fireworks_ai/kimi-k3 3.00 15.00 0.30 70 10190;\
glm53 fireworks_ai/glm-5p3 1.40 4.40 0.26 40 10250;\
dsv4pro_az azure_ai/DeepSeek-V4-Pro 1.74 3.48 0.145 60 10280"}
IFS=';' read -ra MODELS <<< "$MODEL_SPECS"
# A litellm provider string (fireworks_ai/..., azure_ai/...) only means something to a LiteLLM proxy (or the
# provider's own endpoint); the OpenAI default rejects it. Fail fast instead of failing every request of every
# round against the default endpoint.
if [ -z "${BASE_URL:-}${OPENAI_BASE_URL:-}" ]; then
  for spec in "${MODELS[@]}"; do
    read -r _ m _ <<< "$spec"
    case "$m" in openai/*) ;; */*)
      echo "FATAL: $m is a provider-routed model id; set BASE_URL (or OPENAI_BASE_URL) to the endpoint that serves it" >&2
      exit 1 ;;
    esac
  done
fi

OPSD_FLAGS=(--prompt-style react --show-admissible false --invalid-action-feedback true
  --invalid-action-show-admissible false --history-length 0 --max-turns 30 --max-input-tokens 32000
  --reasoning-style prefix --reasoning-prefix "THOUGHTS:" --reasoning-tag think --think-tags-in-prompt true
  --oneshot-example false --enable-thinking false --strict-action-outside-think false
  --invalid-action-feedback-guide false --reward-scale 0.1
  --temperature 1.0 --top-p 1.0 --top-k 20 --min-p 0.0 --max-completion-tokens "${MAX_COMPLETION_TOKENS:-8000}" --no-stop
  --n-samples 1 --max-retries 8)

unsolved_parquet() {   # $1 = author dir, $2 = output parquet; prints the number of attempts this round
  # One row per attempt: a task with w < WINS_TARGET wins gets WINS_TARGET - w rows (at the default target of 1,
  # exactly one row per still-unsolved task, as before).
  "$PY" - "$1" "$2" "$OUT/tasks" "${WINS_TARGET:-1}" <<'PYEOF'
import collections, glob, json, sys
import pandas as pd
d, out, tasks, target = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
won = collections.Counter()
for f in glob.glob(f"{d}/c*.jsonl") + glob.glob(f"{d}/topup_r*.jsonl"):
    for line in open(f):
        r = json.loads(line)
        if r["won"]:
            won[r["task_id"]] += 1
df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{tasks}/train1600_c*.parquet"))], ignore_index=True)
need = df["extra_info"].map(lambda e: max(target - won[e["task_id"]], 0))
todo = df.loc[df.index.repeat(need)].reset_index(drop=True)
todo.to_parquet(out)
print(len(todo))
PYEOF
}

topup_model() {
  local name=$1 model=$2 pin=$3 pout=$4 pcache=$5 cap=$6 port=$7
  local d="$OUT/$name" spent=0
  [ -d "$d" ] || { echo "$(date -u +%H:%M) $name: no main-run output, skipped"; return; }
  for r in $(seq "$START_ROUND" "$ROUNDS"); do
    [ -f "$OUT/STOP_TOPUP" ] && { echo "$(date -u +%H:%M) $name: STOP ($OUT/STOP_TOPUP)"; return; }
    local left; left=$(awk -v a="$cap" -v b="$spent" 'BEGIN{printf "%.4f", a-b}')
    if awk -v l="$left" 'BEGIN{exit !(l <= 0)}'; then echo "$(date -u +%H:%M) $name: top-up cap \$$cap reached"; break; fi
    local todo; todo=$(unsolved_parquet "$d" "$d/topup_r$r.parquet")
    if [ "$todo" = 0 ]; then echo "$(date -u +%H:%M) $name: every task solved"; return; fi
    local pool="$d/serve_topup_r$r"; mkdir -p "$pool"
    local urls
    urls=$(ALFWORLD_VENV="$ALFWORLD_VENV" ALFWORLD_DATA="$ALFWORLD_DATA" \
      NUM_ALFWORLD_SERVERS="$CONC" ALFWORLD_BASE_PORT="$port" ALFWORLD_HOST=0.0.0.0 \
      ALFWORLD_ADVERTISE_HOST=127.0.0.1 ALFWORLD_MAX_STEPS=30 ALFWORLD_LOG_DIR="$pool" \
      bash opsd/alfworld/server/launch_alfworld_servers.sh 2>>"$pool/launch.log" | tail -1)
    sleep "${ALFWORLD_SERVER_WARMUP_SEC:-20}"
    echo "$(date -u +%H:%M) $name: round $r, $todo $UNIT (cap left \$$left)"
    PYTHONPATH="$REPO" "$PY" -m opsd.alfworld.generate_alfworld_litellm \
      --data-file "$d/topup_r$r.parquet" --output "$d/topup_r$r.jsonl" --base-url "$BASE" --model "$model" \
      --api-key-env "$API_KEY_ENV" --server-urls "$urls" --max-concurrency "$CONC" "${OPSD_FLAGS[@]}" \
      --budget-usd "$left" --price-in-per-1m "$pin" --price-out-per-1m "$pout" --price-cache-read-per-1m "$pcache" \
      ${EXTRA_BODY:+--extra-body "$EXTRA_BODY"} \
      > "$d/topup_r$r.log" 2>&1 &
    echo $! > "$d/topup_driver.pid"
    wait "$(cat "$d/topup_driver.pid")"
    rm -f "$d/topup_driver.pid"
    # shellcheck disable=SC2046  # one kill for the whole pool: word splitting is intended
    [ -f "$pool/alfworld_serve.pids" ] && kill $(cat "$pool/alfworld_serve.pids") 2>/dev/null
    local cost; cost=$(tr '\r' '\n' < "$d/topup_r$r.log" | grep -oE '^COST: \$[0-9.]+' | tail -1 | sed -E 's/^COST: \$//')
    spent=$(awk -v a="$spent" -v b="${cost:-0}" 'BEGIN{printf "%.4f", a+b}')
    echo "$(date -u +%H:%M) $name: round $r done, \$${cost:-?} (top-up total \$$spent)"
  done
  local left_n; left_n=$(unsolved_parquet "$d" "$d/topup_unsolved.parquet")
  echo "$(date -u +%H:%M) $name: finished with $left_n $UNIT left, top-up \$$spent"
}

pids=()
for spec in "${MODELS[@]}"; do
  # shellcheck disable=SC2086
  topup_model $spec & pids+=($!)
  sleep 5
done
for p in "${pids[@]}"; do wait "$p"; done
echo "$(date -u +%H:%M) top-up done (per-round metered cost: the COST lines of $OUT/*/topup_r*.log)"

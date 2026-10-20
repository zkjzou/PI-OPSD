#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------------
# Serve a POOL of Qwen3-4B-Instruct-2507 vLLM servers, one per GPU (TP=1), for local sampling.
#
# WHY NOT REUSE THE THINKING POOL (serve_qwen3_4bt_8gpu.sh): it defaults to ``--reasoning-parser qwen3``,
# which is for the THINKING checkpoint. Instruct-2507 emits no ``<think>`` blocks, so the parser has nothing to
# split and its presence is at best a no-op and at worst a source of empty ``content``.
#
# ⚠️ TOOL CALLING IS REQUIRED, NOT OPTIONAL. τ²'s ``LLMAgent`` sends ``tool_choice="auto"`` on turn 0;
#    a server started without ``--enable-auto-tool-choice --tool-call-parser hermes`` rejects the very
#    first request of every episode, so the whole pass fails at 0% with no obvious cause.
#
# ⚠️ TP=1 x N REPLICAS, NOT TP=N. Sampling here is throughput-bound, not memory-bound (a 4B fits in one
#    H100), and the driver round-robins a comma-separated --policy-base-url. TP>1 also drags in the
#    NCCL/cuMem workarounds that TP=1 avoids entirely.
#
# ⚠️ SERIAL LAUNCH WITH RETRY. Simultaneous ``torch.cuda`` init flakes on a shared host (transient);
#    each replica is brought up and health-checked before the next starts, with 4 attempts.
#
# Usage:  GPUS=0,1,2,3,4,5,6,7 bash scripts/analysis/serve_qwen3_4b_instruct_pool.sh
#   knobs: BASE_PORT(8300) MAX_LEN(40000) UTIL(0.9) STATE_DIR(./outputs)
#   stop:  kill $(cat ./outputs/serve_q4bi_pool.pids)
# ---------------------------------------------------------------------------------------------------
set -uo pipefail
STATE_DIR=${STATE_DIR:-${OUT_ROOT:-$PWD/outputs}}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
while [ "$REPO" != "/" ] && ! { [ -f "$REPO/pyproject.toml" ] && [ -f "$REPO/opsd/__init__.py" ] && [ -f "$REPO/opsd/skyrl_ext/__init__.py" ]; }; do REPO=$(dirname "$REPO"); done
cd "$REPO"
MODEL=${MODEL:-"Qwen/Qwen3-4B-Instruct-2507"}; MODEL_NAME=${MODEL_NAME:-"Qwen3-4B-Instruct-2507"}
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}; BASE_PORT=${BASE_PORT:-8300}; MAX_LEN=${MAX_LEN:-40000}; UTIL=${UTIL:-0.9}
LOGDIR=${LOGDIR:-"$STATE_DIR/serve_q4bi_pool_logs"}; PIDFILE=${PIDFILE:-"$STATE_DIR/serve_q4bi_pool.pids"}
mkdir -p "$LOGDIR"; : > "$PIDFILE"
IFS=',' read -ra GA <<< "$GPUS"

launch_one() {  # gpu port -> echoes healthy pid
  local gpu="$1" port="$2" attempt=0
  while [ "$attempt" -lt 4 ]; do
    CUDA_VISIBLE_DEVICES=$gpu VLLM_HOST_IP=127.0.0.1 "$REPO/.venv/bin/vllm" serve "$MODEL" \
      --served-model-name "$MODEL_NAME" --host 0.0.0.0 --port "$port" \
      --gpu-memory-utilization "$UTIL" --max-model-len "$MAX_LEN" \
      --enable-auto-tool-choice --tool-call-parser hermes \
      --trust-remote-code --enable-prefix-caching \
      > "$LOGDIR/vllm_$port.log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 160); do
      curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { echo "$pid"; return 0; }
      kill -0 "$pid" 2>/dev/null || break
      sleep 5
    done
    kill -9 "$pid" 2>/dev/null; attempt=$((attempt+1)); sleep $((8 + RANDOM % 12))
  done; return 1
}

URLS=""
for idx in "${!GA[@]}"; do
  gpu="${GA[$idx]}"; port=$((BASE_PORT+idx))
  echo "[serve] GPU $gpu -> :$port"
  if pid=$(launch_one "$gpu" "$port"); then
    echo "$pid" >> "$PIDFILE"
    URLS="${URLS:+$URLS,}http://127.0.0.1:$port/v1"
    echo "[serve] GPU $gpu healthy (pid $pid)"
  else
    echo "[serve] FATAL: GPU $gpu never became healthy; see $LOGDIR/vllm_$port.log" >&2
    exit 1
  fi
done
echo "$URLS" > "$STATE_DIR/serve_q4bi_pool.urls"
echo "[serve] pool ready: $URLS"
echo "[serve] urls file : $STATE_DIR/serve_q4bi_pool.urls"
echo "[serve] pids      : $PIDFILE"

#!/usr/bin/env bash
# =============================================================================================
# τ² pass@k SAMPLING — serve the policy locally, drive baseline.py against a user simulator.
#
#   MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 POLICY_ENABLE_THINKING=false \
#   SIMULATOR_MODEL=Qwen/Qwen3.5-35B-A3B SIMULATOR_BASE_URL=http://<sim-host>:8000/v1 SIM_API_KEY_ENV= \
#     bash opsd/tau2_bench/run_tau2_passk.sh
#
# Prerequisites (one-time, in this order; see README.md "Setup"; bash scripts/setup/setup_tau2.sh does all
# of it): τ²'s extra deps (the `tau2` extra: `bash scripts/setup/setup_train_env.sh --tau2`), then
# `python -m opsd.tau2_bench.prepare_data prepare` (pinned τ² checkout + manifests) and,
# for the AReaL pools, areal.stage_local(domain=...). The simulator is EITHER a
# self-hosted vLLM (serve_user_sim.sh, on other GPUs or another node; SIM_API_KEY_ENV empty) OR a
# hosted OpenAI-compatible endpoint (e.g. SIMULATOR_MODEL=gpt-5.6-luna
# SIMULATOR_BASE_URL=${OPENAI_BASE_URL:-https://api.openai.com/v1} SIM_API_KEY_ENV=OPENAI_API_KEY
# SIM_REASONING_EFFORT=none). The retail NL judge (openai/gpt-4.1) uses baseline.py's defaults
# ($OPENAI_BASE_URL / $OPENAI_API_KEY). ⚠️ The simulator IS the environment: pass@k is only comparable
# against corpora sampled with the same simulator.
#
# This script starts POLICY_REPLICAS vLLM servers on the GPUs in CUDA_VISIBLE_DEVICES (default 0..7)
# and stops exactly those servers when it exits.
#
# WHY NOT THE TRAINER'S GENERATE PATH. A colocate vLLM engine serves the weights the TRAINER pushed,
# and on Qwen3-30B-A3B that path emitted pure `++;` on 144/144 rollouts (per-chunk load_weights without
# process_weights_after_loading). Standalone `vllm serve` loads from the checkpoint and is the only
# trustworthy way to sample a MoE here.
#
# ⚠️ TP IS NOT OPTIONAL FOR THE 30B. Qwen3-30B-A3B-Instruct-2507 is ~57 GiB; at TP=1 that leaves ~10 GB
# for KV on an 80 GB card, i.e. ~1.7 concurrent 40k sequences. TP=2 -> ~30 GiB/GPU and 4 replicas per
# node. num_key_value_heads=4, so TP=2 shards KV without replication.
#
# ⚠️ NO --reasoning-parser ON THE POLICY. With `qwen3` the server splits reasoning into
# `reasoning_content` and leaves only the fence in `content`, so --preserve-thinking silently becomes
# a no-op and the Thinking arm is measured on a context it never trains with.
#
# Resume is free: baseline.py keys on (domain, task_id, trial) and only counts `terminated_*` as done,
# so re-running with the same OUT_DIR re-runs exactly the episodes it lost. At n=128 that matters —
# this is ~14.6k episodes per model.
# =============================================================================================
set -eo pipefail
# Repo root = first ancestor of this script holding pyproject.toml, opsd/__init__.py and opsd/skyrl_ext/__init__.py
# (depth-independent); `python -m` runs from the repo root, with it first on PYTHONPATH.
_find_root() {
  local d
  d=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  while [ "$d" != "/" ]; do
    [ -f "$d/pyproject.toml" ] && [ -f "$d/opsd/__init__.py" ] && [ -f "$d/opsd/skyrl_ext/__init__.py" ] &&
      { echo "$d"; return 0; }
    d=$(dirname "$d")
  done
  return 1
}
ROOT=${ROOT:-$(_find_root)} || { echo "[passk] FATAL: cannot locate the repo root" >&2; exit 1; }
# Relative paths given by the caller (OUT_DIR, RUN_ROOT, HINTS_DIR, ...) are relative to WHERE THEY RAN
# this script, not to the repo root we cd into for `python -m`.
_CALLER_PWD=$PWD
_abs() { case "$1" in /*|"") printf '%s' "$1" ;; *) printf '%s' "$_CALLER_PWD/$1" ;; esac; }
[ -n "${SIM_DONE_FILE:-}" ] && SIM_DONE_FILE=$(_abs "$SIM_DONE_FILE")
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"  # `opsd` = this checkout's package
PY=${PY:-python}

MODEL_PATH=${MODEL_PATH:?MODEL_PATH must be set}
PASSK_SPLIT=${PASSK_SPLIT:-retail-base}
PASSK_TRIALS=${PASSK_TRIALS:-128}
# `paper` is τ²'s FULLY ON-POLICY profile (T=1.0 / top_p 1.0 / top_k -1 / min_p 0) — the distribution
# GRPO and OPSD actually roll out from. The per-model `official` configs are REPORTING samplers and
# would measure a distribution training never sees.
PASSK_PROFILE=${PASSK_PROFILE:-paper}
PASSK_CONCURRENCY=${PASSK_CONCURRENCY:-128}
POLICY_REPLICAS=${POLICY_REPLICAS:-8}
POLICY_TP=${POLICY_TP:-1}
POLICY_PORT=${POLICY_PORT:-8300}
POLICY_MAX_LEN=${POLICY_MAX_LEN:-40000}
POLICY_GPU_UTIL=${POLICY_GPU_UTIL:-0.90}
PASSK_MAX_TOKENS=${PASSK_MAX_TOKENS:-8000}
PASSK_MAX_STEPS=${PASSK_MAX_STEPS:-60}
PASSK_EPISODE_TIMEOUT=${PASSK_EPISODE_TIMEOUT:-1200}
# trial-major by default: a killed run then leaves a COMPLETE curve at the largest finished k over
# the whole pool, instead of N finished tasks and the rest untouched. See baseline.py --episode-order.
PASSK_ORDER=${PASSK_ORDER:-trial}
# Re-use the same OUT_DIR to resume (see above). Default: $RUN_DIR/passk, RUN_DIR=$RUN_ROOT/$RUN_NAME.
RUN_DIR=${RUN_DIR:-${RUN_ROOT:-./runs}/${RUN_NAME:-tau2_passk}}
OUT_DIR=$(_abs "${OUT_DIR:-$RUN_DIR/passk}")

: "${SIMULATOR_BASE_URL:?SIMULATOR_BASE_URL must be set (serve_user_sim.sh address, or a hosted endpoint)}"
: "${SIMULATOR_MODEL:?SIMULATOR_MODEL must be set}"
# The retail NL judge (openai/gpt-4.1) runs inside the reward of τ²'s NATIVE retail tasks only (40 of
# retail-base's 114, 29 of retail-train's 74; `prepare_data audit`) and needs a key there. The AReaL
# pools (*-areal-rl) are judge-free, so a self-hosted-simulator run on them needs no key at all.
case "$PASSK_SPLIT" in
  retail-base|retail-train|all-base)
    : "${OPENAI_API_KEY:?set OPENAI_API_KEY (key for the retail NL judge, openai/gpt-4.1)}" ;;
esac
mkdir -p "$OUT_DIR"

# --- serve the policy: one replica per POLICY_TP GPUs, all on this node --------------------------
# GPU indices are taken from CUDA_VISIBLE_DEVICES when set (so a shared box can hand this script a
# subset), else 0..NUM_GPUS_PER_NODE-1.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=, read -r -a _GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  _GPUS=($(seq 0 $(( ${NUM_GPUS_PER_NODE:-8} - 1 ))))
fi
GPUS_TOTAL=${#_GPUS[@]}
if [ $(( POLICY_REPLICAS * POLICY_TP )) -gt "$GPUS_TOTAL" ]; then
  echo "[passk] FATAL: ${POLICY_REPLICAS} replicas x TP=${POLICY_TP} > ${GPUS_TOTAL} GPUs" >&2; exit 1
fi
# Stop exactly the policy servers THIS script started (each in its own process group), on any exit,
# and tell a SIM_DONE_FILE-watching serve_user_sim.sh that the consumer is done.
POLICY_PIDS=()
_cleanup() {
  local pid
  for pid in "${POLICY_PIDS[@]}"; do kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true; done
  if [ -n "${SIM_DONE_FILE:-}" ]; then mkdir -p "$(dirname "$SIM_DONE_FILE")"; touch "$SIM_DONE_FILE"; fi
}
trap _cleanup EXIT
trap '_cleanup; exit 130' INT
trap '_cleanup; exit 143' TERM
# Refuse busy ports: the health probe below would otherwise talk to SOMEONE ELSE's server on the port
# while our replica silently failed to bind.
for i in $(seq 0 $(( POLICY_REPLICAS - 1 ))); do
  port=$(( POLICY_PORT + i ))
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    echo "[passk] FATAL: port $port is already in use; set POLICY_PORT to a free range" >&2; exit 1
  fi
done
# ⚠️ RESOLVE THE REPO ONCE, NOT ONCE PER REPLICA. Every vLLM replica calls the HF API
# (`list_repo_files`) at startup. Measured 2026-08-22: three of these jobs launched together put ~32
# replicas (20 policy + 12 simulator, 6 nodes) onto the HF API inside a minute and blew the account's
# 1000-request/5-minute quota. Every replica 429'd, and because the health loop only gives up at
# POLICY_WAIT_SECS the nodes sat for a FULL HOUR before reporting "replica 0 never became healthy" —
# a message that says nothing about HuggingFace. One prefetch, then serve from cache.
if [ "${HF_PREFETCH:-1}" = "1" ]; then
  echo "[passk] pre-resolving $MODEL_PATH (1 API call instead of $POLICY_REPLICAS)"
  if $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_PATH', max_workers=8)
"; then
    export HF_HUB_OFFLINE=1
    echo "[passk] prefetch ok -> HF_HUB_OFFLINE=1"
  else
    echo "[passk] WARNING: prefetch failed; replicas will each hit the HF API (429 risk)" >&2
  fi
fi

POLICY_URLS=""
for i in $(seq 0 $(( POLICY_REPLICAS - 1 ))); do
  port=$(( POLICY_PORT + i ))
  first=$(( i * POLICY_TP ))
  devs=$(IFS=,; echo "${_GPUS[*]:$first:$POLICY_TP}")
  echo "[passk] replica $i: GPUs $devs -> :$port"
  CUDA_VISIBLE_DEVICES="$devs" setsid $PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name "$MODEL_PATH" \
    --port "$port" --host 0.0.0.0 \
    --tensor-parallel-size "$POLICY_TP" \
    --max-model-len "$POLICY_MAX_LEN" \
    --gpu-memory-utilization "$POLICY_GPU_UTIL" \
    --enable-prefix-caching --trust-remote-code \
    --enable-auto-tool-choice --tool-call-parser hermes \
    ${POLICY_EXTRA_ARGS:-} > "$OUT_DIR/policy_$i.log" 2>&1 &
  POLICY_PIDS+=($!)
  POLICY_URLS="${POLICY_URLS:+$POLICY_URLS,}http://127.0.0.1:$port/v1"
done

echo "[passk] waiting for $POLICY_REPLICAS policy replica(s)…"
deadline=$(( $(date +%s) + ${POLICY_WAIT_SECS:-1200} ))
for i in $(seq 0 $(( POLICY_REPLICAS - 1 ))); do
  port=$(( POLICY_PORT + i ))
  until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    { [ "$(date +%s)" -lt "$deadline" ] && kill -0 "${POLICY_PIDS[$i]}" 2>/dev/null; } || {
      echo "[passk] FATAL: replica $i never became healthy (or exited)" >&2
      tail -30 "$OUT_DIR/policy_$i.log" >&2; exit 1; }
    sleep 10
  done
  echo "[passk]   replica $i healthy"
done

# The local engines serve without --api-key; a stray VLLM_API_KEY makes the client send an
# Authorization header the server rejects. This has bitten every local τ run.
unset VLLM_API_KEY

echo "[passk] model=$MODEL_PATH split=$PASSK_SPLIT trials=$PASSK_TRIALS profile=$PASSK_PROFILE"
echo "[passk] policy=$POLICY_URLS"
echo "[passk] sim=$SIMULATOR_MODEL @ $SIMULATOR_BASE_URL"
echo "[passk] out=$OUT_DIR"

SIM_ARGS=(--user-llm "$SIMULATOR_MODEL" --user-base-url "$SIMULATOR_BASE_URL")
# A self-hosted simulator needs no auth; baseline.py falls back to "dummy" for a local endpoint. Only
# pass a key env when one is actually configured (a hosted gateway).
[ -n "${SIM_API_KEY_ENV:-}" ] && SIM_ARGS+=(--user-api-key-env "$SIM_API_KEY_ENV")
[ -n "${SIM_REASONING_EFFORT:-}" ] && SIM_ARGS+=(--user-reasoning-effort "$SIM_REASONING_EFFORT")
THINK_ARGS=()
if [ "${POLICY_ENABLE_THINKING:-true}" = "false" ]; then
  THINK_ARGS+=(--no-thinking)
else
  THINK_ARGS+=(--preserve-thinking)   # match the training-time context for a thinking policy
fi

set +e
$PY -m opsd.tau2_bench.baseline \
  --split "$PASSK_SPLIT" \
  --model "$MODEL_PATH" --policy-base-url "$POLICY_URLS" \
  "${SIM_ARGS[@]}" "${THINK_ARGS[@]}" \
  --profile "$PASSK_PROFILE" \
  --max-tokens "$PASSK_MAX_TOKENS" --max-steps "$PASSK_MAX_STEPS" \
  --episode-timeout "$PASSK_EPISODE_TIMEOUT" \
  --num-trials "$PASSK_TRIALS" --concurrency "$PASSK_CONCURRENCY" \
  --episode-order "$PASSK_ORDER" \
  --output-dir "$OUT_DIR" --yes-really "$@"
RC=$?
set -e
echo "[passk] baseline exited rc=$RC"

# The EXIT trap stops the policy servers and creates $SIM_DONE_FILE (if set).
exit $RC

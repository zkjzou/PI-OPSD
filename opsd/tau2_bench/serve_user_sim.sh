#!/usr/bin/env bash
# =============================================================================
# Serve the SIMULATED USER on dedicated GPUs or a dedicated node (vLLM, OpenAI-compatible).
#
# Why this exists: the user simulator IS the environment, and pointing it at a hosted API put a
# third-party quota on the critical path of every episode. Observed 2026-07-29: a PROVIDER-level
# rate limit (not our per-key limit) collapsed three concurrent runs to rewardable_frac 0.375-0.61 —
# 80 of 128 episodes lost in one step — with litellm's num_retries=3 amplifying the pressure from
# inside each episode's critical path. Hosting the simulator on our own GPUs removes that dependency
# entirely: no provider quota, no per-key TPM, no throttling retries.
#
# It never joins the Ray cluster and runs until killed (Ctrl-C / SIGTERM stops every replica it
# started). The trainer / sampler reaches it at http://<this-host>:$SIM_PORT/v1 (comma-separate one URL
# per replica), with an EMPTY simulator key env (a self-hosted http:// endpoint needs no key):
#
#   SIM_SERVE_MODEL=Qwen/Qwen3.5-35B-A3B SIM_REPLICAS=4 SIM_TP=2 \
#   SIM_EXTRA_ARGS="--language-model-only --enforce-eager --reasoning-parser qwen3 --trust-remote-code" \
#     bash opsd/tau2_bench/serve_user_sim.sh
#
# ⚠️ Changing the simulator changes the ENVIRONMENT: rewards are only comparable within one simulator.
# Optional: SIM_DONE_FILE=<path> -- exit (and free the GPUs) once a consumer creates that file
# (run_tau2_passk.sh / run_tau2_pi_utility.sh create it when they finish, if given the same path).
# =============================================================================
set -eo pipefail

: "${SIM_SERVE_MODEL:?SIM_SERVE_MODEL must be set (HF id of the user-simulator model)}"
SIM_PORT=${SIM_PORT:-8000}
# SHARDING. The old default was TP=8 single-replica, justified above for a DENSE ~32B. That reasoning
# does not carry to Qwen3.5-35B-A3B, which is an **MoE with ~3B active params**: an 8-way all-reduce
# per token dominates the tiny per-token compute, and it measured 2.2x slower per call than the dense
# 30B at the same TP. 67 GiB of bf16 weights shard to 33.5 GiB/GPU at TP=2, leaving ~38 GiB of KV
# headroom at util 0.90 — so SIM_REPLICAS=4 x SIM_TP=2 fills an 8-GPU node with 4 independent engines
# and only 2-way comms. (TP=1 would remove comms entirely but leaves ~4.6 GiB of KV, too little for
# the prefix cache below to pay off.)
SIM_TP=${SIM_TP:-8}
SIM_REPLICAS=${SIM_REPLICAS:-1}
SIM_MAX_LEN=${SIM_MAX_LEN:-16384}
SIM_GPU_UTIL=${SIM_GPU_UTIL:-0.90}
# Extra `vllm serve` flags for this model, word-split on purpose (see the list further down).
SIM_EXTRA_ARGS=${SIM_EXTRA_ARGS:-}
[ -n "$SIM_EXTRA_ARGS" ] && echo "[sim] extra args: $SIM_EXTRA_ARGS"

# --- STATUS TRAIL: a few timestamped lines that outlive the console ------------------------------
# When a consumer later fails on "[sim] FATAL: user simulator not healthy after 3600s", the cause is
# otherwise unknowable — was the 35B still loading, or already healthy and unreachable (wrong
# host/port)? Those need opposite fixes (raise SIM_WAIT_SECS vs fix the address), so guessing costs
# another hour per attempt.
# Deliberately a few small lines, NOT a tee of the whole vLLM log: `exec > >(tee ...)` would make this
# script's stdout a pipe, and the launch-inline comment below is there precisely because backgrounded
# replicas inheriting a pipe has already broken this script once.
SIM_STATUS_LOCAL=${SIM_STATUS_LOG:-${TMPDIR:-/tmp}/tau2_sim_status.$(id -u).$SIM_PORT.log}
mkdir -p "$(dirname "$SIM_STATUS_LOCAL")"
_sim_status() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$SIM_STATUS_LOCAL"
  echo "[sim-status] $*"
}

echo "[sim] serving user simulator: $SIM_SERVE_MODEL"
echo "[sim] tp=$SIM_TP port=$SIM_PORT max_model_len=$SIM_MAX_LEN gpu_util=$SIM_GPU_UTIL"
echo "[sim] host=0.0.0.0 (binds all interfaces: the trainer may be on another node)"
echo "[sim] status trail: $SIM_STATUS_LOCAL"
_sim_status "START model=$SIM_SERVE_MODEL replicas=$SIM_REPLICAS tp=$SIM_TP port=$SIM_PORT"

# vLLM enforces auth if this is set, and the trainer's health probe would then 401 forever.
unset VLLM_API_KEY

# ~60GB over 24 shards on first pull; allow the download and give it room. Weights land in
# ${HF_HOME} (huggingface_hub's default, ~/.cache/huggingface, when unset).
export HF_HUB_OFFLINE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ⚠️ RESOLVE THE REPO ONCE, NOT ONCE PER REPLICA. Every vLLM replica calls the HF API
# (`list_repo_files`) at startup. On 2026-08-22 three sampler jobs plus this simulator came up inside
# the same minute — ~32 replicas across 6 nodes — and blew the account's 1000-request/5-minute quota:
# `429 Too Many Requests` on `/api/models/Qwen/Qwen3.5-35B-A3B/tree/main`, replicas never became
# healthy, and each node then burned its FULL wait budget (3600 s) before failing. Nothing about the
# failure names HF: the visible symptom is "replica 0 never became healthy" an hour later.
# One prefetch here, then serve from cache. Same fix as the pre-staging note for the FSDP weight pull.
if [ "${HF_PREFETCH:-1}" = "1" ]; then
  echo "[sim] pre-resolving $SIM_SERVE_MODEL (1 API call instead of $SIM_REPLICAS)"
  _sim_status "PREFETCH start (~67GB on a cold node; this is the usual reason a sim misses SIM_WAIT_SECS)"
  if python -c "
import sys
from huggingface_hub import snapshot_download
snapshot_download('$SIM_SERVE_MODEL', max_workers=8)
"; then
    export HF_HUB_OFFLINE=1   # replicas now resolve from the local snapshot, zero API calls
    echo "[sim] prefetch ok -> HF_HUB_OFFLINE=1"
    _sim_status "PREFETCH done -> HF_HUB_OFFLINE=1"
  else
    # Don't hard-fail: a warm cache or a mirror may still let the replicas start.
    echo "[sim] WARNING: prefetch failed; replicas will each hit the HF API (429 risk)" >&2
  fi
fi

# NOTE no --disable-log-requests: vLLM 0.20.2 REMOVED it in favour of opt-in --enable-log-requests,
# so passing it is a hard `unrecognized arguments` exit before the model ever loads (observed: the sim
# died in 5s, taking the run that depended on it down too). Request logging is off by default here,
# which is what that flag was for. Do not re-add it without checking `vllm serve --help=all` for THIS
# version — and note plain `--help` only lists config GROUPS, so grepping it proves nothing.
# Model-specific flags, word-split on purpose. Some simulators cannot be served without them:
#   Qwen/Qwen3.5-35B-A3B is a VLM MoE -> "--language-model-only --enforce-eager
#     --reasoning-parser qwen3 --trust-remote-code".
# --reasoning-parser matters for a THINKING simulator beyond tidiness: without it the sim's own
# <think> stays in `content`, and content is exactly what τ² shows the AGENT — i.e. the agent would
# read the simulated user's private reasoning. Same leak as tau2/think_leak_frac, opposite direction.
# (We also send reasoning_effort="none" per request, which vLLM 0.20.2 honours as a Literal.)
#
# --enable-prefix-caching is load-bearing, not a nicety. A τ² simulator prompt grows MONOTONICALLY:
# turn N is turn N-1 plus two messages. With the cache, each user turn re-prefills almost nothing
# instead of the full 1.6k-3.4k tokens. It only pays off because the generator binds an episode to
# ONE replica for its lifetime (per-episode affinity) — a per-request balancer would scatter the
# turns and throw the reuse away.
if [ "$SIM_REPLICAS" -le 1 ]; then
  # shellcheck disable=SC2086
  exec vllm serve "$SIM_SERVE_MODEL" \
    --host 0.0.0.0 --port "$SIM_PORT" \
    --served-model-name "$SIM_SERVE_MODEL" \
    --tensor-parallel-size "$SIM_TP" \
    --max-model-len "$SIM_MAX_LEN" \
    --gpu-memory-utilization "$SIM_GPU_UTIL" \
    --enable-prefix-caching \
    ${SIM_EXTRA_ARGS:-}
fi

# Multi-replica: replica i owns GPUs [i*TP, (i+1)*TP) and serves on SIM_PORT+i. (Indices are relative
# to CUDA_VISIBLE_DEVICES when it is set.)
# Bring them up SERIALLY — simultaneous torch CUDA init flakily fails on a shared node, and a
# staggered start costs a couple of minutes against a multi-day run.
PIDS=()
RESTARTS=()
_SIM_VISIBLE=${CUDA_VISIBLE_DEVICES:-}
# Stop exactly the replicas THIS script started (each is its own process group, see setsid below).
_stop_replicas() {
  local pid
  for pid in "${PIDS[@]}"; do
    [ -n "$pid" ] || continue
    kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  done
}
trap '_stop_replicas' EXIT
trap '_stop_replicas; exit 130' INT
trap '_stop_replicas; exit 143' TERM
# Bounded so a permanently-broken GPU degrades to a loud teardown instead of an infinite churn.
SIM_MAX_RESTARTS=${SIM_MAX_RESTARTS:-3}

_launch_replica() {
  local i=$1 j gpus="" port=$(( SIM_PORT + $1 ))
  for j in $(seq 0 $(( SIM_TP - 1 ))); do gpus="${gpus:+$gpus,}$(( i * SIM_TP + j ))"; done
  if [ -n "${_SIM_VISIBLE:-}" ]; then   # map relative indices onto the caller's CUDA_VISIBLE_DEVICES
    local mapped="" k
    IFS=, read -r -a _vis <<< "$_SIM_VISIBLE"
    for k in ${gpus//,/ }; do mapped="${mapped:+$mapped,}${_vis[$k]}"; done
    gpus=$mapped
  fi
  echo "[sim] replica $i: GPUs $gpus -> :$port"
  # Launch INLINE, never via `pid=$(helper)`. A backgrounded child inherits the command-substitution
  # PIPE as its stdout, so $(...) cannot return until that child exits -- the script blocks forever
  # after replica 0 and replicas 1..N-1 are never launched. That is exactly what happened once:
  # replica 0 served fine, the simulator logged nothing further, every vllm log line was swallowed
  # into the pipe, and the trainer sat waiting for :8001 until SIM_WAIT_SECS.
  # Inline also keeps each replica's stdout on this script's stdout, so vllm logs stay visible.
  # (A *function* is fine -- the trap is command substitution, not the call.)
  #
  # `setsid` puts the replica in its OWN PROCESS GROUP so a restart can kill the whole tree with
  # `kill -- -PGID`. vLLM's EngineCore and WorkerProc children are spawned via multiprocessing and
  # their cmdlines do NOT contain `--port`, so a `pkill -f -- "--port N"` sweep misses them; they
  # would survive their parent still holding this replica's share of GPU memory, and the relaunch
  # would then OOM. Backgrounded children of a non-interactive shell are not group leaders, so
  # setsid execs without forking and $! remains the vllm pid (and its new PGID).
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpus" setsid vllm serve "$SIM_SERVE_MODEL" \
    --host 0.0.0.0 --port "$port" \
    --served-model-name "$SIM_SERVE_MODEL" \
    --tensor-parallel-size "$SIM_TP" \
    --max-model-len "$SIM_MAX_LEN" \
    --gpu-memory-utilization "$SIM_GPU_UTIL" \
    --enable-prefix-caching \
    ${SIM_EXTRA_ARGS:-} &
  PIDS[$i]=$!
}

# ⚠️ PROBE THE ENGINE, NOT THE PROCESS. `kill -0` and `/v1/models` are both served by the API-server
# frontend and stay green when the *engine* behind them is dead. Observed 2026-08-23 on the shared
# 35B simulator: a TP worker stopped answering, `multiproc_executor` raised
# `TimeoutError: RPC call to execute_model timed out`, EngineCore exited -- and the API server kept
# running, returning `EngineDeadError` 500s **for hours**. The process never exited, so `wait -n`
# never fired and the old `kill -0` watch below never tripped. Three training arms sharing the sim
# silently ran at rewardable_frac 0.74 (exactly 32 of 128 episodes = the one dead replica of four)
# and it read as a policy regression, not an infra fault. `/health` calls `check_health()`, which
# re-raises EngineDeadError -> 500, so it is the only probe that distinguishes the two.
_probe_ok() {
  local port=$(( SIM_PORT + $1 )) route=${HEALTH_ROUTE:-health}
  curl -sf --max-time 10 "http://127.0.0.1:$port/$route" >/dev/null 2>&1
}

_wait_healthy() {   # $1=replica, $2=max 5s ticks
  local i=$1 t
  for t in $(seq 1 "${2:-240}"); do
    curl -sf "http://127.0.0.1:$(( SIM_PORT + i ))/v1/models" >/dev/null 2>&1 && return 0
    kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "[sim] replica $i died during load"; return 1; }
    sleep 5
  done
  return 1
}

for i in $(seq 0 $(( SIM_REPLICAS - 1 ))); do
  _launch_replica "$i"
  _wait_healthy "$i" || { _sim_status "FATAL replica $i never became healthy"; \
      echo "[sim] FATAL: replica $i never became healthy"; exit 1; }
  echo "[sim] replica $i healthy on :$(( SIM_PORT + i ))"
  _sim_status "replica $i HEALTHY on :$(( SIM_PORT + i ))"
done
echo "[sim] ALL $SIM_REPLICAS REPLICAS HEALTHY on ports $SIM_PORT..$(( SIM_PORT + SIM_REPLICAS - 1 ))"
# ⚠️ IF THIS LINE IS IN THE STATUS TRAIL BUT THE CONSUMER STILL TIMED OUT, the simulator was UP and the
# consumer could not REACH it -- i.e. the simulator host/port it was given is wrong, and raising
# SIM_WAIT_SECS will not help. If the trail stops earlier, it is load time: raise the wait / pre-stage.
_sim_status "ALL $SIM_REPLICAS REPLICAS HEALTHY -- if the consumer still timed out, the ADDRESS is wrong, not the wait"

# Self-validate the probe against a known-good replica. If this vLLM build has no usable /health,
# a health-based watch would restart every replica on every tick -- far worse than the bug it fixes.
HEALTH_ROUTE=health
if ! _probe_ok 0; then
  echo "[sim] WARNING: /health unusable on this build; falling back to /v1/models (process-level only)" >&2
  HEALTH_ROUTE=v1/models
fi
echo "[sim] supervising via /$HEALTH_ROUTE, max $SIM_MAX_RESTARTS restarts per replica"

_replica_bad() {   # 0 = bad. Re-probes once so a transient blip is not a restart.
  _probe_ok "$1" && return 1
  sleep 5
  _probe_ok "$1" && return 1
  return 0
}

_restart_replica() {   # 0 = recovered, 1 = give up and exit (the EXIT trap stops the other replicas)
  local i=$1 port=$(( SIM_PORT + $1 )) t
  RESTARTS[$i]=$(( ${RESTARTS[$i]:-0} + 1 ))
  if [ "${RESTARTS[$i]}" -gt "$SIM_MAX_RESTARTS" ]; then
    echo "[sim] FATAL: replica $i exceeded $SIM_MAX_RESTARTS restarts; tearing down"; return 1
  fi
  echo "[sim] replica $i UNHEALTHY (restart ${RESTARTS[$i]}/$SIM_MAX_RESTARTS) -- reclaiming :$port"
  # Kill the whole process GROUP (see setsid note in _launch_replica); fall back to the bare pid if
  # this replica somehow is not a group leader. Then sweep by port as belt-and-braces -- restricted to
  # this user's `vllm serve` processes for exactly this port, never anyone else's.
  kill -9 -- -"${PIDS[$i]}" 2>/dev/null || kill -9 "${PIDS[$i]}" 2>/dev/null || true
  wait "${PIDS[$i]}" 2>/dev/null || true          # reap: a zombie still answers `kill -0`
  pkill -9 -u "$(id -u)" -f -- "vllm serve .*--port $port( |\$)" 2>/dev/null || true
  for t in $(seq 1 24); do                        # wait for the port AND its GPU memory to free
    curl -s --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 || break
    sleep 5
  done
  sleep 15
  _launch_replica "$i"
  _wait_healthy "$i" 120 || { echo "[sim] FATAL: replica $i did not come back"; return 1; }
  echo "[sim] replica $i RECOVERED on :$port"
  _sim_status "replica $i RECOVERED (restart ${RESTARTS[$i]})"
  return 0
}

# --- supervise, and exit when the consumer says it is done ---------------------------------------
# This script otherwise runs forever and holds its GPUs after the consumer has finished. If
# SIM_DONE_FILE is set, run_tau2_passk.sh / run_tau2_pi_utility.sh (given the same path) create it on
# exit, and the simulator then releases its GPUs.
SENTINEL=${SIM_DONE_FILE:-}
[ -n "$SENTINEL" ] && echo "[sim] watching for done file: $SENTINEL"

# Repair beats teardown: restarting one replica costs the consumer ~3 min of failed episodes on that
# port, against hours of silent 25% loss (or, with the old `exit 1`, losing every arm outright).
while :; do
  if [ -n "$SENTINEL" ] && [ -e "$SENTINEL" ]; then
    echo "[sim] done file present -- consumer finished, releasing GPUs"
    exit 0   # the EXIT trap stops every replica this script started
  fi
  for i in $(seq 0 $(( SIM_REPLICAS - 1 ))); do
    if _replica_bad "$i"; then
      _restart_replica "$i" || exit 1
    fi
  done
  sleep "${SIM_WATCH_INTERVAL:-60}"
done

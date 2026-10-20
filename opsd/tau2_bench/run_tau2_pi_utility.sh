#!/usr/bin/env bash
# =============================================================================================
# τ² PI UTILITY — Δpass@k between hint-guided and no-hint generation.
#
# Forked from run_tau2_passk.sh, whose no-hint corpus the DEFAULT setup measures against. The fork is
# deliberate and the shared parts must NOT drift: the Δ is only interpretable if the hinted arms sample
# the SAME environment as the baseline. The DEFAULTS reproduce the original retail-base measurement
# against that pass@k corpus:
#
#     split retail-base | profile paper (T=1.0/top_p 1.0/top_k -1/min_p 0) | max-steps 60
#     max-tokens 8000   | episode-timeout 1200 | evaluation-type ALL (default)
#     judge openai/gpt-4.1 (baseline.py default endpoint) | SIMULATOR = self-hosted Qwen3.5-35B-A3B @ T=0
#
# That setup needs retail-base hint files and a task list, NEITHER of which is distributed. The
# PUBLISHED inputs are keyed by training-pool `retail_N` ids, i.e. split retail-areal-rl: the hint trees
# under $OPSD_DATA_ROOT/tau2/hints/t211/ and the task pools under $OPSD_DATA_ROOT/tau2/splits/
# (matched132_tasks.json, pi_m75/tasks_all75.json). The paper's utility matrix
# was sampled on that split with the hosted
# gpt-5.6-luna simulator (T=0, reasoning_effort none) on BOTH the hinted and the nohint arms, n=4,
# max-tokens 4096 for the Instruct students / 8000 for the Thinking student:
#
#   MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 POLICY_ENABLE_THINKING=false UTIL_MAX_TOKENS=4096 \
#   UTIL_SPLIT=retail-areal-rl HINTS_DIR=$OPSD_DATA_ROOT/tau2/hints/t211 \
#   TASK_IDS_PATH=$OPSD_DATA_ROOT/tau2/splits/matched132_tasks.json \
#   SIMULATOR_MODEL=gpt-5.6-luna SIMULATOR_BASE_URL=${OPENAI_BASE_URL:-https://api.openai.com/v1} \
#   SIM_API_KEY_ENV=OPENAI_API_KEY \
#     bash opsd/tau2_bench/run_tau2_pi_utility.sh
#
# HINTS_DIR holds the arms' hint files at the relative paths named in UTIL_ARMS (default layout:
# glm/hints_base/*.json, qwen/hints_base/*.json). TASK_IDS_PATH (default $HINTS_DIR/tasks_utility.json)
# is the JSON list of task ids every arm runs on; UTIL_SPLIT must be the split those ids belong to.
# Every arm's hint file must cover every listed task with a non-empty hint (checked before serving):
# the published qwen3.7-plus trees cover 131 of matched132's 132 tasks, so with that list either keep
# only the glm_* arms in UTIL_ARMS or pass a list every arm covers. Prerequisites and simulator options
# are the same as run_tau2_passk.sh. This script starts POLICY_REPLICAS vLLM servers on the GPUs in
# CUDA_VISIBLE_DEVICES (default 0..7) and stops exactly those servers when it exits.
#
# ⚠️ THE SIMULATOR MUST BE THE SAME ON BOTH SIDES OF THE Δ. It drives the conversation -- swapping it
# changes pass@k far more than any hint does, and the Δ would be measuring a simulator swap with a
# hint-shaped label on it. The default retail-base setup pairs with a 35B-simulator pass@k corpus, so it
# must stay on the 35B (gpt-5.6-luna is the better simulator, but using it there means regenerating the
# no-hint corpus too); the paper's matrix used luna for every arm.
#
# ⚠️ ONE POLICY POOL FOR ALL ARMS. The arms loop INSIDE this script rather than being separate
# invocations. Re-invoking a serve-then-run driver per arm races on the vLLM ports: the second
# invocation finds them taken, silently comes up with 1 replica instead of 8, and the run projects to
# ~8x the wall-clock while looking healthy. Serve once, loop the arms.
#
# ARMS. In the default retail-base setup `nohint` at n=4 is a DRIFT CHECK, not the baseline: the primary
# Δ uses the n=128 pass@k corpus, which is 32x tighter. If this arm's pass@4 disagrees with the corpus's,
# the environment moved and every Δ in the run is suspect -- that is exactly what it exists to reveal.
# The published matrix instead pairs each hinted arm with this grid's own nohint arm (see the
# `definition` field of utility_delta.json).
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
ROOT=${ROOT:-$(_find_root)} || { echo "[util] FATAL: cannot locate the repo root" >&2; exit 1; }
# Relative paths given by the caller (OUT_DIR, RUN_ROOT, HINTS_DIR, ...) are relative to WHERE THEY RAN
# this script, not to the repo root we cd into for `python -m`.
_CALLER_PWD=$PWD
_abs() { case "$1" in /*|"") printf '%s' "$1" ;; *) printf '%s' "$_CALLER_PWD/$1" ;; esac; }
[ -n "${SIM_DONE_FILE:-}" ] && SIM_DONE_FILE=$(_abs "$SIM_DONE_FILE")
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"  # `opsd` = this checkout's package
PY=${PY:-python}

MODEL_PATH=${MODEL_PATH:?MODEL_PATH must be set}
UTIL_SPLIT=${UTIL_SPLIT:-retail-base}
UTIL_TRIALS=${UTIL_TRIALS:-4}
UTIL_PROFILE=${UTIL_PROFILE:-paper}
UTIL_CONCURRENCY=${UTIL_CONCURRENCY:-128}
UTIL_MAX_TOKENS=${UTIL_MAX_TOKENS:-8000}
UTIL_MAX_STEPS=${UTIL_MAX_STEPS:-60}
UTIL_EPISODE_TIMEOUT=${UTIL_EPISODE_TIMEOUT:-1200}
UTIL_ORDER=${UTIL_ORDER:-trial}
POLICY_REPLICAS=${POLICY_REPLICAS:-8}
POLICY_TP=${POLICY_TP:-1}
POLICY_PORT=${POLICY_PORT:-8300}
POLICY_MAX_LEN=${POLICY_MAX_LEN:-40000}
POLICY_GPU_UTIL=${POLICY_GPU_UTIL:-0.90}
# Re-use the same OUT_DIR to resume: baseline.py resumes each arm from its trajectories.jsonl.
RUN_DIR=${RUN_DIR:-${RUN_ROOT:-./runs}/${RUN_NAME:-tau2_pi_utility}}
OUT_DIR=$(_abs "${OUT_DIR:-$RUN_DIR/pi_utility}")
# `<arm>:<relative hints path>`; `nohint` takes no path. Space-separated.
# ':' NOT '=' AS THE INNER SEPARATOR, so a spec survives launchers that split KEY=VALUE pairs on '='.
UTIL_ARMS=${UTIL_ARMS:-"nohint glm_raw:glm/hints_base/hints_raw.json glm_raw_no_think:glm/hints_base/hints_raw_no_think.json glm_actions:glm/hints_base/hints_actions.json qwen37_raw:qwen/hints_base/hints_raw.json qwen37_raw_no_think:qwen/hints_base/hints_raw_no_think.json qwen37_actions:qwen/hints_base/hints_actions.json"}
# Arms may also be `+`-separated (for launchers that cannot pass SPACES) and are normalised here.
# Only `+` -- never `_`, which would split `raw_no_think` into two bogus arms.
UTIL_ARMS=$(printf %s "$UTIL_ARMS" | tr '+' ' ')
# Smoke knob: keep only the first N pooled tasks. 0 = the whole pool.
UTIL_MAX_TASKS=${UTIL_MAX_TASKS:-0}

: "${SIMULATOR_BASE_URL:?SIMULATOR_BASE_URL must be set (serve_user_sim.sh address, or a hosted endpoint)}"
: "${SIMULATOR_MODEL:?SIMULATOR_MODEL must be set}"
: "${HINTS_DIR:?HINTS_DIR must be set (local dir holding the UTIL_ARMS hint files, e.g. glm/ qwen/)}"
# The retail NL judge (openai/gpt-4.1) runs inside the reward of τ²'s NATIVE retail tasks only (40 of
# retail-base's 114, 29 of retail-train's 74; `prepare_data audit`) and needs a key there. The AReaL
# pools (*-areal-rl) are judge-free, so a self-hosted-simulator run on them needs no key at all.
case "$UTIL_SPLIT" in
  retail-base|retail-train|all-base)
    : "${OPENAI_API_KEY:?set OPENAI_API_KEY (key for the retail NL judge, openai/gpt-4.1)}" ;;
esac
mkdir -p "$OUT_DIR"

# --- check the hints + the task pool, and VALIDATE BEFORE SERVING ---------------------------------
# Every failure below is cheap now and expensive later: a missing hint file discovered after the pool
# is up has already burned the replica-load time, and one discovered mid-loop leaves a half-measured
# grid whose arms are not comparable.
HINTS_DIR=$(_abs "$HINTS_DIR")
[ -d "$HINTS_DIR" ] || { echo "[util] FATAL: HINTS_DIR=$HINTS_DIR is not a directory" >&2; exit 1; }
TASKS_JSON=$(_abs "${TASK_IDS_PATH:-$HINTS_DIR/tasks_utility.json}")
[ -s "$TASKS_JSON" ] || { echo "[util] FATAL: task pool $TASKS_JSON missing/empty (set TASK_IDS_PATH)" >&2; exit 1; }

# Expand the pinned pool into repeated --task args. EVERY arm gets this list, nohint included: the
# hinted side is protected by --require-hint, but nothing would pin the no-hint side.
TASK_ARGS=()
while IFS= read -r t; do TASK_ARGS+=(--task "$t"); done < <($PY -c "
import json,sys
for t in json.load(open('$TASKS_JSON')): print(t)
")
NTASK=$(( ${#TASK_ARGS[@]} / 2 ))
[ "$NTASK" -gt 0 ] || { echo "[util] FATAL: empty task pool" >&2; exit 1; }
if [ "$UTIL_MAX_TASKS" -gt 0 ] && [ "$UTIL_MAX_TASKS" -lt "$NTASK" ]; then
  TASK_ARGS=("${TASK_ARGS[@]:0:$(( UTIL_MAX_TASKS * 2 ))}")
  echo "[util] ⚠️ SMOKE: capped to the first $UTIL_MAX_TASKS of $NTASK tasks -- Δ from this run is NOT" \
       "the measurement, it only proves the plumbing"
  NTASK=$UTIL_MAX_TASKS
fi
echo "[util] task pool: $NTASK tasks"

# Validate every arm's hint file covers every pooled task with a NON-EMPTY hint. --require-hint would
# catch a gap at run time, but only after the pool is served and only for the arm that hits it; and it
# does NOT catch blank values (`if hint:` in episode.py skips them, so a "" entry runs UNHINTED while
# counting as covered).
for spec in $UTIL_ARMS; do
  arm=${spec%%:*}; rel=${spec#*:}
  [ "$arm" = "nohint" ] && continue
  f="$HINTS_DIR/$rel"
  [ -s "$f" ] || { echo "[util] FATAL: arm $arm -- hint file missing: $f" >&2; exit 1; }
  $PY - "$f" "$TASKS_JSON" "$arm" <<'PYEOF' || exit 1
import json, sys
hints, tasks, arm = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
def text(v):
    return v.get("task") if isinstance(v, dict) else v
ok = {str(k) for k, v in hints.items()
      if not str(k).startswith("__") and isinstance(text(v), str) and text(v).strip()}
bad = [t for t in tasks if t not in ok]
if bad:
    raise SystemExit(f"[util] FATAL: arm {arm} -- {len(bad)} pooled task(s) have no usable hint: {bad[:5]}")
print(f"[util]   {arm}: {len(tasks)}/{len(tasks)} pooled tasks hinted")
PYEOF
done
echo "[util] all arms validated -- serving the policy"

# --- serve the policy: one pool, shared by every arm ---------------------------------------------
# GPU indices are taken from CUDA_VISIBLE_DEVICES when set, else 0..NUM_GPUS_PER_NODE-1.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=, read -r -a _GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  _GPUS=($(seq 0 $(( ${NUM_GPUS_PER_NODE:-8} - 1 ))))
fi
GPUS_TOTAL=${#_GPUS[@]}
if [ $(( POLICY_REPLICAS * POLICY_TP )) -gt "$GPUS_TOTAL" ]; then
  echo "[util] FATAL: ${POLICY_REPLICAS} replicas x TP=${POLICY_TP} > ${GPUS_TOTAL} GPUs" >&2; exit 1
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
    echo "[util] FATAL: port $port is already in use; set POLICY_PORT to a free range" >&2; exit 1
  fi
done
# Resolve the repo ONCE, not once per replica: ~32 replicas launched together previously blew the HF
# account's 1000-req/5-min quota and every replica 429'd, then sat until POLICY_WAIT_SECS.
if [ "${HF_PREFETCH:-1}" = "1" ]; then
  echo "[util] pre-resolving $MODEL_PATH"
  if $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_PATH', max_workers=8)
"; then export HF_HUB_OFFLINE=1; echo "[util] prefetch ok -> HF_HUB_OFFLINE=1"
  else echo "[util] WARNING: prefetch failed; replicas will each hit the HF API (429 risk)" >&2; fi
fi

POLICY_URLS=""
for i in $(seq 0 $(( POLICY_REPLICAS - 1 ))); do
  port=$(( POLICY_PORT + i ))
  first=$(( i * POLICY_TP ))
  devs=$(IFS=,; echo "${_GPUS[*]:$first:$POLICY_TP}")
  echo "[util] replica $i: GPUs $devs -> :$port"
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

echo "[util] waiting for $POLICY_REPLICAS policy replica(s)…"
deadline=$(( $(date +%s) + ${POLICY_WAIT_SECS:-1200} ))
for i in $(seq 0 $(( POLICY_REPLICAS - 1 ))); do
  port=$(( POLICY_PORT + i ))
  until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    { [ "$(date +%s)" -lt "$deadline" ] && kill -0 "${POLICY_PIDS[$i]}" 2>/dev/null; } || {
      echo "[util] FATAL: replica $i never became healthy (or exited)" >&2
      tail -30 "$OUT_DIR/policy_$i.log" >&2; exit 1; }
    sleep 10
  done
  echo "[util]   replica $i healthy"
done

# The local engines serve without --api-key; a stray VLLM_API_KEY makes the client send an
# Authorization header the server rejects. This has bitten every local τ run.
unset VLLM_API_KEY

SIM_ARGS=(--user-llm "$SIMULATOR_MODEL" --user-base-url "$SIMULATOR_BASE_URL")
[ -n "${SIM_API_KEY_ENV:-}" ] && SIM_ARGS+=(--user-api-key-env "$SIM_API_KEY_ENV")
[ -n "${SIM_REASONING_EFFORT:-}" ] && SIM_ARGS+=(--user-reasoning-effort "$SIM_REASONING_EFFORT")
THINK_ARGS=()
if [ "${POLICY_ENABLE_THINKING:-true}" = "false" ]; then
  THINK_ARGS+=(--no-thinking)
else
  THINK_ARGS+=(--preserve-thinking)
fi

echo "[util] model=$MODEL_PATH split=$UTIL_SPLIT trials=$UTIL_TRIALS profile=$UTIL_PROFILE"
echo "[util] sim=$SIMULATOR_MODEL @ $SIMULATOR_BASE_URL"
echo "[util] arms: $UTIL_ARMS"

WORST=0
for spec in $UTIL_ARMS; do
  arm=${spec%%:*}; rel=${spec#*:}
  ARM_OUT="$OUT_DIR/$arm"; mkdir -p "$ARM_OUT"
  HINT_ARGS=()
  if [ "$arm" != "nohint" ]; then
    # --require-hint: refuse rather than blend hinted and unhinted episodes into one condition.
    HINT_ARGS=(--hints-path "$HINTS_DIR/$rel" --require-hint)
  fi

  # Per-arm resume is local and automatic: baseline.py skips (domain, task_id, trial) already done.
  echo "########## [util] arm=$arm  $(date -u +%H:%M:%S) ##########"
  set +e
  $PY -m opsd.tau2_bench.baseline \
    --split "$UTIL_SPLIT" \
    --model "$MODEL_PATH" --policy-base-url "$POLICY_URLS" \
    "${SIM_ARGS[@]}" "${THINK_ARGS[@]}" "${HINT_ARGS[@]}" "${TASK_ARGS[@]}" \
    --profile "$UTIL_PROFILE" \
    --max-tokens "$UTIL_MAX_TOKENS" --max-steps "$UTIL_MAX_STEPS" \
    --episode-timeout "$UTIL_EPISODE_TIMEOUT" \
    --num-trials "$UTIL_TRIALS" --concurrency "$UTIL_CONCURRENCY" \
    --episode-order "$UTIL_ORDER" \
    --output-dir "$ARM_OUT" --yes-really "$@" > "$ARM_OUT/run.log" 2>&1
  RC=$?
  set -e
  echo "[util] arm=$arm exited rc=$RC  ($(wc -l < "$ARM_OUT/trajectories.jsonl" 2>/dev/null || echo 0) episodes)"
  [ "$RC" -eq 0 ] || { tail -25 "$ARM_OUT/run.log" >&2; WORST=$RC; }
done

echo "[util] all arms done; worst rc=$WORST"

# The EXIT trap stops the policy servers and creates $SIM_DONE_FILE (if set).
exit $WORST

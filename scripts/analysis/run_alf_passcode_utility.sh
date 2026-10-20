#!/usr/bin/env bash
# =============================================================================
# ALFWorld-Passcode in-context PI UTILITY (the ALFWorld-Passcode cheating study).
#
#   utility = success(with hint) - success(no hint), paired per task, where success = CORRECT KEY
#   SUBMITTED. On this env completing the task only REVEALS the key and only the key is scored, so a
#   hint that leaks the key can pay without the task being done. That makes utility's blind spot
#   measurable. task_completed / honest_success / cheat_success (env metrics) keep the two paths apart.
#
#   student    Qwen3-4B-Instruct-2507 (the OPSD Instruct arms' student)
#   tasks      the 200-task utility pool (alfworld/passcode/task_ids_util200.json, all inside train1600)
#   PIs        pc_key pc_rawkey pc_rawwrong pc_wrongraw (build_alfworld_passcode_hints.py) + nohint
#   placement  user   = hint appended to the first user turn (the established utility definition)
#              system = the OPSD teacher's <Secret information> block as a system turn, i.e. exactly
#                       the prompt the frozen teacher scores the student's rollouts with
#   env        the OPSD Instruct TRAINING contract (THOUGHTS: prefix, think_tags_in_prompt, no
#              admissible list, invalid-action feedback, 30 turns, 32k input, 8k/turn, no stop), passcode on.
#              The older utility runs used a different prompt; this one must match what is trained.
#   sampling   T 1.0 / top-p 0.95 / top-k 20 / presence 1.5 (the PI-utility profile)
#
#   TRAIN_PARQUET=<alfworld parquet dir>/train.parquet bash scripts/analysis/run_alf_passcode_utility.sh
#   knobs: GPUS=0,1,2,3  N_SAMPLES=4  SLICE=48  PLACEMENTS="user system"  PIS="pc_key ..."  SMOKE=1
#   inputs: HINTS (default the bundle's alfworld/passcode/hints_v1; hints_v2 for the ao/act formats),
#           TASK_IDS, and the task rows: POOL_PARQUET, or TRAIN_PARQUET (alfworld_dataset.py output) from
#           which the TASK_IDS rows are cut into $OUT/pool.parquet. Single machine, local GPUs only.
#
# Resumable (--resume per shard). Uses this checkout's opsd.envs: the env must have passcode_mode.
# =============================================================================
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ "$PWD" != / ] && ! { [ -f pyproject.toml ] && [ -f opsd/__init__.py ] && [ -f opsd/skyrl_ext/__init__.py ]; }; do cd ..; done; pwd)"
cd "$REPO"
PY=${PY:-$REPO/.venv/bin/python}
VLLM=${VLLM:-$(dirname "$PY")/vllm}
# This checkout FIRST: an editable install elsewhere may point at a tree whose env has no
# passcode_mode (the preflight below checks which module is imported).
export PYTHONPATH="$REPO"
[ -n "${HF_HUB_OFFLINE:-}" ] && export HF_HUB_OFFLINE
: "${ALFWORLD_VENV:?set ALFWORLD_VENV (the ALFWorld server venv)}"
: "${ALFWORLD_DATA:?set ALFWORLD_DATA (the ALFWorld game data root)}"
export ALFWORLD_VENV ALFWORLD_DATA
# Each ALFWorld /reset copies a planner .so into $TMPDIR (see the env-pool note below): keep scratch on a
# disk with room.
export TMPDIR=${TMPDIR_OVERRIDE:-/var/tmp/$USER-passcode}; mkdir -p "$TMPDIR"
unset OPSD_HINTS_PATH   # generate_alfworld_local.py defaults --hints-path to it: nohint would get hints
OPSD_DATA_ROOT=${OPSD_DATA_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data}

MODEL=${MODEL:-"Qwen/Qwen3-4B-Instruct-2507"}; MODEL_NAME=${MODEL_NAME:-"Qwen3-4B-Instruct-2507"}
N_SAMPLES=${N_SAMPLES:-4}
GPUS=${GPUS:-"0,1,2,3"}
SLICE=${SLICE:-48}
SERVE_MAXLEN=${SERVE_MAXLEN:-40960}
GPU_UTIL=${GPU_UTIL:-0.85}
VLLM_BASE_PORT=${VLLM_BASE_PORT:-8723}
ENV_BASE=${ENV_BASE:-10300}
PLACEMENTS=${PLACEMENTS:-"user system"}
PIS=${PIS:-"pc_key pc_rawkey pc_rawwrong pc_wrongraw"}
SALT=${SALT:-alfworld-passcode-v1}
read -r T P K PP <<< "${SAMPLING:-1.0 0.95 20 1.5}"
HINTS=${HINTS:-$OPSD_DATA_ROOT/alfworld/passcode/hints_v1}
TASK_IDS=${TASK_IDS:-$OPSD_DATA_ROOT/alfworld/passcode/task_ids_util200.json}
OUT=${OUT:-outputs/passcode/utility_4bi}
if [ -n "${SMOKE:-}" ]; then N_SAMPLES=1; SLICE=4; OUT="$OUT-smoke"; fi
mkdir -p "$OUT"
if [ -z "${POOL_PARQUET:-}" ]; then
  # The TASK_IDS rows of the train parquet, in the parquet's own order (the order the study's pool file had).
  : "${TRAIN_PARQUET:?set POOL_PARQUET, or TRAIN_PARQUET to the ALFWorld train parquet (alfworld_dataset.py)}"
  POOL_PARQUET="$OUT/pool.parquet"
  $PY - "$TRAIN_PARQUET" "$TASK_IDS" "$POOL_PARQUET" <<'PY' || exit 1
import json, sys
import pandas as pd
src, ids, dst = sys.argv[1:4]
want = set(json.load(open(ids)))
df = pd.read_parquet(src)
sub = df[df["extra_info"].map(lambda e: e["task_id"] in want)].reset_index(drop=True)
assert len(sub) == len(want), f"{len(want) - len(sub)} task ids are not in {src}"
sub.to_parquet(dst)
print(f"pool: {len(sub)} rows -> {dst}")
PY
fi

IFS=',' read -ra GA <<< "$GPUS"; NSERVE=${#GA[@]}
NUM_ENV=$(( NSERVE * SLICE ))
mkdir -p "$OUT/serve_logs" "$OUT/logs"
LAYOUT="NSHARD=$NSERVE N_SAMPLES=$N_SAMPLES SAMPLING=$T/$P/$K/$PP"
if [ -s "$OUT/.layout" ]; then
  [ "$(cat "$OUT/.layout")" != "$LAYOUT" ] && { echo "FATAL: $OUT was written with [$(cat "$OUT/.layout")], this run is [$LAYOUT]; resuming across layouts duplicates rollouts."; exit 1; }
else
  echo "$LAYOUT" > "$OUT/.layout"
fi

echo "===== preflight ====="
$PY - "$REPO" "$HINTS" "$PIS" "$TASK_IDS" "$SALT" <<'PY' || exit 1
import json, sys
from pathlib import Path
repo, hints, pis, ids_path, salt = sys.argv[1], Path(sys.argv[2]), sys.argv[3].split(), sys.argv[4], sys.argv[5]
import opsd.envs.alfworld.passcode as P
assert P.__file__.startswith(repo), f"opsd.envs resolves to {P.__file__}, not this checkout {repo}"
man = json.loads((hints / "manifest.json").read_text())
assert man["salt"] == salt, f"hints built with salt {man['salt']!r}, env would use {salt!r} -- keys would not match"
ids = set(json.load(open(ids_path)))
bad = 0
for pi in pis:
    d = json.load(open(hints / f"hints_{pi}.json"))
    keys = {k for k in d if not k.startswith("__")}
    miss = ids - keys
    empty = sum(1 for k in ids & keys if not str(d[k]["task"] if isinstance(d[k], dict) else d[k]).strip())
    bad += bool(miss or empty)
    print(f"  {pi:<14} tasks={len(ids)} missing={len(miss)} empty={empty} {'OK' if not (miss or empty) else 'FAIL'}")
print(f"  env module {P.__file__}; salt OK")
sys.exit(1 if bad else 0)
PY
for port in $(seq "$VLLM_BASE_PORT" $((VLLM_BASE_PORT + NSERVE - 1))) $(seq "$ENV_BASE" $((ENV_BASE + NUM_ENV - 1))); do
  ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":$port\$" && { echo "FATAL: port $port is in use"; exit 1; }
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPUS" | while IFS=', ' read -r i m; do
  [ "$m" -gt 2000 ] && echo "WARN: GPU $i already has ${m} MiB in use"
done

launch_serve() {
  local gpu="$1" port="$2" attempt=0
  while [ "$attempt" -lt 4 ]; do
    CUDA_VISIBLE_DEVICES=$gpu VLLM_HOST_IP=127.0.0.1 NCCL_CUMEM_ENABLE=0 \
      "$VLLM" serve "$MODEL" --served-model-name "$MODEL_NAME" \
      --host 0.0.0.0 --port "$port" --tensor-parallel-size 1 --gpu-memory-utilization "$GPU_UTIL" \
      --max-model-len "$SERVE_MAXLEN" --trust-remote-code --enable-prefix-caching \
      > "$OUT/serve_logs/vllm_$port.log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 200); do
      curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { echo "$pid"; return 0; }
      kill -0 "$pid" 2>/dev/null || break
      sleep 5
    done
    kill -9 "$pid" 2>/dev/null; attempt=$((attempt+1)); sleep $((8 + RANDOM % 12))
  done; return 1
}
teardown() {
  [ -n "${KEEP_SERVES:-}" ] && return 0
  kill $(cat "$OUT/serve.pids" 2>/dev/null) 2>/dev/null
  kill $(tr ' ' '\n' < "$OUT/serve_logs/alfworld_serve.pids" 2>/dev/null) 2>/dev/null
  return 0
}
die() { echo "$@"; teardown; exit 1; }

# ---- env pool, RECYCLED BEFORE EVERY CONDITION ----
# Each ALFWorld /reset builds a new textworld PDDL env, and fast_downward.load_lib() copies the 34.7 MB
# libdownward.so into a TemporaryDirectory, dlopen()s it, then deletes the dir: the file is unlinked but
# still MAPPED, so its disk space stays pinned in $TMPDIR until the server process exits: ~5,600 resets
# pin ~194 GB, enough to fill a disk and fail a run mid-condition. Restarting the pool per condition bounds the pinned
# space to one condition's resets (<= ~28 GB). Logs of each pool generation are kept under
# serve_logs/pool_<n>/ so the collision audit still covers every condition.
POOL_GEN=0
start_pool() {
  POOL_GEN=$((POOL_GEN + 1))
  POOL=$( NUM_ALFWORLD_SERVERS="$NUM_ENV" ALFWORLD_BASE_PORT="$ENV_BASE" ALFWORLD_MAX_STEPS=30 \
      ALFWORLD_ADVERTISE_HOST=127.0.0.1 ALFWORLD_LOG_DIR="$OUT/serve_logs" \
      bash opsd/alfworld/server/launch_alfworld_servers.sh 2>>"$OUT/serve_logs/pool.log" | tail -1 )
  mapfile -t POOL_URLS < <(echo "$POOL" | $PY -c 'import sys,json;[print(u) for u in json.load(sys.stdin)]' 2>/dev/null)
  [ "${#POOL_URLS[@]}" -lt "$NUM_ENV" ] && return 1
  local n i
  for i in $(seq 1 120); do
    n=0; for u in "${POOL_URLS[@]}"; do curl -sf -m 2 "$u/health" >/dev/null 2>&1 && n=$((n+1)); done
    [ "$n" -ge "$NUM_ENV" ] && return 0
    sleep 5
  done
  echo "  env pool: only $n/$NUM_ENV servers healthy"; return 1
}
stop_pool() {
  kill $(tr ' ' '\n' < "$OUT/serve_logs/alfworld_serve.pids" 2>/dev/null) 2>/dev/null
  local i
  for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -qE ":$ENV_BASE\b" || break; sleep 2; done
  mkdir -p "$OUT/serve_logs/pool_$POOL_GEN"
  mv "$OUT"/serve_logs/alfworld_serve_*.log "$OUT/serve_logs/pool_$POOL_GEN/" 2>/dev/null
  return 0
}
echo "===== env pool ($NUM_ENV servers from :$ENV_BASE; recycled per condition) ====="
start_pool || die "FATAL: env pool ${#POOL_URLS[@]}/$NUM_ENV"

SERVE_URLS=(); : > "$OUT/serve.pids"
for idx in "${!GA[@]}"; do
  port=$((VLLM_BASE_PORT+idx))
  pid=$(launch_serve "${GA[$idx]}" "$port") || die "FATAL: serve :$port failed"
  SERVE_URLS+=("http://127.0.0.1:$port"); printf '%s\n' "$pid" >> "$OUT/serve.pids"
  echo "  serve :$port healthy on GPU ${GA[$idx]} [$((idx+1))/$NSERVE]"
done

gen_shard() {  # cond placement pi serve slice_csv off lim name
  local cond="$1" placement="$2" pi="$3" serve="$4" slice_csv="$5" off="$6" lim="$7" name="$8" attempt=0
  local outdir="$OUT/$cond/$name" log="$OUT/logs/${cond//\//-}-$name.log"
  local hint_args=()
  [ "$pi" != "nohint" ] && hint_args=(--hints-path "$HINTS/hints_$pi.json" --hint-placement "$placement")
  mkdir -p "$outdir"
  while [ "$attempt" -lt 3 ]; do
    $PY opsd/alfworld/generate_alfworld_local.py \
      --model-url "$serve" --served-model-name "$MODEL_NAME" --tokenizer "$MODEL" \
      --data-file "$POOL_PARQUET" --n-samples "$N_SAMPLES" --offset "$off" --limit "$lim" \
      --task-ids-file "$TASK_IDS" "${hint_args[@]}" \
      --prompt-style react --no-show-admissible --invalid-action-feedback --no-invalid-action-show-admissible \
      --history-length 0 --max-turns 30 --max-input-length 32000 --max-generate-length 8000 \
      --temperature "$T" --top-p "$P" --top-k "$K" --min-p 0.0 \
      --presence-penalty "$PP" --repetition-penalty 1.0 --no-stop \
      --no-enable-thinking --no-strict-action-outside-think \
      --reasoning-style prefix --reasoning-prefix "THOUGHTS:" --think-tags-in-prompt --reward-scale 0.1 \
      --passcode --passcode-salt "$SALT" \
      --server-urls "$slice_csv" --max-env-workers "$SLICE" --request-workers "$SLICE" \
      --resume --disable-tqdm --output-dir "$outdir" >> "$log" 2>&1 && return 0
    attempt=$((attempt+1)); echo "  [$cond/$name] attempt $attempt failed; retrying" | tee -a "$log"
    sleep $((10 + RANDOM % 15))
  done; return 1
}

NROWS=$($PY -c "import pandas as pd;print(len(pd.read_parquet('$POOL_PARQUET')))")
NT=$($PY -c "import json;print(len(json.load(open('$TASK_IDS'))))")
[ -n "${SMOKE:-}" ] && { NROWS=8; NT=8; }
chunk=$(( (NROWS + NSERVE - 1) / NSERVE ))
# WITH_NOHINT=0 skips the baseline: for adding PI conditions in a NEW output dir (e.g. on a different GPU
# count, which the layout guard forbids in the old one) against a baseline COPIED into "$OUT/nohint/" first.
CONDS=""
[ "${WITH_NOHINT:-1}" = "1" ] && CONDS="nohint"
for placement in $PLACEMENTS; do for pi in $PIS; do CONDS="$CONDS $placement/$pi"; done; done
if [ "${WITH_NOHINT:-1}" != "1" ] && ! ls "$OUT"/nohint/sh*/trajectories.jsonl >/dev/null 2>&1; then
  die "FATAL: WITH_NOHINT=0 needs a baseline copied into $OUT/nohint/sh*/trajectories.jsonl"
fi
FIRST_COND=1
for cond in $CONDS; do
  if [ "$cond" = "nohint" ]; then placement=user; pi=nohint; else placement=${cond%%/*}; pi=${cond#*/}; fi
  if [ "$FIRST_COND" != 1 ]; then stop_pool; start_pool || die "FATAL: env pool restart failed before $cond"; fi
  FIRST_COND=0
  echo "===== [$cond] $NT tasks x $N_SAMPLES ($NSERVE shards, chunk=$chunk) $(date +%H:%M:%S) ====="
  pids=()
  for i in "${!SERVE_URLS[@]}"; do
    off=$(( i * chunk )); [ "$off" -ge "$NROWS" ] && continue
    lim=$chunk; [ $((off+lim)) -gt "$NROWS" ] && lim=$(( NROWS - off ))
    slice_csv=$(printf '%s,' "${POOL_URLS[@]:$((i*SLICE)):$SLICE}"); slice_csv=${slice_csv%,}
    gen_shard "$cond" "$placement" "$pi" "${SERVE_URLS[$i]}" "$slice_csv" "$off" "$lim" "sh$i" & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || echo "  WARN: a $cond shard failed"; done
  GOT=$(cat "$OUT/$cond"/sh*/trajectories.jsonl 2>/dev/null | wc -l)
  echo "  [$cond] -> $GOT/$(( NT * N_SAMPLES )) $(date +%H:%M:%S)"
done

BAD=$(grep -rl --include='alfworld_serve_*.log' "step() called before reset" "$OUT/serve_logs" 2>/dev/null | wc -l)
NOSPACE=$(grep -rl --include='alfworld_serve_*.log' "No space left on device" "$OUT/serve_logs" 2>/dev/null | wc -l)
echo "===== env-server audit: collisions $BAD, ENOSPC $NOSPACE (both must be 0) ====="

# One aggregation root per placement, each holding its own COPY of nohint (aggregate_utility_delta
# globs <root>/<cond>/**/*.jsonl, so every condition must sit as a directory under the same root).
for placement in $PLACEMENTS; do
  R="$OUT/agg_$placement"; rm -rf "$R"; mkdir -p "$R/nohint"
  cat "$OUT"/nohint/sh*/trajectories.jsonl > "$R/nohint/all.jsonl"
  for pi in $PIS; do mkdir -p "$R/$pi"; cat "$OUT/$placement/$pi"/sh*/trajectories.jsonl > "$R/$pi/all.jsonl"; done
  $PY opsd/alfworld/aggregate_utility_delta.py \
    --root "$R" --baseline nohint --conditions nohint $PIS \
    --title "ALFWorld-Passcode utility ($NT train tasks), $MODEL_NAME, hint placement=$placement, N=$N_SAMPLES" \
    && echo "  -> $R/utility_delta.md"
done
$PY scripts/analysis/alf_passcode_report.py --utility-root "$OUT" --placements $PLACEMENTS --pis $PIS \
  --hints "$HINTS" --out "$OUT/passcode_report.md" && echo "  -> $OUT/passcode_report.md"
teardown
echo "DONE -> $OUT  ($(date +%H:%M:%S))"

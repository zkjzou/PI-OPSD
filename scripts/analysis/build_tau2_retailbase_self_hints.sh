#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build the SELF cells of the tau2 retail-base hint grid: {anlz,summ} x {4bi,4bt,30bi} = 6 cells.
#
# WHY. retail-base had 12 of the 14 PI formats; the two missing were anlz_self / summ_self. The only
# self arms that existed (pi_tau2_hints_self/*_4bi.json) are in the retail_N namespace and join
# retail-base at 0%, and were 4bi-only.
#
# WHAT "SELF" MEANS HERE -- the AUTHOR axis only. Read off the existing arms' own meta:
#   anlz_glm / anlz_qwn BOTH carry src=.../pi_tau2_retailbase/glm/hints_base/hints_raw.json
# i.e. all three authors read the SAME glm-5.2 reference trajectory and only the summariser changes.
# So `self` = the student writes the hint about glm's trajectory. No self-rollouts are involved; this
# is what keeps it a single-variable contrast against anlz_glm/anlz_qwn.
#
# --max-chars 40000. The flag is REQUIRED by the generator and the value the remote glm/qwn arms used
# is not recorded in their meta and no launcher survives, so it cannot be matched by construction.
# Measured on this pool (n=108, p50 17,598 / p90 28,805 / max 38,141 chars): 24000 elides 25% of
# tasks, 32000 elides 5%, 40000 elides 0%. 40000 is chosen so the self arms see the COMPLETE
# trajectory -- eliminating elision as a confounder rather than guessing at one -- and 40000 chars is
# ~10k tokens, well inside each local server's 40k-token context.
#
# MAX_TOKENS=8192, not the 4096 default: 4bt is a thinking model and spends completion budget on
# reasoning before emitting the hint; 1024 once produced ZERO usable hints on long trajectories.
#
# INPUTS. SRC = the retail-base glm-5.2 hints_raw.json (bare-int retail-base ids; retail-base hint
# artifacts are not part of the data bundle, so it must be supplied). 4bi is expected to be served
# already on P4BI_A / P4BI_B (default 8320 / 8321) under its HF id -- the 4bi cells are SKIPPED (WARN)
# when /v1/models reports any other name. Two TP=1 replicas on those ports, e.g.:
#   GPUS=0,4 BASE_PORT=8320 MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
#     bash scripts/analysis/serve_qwen3_4b_instruct_pool.sh       # stop: kill $(cat <state>/serve_q4bi_pool.pids)
# (or `ONLY=4bi bash serve_tau2_m75.sh` -- one replica on :8350 -- with P4BI_A=8350 P4BI_B=8350).
# 4bt and 30bi are launched here on GPU_4BT_A/B and GPU_30BI_A/B and only those servers are stopped at
# the end. Output: SELF_OUT (default $OUT_ROOT/pi_tau2_hints_rb_self); logs: $OUT_ROOT/rb_self_logs.
#   SRC=<retail-base glm hints_raw.json> bash scripts/analysis/build_tau2_retailbase_self_hints.sh
# ---------------------------------------------------------------------------
set -x
. "$(dirname "${BASH_SOURCE[0]}")/_tau2_env.sh"
G=$REPO_ROOT/scripts/analysis/gen_hints_local_models.py
SRC=${SRC:?set SRC to the retail-base glm-5.2 hints_raw.json}
OUT=${SELF_OUT:-$OUT_ROOT/pi_tau2_hints_rb_self}
LOG=$OUT_ROOT/rb_self_logs
P4BI_A=${P4BI_A:-8320}; P4BI_B=${P4BI_B:-8321}
GPU_4BT_A=${GPU_4BT_A:-1}; GPU_4BT_B=${GPU_4BT_B:-2}
GPU_30BI_A=${GPU_30BI_A:-3,5}; GPU_30BI_B=${GPU_30BI_B:-6,7}
mkdir -p "$OUT" "$LOG"
cd "$REPO_ROOT"
unset VLLM_API_KEY

[ -f "$SRC" ] || { echo "FATAL source artifact missing: $SRC"; exit 1; }

MAXCHARS=40000
MAXTOK=8192

# --- assert a port serves the EXACT model before spending anything on it. The hard-won lesson: a
# --- watchdog pinned to the wrong name once replaced 4bt's replicas with 4bi, and only an exact
# --- model-name mismatch caught it before mislabelled hints were written.
assert_model () {  # $1=port $2=expected model id
  got=$(curl -s --max-time 10 "http://127.0.0.1:$1/v1/models" \
        | $PY -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
  [ "$got" = "$2" ] || { echo "FATAL :$1 serves '$got', expected '$2'"; return 1; }
  echo "[assert] :$1 -> $2 OK"
}

# The 30bi servers below are TP=2: same single-host NCCL workarounds as serve_tau2_m75.sh (TCP
# transport, no cuMem allocator, libfabric FI_* cleared); KEEP_NCCL_ENV=1 keeps your own settings.
if [ "${KEEP_NCCL_ENV:-0}" != 1 ]; then
  export NCCL_NET=Socket NCCL_CUMEM_ENABLE=0
  for _v in $(compgen -e); do case "$_v" in FI_*) unset "$_v" ;; esac; done
fi
SERVE_PIDS=()
serve () {  # $1=port $2=model $3=tp $4=cuda_visible_devices
  CUDA_VISIBLE_DEVICES="$4" setsid nohup $PY -m vllm.entrypoints.openai.api_server \
    --model "$2" --served-model-name "$2" --port "$1" --host 0.0.0.0 \
    --tensor-parallel-size "$3" --max-model-len 40000 --gpu-memory-utilization 0.90 \
    --enable-prefix-caching --trust-remote-code \
    > "$LOG/serve_${1}.log" 2>&1 < /dev/null &
  SERVE_PIDS+=("$!")
  echo "[serve] :$1 $2 tp=$3 gpus=$4 pid=$!"
}

wait_healthy () {  # $1..=ports ; generous grace -- a premature kill cost 44 restarts once
  for p in "$@"; do
    for t in $(seq 1 90); do
      [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 "http://127.0.0.1:$p/health" 2>/dev/null)" = "200" ] && { echo "[health] :$p up (~$((t*10))s)"; break; }
      sleep 10
    done
  done
}

cell () {  # $1=port $2=author-model $3=style $4=outfile-tag
  case "$3" in onpolicy_skill) pre=anlz ;; plain) pre=summ ;; esac
  o="$OUT/${pre}_self_${4}.json"
  if [ -f "$o" ]; then echo "[skip] $o exists"; return 0; fi
  $PY "$G" --env tau2 --input-hints "$SRC" \
      --url "http://127.0.0.1:$1" --model "$2" \
      --prompt-style "$3" --outcome success \
      --max-chars $MAXCHARS --max-tokens $MAXTOK --temperature 0.7 --concurrency 16 \
      --retries 4 --timeout 900 --out-schema opsd_v2 --out "$o" \
      > "$LOG/${pre}_self_${4}.log" 2>&1
  echo "[cell] ${pre}_self_${4} rc=$? -> $o"
}

# ---------- 4bi: replicas already serving on P4BI_A / P4BI_B ----------
M4BI=Qwen/Qwen3-4B-Instruct-2507
if assert_model "$P4BI_A" "$M4BI" && assert_model "$P4BI_B" "$M4BI"; then
  cell "$P4BI_A" "$M4BI" onpolicy_skill 4bi &
  cell "$P4BI_B" "$M4BI" plain          4bi &
  wait
else
  echo "WARN 4bi replicas not as expected; skipping 4bi cells"
fi

# ---------- 4bt TP=1 x2, 30bi TP=2 x2 (default gpu1, gpu2, gpu3+5, gpu6+7) ----------
M4BT=Qwen/Qwen3-4B-Thinking-2507
M30=Qwen/Qwen3-30B-A3B-Instruct-2507
serve 8330 "$M4BT" 1 "$GPU_4BT_A"
serve 8331 "$M4BT" 1 "$GPU_4BT_B"
serve 8340 "$M30"  2 "$GPU_30BI_A"
serve 8341 "$M30"  2 "$GPU_30BI_B"
wait_healthy 8330 8331 8340 8341

# NOTE: `wait` with no args waits for EVERY background child -- including the four `serve` jobs,
# which run until killed. That hangs the script before teardown (hit 2026-09-10). Wait on the CELL
# pids only.
CELLPIDS=""
if assert_model 8330 "$M4BT" && assert_model 8331 "$M4BT"; then
  cell 8330 "$M4BT" onpolicy_skill 4bt & CELLPIDS="$CELLPIDS $!"
  cell 8331 "$M4BT" plain          4bt & CELLPIDS="$CELLPIDS $!"
else echo "WARN 4bt assertion failed; skipping 4bt cells"; fi
if assert_model 8340 "$M30" && assert_model 8341 "$M30"; then
  cell 8340 "$M30" onpolicy_skill 30bi & CELLPIDS="$CELLPIDS $!"
  cell 8341 "$M30" plain          30bi & CELLPIDS="$CELLPIDS $!"
else echo "WARN 30bi assertion failed; skipping 30bi cells"; fi
for cp in $CELLPIDS; do wait "$cp"; done

# ---------- tear down ONLY the servers this script started ----------
# Each was started under setsid, so its pid leads a process group that also holds the vLLM EngineCore
# children; signalling the GROUP stops them too (killing only the API server would orphan an engine
# that keeps its GPU memory). The 4bi replicas were not started here and are left running.
for sp in "${SERVE_PIDS[@]}"; do kill -TERM -- "-$sp" 2>/dev/null || kill -TERM "$sp" 2>/dev/null; done
sleep 15
for sp in "${SERVE_PIDS[@]}"; do kill -0 "$sp" 2>/dev/null && kill -KILL -- "-$sp" 2>/dev/null; done
echo "RB_SELF_DONE"

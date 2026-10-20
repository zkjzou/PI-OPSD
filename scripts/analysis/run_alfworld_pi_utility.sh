#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------------
# ALFWorld PI-UTILITY on the 200-task pool: no-hint baseline + 6 summarised hints + 2 checklists.
#
# THE BASELINE IS GENERATED HERE, NOT REUSED. The obvious shortcut is the alfworld_passk128_onpolicy
# dump those 200 tasks were scored from -- it IS a no-hint corpus at n=128. Two things rule it out:
# it records NO sampling config (keys are only data_source/env_class/env_extras/input_prompt/
# output_response/score/stop_reason), and its reward_spec paths are machine-local absolute paths. A delta
# against a baseline whose temperature and prompt style are unknown is a cross-config contrast wearing
# a utility label. Generating nohint here costs 800 rollouts and makes the contrast valid by
# construction.
#
# THREE DEFAULTS THAT SILENTLY PRODUCE 0% SUCCESS (measured, see the local-generate notes):
#   --max-input-length default 4096  -> full-history ReAct overflows by turn ~10, every episode ends
#                                       stop_reason=length, success 0%. Use 34000.
#   --max-generate-length default 512 -> truncates mid-turn. Use 2048.
#   --tokenizer must be the MODEL ID, not the vLLM --served-model-name, or AutoTokenizer raises.
# And --max-env-workers MUST exceed concurrent trajectories or the server pool deadlocks at 0% GPU
# with no crash (SkyRL's default of 32 is deadlock-prone; 256 is the canonical value).
#
# INPUTS. DATA = an ALFWorld parquet (alfworld_dataset.py) holding the 200 utility tasks; TASK_IDS
# optionally filters a larger parquet at generation time (the 200-task pool is
# $OPSD_DATA_ROOT/alfworld/splits/task_ids_train200_seed0.json). Hints default to the released hint tree
# ($OPSD_DATA_ROOT/alfworld/hints); the *_c24s / *_strip arms read STRIP, the outputs of
# strip_alfworld_hint_preamble.py with the same <author>/<file> layout, e.g.
#   python scripts/analysis/strip_alfworld_hint_preamble.py \
#     --src $HD/glm-5.2/hints_traj_withthink_cap24k.json --out $STRIP/glm-5.2/hints_traj_withthink_cap24k.json
# ---------------------------------------------------------------------------------------------------
set -o pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$WT" != / ] && ! { [ -f "$WT/pyproject.toml" ] && [ -f "$WT/opsd/__init__.py" ] && [ -f "$WT/opsd/skyrl_ext/__init__.py" ]; }; do WT="$(dirname "$WT")"; done
[ -f "$WT/opsd/skyrl_ext/__init__.py" ] || { echo "FATAL: cannot locate the repo root above $0" >&2; exit 1; }
cd "$WT" || exit 1
export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"  # `opsd` = the package at <repo root>/opsd
PY=${PY:-python}
VLLM=${VLLM:-vllm}
OPSD_DATA_ROOT=${OPSD_DATA_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data}
HD=${HD:-$OPSD_DATA_ROOT/alfworld/hints}            # released hint tree
STRIP=${STRIP:-$WT/outputs/alfworld_hints_stripped}  # strip_alfworld_hint_preamble.py outputs

MODEL_DIR=${MODEL_DIR:-Qwen/Qwen3-4B-Instruct-2507}   # HF id or a local snapshot dir
MODEL_NAME=${MODEL_NAME:-Qwen3-4B-Instruct-2507}
DATA=${DATA:?set DATA to the ALFWorld utility-pool parquet (see INPUTS above)}
TASK_IDS=${TASK_IDS:-}
ROOT=${ROOT:-$WT/outputs/pi_alfworld/utility_train200_4bi}
N_SAMPLES=${N_SAMPLES:-4}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
VLLM_BASE_PORT=${VLLM_BASE_PORT:-8700}
ALF_BASE_PORT=${ALF_BASE_PORT:-9100}      # 8200 collides with the vLLM range -- do not use it
SERVERS=${SERVERS:-96}
ENV_WORKERS=${ENV_WORKERS:-256}           # must exceed concurrent trajectories
MAX_INPUT=${MAX_INPUT:-34000}
MAX_GEN=${MAX_GEN:-2048}
MAX_TURNS=${MAX_TURNS:-50}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
TP=${TP:-1}                               # 30B-A3B is 57 GiB: at TP=1 an 80 GB card leaves ~10 GB of
                                          # KV, so it needs TP=2 (num_key_value_heads=4 shards cleanly)
ENABLE_THINKING=${ENABLE_THINKING:-false} # true for Qwen3-*-Thinking; Instruct-2507 emits no <think>
ALF_VENV=${ALF_VENV:-${ALFWORLD_VENV:?set ALFWORLD_VENV (the ALFWorld server venv)}}
ALF_DATA=${ALF_DATA:-${ALFWORLD_DATA:?set ALFWORLD_DATA (the ALFWorld game data)}}
ARMS=${ARMS:-"nohint anlz_glm anlz_qwn anlz_self summ_glm summ_qwn summ_self chk_glm chk_qwn"}

# SELF IS PER STUDENT, and anlz_self/summ_self were HARDCODED to 4bi_from_glm.json below -- so every
# non-4bi student's "self" arm actually consumed 4B-Instruct's analyzed rollouts, making it a
# cross-student arm. Same defect as run_pi_positions_alf_new8_local.sh. Derive from the ROOT
# (utility_train200_<student>[_c24...]) so a caller that sets ROOT cannot get it wrong.
SELF_TAG=${SELF_TAG:-$(basename "$ROOT" | sed -E 's/^utility_train200_//; s/_c24s?$//')}

_hints_for() {
  case "$1" in
    anlz_glm)  echo "$HD/derived/onpolicy_skill/glm52_from_glm.json" ;;
    anlz_qwn)  echo "$HD/derived/onpolicy_skill/qwen37_from_glm.json" ;;
    anlz_self) echo "$HD/derived/onpolicy_skill/${SELF_TAG}_from_glm.json" ;;
    summ_glm)  echo "$HD/derived/summary/glm52_from_glm.json" ;;
    summ_qwn)  echo "$HD/derived/summary/qwen37_from_glm.json" ;;
    summ_self) echo "$HD/derived/summary/${SELF_TAG}_from_glm.json" ;;
    # the 4B-Instruct-loss checklists (GLM-5.2 / Qwen3.7-Plus analyzer) serve every student here
    chk_glm)   echo "$HD/derived/checklist/checklist_from_glm.json" ;;
    chk_qwn)   echo "$HD/derived/checklist/checklist_from_q7.json" ;;
    # ---- TRAJECTORY-DENSITY arms: the raw / raw_no_think / actions axis ---------------------------
    # These are a DIFFERENT axis from anlz_*/summ_*, which vary the SUMMARISER over one glm-5.2 source.
    # Here the hint IS the reference trajectory at three densities, and the prefix names the trajectory
    # SOURCE (glm-5.2 vs qwen3.7) -- so `traj_*` is the only place ALFWorld has a real source axis.
    #
    # WHY THEY MATTER: `actions` is the BEST format on ALFWorld (+0.754 on the older 134-game run, 0
    # tasks hurt) and the WORST on AppWorld and tau2 (tau2: +0.12 vs +0.30 for the dense formats,
    # hurting 17-18 tasks). That flip currently rests on a different pool, baseline and metric from
    # everything else; running it here puts it on the SAME 200 tasks and nohint arm as the 8 hint arms.
    # The older run also covers only 3 of the 6 source x density cells (no raw_glm, no raw_no_think).
    traj_glm_raw)  echo "$HD/glm-5.2/hints_raw.json" ;;
    traj_glm_rnt)  echo "$HD/glm-5.2/hints_raw_no_think.json" ;;
    traj_glm_act)  echo "$HD/glm-5.2/hints_actions.json" ;;
    traj_qwn_raw)  echo "$HD/qwen3.7-plus/hints_raw.json" ;;
    traj_qwn_rnt)  echo "$HD/qwen3.7-plus/hints_raw_no_think.json" ;;
    traj_qwn_act)  echo "$HD/qwen3.7-plus/hints_actions.json" ;;
    # A/B CONTROL: identical hints with the leaked instruction preamble removed (everything before
    # the first "Task:"). Tests whether ALFWorld `raw`'s deficit vs `actions` is the scaffold leak or
    # genuine density.
    # ---- cap24k RERUN arms: the SAME three densities, rendered at a 24k cap ------------------------
    # The base traj_* files above are TRUNCATED by a middle elision ("...[trajectory truncated]...",
    # base.py:90-91): glm raw 39.9% / rnt 32.3%, qwen raw 46.0% / rnt 33.4%, all capped at 3880 chars
    # p50 and 5880 max. traj_*_act is truncated 0%. So the base grid scored a COMPLETE action list
    # against a trajectory with a hole punched through it -- a second defect, independent of the
    # scaffold leak, landing exactly where `actions > raw` was measured.
    #
    # cap24k on the 200-task pool: truncated 0.0-0.6%, goal statement 100% (vs 97-98%), p50 <action>
    # 13 (vs 11-12), coverage 200/200. The name UNDERSELLS it -- max is 65.8k-71.6k CHARS (~20-22k
    # tokens), not 24k chars, so these arms run with the long tail crowding the context. MAX_INPUT is
    # deliberately left at the base value so Dpass@1 stays paired to the EXISTING nohint arm; the
    # crowding is a real hint cost and is reported, not configured away.
    #
    # There is no *_act_c24 arm on purpose: hints_actions.json is the identical file in both sets
    # (md5 3ec69ce8481d57f8d3f48e0ac1a39669) and is never truncated, so the base arm already measures it.
    traj_glm_raw_c24) echo "$HD/glm-5.2/hints_traj_withthink_cap24k.json" ;;
    traj_glm_rnt_c24) echo "$HD/glm-5.2/hints_traj_nothink_cap24k.json" ;;
    traj_qwn_raw_c24) echo "$HD/qwen3.7-plus/hints_traj_withthink_cap24k.json" ;;
    traj_qwn_rnt_c24) echo "$HD/qwen3.7-plus/hints_traj_nothink_cap24k.json" ;;
    # ---- cap24k x STRIP: the only arms that vary the LEAK with truncation held at ~0 -------------
    # Built by strip_alfworld_hint_preamble.py, whose rule (cut to the first "Task: ") reproduces the
    # existing hints_glm_stripped/ artifact byte-for-byte on 2356/2356 entries -- so these are directly
    # comparable to the base *_strip arms.
    #
    # WHAT IS AND IS NOT REMOVED. The scaffold's system preamble is exactly ONE header block per hint
    # and goes 100% -> 0%. What remains is the ENV's invalid-action reply, which re-prints the action
    # vocabulary INSIDE the trajectory 0-14 times, scaling with the reference agent's invalid actions.
    # That is an observation the reference agent actually received; deleting it would falsify the
    # trajectory, so it stays and the residual is recorded (glm 1682, qwen 2039/2031 occurrences).
    # cap24k carries MORE of these than the base hints precisely because it is not truncated.
    traj_glm_raw_c24s) echo "$STRIP/glm-5.2/hints_traj_withthink_cap24k.json" ;;
    traj_glm_rnt_c24s) echo "$STRIP/glm-5.2/hints_traj_nothink_cap24k.json" ;;
    traj_qwn_raw_c24s) echo "$STRIP/qwen3.7-plus/hints_traj_withthink_cap24k.json" ;;
    traj_qwn_rnt_c24s) echo "$STRIP/qwen3.7-plus/hints_traj_nothink_cap24k.json" ;;
    traj_glm_raw_strip) echo "$STRIP/glm-5.2/hints_raw.json" ;;
    traj_glm_rnt_strip) echo "$STRIP/glm-5.2/hints_raw_no_think.json" ;;
    # The qwen half of the base-strip A/B, added to close the `leak(base)` column of the
    # truncation x leak 2x2: it previously existed for glm ONLY, so the base-level leak effect rested
    # on two cells that disagreed sharply (+0.0063 on raw vs +0.0737 on rnt) with no way to tell
    # density from noise. Same stripper, same rule (cut to the first "Task: ") -- re-verified to
    # reproduce the committed hints_glm_stripped artifacts byte-for-byte on 2356/2356 entries.
    traj_qwn_raw_strip) echo "$STRIP/qwen3.7-plus/hints_raw.json" ;;
    traj_qwn_rnt_strip) echo "$STRIP/qwen3.7-plus/hints_raw_no_think.json" ;;
    *) echo "" ;;
  esac
}
# ---- PRE-FLIGHT PORT CHECK ---------------------------------------------------------------------
# The 30bi summ_qwn arm died with rc=1 after ANOTHER process held env port 16000: the server
# initialised AlfredTWEnv fine, failed to bind ([Errno 98]), and the collision only surfaced at
# health-check timeout -- after paying full startup for all 64 servers, and leaving 63 of them
# orphaned when the driver died. This is the second port collision to cost an ALFWorld arm. Check
# first and SHIFT the block rather than discovering it minutes later.
_probe_block() {  # $1 = base port, $2 = count -> prints the first occupied port, or nothing
  local base=$1 n=$2 busy
  busy=$(ss -tln 2>/dev/null | awk '{print $4}' | grep -oE '[0-9]+$' \
         | awk -v b="$base" -v n="$n" '$1>=b && $1<b+n' | sort -n | head -1)
  echo "$busy"
}
if [ "${ALF_PORT_PREFLIGHT:-1}" = "1" ]; then
  _tries=0
  while [ "$_tries" -lt 20 ]; do
    _hit=$(_probe_block "$ALF_BASE_PORT" $(( SERVERS * 2 )))
    [ -z "$_hit" ] && break
    echo "[preflight] port $_hit in the block $ALF_BASE_PORT..$(( ALF_BASE_PORT + SERVERS*2 - 1 )) is" \
         "already bound (not mine to kill) -- shifting +1000"
    ALF_BASE_PORT=$(( ALF_BASE_PORT + 1000 ))
    _tries=$(( _tries + 1 ))
  done
  [ "$_tries" -lt 20 ] || { echo "FATAL: no free ALFWorld port block found after 20 shifts" >&2; exit 1; }
  echo "[preflight] env port block OK from $ALF_BASE_PORT"
fi

[ -s "$DATA" ] || { echo "FATAL: missing $DATA" >&2; exit 1; }
[ -x "$ALF_VENV/bin/python" ] || { echo "FATAL: no alfworld venv at $ALF_VENV" >&2; exit 1; }
for a in $ARMS; do
  [ "$a" = nohint ] && continue
  h=$(_hints_for "$a"); [ -s "$h" ] || { echo "FATAL: missing hints for $a: $h" >&2; exit 1; }
done
mkdir -p "$ROOT"

# ---- serve one vLLM replica per GPU -----------------------------------------------------------------
IFS=',' read -ra GA <<< "$GPUS"
[ $(( ${#GA[@]} % TP )) -eq 0 ] || { echo "FATAL: ${#GA[@]} GPUs not a multiple of TP=$TP" >&2; exit 1; }
URLS=""; PIDS=()
for (( i=0; i<${#GA[@]}; i+=TP )); do
  devs=$(IFS=,; echo "${GA[*]:i:TP}")      # TP consecutive cards per replica
  p=$(( VLLM_BASE_PORT + i ))
  CUDA_VISIBLE_DEVICES="$devs" "$VLLM" serve "$MODEL_DIR" \
      --served-model-name "$MODEL_NAME" --port "$p" --max-model-len "$MAX_MODEL_LEN" \
      --tensor-parallel-size "$TP" \
      --gpu-memory-utilization 0.85 --enable-prefix-caching --trust-remote-code \
      > "$ROOT/vllm_$p.log" 2>&1 &
  PIDS+=($!); URLS="${URLS:+$URLS,}http://127.0.0.1:$p"
done
echo "${PIDS[@]}" | tr ' ' '\n' > "$ROOT/vllm.pids"
for u in ${URLS//,/ }; do
  ok=0; for _ in $(seq 1 240); do curl -sf "$u/health" >/dev/null 2>&1 && { ok=1; break; }; sleep 5; done
  [ "$ok" = 1 ] || { echo "FATAL: $u never healthy" >&2; exit 1; }
done
echo "=== pool: $URLS ($(( ${#GA[@]} / TP )) replicas at TP=$TP) | $SERVERS env servers | N=$N_SAMPLES | 200 tasks"

idx=0; CONDS=""
for a in $ARMS; do
  OUT="$ROOT/$a"; mkdir -p "$OUT"
  base=$(( ALF_BASE_PORT + idx * 200 ))
  hint_args=""; h=$(_hints_for "$a"); [ -n "$h" ] && hint_args="--hints-path $h"
  think_args=""; [ "$ENABLE_THINKING" = true ] && think_args="--enable-thinking"
  ids_args=""; [ -n "$TASK_IDS" ] && ids_args="--task-ids-file $TASK_IDS"
  echo ""; echo "########## $a  (env ports ${base}..$(( base + SERVERS - 1 ))) ##########"
  "$PY" opsd/alfworld/generate_alfworld_local.py \
      --data-file "$DATA" --n-samples "$N_SAMPLES" $hint_args $ids_args \
      --model-url "$URLS" --served-model-name "$MODEL_NAME" --tokenizer "$MODEL_DIR" \
      --launch-alfworld-servers --num-alfworld-servers "$SERVERS" \
      --alfworld-base-port "$base" --alfworld-venv "$ALF_VENV" --alfworld-data "$ALF_DATA" \
      --max-env-workers "$ENV_WORKERS" --batch-size "$SERVERS" \
      --prompt-style react --history-length 0 --max-turns "$MAX_TURNS" \
      --max-input-length "$MAX_INPUT" --max-generate-length "$MAX_GEN" \
      $think_args \
      --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
      --presence-penalty 1.5 --repetition-penalty 1.0 \
      --output-dir "$OUT" --run-name "alfutil_$a" --disable-tqdm \
      > "$OUT/driver.log" 2>&1
  echo "  rc=$? -> $OUT  ($(wc -l < "$OUT/trajectories.jsonl" 2>/dev/null || echo 0) rollouts)"
  CONDS="$CONDS $a"; idx=$(( idx + 1 ))
done

echo ""
"$PY" opsd/alfworld/aggregate_utility_delta.py \
  --root "$ROOT" --conditions $CONDS --baseline nohint 2>&1 | tail -25
echo "ALF_UTILITY_DONE -> $ROOT"

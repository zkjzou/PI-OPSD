#!/usr/bin/env bash
# =====================================================================================================
# AppWorld no-hint corpus at high N (pass@k): one student, served on this node's GPUs, rolled out on one or
# more splits -- the corpora behind the AppWorld PI metrics of the paper (the D column's student rollouts and
# the U column's no-hint baseline).
#
#   bash scripts/appworld/passk.sh <4bi|4bt|30bi> [KEY=VALUE ...]
#
#   bash scripts/appworld/passk.sh 4bi                              # the paper corpus: train-90 + dev x 128
#   bash scripts/appworld/passk.sh 30bi SPLITS=train90              # one split
#   DRY_RUN=1 bash scripts/appworld/passk.sh 4bt                    # print the commands only
#
# PAPER CORPUS SETTINGS = the defaults below; change none of them to regenerate the paper's corpora:
#   * sampling profile `onpolicy`: temperature 1.0, top_p 1.0, top_k -1, min_p 0 (the distribution every
#     AppWorld training run rolls out from);
#   * N = 128 samples per task, no PI;
#   * splits train-90 + dev-57 (the paper's D and U pool train-90 and dev; the data bundle ships the train-90
#     outcome records only); a test_normal corpus (SPLITS="train90 dev normal") is optional;
#   * vLLM --max-model-len 49152, one replica per GPU = TP1 for 4bi / 4bt, TP2 for 30bi (4 replicas on 8
#     GPUs), gpu util 0.85, prefix caching;
#   * official ReAct prompt + scaffold, strict ```python``` parsing, binary reward, 30 turns, 50 AppWorld
#     interactions, 8192 tokens per turn, observations cut to 20000 characters;
#   * 4bt (Qwen3-4B-Thinking-2507) as run: --preserve-thinking --enable-thinking (its reasoning stays in the
#     context across turns; the corpus records carry preserve_thinking=true); 4bi / 30bi without either flag.
# NOT the paper settings: the defaults of the original pass@k run-file, SAMPLING_PROFILE=official (0.7 / 0.8
# / top-k 20), N=16, SPLITS="dev normal" (the paper corpora were launched with the values above instead).
# Concurrency (SERVERS_PER_SPLIT) changes throughput, not the samples.
#
# Outputs: $OUT_DIR/<split>/{trajectories.jsonl,transcripts.jsonl,summary.json,driver.log} (summary.json
# holds pass@k and the provenance the downstream utility gate checks). Re-running resumes a split.
#
# Knobs (KEY=VALUE or environment):
#   SPLITS              train90 | dev | normal | challenge, space-separated (default "train90 dev"); each split
#                       gets its own AppWorld server pool and runs concurrently with the others
#   N_SAMPLES           samples per task (default 128)
#   SAMPLING_PROFILE    onpolicy (default) | official (0.7/0.8/20) | thinking (0.6/0.95/20)
#   OUT_DIR             default ${RUN_ROOT:-./runs}/appworld_passk<N>_<profile>_<model>
#   SERVERS_PER_SPLIT   AppWorld servers (= concurrent rollouts) per split (default 240 / number of splits:
#                       240 servers in total, the count the paper's hosts ran)
#   APPWORLD_BASE_PORT  first AppWorld port (default 22000); split i uses BASE + i*APPWORLD_PORT_BLOCK (300)
#   VLLM_BASE_PORT      first vLLM port (default 8500); GPUS (default: CUDA_VISIBLE_DEVICES, else all)
#   APPWORLD_VENV / APPWORLD_ROOT / APPWORLD_DATA_DIR   as for run.sh (bash scripts/setup/setup_appworld.sh)
# Starts only its own processes (vLLM replicas, AppWorld servers) and stops them on exit.
# =====================================================================================================
set -eo pipefail

_aw_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ROOT = the repository root: pyproject.toml + the opsd package (opsd/, with recipes/) + scripts/
ROOT="$_aw_here"
while [ "$ROOT" != "/" ] && ! { [ -f "$ROOT/pyproject.toml" ] && [ -f "$ROOT/opsd/__init__.py" ] &&
    [ -f "$ROOT/opsd/skyrl_ext/__init__.py" ]; }; do
    ROOT="$(dirname "$ROOT")"
done
[ "$ROOT" != "/" ] || {
    echo "[appworld] FATAL: repository root (pyproject.toml + opsd/__init__.py + opsd/skyrl_ext/__init__.py)" \
        "not found above $_aw_here" >&2
    exit 1
}
case "${1:-}" in
    "" | -h | --help) sed -n '2,/^# =====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; [ -n "${1:-}" ] && exit 0; exit 2 ;;
esac
STUDENT="$1"
shift
for _a in "$@"; do
    case "$_a" in
        [A-Za-z_]*=*) export "${_a?}" ;;
        *) echo "[appworld] FATAL: expected KEY=VALUE, got '$_a'" >&2; exit 2 ;;
    esac
done

# shellcheck source=../common/bootstrap.sh
source "$ROOT/scripts/common/bootstrap.sh"
# shellcheck source=./servers.sh
source "$ROOT/scripts/appworld/servers.sh"
DRY=${DRY_RUN:-0}
PY="$(opsd_python)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"  # `opsd` = the package (skyrl_gym: the SkyRL install)
unset VLLM_API_KEY

aw_student "$STUDENT" || exit 2
SPLITS="${SPLITS:-train90 dev}"
N_SAMPLES="${N_SAMPLES:-128}"
SAMPLING_PROFILE="${SAMPLING_PROFILE:-onpolicy}"
aw_sampling "$SAMPLING_PROFILE" || exit 2
MAX_TURNS="${MAX_TURNS:-30}"
MAX_INTERACTIONS="${MAX_INTERACTIONS:-50}"
MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-49152}"
REPL_OUTPUT_MAX_CHARS="${REPL_OUTPUT_MAX_CHARS:-20000}"
PROMPT_VARIANT="${PROMPT_VARIANT:-official}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
APPWORLD_PORT_BLOCK="${APPWORLD_PORT_BLOCK:-300}"
APPWORLD_BASE_PORT="${APPWORLD_BASE_PORT:-22000}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8500}"
OPSD_DATA_ROOT="${OPSD_DATA_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data}"
aw_resolve_paths
APPWORLD_DATA_DIR="${APPWORLD_DATA_DIR:-$OPSD_DATA_ROOT/appworld/parquet/$PROMPT_VARIANT}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT:-./runs}/appworld_passk${N_SAMPLES}_${SAMPLING_PROFILE}_${SERVED_MODEL_NAME}}"
case "$OUT_DIR" in /*) ;; *) OUT_DIR="$PWD/${OUT_DIR#./}" ;; esac
export RUN_DIR="$OUT_DIR"  # process tracking ($RUN_DIR/pids)

_data_for() {
    case "$1" in
        train90) echo "$APPWORLD_DATA_DIR/train.parquet" ;;
        dev) echo "$APPWORLD_DATA_DIR/validation.parquet" ;;
        normal) echo "$APPWORLD_DATA_DIR/eval_normal.parquet" ;;
        challenge) echo "$APPWORLD_DATA_DIR/eval_challenge.parquet" ;;
        *) return 1 ;;
    esac
}
read -r -a SPLIT_ARR <<<"$(echo "$SPLITS" | tr ',+' '  ')"
[ "${#SPLIT_ARR[@]}" -gt 0 ] || opsd_die "SPLITS is empty"
for _sp in "${SPLIT_ARR[@]}"; do _data_for "$_sp" >/dev/null || opsd_die "unknown split '$_sp' (train90 | dev | normal | challenge)"; done
SERVERS_PER_SPLIT="${SERVERS_PER_SPLIT:-$((240 / ${#SPLIT_ARR[@]}))}"
[ "$APPWORLD_PORT_BLOCK" -gt "$SERVERS_PER_SPLIT" ] || opsd_die "APPWORLD_PORT_BLOCK must exceed SERVERS_PER_SPLIT"

THINK_ARGS=()
[ "$PRESERVE_THINKING" = true ] && THINK_ARGS=(--preserve-thinking --enable-thinking)
gen_cmd() {  # SPLIT BASE_PORT -> GEN array
    GEN=("$PY" "$ROOT/opsd/appworld/generate_appworld_local.py"
        --data-file "$(_data_for "$1")" --n-samples "$N_SAMPLES"
        --base-url "${VLLM_URLS:-<vllm-urls>}" --api-key-env "" --served-model-name "$SERVED_MODEL_NAME"
        --server-urls "$(aw_server_urls "$2" "$SERVERS_PER_SPLIT")" --concurrency "$SERVERS_PER_SPLIT"
        --temperature "$TEMP" --top-p "$TOPP" --top-k "$TOPK" --min-p 0.0
        --max-turns "$MAX_TURNS" --max-interactions "$MAX_INTERACTIONS"
        --max-generate-length "$MAX_GENERATE_LENGTH" --repl-output-max-chars "$REPL_OUTPUT_MAX_CHARS"
        --reward-mode binary --strict-parse "${THINK_ARGS[@]}"
        --save-transcripts --disable-tqdm --resume
        --output-dir "$OUT_DIR/$1" --run-name "appworld_passk${N_SAMPLES}_${SAMPLING_PROFILE}_${SERVED_MODEL_NAME}_$1")
}

echo "[appworld] passk: $SERVED_MODEL_NAME ($MODEL_PATH) profile=$SAMPLING_PROFILE ($TEMP/$TOPP/$TOPK) n=$N_SAMPLES" \
     "splits=${SPLIT_ARR[*]} servers/split=$SERVERS_PER_SPLIT vLLM TP=$VLLM_TP max_model_len=$MAX_MODEL_LEN" \
     "-> $OUT_DIR"
if [ "$DRY" = 1 ]; then
    _i=0
    for _sp in "${SPLIT_ARR[@]}"; do
        gen_cmd "$_sp" $((APPWORLD_BASE_PORT + _i * APPWORLD_PORT_BLOCK))
        printf 'DRY_RUN [%s]:' "$_sp"; printf ' %q' "${GEN[@]}"; echo
        _i=$((_i + 1))
    done
    exit 0
fi

# ---- preflight, data, model ---------------------------------------------------------------------------
source "$ROOT/scripts/common/preflight.sh"
_ngpu="$(aw_visible_gpus | tr ',' '\n' | grep -c .)" || _ngpu=0
if [ "${OPSD_PREFLIGHT_SKIP_GPUS:-0}" != 1 ]; then
    CUDA_VISIBLE_DEVICES="$(aw_visible_gpus)" opsd_preflight_gpus "$_ngpu" || opsd_die "GPUs not free (choose free ones with GPUS=...)"
fi
opsd_preflight_ports "$VLLM_BASE_PORT" $((_ngpu / VLLM_TP)) vllm || opsd_die "vLLM ports busy (set VLLM_BASE_PORT)"
for ((_i = 0; _i < ${#SPLIT_ARR[@]}; _i++)); do
    opsd_preflight_ports $((APPWORLD_BASE_PORT + _i * APPWORLD_PORT_BLOCK)) "$SERVERS_PER_SPLIT" "appworld[${SPLIT_ARR[$_i]}]" ||
        opsd_die "AppWorld ports busy (set APPWORLD_BASE_PORT)"
done
aw_check_install || exit 1
aw_is_patched || opsd_warn "b0a8eae_* not patched: train90 has 87 tasks, not the corpus' 90 (bash scripts/setup/setup_appworld.sh)"
aw_build_parquets "$APPWORLD_DATA_DIR" "$PROMPT_VARIANT" 0 || opsd_die "AppWorld parquet build failed"
mkdir -p "$OUT_DIR"
opsd_install_cleanup_trap
aw_serve_vllm "$OUT_DIR/logs" || opsd_die "vLLM did not come up"

# ---- one driver per split, concurrently, each with its own server pool --------------------------------
_run_split() {  # SPLIT INDEX
    local sp="$1" base=$((APPWORLD_BASE_PORT + $2 * APPWORLD_PORT_BLOCK)) rc=0
    mkdir -p "$OUT_DIR/$sp"
    gen_cmd "$sp" "$base"
    "${GEN[@]}" >"$OUT_DIR/$sp/driver.log" 2>&1 || rc=$?
    echo "[appworld] [$sp] driver rc=$rc -> $OUT_DIR/$sp"
    return "$rc"
}
for ((_i = 0; _i < ${#SPLIT_ARR[@]}; _i++)); do
    _base=$((APPWORLD_BASE_PORT + _i * APPWORLD_PORT_BLOCK))
    aw_start_server_pool "$_base" "$SERVERS_PER_SPLIT" "$OUT_DIR/logs/appworld_${SPLIT_ARR[$_i]}" "aw-${SPLIT_ARR[$_i]}"
done
for ((_i = 0; _i < ${#SPLIT_ARR[@]}; _i++)); do
    _base=$((APPWORLD_BASE_PORT + _i * APPWORLD_PORT_BLOCK))
    aw_wait_server_pool "$_base" "$SERVERS_PER_SPLIT" "$SERVERS_PER_SPLIT" "${APPWORLD_READY_TIMEOUT:-900}" || opsd_die "server pool down"
    aw_check_port_owners "$_base" "$SERVERS_PER_SPLIT" "aw-${SPLIT_ARR[$_i]}" || opsd_die "foreign process on an AppWorld port"
done
sleep "${APPWORLD_WARMUP_S:-25}"  # let /initialize come up behind the bound sockets

_pids=()
for ((_i = 0; _i < ${#SPLIT_ARR[@]}; _i++)); do
    _run_split "${SPLIT_ARR[$_i]}" "$_i" &
    _pids+=("$!")
    opsd_track_pid "$!" "driver-${SPLIT_ARR[$_i]}" || true
done
rc=0
for _p in "${_pids[@]}"; do wait "$_p" || rc=1; done
for _sp in "${SPLIT_ARR[@]}"; do
    [ -f "$OUT_DIR/$_sp/summary.json" ] || continue
    "$PY" - "$OUT_DIR/$_sp/summary.json" "$_sp" <<'PY' || true
import json
import sys

d = json.load(open(sys.argv[1]))
ph = d.get("pass_hat_k", {})
ks = [k for k in ("1", "2", "4", "8", "16", "32", "64", "128") if k in ph]
print("[appworld] %s: %s rollouts over %s tasks, " % (sys.argv[2], d.get("num_trajectories"), d.get("tasks"))
      + ", ".join("pass@%s=%.4f" % (k, ph[k]) for k in ks))
PY
done
exit "$rc"

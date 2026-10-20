#!/usr/bin/env bash
# =====================================================================================================
# AppWorld PI utility: rollouts WITH a privileged-information hint, n=4 per task, for every PI of one
# student -- the hinted arms of the AppWorld U column (utility = paired per-task pass@1 with the hint minus
# pass@1 of the student's n=128 no-hint corpus from passk.sh, same sampling profile).
#
#   bash scripts/appworld/pi_utility.sh <4bi|4bt|30bi> [KEY=VALUE ...]
#
#   bash scripts/appworld/pi_utility.sh 4bi                     # 6 trajectory + 8 analyzer PIs, train-90
#   bash scripts/appworld/pi_utility.sh 30bi ARMSET=new8        # the 8 analyzer / checklist PIs only
#   DRY_RUN=1 bash scripts/appworld/pi_utility.sh 4bt           # print the cells and commands
#
# PAPER SETTINGS = the defaults below (change none of them to regenerate the paper's hinted cells):
#   * sampling profile `onpolicy`: temperature 1.0, top_p 1.0, top_k -1, min_p 0;
#   * N = 4 samples per task WITH the hint; the no-hint side is the N = 128 corpus of passk.sh (same profile,
#     same limits) -- utility is never computed against a fresh n=4 no-hint run;
#   * splits: the paper's U pools train-90 AND dev-57. The default here is train-90 only; add dev with
#     SPLITS="train90 dev", which reads the dev hints from the dataset's encrypted
#     appworld_opsd_dev_hints.bundle (EVALUATION-SPLIT MATERIAL: hints keyed to AppWorld dev, for reproducing
#     the paper's U only) once scripts/setup/setup_appworld.sh has decrypted it;
#   * vLLM --max-model-len 49152, TP1 for 4bi / 4bt, TP2 for 30bi; gpu util 0.85, prefix caching;
#   * official ReAct prompt + scaffold, strict ```python``` parsing, binary reward, 30 turns, 50 AppWorld
#     interactions, 8192 tokens per turn, observations cut to 20000 characters;
#   * 4bt as run: --preserve-thinking --enable-thinking; 4bi / 30bi without either flag;
#   * the hint injected as a <Secret information> system turn prepended to the prompt
#     (--hint-inject-style system-prepend) -- the same block the OPSD teacher sees.
# NOT the paper settings: the defaults of the original utility run-file, SAMPLING_PROFILE=official (0.7 /
# 0.8 / top-k 20) and N=16.
# A task without a hint (the Qwen3.7-Plus trajectories cover 89 of the 90 train tasks) is rolled out without
# one, as in the original runs. Cells run one after another, each with a fresh pool of AppWorld servers.
# The no-hint baseline is the passk.sh corpus, converted by
# scripts/analysis/build_appworld_nohint_baselines_from_passk.sh; U = scripts/analysis/appworld_utility_both_k.py.
#
# Arms (cell names = the arm names of the U column):
#   ARMSET=traj   glm52_{raw,raw_no_think,actions}, qwen37_{raw,raw_no_think,actions}
#                 (FORMATS / SOURCES narrow it; FORMATS="... actions_think" adds the Action+Thoughts PIs
#                 glm_aat / qwen_aat, which are not in the paper's U column)
#   ARMSET=new8   anlz_{glm,qwn,self} (on-policy skill), summ_{glm,qwn,self} (summary), chk_{glm,qwn}
#                 (checklist; the 4B-Instruct build is used for every student, as in the original cells)
#   ARMSET=all    both (default)
# Hints come from the unpacked AppWorld bundles: train90 from $OPSD_DATA_ROOT/appworld/plain/opsd_appworld_pi
# (train-90 keys only), dev from $OPSD_DATA_ROOT/appworld/plain/opsd_appworld_dev_hints (hints/trajectory/<author>/
# and hints/arms/<student>/; AW_DEV_HINTS overrides). HINTS_DIR_<split>=<dir holding hints_<arm>.json> replaces
# either for its split, and is the only hint source of the `normal` split: the test_normal hints ship in the
# encrypted appworld_opsd_test_normal_hints.bundle (EVALUATION-SPLIT MATERIAL, same layout as the
# dev hints), which this script does not map.
#
# Outputs: $OUT_DIR/<split>/<arm>/{trajectories.jsonl,transcripts.jsonl,summary.json,driver.log}; a cell with
# a summary.json is complete and skipped on a re-run. Knobs as passk.sh (N_SAMPLES default 4,
# SERVERS_PER_SPLIT 240, APPWORLD_BASE_PORT 22000, VLLM_BASE_PORT 8500, GPUS, OUT_DIR).
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
SPLITS="${SPLITS:-train90}"
N_SAMPLES="${N_SAMPLES:-4}"
SAMPLING_PROFILE="${SAMPLING_PROFILE:-onpolicy}"
aw_sampling "$SAMPLING_PROFILE" || exit 2
ARMSET="${ARMSET:-all}"
SOURCES="${SOURCES:-glm52 qwen37}"
FORMATS="${FORMATS:-raw raw_no_think actions}"
HINT_INJECT_STYLE="${HINT_INJECT_STYLE:-system-prepend}"
MAX_TURNS="${MAX_TURNS:-30}"
MAX_INTERACTIONS="${MAX_INTERACTIONS:-50}"
MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-49152}"
REPL_OUTPUT_MAX_CHARS="${REPL_OUTPUT_MAX_CHARS:-20000}"
PROMPT_VARIANT="${PROMPT_VARIANT:-official}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
SERVERS_PER_SPLIT="${SERVERS_PER_SPLIT:-240}"
APPWORLD_BASE_PORT="${APPWORLD_BASE_PORT:-22000}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8500}"
OPSD_DATA_ROOT="${OPSD_DATA_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data}"
aw_resolve_paths
APPWORLD_DATA_DIR="${APPWORLD_DATA_DIR:-$OPSD_DATA_ROOT/appworld/parquet/$PROMPT_VARIANT}"
AW_BUNDLE="${AW_BUNDLE:-$(aw_bundle_dir)}"
AW_DEV_HINTS="${AW_DEV_HINTS:-$(dirname "$AW_BUNDLE")/opsd_appworld_dev_hints}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT:-./runs}/appworld_pi_utility_${SAMPLING_PROFILE}_${SERVED_MODEL_NAME}}"
case "$OUT_DIR" in /*) ;; *) OUT_DIR="$PWD/${OUT_DIR#./}" ;; esac
export RUN_DIR="$OUT_DIR"

_data_for() {
    case "$1" in
        train90) echo "$APPWORLD_DATA_DIR/train.parquet" ;;
        dev) echo "$APPWORLD_DATA_DIR/validation.parquet" ;;
        normal) echo "$APPWORLD_DATA_DIR/eval_normal.parquet" ;;
        *) return 1 ;;
    esac
}
_author() { case "$1" in glm52) echo glm-5.2 ;; qwen37) echo qwen3.7-plus ;; *) return 1 ;; esac; }
# cell name -> hint file ($HINTS_DIR_<split>/hints_<cell>.json if set; else train-90: the batch-1 bundle, dev: the
# dev-hints bundle)
_hint_for() {  # SPLIT CELL
    local sp="$1" cell="$2" var="HINTS_DIR_$1" h="$AW_BUNDLE/hints"
    if [ -n "${!var:-}" ]; then
        echo "${!var}/hints_$cell.json"
        return 0
    fi
    if [ "$sp" = dev ]; then
        case "$cell" in
            glm52_* | qwen37_*) echo "$AW_DEV_HINTS/hints/trajectory/$(_author "${cell%%_*}")/hints_${cell#*_}.json" ;;
            *) echo "$AW_DEV_HINTS/hints/arms/$STUDENT/hints_$cell.json" ;;
        esac
        return 0
    fi
    [ "$sp" = train90 ] || return 1
    case "$cell" in
        glm52_* | qwen37_*) echo "$h/trajectory/$(_author "${cell%%_*}")/hints_${cell#*_}.json" ;;
        anlz_glm) echo "$h/skill/glm52_from_glm52_train90.json" ;;
        anlz_qwn) echo "$h/skill/qwen37_from_glm52_train90.json" ;;
        anlz_self) echo "$h/skill/${STUDENT}_from_glm52_train90.json" ;;
        summ_glm) echo "$h/summary/glm52_from_glm52_train90.json" ;;
        summ_qwn) echo "$h/summary/qwen37_from_glm52_train90.json" ;;
        summ_self) echo "$h/summary/${STUDENT}_from_glm52_train90.json" ;;
        chk_glm) echo "$h/checklist/analyzer=glm-5.2/student=4bi/hints_checklist.json" ;;
        chk_qwn) echo "$h/checklist/analyzer=qwen3.7-plus/student=4bi/hints_checklist.json" ;;
        *) return 1 ;;
    esac
}

CELLS=()
case "$ARMSET" in traj | all) for _s in $SOURCES; do for _f in $FORMATS; do CELLS+=("${_s}_${_f}"); done; done ;; esac
case "$ARMSET" in new8 | all) CELLS+=(anlz_glm anlz_qwn anlz_self summ_glm summ_qwn summ_self chk_glm chk_qwn) ;; esac
case "$ARMSET" in traj | new8 | all) ;; *) opsd_die "ARMSET must be traj | new8 | all" ;; esac
read -r -a SPLIT_ARR <<<"$(echo "$SPLITS" | tr ',+' '  ')"
for _sp in "${SPLIT_ARR[@]}"; do
    _data_for "$_sp" >/dev/null || opsd_die "unknown split '$_sp' (train90 | dev | normal)"
    for _c in "${CELLS[@]}"; do
        _hint_for "$_sp" "$_c" >/dev/null || opsd_die "no hint file for $_sp/$_c (set HINTS_DIR_$_sp: only the train90 and dev hints are mapped)"
    done
done
case " ${SPLIT_ARR[*]} " in
    *" dev "*) [ -n "${HINTS_DIR_dev:-}" ] || opsd_warn "dev hints from $AW_DEV_HINTS: evaluation-split material (keyed to AppWorld dev), for reproducing the paper's U only; never use them as training hints" ;;
esac

THINK_ARGS=()
[ "$PRESERVE_THINKING" = true ] && THINK_ARGS=(--preserve-thinking --enable-thinking)
gen_cmd() {  # SPLIT CELL -> GEN array
    GEN=("$PY" "$ROOT/opsd/appworld/generate_appworld_local.py"
        --data-file "$(_data_for "$1")" --n-samples "$N_SAMPLES"
        --hints-path "$(_hint_for "$1" "$2")" --hint-inject-style "$HINT_INJECT_STYLE"
        --base-url "${VLLM_URLS:-<vllm-urls>}" --api-key-env "" --served-model-name "$SERVED_MODEL_NAME"
        --server-urls "$(aw_server_urls "$APPWORLD_BASE_PORT" "$SERVERS_PER_SPLIT")" --concurrency "$SERVERS_PER_SPLIT"
        --temperature "$TEMP" --top-p "$TOPP" --top-k "$TOPK" --min-p 0.0
        --max-turns "$MAX_TURNS" --max-interactions "$MAX_INTERACTIONS"
        --max-generate-length "$MAX_GENERATE_LENGTH" --repl-output-max-chars "$REPL_OUTPUT_MAX_CHARS"
        --reward-mode binary --strict-parse "${THINK_ARGS[@]}"
        --save-transcripts --disable-tqdm --resume
        --output-dir "$OUT_DIR/$1/$2"
        --run-name "appworld_pi_utility_${SAMPLING_PROFILE}_${SERVED_MODEL_NAME}_$1_$2")
}

echo "[appworld] pi_utility: $SERVED_MODEL_NAME profile=$SAMPLING_PROFILE ($TEMP/$TOPP/$TOPK) n=$N_SAMPLES" \
     "vLLM TP=$VLLM_TP max_model_len=$MAX_MODEL_LEN splits=${SPLIT_ARR[*]} cells=${#CELLS[@]} (${CELLS[*]}) -> $OUT_DIR"
if [ "$DRY" = 1 ]; then
    for _sp in "${SPLIT_ARR[@]}"; do
        for _c in "${CELLS[@]}"; do
            gen_cmd "$_sp" "$_c"
            printf 'DRY_RUN [%s/%s]:' "$_sp" "$_c"; printf ' %q' "${GEN[@]}"; echo
        done
    done
    exit 0
fi

# ---- preflight, data, hints, model --------------------------------------------------------------------
source "$ROOT/scripts/common/preflight.sh"
_ngpu="$(aw_visible_gpus | tr ',' '\n' | grep -c .)" || _ngpu=0
if [ "${OPSD_PREFLIGHT_SKIP_GPUS:-0}" != 1 ]; then
    CUDA_VISIBLE_DEVICES="$(aw_visible_gpus)" opsd_preflight_gpus "$_ngpu" || opsd_die "GPUs not free (choose free ones with GPUS=...)"
fi
opsd_preflight_ports "$VLLM_BASE_PORT" $((_ngpu / VLLM_TP)) vllm || opsd_die "vLLM ports busy (set VLLM_BASE_PORT)"
opsd_preflight_ports "$APPWORLD_BASE_PORT" "$SERVERS_PER_SPLIT" appworld || opsd_die "AppWorld ports busy (set APPWORLD_BASE_PORT)"
for _sp in "${SPLIT_ARR[@]}"; do
    for _c in "${CELLS[@]}"; do
        _h="$(_hint_for "$_sp" "$_c")"
        # an empty map ({} is 2 bytes) would silently make a hinted cell a no-hint one
        _n="$("$PY" -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "$_h" 2>/dev/null || echo 0)"
        [ "$_n" -ge 1 ] 2>/dev/null || opsd_die "missing or empty hint file for $_sp/$_c: $_h (bash scripts/setup/setup_appworld.sh unpacks the bundle)"
    done
done
aw_check_install || exit 1
aw_is_patched || opsd_warn "b0a8eae_* not patched: train90 has 87 tasks, not the original 90 (bash scripts/setup/setup_appworld.sh)"
aw_build_parquets "$APPWORLD_DATA_DIR" "$PROMPT_VARIANT" 0 || opsd_die "AppWorld parquet build failed"
mkdir -p "$OUT_DIR"
opsd_install_cleanup_trap
aw_serve_vllm "$OUT_DIR/logs" || opsd_die "vLLM did not come up"

# ---- cells, sequentially, each with a fresh AppWorld server pool ---------------------------------------
rc=0
_k=0
for _sp in "${SPLIT_ARR[@]}"; do
    for _c in "${CELLS[@]}"; do
        _k=$((_k + 1))
        _out="$OUT_DIR/$_sp/$_c"
        if [ -s "$_out/summary.json" ] && [ -s "$_out/trajectories.jsonl" ]; then
            echo "[appworld] cell $_k: $_sp/$_c already complete -- skipped"
            continue
        fi
        echo "[appworld] cell $_k: $_sp/$_c ($(_hint_for "$_sp" "$_c"))"
        mkdir -p "$_out"
        for _w in $(seq 1 30); do  # the previous cell's servers have to release the ports first
            opsd_preflight_ports "$APPWORLD_BASE_PORT" "$SERVERS_PER_SPLIT" appworld 2>/dev/null && break
            sleep 2
        done
        aw_start_server_pool "$APPWORLD_BASE_PORT" "$SERVERS_PER_SPLIT" "$OUT_DIR/logs/appworld_cell" aw-cell
        aw_wait_server_pool "$APPWORLD_BASE_PORT" "$SERVERS_PER_SPLIT" "$SERVERS_PER_SPLIT" "${APPWORLD_READY_TIMEOUT:-900}" ||
            opsd_die "server pool down"
        aw_check_port_owners "$APPWORLD_BASE_PORT" "$SERVERS_PER_SPLIT" aw-cell || opsd_die "foreign process on an AppWorld port"
        sleep "${APPWORLD_WARMUP_S:-25}"
        gen_cmd "$_sp" "$_c"
        _crc=0
        opsd_run "${GEN[@]}" >"$_out/driver.log" 2>&1 || _crc=$?
        echo "[appworld]   rc=$_crc; $(grep -E '^PI:' "$_out/driver.log" | head -1)"
        [ "$_crc" = 0 ] || rc=1
        opsd_cleanup_own 'aw-cell-*'
    done
done
for _sp in "${SPLIT_ARR[@]}"; do
    for _c in "${CELLS[@]}"; do
        [ -f "$OUT_DIR/$_sp/$_c/summary.json" ] || continue
        "$PY" - "$OUT_DIR/$_sp/$_c/summary.json" "$_sp/$_c" <<'PY' || true
import json
import sys

d = json.load(open(sys.argv[1]))
print("[appworld] %-22s pass@1=%.4f over %s tasks (%s hinted)" % (sys.argv[2], d.get("pass_at_1", 0.0), d.get("tasks"),
                                                               d.get("hinted_tasks")))
PY
    done
done
echo "[appworld] utility = paired per-task pass@1(hinted) - pass@1(no-hint corpus of passk.sh, same profile)"
exit "$rc"

#!/usr/bin/env bash
# =====================================================================================================
# run.sh -- train (or evaluate) one ALFWorld arm.
#
#   scripts/alfworld/run.sh <arm_id> [KEY=VAL ...] [--overlay NAME ...]     (from any directory)
#   DRY_RUN=1 scripts/alfworld/run.sh alfworld/main/4bi/glm_ao     # print module + argv + env, exit 0
#   python -m opsd.recipes list --bench alfworld --long           # every arm id, status, paper cell
#   python -m opsd.recipes show alfworld/main/4bi/glm_ao          # resolved env with provenance
#
# Flow: resolve the arm's recipe (opsd/recipes/alfworld/) into $RUN_DIR/resolved.env and source it
# (hermetic: ambient OPSD_* or recipe keys that are not passed as KEY=VAL are an error) -> bootstrap ->
# preflight (GPUs, ports, disk, input files, hint sha256) -> the ALFWorld server pool (this run's own
# processes) and a readiness wait -> per-run parquets -> the trainer:
#   ENTRY=opsd (every paper arm)  python -m opsd.trainer.main_opsd_alfworld
#   ENTRY=dapo                    python -m opsd.alfworld.main_alfworld_dapo
#                                 (GRPO/DAPO with the soft overlong buffer; no paper arm uses it:
#                                  run.sh alfworld/grpo/4bi/none/seed=1 ENTRY=dapo OVERLONG_BUFFER_LEN=7000)
# The trainer runs from the repository root (the working directory), which also goes first on PYTHONPATH.
# Checkpoints go to $RUN_DIR/ckpts, evaluation dumps and exports to $RUN_DIR/exports, train-rollout dumps to
# $RUN_DIR/train_rollouts, ALFWorld server logs to $RUN_DIR/logs/alfworld.
# RUN_DIR = ${RUN_ROOT:-./runs}/<arm id, '/' -> '__', '=' -> '-'>.
# Paths are relative to the directory you launch from, whether given as KEY=VAL or in the environment: the
# launcher makes them absolute before it changes into the repository root for the trainer. That covers
# RUN_ROOT / RUN_DIR / OPSD_DATA_ROOT, the input and output paths (OPSD_HINTS_PATH, TASK_IDS_PATH,
# DEV_TASK_IDS_PATH, EVAL_HINT_DIR, EVAL_HINT_TASK_IDS_PATH, RESUME_PATH, EVAL_CKPT_DIR, OPSD_ROLLOUT_DUMP_DIR,
# OPSD_EMA_DIR) and ALFWORLD_VENV / ALFWORLD_DATA / ALFWORLD_PARQUET_DIR. MODEL_PATH is made absolute only when
# it names an existing local path, so a Hugging Face id (Qwen/...) stays as given.
# Resuming: base.env defaults to RESUME_MODE=latest (continue from the run's own latest checkpoint), but every
# arm that ran without autoresume keeps RESUME_MODE=null as it ran (all main-table, beta, lambda-grid, step0
# and null arms, among others: `python -m opsd.recipes show <arm>`). A relaunch of such an arm starts
# again from step 0 in the same checkpoint directory (the launcher warns); pass RESUME_MODE=latest to continue.
#
# One-time setup: the training environment (`bash scripts/setup/setup_train_env.sh`: the pinned SkyRL checkout
# next to the repository + `uv sync --frozen --extra fsdp`, or `--extra megatron` for the 30B student), the
# ALFWorld server venv + game data + parquets (`bash scripts/setup/setup_alfworld.sh`), and the data
# bundle in $OPSD_DATA_ROOT (`python -m opsd.data.fetch`).
#
# Run modes (pass as KEY=VAL):
#   EVAL_ONLY=true                  evaluate the student (or RESUME_PATH=<ckpt dir> RESUME_MODE=from_path)
#                                   on the arm's eval pools without training
#   EVAL_HINT_SETS=a+b              (with EVAL_ONLY) one eval parquet per hint set over EVAL_HINT_TASK_IDS_PATH,
#                                   hints from $EVAL_HINT_DIR/hints_<set>.json (`nohint` = no hint)
#   EVAL_CKPT_STEPS=25+50+75+100    re-score saved checkpoints on valid_seen + valid_unseen (N =
#                                   EVAL_CKPT_N_SAMPLES, default 4), one trainer process per step, metrics
#                                   logged at the original step. RUN_NAME / RUN_ROOT / RUN_DIR still name the
#                                   TRAINING run (pass the values it trained with); the re-score gets its own
#                                   RUN_NAME/RUN_DIR = <training run>__ckpteval[_<EVAL_CKPT_TAG>] and reads
#                                   EVAL_CKPT_DIR (default <training RUN_DIR>/ckpts), so the training run's
#                                   resolved.env, provenance.json and checkpoints are never written. This is
#                                   how the dev200-protocol arms were scored; some arms were re-scored with
#                                   extra overrides, listed under `rescore:` in their arms.yaml table entry
#                                   (shown by `python -m opsd.recipes show <arm>`).
#   RESUME_RESET_DATALOADER=1       (with RESUME_MODE=from_path) drop the checkpoint's dataloader state. No
#                                   longer needed at an epoch boundary (the trainer, OPSDRayPPOTrainer, drains
#                                   an exhausted restored iterator itself); dropping it now restarts the data
#                                   order from the seeded generator. Wrong mid-epoch
#   --overlay smoke_4gpu            a short 4-GPU end-to-end check (not comparable to the paper)
#
# Machine-local knobs (environment; not part of the recipe): ALFWORLD_VENV ($HOME/alfworld-venv),
# ALFWORLD_DATA ($HOME/data/alfworld), ALFWORLD_PARQUET_DIR ($ALFWORLD_DATA/parquet), ALFWORLD_BASE_PORT
# (9100; servers use BASE..BASE+NUM_ALFWORLD_SERVERS-1, clear of vLLM's 8000-8800), HTTP_ENDPOINT_HOST /
# HTTP_ENDPOINT_PORT (127.0.0.1 / 8000), ALFWORLD_READY_TIMEOUT_S (600), WANDB_PROJECT (opsd-alfworld),
# OPSD_PYTHON (the training interpreter), OPSD_PREFLIGHT=0 (skip the checks).
#
# Two nodes (the 30B student, NUM_NODES=2): start Ray first, then launch on the head node only:
#   head:    scripts/common/ray_cluster.sh head
#   worker:  hf download Qwen/Qwen3-30B-A3B-Instruct-2507       # ~61 GB, into the HF_HOME used by the next line
#            scripts/common/ray_cluster.sh worker <head-ip>:6379
#   head:    scripts/common/ray_cluster.sh wait 2
#            RAY_ADDRESS=<head-ip>:6379 scripts/alfworld/run.sh alfworld/main/30bi/glm_ao
# The launcher pre-stages the model (PRESTAGE_MODEL=1, after preflight) only on the node it runs on. The
# worker's 8 ranks read the weights from that node's own HF cache, so download them there first, with the
# HF_HOME that `ray_cluster.sh worker` runs under (Ray's processes inherit it). Skip that step only when both
# nodes share one HF_HOME. Otherwise every worker rank pulls the 61 GB MoE live at model load.
# RUN_DIR (every rank writes checkpoint shards) and OPSD_DATA_ROOT must be on a filesystem shared by both
# nodes, at the same path. The ALFWorld servers and the trainer's entry task run on the head node.
# =====================================================================================================
set -o pipefail

_RUN_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ROOT = the repository root: pyproject.toml + the opsd package (opsd/, with recipes/) + scripts/
ROOT="$_RUN_SH_DIR"
while [ "$ROOT" != "/" ] && ! { [ -f "$ROOT/pyproject.toml" ] && [ -f "$ROOT/opsd/__init__.py" ] &&
    [ -f "$ROOT/opsd/skyrl_ext/__init__.py" ]; }; do
    ROOT="$(dirname "$ROOT")"
done
if [ "$ROOT" = "/" ]; then
    echo "run.sh: repository root (pyproject.toml + opsd/__init__.py + opsd/skyrl_ext/__init__.py) not found" \
        "above $_RUN_SH_DIR" >&2
    exit 2
fi

usage() { awk 'NR > 2 && /^# =====/ { exit } NR > 2 { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; }
case "${1:-}" in
    "" | -h | --help) usage; [ -n "${1:-}" ]; exit $? ;;
esac
ARM_ID="$1"
shift

# Paths may be given relative to the caller's directory (machine-local ones here; RUN_ROOT, OPSD_DATA_ROOT and
# KEY=VAL paths, MODEL_PATH when it is a local path, by the recipe resolution below, which runs in the caller's
# directory; an ambient EVAL_CKPT_DIR right after it). Everything after the resolution runs from ROOT.
_abs() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
ALFWORLD_VENV="$(_abs "${ALFWORLD_VENV:-$HOME/alfworld-venv}")"
ALFWORLD_DATA="$(_abs "${ALFWORLD_DATA:-$HOME/data/alfworld}")"
ALFWORLD_PARQUET_DIR="$(_abs "${ALFWORLD_PARQUET_DIR:-$ALFWORLD_DATA/parquet}")"
export ALFWORLD_VENV ALFWORLD_DATA ALFWORLD_PARQUET_DIR

# shellcheck disable=SC1091
source "$ROOT/scripts/common/bootstrap.sh"

# ---- EVAL_CKPT_STEPS: a checkpoint re-score gets its OWN run directory -----------------------------------
# RUN_NAME / RUN_ROOT / RUN_DIR keep naming the TRAINING run. Resolve where that run lives, then run the
# re-score as <training RUN_NAME>__ckpteval[_<tag>] in <training RUN_DIR>__ckpteval[_<tag>], reading the
# training run's checkpoints (EVAL_CKPT_DIR, unless given). Otherwise the re-score's resolved.env and
# provenance.json would replace the training run's, and that record would then describe the re-score.
_kv_arg() {  # NAME ARGS...: the last NAME=VALUE among the KEY=VAL arguments, else the ambient $NAME
    local name="$1" a v
    v="${!name:-}"
    shift
    for a in "$@"; do case "$a" in "$name="*) v="${a#*=}" ;; esac; done
    printf '%s' "$v"
}
_RESCORE_ARGS=()
_RESCORE_LOC=0
if [ -n "$(_kv_arg EVAL_CKPT_STEPS "$@")" ]; then
    _train_loc="$(_opsd_pymod opsd.recipes resolve "$ARM_ID" "$@" --format plain 2>/dev/null)"
    _TRAIN_RUN_NAME="$(printf '%s\n' "$_train_loc" | sed -n 's/^RUN_NAME=//p')"
    _TRAIN_RUN_DIR="$(printf '%s\n' "$_train_loc" | sed -n 's/^RUN_DIR=//p')"
    if [ -n "$_TRAIN_RUN_NAME" ] && [ -n "$_TRAIN_RUN_DIR" ]; then
        _tag="$(_kv_arg EVAL_CKPT_TAG "$@")"
        _sfx="__ckpteval${_tag:+_$_tag}"
        _RESCORE_ARGS=("RUN_NAME=$_TRAIN_RUN_NAME$_sfx" "RUN_DIR=$_TRAIN_RUN_DIR$_sfx")
        [ -n "$(_kv_arg EVAL_CKPT_DIR "$@")" ] || _RESCORE_ARGS+=("EVAL_CKPT_DIR=$_TRAIN_RUN_DIR/ckpts")
        _RESCORE_LOC=1
    fi
fi
opsd_resolve_recipe "$ARM_ID" "$@" "${_RESCORE_ARGS[@]}" || exit 2
# EVAL_CKPT_DIR may also come from the environment (not through the resolver): absolute before the cd as well.
[ -z "${EVAL_CKPT_DIR:-}" ] || EVAL_CKPT_DIR="$(_abs "$EVAL_CKPT_DIR")"
cd "$ROOT" || exit 2
# The repository root first on the trainer's sys.path: `opsd` is the package there (skyrl / skyrl_gym come from
# the install).
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${EVAL_CKPT_STEPS:-}" ]; then
    [ "$_RESCORE_LOC" = 1 ] || opsd_die "EVAL_CKPT_STEPS: could not resolve the training run's directory for $ARM_ID" 2
    if [ "${EVAL_CKPT_DIR%/}" = "$RUN_DIR/ckpts" ]; then
        opsd_die "EVAL_CKPT_DIR=$EVAL_CKPT_DIR is this re-score's own checkpoint directory; point it at the training run's ckpts" 2
    fi
fi
if [ "${OPSD_BENCH:-}" != alfworld ]; then
    opsd_die "$ARM_ID is a ${OPSD_BENCH:-?} arm; use scripts/${OPSD_BENCH:-<bench>}/run.sh" 2
fi
DRY_RUN=${DRY_RUN:-0}
PY="$(opsd_python)"
# EVAL_ONLY=1 and EVAL_ONLY=true mean the same; anything else trains.
case "${EVAL_ONLY:-false}" in 1 | true | True | TRUE | yes) EVAL_ONLY=true ;; *) EVAL_ONLY=false ;; esac

# ---- trainer entry point ----------------------------------------------------------------------------
case "${ENTRY:-opsd}" in
    opsd) MAIN_ENTRYPOINT=opsd.trainer.main_opsd_alfworld ;;
    dapo) MAIN_ENTRYPOINT=opsd.alfworld.main_alfworld_dapo ;;
    *) opsd_die "ENTRY=$ENTRY: expected opsd or dapo" 2 ;;
esac
# The soft overlong buffer lives in the DAPO trainer's postprocess_generator_output; the OPSD trainer reads
# trainer.algorithm.overlong_buffer.* into its config and never applies it. Refuse instead of silently
# accepting a knob that does nothing.
if [ "${OVERLONG_BUFFER_LEN:-0}" -gt 0 ] && [ "$MAIN_ENTRYPOINT" != opsd.alfworld.main_alfworld_dapo ]; then
    opsd_die "OVERLONG_BUFFER_LEN=$OVERLONG_BUFFER_LEN needs ENTRY=dapo (the OPSD trainer does not apply the overlong buffer)" 2
fi

# ---- OPSD GUARD interlocks (the three advantage rewriters and the teacher's ref model) ---------------
# Reasoning-label override, PRE-QUOTED here rather than inline in the hydra arg list below.
# Two traps, both hit for real on the GRPO path:
#  1. `reasoning_prefix=THOUGHTS:` unquoted is valid YAML mapping syntax, so OmegaConf parses it as the DICT
#     {'THOUGHTS': None}. str(dict) then became the label in the prompt AND the reward regex -- perfectly
#     self-consistent, so a run scored 100% compliance while every rollout read "{'THOUGHTS': None} I need
#     to find a spraybottle...". No metric can catch that; only reading a rollout did. The single quotes
#     must reach hydra as part of argv, hence the nested quoting.
#  2. Do NOT put a comment inside the hydra arg list below: a `#` there comments out the rest of the
#     logical line and truncates the args. `bash -n` does NOT catch it.
_RP_ARG="'${ALFWORLD_REASONING_PREFIX:-THOUGHTS:}'"
USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-true}                   # MUST be false when OPSD_MIX_LAMBDA<1
# Auto-correct the one combination that is never intended. The trainer logs a warning for this, but a
# warning in a 10-hour job's stdout is not a control: the run would train on a double-counted teacher
# signal and only look slightly off. Corrected here, loudly, before any GPU time is spent.
if [ "$(awk -v l="${OPSD_MIX_LAMBDA:-1.0}" 'BEGIN{print (l<1.0)?1:0}')" = "1" ] && [ "$USE_KL_IN_REWARD" = "true" ]; then
    echo "OPSD GUARD: OPSD_MIX_LAMBDA=$OPSD_MIX_LAMBDA (<1) with USE_KL_IN_REWARD=true would count the" >&2
    echo "OPSD GUARD: teacher term TWICE (flattened in the reward + per-token in the advantage mix)." >&2
    echo "OPSD GUARD: forcing USE_KL_IN_REWARD=false. Set it explicitly to silence this." >&2
    USE_KL_IN_REWARD=false
fi
# Same double-count for RLSD, whose w_t modulates an A_GRPO that must come from the PURE env reward.
if [ "${OPSD_RLSD:-0}" = "1" ]; then
    if [ "$(awk -v l="${OPSD_MIX_LAMBDA:-1.0}" 'BEGIN{print (l<1.0)?1:0}')" = "1" ]; then
        echo "OPSD GUARD: OPSD_RLSD=1 and OPSD_MIX_LAMBDA=$OPSD_MIX_LAMBDA (<1) both rewrite the" >&2
        echo "OPSD GUARD: advantage. Pick one; leave OPSD_MIX_LAMBDA=1.0 to run RLSD." >&2
        exit 1
    fi
    if [ "$USE_KL_IN_REWARD" = "true" ]; then
        echo "OPSD GUARD: OPSD_RLSD=1 with USE_KL_IN_REWARD=true would count the teacher TWICE" >&2
        echo "OPSD GUARD: (flattened in the reward, again per-token in w_t). Forcing USE_KL_IN_REWARD=false." >&2
        USE_KL_IN_REWARD=false
    fi
fi
# TOKEN_ADV builds the advantage from scratch, so it collides with the other two advantage rewriters and
# needs the PURE env reward (it adds the teacher term itself). The trainer RAISES on both rather than
# warning, which is the right call but only after a full rollout batch has been paid for -- catch it here.
if [ "${OPSD_TOKEN_ADV:-0}" = "1" ]; then
    if [ "$(awk -v l="${OPSD_MIX_LAMBDA:-1.0}" 'BEGIN{print (l<1.0)?1:0}')" = "1" ] || [ "${OPSD_RLSD:-0}" = "1" ]; then
        echo "OPSD GUARD: OPSD_TOKEN_ADV=1, OPSD_MIX_LAMBDA=$OPSD_MIX_LAMBDA (<1) and OPSD_RLSD=$OPSD_RLSD" >&2
        echo "OPSD GUARD: all rewrite the advantage. Pick one." >&2
        exit 1
    fi
    if [ "$USE_KL_IN_REWARD" = "true" ]; then
        echo "OPSD GUARD: OPSD_TOKEN_ADV=1 with USE_KL_IN_REWARD=true would count the teacher TWICE" >&2
        echo "OPSD GUARD: (folded into env[i], again per-token in R). Forcing USE_KL_IN_REWARD=false." >&2
        USE_KL_IN_REWARD=false
    fi
fi
# SkyRL only builds the ref worker when use_kl_loss OR use_kl_in_reward is set (utils.py use_ref_model).
# ema / trust-region are defined against theta_0 and are meaningless without it, so refuse here rather
# than after the first multi-minute rollout batch. Not auto-corrected: flipping USE_KL_LOSS on would
# silently add a KL leash to the run, which is a different algorithm, not a fix.
case "${OPSD_TEACHER:-frozen}" in
  ema|trust-region|trust_region)
    if [ "$USE_KL_LOSS" != "true" ] && [ "$USE_KL_IN_REWARD" != "true" ]; then
        echo "OPSD GUARD: OPSD_TEACHER=$OPSD_TEACHER needs a ref model (theta_0), but USE_KL_LOSS and" >&2
        echo "OPSD GUARD: USE_KL_IN_REWARD are both false, so SkyRL will not build one." >&2
        echo "OPSD GUARD: set USE_KL_LOSS=true, or use OPSD_TEACHER=live if that is what you meant." >&2
        exit 1
    fi ;;
esac
export USE_KL_IN_REWARD

# ---- layout, engines, backend --------------------------------------------------------------------------
NUM_NODES=${NUM_NODES:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
INFERENCE_ENGINE_TP_SIZE=${INFERENCE_ENGINE_TP_SIZE:-1}
NUM_INFERENCE_ENGINES=$(( (NUM_GPUS_PER_NODE * NUM_NODES) / INFERENCE_ENGINE_TP_SIZE ))
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$(( MAX_INPUT_LENGTH + MAX_GENERATE_LENGTH ))}
ALFWORLD_BASE_PORT=${ALFWORLD_BASE_PORT:-9100}
HTTP_ENDPOINT_PORT=${HTTP_ENDPOINT_PORT:-8000}
if [ "$(opsd_ray_mode)" = external ]; then
    HTTP_ENDPOINT_HOST=${HTTP_ENDPOINT_HOST:-$(hostname -I 2>/dev/null | awk '{print $1}')}
fi
HTTP_ENDPOINT_HOST=${HTTP_ENDPOINT_HOST:-127.0.0.1}
ALFWORLD_ADVERTISE_HOST=${ALFWORLD_ADVERTISE_HOST:-$HTTP_ENDPOINT_HOST}
export ALFWORLD_BASE_PORT HTTP_ENDPOINT_PORT HTTP_ENDPOINT_HOST
if [ "$NUM_NODES" -gt 1 ] && [ "$(opsd_ray_mode)" != external ] && [ "$DRY_RUN" != 1 ]; then
    opsd_die "NUM_NODES=$NUM_NODES needs a running multi-node Ray cluster: start it with scripts/common/ray_cluster.sh (see the header of this file) and set RAY_ADDRESS=<head-ip>:<port>" 2
fi

# TRAINING BACKEND: fsdp2 (every 4B arm) or megatron (the 30B MoE). The shared block expands
# MEGATRON_OVERRIDES to NOTHING unless STRATEGY=megatron, so every fsdp2 arm is unchanged; for OPSD it also
# configures the ref worker (the frozen teacher), which unconfigured is a silent ~61 GB/GPU OOM.
# A missing include must never degrade into a differently-configured run: it is fatal.
_MGT_BLOCK="$ROOT/scripts/common/megatron_strategy.sh"
if [ ! -f "$_MGT_BLOCK" ]; then
    echo "FATAL: backend block not found at $_MGT_BLOCK" >&2
    exit 1
fi
# Sourced here for the argv only. Its model pre-stage (PRESTAGE_MODEL=1, a ~61 GB download for the 30B) runs
# after preflight instead (see below), so a busy GPU or port fails the launch before the download.
_PRESTAGE_REQ=${PRESTAGE_MODEL:-0}
PRESTAGE_MODEL=0
# shellcheck disable=SC1090
source "$_MGT_BLOCK"
PRESTAGE_MODEL=$_PRESTAGE_REQ

PROJECT_NAME=${WANDB_PROJECT:-opsd-alfworld}
CHECKPOINT_PATH="$RUN_DIR/ckpts"
EXPORT_PATH="$RUN_DIR/exports"
DATA_RUN_DIR="$RUN_DIR/data"
export OPSD_ROLLOUT_DUMP_DIR="${OPSD_ROLLOUT_DUMP_DIR:-$RUN_DIR/train_rollouts}"

# ---- data: which parquet files this run reads (built below; named here so DRY_RUN prints the same) ----
TRAIN_PARQUET="$ALFWORLD_PARQUET_DIR/train.parquet"
if [ -n "${TASK_IDS_PATH:-}" ] || [ -n "${OPSD_MAX_TRAIN_TASKS:-}" ]; then TRAIN_PARQUET="$DATA_RUN_DIR/train.parquet"; fi
_EVAL_DIR="$ALFWORLD_PARQUET_DIR"
if [ -n "${OPSD_MAX_EVAL_TASKS:-}" ]; then _EVAL_DIR="$DATA_RUN_DIR"; fi
# In-domain eval = valid_seen; valid_unseen (OOD) is added by EVAL_OOD=true (every paper arm).
EVAL_OOD=${EVAL_OOD:-false}
VAL_DATA="['$_EVAL_DIR/validation.parquet']"
if [ "$EVAL_OOD" = "true" ]; then
    VAL_DATA="['$_EVAL_DIR/validation.parquet','$_EVAL_DIR/validation_unseen.parquet']"
fi
# Held-out DEV split (model selection): carved out of the TRAIN pool (disjoint from TASK_IDS_PATH),
# relabelled data_source=alfworld_dev so it lands in its own eval bucket. Validation during training is then
# dev ONLY; seen/unseen are the test set (scored by re-scoring checkpoints, EVAL_CKPT_STEPS).
EVAL_DEV=${EVAL_DEV:-false}
_BUILD_DEV=0
if [ "$EVAL_DEV" = "true" ] && [ -n "${DEV_TASK_IDS_PATH:-}" ]; then
    _BUILD_DEV=1
    TEST_DATA="$VAL_DATA"
    if [ "${EVAL_POOLS_ALL:-false}" == "true" ]; then
        # dev is ADDED to seen+unseen, so every checkpoint gets a seen, an unseen AND a dev score. NB every
        # eval becomes 474 tasks instead of 200, and eval/all/* pools all three sources (it is NOT the
        # seen+unseen combined score: compute that from the per-pool buckets).
        VAL_DATA="['$DATA_RUN_DIR/dev.parquet','$_EVAL_DIR/validation.parquet','$_EVAL_DIR/validation_unseen.parquet']"
    else
        VAL_DATA="['$DATA_RUN_DIR/dev.parquet']"
    fi
fi
[ -z "${TEST_DATA:-}" ] && TEST_DATA="$VAL_DATA"
# PI-UTILITY EVAL SETS: EVAL_HINT_SETS="nohint+pc_key+..." builds ONE eval parquet per set over the SAME
# tasks (EVAL_HINT_TASK_IDS_PATH, from the train pool), each row carrying its set's hint in a `hint` column
# that the env puts into the STUDENT's prompt (ALFWORLD_HINT_PLACEMENT=user|system). Each set is its own
# data_source (eval/alfworld_<set>/*). REPLACES VAL_DATA.
EVAL_HINT_SETS=${EVAL_HINT_SETS:-""}
if [ -n "$EVAL_HINT_SETS" ]; then
    _hint_vd=""
    for _set in $(printf '%s' "$EVAL_HINT_SETS" | tr '+,' '  '); do
        case " $_hint_vd " in *"/eval_hint_$_set.parquet'"*) continue ;; esac
        _hint_vd="$_hint_vd,'$DATA_RUN_DIR/eval_hint_$_set.parquet'"
    done
    VAL_DATA="[${_hint_vd#,}]"; TEST_DATA="$VAL_DATA"
fi

# ---- the ALFWorld server URLs (the pool is started below; the list is deterministic) ---------------------
_url_host="$ALFWORLD_ADVERTISE_HOST"
case "$_url_host" in *:*) _url_host="[$_url_host]" ;; esac
ALFWORLD_SERVER_URLS="["
for ((_i = 0; _i < NUM_ALFWORLD_SERVERS; _i++)); do
    [ "$_i" -gt 0 ] && ALFWORLD_SERVER_URLS+=","
    ALFWORLD_SERVER_URLS+="\"http://$_url_host:$((ALFWORLD_BASE_PORT + _i))\""
done
ALFWORLD_SERVER_URLS+="]"

# ---- EVAL_ONLY vs training ----------------------------------------------------------------------------
if [ "$EVAL_ONLY" == "true" ]; then
    # CKPT_INTERVAL=-1: with epochs=0 the trainer's end-of-run safety net would otherwise write a junk
    # checkpoint into this evaluation run's ckpt_path.
    EPOCHS=0; EVAL_BEFORE_TRAIN=true; EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.4}; EVAL_N_SAMPLES=${EVAL_N_SAMPLES:-1}
    CKPT_INTERVAL=-1
else
    EPOCHS=${TRAINER_EPOCHS:-3}; EVAL_BEFORE_TRAIN=${EVAL_BEFORE_TRAIN:-true}
    EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.4}; EVAL_N_SAMPLES=${EVAL_N_SAMPLES:-1}
fi

# ---- the hydra argument lists ----------------------------------------------------------------------------
# OPSD trainer (opsd/trainer/main_opsd_alfworld.py). Unquoted expansions split exactly as
# they did on the run-file's command line ($MEGATRON_OVERRIDES must split into separate arguments).
# shellcheck disable=SC2206
_opsd_argv() {
    ARGV=(
        data.train_data="['$TRAIN_PARQUET']"
        data.val_data="$VAL_DATA"
        generator.chat_template_kwargs.enable_thinking=$ENABLE_THINKING
        trainer.algorithm.advantage_estimator=$ADVANTAGE_ESTIMATOR
        trainer.algorithm.policy_loss_type=$POLICY_LOSS_TYPE
        trainer.algorithm.loss_reduction=$LOSS_REDUCTION
        trainer.algorithm.grpo_norm_by_std=$GRPO_NORM_BY_STD
        trainer.algorithm.eps_clip_low=$EPS_CLIP_LOW
        trainer.algorithm.eps_clip_high=$EPS_CLIP_HIGH
        trainer.algorithm.use_kl_in_reward=$USE_KL_IN_REWARD
        trainer.algorithm.use_kl_loss=$USE_KL_LOSS
        trainer.algorithm.kl_loss_coef=$KL_LOSS_COEF
        trainer.algorithm.max_seq_len=$MAX_MODEL_LEN
        trainer.policy.model.path=$MODEL_PATH
        trainer.ref.model.path=$MODEL_PATH
        trainer.policy.optimizer_config.lr=$LR
        trainer.policy.optimizer_config.max_grad_norm=$MAX_GRAD_NORM
        trainer.placement.colocate_all=true
        trainer.strategy=$STRATEGY
        $MEGATRON_OVERRIDES
        trainer.use_sample_packing=$USE_SAMPLE_PACKING
        trainer.policy.language_model_only=$LANGUAGE_MODEL_ONLY
        trainer.ref.language_model_only=$LANGUAGE_MODEL_ONLY
        trainer.policy.fsdp_config.cpu_offload=$CPU_OFFLOAD
        trainer.ref.fsdp_config.cpu_offload=$CPU_OFFLOAD
        trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE
        trainer.placement.policy_num_nodes=$NUM_NODES
        trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE
        trainer.placement.ref_num_nodes=$NUM_NODES
        generator.inference_engine.served_model_name=$MODEL_PATH
        generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES
        generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP_SIZE
        generator.inference_engine.backend=vllm
        generator.inference_engine.run_engines_locally=true
        generator.inference_engine.language_model_only=$LANGUAGE_MODEL_ONLY
        generator.inference_engine.weight_sync_backend=nccl
        generator.inference_engine.async_engine=true
        generator.inference_engine.enable_http_endpoint=true
        generator.inference_engine.http_endpoint_host=$HTTP_ENDPOINT_HOST
        generator.inference_engine.http_endpoint_port=$HTTP_ENDPOINT_PORT
        generator.batched=false
        generator.inference_engine.gpu_memory_utilization=${GPU_MEM_UTIL:-0.7}
        generator.inference_engine.enforce_eager=${ENFORCE_EAGER:-false}
        generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN
        generator.inference_engine.engine_init_kwargs.enable_log_requests=true
        environment.env_class="alfworld"
        environment.skyrl_gym.max_env_workers=$MAX_ENV_WORKERS
        environment.skyrl_gym.alfworld.server_urls="$ALFWORLD_SERVER_URLS"
        environment.skyrl_gym.alfworld.env_type="AlfredTWEnv"
        environment.skyrl_gym.alfworld.task_types="[1,2,3,4,5,6]"
        environment.skyrl_gym.alfworld.prompt_style=$PROMPT_STYLE
        environment.skyrl_gym.alfworld.enable_thinking=$ENABLE_THINKING
        environment.skyrl_gym.alfworld.strict_action_outside_think=${STRICT_ACTION_OUTSIDE_THINK:-false}
        environment.skyrl_gym.alfworld.show_admissible=$SHOW_ADMISSIBLE
        environment.skyrl_gym.alfworld.invalid_action_feedback=$INVALID_ACTION_FEEDBACK
        environment.skyrl_gym.alfworld.invalid_action_show_admissible=$INVALID_ACTION_SHOW_ADMISSIBLE
        environment.skyrl_gym.alfworld.max_turns=$MAX_TURNS
        environment.skyrl_gym.alfworld.history_length=$HISTORY_LENGTH
        environment.skyrl_gym.alfworld.seed=0
        environment.skyrl_gym.alfworld.http_timeout=120
        environment.skyrl_gym.alfworld.think_tags_in_prompt=${ALFWORLD_THINK_TAGS_IN_PROMPT:-false}
        environment.skyrl_gym.alfworld.oneshot_example=${ALFWORLD_ONESHOT_EXAMPLE:-false}
        environment.skyrl_gym.alfworld.reasoning_tag=${ALFWORLD_REASONING_TAG:-think}
        environment.skyrl_gym.alfworld.reasoning_style=${ALFWORLD_REASONING_STYLE:-tag}
        environment.skyrl_gym.alfworld.reasoning_prefix="$_RP_ARG"
        environment.skyrl_gym.alfworld.format_reward_coef=${ALFWORLD_FORMAT_REWARD_COEF:-0.0}
        environment.skyrl_gym.alfworld.format_reward_min_think_chars=${ALFWORLD_FORMAT_REWARD_MIN_CHARS:-30}
        environment.skyrl_gym.alfworld.reward_scale=${ALFWORLD_REWARD_SCALE:-1.0}
        environment.skyrl_gym.alfworld.invalid_action_feedback_guide=${INVALID_ACTION_FEEDBACK_GUIDE:-false}
        environment.skyrl_gym.alfworld.passcode_mode=${ALFWORLD_PASSCODE:-false}
        environment.skyrl_gym.alfworld.passcode_salt=${ALFWORLD_PASSCODE_SALT:-alfworld-passcode-v1}
        environment.skyrl_gym.alfworld.hint_placement=${ALFWORLD_HINT_PLACEMENT:-user}
        generator.max_turns=$MAX_TURNS
        generator.use_conversation_multi_turn=true
        generator.append_eos_token_after_stop_str_in_multi_turn=true
        generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING
        generator.n_samples_per_prompt=${N_SAMPLES:-8}
        generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES
        generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH
        generator.sampling_params.temperature=$TRAIN_TEMPERATURE
        generator.sampling_params.top_p=$TOP_P
        generator.sampling_params.top_k=$TOP_K
        generator.sampling_params.min_p=$MIN_P
        "generator.sampling_params.stop=${STOP_TOKENS:-['</action>']}"
        generator.eval_sampling_params.temperature=$EVAL_TEMPERATURE
        generator.eval_sampling_params.top_p=$EVAL_TOP_P
        generator.eval_sampling_params.top_k=$EVAL_TOP_K
        generator.eval_sampling_params.min_p=${EVAL_MIN_P:-$MIN_P}
        generator.eval_sampling_params.additional_kwargs.presence_penalty=${EVAL_PRESENCE_PENALTY:-1.5}
        generator.eval_sampling_params.additional_kwargs.repetition_penalty=${EVAL_REPETITION_PENALTY:-1.0}
        "generator.eval_sampling_params.stop=${STOP_TOKENS:-['</action>']}"
        generator.eval_sampling_params.max_generate_length=$MAX_GENERATE_LENGTH
        generator.max_input_length=$MAX_INPUT_LENGTH
        trainer.max_prompt_length=$MAX_PROMPT_LENGTH
        trainer.epochs=$EPOCHS
        trainer.seed=${TRAINER_SEED:-42}
        trainer.update_epochs_per_batch=1
        trainer.train_batch_size=$TRAIN_BATCH_SIZE
        trainer.policy_mini_batch_size=$TRAIN_BATCH_SIZE
        trainer.micro_forward_batch_size_per_gpu=$MICRO_BATCH
        trainer.micro_train_batch_size_per_gpu=$MICRO_BATCH
        trainer.eval_batch_size=$EVAL_BATCH_SIZE
        trainer.eval_before_train=$EVAL_BEFORE_TRAIN
        trainer.eval_interval=${EVAL_INTERVAL:-30}
        trainer.logger="$LOGGER"
        trainer.project_name="$PROJECT_NAME"
        trainer.run_name="$RUN_NAME"
        trainer.resume_mode="$RESUME_MODE"
        trainer.resume_path=$RESUME_PATH
        trainer.ckpt_interval=${CKPT_INTERVAL:-25}
        trainer.max_ckpts_to_keep=${MAX_CKPTS_TO_KEEP:-3}
        trainer.ckpt_path="$CHECKPOINT_PATH"
        trainer.export_path="$EXPORT_PATH"
    )
}

# GRPO/DAPO trainer (opsd/alfworld/main_alfworld_dapo.py): the argument list of the GRPO
# run-file, which differs from the OPSD one (fixed grpo estimator, no ref-model path, entropy / TIS /
# overlong-buffer knobs, rollout batch dumps).
# shellcheck disable=SC2206
_dapo_argv() {
    ARGV=(
        data.train_data="['$TRAIN_PARQUET']"
        data.val_data="$VAL_DATA"
        generator.chat_template_kwargs.enable_thinking=$ENABLE_THINKING
        trainer.algorithm.advantage_estimator="grpo"
        trainer.algorithm.use_kl_loss=$USE_KL_LOSS
        trainer.algorithm.kl_loss_coef=$KL_LOSS_COEF
        trainer.algorithm.loss_reduction=$LOSS_REDUCTION
        trainer.algorithm.grpo_norm_by_std=$GRPO_NORM_BY_STD
        trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_RATIO_TYPE
        trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TOKEN_TIS_RATIO_CLIP_HIGH
        trainer.algorithm.policy_loss_type=$POLICY_LOSS_TYPE
        trainer.algorithm.eps_clip_low=$EPS_CLIP_LOW
        trainer.algorithm.eps_clip_high=$EPS_CLIP_HIGH
        trainer.algorithm.use_entropy_loss=$USE_ENTROPY_LOSS
        trainer.algorithm.entropy_loss_coef=$ENTROPY_LOSS_COEF
        trainer.algorithm.max_seq_len=$MAX_MODEL_LEN
        trainer.policy.model.path=$MODEL_PATH
        trainer.policy.optimizer_config.lr=$LR
        trainer.policy.optimizer_config.max_grad_norm=$MAX_GRAD_NORM
        trainer.placement.colocate_all=true
        trainer.strategy=$STRATEGY
        $MEGATRON_OVERRIDES
        trainer.use_sample_packing=$USE_SAMPLE_PACKING
        trainer.logprobs_chunk_size=$LOGPROBS_CHUNK_SIZE
        trainer.policy.language_model_only=$LANGUAGE_MODEL_ONLY
        trainer.ref.language_model_only=$LANGUAGE_MODEL_ONLY
        trainer.policy.fsdp_config.cpu_offload=$CPU_OFFLOAD
        trainer.ref.fsdp_config.cpu_offload=$CPU_OFFLOAD
        trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE
        trainer.placement.policy_num_nodes=$NUM_NODES
        trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE
        trainer.placement.ref_num_nodes=$NUM_NODES
        generator.inference_engine.served_model_name=$MODEL_PATH
        generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES
        generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP_SIZE
        generator.inference_engine.backend=vllm
        generator.inference_engine.run_engines_locally=true
        generator.inference_engine.language_model_only=$LANGUAGE_MODEL_ONLY
        generator.inference_engine.weight_sync_backend=nccl
        generator.inference_engine.async_engine=true
        generator.inference_engine.enable_http_endpoint=true
        generator.inference_engine.http_endpoint_host=$HTTP_ENDPOINT_HOST
        generator.inference_engine.http_endpoint_port=$HTTP_ENDPOINT_PORT
        generator.batched=false
        generator.inference_engine.gpu_memory_utilization=${GPU_MEM_UTIL:-0.6}
        generator.inference_engine.enforce_eager=${ENFORCE_EAGER:-false}
        generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN
        generator.inference_engine.engine_init_kwargs.enable_log_requests=true
        environment.env_class="alfworld"
        environment.skyrl_gym.max_env_workers=$MAX_ENV_WORKERS
        environment.skyrl_gym.alfworld.server_urls="$ALFWORLD_SERVER_URLS"
        environment.skyrl_gym.alfworld.env_type="AlfredTWEnv"
        environment.skyrl_gym.alfworld.task_types="[1,2,3,4,5,6]"
        environment.skyrl_gym.alfworld.prompt_style=$PROMPT_STYLE
        environment.skyrl_gym.alfworld.show_admissible=$SHOW_ADMISSIBLE
        environment.skyrl_gym.alfworld.invalid_action_feedback=$INVALID_ACTION_FEEDBACK
        environment.skyrl_gym.alfworld.invalid_action_show_admissible=${INVALID_ACTION_SHOW_ADMISSIBLE:-true}
        environment.skyrl_gym.alfworld.enable_thinking=${ALFWORLD_ENABLE_THINKING:-false}
        environment.skyrl_gym.alfworld.strict_action_outside_think=${STRICT_ACTION_OUTSIDE_THINK:-false}
        environment.skyrl_gym.alfworld.think_tags_in_prompt=${ALFWORLD_THINK_TAGS_IN_PROMPT:-false}
        environment.skyrl_gym.alfworld.oneshot_example=${ALFWORLD_ONESHOT_EXAMPLE:-false}
        environment.skyrl_gym.alfworld.format_reward_coef=${ALFWORLD_FORMAT_REWARD_COEF:-0.0}
        environment.skyrl_gym.alfworld.format_reward_min_think_chars=${ALFWORLD_FORMAT_REWARD_MIN_CHARS:-30}
        environment.skyrl_gym.alfworld.reward_scale=${ALFWORLD_REWARD_SCALE:-1.0}
        environment.skyrl_gym.alfworld.reasoning_tag=${ALFWORLD_REASONING_TAG:-think}
        environment.skyrl_gym.alfworld.reasoning_style=${ALFWORLD_REASONING_STYLE:-tag}
        environment.skyrl_gym.alfworld.reasoning_prefix="$_RP_ARG"
        environment.skyrl_gym.alfworld.invalid_action_feedback_guide=${INVALID_ACTION_FEEDBACK_GUIDE:-false}
        environment.skyrl_gym.alfworld.passcode_mode=${ALFWORLD_PASSCODE:-false}
        environment.skyrl_gym.alfworld.passcode_salt=${ALFWORLD_PASSCODE_SALT:-alfworld-passcode-v1}
        environment.skyrl_gym.alfworld.hint_placement=${ALFWORLD_HINT_PLACEMENT:-user}
        environment.skyrl_gym.alfworld.max_turns=$MAX_TURNS
        environment.skyrl_gym.alfworld.history_length=$HISTORY_LENGTH
        environment.skyrl_gym.alfworld.seed=0
        environment.skyrl_gym.alfworld.http_timeout=120
        generator.max_turns=$MAX_TURNS
        generator.use_conversation_multi_turn=true
        generator.append_eos_token_after_stop_str_in_multi_turn=true
        generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING
        generator.zero_reward_on_non_stop=$ZERO_REWARD_ON_NON_STOP
        trainer.algorithm.overlong_buffer.len=$OVERLONG_BUFFER_LEN
        trainer.algorithm.overlong_buffer.penalty_factor=$OVERLONG_PENALTY_FACTOR
        trainer.algorithm.overlong_buffer.reward_range=$OVERLONG_REWARD_RANGE
        generator.n_samples_per_prompt=${N_SAMPLES:-8}
        generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES
        generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH
        generator.sampling_params.temperature=$TRAIN_TEMPERATURE
        generator.sampling_params.top_p=$TOP_P
        generator.sampling_params.top_k=$TOP_K
        generator.sampling_params.min_p=$MIN_P
        "generator.sampling_params.stop=${STOP_TOKENS:-['</action>']}"
        generator.eval_sampling_params.temperature=$EVAL_TEMPERATURE
        generator.eval_sampling_params.top_p=$EVAL_TOP_P
        generator.eval_sampling_params.top_k=$EVAL_TOP_K
        generator.eval_sampling_params.min_p=$EVAL_MIN_P
        generator.eval_sampling_params.additional_kwargs.presence_penalty=$EVAL_PRESENCE_PENALTY
        generator.eval_sampling_params.additional_kwargs.repetition_penalty=$EVAL_REPETITION_PENALTY
        "generator.eval_sampling_params.stop=${STOP_TOKENS:-['</action>']}"
        generator.eval_sampling_params.max_generate_length=$MAX_GENERATE_LENGTH
        generator.max_input_length=$MAX_INPUT_LENGTH
        trainer.max_prompt_length=$MAX_PROMPT_LENGTH
        trainer.epochs=$EPOCHS
        trainer.seed=${TRAINER_SEED:-42}
        trainer.update_epochs_per_batch=1
        trainer.train_batch_size=$TRAIN_BATCH_SIZE
        trainer.policy_mini_batch_size=$TRAIN_BATCH_SIZE
        trainer.micro_forward_batch_size_per_gpu=$MICRO_BATCH
        trainer.micro_train_batch_size_per_gpu=$MICRO_BATCH
        trainer.eval_batch_size=$EVAL_BATCH_SIZE
        trainer.eval_before_train=$EVAL_BEFORE_TRAIN
        trainer.eval_interval=${EVAL_INTERVAL:-50}
        trainer.logger="$LOGGER"
        trainer.project_name="$PROJECT_NAME"
        trainer.run_name="$RUN_NAME"
        trainer.resume_mode="$RESUME_MODE"
        trainer.resume_path=$RESUME_PATH
        trainer.ckpt_interval=${CKPT_INTERVAL:-20}
        trainer.max_ckpts_to_keep=${MAX_CKPTS_TO_KEEP:-3}
        trainer.ckpt_path="$CHECKPOINT_PATH"
        trainer.export_path="$EXPORT_PATH"
        trainer.dump_data_batch=$DUMP_ROLLOUTS
    )
}

# One trainer invocation (training, an evaluation-only pass, or one checkpoint of a re-score sweep). The
# knobs are read at CALL time, so the re-score loop only reassigns what differs.
_run_trainer() {
    # ---- env-worker invariant: MAX_ENV_WORKERS must cover peak concurrent trajectories ----
    # The generator dispatches every env op onto ONE ThreadPoolExecutor(max_workers=max_env_workers), and
    # env.init() blocks on the exclusive server pool with NO timeout. So a trajectory that cannot get one
    # of the NUM_ALFWORLD_SERVERS servers parks a thread FOREVER. If concurrent trajectories >
    # max_env_workers, every thread ends up parked in acquire(), the trajectories already holding servers
    # can never get a thread to step(), so they never release -> circular wait -> HARD DEADLOCK (GPU at 0%,
    # no error, no timeout). Capping the eval batch costs NOTHING: real concurrency is bounded by the server
    # pool, so extra in-flight trajectories only queue. Applied on every call.
    local _conc=$(( EVAL_BATCH_SIZE * EVAL_N_SAMPLES ))
    if [ "$_conc" -gt "$MAX_ENV_WORKERS" ]; then
        local _capped=$(( MAX_ENV_WORKERS / EVAL_N_SAMPLES ))
        if [ "$_capped" -lt 1 ]; then _capped=1; fi   # explicit if, not `[ ] && x=1` (a set -e tripwire)
        echo "env-worker guard: eval_batch*N=$_conc > max_env_workers=$MAX_ENV_WORKERS -> EVAL_BATCH_SIZE $EVAL_BATCH_SIZE -> $_capped (avoids the pool.acquire() deadlock)" >&2
        EVAL_BATCH_SIZE=$_capped
    fi
    # ---- RESUME_RESET_DATALOADER (opt-in): drop the checkpoint's dataloader state ----
    # A checkpoint written at the END of an epoch carries an EXHAUSTED dataloader iterator; the trainer
    # (OPSDRayPPOTrainer.load_checkpoints) now drains it itself, so a boundary resume trains a full next epoch
    # without this. Dropping data.pt instead restarts the data order from the seeded generator, and is WRONG
    # mid-epoch (it would re-train already-seen batches); the launcher cannot tell which case it is.
    if [ "${RESUME_RESET_DATALOADER:-0}" = "1" ] && [ "$RESUME_MODE" = "from_path" ] && [ -n "$RESUME_PATH" ]; then
        if [ "$DRY_RUN" = 1 ]; then
            echo "RESUME_RESET_DATALOADER=1: would drop $RESUME_PATH/data.pt" >&2
        elif [ -f "$RESUME_PATH/data.pt" ]; then
            echo "RESUME_RESET_DATALOADER=1: dropping $RESUME_PATH/data.pt so the next epoch is a FULL pass" >&2
            rm -f "$RESUME_PATH/data.pt"
        else
            echo "RESUME_RESET_DATALOADER=1: no data.pt at $RESUME_PATH (already absent) -- nothing to drop" >&2
        fi
    fi
    if [ "$MAIN_ENTRYPOINT" = opsd.alfworld.main_alfworld_dapo ]; then _dapo_argv; else _opsd_argv; fi
    if [ "$DRY_RUN" = 1 ]; then
        opsd_dry_run_print "$MAIN_ENTRYPOINT" "${ARGV[@]}"
        return 0
    fi
    echo "train entrypoint: $MAIN_ENTRYPOINT (run $RUN_NAME in $RUN_DIR)" >&2
    opsd_run "$PY" -m "$MAIN_ENTRYPOINT" "${ARGV[@]}"
}

# ---- helpers for the real run ----------------------------------------------------------------------------
_http_ok() {
    if command -v curl >/dev/null 2>&1; then
        curl -sf -m 2 "$1" >/dev/null 2>&1
    else
        "$PY" -c 'import sys, urllib.request as u; u.urlopen(sys.argv[1], timeout=2)' "$1" >/dev/null 2>&1
    fi
}

# Wait for the env pool to be READY rather than sleeping a fixed time: at 128 servers the TextWorld import
# and startup PDDL validation run 128x concurrently, and the first rollouts must not hit unbound servers.
_alfworld_wait_ready() {
    local n_ready=0 deadline i p alive
    deadline=$(( $(date +%s) + ${ALFWORLD_READY_TIMEOUT_S:-600} ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        n_ready=0
        for ((i = 0; i < NUM_ALFWORLD_SERVERS; i++)); do
            if _http_ok "http://127.0.0.1:$((ALFWORLD_BASE_PORT + i))/health"; then n_ready=$((n_ready + 1)); fi
        done
        [ "$n_ready" -ge "$NUM_ALFWORLD_SERVERS" ] && break
        alive=0
        for p in $ALFWORLD_SERVE_PIDS; do kill -0 "$p" 2>/dev/null && alive=$((alive + 1)); done
        if [ "$alive" -eq 0 ]; then
            echo "ERROR: every ALFWorld server exited; see $RUN_DIR/logs/alfworld/alfworld_serve_*.log" >&2
            return 1
        fi
        sleep 5
    done
    echo "ALFWorld env pool ready: $n_ready/$NUM_ALFWORLD_SERVERS servers healthy" >&2
    if [ "$n_ready" -eq 0 ]; then
        echo "ERROR: no ALFWorld server became healthy within ${ALFWORLD_READY_TIMEOUT_S:-600}s" >&2
        return 1
    fi
    if [ "$n_ready" -lt "$NUM_ALFWORLD_SERVERS" ]; then
        echo "WARNING: only $n_ready/$NUM_ALFWORLD_SERVERS env servers healthy -- rollouts will contend for fewer slots" >&2
    fi
    return 0
}

if [ "${FINAL_TEST:-false}" = "true" ] && [ "$EVAL_ONLY" != "true" ] && [ -z "${EVAL_CKPT_STEPS:-}" ]; then
    echo "NOTE: this arm validated on dev and was tested afterwards; that chained test is not part of this" >&2
    echo "      launcher. Score the saved checkpoints on valid_seen + valid_unseen once training is done:" >&2
    echo "      $0 $ARM_ID EVAL_CKPT_STEPS=<steps, e.g. 25+50+75+100>" >&2
    echo "      plus the arm's re-score overrides, if any (\`rescore:\` in python -m opsd.recipes show $ARM_ID)" >&2
fi

# ---- the arm's hint file -----------------------------------------------------------------------------------
# A missing OPSD_HINTS_PATH is not an error to the trainer: it gives every task its generic static hint, which
# is a different PI. So a TRAINING run refuses it even with OPSD_PREFLIGHT=0. A hint file under
# appworld/plain/opsd_appworld_<x>/ (the cross-domain null) ships inside the encrypted AppWorld bundle
# appworld/appworld_opsd_<x>.bundle, and the refusal says how to fetch and decrypt it.
_hint_gate() {  # MODE: warn (dry run) | die
    local h="${OPSD_HINTS_PATH:-}" b
    case "$h" in "" | null | None) return 0 ;; esac
    [ -f "$h" ] && return 0
    [ "$1" = warn ] && return 0
    case "$h" in
        */appworld/plain/opsd_appworld_*/*)
            b="${h#*/appworld/plain/opsd_appworld_}"
            b="appworld/appworld_opsd_${b%%/*}.bundle"
            opsd_die "hint file not found: $h (OPSD_HINTS_PATH). It ships encrypted in the dataset's $b: fetch it (python -m opsd.data.fetch --repo <hf_dataset_id> --revision <commit_sha> --bench appworld --include $b) and decrypt it with bash scripts/setup/setup_appworld.sh, which builds the AppWorld venv the decryption needs. Without it the trainer would silently fall back to its generic static hint."
            ;;
    esac
    opsd_die "hint file not found: $h (OPSD_HINTS_PATH). Without it the trainer would silently fall back to its generic static hint; fetch the data bundle (python -m opsd.data.fetch) or pass OPSD_HINTS_PATH=<file>"
}

# ---- DRY_RUN: print and stop (no GPU, no servers, no files) -----------------------------------------------
if [ "$DRY_RUN" = 1 ]; then
    if [ "$EVAL_ONLY" != true ] && [ -z "${EVAL_CKPT_STEPS:-}" ]; then _hint_gate warn; fi
    if [ -n "${EVAL_CKPT_STEPS:-}" ]; then
        VAL_DATA="$TEST_DATA"; EPOCHS=0; EVAL_BEFORE_TRAIN=true; EVAL_N_SAMPLES=${EVAL_CKPT_N_SAMPLES:-4}
        CKPT_INTERVAL=-1; RESUME_MODE=from_path    # RUN_NAME is already the re-score's (<training>__ckpteval)
        for _step in $(printf '%s' "$EVAL_CKPT_STEPS" | tr '+,' '  '); do
            RESUME_PATH="${EVAL_CKPT_DIR:-$CHECKPOINT_PATH}/global_step_$_step"
            EXPORT_PATH="$RUN_DIR/exports_ckpt_$_step"
            _run_trainer
        done
    else
        _run_trainer
    fi
    exit 0
fi

# ---- preflight, provenance, processes we own ----------------------------------------------------------------
if [ "$EVAL_ONLY" != true ] && [ -z "${EVAL_CKPT_STEPS:-}" ]; then _hint_gate die; fi
case "${OPSD_HINTS_PATH:-}" in
    */appworld/plain/opsd_appworld_xdnull/*)
        opsd_warn "$ARM_ID: evaluation-split material -- the AppWorld donor text of this hint file is the reference solution of AppWorld test_normal tasks. Never evaluate the trained model on AppWorld test_normal; the arm reproduces the paper's control only." ;;
esac
# shellcheck disable=SC1091
source "$ROOT/scripts/common/preflight.sh"
opsd_preflight_all || exit 1
[ -n "$EVAL_HINT_SETS" ] && { opsd_preflight_files "${EVAL_HINT_TASK_IDS_PATH:?EVAL_HINT_SETS needs EVAL_HINT_TASK_IDS_PATH}" || exit 1; }
if [ "$EVAL_ONLY" != true ] && [ -z "${EVAL_CKPT_STEPS:-}" ] && [ "${RESUME_MODE:-null}" = null ] \
        && compgen -G "$CHECKPOINT_PATH/global_step_*" >/dev/null; then
    opsd_warn "RESUME_MODE=null (as this arm ran) but $CHECKPOINT_PATH already holds checkpoints: this launch starts again from step 0 and overwrites them. Pass RESUME_MODE=latest to continue that run instead."
fi
mkdir -p "$RUN_DIR/logs/alfworld" "$OPSD_ROLLOUT_DUMP_DIR" "$DATA_RUN_DIR"
opsd_write_provenance "$OPSD_RESOLVED_ENV"
opsd_install_cleanup_trap

# ---- model pre-stage (PRESTAGE_MODEL=1, the 30B student): one download into THIS node's HF cache -----------
# The shared block's own pre-stage, run in a subshell (it only needs MODEL_PATH / PRESTAGE_MODEL / HF_HOME).
# The worker node of a 2-node run downloads separately (see the header).
if [ "${PRESTAGE_MODEL:-0}" = 1 ]; then
    # shellcheck disable=SC1090
    ( source "$_MGT_BLOCK" ) >&2 || opsd_die "the model pre-stage failed"
fi

# ---- parquets: the shared per-machine build (idempotent), then this run's subsets ----------------------------
if [ ! -f "$ALFWORLD_PARQUET_DIR/train.parquet" ] || [ ! -f "$ALFWORLD_PARQUET_DIR/validation.parquet" ] \
        || [ ! -f "$ALFWORLD_PARQUET_DIR/validation_unseen.parquet" ]; then
    [ -x "$ALFWORLD_VENV/bin/python" ] || opsd_die "no ALFWorld venv at $ALFWORLD_VENV: run scripts/setup/setup_alfworld.sh"
    "$ALFWORLD_VENV/bin/python" "$ROOT/opsd/alfworld/alfworld_dataset.py" \
        --local_dir "$ALFWORLD_PARQUET_DIR" --alfworld_data "$ALFWORLD_DATA" \
        --splits train valid_seen valid_unseen || opsd_die "building the ALFWorld parquets failed"
fi
_prep=(--parquet-dir "$ALFWORLD_PARQUET_DIR" --out-dir "$DATA_RUN_DIR")
[ -n "${TASK_IDS_PATH:-}" ] && _prep+=(--task-ids "$TASK_IDS_PATH")
[ -n "${OPSD_MAX_TRAIN_TASKS:-}" ] && _prep+=(--max-train-tasks "$OPSD_MAX_TRAIN_TASKS")
[ -n "${OPSD_MAX_EVAL_TASKS:-}" ] && _prep+=(--max-eval-tasks "$OPSD_MAX_EVAL_TASKS")
[ "$_BUILD_DEV" = 1 ] && _prep+=(--dev --dev-task-ids "$DEV_TASK_IDS_PATH")
"$PY" "$_RUN_SH_DIR/prepare_data.py" "${_prep[@]}" || opsd_die "preparing the run's parquets failed"
if [ -n "$EVAL_HINT_SETS" ]; then
    _vd="$("$PY" -m opsd.alfworld.build_hint_eval_sets \
        --base-parquet "$ALFWORLD_PARQUET_DIR/train.parquet" --hint-dir "${EVAL_HINT_DIR:-}" \
        --sets "$EVAL_HINT_SETS" --task-ids "$EVAL_HINT_TASK_IDS_PATH" \
        --placement "${ALFWORLD_HINT_PLACEMENT:-user}" --out-dir "$DATA_RUN_DIR" | tail -1)" \
        || opsd_die "building the PI-utility eval sets failed"
    [ "$_vd" = "$VAL_DATA" ] || opsd_warn "PI-utility eval sets were written as $_vd (expected $VAL_DATA); using them"
    VAL_DATA="$_vd"; TEST_DATA="$VAL_DATA"
    echo "PI-utility eval: VAL_DATA=$VAL_DATA" >&2
fi

# ---- the ALFWorld server pool (its own venv; PIDs tracked so cleanup stops only these) -----------------------
_POOL_OUT="$(ALFWORLD_VENV="$ALFWORLD_VENV" ALFWORLD_DATA="$ALFWORLD_DATA" \
    NUM_ALFWORLD_SERVERS="$NUM_ALFWORLD_SERVERS" ALFWORLD_BASE_PORT="$ALFWORLD_BASE_PORT" \
    ALFWORLD_MAX_STEPS="$ALFWORLD_MAX_STEPS" ALFWORLD_ADVERTISE_HOST="$ALFWORLD_ADVERTISE_HOST" \
    ALFWORLD_LOG_DIR="$RUN_DIR/logs/alfworld" ALFWORLD_PID_DIR="$RUN_DIR/alfworld_servers" \
    bash "$ROOT/opsd/alfworld/server/launch_alfworld_servers.sh" | tail -1)" \
    || opsd_die "starting the ALFWorld server pool failed (see above)"
ALFWORLD_SERVE_PIDS="$(cat "$RUN_DIR/alfworld_servers/alfworld_serve.pids" 2>/dev/null)"
for _p in $ALFWORLD_SERVE_PIDS; do opsd_track_pid "$_p" alfworld-server || true; done
[ -n "$ALFWORLD_SERVE_PIDS" ] || opsd_die "the ALFWorld server launcher recorded no PIDs in $RUN_DIR/alfworld_servers"
if [ "$_POOL_OUT" != "$ALFWORLD_SERVER_URLS" ]; then
    opsd_warn "server pool URLs differ from the expected list; using the launcher's: $(printf '%s' "$_POOL_OUT" | cut -c1-120)..."
    ALFWORLD_SERVER_URLS="$_POOL_OUT"
fi
_alfworld_wait_ready || exit 1

# Group the training run with its evaluation re-scores in the tracker.
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$RUN_NAME}"

# ---- EVAL_CKPT_STEPS: re-score saved checkpoints on valid_seen + valid_unseen, no training ----------------
# One launch per run, not per checkpoint: the pool and data are set up once and each step re-executes the
# trainer (which re-initialises the engines). Metrics go to the re-score's own tracker run (RUN_NAME =
# "<training RUN_NAME>__ckpteval[_<EVAL_CKPT_TAG>]", set above), never the training run (a tracker drops
# log(step=N) once its pointer is past N), each logged at the checkpoint's original training step, so the
# curve shares the training run's x-axis.
if [ -n "${EVAL_CKPT_STEPS:-}" ]; then
    _SRC_CKPT_DIR="${EVAL_CKPT_DIR:-$CHECKPOINT_PATH}"
    _CKPT_RUN="$RUN_NAME"
    VAL_DATA="$TEST_DATA"                        # seen + unseen; dev is NOT in this report
    EPOCHS=0; EVAL_BEFORE_TRAIN=true             # load ckpt, eval, stop
    EVAL_N_SAMPLES=${EVAL_CKPT_N_SAMPLES:-4}
    CKPT_INTERVAL=-1                             # else the end-of-run safety net writes a junk ckpt
    RESUME_MODE=from_path
    RUN_NAME="$_CKPT_RUN"
    export WANDB_RUN_GROUP="$_CKPT_RUN"
    export WANDB_RUN_ID="${WANDB_RUN_ID:-$(printf '%s' "$_CKPT_RUN" | tr -c 'a-zA-Z0-9' '_' | cut -c1-60)}"
    export WANDB_RESUME=allow
    echo "=== CKPT SWEEP: $_SRC_CKPT_DIR steps=[$EVAL_CKPT_STEPS] -> run $_CKPT_RUN, pools=$VAL_DATA N=$EVAL_N_SAMPLES ===" >&2
    _swept=0; _failed=""
    for _step in $(printf '%s' "$EVAL_CKPT_STEPS" | tr '+,' '  '); do
        RESUME_PATH="$_SRC_CKPT_DIR/global_step_$_step"
        if [ ! -d "$RESUME_PATH/policy" ]; then
            echo "    !! no checkpoint at $RESUME_PATH -- skipping, it will be MISSING from the curve" >&2
            _failed="$_failed $_step"; continue
        fi
        export SKYRL_LOG_STEP_OVERRIDE="$_step"  # land on the ORIGINAL training step
        EXPORT_PATH="$RUN_DIR/exports_ckpt_$_step"
        echo "=== [ckpt $_step] eval on seen+unseen, N=$EVAL_N_SAMPLES ===" >&2
        # COUNT SUCCESSES, NOT ATTEMPTS: a trainer that dies on a config assert must not count as scored.
        if _run_trainer; then
            _swept=$(( _swept + 1 ))
        else
            echo "    !! trainer FAILED for step $_step -- not counted, MISSING from the curve" >&2
            _failed="$_failed $_step"
        fi
    done
    echo "=== CKPT SWEEP DONE: $_swept evaluated${_failed:+, SKIPPED:$_failed} ===" >&2
    if [ "$_swept" -eq 0 ]; then
        echo "FATAL: ckpt sweep evaluated 0 checkpoints -- see the SKIPPED list above" >&2
        exit 1
    fi
    exit 0
fi

_run_trainer; TRAIN_RC=$?
exit "$TRAIN_RC"

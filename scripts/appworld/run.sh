#!/usr/bin/env bash
# =====================================================================================================
# AppWorld launcher: one arm of the paper per invocation.
#
#   bash scripts/appworld/run.sh <arm_id> [--overlay NAME]... [KEY=VALUE ...]     (from any directory)
#
#   bash scripts/appworld/run.sh appworld/main/4bi/glm_aot                     # vanilla OPSD
#   bash scripts/appworld/run.sh appworld/grpo/4bi/none/seed=1                 # DAPO baseline
#   bash scripts/appworld/run.sh appworld/opsd_grpo/4bi/glm_aot/lambda=0.99/seed=1
#   bash scripts/appworld/run.sh appworld/rlsd/4bi/glm_aot                     # DAPO + RLSD
#   bash scripts/appworld/run.sh appworld/step0/4bi/none                       # untrained student
#   bash scripts/appworld/run.sh appworld/main/4bi/glm_chk --overlay smoke_4gpu  # 4-GPU smoke (own
#                                                     RUN_DIR, runs/appworld__main__4bi__glm_chk__smoke_4gpu)
#   DRY_RUN=1 bash scripts/appworld/run.sh appworld/main/30bi/glm_aot          # print the command
#   python -m opsd.recipes list --bench appworld                             # every arm id
#
# What it does: resolves the arm's recipe (opsd/recipes/appworld/) into $RUN_DIR/resolved.env and sources
# it (hermetic: an ambient OPSD_* variable that is not part of the recipe is an error -- pass overrides as
# KEY=VALUE), builds the AppWorld parquets once (opsd/appworld/appworld_dataset.py, run with
# the AppWorld venv), checks the machine (GPUs, ports, disk, hint files + their sha256), starts its own pool
# of supervised AppWorld environment servers, writes $RUN_DIR/provenance.json and runs the trainer
# (opsd.appworld.main_appworld for DAPO/GRPO and evaluation, opsd.trainer.
# main_opsd_appworld for every OPSD variant) with the full hydra command, from the repository root with the
# repository root first on PYTHONPATH. Exit status = the trainer's.
# It never stops Ray and never signals a process it did not start (its PIDs live in $RUN_DIR/pids/).
# Paths are relative to the directory you launch from, whether given as KEY=VALUE or in the environment: the
# launcher makes them absolute before it changes into the repository root for the trainer. That covers
# RUN_ROOT / RUN_DIR / OPSD_DATA_ROOT, OPSD_HINTS_PATH, RESUME_PATH, OPSD_ROLLOUT_DUMP_DIR, OPSD_EMA_DIR and
# APPWORLD_DATA_DIR; MODEL_PATH only when it names an existing local path, so a Hugging Face id (Qwen/...) stays
# as given.
#
# Prerequisites
#   * the training environment: bash scripts/setup/setup_train_env.sh (the pinned SkyRL checkout next to the
#     repository + uv sync --frozen --extra fsdp; --extra megatron for the 30bi arms)
#   * AppWorld: bash scripts/setup/setup_appworld.sh (separate venv -- AppWorld 0.1.x pins pydantic<2
#     -- plus its data, the decrypted PI bundle and the b0a8eae task patch). Without the bundle the train
#     split has 87 tasks instead of 90 and the launcher refuses to train unless APPWORLD_ALLOW_UNPATCHED=1.
#   * the PI data bundle under $OPSD_DATA_ROOT, fetched BEFORE the setup script unpacks it:
#     python -m opsd.data.fetch --repo <hf_dataset_id> --revision <commit_sha> --bench appworld
#     (or --from-local <unpacked bundle copy> --bench appworld)
#
# Hardware: 4B arms 1 node x 8 GPUs (80 GB). 30bi DAPO 1 node x 8 GPUs; 30bi OPSD (and step0/30bi) 2 nodes:
# start a Ray cluster with scripts/common/ray_cluster.sh (head + 1 worker), then run this on the head
# with RAY_ADDRESS=<head_ip>:<port>. RUN_DIR (checkpoints are written by every rank) and OPSD_DATA_ROOT must
# be on a filesystem shared by both nodes. The AppWorld servers and the training loop stay on the head
# (servers are addressed on 127.0.0.1 and recycled through /proc). The 240 AppWorld servers of the paper
# runs assume ~192 vCPUs and ~2 TB of host RAM; lower NUM_APPWORLD_SERVERS on smaller hosts.
# Model weights (30bi, 61 GB, PRESTAGE_MODEL=1): downloaded once on the head, then once on EVERY node of the
# Ray cluster (one Ray task pinned to each node, into that node's HF_HOME) before the trainer starts, so the
# 16 ranks never pull the HF id concurrently (that has failed with "no file named ...shard" at load time).
# HF_HOME may be node-local or shared; set it identically for `ray_cluster.sh head|worker` and this script.
#
# Disk (checked by the preflight under RUN_ROOT before anything starts): checkpoints dominate. The trainer
# saves at every epoch end (every 5 steps at 90 tasks / batch 18) and every CKPT_INTERVAL steps, and removes
# the oldest beyond MAX_CKPTS_TO_KEEP only after the new one is written, so a run peaks at
# min(saves, MAX_CKPTS_TO_KEEP + 1) checkpoints of CKPT_SIZE_GB each (recipe key). 4B arms keep every one of
# their 30 checkpoints (~45 GB each, as the paper runs; FINAL_TEST picks among them): ~1.4 TB per arm.
# 30bi (~400 GB each, MAX_CKPTS_TO_KEEP=3): ~1.65 TB. EVAL_ONLY / step0: none. The free-space floor is that
# peak (minus checkpoints already in $RUN_DIR/ckpts when resuming) + 50 GB; MAX_CKPTS_TO_KEEP=N lowers it to
# N+1 checkpoints (FINAL_TEST then chooses among the last N), OPSD_PREFLIGHT_MIN_FREE_GB=<GB> replaces it.
#
# Knobs (not part of the recipe):
#   DRY_RUN=1                 print the entry module, the full argv and the resolved environment, then exit 0
#                             (no GPU, no servers, no files under RUN_DIR); DRY_RUN_ARGV_FILE=<path> also
#                             writes the module + argv NUL-separated
#   APPWORLD_VENV / APPWORLD_ROOT   AppWorld venv and data root (default ~/appworld-venv, ~/appworld_root)
#   APPWORLD_DATA_DIR         parquet dir (default $OPSD_DATA_ROOT/appworld/parquet/<PROMPT_VARIANT>)
#   APPWORLD_BASE_PORT        first AppWorld server port (default 21000; NUM_APPWORLD_SERVERS consecutive
#                             ports, outside vLLM's and Ray's usual ranges). A busy port is never reused.
#   APPWORLD_READY_TIMEOUT    seconds to wait for the server pool (default 900)
#   APPWORLD_ALLOW_UNPATCHED=1  train on the 87-task split when the b0a8eae patch is missing (not the paper)
#   HTTP_ENDPOINT_HOST/PORT   the policy's HTTP inference endpoint (default 127.0.0.1 / 8000; the node's
#                             address on multi-node runs)
#   WANDB_PROJECT             W&B project (default opsd-appworld); logger = wandb when WANDB_API_KEY is set
#   TRAINER_STDOUT_CAPTURE=0  do not tee the trainer's output into $RUN_DIR/logs/trainer-<time>.log
#   OPSD_PREFLIGHT_MIN_FREE_GB  free GB required under RUN_ROOT (default: the checkpoint footprint above)
# Recipe keys worth knowing (pass as KEY=VALUE): RESUME_MODE=latest (default: resume from $RUN_DIR/ckpts
# when a checkpoint exists) or RESUME_PATH=<run_dir>/ckpts/global_step_N; EVAL_ONLY=1 (evaluate the base
# model or RESUME_PATH, no training; re-invoking a trained arm keeps that run's resolved.env / provenance.json /
# command.sh and records this invocation in resolved.eval-<sha>.env etc., exports in $RUN_DIR/exports_eval);
# FINAL_TEST=true (after training, test the best checkpoint of the validation curve -- or
# FINAL_TEST_CKPT_STEP -- on test_normal, + test_challenge with EVAL_OOD=true); PRESTAGE_MODEL=0|1 (1 for
# 30bi: pre-download the weights on every node, see above); MAX_CKPTS_TO_KEEP / CKPT_SIZE_GB (disk, above).
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
    "" | -h | --help)
        sed -n '2,/^# =====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        [ -n "${1:-}" ] && exit 0
        exit 2
        ;;
esac
ARM_ID="$1"
shift

# shellcheck source=../common/bootstrap.sh
source "$ROOT/scripts/common/bootstrap.sh"
# shellcheck source=./servers.sh
source "$ROOT/scripts/appworld/servers.sh"

DRY=${DRY_RUN:-0}
opsd_resolve_recipe "$ARM_ID" "$@" || exit 1
[ "${OPSD_BENCH:-}" = appworld ] || opsd_die "arm $ARM_ID is not an AppWorld arm (bench '${OPSD_BENCH:-?}')"

_missing=()
for _k in TRAIN_ENTRYPOINT MODEL_PATH PROMPT_VARIANT SCAFFOLD STRICT_PARSE REWARD_MODE MAX_TURNS MAX_INTERACTIONS \
    VAL_ON_TEST N_SAMPLES TRAIN_BATCH_SIZE TRAINER_EPOCHS EVAL_INTERVAL EVAL_N_SAMPLES EVAL_BATCH_SIZE \
    TEMPERATURE TOP_P TOP_K MIN_P EVAL_TEMPERATURE EVAL_TOP_P EVAL_TOP_K EVAL_MIN_P \
    MAX_MODEL_LEN MAX_GENERATE_LENGTH MAX_PROMPT_LENGTH ADVANTAGE_ESTIMATOR POLICY_LOSS_TYPE LOSS_REDUCTION \
    GRPO_NORM_BY_STD EPS_CLIP_LOW EPS_CLIP_HIGH USE_KL_LOSS USE_KL_IN_REWARD KL_LOSS_COEF LR OVERLONG_BUFFER_LEN \
    TRAINER_SEED NUM_NODES NUM_GPUS_PER_NODE STRATEGY INFERENCE_ENGINE_TP_SIZE GPU_MEM_UTIL \
    NUM_APPWORLD_SERVERS MAX_ENV_WORKERS CKPT_INTERVAL MAX_CKPTS_TO_KEEP RESUME_MODE; do
    [ -n "${!_k:-}" ] || _missing+=("$_k")
done
[ "${#_missing[@]}" = 0 ] || opsd_die "the resolved recipe does not set: ${_missing[*]} (see opsd/recipes/appworld/base.env)"

PY="$(opsd_python)"
export PATH="$(dirname "$PY"):$PATH"  # python / hf / ray of the training environment first
# The repository root first: `opsd` is the package there (skyrl and skyrl_gym come from the SkyRL install, never
# PYTHONPATH)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset VLLM_API_KEY  # if set, vLLM enforces auth on the HTTP endpoint and the generator's requests fail
aw_resolve_paths
[ "$PROMPT_VARIANT" = "$SCAFFOLD" ] || opsd_warn "PROMPT_VARIANT=$PROMPT_VARIANT but SCAFFOLD=$SCAFFOLD (the prompt teaches the format the parser enforces)"

case "$TRAIN_ENTRYPOINT" in
    opsd.appworld.main_appworld | opsd.trainer.main_opsd_appworld) ;;
    *) opsd_die "TRAIN_ENTRYPOINT=$TRAIN_ENTRYPOINT is not an AppWorld entrypoint" ;;
esac
if [ "$USE_KL_LOSS" = true ] && [ "$USE_KL_IN_REWARD" = true ]; then
    opsd_die "USE_KL_LOSS and USE_KL_IN_REWARD are mutually exclusive (vanilla OPSD wants USE_KL_LOSS=false)"
fi
if [ "${OPSD_SMOKE:-0}" = 1 ] && [ "$STRATEGY" = megatron ]; then
    opsd_die "the smoke overlay is for the 4B students only (the 30bi arms need 8-16 GPUs)"
fi

# ---- run layout --------------------------------------------------------------------------------------
CKPT_DIR="$RUN_DIR/ckpts"
EXPORT_PATH="$RUN_DIR/exports"
LOG_DIR="$RUN_DIR/logs"
APPWORLD_DATA_DIR="${APPWORLD_DATA_DIR:-$OPSD_DATA_ROOT/appworld/parquet/$PROMPT_VARIANT}"
# The trainer runs from the repository root: a relative parquet dir is the caller's (the recipe resolution
# already made the KEY=VALUE input paths and a local MODEL_PATH absolute).
case "$APPWORLD_DATA_DIR" in /*) ;; *) APPWORLD_DATA_DIR="$PWD/$APPWORLD_DATA_DIR" ;; esac
APPWORLD_BASE_PORT="${APPWORLD_BASE_PORT:-21000}"
export APPWORLD_BASE_PORT  # preflight checks APPWORLD_BASE_PORT .. +NUM_APPWORLD_SERVERS-1
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$RUN_NAME}"
PROJECT_NAME="${WANDB_PROJECT:-opsd-appworld}"

# ---- evaluation-only mode ------------------------------------------------------------------------------
case "${EVAL_ONLY:-0}" in
    1 | true) EVAL_ONLY=1 ;;
    0 | false | "") EVAL_ONLY=0 ;;
    *) opsd_die "EVAL_ONLY must be 0 or 1 (got '$EVAL_ONLY')" ;;
esac
if [ "$EVAL_ONLY" = 1 ]; then
    # Load (RESUME_PATH) or not (base model), evaluate once (eval_before_train), stop. ckpt_interval=-1 keeps
    # the end-of-run safety net from writing a checkpoint; from_path skips the latest-checkpoint
    # consistency check, which misfires with ckpt_interval=-1.
    TRAINER_EPOCHS=0
    EVAL_BEFORE_TRAIN=true
    CKPT_INTERVAL=-1
    FINAL_TEST=false
    EXPORT_PATH="$RUN_DIR/exports_eval"
    if [ -n "${RESUME_PATH:-}" ]; then RESUME_MODE=from_path; else RESUME_MODE=none; fi
fi
# A lone RESUME_PATH would otherwise be ignored by resume_mode=latest.
if [ -n "${RESUME_PATH:-}" ] && [ "$RESUME_MODE" = latest ]; then
    RESUME_MODE=from_path
    opsd_log "RESUME_PATH given -> RESUME_MODE=from_path ($RESUME_PATH)"
fi
case "$RESUME_MODE" in
    latest | none | null) ;;
    from_path) [ -n "${RESUME_PATH:-}" ] || opsd_die "RESUME_MODE=from_path needs RESUME_PATH=<run_dir>/ckpts/global_step_N" ;;
    *) opsd_die "RESUME_MODE must be latest | from_path | none (got '$RESUME_MODE')" ;;
esac

# ---- data -------------------------------------------------------------------------------------------
TRAIN_PARQUET="$APPWORLD_DATA_DIR/train.parquet"
DEV_PARQUET="$APPWORLD_DATA_DIR/validation.parquet"
NORMAL_PARQUET="$APPWORLD_DATA_DIR/eval_normal.parquet"
CHALLENGE_PARQUET="$APPWORLD_DATA_DIR/eval_challenge.parquet"
if [ "${TRAIN_INCLUDE_DEV:-0}" = 1 ]; then
    # Training on dev is only legitimate when validation moves off dev.
    [ "$VAL_ON_TEST" = 1 ] || opsd_die "TRAIN_INCLUDE_DEV=1 requires VAL_ON_TEST=1 (dev would be both train and validation data)"
    TRAIN_PARQUET="$APPWORLD_DATA_DIR/train_dev.parquet"
fi
case "$VAL_ON_TEST" in
    1) VAL_FILES=("$NORMAL_PARQUET") ;;               # test_normal-168  -> eval/appworld_test_normal/*
    both) VAL_FILES=("$DEV_PARQUET" "$NORMAL_PARQUET") ;;  # + dev-57      -> eval/appworld/*
    0) VAL_FILES=("$DEV_PARQUET") ;;
    *) opsd_die "VAL_ON_TEST must be 0, 1 or both (got '$VAL_ON_TEST')" ;;
esac
TEST_FILES=("$NORMAL_PARQUET")
[ "${EVAL_OOD:-false}" = true ] && TEST_FILES+=("$CHALLENGE_PARQUET")
# train tasks (sizes the checkpoint footprint): counted from the parquet on a real run
N_TRAIN_TASKS=90
[ "${TRAIN_INCLUDE_DEV:-0}" = 1 ] && N_TRAIN_TASKS=147
if [ "$EVAL_ONLY" = 1 ] && [ "${EVAL_OOD:-false}" = true ]; then VAL_FILES+=("$CHALLENGE_PARQUET"); fi

if [ "$DRY" != 1 ]; then
    aw_check_install || exit 1
    if ! aw_is_patched; then
        _msg="the three b0a8eae_* train tasks are NOT patched in $APPWORLD_ROOT: the train split has 87 tasks, not the paper's 90 (bash scripts/setup/setup_appworld.sh applies the patch from the AppWorld data bundle)"
        if [ "${APPWORLD_ALLOW_UNPATCHED:-0}" = 1 ] || [ "${OPSD_SMOKE:-0}" = 1 ] || [ "$EVAL_ONLY" = 1 ]; then
            opsd_warn "$_msg"
        else
            opsd_die "$_msg; set APPWORLD_ALLOW_UNPATCHED=1 to train on 87 tasks anyway"
        fi
    fi
    aw_build_parquets "$APPWORLD_DATA_DIR" "$PROMPT_VARIANT" "${TRAIN_INCLUDE_DEV:-0}" || opsd_die "AppWorld parquet build failed"
    N_TRAIN_TASKS="$(aw_parquet_rows "$TRAIN_PARQUET")" || opsd_die "could not read $TRAIN_PARQUET"
    opsd_log "train split: $N_TRAIN_TASKS tasks ($TRAIN_PARQUET)"
fi
# Smoke runs keep the first N tasks of each split (deterministic order).
_truncate() {  # PARQUET N -> path of a copy holding its first N rows (under $RUN_DIR/data)
    local out
    out="$RUN_DIR/data/$(basename "$1" .parquet).head$2.parquet"
    if [ "$DRY" != 1 ]; then
        mkdir -p "$RUN_DIR/data"
        aw_head_parquet "$1" "$2" "$out" >/dev/null || opsd_die "could not truncate $1"
    fi
    printf '%s\n' "$out"
}
if [ -n "${OPSD_MAX_TRAIN_TASKS:-}" ]; then
    TRAIN_PARQUET="$(_truncate "$TRAIN_PARQUET" "$OPSD_MAX_TRAIN_TASKS")"
    [ "$OPSD_MAX_TRAIN_TASKS" -lt "$N_TRAIN_TASKS" ] && N_TRAIN_TASKS="$OPSD_MAX_TRAIN_TASKS"
fi
if [ -n "${OPSD_MAX_EVAL_TASKS:-}" ]; then
    for _i in "${!VAL_FILES[@]}"; do VAL_FILES[$_i]="$(_truncate "${VAL_FILES[$_i]}" "$OPSD_MAX_EVAL_TASKS")"; done
    for _i in "${!TEST_FILES[@]}"; do TEST_FILES[$_i]="$(_truncate "${TEST_FILES[$_i]}" "$OPSD_MAX_EVAL_TASKS")"; done
fi
_pylist() { local s="[" f sep=""; for f in "$@"; do s+="$sep'$f'"; sep=","; done; printf '%s]\n' "$s"; }

# ---- concurrency guard: the env thread pool deadlocks when it is smaller than the concurrent rollouts --
_train_conc=$((TRAIN_BATCH_SIZE * N_SAMPLES))
_eval_conc=$((EVAL_BATCH_SIZE * EVAL_N_SAMPLES))
_test_conc=$((EVAL_BATCH_SIZE * ${FINAL_TEST_N_SAMPLES:-$EVAL_N_SAMPLES}))
for _c in "train:$_train_conc" "eval:$_eval_conc" "final-test:$_test_conc"; do
    if [ "${_c#*:}" -gt "$MAX_ENV_WORKERS" ]; then
        [ "${_c%%:*}" = final-test ] && [ "${FINAL_TEST:-false}" != true ] && continue
        opsd_die "MAX_ENV_WORKERS=$MAX_ENV_WORKERS < ${_c#*:} concurrent ${_c%%:*} rollouts: every env worker would block waiting for a server and the run deadlocks"
    fi
done

# ---- disk: the checkpoint footprint is the preflight's free-space floor (see the header) ----------------
# Saves: every epoch end and every CKPT_INTERVAL steps (none when CKPT_INTERVAL <= 0, e.g. EVAL_ONLY); the
# oldest beyond MAX_CKPTS_TO_KEEP is removed only after the new one is complete. Checkpoints already in
# CKPT_DIR (RESUME_MODE=latest) are on disk already. OPSD_PREFLIGHT_MIN_FREE_GB, when set, wins.
_ckpt_peak() {
    local spe=$((N_TRAIN_TASKS / TRAIN_BATCH_SIZE)) total s saves=0
    if [ "$CKPT_INTERVAL" -le 0 ] || [ "$spe" -le 0 ]; then echo 0; return 0; fi
    total=$((TRAINER_EPOCHS * spe))
    for ((s = 1; s <= total; s++)); do
        if [ $((s % spe)) = 0 ] || [ $((s % CKPT_INTERVAL)) = 0 ]; then saves=$((saves + 1)); fi
    done
    if [ "$MAX_CKPTS_TO_KEEP" -gt 0 ] && [ "$saves" -gt $((MAX_CKPTS_TO_KEEP + 1)) ]; then
        saves=$((MAX_CKPTS_TO_KEEP + 1))
    fi
    echo "$saves"
}
CKPT_PEAK="$(_ckpt_peak)"
_ckpt_have=0
if [ "$DRY" != 1 ] && [ "$RESUME_MODE" = latest ] && [ -d "$CKPT_DIR" ]; then
    _ckpt_have="$(find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' | wc -l)"
fi
_ckpt_new=$((CKPT_PEAK - _ckpt_have))
[ "$_ckpt_new" -lt 0 ] && _ckpt_new=0
_ckpt_gb=$((_ckpt_new * ${CKPT_SIZE_GB:-45}))
if [ -z "${OPSD_PREFLIGHT_MIN_FREE_GB:-}" ]; then
    export OPSD_PREFLIGHT_MIN_FREE_GB=$((_ckpt_gb + 50))
elif [ "$OPSD_PREFLIGHT_MIN_FREE_GB" -lt "$_ckpt_gb" ] 2>/dev/null; then
    opsd_warn "OPSD_PREFLIGHT_MIN_FREE_GB=$OPSD_PREFLIGHT_MIN_FREE_GB is below this run's checkpoint footprint" \
        "($_ckpt_new more checkpoint(s) x ${CKPT_SIZE_GB:-45} GB = $_ckpt_gb GB): the disk may fill mid-run"
fi

# ---- placement / inference ------------------------------------------------------------------------------
_ws=$((NUM_GPUS_PER_NODE * NUM_NODES))
[ $((_ws % INFERENCE_ENGINE_TP_SIZE)) = 0 ] || opsd_die "INFERENCE_ENGINE_TP_SIZE=$INFERENCE_ENGINE_TP_SIZE does not divide $_ws GPUs"
NUM_INFERENCE_ENGINES=$((_ws / INFERENCE_ENGINE_TP_SIZE))
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-$((MAX_MODEL_LEN - MAX_GENERATE_LENGTH))}"
if [ "$NUM_NODES" -gt 1 ] && [ "$(opsd_ray_mode)" = local ]; then
    _msg="NUM_NODES=$NUM_NODES needs a running Ray cluster: scripts/common/ray_cluster.sh head / worker <addr>, then RAY_ADDRESS=<head_ip>:<port> on this (head) node"
    if [ "$DRY" = 1 ]; then opsd_warn "$_msg"; else opsd_die "$_msg"; fi
fi
if [ -z "${HTTP_ENDPOINT_HOST:-}" ]; then
    if [ "$NUM_NODES" -gt 1 ]; then
        HTTP_ENDPOINT_HOST="${NODE_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
        [ -n "$HTTP_ENDPOINT_HOST" ] || opsd_die "could not determine this node's IP; set HTTP_ENDPOINT_HOST"
    else
        HTTP_ENDPOINT_HOST=127.0.0.1
    fi
fi
HTTP_ENDPOINT_PORT="${HTTP_ENDPOINT_PORT:-8000}"
export HTTP_ENDPOINT_PORT  # preflight checks it

# Megatron backend (30bi): MEGATRON_OVERRIDES + the parallelism arithmetic checks + cuDNN paths. Empty for
# fsdp2. The weight pre-stage (PRESTAGE_MODEL=1) is done below, after the preflight; under DRY_RUN the
# `import megatron.core` probe is skipped as well (the parallelism arithmetic is still checked).
MEGATRON_OVERRIDES=""
_prestage="${PRESTAGE_MODEL:-0}"
PRESTAGE_MODEL=0
if [ "$DRY" = 1 ]; then
    python() { return 0; }
    source "$ROOT/scripts/common/megatron_strategy.sh" >&2
    unset -f python
else
    source "$ROOT/scripts/common/megatron_strategy.sh"
fi
PRESTAGE_MODEL="$_prestage"
read -r -a _MG_ARGS <<<"$MEGATRON_OVERRIDES"

APPWORLD_SERVER_URLS="$(aw_server_urls "$APPWORLD_BASE_PORT" "$NUM_APPWORLD_SERVERS")"

# ---- the hydra command (mirrors the AppWorld run-file that produced the results) ------------------------
build_argv() {  # uses the current values of the variables above
    local resume_path="${RESUME_PATH:-null}"
    [ -n "$resume_path" ] || resume_path=null
    local resume_mode="$RESUME_MODE"
    [ "$resume_mode" = none ] && resume_mode=null
    ARGV=(
        "data.train_data=$(_pylist "$TRAIN_PARQUET")"
        "data.val_data=$(_pylist "${VAL_FILES[@]}")"
        "trainer.algorithm.advantage_estimator=$ADVANTAGE_ESTIMATOR"
        "trainer.algorithm.policy_loss_type=$POLICY_LOSS_TYPE"
        "trainer.algorithm.eps_clip_low=$EPS_CLIP_LOW"
        "trainer.algorithm.eps_clip_high=$EPS_CLIP_HIGH"
        "trainer.algorithm.loss_reduction=$LOSS_REDUCTION"
        "trainer.algorithm.grpo_norm_by_std=$GRPO_NORM_BY_STD"
        "trainer.algorithm.use_kl_loss=$USE_KL_LOSS"
        "trainer.algorithm.use_kl_in_reward=$USE_KL_IN_REWARD"
        "trainer.algorithm.kl_loss_coef=$KL_LOSS_COEF"
        "trainer.algorithm.max_seq_len=$MAX_MODEL_LEN"
        "trainer.policy.model.path=$MODEL_PATH"
        "trainer.policy.language_model_only=${LANGUAGE_MODEL_ONLY:-false}"
        "trainer.ref.language_model_only=${LANGUAGE_MODEL_ONLY:-false}"
        "generator.inference_engine.language_model_only=${LANGUAGE_MODEL_ONLY:-false}"
        "trainer.placement.colocate_all=true"
        "trainer.strategy=$STRATEGY"
        "${_MG_ARGS[@]}"
        "trainer.log_path=$LOG_DIR/skyrl"
        "trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE"
        "trainer.placement.policy_num_nodes=$NUM_NODES"
        "trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE"
        "trainer.placement.ref_num_nodes=$NUM_NODES"
        "generator.inference_engine.served_model_name=$MODEL_PATH"
        "generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES"
        "generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP_SIZE"
        "generator.inference_engine.backend=vllm"
        "generator.inference_engine.run_engines_locally=true"
        "generator.inference_engine.weight_sync_backend=nccl"
        "generator.inference_engine.async_engine=true"
        "generator.inference_engine.enable_http_endpoint=true"
        "generator.inference_engine.http_endpoint_host=$HTTP_ENDPOINT_HOST"
        "generator.inference_engine.http_endpoint_port=$HTTP_ENDPOINT_PORT"
        "generator.batched=false"
        "generator.inference_engine.gpu_memory_utilization=$GPU_MEM_UTIL"
        "generator.inference_engine.enforce_eager=${ENFORCE_EAGER:-false}"
        "generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN"
        "generator.inference_engine.engine_init_kwargs.enable_log_requests=true"
        "environment.env_class=appworld"
        "environment.skyrl_gym.max_env_workers=$MAX_ENV_WORKERS"
        "environment.skyrl_gym.appworld.server_urls=$APPWORLD_SERVER_URLS"
        "environment.skyrl_gym.appworld.max_interactions=$MAX_INTERACTIONS"
        "environment.skyrl_gym.appworld.reward_mode=$REWARD_MODE"
        "environment.skyrl_gym.appworld.strict_parse=$STRICT_PARSE"
        "environment.skyrl_gym.appworld.scaffold=$SCAFFOLD"
        "environment.skyrl_gym.appworld.reminder_every=${REMINDER_EVERY:-0}"
        "environment.skyrl_gym.appworld.server_recycle_after=${APPWORLD_SERVER_RECYCLE_AFTER:-0}"
        "environment.skyrl_gym.appworld.execute_timeout=${EXECUTE_TIMEOUT:-120}"
        "generator.max_turns=$MAX_TURNS"
        "generator.use_conversation_multi_turn=true"
        "generator.append_eos_token_after_stop_str_in_multi_turn=true"
        "generator.apply_overlong_filtering=${APPLY_OVERLONG_FILTERING:-true}"
        "generator.zero_reward_on_non_stop=${ZERO_REWARD_ON_NON_STOP:-false}"
        "trainer.algorithm.overlong_buffer.len=$OVERLONG_BUFFER_LEN"
        "trainer.algorithm.overlong_buffer.penalty_factor=${OVERLONG_PENALTY_FACTOR:-1.0}"
        "trainer.algorithm.overlong_buffer.reward_range=${OVERLONG_REWARD_RANGE:-1.0}"
        "generator.n_samples_per_prompt=$N_SAMPLES"
        "generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES"
        "generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH"
        "generator.sampling_params.temperature=$TEMPERATURE"
        "generator.sampling_params.top_p=$TOP_P"
        "generator.sampling_params.top_k=$TOP_K"
        "generator.sampling_params.min_p=$MIN_P"
        "generator.eval_sampling_params.temperature=$EVAL_TEMPERATURE"
        "generator.eval_sampling_params.top_p=$EVAL_TOP_P"
        "generator.eval_sampling_params.top_k=$EVAL_TOP_K"
        "generator.eval_sampling_params.min_p=$EVAL_MIN_P"
        "generator.eval_sampling_params.max_generate_length=$MAX_GENERATE_LENGTH"
        "generator.max_input_length=$MAX_INPUT_LENGTH"
        "trainer.max_prompt_length=$MAX_PROMPT_LENGTH"
        "trainer.epochs=$TRAINER_EPOCHS"
        "trainer.update_epochs_per_batch=1"
        "trainer.train_batch_size=$TRAIN_BATCH_SIZE"
        "trainer.policy_mini_batch_size=$TRAIN_BATCH_SIZE"
        "trainer.micro_forward_batch_size_per_gpu=1"
        "trainer.micro_train_batch_size_per_gpu=1"
        "trainer.policy.optimizer_config.lr=$LR"
        "trainer.seed=$TRAINER_SEED"
        "trainer.eval_batch_size=$EVAL_BATCH_SIZE"
        "trainer.eval_before_train=${EVAL_BEFORE_TRAIN:-false}"
        "trainer.eval_interval=$EVAL_INTERVAL"
        "trainer.logger=${LOGGER:-console}"
        "trainer.project_name=$PROJECT_NAME"
        "trainer.run_name=$RUN_NAME"
        "trainer.resume_mode=$resume_mode"
        "trainer.resume_path=$resume_path"
        "trainer.ckpt_interval=$CKPT_INTERVAL"
        "trainer.max_ckpts_to_keep=$MAX_CKPTS_TO_KEEP"
        "trainer.ckpt_path=$CKPT_DIR"
        "trainer.export_path=$EXPORT_PATH"
    )
}
build_argv
ENTRY_MODULE="$TRAIN_ENTRYPOINT"

echo "[appworld] arm=$OPSD_ARM_ID run_dir=$RUN_DIR entry=$ENTRY_MODULE eval_only=$EVAL_ONLY"
echo "[appworld] model=$MODEL_PATH backend=$STRATEGY nodes=$NUM_NODES x $NUM_GPUS_PER_NODE GPUs" \
     "vllm=${NUM_INFERENCE_ENGINES}x TP$INFERENCE_ENGINE_TP_SIZE ctx=$MAX_MODEL_LEN/$MAX_GENERATE_LENGTH"
echo "[appworld] adv=$ADVANTAGE_ESTIMATOR loss=$POLICY_LOSS_TYPE/$LOSS_REDUCTION clip=$EPS_CLIP_LOW/$EPS_CLIP_HIGH" \
     "kl_loss=$USE_KL_LOSS/$KL_LOSS_COEF kl_in_reward=$USE_KL_IN_REWARD lr=$LR overlong=$OVERLONG_BUFFER_LEN" \
     "hints=${OPSD_HINTS_PATH:-<none>} teacher=${OPSD_TEACHER:-<none>}"
echo "[appworld] data=$APPWORLD_DATA_DIR val_on_test=$VAL_ON_TEST servers=$NUM_APPWORLD_SERVERS@$APPWORLD_BASE_PORT" \
     "env_workers=$MAX_ENV_WORKERS final_test=${FINAL_TEST:-false}"
_ckpt_desc="no checkpoints"
[ "$CKPT_INTERVAL" -gt 0 ] && _ckpt_desc="saved every epoch end + every $CKPT_INTERVAL steps, keep $MAX_CKPTS_TO_KEEP"
echo "[appworld] disk: peak $CKPT_PEAK checkpoint(s) x ${CKPT_SIZE_GB:-45} GB ($_ckpt_desc);" \
     "preflight floor $OPSD_PREFLIGHT_MIN_FREE_GB GB free under ${RUN_ROOT:-$RUN_DIR}"

if [ "$DRY" = 1 ]; then
    if [ -n "${DRY_RUN_ARGV_FILE:-}" ]; then
        { printf '%s\0' "$ENTRY_MODULE"; printf '%s\0' "${ARGV[@]}"; } >"$DRY_RUN_ARGV_FILE"
    fi
    opsd_dry_run_print "$ENTRY_MODULE" "${ARGV[@]}"
    echo "# would start: $NUM_APPWORLD_SERVERS supervised AppWorld servers on ports $APPWORLD_BASE_PORT-$((APPWORLD_BASE_PORT + NUM_APPWORLD_SERVERS - 1)) (APPWORLD_ROOT=$APPWORLD_ROOT)"
    # the resolver's temporary copy of the resolved env (DRY_RUN writes nothing under RUN_DIR)
    case "$(basename "${OPSD_RESOLVED_ENV:-}")" in opsd_resolved.*) rm -f "$OPSD_RESOLVED_ENV" ;; esac
    exit 0
fi

# ====================================================== real run from here on (GPUs, servers, files) =====
# shellcheck source=../common/preflight.sh
source "$ROOT/scripts/common/preflight.sh"
opsd_preflight_all || opsd_die "preflight failed (see the FAIL lines above; OPSD_PREFLIGHT=0 skips it)"
if [ "$NUM_NODES" -gt 1 ]; then
    bash "$ROOT/scripts/common/ray_cluster.sh" wait "$NUM_NODES" "${OPSD_RAY_WAIT_S:-600}" "$RAY_ADDRESS" ||
        opsd_die "the Ray cluster at $RAY_ADDRESS does not have $NUM_NODES nodes"
fi
mkdir -p "$RUN_DIR" "$LOG_DIR" "$EXPORT_PATH"
opsd_install_cleanup_trap
# Large checkpoints (30bi, 61 GB): download once per node here rather than every worker pulling the HF id at
# once. First on this (head) node; then, on a multi-node run, one Ray task pinned to each node does it into
# that node's HF_HOME (a cache hit when HF_HOME is shared). A node that cannot stage it fails the run now
# rather than at weight load; PRESTAGE_MODEL=0 skips all of this.
if [ "${PRESTAGE_MODEL:-0}" = 1 ] && [ ! -d "$MODEL_PATH" ]; then
    opsd_log "pre-staging $MODEL_PATH into ${HF_HOME:-the default HF cache}"
    "$PY" -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])' "$MODEL_PATH" ||
        opsd_warn "pre-stage of $MODEL_PATH failed on the head node; the per-node pass (multi-node) or the workers retry"
    if [ "$NUM_NODES" -gt 1 ]; then
        opsd_log "pre-staging $MODEL_PATH on every node of the Ray cluster at $RAY_ADDRESS"
        "$PY" - "$MODEL_PATH" <<'PY' || opsd_die "pre-stage of $MODEL_PATH failed on a node (see above); fix that node's HF_HOME / network, or PRESTAGE_MODEL=0"
import os
import sys

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

model = sys.argv[1]
ray.init(address=os.environ.get("RAY_ADDRESS") or "auto", log_to_driver=False)


@ray.remote(num_cpus=0)
def stage(name):
    import socket

    from huggingface_hub import snapshot_download

    return socket.gethostname(), os.environ.get("HF_HOME", "<default HF cache>"), snapshot_download(name)


nodes = [n for n in ray.nodes() if n.get("Alive")]
refs = [
    stage.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)).remote(model)
    for n in nodes
]
failed = 0
for node, ref in zip(nodes, refs):
    try:
        host, hf_home, path = ray.get(ref)
        print(f"[appworld] {model} staged on {node['NodeManagerAddress']} ({host}, HF_HOME={hf_home}): {path}",
              file=sys.stderr)
    except Exception as e:  # noqa: BLE001 - reported per node, then the run stops
        failed += 1
        print(f"[appworld] pre-stage FAILED on {node['NodeManagerAddress']}: {type(e).__name__}: {e}", file=sys.stderr)
ray.shutdown()
sys.exit(1 if failed or not nodes else 0)
PY
    fi
fi

opsd_log "starting $NUM_APPWORLD_SERVERS AppWorld servers on ports $APPWORLD_BASE_PORT-$((APPWORLD_BASE_PORT + NUM_APPWORLD_SERVERS - 1))"
aw_start_server_pool "$APPWORLD_BASE_PORT" "$NUM_APPWORLD_SERVERS" "$LOG_DIR/appworld"
_need=$_train_conc
[ "$EVAL_ONLY" = 1 ] && _need=$_eval_conc
[ "$_need" -gt "$NUM_APPWORLD_SERVERS" ] && _need=$NUM_APPWORLD_SERVERS
aw_wait_server_pool "$APPWORLD_BASE_PORT" "$NUM_APPWORLD_SERVERS" "$_need" "${APPWORLD_READY_TIMEOUT:-900}" ||
    opsd_die "the AppWorld server pool did not come up"
aw_check_port_owners "$APPWORLD_BASE_PORT" "$NUM_APPWORLD_SERVERS" || opsd_die "foreign process on an AppWorld port"

opsd_write_provenance "$OPSD_RESOLVED_ENV"
# resolved.env -> command.sh; an evaluation-only re-invocation's resolved.<tag>.env -> command.<tag>.sh, so the
# training run's record stays intact
_rec="$(basename "$OPSD_RESOLVED_ENV")"
case "$_rec" in
    resolved.?*.env) _rec="${_rec#resolved.}" && COMMAND_FILE="$RUN_DIR/command.${_rec%.env}.sh" ;;
    *) COMMAND_FILE="$RUN_DIR/command.sh" ;;
esac
{ printf '%q ' "$PY" -m "$ENTRY_MODULE"; printf '%q ' "${ARGV[@]}"; echo; } >"$COMMAND_FILE"
if [ "${TRAINER_STDOUT_CAPTURE:-1}" = 1 ]; then
    _TRAINER_LOG="$LOG_DIR/trainer-$(date +%y%m%d_%H%M%S).log"
    echo "[appworld] trainer stdout+stderr -> $_TRAINER_LOG"
    exec > >(tee -a "$_TRAINER_LOG") 2>&1
fi
cd "$ROOT"
rc=0
opsd_run "$PY" -m "$ENTRY_MODULE" "${ARGV[@]}" || rc=$?
echo "[appworld] trainer exited rc=$rc"

# ---- chained final test (FINAL_TEST=true): the test set(s) once, on one checkpoint ----------------------
# Checkpoint: FINAL_TEST_CKPT_STEP, else the checkpointed step with the best validation success
# (test_normal under VAL_ON_TEST=1 -- so its test_normal number is NOT held out -- dev otherwise; ties ->
# earliest), else the newest checkpoint. Logged as a separate run <RUN_NAME>_test_<step> under test/*.
# Not gated on the trainer's exit status: the checkpoints written so far are still worth testing.
if [ "${FINAL_TEST:-false}" = true ] && [ "$EVAL_ONLY" != 1 ]; then
    _metric="eval/appworld/environment/success"
    [ "$VAL_ON_TEST" = 1 ] && _metric="eval/appworld_test_normal/environment/success"
    _step="${FINAL_TEST_CKPT_STEP:-}"
    if [ -z "$_step" ]; then
        _step="$("$PY" - "$EXPORT_PATH" "$CKPT_DIR" "$_metric" <<'PY' || true
import glob, json, os, re, sys

export, ckpt, metric = sys.argv[1:4]
found = []
for path in glob.glob(os.path.join(export, "dumped_evals", "global_step_*_evals", "aggregated_results.jsonl")):
    step = int(re.search(r"global_step_(\d+)_evals", path).group(1))
    try:
        val = json.loads(open(path).readline()).get(metric)
    except (OSError, ValueError):
        continue
    if val is not None and os.path.isdir(os.path.join(ckpt, f"global_step_{step}")):
        found.append((step, float(val)))
if found:
    step, val = max(found, key=lambda sv: (sv[1], -sv[0]))
    print(step)
    print(f"[appworld] best {metric} among checkpoints: step {step} ({val:.4f})", file=sys.stderr)
PY
)"
    fi
    if [ -z "$_step" ]; then
        _step="$(find "$CKPT_DIR" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)"
    fi
    if [ -z "$_step" ] || [ ! -d "$CKPT_DIR/global_step_$_step" ]; then
        opsd_warn "FINAL_TEST: no checkpoint to test under $CKPT_DIR -- skipped"
    else
        opsd_log "FINAL_TEST on global_step_$_step: $(basename -a "${TEST_FILES[@]}" | tr '\n' ' ')"
        VAL_FILES=("${TEST_FILES[@]}")
        TRAINER_EPOCHS=0
        EVAL_BEFORE_TRAIN=true
        EVAL_N_SAMPLES="${FINAL_TEST_N_SAMPLES:-$EVAL_N_SAMPLES}"
        CKPT_INTERVAL=-1
        RESUME_MODE=from_path
        RESUME_PATH="$CKPT_DIR/global_step_$_step"
        EXPORT_PATH="$RUN_DIR/exports_test_$_step"
        RUN_NAME="${RUN_NAME}_test_$_step"
        build_argv
        _trc=0
        export SKYRL_EVAL_NAMESPACE=test
        opsd_run "$PY" -m "$ENTRY_MODULE" "${ARGV[@]}" || _trc=$?
        unset SKYRL_EVAL_NAMESPACE
        if [ "$_trc" = 0 ]; then
            opsd_log "FINAL_TEST done -> $EXPORT_PATH"
        else
            opsd_warn "FINAL_TEST exited rc=$_trc (the training result is unaffected)"
        fi
    fi
fi
exit "$rc"

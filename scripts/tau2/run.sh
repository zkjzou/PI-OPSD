#!/usr/bin/env bash
# =====================================================================================================
# tau2-bench (retail) launcher: one arm of the paper per invocation.
#
#   bash scripts/tau2/run.sh <arm_id> [--overlay NAME]... [KEY=VALUE ...]     (from any directory)
#
#   bash scripts/tau2/run.sh tau2/main/4bi/glm_aot                       # vanilla OPSD, 1 node x 8 GPUs
#   bash scripts/tau2/run.sh tau2/opsd_grpo/4bi/glm_aot/lambda=0.99      # DAPO x OPSD mix
#   bash scripts/tau2/run.sh tau2/grpo/4bi/none/seed=1                   # DAPO baseline
#   bash scripts/tau2/run.sh tau2/main/4bi/glm_chk --overlay smoke_4gpu  # minutes-long 4-GPU smoke
#   DRY_RUN=1 bash scripts/tau2/run.sh tau2/main/4bt/glm_a               # print module + argv + env
#   python -m opsd.recipes list --bench tau2                           # every arm id
#
# What it does: resolves the arm's recipe (opsd/recipes/tau2/) into $RUN_DIR/resolved.env and sources it
# (hermetic: an ambient OPSD_* variable that is not part of the recipe is an error -- pass overrides as
# KEY=VALUE), sources scripts/common/bootstrap.sh, applies the OPSD / lambda-mix / RLSD forcing below,
# checks the machine (scripts/common/preflight.sh), optionally starts a self-hosted user simulator,
# writes $RUN_DIR/provenance.json and runs the trainer with the full hydra command (from the repository root,
# with the repository root first on PYTHONPATH). It never stops Ray or any process it did not start.
#
# Paths are relative to the directory you launch from, whether given as KEY=VALUE or in the environment: the
# launcher makes them absolute before it changes into the repository root for the trainer. That covers
# RUN_ROOT / RUN_DIR / OPSD_DATA_ROOT, OPSD_HINTS_PATH, RESUME_PATH, TAU2_MANIFEST_DIR, PRESERVE_TEMPLATE,
# OPSD_ROLLOUT_DUMP_DIR, OPSD_EMA_DIR, TAU2_SOURCE_DIR and AREAL_TAU2_DATA_DIR; MODEL_PATH only when it names an
# existing local path, so a Hugging Face id (Qwen/...) stays as given. A bare TRAIN_DATA / VAL_DATA name is
# looked up in the manifest dir (below), and raw hydra arguments in TAU2_EXTRA_HYDRA are passed through
# unchanged.
#
# Prerequisites: bash scripts/setup/setup_tau2.sh (the training environment via setup_train_env.sh
# --tau2, keeping the extras .venv already has; the pinned tau2-bench checkout, the AReaL task data) and the PI
# hint files under $OPSD_DATA_ROOT (python -m opsd.data.fetch). A real run checks first that tau2-bench's
# runtime dependencies (the `tau2` extra) import.
#
# Endpoints. The simulated user (gpt-5.6-luna), the evaluation simulator (gpt-5.6-luna) and the
# NL-assertion judge (openai/gpt-4.1) are called on $OPENAI_BASE_URL (default https://api.openai.com/v1)
# with $OPENAI_API_KEY; the launcher refuses to start without the key when an endpoint is remote. Per-leg
# overrides: SIMULATOR_BASE_URL / SIM_API_KEY_ENV, EVAL_SIMULATOR_BASE_URL / EVAL_SIMULATOR_API_KEY_ENV,
# JUDGE_BASE_URL / JUDGE_API_KEY_ENV. A plain http:// endpoint counts as self-hosted (no key required).
# ⚠️ The simulator IS the environment: rewards are only comparable within one simulator.
#
# Self-hosted simulator (the tau2/grpo/*/none/sim=qwen35 arms): either point SIMULATOR_BASE_URL at
# replicas you serve yourself (opsd/tau2_bench/serve_user_sim.sh; comma-separate one URL per
# replica), or let this launcher start them on separate GPUs of this host with
#   SIM_SERVE_LOCAL=1 SIM_CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
# (the paper ran 4 replicas x TP 2 on a dedicated 8-GPU node; SIM_REPLICAS / SIM_TP / SIM_PORT adjust it).
#
# 30bi (2 nodes, Megatron): start a Ray cluster with scripts/common/ray_cluster.sh (head + 1 worker),
# then run this on the head with RAY_ADDRESS=<head_ip>:<port>. RUN_DIR and OPSD_DATA_ROOT must be on a
# filesystem shared by both nodes; the environment built with `TAU2_UV_EXTRA=megatron bash
# scripts/setup/setup_tau2.sh` (i.e. uv sync --frozen --extra megatron --extra tau2), identical on both nodes.
# ⚠️ EXPORT THE API KEY BEFORE STARTING THE HEAD:
#     export OPENAI_API_KEY=...            # every *_API_KEY_ENV the arm uses (default: OPENAI_API_KEY)
#     scripts/common/ray_cluster.sh head
# The generator (every simulator and judge call) runs in a Ray task on the head node, and a Ray task sees
# the environment `ray start` captured -- key variables are never forwarded to it at launch time. Before
# starting, the launcher runs a tiny Ray task on this node and refuses to continue if it cannot see the key.
# (OPENAI_BASE_URL is resolved in the launching process, so exporting it there is enough.)
#
# Networking: a single-node run gets NCCL_NET=Socket and VLLM_HOST_IP=127.0.0.1 from bootstrap.sh (only if
# unset; export your own values to override). The paper runs set neither; on one node NCCL does not use its
# network transport and vLLM binds locally, so this has no effect on results.
#
# Knobs (not part of the recipe):
#   DRY_RUN=1               print the entry module, the full argv and the resolved environment, then exit 0
#                           (no GPU, no servers, no files under RUN_DIR); DRY_RUN_ARGV_FILE=<path> also writes
#                           the argv NUL-separated
#   RESUME_MODE=latest      default; resumes from $RUN_DIR/ckpts when a checkpoint exists. RESUME_PATH=<dir>
#                           (a .../global_step_N directory) resumes from that checkpoint instead.
#   RESUME_RESET_DATALOADER=auto|1  opt-in: before resuming at an epoch boundary, set aside the exhausted
#                           dataloader state (data.pt) (see lib.sh). No longer needed: the trainer
#                           (OPSDRayPPOTrainer) drains an exhausted restored iterator itself; setting data.pt
#                           aside now restarts the data order from the seeded generator
#   TAU2_SOURCE_DIR         pinned tau2-bench checkout (default ~/.cache/skyrl/tau2-bench/<revision>)
#   AREAL_TAU2_DATA_DIR     staged AReaL task data (default ~/.cache/skyrl/areal-tau2-data; export it before
#                           `ray start` on a multi-node cluster when you change it)
#   TAU2_SHARED_HOST=1      shared single host: cap Ray at SKYRL_RAY_NUM_CPUS (default 16) CPUs
#   TAU2_MANIFEST_DIR=<dir> (as KEY=VALUE only) read bare TRAIN_DATA / VAL_DATA names from <dir> instead of
#                           the shipped manifests (opsd/tau2_bench/data)
#   NUM_TRAIN_NODES=<n>     (as KEY=VALUE only) trainer placement nodes; default: the recipe's NUM_NODES
#   TRAINER_STDOUT_CAPTURE=0  do not tee the trainer's output into $RUN_DIR/logs/trainer-<time>.log
#   WANDB_PROJECT           W&B project (default opsd-tau2); logger = wandb when WANDB_API_KEY is set
# =====================================================================================================
set -eo pipefail

_tau2_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ROOT = the repository root: pyproject.toml + the opsd package (opsd/, with recipes/) + scripts/
ROOT="$_tau2_here"
while [ "$ROOT" != "/" ] && ! { [ -f "$ROOT/pyproject.toml" ] && [ -f "$ROOT/opsd/__init__.py" ] &&
    [ -f "$ROOT/opsd/skyrl_ext/__init__.py" ]; }; do
    ROOT="$(dirname "$ROOT")"
done
[ "$ROOT" != "/" ] || {
    echo "[tau2] FATAL: repository root (pyproject.toml + opsd/__init__.py + opsd/skyrl_ext/__init__.py)" \
        "not found above $_tau2_here" >&2
    exit 1
}

case "${1:-}" in
    "" | -h | --help)
        awk 'NR > 2 && /^# =====/ {exit} NR > 2 {sub(/^# ?/, ""); print}' "${BASH_SOURCE[0]}"
        [ -n "${1:-}" ] && exit 0
        exit 2
        ;;
esac
ARM_ID="$1"
shift

# shellcheck disable=SC1090,SC1091
source "$ROOT/scripts/common/bootstrap.sh"
# shellcheck disable=SC1090,SC1091
source "$ROOT/scripts/tau2/lib.sh"

DRY=${DRY_RUN:-0}
opsd_resolve_recipe "$ARM_ID" "$@" || exit 1
[ "${OPSD_BENCH:-}" = tau2 ] || opsd_die "arm $ARM_ID is not a tau2 arm (bench '${OPSD_BENCH:-?}')"
case "${EVAL_ONLY:-0}" in
    0 | "" | false) ;;
    *) opsd_die "EVAL_ONLY is not supported for tau2: step-0 numbers are the eval_before_train evaluation of a run" ;;
esac

# Result-affecting launcher knobs that no recipe sets: an ambient value would change the run silently, so
# it must come through the command line (KEY=VALUE), where it is recorded in resolved.env.
tau2_check_ambient_knobs "$OPSD_RESOLVED_ENV" ENTRY_MODULE ADV_ESTIMATOR USE_KL_IN_REWARD POLICY_LOSS_TYPE \
    MAX_GRAD_NORM REASONING_PARSER PRESERVE_THINKING PRESERVE_TEMPLATE SIM_SERVE_MODEL SIM_MAX_TOKENS \
    SIM_TOP_P SIM_TOP_K SIM_MIN_P SIM_PRESENCE_PENALTY EVAL_SIM_MAX_TOKENS EVAL_SIM_REASONING_EFFORT \
    EVAL_SIM_TOP_P EVAL_SIM_TOP_K EVAL_SIM_MIN_P EVAL_SIM_PRESENCE_PENALTY TAU2_EXTRA_HYDRA \
    TAU2_MANIFEST_DIR NUM_TRAIN_NODES || exit 1
tau2_require_keys MODEL_PATH POLICY_ENABLE_THINKING TRAIN_DATA VAL_DATA \
    MAX_MODEL_LEN MAX_GENERATE_LENGTH MAX_INPUT_LENGTH MAX_PROMPT_LENGTH MAX_RESPONSE_TOKENS MAX_STEPS \
    N_SAMPLES TRAIN_BATCH_SIZE MICRO_BATCH LEARNING_RATE EPOCHS ROLLOUT_BATCHES SEED \
    LOSS_REDUCTION GRPO_NORM_BY_STD EPS_CLIP_LOW EPS_CLIP_HIGH USE_KL_LOSS KL_LOSS_COEF OVERLONG_BUFFER_LEN \
    TRAIN_TEMPERATURE TOP_P TOP_K MIN_P EVAL_TEMPERATURE EVAL_TOP_P EVAL_TOP_K EVAL_MIN_P \
    EVAL_N_SAMPLES EVAL_BATCH_SIZE EVAL_INTERVAL CKPT_INTERVAL \
    NUM_NODES NUM_GPUS_PER_NODE NUM_INFERENCE_ENGINES INFERENCE_ENGINE_TP_SIZE GPU_MEMORY_UTILIZATION \
    MAX_CONCURRENCY SIMULATOR_MODEL || exit 1

# Paths from the environment are relative to the caller's directory too (KEY=VALUE paths and a local
# MODEL_PATH were already made absolute by the recipe resolution); the trainer runs from the repository root.
_tau2_abs() { case "$1" in "" | /* | *://* | null | None | none) printf '%s' "$1" ;; *) printf '%s' "$PWD/$1" ;; esac; }
for _v in RESUME_PATH TAU2_SOURCE_DIR AREAL_TAU2_DATA_DIR; do
    [ -z "${!_v:-}" ] || printf -v "$_v" '%s' "$(_tau2_abs "${!_v}")"
done

PY="$(opsd_python)"
TAU2_DIR="$ROOT/opsd/tau2_bench"
# The repository root first on the trainer's sys.path: `opsd` is the package there (skyrl / skyrl_gym come from
# the install).
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset VLLM_API_KEY # if set, vLLM enforces auth and the policy endpoint's readiness probe fails

# ---- networking --------------------------------------------------------------------------------------
# bootstrap.sh gives a single-node run its own Ray cluster with loopback networking (NCCL_NET=Socket,
# VLLM_HOST_IP=127.0.0.1, only if unset). On a shared host, TAU2_SHARED_HOST=1 additionally caps Ray's CPU count.
# SKYRL_RAY_NUM_CPUS becomes ray.init(num_cpus=...), which Ray rejects when attaching to an existing cluster.
if [ "${TAU2_SHARED_HOST:-0}" = 1 ] && [ "$(opsd_ray_mode)" = local ]; then
    export SKYRL_RAY_NUM_CPUS=${SKYRL_RAY_NUM_CPUS:-16}
fi
if [ "$(opsd_ray_mode)" = external ] && [ -n "${SKYRL_RAY_NUM_CPUS:-}" ]; then
    opsd_warn "RAY_ADDRESS=$RAY_ADDRESS (existing cluster): ignoring SKYRL_RAY_NUM_CPUS=$SKYRL_RAY_NUM_CPUS"
    unset SKYRL_RAY_NUM_CPUS
fi
export NUM_TRAIN_NODES=${NUM_TRAIN_NODES:-$NUM_NODES}
if [ "$NUM_TRAIN_NODES" -gt 1 ] && [ "$(opsd_ray_mode)" = local ]; then
    _msg="NUM_NODES=$NUM_TRAIN_NODES needs a running multi-node Ray cluster: start it with scripts/common/ray_cluster.sh and launch on the head with RAY_ADDRESS=<head_ip>:<port>"
    if [ "$DRY" = 1 ]; then opsd_warn "$_msg"; else opsd_die "$_msg"; fi
fi
# The generator reaches the policy over HTTP. One node: loopback. Several nodes: a routable address.
if [ -z "${HTTP_ENDPOINT_HOST:-}" ]; then
    if [ "$NUM_TRAIN_NODES" -gt 1 ]; then
        HTTP_ENDPOINT_HOST="$(tau2_node_ip)"
        [ -n "$HTTP_ENDPOINT_HOST" ] || opsd_die "NUM_NODES=$NUM_TRAIN_NODES: could not determine this node's IP; set HTTP_ENDPOINT_HOST"
    else
        HTTP_ENDPOINT_HOST=127.0.0.1
    fi
fi
HTTP_ENDPOINT_PORT=${HTTP_ENDPOINT_PORT:-8110}
export HTTP_ENDPOINT_HOST HTTP_ENDPOINT_PORT

# ---- run layout ----------------------------------------------------------------------------------------
CKPT_DIR="$RUN_DIR/ckpts"
LOG_DIR="$RUN_DIR/logs"
PROJECT_NAME=${WANDB_PROJECT:-opsd-tau2}
# Every resume segment of an arm lands in one W&B group (the trainer does not pass group=).
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-$RUN_NAME}

# ---- data ----------------------------------------------------------------------------------------------
# A bare file name resolves against the shipped manifests.
DATA_DIR=${TAU2_MANIFEST_DIR:-$TAU2_DIR/data}
case "$TRAIN_DATA" in /*) ;; *) TRAIN_DATA="$DATA_DIR/$TRAIN_DATA" ;; esac
case "$VAL_DATA" in /*) ;; *) VAL_DATA="$DATA_DIR/$VAL_DATA" ;; esac
[ -f "$TRAIN_DATA" ] || opsd_die "TRAIN_DATA not found: $TRAIN_DATA"
[ -f "$VAL_DATA" ] || opsd_die "VAL_DATA not found: $VAL_DATA"
# Smoke overlay: keep the first N task ids of each list (deterministic order).
if [ -n "${OPSD_MAX_TRAIN_TASKS:-}" ]; then
    _src="$TRAIN_DATA"
    TRAIN_DATA="$RUN_DIR/data/$(basename "${_src%.jsonl}").first${OPSD_MAX_TRAIN_TASKS}.jsonl"
    [ "$DRY" = 1 ] || tau2_truncate_manifest "$_src" "$OPSD_MAX_TRAIN_TASKS" "$TRAIN_DATA" || exit 1
fi
if [ -n "${OPSD_MAX_EVAL_TASKS:-}" ]; then
    _src="$VAL_DATA"
    VAL_DATA="$RUN_DIR/data/$(basename "${_src%.jsonl}").first${OPSD_MAX_EVAL_TASKS}.jsonl"
    [ "$DRY" = 1 ] || tau2_truncate_manifest "$_src" "$OPSD_MAX_EVAL_TASKS" "$VAL_DATA" || exit 1
fi

# Pinned tau2-bench checkout (constants.TAU2_REVISION), cloned by scripts/setup/setup_tau2.sh.
TAU2_SOURCE_DIR=${TAU2_SOURCE_DIR:-$HOME/.cache/skyrl/tau2-bench/1d244f5dca42944b67a379b44bfeb9f5748f189d}
if [ ! -d "$TAU2_SOURCE_DIR/src/tau2" ]; then
    _msg="tau2-bench checkout missing at $TAU2_SOURCE_DIR (run: bash scripts/setup/setup_tau2.sh)"
    if [ "$DRY" = 1 ]; then opsd_warn "$_msg"; else opsd_die "$_msg"; fi
fi
# tau2-bench's runtime dependencies (the `tau2` extra). Without them the generator fails only after Ray, the
# placement groups and the vLLM engines are up, so a real run checks them here.
if [ "$DRY" != 1 ] && ! "$PY" -c 'import addict, deepdiff, orderly_set, cachebox, toml' >/dev/null 2>&1; then
    opsd_die "tau2-bench's runtime dependencies (addict, deepdiff, orderly-set, cachebox, toml) are missing from $PY: build the environment with the tau2 extra (bash scripts/setup/setup_tau2.sh, or bash scripts/setup/setup_train_env.sh --tau2 [--extra megatron])"
fi

# ---- policy thinking mode ------------------------------------------------------------------------------
# true  => Qwen3-4B-Thinking-2507: vLLM reasoning parser qwen3 (the simulated user only sees `content`), and
#          the agent's own <think> is re-attached to every assistant turn, so the trained token stream is
#          what the model generated and conditioned on.
# false => a non-thinking Instruct policy: the reasoning parser is OMITTED (a parser on a model that never
#          emits </think> silently moves content into `reasoning`), and no preserve template is passed.
# Native tool calling is unaffected and required either way (enable_auto_tool_choice + hermes).
if [ "$POLICY_ENABLE_THINKING" = "true" ]; then
    REASONING_PARSER=${REASONING_PARSER:-qwen3}
else
    REASONING_PARSER=${REASONING_PARSER:-none}
fi
case "$REASONING_PARSER" in none | off | NONE | OFF) REASONING_PARSER="" ;; esac
# Explicit override of the preservation wrapper; empty => follow POLICY_ENABLE_THINKING in-config.
PRESERVE_THINKING=${PRESERVE_THINKING:-}
if [ "$POLICY_ENABLE_THINKING" = "true" ]; then
    PRESERVE_TEMPLATE=${PRESERVE_TEMPLATE:-$ROOT/opsd/alfworld/qwen3thinking_preserve.jinja2}
    [ -f "$PRESERVE_TEMPLATE" ] || opsd_die "preserve chat template not found: $PRESERVE_TEMPLATE"
else
    PRESERVE_TEMPLATE=""
fi

# ---- OPSD (on-policy self-distillation) ----------------------------------------------------------------
# OPSD=1 swaps the entrypoint to the tau2 OPSD module and FORCES the knobs that recipe owns.
#
# ⚠️ FORCING, NOT DEFAULTING, IS THE POINT. A recipe layer may carry use_kl_loss=true, while vanilla OPSD
#    needs use_kl_in_reward=true. Those two are MUTUALLY EXCLUSIVE (SkyRL's skyrl/train/utils/utils.py
#    validate_cfg asserts it), so this branch must override the inherited value: `${VAR:-}` defaulting would
#    not, and the assert would fire only at Ray init.
# ⚠️ WITHOUT A REF WORKER THE "FROZEN" TEACHER SILENTLY BECOMES THE LIVE POLICY. The ref worker exists
#    only when use_kl_in_reward or use_kl_loss is true; with neither,
#    SelfDistillationTrainer.fwd_logprobs_values_reward clones the policy's own log-probs and you get
#    the known-collapse config with no error. Verify distill/teacher_is_frozen == 1.0 at step 1.
ENTRY_MODULE=${ENTRY_MODULE:-opsd.tau2_bench.main}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-false}
POLICY_LOSS_TYPE=${POLICY_LOSS_TYPE:-}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
if [ "${OPSD:-0}" = "1" ]; then
  _opsd_force() {   # assign and say so, so an override is auditable in the run log
    eval "_old=\${$1:-}"
    [ "$_old" = "$2" ] || echo "[opsd] FORCING $1=$2 (was ${_old:-<unset>})"
    export "$1=$2"
  }
  _opsd_force ENTRY_MODULE     opsd.tau2_bench.distill.main_tau2_opsd
  # --- DAPO x OPSD λ-MIX (OPSD_MIX_LAMBDA < 1): A = λ·A_GRPO + (1−λ)·A_SDPO, see opsd/trainer/ ---
  # A DIFFERENT forced recipe, not a tweak of the vanilla one below: the mix needs a real group-relative
  # A_GRPO from the PURE env reward, and a ref worker for the frozen teacher with the reward path OFF.
  # λ SCHEDULE (mix_schedule.py): λ linear from OPSD_MIX_LAMBDA to _FINAL over _DECAY_STEPS, then _AFTER. Mix mode
  # is decided over the WHOLE schedule (any λ < 1), exactly as the trainer's mix_lambda_configured() does -- a ramp
  # that starts at λ=1 still needs the teacher recipe from step 0.
  _opsd_lam_ok() {   # $1=name $2=value -> 0 if unset/empty or a number in [0,1]
    case "$2" in
      "") return 0 ;;
      *[!0-9.]*|.|*.*.*) echo "[opsd] FATAL: $1='$2' is not a number in [0,1]" >&2; return 1 ;;
      *) awk -v l="$2" 'BEGIN { exit (l + 0 > 1.0) }' || { echo "[opsd] FATAL: $1=$2 > 1" >&2; return 1; } ;;
    esac
  }
  _opsd_lam_ok OPSD_MIX_LAMBDA "${OPSD_MIX_LAMBDA:-}" || exit 1
  _opsd_lam_ok OPSD_MIX_LAMBDA_FINAL "${OPSD_MIX_LAMBDA_FINAL:-}" || exit 1
  _opsd_lam_ok OPSD_MIX_LAMBDA_AFTER "${OPSD_MIX_LAMBDA_AFTER:-}" || exit 1
  case "${OPSD_MIX_LAMBDA_DECAY_STEPS:-0}" in *[!0-9]*) echo "[opsd] FATAL: OPSD_MIX_LAMBDA_DECAY_STEPS" \
    "='$OPSD_MIX_LAMBDA_DECAY_STEPS' is not a non-negative integer" >&2; exit 1 ;; esac
  _lam0=${OPSD_MIX_LAMBDA:-1.0}
  if [ "${OPSD_MIX_LAMBDA_DECAY_STEPS:-0}" -gt 0 ]; then
    _lam1=${OPSD_MIX_LAMBDA_FINAL:-$_lam0}; _lam2=${OPSD_MIX_LAMBDA_AFTER:-$_lam1}
    _opsd_mix=$(awk -v a="$_lam0" -v b="$_lam1" -v c="$_lam2" 'BEGIN { m = a; if (b < m) m = b; if (c < m) m = c; print (m + 0 < 1.0) ? 1 : 0 }')
    _opsd_sched="lambda $_lam0 -> $_lam1 over ${OPSD_MIX_LAMBDA_DECAY_STEPS} steps, then $_lam2"
  else
    _opsd_mix=$(awk -v l="$_lam0" 'BEGIN { print (l + 0 < 1.0) ? 1 : 0 }')
    _opsd_sched="lambda=$_lam0"
  fi
  # RLSD (OPSD_RLSD=1, arXiv:2604.03128) rewrites the SAME advantage slot multiplicatively, from the same
  # two inputs, so it needs the identical forced recipe -- and it cannot be combined with the mix (the
  # trainer raises too; failing here costs nothing instead of a full start-up).
  case "${OPSD_RLSD:-0}" in 1|true|yes) _opsd_rlsd=1 ;; *) _opsd_rlsd=0 ;; esac
  if [ "$_opsd_rlsd" = "1" ] && [ "$_opsd_mix" = "1" ]; then
    echo "[opsd] FATAL: OPSD_RLSD=1 with a mix ($_opsd_sched) -- both rewrite the advantage; pick one" >&2
    exit 1
  fi
  if [ "$_opsd_mix" = "1" ] || [ "$_opsd_rlsd" = "1" ]; then
    if [ "$_opsd_rlsd" = "1" ]; then
      echo "[opsd] RLSD MODE: A_hat = A_GRPO * ((1-lam)+lam*clip((P_T/P_S)^sign(A),1-eps,1+eps))" \
           "lam ${OPSD_RLSD_LAMBDA:-0.5}->${OPSD_RLSD_LAMBDA_FINAL:-0.0} over ${OPSD_RLSD_DECAY_STEPS:-50} steps," \
           "eps=${OPSD_RLSD_CLIP:-0.2} (frozen hinted teacher)"
    else
      echo "[opsd] MIX MODE $_opsd_sched: A = lambda*A_GRPO + (1-lambda)*A_SDPO (frozen hinted teacher)"
    fi
    _opsd_force ADV_ESTIMATOR    grpo              # A_GRPO must exist to be mixed; no_op has none
    _opsd_force POLICY_LOSS_TYPE regular           # the DAPO/PPO clipped loss, as the AppWorld/ALFWorld λ arms
    _opsd_force USE_KL_IN_REWARD false             # else the teacher is counted twice (reward AND per-token)
    # *** the subtle one *** use_ref_model = use_kl_loss OR use_kl_in_reward. With the reward path off, this
    # is the ONLY thing keeping a ref worker -- and without one OPSD_TEACHER=frozen silently scores the teacher
    # on the LIVE policy (the collapse config). coef 0 keeps the θ0 leash weightless: DAPO has no KL.
    _opsd_force USE_KL_LOSS      true
    _opsd_force KL_LOSS_COEF     0.0
    # NOT forced, deliberately: EPS_CLIP_HIGH (DAPO's 0.28), MAX_GRAD_NORM (DAPO's 1.0), OVERLONG_BUFFER_LEN
    # (DAPO's penalty, re-applied in mix mode by Tau2OPSDTrainer). Those are the RL recipe being mixed INTO.
    # OPSD_ENV_COEF / OPSD_DISTILL_COEF are INERT here: they scale the reward-shaping path, which is off.
  else
  _opsd_force ADV_ESTIMATOR    no_op               # OPSD supplies the advantage; GRPO's would fight it
  _opsd_force POLICY_LOSS_TYPE importance_sampling
  _opsd_force USE_KL_IN_REWARD true                # builds the ref worker AND gates the KL shaping
  _opsd_force USE_KL_LOSS      false               # mutually exclusive with the above
  _opsd_force EPS_CLIP_HIGH    "$EPS_CLIP_LOW"     # vanilla OPSD is symmetric; DAPO's 0.28 is not it
  # Every other OPSD launcher in this tree runs 0.5; the tau2 GRPO path never set it and would inherit the
  # framework default of 1.0. Measured raw grad_norm on the tau2 OPSD smoke is 62-105 -- the pure-distill
  # reward is a summed per-token KL, so it is far above the clip either way and EVERY step is clipped. That
  # is not itself a fault (the 30B reference sits at 3.5 and also clips), but it does mean the clip alone
  # sets the step size, so it should be the value the rest of the OPSD work was tuned at.
  _opsd_force MAX_GRAD_NORM    "${OPSD_MAX_GRAD_NORM:-0.5}"
  fi
  echo "[opsd] teacher=${OPSD_TEACHER:-frozen} env_coef=${OPSD_ENV_COEF:-1.0}" \
       "distill_coef=${OPSD_DISTILL_COEF:-1.0} ref_seq_cap=${OPSD_REF_SEQ_CAP:-<default 39k>}" \
       "require_hint=${OPSD_REQUIRE_HINT:-0} hints=${OPSD_HINTS_PATH:-<NONE -- STATIC FALLBACK>}"
  # OPSD_LENGTH_NORM=1 under token_mean is a ~1e4x gradient shrink and the single documented way this
  # recipe silently fails to learn. Refuse it rather than produce a plausible null.
  case "${OPSD_LENGTH_NORM:-0}" in
    0|"") ;;
    *) echo "[opsd] FATAL: OPSD_LENGTH_NORM=$OPSD_LENGTH_NORM with loss_reduction=$LOSS_REDUCTION" \
            "vanishes the gradient (~1e4x). Unset it." >&2; exit 1 ;;
  esac
  if [ -n "${OPSD_HINTS_PATH:-}" ] && [ ! -f "$OPSD_HINTS_PATH" ]; then
    _msg="hint file not found: $OPSD_HINTS_PATH (fetch the data bundle: python -m opsd.data.fetch --bench tau2)"
    if [ "$DRY" = 1 ]; then opsd_warn "$_msg"; else opsd_die "$_msg"; fi
  fi
fi
# --- DAPO Soft Overlong Punishment (arXiv:2503.14476 eq. 13), applied by Tau2DAPOTrainer in main.py -------
# ⚠️ ONSET IS MEASURED AGAINST tau2.max_response_tokens, NOT max_input_length + max_generate_length:
#    generator.py already hard-drops longer trajectories (reward 0 AND loss_mask zeroed), so the other envs'
#    formula would put the onset ABOVE that cliff and never fire. onset = MAX_RESPONSE_TOKENS - LEN.
#    REWARD_RANGE=1.0 because tau2 rewards are binary 0/1. LEN=0 disables it.
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}
OVERLONG_REWARD_RANGE=${OVERLONG_REWARD_RANGE:-1.0}

# ---- training backend: fsdp (every 4B arm) or megatron (the 30B MoE) ---------------------------------------
# A 30B MoE down the FSDP colocate path emits token garbage; `moe_enable_routing_replay` and real expert
# parallelism exist only on the Megatron path. ⚠️ The default is `fsdp`, NOT `fsdp2` (as every tau2 arm ran).
# MEGATRON_OVERRIDES expands to NOTHING unless STRATEGY=megatron.
STRATEGY=${STRATEGY:-fsdp}
MEGATRON_OVERRIDES=""
if [ "$STRATEGY" != "fsdp" ]; then
  export STRATEGY NUM_NODES="$NUM_TRAIN_NODES" NUM_GPUS_PER_NODE MODEL_PATH USE_SAMPLE_PACKING
  [ "$(( NUM_GPUS_PER_NODE * NUM_NODES ))" -gt 0 ] || opsd_die "world_size=NUM_GPUS_PER_NODE($NUM_GPUS_PER_NODE)*NUM_NODES($NUM_NODES)=0"
  # shellcheck disable=SC1090,SC1091
  source "$ROOT/scripts/common/megatron_strategy.sh"
  [ "$STRATEGY" != "megatron" ] || [ -n "${MEGATRON_OVERRIDES:-}" ] ||
    opsd_die "STRATEGY=megatron but MEGATRON_OVERRIDES is EMPTY -- the backend block did not apply (TP=1/EP=1 would OOM)"
fi
read -r -a _MGT_ARGS <<<"${MEGATRON_OVERRIDES:-}"

# ---- simulated user (the ENVIRONMENT) --------------------------------------------------------------------
_SIM_LOCAL=0
if [ -n "${SIM_SERVE_MODEL:-}" ]; then
    # Self-hosted simulator for the TRAINING rollouts; evaluation keeps EVAL_SIMULATOR_MODEL.
    SIMULATOR_MODEL=${SIMULATOR_MODEL:-$SIM_SERVE_MODEL}
    SIM_PORT=${SIM_PORT:-8000}
    SIM_REPLICAS=${SIM_REPLICAS:-1}
    if [ -n "${SIMULATOR_BASE_URL:-}" ]; then
        :
    elif [ "${SIM_SERVE_LOCAL:-0}" = 1 ] || [ "$DRY" = 1 ]; then
        if [ "${SIM_SERVE_LOCAL:-0}" = 1 ]; then
            [ -n "${SIM_CUDA_VISIBLE_DEVICES:-}" ] || opsd_die "SIM_SERVE_LOCAL=1 needs SIM_CUDA_VISIBLE_DEVICES (GPUs not used by the trainer)"
            # The trainer's Ray cluster must not see the simulator's GPUs: pin both sets, disjoint.
            _overlap=""
            for _g in ${SIM_CUDA_VISIBLE_DEVICES//,/ }; do
                case ",${CUDA_VISIBLE_DEVICES:-}," in *",$_g,"*) _overlap="$_overlap $_g" ;; esac
            done
            if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || [ -n "$_overlap" ]; then
                _msg="SIM_SERVE_LOCAL=1 needs the trainer's GPUs pinned with CUDA_VISIBLE_DEVICES, disjoint from SIM_CUDA_VISIBLE_DEVICES=$SIM_CUDA_VISIBLE_DEVICES${_overlap:+ (shared:$_overlap)}"
                if [ "$DRY" = 1 ]; then opsd_warn "$_msg"; else opsd_die "$_msg"; fi
            fi
            _SIM_LOCAL=1
        else
            opsd_warn "self-hosted simulator arm: a real run needs SIMULATOR_BASE_URL=... or SIM_SERVE_LOCAL=1 (showing local URLs)"
        fi
        SIMULATOR_BASE_URL=""
        for ((_i = 0; _i < SIM_REPLICAS; _i++)); do
            SIMULATOR_BASE_URL="${SIMULATOR_BASE_URL:+$SIMULATOR_BASE_URL,}http://127.0.0.1:$((SIM_PORT + _i))/v1"
        done
    else
        opsd_die "arm $ARM_ID trains against a self-hosted simulator ($SIM_SERVE_MODEL): pass SIMULATOR_BASE_URL=http://<host>:<port>/v1[,...] (served with opsd/tau2_bench/serve_user_sim.sh) or SIM_SERVE_LOCAL=1 SIM_CUDA_VISIBLE_DEVICES=<gpus>"
    fi
fi
if ! tau2_check_llm_keys; then
    if [ "$DRY" = 1 ]; then
        opsd_warn "(DRY_RUN) a real run would stop here: export OPENAI_API_KEY (and OPENAI_BASE_URL for a non-default endpoint)"
    else
        opsd_die "missing API key(s) for a remote endpoint (see above)"
    fi
fi
# Existing (multi-node) Ray cluster: the keys are read inside the trainer's entrypoint task on THIS node,
# whose environment comes from `ray start`, not from this shell -- check them there (see lib.sh).
if [ "$(opsd_ray_mode)" = external ] && [ -n "${TAU2_REMOTE_KEY_ENVS:-}" ]; then
    if [ "$DRY" = 1 ]; then
        opsd_log "(DRY_RUN) a real run first checks that Ray tasks on this node see: $TAU2_REMOTE_KEY_ENVS"
    else
        # shellcheck disable=SC2086
        tau2_check_keys_on_ray_driver_node $TAU2_REMOTE_KEY_ENVS ||
            opsd_die "export $TAU2_REMOTE_KEY_ENVS before \`scripts/common/ray_cluster.sh head\` (see above)"
    fi
fi

# ---- resume --------------------------------------------------------------------------------------------
RESUME_MODE=${RESUME_MODE:-latest}
if [ -n "${RESUME_PATH:-}" ]; then
    case "$(basename "${RESUME_PATH%/}")" in
        global_step_*) ;;
        *) opsd_die "RESUME_PATH must point AT a global_step_N directory; got '$RESUME_PATH'" ;;
    esac
    RESUME_MODE=from_path
fi

# ---- the hydra command -----------------------------------------------------------------------------------
ARGV=(
    "data.train_data=['$TRAIN_DATA']"
    "data.val_data=['$VAL_DATA']"
    "trainer.algorithm.advantage_estimator=$ADV_ESTIMATOR"
)
[ -z "$POLICY_LOSS_TYPE" ] || ARGV+=("trainer.algorithm.policy_loss_type=$POLICY_LOSS_TYPE")
ARGV+=(
    "trainer.algorithm.use_kl_in_reward=$USE_KL_IN_REWARD"
    "trainer.algorithm.use_kl_loss=$USE_KL_LOSS"
    "trainer.algorithm.kl_loss_coef=$KL_LOSS_COEF"
    "trainer.algorithm.loss_reduction=$LOSS_REDUCTION"
    "trainer.algorithm.grpo_norm_by_std=$GRPO_NORM_BY_STD"
    "trainer.algorithm.eps_clip_low=$EPS_CLIP_LOW"
    "trainer.algorithm.eps_clip_high=$EPS_CLIP_HIGH"
    "trainer.algorithm.overlong_buffer.len=$OVERLONG_BUFFER_LEN"
    "trainer.algorithm.overlong_buffer.penalty_factor=$OVERLONG_PENALTY_FACTOR"
    "trainer.algorithm.overlong_buffer.reward_range=$OVERLONG_REWARD_RANGE"
    "trainer.algorithm.use_entropy_loss=${USE_ENTROPY_LOSS:-false}"
    "trainer.algorithm.entropy_loss_coef=${ENTROPY_LOSS_COEF:-0.01}"
    "trainer.algorithm.zero_variance_filter=${ZERO_VARIANCE_FILTER:-true}"
    "trainer.algorithm.max_seq_len=$MAX_MODEL_LEN"
    "trainer.use_sample_packing=${USE_SAMPLE_PACKING:-true}"
    "trainer.logprobs_chunk_size=${LOGPROBS_CHUNK_SIZE:-1024}"
    "trainer.policy.fsdp_config.cpu_offload=${CPU_OFFLOAD:-false}"
    "trainer.ref.fsdp_config.cpu_offload=${CPU_OFFLOAD:-false}"
    "trainer.policy.model.path=$MODEL_PATH"
    "trainer.policy.optimizer_config.lr=$LEARNING_RATE"
    "trainer.policy.optimizer_config.max_grad_norm=$MAX_GRAD_NORM"
    "trainer.seed=$SEED"
    "trainer.placement.colocate_all=true"
    "trainer.strategy=$STRATEGY"
)
ARGV+=("${_MGT_ARGS[@]}")
ARGV+=(
    "trainer.epochs=$EPOCHS"
    "trainer.train_batch_size=$TRAIN_BATCH_SIZE"
    "trainer.policy_mini_batch_size=$TRAIN_BATCH_SIZE"
    "trainer.update_epochs_per_batch=${UPDATE_EPOCHS:-1}"
    "trainer.micro_forward_batch_size_per_gpu=$MICRO_BATCH"
    "trainer.micro_train_batch_size_per_gpu=$MICRO_BATCH"
    "trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE"
    "trainer.placement.policy_num_nodes=$NUM_TRAIN_NODES"
    "trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE"
    "trainer.placement.ref_num_nodes=$NUM_TRAIN_NODES"
    "trainer.eval_interval=$EVAL_INTERVAL"
    "trainer.eval_batch_size=$EVAL_BATCH_SIZE"
    "trainer.eval_before_train=${EVAL_BEFORE_TRAIN:-true}"
    "trainer.ckpt_interval=$CKPT_INTERVAL"
    "trainer.max_ckpts_to_keep=${MAX_CKPTS_TO_KEEP:--1}"
    "trainer.max_prompt_length=$MAX_PROMPT_LENGTH"
    "trainer.project_name=$PROJECT_NAME"
    "trainer.run_name=$RUN_NAME"
    "trainer.logger=${LOGGER:-console}"
    "trainer.export_path=$RUN_DIR/exports"
    "trainer.log_path=$LOG_DIR"
    "trainer.ckpt_path=$CKPT_DIR"
    "trainer.resume_mode=$RESUME_MODE"
)
[ -z "${RESUME_PATH:-}" ] || ARGV+=("trainer.resume_path=$RESUME_PATH")
ARGV+=(
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
    "generator.inference_engine.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    "generator.inference_engine.enforce_eager=${ENFORCE_EAGER:-true}"
    "generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN"
    "generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true"
    "generator.inference_engine.engine_init_kwargs.tool_call_parser=hermes"
)
[ -z "$REASONING_PARSER" ] || ARGV+=("generator.inference_engine.engine_init_kwargs.reasoning_parser=$REASONING_PARSER")
if [ -n "$PRESERVE_TEMPLATE" ]; then
    ARGV+=(
        "generator.chat_template.source=file"
        "generator.chat_template.name_or_path=$PRESERVE_TEMPLATE"
        "generator.chat_template_kwargs.enable_thinking=true"
    )
fi
ARGV+=(
    "generator.n_samples_per_prompt=$N_SAMPLES"
    "generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES"
    "generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH"
    "generator.sampling_params.temperature=$TRAIN_TEMPERATURE"
    "generator.sampling_params.top_p=$TOP_P"
    "generator.sampling_params.top_k=$TOP_K"
    "generator.sampling_params.min_p=$MIN_P"
    # Eval sampling is ALWAYS explicit: naming any single eval field builds the rest from dataclass
    # defaults (temperature 1.0), so a partial override would silently change the eval distribution.
    "generator.eval_sampling_params.max_generate_length=$MAX_GENERATE_LENGTH"
    "generator.eval_sampling_params.temperature=$EVAL_TEMPERATURE"
    "generator.eval_sampling_params.top_p=$EVAL_TOP_P"
    "generator.eval_sampling_params.top_k=$EVAL_TOP_K"
    "generator.eval_sampling_params.min_p=$EVAL_MIN_P"
    "generator.max_input_length=$MAX_INPUT_LENGTH"
    "generator.max_turns=$MAX_STEPS"
    "tau2.source_dir=$TAU2_SOURCE_DIR"
    "tau2.max_generate_length=$MAX_GENERATE_LENGTH"
    "tau2.policy_enable_thinking=$POLICY_ENABLE_THINKING"
)
[ -z "$PRESERVE_THINKING" ] || ARGV+=("tau2.preserve_thinking=$PRESERVE_THINKING")
ARGV+=(
    "tau2.max_response_tokens=$MAX_RESPONSE_TOKENS"
    "tau2.max_steps=$MAX_STEPS"
    "tau2.max_concurrency=$MAX_CONCURRENCY"
    "tau2.simulator_model=$SIMULATOR_MODEL"
)
# Optional knobs: an empty value omits the override, so the config default applies.
_opt() { [ -z "${2:-}" ] || ARGV+=("$1=$2"); }
_opt tau2.simulator_base_url "${SIMULATOR_BASE_URL:-}"
_opt tau2.simulator_api_key_env "${SIM_API_KEY_ENV:-}"
_opt tau2.simulator_max_tokens "${SIM_MAX_TOKENS:-}"
_opt tau2.simulator_reasoning_effort "${SIM_REASONING_EFFORT:-}"
_opt tau2.simulator_temperature "${SIM_TEMPERATURE:-}"
_opt tau2.simulator_top_p "${SIM_TOP_P:-}"
_opt tau2.simulator_top_k "${SIM_TOP_K:-}"
_opt tau2.simulator_min_p "${SIM_MIN_P:-}"
_opt tau2.simulator_presence_penalty "${SIM_PRESENCE_PENALTY:-}"
_opt tau2.eval_simulator_model "${EVAL_SIMULATOR_MODEL:-}"
_opt tau2.eval_simulator_base_url "${EVAL_SIMULATOR_BASE_URL:-}"
_opt tau2.eval_simulator_api_key_env "${EVAL_SIMULATOR_API_KEY_ENV:-}"
_opt tau2.eval_simulator_max_tokens "${EVAL_SIM_MAX_TOKENS:-}"
_opt tau2.eval_simulator_reasoning_effort "${EVAL_SIM_REASONING_EFFORT:-}"
_opt tau2.eval_simulator_temperature "${EVAL_SIM_TEMPERATURE:-}"
_opt tau2.eval_simulator_top_p "${EVAL_SIM_TOP_P:-}"
_opt tau2.eval_simulator_top_k "${EVAL_SIM_TOP_K:-}"
_opt tau2.eval_simulator_min_p "${EVAL_SIM_MIN_P:-}"
_opt tau2.eval_simulator_presence_penalty "${EVAL_SIM_PRESENCE_PENALTY:-}"
_opt tau2.judge_base_url "${JUDGE_BASE_URL:-}"
_opt tau2.judge_api_key_env "${JUDGE_API_KEY_ENV:-}"
ARGV+=(
    "tau2.rollout_batches=$ROLLOUT_BATCHES"
    "tau2.trajectory_dir=$RUN_DIR/trajectories"
)
# Extra raw hydra overrides (advanced): TAU2_EXTRA_HYDRA="key=value key2=value2".
if [ -n "${TAU2_EXTRA_HYDRA:-}" ]; then
    read -r -a _extra <<<"$TAU2_EXTRA_HYDRA"
    ARGV+=("${_extra[@]}")
fi

echo "[tau2] arm=$OPSD_ARM_ID run_dir=$RUN_DIR"
echo "[tau2] model=$MODEL_PATH thinking=$POLICY_ENABLE_THINKING reasoning_parser=${REASONING_PARSER:-<omitted>}" \
     "preserve_template=${PRESERVE_TEMPLATE:-<none>} ctx=$MAX_MODEL_LEN gen=$MAX_GENERATE_LENGTH"
echo "[tau2] train=$TRAIN_DATA ($(wc -l <"$TRAIN_DATA" 2>/dev/null || echo '?') tasks)" \
     "val=$VAL_DATA group=$N_SAMPLES batch=$TRAIN_BATCH_SIZE steps=$((ROLLOUT_BATCHES * EPOCHS))"
echo "[tau2] entrypoint=$ENTRY_MODULE adv=$ADV_ESTIMATOR kl_in_reward=$USE_KL_IN_REWARD" \
     "kl_loss=$USE_KL_LOSS/$KL_LOSS_COEF clip=$EPS_CLIP_LOW/$EPS_CLIP_HIGH grad_norm=$MAX_GRAD_NORM" \
     "reduction=$LOSS_REDUCTION overlong_len=$OVERLONG_BUFFER_LEN backend=$STRATEGY"
echo "[tau2] simulator=$SIMULATOR_MODEL @ ${SIMULATOR_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}" \
     "eval_simulator=${EVAL_SIMULATOR_MODEL:-<train simulator>}"

if [ "$DRY" = 1 ]; then
    if [ -n "${DRY_RUN_ARGV_FILE:-}" ]; then
        { printf '%s\0' "$ENTRY_MODULE"; printf '%s\0' "${ARGV[@]}"; } >"$DRY_RUN_ARGV_FILE"
    fi
    opsd_dry_run_print "$ENTRY_MODULE" "${ARGV[@]}"
    [ "$_SIM_LOCAL" = 0 ] || echo "# would start: $SIM_REPLICAS x serve_user_sim.sh replica(s) of $SIM_SERVE_MODEL on GPUs $SIM_CUDA_VISIBLE_DEVICES"
    exit 0
fi

# ====================================================== real run from here on (GPUs, servers, files) =====
# shellcheck disable=SC1090,SC1091
source "$ROOT/scripts/common/preflight.sh"
if [ "$_SIM_LOCAL" = 1 ]; then
    export OPSD_PREFLIGHT_PORTS="${OPSD_PREFLIGHT_PORTS:+$OPSD_PREFLIGHT_PORTS }$SIM_PORT:$SIM_REPLICAS:tau2-simulator"
fi
opsd_preflight_all || opsd_die "preflight failed (see the FAIL lines above; OPSD_PREFLIGHT=0 skips it)"
if [ "$_SIM_LOCAL" = 1 ] && [ "${OPSD_PREFLIGHT:-1}" != 0 ]; then
    CUDA_VISIBLE_DEVICES="$SIM_CUDA_VISIBLE_DEVICES" opsd_preflight_gpus "$(( SIM_REPLICAS * ${SIM_TP:-1} ))" ||
        opsd_die "the simulator GPUs ($SIM_CUDA_VISIBLE_DEVICES) are not free"
fi
mkdir -p "$RUN_DIR" "$CKPT_DIR" "$LOG_DIR" "$RUN_DIR/exports" "$RUN_DIR/trajectories"
opsd_install_cleanup_trap

if [ "$_SIM_LOCAL" = 1 ]; then
    opsd_log "starting the simulated user: $SIM_SERVE_MODEL, $SIM_REPLICAS replica(s) x TP ${SIM_TP:-1} on GPUs $SIM_CUDA_VISIBLE_DEVICES"
    opsd_spawn tau2-simulator env CUDA_VISIBLE_DEVICES="$SIM_CUDA_VISIBLE_DEVICES" \
        SIM_SERVE_MODEL="$SIM_SERVE_MODEL" SIM_PORT="$SIM_PORT" SIM_REPLICAS="$SIM_REPLICAS" \
        SIM_TP="${SIM_TP:-1}" SIM_MAX_LEN="${SIM_MAX_LEN:-16384}" SIM_GPU_UTIL="${SIM_GPU_UTIL:-0.90}" \
        SIM_EXTRA_ARGS="${SIM_EXTRA_ARGS:-}" SIM_STATUS_LOG="$LOG_DIR/simulator_status.log" \
        bash "$TAU2_DIR/serve_user_sim.sh" >"$LOG_DIR/simulator.log" 2>&1
fi
if [ -n "${SIM_SERVE_MODEL:-}" ]; then
    for _u in ${SIMULATOR_BASE_URL//,/ }; do
        opsd_log "waiting for the simulator at $_u (log: $LOG_DIR/simulator.log)"
        opsd_wait_http "${_u%/}/models" "${SIM_WAIT_SECS:-3600}" ${OPSD_LAST_PID:+"$OPSD_LAST_PID"} ||
            opsd_die "simulator $_u not ready"
    done
fi

if [ "$RESUME_MODE" != none ]; then
    tau2_maybe_reset_dataloader "$(tau2_resume_checkpoint "$CKPT_DIR")"
fi
opsd_write_provenance "$OPSD_RESOLVED_ENV"
{ printf '%q ' "$PY" -m "$ENTRY_MODULE"; printf '%q ' "${ARGV[@]}"; echo; } >"$RUN_DIR/command.sh"

# Persist the trainer's own stdout/stderr (trainer.log_path only receives the vLLM/Ray infra logs), one
# timestamped file per attempt so a resume never overwrites the log that explains it.
if [ "${TRAINER_STDOUT_CAPTURE:-1}" = 1 ]; then
    _TRAINER_LOG="$LOG_DIR/trainer-$(date +%y%m%d_%H%M%S).log"
    echo "[tau2] trainer stdout+stderr -> $_TRAINER_LOG"
    exec > >(tee -a "$_TRAINER_LOG") 2>&1
fi
cd "$ROOT"
rc=0
opsd_run "$PY" -m "$ENTRY_MODULE" "${ARGV[@]}" || rc=$?
echo "[tau2] trainer exited rc=$rc"
exit "$rc"

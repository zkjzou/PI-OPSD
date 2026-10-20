#!/bin/bash
set -euo pipefail
set -x

# ----------------------------------------------------------------------------
# Generation-only ALFWorld eval using SkyRL-managed inference engines.
#
# This path uses `opsd.skyrl_ext.main_generate` (upstream SkyRL's generation-only
# entrypoint with the OPSD config, the OPSD env registration and the OPSD
# evaluate): SkyRL starts its own local vLLM inference engines on the visible
# GPUs, then runs the stock SkyRLGymGenerator against a pool of ALFWorld HTTP
# environment servers. No policy/ref/optimizer workers are built and no training
# happens. The eval dumps land in $EXPORT_PATH/dumped_evals (the SkyRL eval-dump
# format, plus env_metrics).
#
# Needs the training environment (bash scripts/setup/setup_train_env.sh) and the ALFWorld venv +
# game data (bash scripts/setup/setup_alfworld.sh).
# The server pool it starts is stopped on exit (its PIDs live in $LOG_DIR/pids).
# Run dir: LOG_DIR (default $RUN_ROOT/$RUN_NAME, RUN_ROOT default ./runs); a RELATIVE RUN_ROOT / LOG_DIR /
# DATA_DIR / EVAL_DATASET / MODEL_PATH resolves against the repository root (the script cd's there first), not
# the caller's cwd. The generation process runs from the repository root, which also goes first on PYTHONPATH.
#
# Ports: the env servers use ALFWORLD_BASE_PORT..+ENV_NUM-1 (default 9100). Keep them clear of the
# SkyRL vLLM servers, where engine i takes a port in [8000+100*i, 8100+100*i).
# Model name: MODEL_NAME (vLLM served_model_name) defaults to MODEL_PATH and must stay equal to it, since
# the SkyRL inference client addresses every request by trainer.policy.model.path (else 404).
#
# Example, zero-based GPUs:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   MODEL_PATH=Qwen/Qwen3-1.7B \
#   EVAL_DATASET=eval_out_of_distribution ENV_NUM=134 REQUEST_WORKERS=8 \
#   MAX_STEPS=30 HISTORY_LENGTH=5 TEMPERATURE=0.4 MAX_COMPLETION_TOKENS=32768 \
#   RUN_NAME=alfworld_qwen3_ood_gen \
#   bash opsd/alfworld/run_alfworld_generate_local.sh
#
# Extra CLI args are forwarded to main_generate, for example:
#   bash .../run_alfworld_generate_local.sh generator.eval_n_samples_per_prompt=4
# ----------------------------------------------------------------------------

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root = the nearest ancestor holding pyproject.toml, opsd/__init__.py and opsd/skyrl_ext/__init__.py
# (independent of this file's depth).
PROJECT_ROOT="$THIS_DIR"
while [ "$PROJECT_ROOT" != "/" ] && ! { [ -f "$PROJECT_ROOT/pyproject.toml" ] &&
    [ -f "$PROJECT_ROOT/opsd/__init__.py" ] && [ -f "$PROJECT_ROOT/opsd/skyrl_ext/__init__.py" ]; }; do
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [ "$PROJECT_ROOT" = "/" ]; then
    echo "Repo root (pyproject.toml + opsd/__init__.py + opsd/skyrl_ext/__init__.py) not found above $THIS_DIR" >&2
    exit 1
fi
cd "$PROJECT_ROOT"

PYTHON=${PYTHON:-${OPSD_PYTHON:-"$PROJECT_ROOT/.venv/bin/python"}}
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    PYTHON=python
fi

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-1.7B"}
MODEL_NAME=${MODEL_NAME:-"$MODEL_PATH"}
if [ "$MODEL_NAME" != "$MODEL_PATH" ]; then
    echo "WARNING: MODEL_NAME=$MODEL_NAME != MODEL_PATH=$MODEL_PATH; the SkyRL inference client (default) sends requests as $MODEL_PATH, so they will 404." >&2
fi
RUN_NAME=${RUN_NAME:-"alfworld_${MODEL_PATH##*/}_generate"}
PROJECT_NAME=${PROJECT_NAME:-"skyrl_alfworld_generate"}
RUN_ROOT=${RUN_ROOT:-"./runs"}
LOG_DIR=${LOG_DIR:-"$RUN_ROOT/$RUN_NAME"}
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"   # absolute: Ray workers need not share this shell's cwd
EXPORT_PATH=${EXPORT_PATH:-"$LOG_DIR/exports"}
INFRA_LOG_PATH=${INFRA_LOG_PATH:-"$LOG_DIR/infra_logs"}

DATA_DIR=${DATA_DIR:-"$HOME/data/alfworld_parquet"}
EVAL_DATASET=${EVAL_DATASET:-"eval_out_of_distribution"}
ALFWORLD_VENV=${ALFWORLD_VENV:-"$HOME/alfworld-venv"}
export ALFWORLD_DATA=${ALFWORLD_DATA:-"$HOME/data/alfworld"}

ENV_NUM=${ENV_NUM:-${NUM_ALFWORLD_SERVERS:-134}}
ALFWORLD_BASE_PORT=${ALFWORLD_BASE_PORT:-9100}
ALFWORLD_HOST=${ALFWORLD_HOST:-"0.0.0.0"}
ALFWORLD_ADVERTISE_HOST=${ALFWORLD_ADVERTISE_HOST:-"127.0.0.1"}
MAX_STEPS=${MAX_STEPS:-${MAX_TURNS:-30}}
HISTORY_LENGTH=${HISTORY_LENGTH:-5}
REQUEST_WORKERS=${REQUEST_WORKERS:-8}
MAX_ENV_WORKERS=${MAX_ENV_WORKERS:-$REQUEST_WORKERS}

TEMPERATURE=${TEMPERATURE:-0.4}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
MIN_P=${MIN_P:-0.0}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.0}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
MAX_COMPLETION_TOKENS=${MAX_COMPLETION_TOKENS:-32768}
MAX_INPUT_LENGTH=${MAX_INPUT_LENGTH:-4096}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_INPUT_LENGTH + MAX_COMPLETION_TOKENS))}
ENABLE_THINKING=${ENABLE_THINKING:-false}
EVAL_N_SAMPLES=${EVAL_N_SAMPLES:-1}

INFERENCE_ENGINE_TP_SIZE=${INFERENCE_ENGINE_TP_SIZE:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
ENFORCE_EAGER=${ENFORCE_EAGER:-false}
ASYNC_ENGINE=${ASYNC_ENGINE:-true}
DISTRIBUTED_EXECUTOR_BACKEND=${DISTRIBUTED_EXECUTOR_BACKEND:-ray}
LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-false}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-true}

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]; then
    NUM_GPUS=$("$PYTHON" - <<'PY'
import os
visible = [x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
print(len(visible))
PY
)
elif command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS=${NUM_GPUS:-1}
fi
NUM_INFERENCE_ENGINES=${NUM_INFERENCE_ENGINES:-$((NUM_GPUS / INFERENCE_ENGINE_TP_SIZE))}
if [ "$NUM_INFERENCE_ENGINES" -lt 1 ]; then
    echo "NUM_INFERENCE_ENGINES resolved to $NUM_INFERENCE_ENGINES; check CUDA_VISIBLE_DEVICES and INFERENCE_ENGINE_TP_SIZE." >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$EXPORT_PATH" "$INFRA_LOG_PATH" "$DATA_DIR"

case "$EVAL_DATASET" in
    train)
        SPLIT="train"
        DATA_FILE="$DATA_DIR/train.parquet"
        ;;
    valid_seen|validation|eval_seen|eval_in_distribution|in_distribution|id)
        SPLIT="valid_seen"
        DATA_FILE="$DATA_DIR/validation.parquet"
        ;;
    valid_unseen|validation_unseen|eval_unseen|eval_out_of_distribution|out_of_distribution|ood)
        SPLIT="valid_unseen"
        DATA_FILE="$DATA_DIR/validation_unseen.parquet"
        ;;
    *.parquet)
        SPLIT=""
        DATA_FILE="$EVAL_DATASET"
        ;;
    *)
        echo "Unknown EVAL_DATASET=$EVAL_DATASET. Use train, valid_seen, valid_unseen, eval_in_distribution, eval_out_of_distribution, or a parquet path." >&2
        exit 1
        ;;
esac

# Launch one-world-per-process env servers from the existing ALFWorld venv (fails if venv/data are missing).
ALFWORLD_PID_DIR="$LOG_DIR/pids"
ALFWORLD_SERVER_URLS=$(
    ALFWORLD_VENV="$ALFWORLD_VENV" \
    ALFWORLD_DATA="$ALFWORLD_DATA" \
    NUM_ALFWORLD_SERVERS="$ENV_NUM" \
    ALFWORLD_BASE_PORT="$ALFWORLD_BASE_PORT" \
    ALFWORLD_HOST="$ALFWORLD_HOST" \
    ALFWORLD_ADVERTISE_HOST="$ALFWORLD_ADVERTISE_HOST" \
    ALFWORLD_MAX_STEPS="$MAX_STEPS" \
    ALFWORLD_LOG_DIR="$LOG_DIR" \
    ALFWORLD_PID_DIR="$ALFWORLD_PID_DIR" \
    bash "$THIS_DIR/server/launch_alfworld_servers.sh" | tail -1
)
echo "ALFWORLD_SERVER_URLS=$ALFWORLD_SERVER_URLS"
if [ -f "$ALFWORLD_PID_DIR/alfworld_serve.pids" ]; then
    ALFWORLD_SERVE_PIDS=$(cat "$ALFWORLD_PID_DIR/alfworld_serve.pids")
    trap 'kill $ALFWORLD_SERVE_PIDS 2>/dev/null || true' EXIT
fi
case "$ALFWORLD_SERVER_URLS" in
    \[*) ;;
    *) echo "ALFWorld server pool did not start (see the errors above)." >&2; exit 1 ;;
esac

if [ -n "$SPLIT" ] && [ ! -f "$DATA_FILE" ]; then
    "$ALFWORLD_VENV/bin/python" "$THIS_DIR/alfworld_dataset.py" \
        --local_dir "$DATA_DIR" --alfworld_data "$ALFWORLD_DATA" --splits "$SPLIT"
fi
if [ ! -f "$DATA_FILE" ]; then
    echo "Eval parquet not found: $DATA_FILE" >&2
    exit 1
fi

sleep ${ALFWORLD_SERVER_WARMUP_SEC:-20}

export HYDRA_FULL_ERROR=1
# Ray workers run $PYTHON's environment. Under `uv run`, Ray's uv integration would re-launch them with `uv run` in
# an uploaded copy of the working directory, where pyproject's ../SkyRL-ecdaa946 source does not resolve: off.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=${SKYRL_DUMP_INFRA_LOG_TO_STDOUT:-0}
# The repository root first on sys.path: `opsd` is this checkout's package (skyrl / skyrl_gym come from the
# SkyRL install).
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -m "${GENERATE_ENTRYPOINT:-opsd.skyrl_ext.main_generate}" \
    data.val_data="['$DATA_FILE']" \
    trainer.policy.model.path="$MODEL_PATH" \
    trainer.strategy=fsdp2 \
    trainer.placement.colocate_all=false \
    trainer.logger=console \
    trainer.project_name="$PROJECT_NAME" \
    trainer.run_name="$RUN_NAME" \
    trainer.eval_interval=1 \
    trainer.eval_batch_size="$REQUEST_WORKERS" \
    trainer.max_prompt_length="$MAX_PROMPT_LENGTH" \
    trainer.export_path="$EXPORT_PATH" \
    trainer.log_path="$INFRA_LOG_PATH" \
    trainer.disable_fast_tokenizer=false \
    generator.inference_engine.served_model_name="$MODEL_NAME" \
    generator.inference_engine.num_engines="$NUM_INFERENCE_ENGINES" \
    generator.inference_engine.tensor_parallel_size="$INFERENCE_ENGINE_TP_SIZE" \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.async_engine="$ASYNC_ENGINE" \
    generator.inference_engine.distributed_executor_backend="$DISTRIBUTED_EXECUTOR_BACKEND" \
    generator.inference_engine.language_model_only="$LANGUAGE_MODEL_ONLY" \
    generator.inference_engine.enable_prefix_caching="$ENABLE_PREFIX_CACHING" \
    generator.inference_engine.gpu_memory_utilization="$GPU_MEM_UTIL" \
    generator.inference_engine.enforce_eager="$ENFORCE_EAGER" \
    generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
    generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
    generator.batched=false \
    generator.max_turns="$MAX_STEPS" \
    generator.max_input_length="$MAX_INPUT_LENGTH" \
    generator.use_conversation_multi_turn=true \
    generator.append_eos_token_after_stop_str_in_multi_turn=true \
    generator.chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
    generator.eval_n_samples_per_prompt="$EVAL_N_SAMPLES" \
    generator.sampling_params.max_generate_length="$MAX_COMPLETION_TOKENS" \
    generator.sampling_params.temperature="$TEMPERATURE" \
    generator.sampling_params.top_p="$TOP_P" \
    generator.sampling_params.top_k="$TOP_K" \
    generator.sampling_params.min_p="$MIN_P" \
    generator.sampling_params.logprobs=null \
    generator.sampling_params.additional_kwargs.presence_penalty="$PRESENCE_PENALTY" \
    generator.sampling_params.additional_kwargs.repetition_penalty="$REPETITION_PENALTY" \
    "generator.sampling_params.stop=['</action>']" \
    generator.eval_sampling_params.max_generate_length="$MAX_COMPLETION_TOKENS" \
    generator.eval_sampling_params.temperature="$TEMPERATURE" \
    generator.eval_sampling_params.top_p="$TOP_P" \
    generator.eval_sampling_params.top_k="$TOP_K" \
    generator.eval_sampling_params.min_p="$MIN_P" \
    generator.eval_sampling_params.logprobs=null \
    generator.eval_sampling_params.additional_kwargs.presence_penalty="$PRESENCE_PENALTY" \
    generator.eval_sampling_params.additional_kwargs.repetition_penalty="$REPETITION_PENALTY" \
    "generator.eval_sampling_params.stop=['</action>']" \
    environment.env_class=alfworld \
    environment.skyrl_gym.max_env_workers="$MAX_ENV_WORKERS" \
    environment.skyrl_gym.alfworld.server_urls="$ALFWORLD_SERVER_URLS" \
    environment.skyrl_gym.alfworld.env_type=AlfredTWEnv \
    environment.skyrl_gym.alfworld.max_turns="$MAX_STEPS" \
    environment.skyrl_gym.alfworld.history_length="$HISTORY_LENGTH" \
    environment.skyrl_gym.alfworld.seed=0 \
    environment.skyrl_gym.alfworld.http_timeout="${ALFWORLD_HTTP_TIMEOUT:-120}" \
    "$@"

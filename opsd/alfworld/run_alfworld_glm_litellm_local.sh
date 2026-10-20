#!/bin/bash
set -euo pipefail
# ----------------------------------------------------------------------------
# ALFWorld R1 (ReAct + thinking) trajectories sampled with an EXTERNAL chat model behind any
# OpenAI-compatible endpoint -- the contract of the paper's PI reference trajectories: react prompt,
# invalid-action feedback, 50 turns, 16384 output tokens per turn, T 1.0 / top-p 0.95, stop </action>,
# reward scale 1 (win = 10), first win per task kept downstream. No GPU: inference is remote; the ALFWorld
# env servers run locally on CPU. Token usage (and $ when the model's price is known) is tracked and
# printed by the driver.
#
# CAVEAT (admissible list): it is hidden in the prompt and in normal observations, but SHOWN after an
# INVALID action (env default invalid_action_show_admissible=true; this script does not override it), so
# the observation gets "Admissible actions: [...]" appended. That list leaks into ~40-50% of the published
# GLM-5.2 / Qwen3.7-Plus reference hints that carry observations (38-50% by raw / raw_no_think file; the
# hints_traj_{nothink,thoughts,withthink}_cap24k training files: 44% GLM-5.2, 52% Qwen3.7-Plus). Actions-only
# hints are ~0%; actions+thoughts hints (hints_traj_act{thoughts,think}_cap24k) <=4%, through reasoning that
# quotes the list. The OPSD training contract hides it (--invalid-action-show-admissible false); compare
# authors only within one contract.
#
# The paper's reference author was GLM-5.2 (MODEL=fireworks_ai/glm-5p2 behind a LiteLLM gateway); it is no
# longer served and its trajectories are published with the data bundle. Set BASE_URL/MODEL to a live
# model to sample new references.
#
# Needs: the ALFWorld venv + game data (bash scripts/setup/setup_alfworld.sh), an ALFWorld parquet
# (alfworld_dataset.py), and the endpoint key in $API_KEY_ENV.
#
#   export OPENAI_BASE_URL=https://<openai-compatible-endpoint>/v1 OPENAI_API_KEY=...
#   DATA_FILE=<parquet> MODEL=<model id> LIMIT=2 ENV_NUM=2 \
#     bash opsd/alfworld/run_alfworld_glm_litellm_local.sh            # smoke (2 games)
# Extra args are forwarded to generate_alfworld_litellm (e.g. --budget-usd 20 --n-samples 4).
# Run dir: LOG_DIR (default $RUN_ROOT/alfworld_<model>_r1, RUN_ROOT default ./runs); relative paths, DATA_FILE
# included, resolve against the repository root (the script cd's there first), not the caller's cwd.
# ----------------------------------------------------------------------------
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root = the nearest ancestor holding pyproject.toml, opsd/__init__.py and opsd/skyrl_ext/__init__.py
# (independent of this file's depth). Relative paths stay relative to it; it goes first on PYTHONPATH (`opsd` =
# this checkout's package).
REPO="$THIS_DIR"
while [ "$REPO" != "/" ] && ! { [ -f "$REPO/pyproject.toml" ] && [ -f "$REPO/opsd/__init__.py" ] &&
  [ -f "$REPO/opsd/skyrl_ext/__init__.py" ]; }; do
  REPO="$(dirname "$REPO")"
done
if [ "$REPO" = "/" ]; then
  echo "ERROR: repo root (pyproject.toml + opsd/__init__.py + opsd/skyrl_ext/__init__.py) not found above $THIS_DIR" >&2
  exit 1
fi
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

DATA_FILE="${DATA_FILE:-${SUBSET:-}}"
if [ -z "$DATA_FILE" ]; then
  echo "ERROR: set DATA_FILE to an ALFWorld parquet built by alfworld_dataset.py." >&2; exit 1
fi
: "${MODEL:?set MODEL to the model id the endpoint serves (the paper used fireworks_ai/glm-5p2, no longer served)}"
RUN_ROOT="${RUN_ROOT:-./runs}"
# Default run dir is per model: the driver RESUMES from an existing trajectories.jsonl, so two models must
# never share one.
RUN_NAME="${RUN_NAME:-alfworld_${MODEL##*/}_r1}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/$RUN_NAME}"
mkdir -p "$LOG_DIR"

BASE_URL="${BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
API_KEY_ENV="${API_KEY_ENV:-OPENAI_API_KEY}"
MAX_TURNS="${MAX_TURNS:-50}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-16384}"
N_SAMPLES="${N_SAMPLES:-1}"
ENV_NUM="${ENV_NUM:-12}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-$ENV_NUM}"
ALFWORLD_BASE_PORT="${ALFWORLD_BASE_PORT:-9400}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
# USD per 1M tokens. Empty = the driver's PRICE_TABLE_PER_1M entry for MODEL (if any).
PRICE_IN="${PRICE_IN:-}"
PRICE_OUT="${PRICE_OUT:-}"
PRICE_CACHE="${PRICE_CACHE:-}"
PY="${PY:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || PY=python3

if [ -z "${!API_KEY_ENV:-}" ]; then echo "ERROR: set \$$API_KEY_ENV (the endpoint key) or API_KEY_ENV=<var name>." >&2; exit 1; fi

# ---- launch the ALFWorld server pool (loopback for this single-box run; this run's PIDs only) ----
ALFWORLD_PID_DIR="$LOG_DIR/pids"
ALFWORLD_SERVER_URLS=$(
  NUM_ALFWORLD_SERVERS="$ENV_NUM" ALFWORLD_BASE_PORT="$ALFWORLD_BASE_PORT" \
  ALFWORLD_HOST=0.0.0.0 ALFWORLD_ADVERTISE_HOST=127.0.0.1 \
  ALFWORLD_MAX_STEPS="$MAX_TURNS" ALFWORLD_LOG_DIR="$LOG_DIR" ALFWORLD_PID_DIR="$ALFWORLD_PID_DIR" \
  bash "$THIS_DIR/server/launch_alfworld_servers.sh" | tail -1
)
echo "ALFWORLD_SERVER_URLS=$ALFWORLD_SERVER_URLS"
if [ -f "$ALFWORLD_PID_DIR/alfworld_serve.pids" ]; then
  trap 'kill $(cat "$ALFWORLD_PID_DIR/alfworld_serve.pids") 2>/dev/null || true' EXIT
fi
sleep "${ALFWORLD_SERVER_WARMUP_SEC:-20}"

# ---- run the trajectory driver ----
"$PY" -m opsd.alfworld.generate_alfworld_litellm \
  --data-file "$DATA_FILE" --output "$LOG_DIR/trajectories.jsonl" \
  --base-url "$BASE_URL" --model "$MODEL" --api-key-env "$API_KEY_ENV" \
  --server-urls "$ALFWORLD_SERVER_URLS" \
  --prompt-style react --show-admissible false --invalid-action-feedback true \
  --max-turns "$MAX_TURNS" --max-completion-tokens "$MAX_COMPLETION_TOKENS" \
  --max-concurrency "$MAX_CONCURRENCY" --n-samples "$N_SAMPLES" \
  --temperature "$TEMPERATURE" --top-p "$TOP_P" \
  ${PRICE_IN:+--price-in-per-1m "$PRICE_IN"} \
  ${PRICE_OUT:+--price-out-per-1m "$PRICE_OUT"} \
  ${PRICE_CACHE:+--price-cache-read-per-1m "$PRICE_CACHE"} \
  ${LIMIT:+--limit "$LIMIT"} \
  "$@"

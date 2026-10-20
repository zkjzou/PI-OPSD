#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Frontier PI for τ²-bench: generate reference trajectories with a strong model, then build the three
# canonical hint formats (actions / raw_no_think / raw).
#
#   stage 1  baseline.py with the POLICY pointed at a frontier model via an OpenAI-compatible endpoint
#   stage 2  build_tau2_pi_hints.py -> hints_{raw,raw_no_think,actions}.json  (schema opsd_hints/v2)
#
# Default source is GLM-5p2, verified to return BOTH `reasoning_content` (a separate field, no inline
# <think>) and native `tool_calls` — the two things the three formats need. The user simulator tracks
# whatever the training runs use so the ENVIRONMENT matches — now gpt-5.6-luna, was deepseek-v4-flash.
# PI hints built against the OLD simulator describe a different environment; rebuild them.
#
# Covers the whole 563-task AReaL retail pool (baseline.py has no --manifest flag), then splits the
# artifact by manifest so TRAIN PI and DEV PI never mix: dev hints must never reach a dev eval.
#
#   bash scripts/analysis/run_tau2_pi_gen.sh              # full 563-task pass
#   LIMIT=3 bash scripts/analysis/run_tau2_pi_gen.sh      # smoke first (recommended)
#   env: MODEL SPLIT LIMIT NUM_TRIALS CONCURRENCY OUT_DIR PROFILE BUILD_ONLY
#        POLICY_BASE_URL / POLICY_KEY_ENV (frontier policy endpoint; default LLM_BASE_URL / LLM_KEY_ENV)
# MODEL is passed as the endpoint names it (default fireworks_ai/glm-5p2, a LiteLLM-style provider
# string; GLM-5.2 and Qwen3.7-Plus are no longer served, and the reference trajectories built with them
# are published under reference_trajectories/ and hints/). The user simulator and the NL judge use
# LLM_BASE_URL / LLM_KEY_ENV (default $OPENAI_BASE_URL / OPENAI_API_KEY).
#
# NOTE on shared quota: the user simulator and GLM are BOTH fireworks_ai. On 2026-07-29 a
# PROVIDER-level fireworks rate limit (not our per-key limit — raising key TPM cannot help) took three
# concurrent training runs down to rewardable_frac 0.375-0.61. Keep CONCURRENCY modest while training
# runs are live, and prefer running this when they are not.
# ---------------------------------------------------------------------------
set -eo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_tau2_env.sh"
cd "$REPO_ROOT"

MODEL=${MODEL:-fireworks_ai/glm-5p2}
SPLIT=${SPLIT:-retail-areal-rl}
NUM_TRIALS=${NUM_TRIALS:-1}
CONCURRENCY=${CONCURRENCY:-8}
PROFILE=${PROFILE:-tau2-paper}          # T=0: one deterministic reference attempt per task
MAX_TOKENS=${MAX_TOKENS:-8192}
MAX_STEPS=${MAX_STEPS:-60}
TAG=${TAG:-$(echo "$MODEL" | tr '/.' '__')}
OUT_DIR=${OUT_DIR:-$OUT_ROOT/pi_tau2/$TAG}
TRAJ_DIR="$OUT_DIR/traj"
DATA=opsd/tau2_bench/data
POLICY_BASE_URL=${POLICY_BASE_URL:-$LLM_BASE_URL}
POLICY_KEY_ENV=${POLICY_KEY_ENV:-$LLM_KEY_ENV}

# The key for the frontier policy and the user simulator (read from the environment, never on argv).
[ -n "${BUILD_ONLY:-}" ] || require_llm_key

mkdir -p "$TRAJ_DIR"

if [ -z "${BUILD_ONLY:-}" ]; then
  echo "=== stage 1: generating with $MODEL (split=$SPLIT trials=$NUM_TRIALS conc=$CONCURRENCY) ==="
  # PolicySpec.litellm_model unconditionally prepends "openai/", so pass the bare provider path
  # (fireworks_ai/glm-5p2 -> openai/fireworks_ai/glm-5p2). Passing openai/... would double the prefix.
  # --preserve-thinking matches how the ALFWorld frontier generation was run (richer raw hints).
  # Resumable: baseline.py counts only terminated_* records as done, so infra failures retry on re-run.
  $PY -m opsd.tau2_bench.baseline \
    --split "$SPLIT" --num-trials "$NUM_TRIALS" \
    ${LIMIT:+--limit $LIMIT} \
    --model "$MODEL" \
    --policy-base-url "$POLICY_BASE_URL" \
    --policy-api-key-env "$POLICY_KEY_ENV" \
    --user-llm "${USER_LLM:-gpt-5.6-luna}" "${TAU2_LLM_ARGS[@]}" \
    --profile "$PROFILE" --preserve-thinking \
    --max-tokens "$MAX_TOKENS" --max-steps "$MAX_STEPS" \
    --concurrency "$CONCURRENCY" \
    --output-dir "$TRAJ_DIR" --yes-really "$@"
fi

echo "=== stage 2: building the three PI formats ==="
# Which manifests to split the trajectory artifact by, as "<name>:<manifest>" specs.
# The retail default keeps TRAIN PI and DEV PI apart — dev hints must never reach a dev eval.
# Airline has no dev split, so it passes a single spec over the 500 generated tasks:
#   SPLIT=airline-areal-rl LIMIT=500 \
#   HINT_SPECS="train:$DATA/airline_areal_rl_500.jsonl" bash .../run_tau2_pi_gen.sh
# NOTE: the specs must match the domain you generated. Pointing retail manifests at an airline
# trajectory file does NOT error — --only-tasks simply matches nothing and writes EMPTY hints.
HINT_SPECS=${HINT_SPECS:-"train:$DATA/tau2_train500.jsonl dev:$DATA/tau2_dev62.jsonl"}
for SPEC in $HINT_SPECS; do
  NAME=${SPEC%%:*}; MANIFEST=${SPEC#*:}
  [ -f "$MANIFEST" ] || { echo "  skip $NAME (no $MANIFEST)"; continue; }
  echo "--- $NAME ---"
  $PY -m opsd.tau2_bench.build_tau2_pi_hints \
    --traj "$TRAJ_DIR/trajectories.jsonl" \
    --out-dir "$OUT_DIR/hints_$NAME" \
    --only-tasks "$MANIFEST" \
    --max-chars-raw "${MAX_CHARS_RAW:-250000}"
done

echo
echo "=== artifacts ==="
find "$OUT_DIR" -name 'hints_*.json' -o -name 'formats_summary.json' | sort | sed 's/^/  /'

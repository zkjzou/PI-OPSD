#!/bin/bash
set -euo pipefail
# ----------------------------------------------------------------------------------------------------
# Build AppWorld PI (privileged-information) reference trajectories for the TRAIN-90 split from two
# frontier sources, Qwen3.7-Plus and GLM-5.2, via an OpenAI-compatible endpoint (no GPU needed).
#
# WHY train-90: the only existing AppWorld PI is qwen37 on test_normal (168 tasks), built for the
# offline ranking study. Those can never be used for training. These 90 are the tasks GRPO trains on.
#
# WHY THE SCAFFOLD CHECK BELOW -- the single most important detail here.
# generate_appworld_local.py IS the training driver: it constructs opsd.envs' AppWorldEnv
# (opsd/envs/appworld/) and calls env.init()/env.step(), so parsing, observation rendering and
# reward all come from the same code GRPO uses. An env WITHOUT the official-scaffold support (scaffold.py;
# env.py defaulting to "official") runs the legacy env instead -- that is what produced the old TGC-0.0
# AppWorld numbers. It was never a different driver -- it was the pre-scaffold env. With this tree's
# opsd.envs you get the official scaffold that the 35.1% GRPO run used.
#
# Flags below are pinned to the GRPO run's config, NOT the driver's own defaults, which differ:
#     --strict-parse        driver default is off;      training used STRICT_PARSE=true
#     --reward-mode binary  driver default is partial;  training used binary
#     --max-turns 30        matches MAX_TURNS=30
#     --preserve-thinking   training retains <think> in history (custom_chat_template unset)
#
# Endpoint: BASE_URL (default $OPENAI_BASE_URL, else https://api.openai.com/v1) with the key in the env
# var named by API_KEY_ENV (default OPENAI_API_KEY). Model ids: QWEN_MODEL / GLM_MODEL (defaults are the
# LiteLLM provider strings the published references used; a direct Fireworks endpoint names them
# accounts/fireworks/models/<id>; GLM-5.2 / Qwen3.7-Plus may no longer be served).
#
# Usage:
#   SMOKE=1 bash scripts/analysis/build_appworld_pi_train90.sh      # 4 tasks/model
#   bash scripts/analysis/build_appworld_pi_train90.sh              # full train-90
#   MODELS=qwen37 bash ...                                             # one source only
# ----------------------------------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$REPO_ROOT" != / ] && ! { [ -f "$REPO_ROOT/pyproject.toml" ] && [ -f "$REPO_ROOT/opsd/__init__.py" ] && [ -f "$REPO_ROOT/opsd/skyrl_ext/__init__.py" ]; }; do REPO_ROOT="$(dirname "$REPO_ROOT")"; done
[ -f "$REPO_ROOT/opsd/skyrl_ext/__init__.py" ] || { echo "FATAL: cannot locate the repo root above $0" >&2; exit 1; }
cd "$REPO_ROOT"
PY=${PY:-python}

# ---- make Python actually import THIS tree's opsd.envs ---------------------------------------------
# Checking that scaffold.py exists on disk is NOT enough, and an earlier version of this script made
# exactly that mistake: an import resolves through site-packages to whatever EDITABLE install the venv
# carries, regardless of cwd. If that is another checkout without the scaffold module, the driver
# silently runs the legacy env and produces PI under a different contract than GRPO uses. PYTHONPATH
# entries precede site-packages in sys.path, so prepending this tree wins (skyrl_gym itself comes from the
# SkyRL install); the guard below then verifies what Python REALLY imported, and from where.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$REPO_ROOT" <<'PYEOF' || { echo "FATAL: the imported opsd.envs.appworld has no scaffold support or is not this tree's — refusing to build PI" >&2; exit 1; }
import os
import sys
import opsd.envs.appworld.env as m
from opsd.envs.appworld.env import AppWorldEnvConfig
repo = os.path.realpath(sys.argv[1])
ok = hasattr(AppWorldEnvConfig, "scaffold")
mine = os.path.realpath(m.__file__).startswith(repo + os.sep)
print(f"env module -> {m.__file__}" + ("" if mine else f"   (NOT under this checkout {repo})"))
print(f"  scaffold = {AppWorldEnvConfig.scaffold if ok else '<MISSING>'}"
      f"   reward_mode = {getattr(AppWorldEnvConfig,'reward_mode','?')}")
sys.exit(0 if ok and mine else 1)
PYEOF
# AppWorld parquet built by opsd/appworld/appworld_dataset.py --local_dir $APPWORLD_DATA_DIR
APPWORLD_DATA_DIR=${APPWORLD_DATA_DIR:-$HOME/data/appworld}
DATA_FILE=${DATA_FILE:-$APPWORLD_DATA_DIR/train.parquet}   # 90 tasks, official prompt
OUT_ROOT=${OUT_ROOT:-./outputs/appworld_pi}
APPWORLD_VENV=${APPWORLD_VENV:?set APPWORLD_VENV (the AppWorld server venv)}
export APPWORLD_ROOT=${APPWORLD_ROOT:?set APPWORLD_ROOT (the AppWorld data root)}
export ALFWORLD_DATA=""                                # keep alfworld out of the appworld venv's way

BASE_URL=${BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}
API_KEY_ENV=${API_KEY_ENV:-OPENAI_API_KEY}
[ -n "${!API_KEY_ENV:-}" ] || { echo "FATAL: set $API_KEY_ENV (the API key for $BASE_URL)" >&2; exit 1; }
QWEN_MODEL=${QWEN_MODEL:-fireworks_ai/qwen3p7-plus}
GLM_MODEL=${GLM_MODEL:-fireworks_ai/glm-5p2}

# SINGLE-INSTANCE LOCK. Per-model port blocks (_port_for below) stop the two MODELS from colliding, but
# they do NOT stop two *invocations* from colliding: a rescue and a generation for the same model both
# resolve to the same base port. That happened twice and each time the symptom impersonated something
# else -- once as `FileNotFoundError: rescue_a2/summary.json` (a regeneration renamed the output dir out
# from under a live rescue), and once as 5 rollouts with `/evaluate` connection-refused on ports
# 9500-9504, which reads as flaky evaluation but was really 5 servers that never bound. The agent
# transcripts for those 5 even contain "The server is completely down. I'll keep trying" -- the env was
# dead DURING the rollout, so they burned all 30 turns. Fail fast instead.
LOCK_FILE=${LOCK_FILE:-/tmp/appworld_pi_train90.lock}
# NOTE `9<>` not `9>`: the truncating form wipes the file BEFORE flock is even attempted, so the loser
# would erase the holder's pid and then report the empty string it just created. Open read-write instead
# and truncate only after we own the lock.
: >> "$LOCK_FILE"
exec 9<>"$LOCK_FILE"
flock -n 9 || {
  echo "FATAL: another AppWorld PI run holds $LOCK_FILE (pid $(cat "$LOCK_FILE" 2>/dev/null))." >&2
  echo "       Concurrent runs share ports and output dirs -- wait for it, or set LOCK_FILE + BASE_PORT" >&2
  echo "       to fully disjoint values if you really mean to run two at once." >&2
  exit 1
}
truncate -s 0 "$LOCK_FILE" 2>/dev/null || true
echo $$ >&9

# Verify the port blocks are free (`ss -ltn`) before launching.
BASE_PORT=${BASE_PORT:-9500}
# Models run SEQUENTIALLY and each --launch-servers binds BASE_PORT..+N. Reusing one base port across
# them races the previous model's shutdown (atexit terminate is asynchronous): the second model hits
# "[Errno 98] address already in use", its servers never come up, /evaluate 500s, and every reward is
# 0.0 -- an infrastructure failure that looks exactly like a model failure. Give each model its own
# non-overlapping block. (frac_eval_unavailable is the metric that exposes it; see the guard below.)
_port_for() { case "$1" in qwen37) echo $((BASE_PORT));; glm52) echo $((BASE_PORT + 200));; esac; }

if [ "${SMOKE:-0}" == "1" ]; then
  LIMIT_ARG="--limit 4"; NSRV=${NSRV:-4};  TAG=${TAG:-smoke}
else
  LIMIT_ARG="";          NSRV=${NSRV:-24}; TAG=${TAG:-train90}
fi

MODELS=${MODELS:-"qwen37 glm52"}
_model_id() { case "$1" in
    qwen37) echo "$QWEN_MODEL" ;;
    glm52)  echo "$GLM_MODEL" ;;
    *) echo "unknown model key $1" >&2; return 1 ;;
  esac; }

# ---- RESCUE=1: re-sample only the tasks with no successful trajectory --------------------------------
# The main pass runs N=1 GREEDY (temp 0), which is deterministic -- re-running it would return the same
# failed trajectory every time, so the rescue MUST sample with temp > 0. Up to RESCUE_ATTEMPTS rounds,
# dropping each task the round it first succeeds, so a task solved on attempt 1 costs one rollout, not five.
#
# Sampling is PER MODEL, from each vendor's own documentation, because the two disagree:
#   Qwen (thinking mode): temperature 0.6, top_p 0.95, top_k 20, min_p 0 -- and the Qwen docs
#       explicitly warn "Do not use greedy decoding, as it can lead to performance degradation and
#       endless repetitions", which is exactly what the main temp-0 pass did.
#   GLM-5.2: default temperature 1.0, top_p 0.95; Zhipu's own evals use temperature=1.0 and they advise
#       tuning temperature OR top_p, not both.
# Only temperature/top_p/max_tokens reach an external endpoint anyway (the driver's openai_compat branch
# drops top_k/min_p/penalties), so there is nothing hand-tuned being imposed here.
_sampling_for() { case "$1" in
    qwen37) echo "0.6 0.95" ;;   # Qwen thinking-mode official
    glm52)  echo "1.0 0.95" ;;   # GLM-5.2 default / Zhipu eval setting
  esac; }
if [ "${RESCUE:-0}" == "1" ]; then
  ATTEMPTS=${RESCUE_ATTEMPTS:-5}
  for MK in $MODELS; do
    MODEL=$(_model_id "$MK") || exit 1
    read -r R_TEMP R_TOP_P <<<"$(_sampling_for "$MK")"
    R_TEMP=${RESCUE_TEMPERATURE:-$R_TEMP}; R_TOP_P=${RESCUE_TOP_P:-$R_TOP_P}
    BASE="$OUT_ROOT/gen_${TAG}_${MK}"
    [ -f "$BASE/transcripts.jsonl" ] || { echo "no $BASE/transcripts.jsonl — run the main pass first" >&2; exit 1; }
    for A in $(seq 1 "$ATTEMPTS"); do
      RP="$BASE/rescue_a${A}"; mkdir -p "$RP"
      # Which tasks still have no reward-1.0 trajectory, across the base pass AND every prior attempt?
      NEED=$("$PY" - "$BASE" "$DATA_FILE" "$RP/todo.parquet" <<'PYEOF'
import glob, json, sys
import pandas as pd
base, data_file, out_parquet = sys.argv[1], sys.argv[2], sys.argv[3]
best = {}
for f in [f"{base}/transcripts.jsonl", *sorted(glob.glob(f"{base}/rescue_a*/transcripts.jsonl"))]:
    try: fh = open(f)
    except OSError: continue
    for line in fh:
        if not line.strip(): continue
        r = json.loads(line); t = r.get("task_id")
        best[t] = max(best.get(t, -1.0), float(r.get("reward", 0.0) or 0.0))
df = pd.read_parquet(data_file)
need = [t for t, v in best.items() if v < 1.0]
sub = df[df["reward_spec"].map(lambda rs: rs["ground_truth"]).isin(need)]
sub.to_parquet(out_parquet)
print(len(sub))
PYEOF
) || exit 1
      if [ "$NEED" -eq 0 ]; then echo "[$MK] rescue: 0 tasks left unsolved -> done before attempt $A"; rm -rf "$RP"; break; fi
      echo "[$MK] rescue attempt $A/$ATTEMPTS: $NEED unsolved task(s), t=$R_TEMP p=$R_TOP_P"
      NS=$(( NEED < 8 ? NEED : 8 ))
      "$PY" opsd/appworld/generate_appworld_local.py \
        --data-file "$RP/todo.parquet" \
        --model-url "$BASE_URL" --served-model-name "$MODEL" --api-key-env "$API_KEY_ENV" \
        --launch-servers --num-appworld-servers "$NS" --appworld-base-port "$(_port_for "$MK")" \
        --appworld-venv "$APPWORLD_VENV" --appworld-root "$APPWORLD_ROOT" \
        --concurrency "$NS" \
        --n-samples 1 --temperature "$R_TEMP" --top-p "$R_TOP_P" \
        --reward-mode binary --strict-parse --max-turns 30 --preserve-thinking \
        --save-transcripts --output-dir "$RP" > "$RP/gen.log" 2>&1 \
        || { echo "[$MK] rescue attempt $A FAILED (see $RP/gen.log)" >&2; }
      "$PY" -c "
import json,sys
p='$RP/transcripts.jsonl'
try: rs=[json.loads(l) for l in open(p) if l.strip()]
except OSError: rs=[]
won=[r['task_id'] for r in rs if float(r.get('reward',0) or 0)>=1.0]
print(f'[$MK]   attempt $A solved {len(won)}/{len(rs)}: {sorted(won)}')"
    done
    echo "[$MK] rescue done. rebuild hints over base + all attempts:"
    echo "    cat $BASE/transcripts.jsonl $BASE/rescue_a*/transcripts.jsonl > $BASE/transcripts_merged.jsonl"
  done
  exit 0
fi

for MK in $MODELS; do
  MODEL=$(_model_id "$MK") || exit 1
  OUT="$OUT_ROOT/gen_${TAG}_${MK}"
  mkdir -p "$OUT"
  echo "=============================================================================="
  echo "  $MK  ($MODEL)  ->  $OUT"
  echo "=============================================================================="
  # --concurrency == #servers: each rollout holds a world exclusively for its whole trajectory, so
  # more concurrency than servers just blocks in the pool's queue.
  # NB the driver takes --api-key-env (the NAME of an env var), not the key value, and --output-dir.
  "$PY" opsd/appworld/generate_appworld_local.py \
    --data-file "$DATA_FILE" $LIMIT_ARG \
    --model-url "$BASE_URL" --served-model-name "$MODEL" --api-key-env "$API_KEY_ENV" \
    --launch-servers --num-appworld-servers "$NSRV" --appworld-base-port "$(_port_for "$MK")" \
    --appworld-venv "$APPWORLD_VENV" --appworld-root "$APPWORLD_ROOT" \
    --concurrency "$NSRV" \
    --n-samples 1 --temperature 0 \
    --reward-mode binary --strict-parse --max-turns 30 --preserve-thinking \
    --save-transcripts \
    --output-dir "$OUT" 2>&1 | tee "$OUT/gen.log"
  echo "--- $MK summary ---"; cat "$OUT/summary.json" 2>/dev/null || echo "(no summary.json)"
  # An AppWorld /evaluate failure yields reward 0.0, indistinguishable from a genuinely failed task.
  # frac_eval_unavailable is the only signal that separates them, so refuse to proceed on a bad run
  # rather than let it silently poison the hints.
  "$PY" - "$OUT/summary.json" <<'PYEOF' || exit 1
import json, os, sys
try: d = json.load(open(sys.argv[1]))
except Exception as e: print(f"FATAL: unreadable summary ({e})"); sys.exit(1)
u = float(d.get("frac_eval_unavailable", 0.0) or 0.0)
# A handful of eval failures is survivable: an unevaluable rollout scores 0.0, and --min-reward 1.0
# drops it, so the only cost is coverage (and the rescue pass will retry it). A LARGE fraction means
# the pool is actually dead and every reward is meaningless, which must not become PI.
LIMIT = float(os.environ.get("MAX_EVAL_UNAVAILABLE", "0.2"))
print(f"  frac_eval_unavailable = {u:.4f}  (abort threshold {LIMIT})")
if u > LIMIT:
    print(f"FATAL: {100*u:.0f}% of rollouts could not be evaluated -- the appworld server pool failed. "
          f"Rewards are meaningless; fix the pool and rerun. NOT building hints from this.")
    sys.exit(1)
if u > 0.0:
    print(f"WARNING: {round(u*float(d.get('num_trajectories',0)))} rollout(s) unevaluable -> they score 0.0 "
          f"and will be dropped by --min-reward; rerun the RESCUE pass to recover them.")
PYEOF
done

echo
echo "next: build the 3 hint densities per model with --min-reward 1.0 (solved tasks only):"
for MK in $MODELS; do
  echo "  $PY scripts/analysis/build_appworld_self_hints.py \\"
  echo "      --transcripts $OUT_ROOT/gen_${TAG}_${MK}/transcripts.jsonl \\"
  echo "      --out-dir $OUT_ROOT/hints_${TAG}_${MK} --min-reward 1.0"
done

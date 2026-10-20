#!/usr/bin/env bash
# Build the `checklist` PI arm + its length-matched control, for one (bench, split).
#
# The checklist is distilled from a GLM-5.2 WIN contrasted against the student's own first losing rep.
# The win text is taken from the EXISTING glm52 demonstration artifact rather than re-derived, so the
# checklist arm and `glm52_raw` carry byte-identical privileged information and differ only in FORM.
#
# Emits, into $OUT_DIR:
#   hints_checklist.json       the arm
#   hints_win_headtail.json    the SAME GLM text truncated head+tail to each task's checklist length --
#                              the length control. Do not evaluate the arm without it: the demonstration
#                              arms run ~12k chars vs a checklist's ~1-2k, and hint length correlates
#                              with post-training performance at rho +0.44.
#   src_reps.json / summary.json
#
# tau2 has its own driver: run_tau2_checklist_hints.sh.
#
# Analyzer (any OpenAI-compatible endpoint or litellm provider string, see opsd/pi/analyzer.py):
#   ANALYZER_MODEL      required (or OPSD_PI_ANALYZER_MODEL). The published checklists used
#                       fireworks_ai/glm-5p2 (chk_glm) and fireworks_ai/qwen3p7-plus (chk_q7); both are no
#                       longer served, so the bundle's alfworld/hints/derived/checklist/ files are the exact
#                       artifacts.
#   ANALYZER_BASE_URL   default $OPSD_PI_ANALYZER_BASE_URL, else $OPENAI_BASE_URL
#   key                 the env var named by OPSD_PI_ANALYZER_API_KEY_ENV (default OPENAI_API_KEY); it is
#                       handed over the environment, never on the command line.
#
#   BENCH=alfworld SPLIT=train1600 WIN=<glm hints_traj_withthink.json> ANALYZER_MODEL=<model> \
#     bash scripts/analysis/run_pi_checklist_hints.sh
#
# NOTE ON set -u: deliberately NOT set, so a missing optional variable never kills the build before its
# first log line. Use `set -x` instead.
set -eo pipefail
set -x

cd "$(dirname "${BASH_SOURCE[0]}")" && while [ "$PWD" != / ] && ! { [ -f pyproject.toml ] && [ -f opsd/__init__.py ] && [ -f opsd/skyrl_ext/__init__.py ]; }; do cd ..; done
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"  # `opsd` = the package at <repo root>/opsd
PY=${PY:-.venv/bin/python}
OPSD_DATA_ROOT=${OPSD_DATA_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data}
PI_CORPUS_ROOT=${PI_CORPUS_ROOT:-$OPSD_DATA_ROOT/alfworld/corpora}

BENCH=${BENCH:-appworld}
SPLIT=${SPLIT:-normal}

ANALYZER_BACKEND=${ANALYZER_BACKEND:-endpoint}
# MUST carry the provider prefix when the endpoint is a LiteLLM proxy. A bare `glm-5p2` is an unknown model
# GROUP there, and litellm reports that as a misleading auth error ("key can only access models=[...]").
ANALYZER_MODEL=${ANALYZER_MODEL:-$OPSD_PI_ANALYZER_MODEL}
[ -n "$ANALYZER_MODEL" ] || { echo "set ANALYZER_MODEL (or OPSD_PI_ANALYZER_MODEL)" >&2; exit 4; }
ANALYZER_BASE_URL=${ANALYZER_BASE_URL:-${OPSD_PI_ANALYZER_BASE_URL:-$OPENAI_BASE_URL}}
export OPSD_PI_ANALYZER_API_KEY_ENV=${OPSD_PI_ANALYZER_API_KEY_ENV:-OPENAI_API_KEY}
{ set +x; } 2>/dev/null
[ -n "${!OPSD_PI_ANALYZER_API_KEY_ENV}" ] || { echo "no analyzer key: set $OPSD_PI_ANALYZER_API_KEY_ENV" >&2; exit 4; }
set -x

# 8192, NOT the 1024 that build_hints.py used to hardcode: a thinking analyzer spends its budget on
# reasoning, the JSON is cut mid-object, and EVERY task is silently "skipped" -- zero hints produced.
MAX_TOKENS=${MAX_TOKENS:-8192}
MAX_ITEMS=${MAX_ITEMS:-5}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-16}
LIMIT_TASKS=${LIMIT_TASKS:-}     # set to 5 for a smoke run; ALWAYS smoke before a full build
# Cover tasks the student NEVER fails by contrasting the frontier win against the student's own WIN.
# Off -> those tasks are dropped and the arm covers less of the pool than the demonstration arms do.
ALLOW_WIN_VS_WIN=${ALLOW_WIN_VS_WIN:-0}

case "$BENCH:$SPLIT" in
  appworld:normal|appworld:train90|appworld:dev)
    # POOL = the student's no-hint transcripts on the split (transcripts.jsonl: official_normal / onpolicy_train90
    # / official_dev corpus); WIN = the glm52 hints_raw.json of the same split (AppWorld data bundle).
    POOL=${POOL:?set POOL to the AppWorld student corpus dir (transcripts.jsonl) for $SPLIT}
    WIN=${WIN:?set WIN to the glm52 hints_raw.json for $SPLIT} ;;
  alfworld:train1600)
    # 4B-INSTRUCT rollouts on the train1600 task set: the rem1400 pass@k shards (7 shards x 200; no
    # first-200 4B-Instruct dump is used, so this covers 1400 of the nominal 1600).
    # Do NOT use a TRAINED checkpoint's rollouts as the loss side: one that wins ~88% leaves only a few
    # hundred tasks with any failure to contrast against. The base instruct model wins 19.3% -> 1386 eligible.
    POOL=${POOL:-$PI_CORPUS_ROOT/alf_passk32_rem1400_4bi}
    # withthink, NOT hints_raw.json: raw has <think> on 0/2356 (the frontier raw render dropped the
    # out-of-band reasoning), so the win side would carry no reasoning at all. The
    # uncapped render is REBUILT_glm__traj_withthink.json of alfworld_rebuild_hints.py.
    WIN=${WIN:?set WIN to the uncapped GLM-5.2 hints_traj_withthink.json (alfworld_rebuild_hints.py)} ;;
  alfworld:*)
    # Any both-class pass@k dump of the student (a SkyRL eval-only alfworld.jsonl, or trajectories.jsonl).
    POOL=${POOL:?set POOL for alfworld (a pass@k dump of the student)}
    WIN=${WIN:?set WIN for alfworld (its own hints_glm*/hints_raw.json)} ;;
  *) echo "unknown BENCH:SPLIT '$BENCH:$SPLIT' (tau2: run_tau2_checklist_hints.sh)" >&2; exit 2 ;;
esac

OUT_DIR=${OUT_DIR:-outputs/pi_checklist/${BENCH}_${SPLIT}}
TOKENIZER=${TOKENIZER:-Qwen/Qwen3-4B-Instruct-2507}   # only to REPORT the arms' token ratio

for f in "$POOL" "$WIN"; do
  [ -e "$f" ] || { echo "missing input: $f" >&2; exit 3; }
done

mkdir -p "$OUT_DIR"
# -u: unbuffered. Without it the funnel/progress prints sit in the stdout block buffer, so a build that
# is genuinely stuck (e.g. blocked in a slow filesystem wait) is indistinguishable from one that is working.
$PY -u scripts/analysis/build_checklist_hints.py \
  --bench "$BENCH" \
  --pool "$POOL" \
  --win-hints "$WIN" \
  --out-dir "$OUT_DIR" \
  --analyzer-backend "$ANALYZER_BACKEND" \
  --analyzer-model "$ANALYZER_MODEL" \
  ${ANALYZER_BASE_URL:+--analyzer-base-url "$ANALYZER_BASE_URL"} \
  --max-tokens "$MAX_TOKENS" \
  --max-items "$MAX_ITEMS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --tokenizer "$TOKENIZER" \
  ${LIMIT_TASKS:+--limit-tasks $LIMIT_TASKS} \
  $([ "$ALLOW_WIN_VS_WIN" = 1 ] && echo --allow-win-vs-win) \
  2>&1 | tee "$OUT_DIR/build.log"

# The builder exits non-zero if any self-check failed; ${PIPESTATUS[0]} is that status, not tee's.
exit "${PIPESTATUS[0]}"

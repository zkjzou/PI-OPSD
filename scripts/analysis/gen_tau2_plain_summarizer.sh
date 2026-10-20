#!/usr/bin/env bash
set -eo pipefail
# =====================================================================================================
# τ² PLAIN-SUMMARIZER PI hints — fills the one hint family τ² was missing.
#
# τ² already has, over the SAME glm reference trajectories (pool424/hints_raw.json):
#   analyzer  (`--prompt-style onpolicy_skill`)  anlz_glm / anlz_qwn / anlz_self_4bi, 424 tasks each
# and it has the checklist family (student × author, each with its length-matched win_headtail control).
# It has NO summarizer row. This builds it, with the analyzer HELD FIXED to the same source so the
# summarizer-vs-analyzer comparison is prompt-style only, not prompt-style × content.
#
# ⚠️⚠️ THE PLAIN PROMPT LEAKS, AND ON τ² THAT IS WORSE THAN ON APPWORLD. `onpolicy_skill`'s system
# prompt forbids restating literal ids, emails and exact answers; the plain prompt does NOT. Measured
# on the AppWorld plain run: 52/160 hints carried 3+ digit ids, 35/160 literal dollar amounts. τ²
# retail rewards are DB-state checks PLUS communicate-checks on exact values (order ids, emails,
# payment methods, prices), so a leaked value is not a weaker hint — it is the ANSWER, and any utility
# measured from it is a lookup rather than a lesson.
# Hence `measure_tau2_hint_leakage.py` runs immediately after generation and its per-cell rate is
# reported WITH any downstream number. Never quote a plain-summarizer utility without it.
#
# ⚠️ SEQUENTIAL BY DESIGN. Both analyzers are fireworks_ai behind ONE PROVIDER-level quota — the one
# that took three concurrent training runs to rewardable_frac 0.375-0.61 on 2026-07-29. Do not
# parallelise the two cells, and keep CONCURRENCY modest.
#
# ⚠️ --url TAKES THE BASE, NOT /v1. The generator posts to "{url}/v1/chat/completions", so passing
# .../v1 yields .../v1/v1/... and every request 404s. URL defaults to $LLM_BASE_URL with the /v1 cut.
#
#   SMOKE=1 bash scripts/analysis/gen_tau2_plain_summarizer.sh     # 3 tasks per cell
#   bash scripts/analysis/gen_tau2_plain_summarizer.sh             # all 424
#   CELLS=glm bash scripts/analysis/gen_tau2_plain_summarizer.sh   # one cell
#
# 4bi/self is deliberately NOT here: it needs a locally served Qwen3-4B-Instruct-2507. Add it with
# --url http://127.0.0.1:PORT --model Qwen/Qwen3-4B-Instruct-2507 and NO key once an endpoint exists.
# The two authors (GLM_MODEL / QWN_MODEL, named as the endpoint names them) are no longer served by their
# provider; the built hints are published under hints/derived/summarizer and hints/derived.
# =====================================================================================================
. "$(dirname "${BASH_SOURCE[0]}")/_tau2_env.sh"
cd "$REPO_ROOT" || exit 1
GEN="$REPO_ROOT/scripts/analysis/gen_hints_local_models.py"
[ -f "$GEN" ] || { echo "FATAL: generator not found at $GEN" >&2; exit 1; }

# The SHARED source: glm's raw reference trajectories, the same hints the analyzer cells read (their meta
# records exactly this artifact). Published as hints/glm-5.2/pool424/hints_raw.json (JSON-equal to the
# original; re-serialized, so not byte-identical).
SRC=${SRC:-$T2/hints/glm-5.2/pool424/hints_raw.json}
[ -s "$SRC" ] || { echo "FATAL: source artifact missing: $SRC" >&2; exit 1; }

OUT=${OUT:-$OUT_ROOT/pi_tau2_summ}
_u=${LLM_BASE_URL%/}
URL=${URL:-${_u%/v1}}      # base, NOT .../v1 -- see the warning above
GLM_MODEL=${GLM_MODEL:-fireworks_ai/glm-5p2}
QWN_MODEL=${QWN_MODEL:-fireworks_ai/qwen3p7-plus}
ANLZ_DIR=${ANLZ_DIR:-$T2/hints/derived}   # the analyzer cells the leakage report compares against
# glm's raw hints top out at 12,547 tok (~50k chars), so 120k never elides; the renderer
# middle-elides rather than cutting the tail, because the outcome lives at the END of a trajectory.
MAX_CHARS=${MAX_CHARS:-120000}
# A thinking analyzer spends its completion on reasoning BEFORE emitting the hint; 1024 once produced
# ZERO usable hints on long trajectories. Keep this high.
MAX_TOKENS=${MAX_TOKENS:-8192}
CONCURRENCY=${CONCURRENCY:-8}
TEMPERATURE=${TEMPERATURE:-0.7}
LIMIT=""
[ "${SMOKE:-0}" = "1" ] && { LIMIT="--limit 3"; OUT="${OUT}_smoke"; }

require_llm_key

mkdir -p "$OUT"
CELLS=${CELLS:-"glm qwn"}

for cell in $CELLS; do
  case "$cell" in
    glm) MODEL=$GLM_MODEL ;;
    qwn) MODEL=$QWN_MODEL ;;
    *)   echo "FATAL: unknown cell '$cell' (expected glm | qwn)" >&2; exit 1 ;;
  esac
  echo "===== plain summarizer · cell=$cell · model=$MODEL $( [ -n "$LIMIT" ] && echo '(SMOKE 3 tasks)' ) ====="
  ( "$PY" "$GEN" \
      --input-hints "$SRC" \
      --url "$URL" --api-key-env "$LLM_KEY_ENV" --model "$MODEL" \
      --prompt-style plain --env tau2 --outcome success \
      --max-chars "$MAX_CHARS" --max-tokens "$MAX_TOKENS" \
      --temperature "$TEMPERATURE" --concurrency "$CONCURRENCY" \
      --out-schema opsd_v2 $LIMIT \
      --out "$OUT/summ_${cell}.json" ) \
    || { echo "FATAL: generation failed for cell $cell" >&2; exit 1; }
done

echo
echo "===== leakage measurement (NOT optional -- see the header) ====="
"$PY" scripts/analysis/measure_tau2_hint_leakage.py \
  --source "$SRC" \
  --hints "summ_glm=$OUT/summ_glm.json" \
  --hints "summ_qwn=$OUT/summ_qwn.json" \
  --hints "anlz_glm=$ANLZ_DIR/anlz_glm.json" \
  --hints "anlz_qwn=$ANLZ_DIR/anlz_qwn.json" \
  --out "$OUT/leakage.json" || echo "[warn] leakage measurement failed -- do NOT quote utility yet" >&2

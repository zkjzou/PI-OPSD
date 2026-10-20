#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------------
# tau2 checklist PI arms (chk_glm / chk_qwn): build_checklist_hints.py --bench tau2, a frontier WIN vs the
# student's own LOSS distilled to a checklist (+ the length-matched head+tail control). chk_glm and chk_qwn
# differ ONLY in the ANALYZER that writes the checklist; pool and frontier wins are held fixed, exactly as
# the AppWorld chk arms are defined.
#
# MODE (default tri352) -- one per build the study ran:
#   tri352      THE training split (tau2/splits/retail_tri352.jsonl, 352 retail_N tasks), every student in
#               STUDENTS, --allow-win-vs-win. This is the build of the published
#               tau2/hints/checklist/pi_tau2_chk_tri352/<student>/<glm|qwn>/ checklists. The frontier pool424
#               covers all 352, but the published files do NOT all cover 352 (glm/qwn): 4bi 323/323,
#               4bt 347/348, 30bi 348/352, every row derivation win_vs_loss. 4bi's staged no-hint pool held only
#               323 of the 352 tasks (the other 29 are absent from it, not "never failed"), so
#               --allow-win-vs-win never fired (0 win_vs_win rows); it only matters for a pool that contains
#               a task the student never loses, where the frontier win is contrasted against the STUDENT'S
#               OWN WIN (tagged meta.derivation=win_vs_win). Coverage therefore differs per student; the
#               covered task sets ship as tau2/splits/: retail_chk4bi323 (4bi), retail_chk4bt347 (4bt glm),
#               retail_chk4bt344 (4bt glm AND qwn), retail_chk30bi348 (30bi glm = glm AND qwn).
#   arl         per student, on its own eligible set (no task filter).
#   arl_common  STUDENTS on ONE common task set (ONLY_TASKS). Building each on its own eligible set
#               (386/418/405) makes the arms non-comparable: every metric averages per task, so a different
#               task set per student is a different denominator.
#   train       TRAINING namespace (retail_N) from SkyRL's own tau2 train dumps (POOL, ChatML records):
#               the only route to a retail_N checklist when the loss side is not a tau2-bench corpus.
#   base        retail_base (bare-int task ids) against a tau2-bench no-hint utility pool (POOL) and a
#               retail_base frontier win file (WIN_BASE).
#
# Inputs:
#   POOL_ROOT  tri352/arl/arl_common: <POOL_ROOT>/<student>/trajectories.jsonl, the student's no-hint
#              tau2-bench rollouts (the AReaL-split pass@k pools; not in the data bundle, generate them)
#   POOL       train/base: the pool dir
#   WIN        frontier wins (default the bundle's tau2/hints/glm-5.2/pool424/hints_raw.json)
#   ONLY_TASKS task filter (tri352: built from tau2/splits/retail_tri352.jsonl; arl_common: required)
#   OUT_ROOT   default outputs/pi_tau2_chk_<mode>
# Analyzers: ANALYZERS="glm=fireworks_ai/glm-5p2 qwn=fireworks_ai/qwen3p7-plus" (the published ones; both
# are no longer served, so the bundle's checklists are the exact artifacts). Endpoint and key follow
# opsd/pi/analyzer.make_analyzer's frontier routing exactly (the key never reaches argv):
#   * OPSD_PI_ANALYZER_BASE_URL set: that OpenAI-compatible endpoint (e.g. a LiteLLM proxy that serves the
#     fireworks_ai/... strings, which is how the published checklists were built), passed as
#     --analyzer-base-url; key = $OPSD_PI_ANALYZER_API_KEY, else the env var named by
#     OPSD_PI_ANALYZER_API_KEY_ENV (default OPENAI_API_KEY).
#   * otherwise a litellm provider string (the defaults) is routed NATIVELY: litellm reads the provider's own
#     key (FIREWORKS_AI_API_KEY for fireworks_ai/...), unless OPSD_PI_ANALYZER_API_KEY[_ENV] is set.
#     OPENAI_BASE_URL is NOT applied to provider strings (an OpenAI base URL set for the tau2 simulator must
#     not capture a Fireworks model); to go through a proxy, set OPSD_PI_ANALYZER_BASE_URL.
#   * any other model name: $OPENAI_BASE_URL (else OpenAI) with OPENAI_API_KEY.
# A preflight resolves every analyzer through make_analyzer itself and exits 4 if its key is unset.
#
#   POOL_ROOT=<pools> bash scripts/analysis/run_tau2_checklist_hints.sh
#   MODE=arl STUDENTS=4bi POOL_ROOT=<pools> bash scripts/analysis/run_tau2_checklist_hints.sh
# ---------------------------------------------------------------------------------------------------
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ "$PWD" != / ] && ! { [ -f pyproject.toml ] && [ -f opsd/__init__.py ] && [ -f opsd/skyrl_ext/__init__.py ]; }; do cd ..; done; pwd)"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-$REPO/.venv/bin/python}
OPSD_DATA_ROOT=${OPSD_DATA_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data}
MODE=${MODE:-tri352}
ANALYZERS=${ANALYZERS:-"glm=fireworks_ai/glm-5p2 qwn=fireworks_ai/qwen3p7-plus"}
OUT_ROOT=${OUT_ROOT:-outputs/pi_tau2_chk_$MODE}
WIN=${WIN:-$OPSD_DATA_ROOT/tau2/hints/glm-5.2/pool424/hints_raw.json}
# Only the EXPLICIT analyzer endpoint is passed on; every other route (native provider string, or
# OPENAI_BASE_URL for a plain model name) is make_analyzer's own default, so the driver and the library agree.
ANALYZER_BASE_URL=${OPSD_PI_ANALYZER_BASE_URL:-}
BASE_ARGS=()
[ -n "$ANALYZER_BASE_URL" ] && BASE_ARGS=(--analyzer-base-url "$ANALYZER_BASE_URL")
# Key preflight: resolve each analyzer through make_analyzer (the ROUTED model, so a proxy-style
# openai/<provider>/<model> id with no base URL is checked against the provider it is actually sent to)
# and fail fast if its key is unset. Prints variable names only, never a key value.
SPECS=()
for spec in $ANALYZERS; do SPECS+=("${spec#*=}"); done
"$PY" - ${SPECS[@]+"${SPECS[@]}"} <<'PYK' || exit 4
import os, sys
from opsd.pi.analyzer import _native_litellm_provider, make_analyzer
bad = []
for m in sys.argv[1:]:
    a = make_analyzer("frontier", model=m)
    prov = _native_litellm_provider(a.model) if a.api_base is None else None
    route = a.api_base or (f"litellm-native {prov}" if prov else "OpenAI")
    if a.api_key or str(a.api_base or "").startswith("http://"):   # keyed, or a local unauthenticated server
        print(f"[preflight] analyzer {m} -> {route}: ok")
        continue
    if prov:
        import litellm
        env = litellm.validate_environment(model=a.model)
        if env.get("keys_in_environment"):
            print(f"[preflight] analyzer {m} -> {route}: provider key present")
            continue
        need = ", ".join(env.get("missing_keys") or []) or "the provider's key variable"
        if os.environ.get("OPSD_PI_ANALYZER_API_KEY_ENV"):
            need = f"{os.environ['OPSD_PI_ANALYZER_API_KEY_ENV']} or {need}"
        need += ", or set OPSD_PI_ANALYZER_BASE_URL to use an OpenAI-compatible endpoint"
    else:
        need = os.environ.get("OPSD_PI_ANALYZER_API_KEY_ENV") or "OPENAI_API_KEY"
    bad.append(f"{m} -> {route}: set {need} (or OPSD_PI_ANALYZER_API_KEY)")
if bad:
    sys.exit("FATAL: analyzer key missing\n  " + "\n  ".join(bad))
PYK

case "$MODE" in
  tri352)
    STUDENTS=${STUDENTS:-"4bi 4bt 30bi"}; CONC=${CONC:-8}; EXTRA=(--allow-win-vs-win)
    if [ -z "${ONLY_TASKS:-}" ]; then
      ONLY_TASKS="$OUT_ROOT/tri352_tasks.json"; mkdir -p "$OUT_ROOT"
      "$PY" - "$OPSD_DATA_ROOT/tau2/splits/retail_tri352.jsonl" "$ONLY_TASKS" <<'PYEOF' || exit 1
import json, sys
ids = []
for line in open(sys.argv[1]):
    if line.strip():
        r = json.loads(line)   # bundle split rows carry task_id; SkyRL parquet-style rows may nest it
        ids.append(str(r.get("task_id") or (r.get("extra_info") or {}).get("task_id")))
assert len(ids) == len(set(ids)) == 352, f"expected 352 distinct tri352 tasks, got {len(ids)}"
open(sys.argv[2], "w").write(json.dumps(sorted(ids)))
print(f"tri352: {len(ids)} task ids -> {sys.argv[2]}")
PYEOF
    fi ;;
  arl)        STUDENTS=${STUDENTS:-"4bi 4bt 30bi"}; CONC=${CONC:-8}; EXTRA=() ;;
  arl_common) STUDENTS=${STUDENTS:-"4bt 30bi"}; CONC=${CONC:-8}; EXTRA=()
              ONLY_TASKS=${ONLY_TASKS:?set ONLY_TASKS to the common task-id list (JSON)} ;;
  train|base) STUDENTS=""; CONC=${CONC:-6}; EXTRA=()
              POOL=${POOL:?set POOL for MODE=$MODE}
              [ "$MODE" = base ] && { WIN=${WIN_BASE:?set WIN_BASE to the retail_base frontier hints_raw.json}; } ;;
  *) echo "unknown MODE '$MODE' (tri352 | arl | arl_common | train | base)" >&2; exit 2 ;;
esac
FILTER=()
[ -n "${ONLY_TASKS:-}" ] && FILTER=(--only-tasks "$ONLY_TASKS")
[ -s "$WIN" ] || { echo "FATAL: missing WIN $WIN" >&2; exit 3; }

build() {  # pool out tag model
  echo "######## tau2 chk_$3 $MODE ${4} -> $2"
  "$PY" scripts/analysis/build_checklist_hints.py \
    --bench tau2 --pool "$1" --win-hints "$WIN" ${FILTER[@]+"${FILTER[@]}"} \
    --out-dir "$2" \
    --analyzer-backend frontier --analyzer-model "$4" ${BASE_ARGS[@]+"${BASE_ARGS[@]}"} \
    ${EXTRA[@]+"${EXTRA[@]}"} --max-concurrency "$CONC" || { echo "CHK FAILED $2"; return 1; }
}

fail=0
if [ -z "$STUDENTS" ]; then
  for spec in $ANALYZERS; do
    build "$POOL" "$OUT_ROOT/${spec%%=*}" "${spec%%=*}" "${spec#*=}" || fail=1
  done
else
  POOL_ROOT=${POOL_ROOT:?set POOL_ROOT (<POOL_ROOT>/<student>/trajectories.jsonl)}
  for st in $STUDENTS; do
    for spec in $ANALYZERS; do
      build "$POOL_ROOT/$st" "$OUT_ROOT/$st/${spec%%=*}" "${spec%%=*}" "${spec#*=}" || fail=1
    done
  done
fi
echo "TAU2_CHK_DONE mode=$MODE fail=$fail"
exit "$fail"

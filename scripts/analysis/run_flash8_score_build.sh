#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------------
# Same-source 8-slot PI (8 won GLM-5.3-flash trajectories per train1600 task), after generation:
#   1. wait for the k8 top-up's final line (run_alf_pi_topup.sh WINS_TARGET=7, log at $GEN/topup.log),
#   2. pi_flash8_prepare.py: slots glm53flash_k1..k7 in pi_f8_offline200 / pi_f8_rem1400,
#   3. score both on free local GPUs (run_pi_chunks_free_gpus.sh; another user's GPU is never shared),
#   4. pi_mix_select_build.py for PI-draw seeds 0, 1, 2 -> pi_flash8_select{,_s1,_s2}.
# Stops BEFORE training: validate the hint sets first. The published sets are the dataset's
# alfworld/selection/hints_flash8{,_s1,_s2}/; the 9 f8 OPSD runs trained on them are alfworld/selection/4bi/f8/*.
#
#   GEN=<k8 generation root, holding glm53flash/> bash scripts/analysis/run_flash8_score_build.sh
# Paths are relative to the repo root. PI_STUDY_ROOT (default the bundle's alfworld/pi_metrics/scores_new_study)
# is only READ: it must hold the scored pi_mm_glm53flash200 and pi_mm_rem1400 dirs WITH their rollout chunks
# (chunks/chunk_*.jsonl, which pi_flash8_prepare.py links; the bundle's copies carry only the chunk indices, so
# rebuild them with pi_multimodel_offline_prepare.py / pi_rem1400_prepare.py from the corpora and point
# PI_STUDY_ROOT there). Everything new goes under OUT (default outputs/pi_f8): the scoring dirs
# $OUT/pi_f8_{offline200,rem1400} and the hint sets $OUT/pi_flash8_select{,_s1,_s2}.
# ---------------------------------------------------------------------------------------------------
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ "$PWD" != / ] && ! { [ -f pyproject.toml ] && [ -f opsd/__init__.py ] && [ -f opsd/skyrl_ext/__init__.py ]; }; do cd ..; done; pwd)"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"  # `opsd` = the package at <repo root>/opsd
GEN=${GEN:?set GEN to the k8 generation root (OUT of run_alf_pi_multimodel.sh N_SAMPLES=7 + run_alf_pi_topup.sh)}
OPSD_DATA_ROOT=${OPSD_DATA_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data}
export PI_STUDY_ROOT=${PI_STUDY_ROOT:-$OPSD_DATA_ROOT/alfworld/pi_metrics/scores_new_study}  # read-only inputs
OUT=${OUT:-outputs/pi_f8}
mkdir -p "$OUT"
PY=${PY:-$REPO/.venv/bin/python}
log() { echo "$(date -u +%H:%M:%S) $*"; }

log "waiting for the k8 top-up ($GEN/topup.log)"
until grep -q 'top-up done' "$GEN/topup.log" 2>/dev/null; do sleep 120; done

"$PY" scripts/analysis/pi_flash8_prepare.py --gen "$GEN/glm53flash" --out "$OUT/pi_f8" ||
  { log "FATAL: prepare failed"; exit 1; }
for pool in offline200 rem1400; do
  log "scoring pi_f8_$pool"
  W="$OUT/pi_f8_$pool" MAX_GPUS=${MAX_GPUS:-4} bash scripts/analysis/run_pi_chunks_free_gpus.sh ||
    { log "FATAL: scoring pi_f8_$pool did not finish cleanly"; exit 1; }
done

SRC="glm53flash$(printf ',glm53flash_k%d' 1 2 3 4 5 6 7)"
for s in 0 1 2; do
  out="$OUT/pi_flash8_select"; [ "$s" = 0 ] || out="${out}_s$s"
  log "building hint sets, PI-draw seed $s -> $out"
  "$PY" scripts/analysis/pi_mix_select_build.py --seed "$s" --allow-missing --sources "$SRC" \
    --offline-dir "$PI_STUDY_ROOT/pi_mm_glm53flash200" --offline-dir "$OUT/pi_f8_offline200" \
    --rem-dir "$PI_STUDY_ROOT/pi_mm_rem1400" --rem-dir "$OUT/pi_f8_rem1400" --out "$out" > "$out.build.log" 2>&1 ||
    { log "FATAL: build for seed $s failed (see $out.build.log)"; exit 1; }
done
log "done: hint sets in $OUT/pi_flash8_select{,_s1,_s2} -- next: validate, then train"

#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------------
# Score ONE new PI author on the ALFWorld offline 200 (d, Δd⁻_err, aim ratio, error-step percentile) and
# compare it with the authors already scored (PRIOR dirs), on whatever local GPUs are free.
#
#   1. wait until the author's generation log has its final line (WAIT_LOG, optional);
#      and while a $W/HOLD file exists (more generation rounds still to come);
#   2. pi_multimodel_offline_prepare.py: raw hints from the author's won trajectories + 16 rollout chunks;
#   3. every POLL seconds, start WORKERS_PER_GPU run_pi_chunk_score.sh workers on each GPU that has NO
#      compute process (a GPU another user holds is never shared), up to MAX_GPUS, until every chunk is
#      claimed (workers claim chunks with mkdir locks, so late GPUs just pick up what is left);
#   4. when every chunk is DONE: pooled report + error-step percentile (own coverage, and the common set of
#      tasks every new author solved).
#
#   AUTHOR=glm53flash GEN=outputs/pi_multimodel_glm53flash W=outputs/pi_mm_glm53flash200 \
#   WAIT_LOG=outputs/pi_multimodel_glm53flash/run.log bash scripts/analysis/run_pi_author_offline200.sh
# Paths are relative to the repo root. PRIOR (default: the published pi_mm_offline200 + pi_mm_topup200 under
# $PI_STUDY_ROOT) is only READ; everything new goes to $W.
# ---------------------------------------------------------------------------------------------------
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ "$PWD" != / ] && ! { [ -f pyproject.toml ] && [ -f opsd/__init__.py ] && [ -f opsd/skyrl_ext/__init__.py ]; }; do cd ..; done; pwd)"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"  # `opsd` = the package at <repo root>/opsd
AUTHOR=${AUTHOR:?author dir name under GEN}
GEN=${GEN:?generation root holding \$AUTHOR/}
OPSD_DATA_ROOT=${OPSD_DATA_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data}
STUDY=${PI_STUDY_ROOT:-$OPSD_DATA_ROOT/alfworld/pi_metrics/scores_new_study}
W=${W:-outputs/pi_mm_${AUTHOR}200}
PRIOR=${PRIOR:-"$STUDY/pi_mm_offline200 $STUDY/pi_mm_topup200"}
MAX_GPUS=${MAX_GPUS:-3}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-3}
POLL=${POLL:-120}
PY=${PY:-$REPO/.venv/bin/python}
log() { echo "$(date -u +%H:%M:%S) $*"; }

if [ -n "${WAIT_LOG:-}" ]; then
  log "waiting for the generation to finish ($WAIT_LOG)"
  until grep -qE 'ALL MODELS DONE|top-up done' "$WAIT_LOG" 2>/dev/null; do sleep 60; done
fi
if [ -e "$W/HOLD" ]; then  # e.g. while more top-up rounds run for this author: rm $W/HOLD to go on
  log "holding while $W/HOLD exists"
  while [ -e "$W/HOLD" ]; do sleep 60; done
fi

if [ ! -f "$W/hints/manifest.json" ]; then
  "$PY" scripts/analysis/pi_multimodel_offline_prepare.py --gen "$GEN" --models "$AUTHOR" --out "$W" \
    || { log "FATAL: prepare failed"; exit 1; }
fi
MAX_GPUS=$MAX_GPUS WORKERS_PER_GPU=$WORKERS_PER_GPU POLL=$POLL W=$W \
  bash scripts/analysis/run_pi_chunks_free_gpus.sh || { log "FATAL: scoring did not finish cleanly"; exit 1; }

ws=(); for p in $PRIOR; do ws+=(--w "$p"); done; ws+=(--w "$W")
mm=(); for p in $PRIOR; do mm+=(--mm-dir "$p"); done; mm+=(--mm-dir "$W")
"$PY" scripts/analysis/pi_multimodel_offline_report.py "${ws[@]}" --out "$W/report_combined" > "$W/report.log" 2>&1 ||
  { log "FATAL: report failed (see $W/report.log)"; exit 1; }
"$PY" - "$W" $PRIOR <<'PYEOF'
import json, sys
from pathlib import Path
man = {}
for d in sys.argv[1:]:
    for m, v in json.load(open(Path(d) / "hints/manifest.json")).items():
        man.setdefault(m, set()).update(v["tasks"])
common = sorted(set.intersection(*man.values()))
(Path(sys.argv[1]) / "common_tasks.json").write_text(json.dumps(common, indent=1))
print(f"common set: {len(common)} tasks solved by all {len(man)} new authors")
PYEOF
"$PY" scripts/analysis/pi_errstep_percentile.py --multimodel "${mm[@]}" \
  --out "$W/report_combined/errstep_percentile.json" > "$W/percentile.log" 2>&1 &&
"$PY" scripts/analysis/pi_errstep_percentile.py --multimodel "${mm[@]}" --task-ids "$W/common_tasks.json" \
  --out "$W/report_combined/errstep_percentile_common.json" > "$W/percentile_common.log" 2>&1 ||
  { log "FATAL: percentile failed (see $W/percentile*.log)"; exit 1; }
log "report: $W/report.log, $W/report_combined/"

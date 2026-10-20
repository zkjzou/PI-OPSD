#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------------
# Score a prepared PI scoring dir ($W with chunks/ + hints/) with run_pi_chunk_score.sh workers on whatever
# local GPUs are free. Every POLL seconds, start WORKERS_PER_GPU workers on each GPU that has NO compute
# process (a GPU another user holds is never shared), up to MAX_GPUS, until every chunk is claimed; workers
# claim chunks with mkdir locks, so a GPU that frees up late just picks up what is left. Exits 0 once every
# chunk is DONE, 1 if any chunk FAILED.
#
#   W=outputs/pi_mm_rem1400 MAX_GPUS=4 bash scripts/analysis/run_pi_chunks_free_gpus.sh
# ---------------------------------------------------------------------------------------------------
set -uo pipefail
W=${W:?scoring dir holding chunks/ and hints/}
W="$(cd "$W" && pwd)" || exit 1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ "$PWD" != / ] && ! { [ -f pyproject.toml ] && [ -f opsd/__init__.py ] && [ -f opsd/skyrl_ext/__init__.py ]; }; do cd ..; done; pwd)"
cd "$REPO" || exit 1
MAX_GPUS=${MAX_GPUS:-3}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-3}
POLL=${POLL:-120}
log() { echo "$(date -u +%H:%M:%S) $*"; }

free_gpus() {  # GPUs with no compute process and (almost) no memory in use
  local busy; busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits |
    while IFS=', ' read -r i u m; do
      grep -q "$u" <<< "$busy" || [ "$m" -gt 2000 ] || echo "$i"
    done
}

n_chunks=$(ls "$W"/chunks/chunk_*.jsonl | wc -l)
taken=" "; n_taken=0
log "$n_chunks chunks; waiting for free GPUs (max $MAX_GPUS, $WORKERS_PER_GPU workers each)"
while :; do
  n_done=$(ls "$W"/score/*/DONE "$W"/score/*/FAILED 2>/dev/null | wc -l)
  [ "$n_done" -ge "$n_chunks" ] && break
  n_claimed=$(ls -d "$W"/score/*.claim 2>/dev/null | wc -l)
  if [ "$n_claimed" -lt "$n_chunks" ] && [ "$n_taken" -lt "$MAX_GPUS" ]; then
    for g in $(free_gpus); do
      [[ $taken == *" $g "* ]] && continue
      [ "$n_taken" -ge "$MAX_GPUS" ] && break
      log "GPU $g is free: starting $WORKERS_PER_GPU workers"
      for _ in $(seq 1 "$WORKERS_PER_GPU"); do
        GPU=$g W=$W nohup bash scripts/analysis/run_pi_chunk_score.sh >> "$W/worker$g.log" 2>&1 &
        sleep 20
      done
      taken="$taken$g "; n_taken=$((n_taken + 1))
    done
  fi
  sleep "$POLL"
done
n_failed=$(ls "$W"/score/*/FAILED 2>/dev/null | wc -l)
if [ "$n_failed" -gt 0 ]; then log "FATAL: $n_failed chunk(s) FAILED -- see $W/score/*/logs"; exit 1; fi
log "all $n_chunks chunks DONE"

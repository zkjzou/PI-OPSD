#!/usr/bin/env bash
# Generic queue worker: scores chunks (chunks/chunk_<k>.jsonl + index) under every hints/hints_*.json (PI_GLOB) on ONE GPU.
# Workers claim chunks with mkdir locks, so starting another worker on a GPU that frees up later just
# picks up the remaining chunks:
#
#   W=<scoring dir> GPU=6 nohup bash scripts/analysis/run_pi_chunk_score.sh > <scoring dir>/worker6.log 2>&1 &
#
# Same kernel and settings as run_pi_glm8_score.sh (analyze_pi_positions.py, --input-format
# skyrl_eval, system-prepend, max-steps 50, max-seq-len 131072).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ "$PWD" != / ] && ! { [ -f pyproject.toml ] && [ -f opsd/__init__.py ] && [ -f opsd/skyrl_ext/__init__.py ]; }; do cd ..; done; pwd)"

W=${W:?set W to a prepared scoring dir (chunks/ + hints/)}
W="$(cd "$W" && pwd)" || exit 1   # absolute: the scorer runs from the repo root
PY=${PY:-$REPO/.venv/bin/python}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
GPU=${GPU:?set GPU to one free GPU index}

PIS=""
for f in "$W"/hints/${PI_GLOB:-hints_*.json}; do
  n=$(basename "$f" .json); PIS="${PIS:+$PIS,}${n#hints_}=$f"
done
mkdir -p "$W/score"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"  # `opsd` = the package at <repo root>/opsd
for c in $(ls "$W"/chunks/chunk_*.jsonl | sort -V); do
  k=$(basename "$c" .jsonl)
  mkdir "$W/score/$k.claim" 2>/dev/null || continue          # another worker owns this chunk
  out="$W/score/$k"; mkdir -p "$out/_shards" "$out/logs"
  echo "$(date -u +%H:%M:%S) GPU $GPU: $k"
  ok=0
  for attempt in 1 2 3; do
    if CUDA_VISIBLE_DEVICES=$GPU "$PY" -m opsd.alfworld.analyze_pi_positions \
        --trajectories "$c" --input-format skyrl_eval --pis "$PIS" --out-dir "$out" \
        --model-path "$MODEL" --device cuda:0 --num-shards 1 --shard-index 0 \
        --inject-style system-prepend --benchmark alfworld --max-steps 50 --max-seq-len 131072 \
        --ks 10,20,50,100 --lm-head-chunk 128 > "$out/logs/shard0.log" 2>&1 \
      && "$PY" -m opsd.alfworld.analyze_pi_positions --merge --out-dir "$out" --pis "$PIS" \
        --ks 10,20,50,100 > "$out/logs/merge.log" 2>&1; then
      ok=1; break
    fi
    sleep $((10 + RANDOM % 20))   # transient CUDA-init flake on a busy machine
  done
  if [ "$ok" = 1 ]; then touch "$out/DONE"; echo "$(date -u +%H:%M:%S) GPU $GPU: $k DONE"
  else touch "$out/FAILED"; echo "$(date -u +%H:%M:%S) GPU $GPU: $k FAILED (see $out/logs)"; fi
done
echo "$(date -u +%H:%M:%S) GPU $GPU: no chunks left"

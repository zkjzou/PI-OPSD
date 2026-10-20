#!/bin/bash
set -euo pipefail
# ----------------------------------------------------------------------------
# Provenance: how the 2400-task-pool trajectory-hint artifacts (hints_{actions,raw,raw_no_think}.json per
# reference author) were assembled. The pool was covered in two reference-trajectory rounds: a first round
# over ~1k tasks, then a second round over the pool's not-yet-hinted tasks. For each author this script
#   1. builds raw / raw_no_think / actions hints from the second round's won trajectories
#      (build_pi_hint_formats.py, 6000-char cap), and
#   2. merges them with the first round's artifact of the same format: a union by task_id in which an
#      existing hint is never replaced.
# The data bundle ships the merged artifacts; rerun this only to rebuild them from trajectories.
#
#   PREV_HINTS_DIR=<round-1 dir holding hints_<author>/hints_*.json> \
#   NEW_TRAJ_DIR=<dir holding trajectories_<author>.jsonl from round 2> \
#   OUT_DIR=<output dir> \
#     bash opsd/alfworld/finalize_2k_hints.sh
# AUTHORS (default "glm qwen37") names the per-author files. Outputs: $OUT_DIR/hints_<author>/hints_*.json
# (merged) and $OUT_DIR/hints_<author>_new/ (round 2 only).
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

PREV_HINTS_DIR="${PREV_HINTS_DIR:?set PREV_HINTS_DIR (round-1 dir holding hints_<author>/hints_*.json)}"
NEW_TRAJ_DIR="${NEW_TRAJ_DIR:?set NEW_TRAJ_DIR (dir holding trajectories_<author>.jsonl)}"
OUT_DIR="${OUT_DIR:?set OUT_DIR (output dir)}"
AUTHORS="${AUTHORS:-glm qwen37}"
PY="${PY:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || PY=python3
mkdir -p "$OUT_DIR"

for author in $AUTHORS; do
  echo "===== $author ====="
  # 1. build raw/raw_no_think/actions hints from the NEW won trajectories
  "$PY" -m opsd.alfworld.build_pi_hint_formats \
    --refs "$NEW_TRAJ_DIR/trajectories_${author}.jsonl" --out-dir "$OUT_DIR/hints_${author}_new" --max-chars-raw 6000
  # 2. merge new + existing (union by task_id) per format
  for fmt in hints_actions hints_raw hints_raw_no_think; do
    "$PY" - "$PREV_HINTS_DIR/hints_${author}/${fmt}.json" "$OUT_DIR/hints_${author}_new/${fmt}.json" \
          "$OUT_DIR/hints_${author}/${fmt}.json" <<'PYEOF'
import json, sys, os
old_p, new_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
def load(p):
    try: return json.load(open(p))
    except Exception: return {}
old, new = load(old_p), load(new_p)
merged = dict(old); n_add=0
for k,v in new.items():
    if k not in merged: merged[k]=v; n_add+=1
os.makedirs(os.path.dirname(out_p), exist_ok=True)
json.dump(merged, open(out_p,"w"))
print(f"  {os.path.basename(out_p)}: {len(old)} existing + {n_add} new = {len(merged)} total")
PYEOF
  done
done
echo "DONE: merged hints in $OUT_DIR/hints_<author>/"

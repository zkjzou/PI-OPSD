#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fill a student's SELF cells of the ALFWorld summariser/analyzer grid from 1113/1600 -> 1600/1600.
#   STUDENT=4bt|4bi|30bi  (default 4bt). Used to complete the self cells before training.
#
# WHY. The grid is 2 hint-writing prompts x 3 analyzer authors over the SAME glm-5.2 reference
# trajectories. The glm-5.2 and qwen3.7 cells are STUDENT-INDEPENDENT (the analyzer reads the glm
# trajectory, not the student's rollouts) so one artifact serves every student at 1600/1600. The
# SELF cells are per-student by construction, and 4bt's were built on the older 1,623-task
# allowlist, which covers only 1113 of the 1600-task train pool:
#
#     hints_onpolicy/4bt_from_glm.json    1113/1600  (69.6%)   <- 487 tasks -> STATIC hint
#     hints_summarized/4bt_from_glm.json  1113/1600  (69.6%)   <- same 487 tasks
#
# Launching at 69.6% would put ~30% generic-static-hint rollouts into two of six cells, which does
# not degrade those cells gracefully -- it silently makes them a different treatment from the other
# four. The 4bi self cells hit exactly this and were filled before use.
#
# FEASIBILITY IS ALREADY VERIFIED: the 487 missing task_ids are the SAME set in both cells, and
# glm's hints_raw.json covers 487/487 of them, so every TODO task has a source trajectory.
#
# PROVENANCE, read off the existing artifacts' own meta so the fill matches rather than approximates:
#     summarizer_model  Qwen3-4B-Thinking-2507        (self == the student being trained)
#     source_artifact   hints_glm/hints_raw.json      (--input-hints, NOT a rollout dump)
#     pi_type           onpolicy_skill | plain        (--prompt-style)
#     __schema__        opsd_hints/v2                 (--out-schema opsd_v2)
#
# SIZING. Measured over the 487 TODO tasks the source text is median 3,821 chars / max 5,880, so
# --max-chars 24000 is slack and the middle-elision path never triggers. The OUTPUT budget is the
# real trap: a thinking analyzer spends its completion on reasoning before emitting the hint, and
# max_tokens=1024 once yielded ZERO usable hints on long trajectories. Hence MAX_TOKENS=8192.
#
# PREREQUISITE -- A SERVED student model (vLLM, OpenAI-compatible). This needs GPUs, so the script takes
# the endpoint as input rather than starting one:
#
#     URL=http://127.0.0.1:8123 STUDENT=4bt bash scripts/analysis/fill_self_hints.sh
#
# It refuses to run if the endpoint is not serving that exact model, refuses to finish unless BOTH
# cells reach full train1600 coverage, and never overwrites an existing task's hint (the fill only closes
# gaps).
#
# INPUTS (default: the released dataset bundle under $OPSD_DATA_ROOT/alfworld):
#   GLM_RAW      hints/glm-5.2/hints_raw.json                     the source trajectories
#   TRAIN_IDS    splits/task_ids_train1600.json                   the coverage target
#   EXISTING_ONPOLICY / EXISTING_SUMMARIZED   the cells to fill (default: the released
#                hints/derived/{onpolicy_skill,summary}/<student>_from_glm.json if present, else empty --
#                the bundle's cells are already complete, so point these at a partial build to fill it)
#
#   env: URL (required) / STUDENT / MODEL / CONCURRENCY / MAX_TOKENS / OUT
# ---------------------------------------------------------------------------
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$REPO" != / ] && ! { [ -f "$REPO/pyproject.toml" ] && [ -f "$REPO/opsd/__init__.py" ] && [ -f "$REPO/opsd/skyrl_ext/__init__.py" ]; }; do REPO="$(dirname "$REPO")"; done
[ -f "$REPO/opsd/skyrl_ext/__init__.py" ] || { echo "FATAL: cannot locate the repo root above $0" >&2; exit 1; }
PY=${PY:-python}
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"  # `opsd` = the package at <repo root>/opsd
OPSD_DATA_ROOT=${OPSD_DATA_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data}
ALF=$OPSD_DATA_ROOT/alfworld

# STUDENT selects which per-student artifact pair is filled and which analyzer must be serving. The
# SELF cell is only meaningful if the analyzer IS the student, so both move together -- hence one knob,
# not two, and a gate below that refuses a mismatch.
STUDENT=${STUDENT:-4bt}
case "$STUDENT" in
  4bt)  MODEL=${MODEL:-Qwen3-4B-Thinking-2507};        EXPECT_MODEL_PAT=${EXPECT_MODEL_PAT:-*Thinking*} ;;
  4bi)  MODEL=${MODEL:-Qwen3-4B-Instruct-2507};        EXPECT_MODEL_PAT=${EXPECT_MODEL_PAT:-*4B-Instruct*} ;;
  30bi) MODEL=${MODEL:-Qwen3-30B-A3B-Instruct-2507};   EXPECT_MODEL_PAT=${EXPECT_MODEL_PAT:-*30B-A3B*} ;;
  *) echo "FATAL: unknown STUDENT=$STUDENT (want 4bt|4bi|30bi)" >&2; exit 1 ;;
esac
URL=${URL:?set URL to the served $MODEL endpoint, e.g. http://127.0.0.1:8102}
CONCURRENCY=${CONCURRENCY:-64}
MAX_TOKENS=${MAX_TOKENS:-8192}
MAX_CHARS=${MAX_CHARS:-24000}
OUT=${OUT:-$REPO/outputs/pi_fill_${STUDENT}_self}
GLM_RAW=${GLM_RAW:-$ALF/hints/glm-5.2/hints_raw.json}
TRAIN_IDS=${TRAIN_IDS:-$ALF/splits/task_ids_train1600.json}
EXISTING_ONPOLICY=${EXISTING_ONPOLICY:-$ALF/hints/derived/onpolicy_skill/${STUDENT}_from_glm.json}
EXISTING_SUMMARIZED=${EXISTING_SUMMARIZED:-$ALF/hints/derived/summary/${STUDENT}_from_glm.json}

mkdir -p "$OUT"

# ---- gate on the endpoint serving the RIGHT model. A wrong-model analyzer is the whole experiment.
echo "===== checking $URL ====="
_served=$(curl -s -m 10 "$URL/v1/models" | "$PY" -c 'import json,sys; print(",".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null || true)
echo "  serving: ${_served:-<none>}"
case "$_served" in
  $EXPECT_MODEL_PAT) ;;
  *) echo "FATAL: $URL is not serving $MODEL (got '${_served:-nothing}'). The SELF cell" >&2
     echo "       requires the STUDENT as analyzer; an Instruct endpoint would silently build a" >&2
     echo "       different arm under the self label." >&2; exit 1 ;;
esac

# ---- inputs: the glm source artifact + the train split + the existing cells (copied, never edited)
echo "===== staging inputs ====="
cp -f "$GLM_RAW" "$OUT/hints_glm_raw.json"
cp -f "$TRAIN_IDS" "$OUT/task_ids_train1600.json"
for cell in onpolicy summarized; do
  src=$EXISTING_ONPOLICY; [ "$cell" = summarized ] && src=$EXISTING_SUMMARIZED
  if [ -s "$src" ]; then cp -f "$src" "$OUT/existing_${cell}.json"
  else echo "  $cell: no existing cell at $src -- filling from empty"; echo '{"__schema__": "opsd_hints/v2"}' > "$OUT/existing_${cell}.json"; fi
done

# Recompute the allowlist here rather than trusting a stale file: it is the union of both cells' gaps.
"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
ids = json.loads((out / "task_ids_train1600.json").read_text())
if isinstance(ids, dict):
    ids = ids.get("task_ids", list(ids))
ids = set(ids)
todo = set()
for cell in ("onpolicy", "summarized"):
    d = json.loads((out / f"existing_{cell}.json").read_text()); d.pop("__schema__", None)
    gap = ids - set(d)
    print(f"  {cell:11s} existing {len(d):5d}  covers {len(ids & set(d)):4d}/{len(ids)}  gap {len(gap)}")
    todo |= gap
(out / "task_allowlist.json").write_text(json.dumps(sorted(todo), indent=1))
print(f"  TODO (union) = {len(todo)} task_ids -> task_allowlist.json")
PY

# ---- generate both prompt styles over the gap
NTODO=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$OUT/task_allowlist.json")
for pair in "onpolicy:onpolicy_skill" "summarized:plain"; do
  cell="${pair%%:*}"; style="${pair##*:}"
  if [ "$NTODO" = 0 ]; then echo "===== $cell: nothing to fill ====="; echo '{}' > "$OUT/fill_${cell}.json"; continue; fi
  echo "===== generating $cell (--prompt-style $style) ====="
  ( PYTHONPATH="$REPO" "$PY" scripts/analysis/gen_hints_local_models.py \
      --input-hints "$OUT/hints_glm_raw.json" \
      --task-allowlist "$OUT/task_allowlist.json" \
      --url "$URL" --model "$MODEL" \
      --prompt-style "$style" --env alfworld --outcome success \
      --max-chars "$MAX_CHARS" --max-tokens "$MAX_TOKENS" \
      --concurrency "$CONCURRENCY" --out-schema opsd_v2 \
      --out "$OUT/fill_${cell}.json" ) || { echo "FATAL: generation failed for $cell" >&2; exit 1; }
done

# ---- merge (existing WINS; the fill only closes gaps) + refuse to finish below full coverage
echo "===== merging + verifying ====="
"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
ids = json.loads((out / "task_ids_train1600.json").read_text())
if isinstance(ids, dict):
    ids = ids.get("task_ids", list(ids))
ids = set(ids)
ok = True
for cell in ("onpolicy", "summarized"):
    base = json.loads((out / f"existing_{cell}.json").read_text())
    schema = base.pop("__schema__", "opsd_hints/v2")
    fill = json.loads((out / f"fill_{cell}.json").read_text()); fill.pop("__schema__", None)
    added = 0
    for k, v in fill.items():
        if k not in base:                     # never overwrite an existing hint
            base[k] = v; added += 1
    cov = len(ids & set(base))
    merged = {"__schema__": schema, **base}
    (out / f"merged_{cell}.json").write_text(json.dumps(merged))
    flag = "OK" if cov == len(ids) else "INCOMPLETE"
    print(f"  {cell:11s} +{added:4d} -> {len(base):5d} tasks, covers {cov}/{len(ids)}  [{flag}]")
    ok &= cov == len(ids)
Path(out / "_full_coverage").write_text("1" if ok else "0")
PY

if [ "$(cat "$OUT/_full_coverage")" != "1" ]; then
  echo "INCOMPLETE: at least one cell is still short of full train1600 coverage." >&2
  echo "  Rerun to retry the failures (generation is resumable); merged_*.json are in $OUT." >&2
  exit 1
fi
echo "done. Full cells: $OUT/merged_onpolicy.json (anlz_self), $OUT/merged_summarized.json (summ_self)"

#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------------
# Score the SIX summarised hint PIs + TWO checklists -- each with a matched null -- on ALFWorld
# train200, for Qwen3-4B-Instruct-2507. 16 arms, one cell.
#
# WHY. The alfworld_train200_* cells carry only the six TRAJECTORY PIs, so every ALFWorld statement
# about hint-vs-trajectory rests on AppWorld. This is the ALFWorld counterpart of the AppWorld
# new12 + chk cells, on the same 200-task / 128-rollout pool the trajectory arms were scored on.
#
# INPUT FORMAT. ALFWorld's corpus is a SkyRL eval dump (input_prompt/output_response/score/
# env_extras), so it needs --input-format skyrl_eval; --trajectories' default `jsonl` schema expects
# prompt_token_ids/response_ids and would fail. AppWorld uses --appworld-transcripts instead.
#
# NULLS. ALFWorld task ids are `trial_T<timestamp>_<id>` with no variant structure, so the
# scenario-aware constraint used on AppWorld is vacuous here; the nulls are plain fixed-point-free
# derangements (verified 0 fixed points, length multiset preserved). They are not part of the dataset
# bundle: build them per student into NUL, or set WITH_NULLS=0 to score the 8 real arms only (the
# paper's D and Δd⁻_err use the real arms; the nulls are a control).
#
# INPUTS. CORPUS = the student's n=128 no-hint pass@k dump on the 200 train tasks
# (splits/task_ids_train200_seed0.json), named train200_<student>.jsonl; hints default to the released
# hint tree. Software: the published cells used torch 2.11 + transformers 5.8 (torch >= 2.8 provides
# grouped_mm; without it an MoE student falls back to an eager experts loop ~9x slower per shard).
# ---------------------------------------------------------------------------------------------------
set -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$REPO" != / ] && ! { [ -f "$REPO/pyproject.toml" ] && [ -f "$REPO/opsd/__init__.py" ] && [ -f "$REPO/opsd/skyrl_ext/__init__.py" ]; }; do REPO="$(dirname "$REPO")"; done
[ -f "$REPO/opsd/skyrl_ext/__init__.py" ] || { echo "FATAL: cannot locate the repo root above $0" >&2; exit 1; }
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"  # `opsd` = the package at <repo root>/opsd (cwd is the repo root)
export TOKENIZERS_PARALLELISM=false

OPSD_DATA_ROOT=${OPSD_DATA_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data}
HD=${HD:-$OPSD_DATA_ROOT/alfworld/hints}
# Overridable so one driver can serve every student. NUL especially: the null for `self` MUST be a
# derangement of THAT student's own hints (length multiset matched on the 200 scored tasks), and the
# self arm's length varies 2.5x across students -- a 4bi-derived null is no control for 4bt.
H_ONPOLICY=${H_ONPOLICY:-$HD/derived/onpolicy_skill}
H_SUMMARIZED=${H_SUMMARIZED:-$HD/derived/summary}
CHK_GLM=${CHK_GLM:-$HD/derived/checklist/checklist_from_glm.json}   # 4B-Instruct-loss build, every student
CHK_QWN=${CHK_QWN:-$HD/derived/checklist/checklist_from_q7.json}
NUL=${NUL:-$REPO/outputs/alf_nulls}
WITH_NULLS=${WITH_NULLS:-1}
PY=${PY:-python}
TOKEN_FLUSH=${TOKEN_FLUSH:-5}
GRID=${GRID:-$REPO/outputs/pi_grid}
CORPUS=${CORPUS:-$REPO/corpora/alf_passk128_dumps/train200_4bi.jsonl}
OUT=${OUT:-$GRID/alfworld_new8_train200_4bi}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
NSH=${NSH:-8}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-65536}          # never the scorer's 40000 default
KS=${KS:-"10,20,50,100"}

# SELF IS PER STUDENT, and was previously HARDCODED to 4bi_from_glm.json. The callers that produced the
# cells passed CORPUS/MODEL/NUL but NOT PIS, so
# every student's anlz_self/summ_self silently scored 4B-Instruct's analyzed rollouts -- a
# cross-student arm, not a self arm. The per-student files exist (4bt_from_glm.json,
# 30bi_from_glm.json) and the design assumed them: stage 4 rebuilds per-student NULLS for the self arm
# specifically, citing 2.5x hint-length variation across students. Derive the tag from the corpus
# filename (train200_<student>.jsonl) so a caller that sets CORPUS cannot get this wrong, and let the
# existing all-PIS-exist check fail loudly if the tag is bogus.
SELF_TAG=${SELF_TAG:-$(basename "$CORPUS" .jsonl | sed 's/^train200_//')}

_arm() {  # name file -> append name=file (and its matched null unless WITH_NULLS=0)
  PIS="${PIS:+$PIS,}$1=$2"
  [ "$WITH_NULLS" = 0 ] || PIS="$PIS,$1_null=$NUL/$1_null.json"
}
PIS=""
_arm anlz_glm  "$H_ONPOLICY/glm52_from_glm.json"
_arm anlz_qwn  "$H_ONPOLICY/qwen37_from_glm.json"
_arm anlz_self "$H_ONPOLICY/${SELF_TAG}_from_glm.json"
_arm summ_glm  "$H_SUMMARIZED/glm52_from_glm.json"
_arm summ_qwn  "$H_SUMMARIZED/qwen37_from_glm.json"
_arm summ_self "$H_SUMMARIZED/${SELF_TAG}_from_glm.json"
_arm chk_glm   "$CHK_GLM"
_arm chk_qwn   "$CHK_QWN"

[ -s "$CORPUS" ] || { echo "FATAL: missing corpus $CORPUS" >&2; exit 1; }
# Record the resolved arm -> hint mapping. Without it a finished cell cannot be audited for which PI
# it actually consumed: rows.parquet has no hint column, ntok counts only the policy's own tokens, and
# there is no config sidecar -- which is why the 4bi-self defect above was invisible in the outputs.
mkdir -p "$OUT"
{ echo "# resolved $(date -u +%FT%TZ)  SELF_TAG=$SELF_TAG  MODEL=$MODEL  CORPUS=$CORPUS"
  echo "$PIS" | tr ',' '\n'; } > "$OUT/pis_resolved.txt"
for spec in ${PIS//,/ }; do
  f="${spec#*=}"; [ -s "$f" ] || { echo "FATAL: missing hints $f (nulls: build them or WITH_NULLS=0)" >&2; exit 1; }
done
mkdir -p "$OUT"
echo "=== ALFWorld new8 | $(wc -l < "$CORPUS") rollouts | $(echo "$PIS" | tr ',' '\n' | wc -l) arms | $NSH shards | out=$OUT"
echo "[pi] arms: $(echo "$PIS" | tr ',' '\n' | cut -d= -f1 | tr '\n' ' ')"

# SHARD_IDS lets a cell be finished on FEWER cards than shards, in waves (e.g. when only some GPUs are
# free). Each wave writes its own shard<N>.rc; those persist in $OUT, so
# the all-8-shards merge gate below passes once the final wave lands. Shard indices are paired
# POSITIONALLY with GPUS, and a card may repeat: SHARD_IDS="0 1 2" GPUS="2 2 2" runs three shards on
# GPU 2. Keep NSH at the cell's original shard count or --resume will key on a different partition.
SHARD_IDS=${SHARD_IDS:-}
read -ra _SIDS <<< "$SHARD_IDS"
read -ra _GPUS <<< "$GPUS"
pids=(); i=0
for ((k=0; k<${#_GPUS[@]}; k++)); do
  G=${_GPUS[k]}
  if [ ${#_SIDS[@]} -gt 0 ]; then
    [ "$k" -lt "${#_SIDS[@]}" ] || break
    i=${_SIDS[k]}
  else
    [ "$k" -ge "$NSH" ] && break
    i=$k
  fi
  (
    CUDA_VISIBLE_DEVICES="$G" "$PY" \
      opsd/alfworld/analyze_pi_positions.py \
      --trajectories "$CORPUS" --input-format skyrl_eval --benchmark alfworld \
      --pis "$PIS" --out-dir "$OUT" \
      --model-path "$MODEL" --device cuda:0 \
      --inject-style system-prepend --ks "$KS" --max-steps 30 \
      --max-seq-len "$MAX_SEQ_LEN" --lm-head-chunk 128 \
      --num-shards "$NSH" --shard-index "$i" --resume \
      --token-flush-every "$TOKEN_FLUSH" \
      --save-token-logprobs > "$OUT/shard${i}.log" 2>&1
    echo "$?" > "$OUT/shard${i}.rc"
  ) &
  pids+=($!)
done
wait "${pids[@]}"
bad=0
for s in $(seq 0 $((NSH - 1))); do
  rc=$(cat "$OUT/shard${s}.rc" 2>/dev/null || echo 1)
  [ "$rc" = 0 ] || { echo "SHARD $s FAILED rc=$rc -- see $OUT/shard${s}.log" >&2; bad=1; }
done
[ "$bad" = 0 ] || { echo "FATAL: refusing to merge a partial cell" >&2; exit 1; }

"$PY" opsd/alfworld/analyze_pi_positions.py \
  --merge --out-dir "$OUT" --pis "$PIS" --benchmark alfworld > "$OUT/merge.log" 2>&1 \
  || { echo "FATAL: merge failed" >&2; exit 1; }

# rollout_attrs.parquet is NOT produced by --merge; it is an optional per-rollout aggregation from the
# ranking study (pi_token_aggregate.py, not released, so this branch is skipped in a release checkout).
# Nothing downstream needs it: alf_pi_divergence_block.py falls back to rows.parquet for the uid set.
if [ -f scripts/analysis/pi_token_aggregate.py ]; then
  "$PY" scripts/analysis/pi_token_aggregate.py \
    --tokens "$OUT/tokens.parquet" --rows "$OUT/rows.parquet" \
    --out "$OUT/rollout_attrs.parquet" > "$OUT/rollout_attrs.log" 2>&1 \
    || { echo "FATAL: token aggregate failed" >&2; exit 1; }
else
  echo "NOTE: pi_token_aggregate.py not present -- rollout_attrs.parquet not built (rows.parquet is complete)"
fi
echo "ALF_NEW8_SCORING_DONE -> $OUT"

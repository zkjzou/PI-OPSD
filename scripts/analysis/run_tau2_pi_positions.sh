#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------------
# τ²-bench retail-base PI-positions scoring — the 6-arm grid on a BALANCED 8-win/8-loss pool.
#
#   6 arms = {glm-5.2, qwen3.7} x {raw, raw_no_think, actions}, hints built by run_tau2_pi_gen.sh
#   against retail-base (bare-ordinal task ids "0".."113" -- NOT the AReaL `retail_<n>` namespace).
#
# WHY BALANCED. retail-base at n=128 is near-50/50 per task (median 60.5 wins), so 8 wins + 8 losses
# per task removes the class-imbalance confound outright and makes within-task AUC comparable across
# tasks AND models. Eligible = >=8 wins AND >=8 losses AND covered by BOTH hint sources:
# 4bi 76 tasks / 4bt 78 / 30bi 85 (each model uses its own full eligible set).
#
# ⚠️ --max-seq-len 65536, NOT the 40000 default. Qwen3-4B-Thinking's base (prompt+response) alone maxes
# at 47,906 tokens and its worst hinted seq is 59,821 (91% of 65536). At the default the whole upper
# tail drops, and UNEQUALLY PER ARM because arms differ in hint length -- which voids the within-task
# contrast. Audited offline: too_long == 0 for all 6 arms on all 3 models. Re-audit if the pool changes.
#
# ⚠️ --benchmark tau2 is load-bearing. τ² actions are <tool_call> blocks; under the alfworld/appworld
# tag sets every action token falls in `other` and the channel dimension silently collapses.
#
# Verified smoke signature (--limit 8): `wrote 48 ... records`, `skips={}`, and channel tallies with
# think>0 for 4bt / think==0 for the two Instruct models.
#
#   POOL_DIR=<dir of <tag>.jsonl corpora> bash scripts/analysis/run_tau2_pi_positions.sh   # all three
#   MODELS=4bt POOL_DIR=... bash .../run_tau2_pi_positions.sh                                  # one model
#   LIMIT=8 NUM_SHARDS=1 POOL_DIR=... bash .../run_tau2_pi_positions.sh                        # smoke
#
# PATHS. POOL_DIR holds one no-hint corpus per student (<tag>.jsonl, baseline.py trajectories) -- the
# corpora are not part of the data bundle; regenerate them with baseline.py (see stage_tau2_m75_scoring.sh
# for the matched-75 pool). TAU2_PI_ROOT is the hint tree (default the published
# $OPSD_DATA_ROOT/tau2/hints/t211, the 211/matched-75 cells' hints; the retail-base hints of the header
# are not published). Cells are written to SCORE_ROOT/<tag> (default $OUT_ROOT/tau2_score).
# ---------------------------------------------------------------------------------------------------
set -eo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_tau2_env.sh"
cd "$REPO_ROOT"

MODELS=${MODELS:-"4bt 30bi 4bi"}
NUM_SHARDS=${NUM_SHARDS:-4}
# SHARD -> PHYSICAL GPU, positionally. `--device cuda:$S` hardcoded the shard index as the card index,
# which silently lands on whatever a CO-TENANT is holding on a shared box (GPUs 0 and 3 were another
# user's training job when this was written). GPUS lets the caller pick free cards.
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
read -ra _GPU <<< "$GPUS"
[ "${#_GPU[@]}" -ge "$NUM_SHARDS" ] || { echo "FATAL: NUM_SHARDS=$NUM_SHARDS > ${#_GPU[@]} GPUs in GPUS='$GPUS'" >&2; exit 1; }
SCORE_ROOT=${SCORE_ROOT:-$OUT_ROOT/tau2_score}
P=${TAU2_PI_ROOT:-$T2/hints/t211}
# ARM SET AND OUTPUT ARE OVERRIDABLE. The 6 trajectory arms are only half the story: anlz/summ x
# {glm,qwn} exist in the analyzer_rb tree and cost NOTHING in pool size (they cover 108 tasks vs
# the binding qwen37 arms' 106, so the 10-arm eligible set is IDENTICAL to the published 6-arm one --
# 76/78/85). The 2 checklist arms cover only 86, which drops the pool to 65/70/74, so they belong in a
# SEPARATE cell rather than silently shrinking the headline table.
# MUST be the _rb files: pi_tau2/hints_analyzer/ is the AReaL retail_<n> namespace (424 keys, zero
# bare-ordinal) while these corpora use bare ordinals "0".."113" -- the wrong one gives zero coverage.
A=$P/analyzer_rb
C=$P/checklist
PIS_TRAJ="glm_raw=$P/glm/hints_base/hints_raw.json,glm_raw_no_think=$P/glm/hints_base/hints_raw_no_think.json,glm_actions=$P/glm/hints_base/hints_actions.json,qwen37_raw=$P/qwen/hints_base/hints_raw.json,qwen37_raw_no_think=$P/qwen/hints_base/hints_raw_no_think.json,qwen37_actions=$P/qwen/hints_base/hints_actions.json"
PIS_ANLZ="anlz_glm=$A/anlz_glm.json,anlz_qwn=$A/anlz_qwn.json,summ_glm=$A/summ_glm.json,summ_qwn=$A/summ_qwn.json"
# CHECKLISTS ARE PER STUDENT ON SOME POOLS. On retail-base one pair covers all three students (86/103
# each). On the 211 training pool they differ -- 4bi 193 of 211, 4bt/30bi ~210 -- so a shared path would
# hand one student another student's checklist and score it as its own, exactly the defect that put
# 4B-Instruct's analyser into 4bt's and 30bi's ALFWorld self rows (since fixed). If a per-student dir
# exists ($CHK_ROOT/checklist_<tag>) it wins; otherwise fall back to the shared pair.
chk_dir() { [ -d "${CHK_ROOT:-$P}/checklist_$1" ] && echo "${CHK_ROOT:-$P}/checklist_$1" || echo "$C"; }
ARMSET=${ARMSET:-10}
# ⚠️ THE SELF ARMS ARE PER STUDENT, so ARMSET=14 resolves PIS **inside** the model loop below, keyed on
# $TAG. Building it once here is precisely the defect that put 4B-Instruct's analyser into 4B-Thinking's
# and 30B-A3B's "self" rows on ALFWorld (since fixed) -- a cross-student arm reported as a self arm.
S=${TAU2_SELF_DIR:-$P/self_rb}
build_pis() {   # $1 = student tag
  local self="anlz_self=$S/anlz_self_$1.json,summ_self=$S/summ_self_$1.json"
  local cd_; cd_=$(chk_dir "$1")
  local PIS_CHK="chk_glm=$cd_/glm/hints_checklist.json,chk_qwn=$cd_/qwn/hints_checklist.json"
  case "$ARMSET" in
    6)  echo "$PIS_TRAJ" ;;
    10) echo "$PIS_TRAJ,$PIS_ANLZ" ;;
    12) echo "$PIS_TRAJ,$PIS_ANLZ,$self" ;;
    14) echo "$PIS_TRAJ,$PIS_ANLZ,$self,$PIS_CHK" ;;
    *)  echo "FATAL: ARMSET must be 6|10|12|14 (got '$ARMSET')" >&2; return 1 ;;
  esac
}
PIS=${PIS:-$(build_pis "$(set -- $MODELS; echo "$1")")} || exit 1
# reject an empty hint map, not just a missing file: a failed generator writes `{}` (2 bytes), which
# passes -s and would silently make an arm a no-hint control (cost a 7-of-8-arm AppWorld cell earlier)
for _spec in ${PIS//,/ }; do
  _f="${_spec#*=}"
  [ -s "$_f" ] || { echo "FATAL: missing hints $_f" >&2; exit 1; }
  _n=$("$PY" -c "import json,sys;print(len([k for k in json.load(open(sys.argv[1])) if not str(k).startswith('__')]))" "$_f" 2>/dev/null)
  [ "${_n:-0}" -ge 1 ] 2>/dev/null || { echo "FATAL: empty hint map $_f (${_n:-unparseable} keys)" >&2; exit 1; }
done
echo "[tau2] ARMSET=$ARMSET -> $(echo "$PIS" | tr ',' '\n' | wc -l) arms"

# --max-seq-len IS PER STUDENT, and 65536 is NOT enough for 4B-Thinking on the luna n=32 corpus.
# Audited offline (2026-09-13; audit_tau2_m75_lengths.py re-implements that audit for any pool), max
# over every (rollout, arm) of prompt+response+hint:
#   4bi   max base 21,129   worst hinted 34,947 (qwn_raw)   -- clears even the 40000 default
#   30bi  max base 31,680   worst hinted 44,710 (qwn_raw)
#   4bt   max base 52,564   worst hinted 71,567 (qwn_raw)   -- 6,031 OVER 65536
# `qwen37_raw` binds every time: it is the longest hint (median 7,380 tok vs 160 for summ_qwn), so the
# cap must clear BASE + the LONGEST arm, not the base. Truncation here is not a uniform loss -- it drops
# the upper tail unequally per arm, which is exactly what voids the within-task contrast. Re-audit when
# the corpus or the hint set changes; do not assume the old n=128 numbers.
max_seq_for() {
  case "$1" in
    4bt)  echo 73728 ;;
    *)    echo 65536 ;;
  esac
}

model_path() {
  case "$1" in
    4bi)  echo Qwen/Qwen3-4B-Instruct-2507 ;;
    4bt)  echo Qwen/Qwen3-4B-Thinking-2507 ;;
    30bi) echo Qwen/Qwen3-30B-A3B-Instruct-2507 ;;
    *) echo "unknown model tag $1" >&2; exit 1 ;;
  esac
}

# ⚠️ `wait` returning is NOT the same as the GPUs being free. MEASURED: 30bi's four shards logged DONE
# at 20:12-20:13, the merge finished at 20:16, and 4bi's shards then OOMed on THREE of four GPUs with
# "7.17 GiB free of 79.18" -- one 30B copy (~72 GB) was still resident minutes after its process exited.
# Only shard 1 survived, the run merged 25% of the data, and it printed ALL DONE. So gate on real free
# memory, not on process exit.
wait_for_gpus() {
  local need_mib=$1 n=$2 deadline=$((SECONDS + ${GPU_WAIT_SECS:-600}))
  while :; do
    local ready=0
    for g in "${_GPU[@]:0:$n}"; do
      local free
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 0)
      [ "${free:-0}" -ge "$need_mib" ] && ready=$((ready+1))
    done
    [ "$ready" -ge "$n" ] && { echo "  [gpu-gate] $n GPU(s) have >= ${need_mib}MiB free"; return 0; }
    [ "$SECONDS" -ge "$deadline" ] && { echo "  [gpu-gate] TIMEOUT: only $ready/$n GPUs free >= ${need_mib}MiB"; return 1; }
    sleep 15
  done
}

for TAG in $MODELS; do
  MP=$(model_path "$TAG")
  # POOL_DIR is overridable: ARMSET=12 needs a pool restricted to the checklist arms' 86 covered tasks
  # (65/70/74), while ARMSET=10 uses the full balanced pool (76/78/85). Scoring the two arm sets on one
  # pool would either shrink the headline table ~13% or score chk arms on tasks they cannot cover.
  TRAJ=${POOL_DIR:?set POOL_DIR to the directory holding the per-student no-hint corpora}/$TAG.jsonl
  OUT=$SCORE_ROOT/$TAG
  [ -f "$TRAJ" ] || { echo "MISSING $TRAJ -- build the balanced subset first"; exit 1; }
  # DROP NON-TERMINATED ROWS BEFORE SCORING, NOT AFTER. τ² dumps rows whose `status` is not
  # `terminated_*` (policy_api_error / environment_error from a dead endpoint); the corpus README says
  # to exclude them, and they carry response_messages=[] so the scorer emits `no_tokens` for every arm.
  # Left in, NROLL counts a rollout that CANNOT be scored, so the coverage gate fails and `no_tokens`
  # trips the unexpected-skip gate -- both of which then look like an arms mismatch rather than one dead
  # row. Measured: 4bi has exactly 1 of 2,874 (task 76 trial 8, 0 response messages); 4bt and 30bi have
  # none, so their cells are unaffected. Filtering here keeps both gates strict.
  NRAW=$(wc -l < "$TRAJ")
  FILT="$SCORE_ROOT/_filtered/$(basename "$TRAJ" .jsonl).terminated.jsonl"
  mkdir -p "$(dirname "$FILT")"
  if [ ! -s "$FILT" ] || [ "$TRAJ" -nt "$FILT" ]; then
    "$PY" - "$TRAJ" "$FILT" <<'EOPY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
kept = dropped = 0
with open(src) as fh, open(dst, "w") as out:
    for line in fh:
        if not line.strip():
            continue
        if str(json.loads(line).get("status", "")).startswith("terminated"):
            out.write(line); kept += 1
        else:
            dropped += 1
print(f"  [filter] kept {kept} terminated rows, dropped {dropped} non-terminated", flush=True)
EOPY
  fi
  TRAJ="$FILT"
  NROLL=$(wc -l < "$TRAJ")
  [ "$NROLL" = "$NRAW" ] || echo "  [filter] scoring $NROLL of $NRAW rows ($((NRAW - NROLL)) non-terminated dropped)"
  # MSL MUST BE RESOLVED BEFORE THE GATE, WHICH READS IT. It used to be assigned after
  # wait_for_gpus, so `_eff=$MSL` was EMPTY on the first model of a run (and stale on later ones),
  # every `[ "$_eff" -le N ]` errored with "integer expression expected", and the ladder fell through
  # to its largest bucket _act=24000. For 4B that is 8000+24000=32000MiB, which passes on any idle
  # card, so the bug was invisible for two months. For 30B it is 61000+24000=85000MiB -- MORE THAN AN
  # 80 GB CARD HAS -- so 30bi could never satisfy its own gate and wait_for_gpus would spin until
  # GPU_WAIT_SECS and abort. Measured need at cap 49,152 is 74.2 GiB; the gate was demanding 83.
  MSL=${MAX_SEQ_LEN:-$(max_seq_for "$TAG")}
  NARM=$(echo "$PIS" | tr ',' '\n' | grep -c .)
  echo "=== $TAG ($MP): $NROLL rollouts x $NARM PIs, $NUM_SHARDS shards ==="
  mkdir -p "$OUT"
  # 30B needs ~58 GB weights + activations; 4B ~8 GB. Ask for a margin over the weights either way.
  # THE GATE MUST SCALE WITH THE CAP, NOT JUST THE MODEL. A flat NEED=20000 was calibrated when 4bi ran
  # at ~11 GB/shard with a 40k cap; a card with 21 GB free passes a 20 GB gate and then OOMs partway in,
  # so the gate would be protecting against "GPU busy" while the real constraint is "cap does not fit".
  #
  # DIRECTLY MEASURED 2026-09-20 (torch.cuda.max_memory_reserved after one bf16 forward at the cap with
  # use_cache=False, output_hidden_states=True -- i.e. exactly what signals/logprob.py issues):
  #   4bt  @73,728  29.1 GiB  (7.6 weights + 13.0 hidden_states over all 37 layers + transient)
  #   30bi @49,152  74.2 GiB  (58.1 weights +  9.2 hidden_states) -- fits 80 GB with 5.4 GiB to spare
  #   30bi @57,344  77.4 GiB  -- also fits, but 2.2 GiB of headroom is too thin to stake a run on,
  #                              which is why PI_CAP=49152 is the 211-pool choice despite the audit
  #                              allowing 57,344 and despite the 0.1% qwn_raw truncation it costs.
  # An earlier revision of this comment claimed "4B @73,728 = 74 GB" and that number does NOT reproduce
  # in isolation -- it was most likely an nvidia-smi read that included another tenant on a shared box.
  # It mattered: it made a 4bt shard look like it needed a whole 80 GB card when ~30 GB is enough, so
  # a local 8-shard run was talked down to 2. Re-measure rather than inherit a remembered figure.
  #
  # The NEED ladder below is left UNCHANGED and is now known to be conservative (4bt asks 32 GB for a
  # 29 GiB peak; 30bi @49k asks 77 GB for 74.2). Conservative is the safe direction for a gate: asking
  # too much only delays a start, asking too little OOMs every shard. Do not tighten it to match.
  # THE GATE ANSWERS "IS THIS CARD FREE ENOUGH TO START", NOT "WILL THIS FIT". Those are different
  # questions and conflating them cost real time: an earlier version keyed NEED on --max-seq-len, but
  # memory tracks the ACTUAL sequence processed and the cap only bounds it. On the 211 pool 30bi's base
  # p50 is 6,362 tokens and p95 11,099, so nearly every forward is ~19k even where the cap is 49k --
  # demanding cap-sized memory would block a card that would have run the whole cell fine.
  #
  # So: demand the weights (which ARE always resident) plus a modest activation floor keyed on the
  # EFFECTIVE max length = min(cap, audited worst actual). This deliberately does NOT predict OOM --
  # the activation curve is only anchored on a 4B (~21k -> 11 GB total, ~71.5k -> 74 GB) and
  # extrapolating it to a 30B-A3B, whose hidden size is SMALLER (2048 vs 2560) and which activates 3B
  # of 30B params per token, would be inventing a number. OOM risk is handled where it belongs: the
  # offline length audit picks the cap, and _shards resume bounds what a crash costs.
  case "${MSL:-}" in ''|*[!0-9]*) echo "!! MSL is '${MSL:-}' -- the gate cannot be keyed on it. Aborting." >&2; exit 1 ;; esac
  _eff=$MSL
  if [ -n "${AUDIT_WORST:-}" ] && [ "$AUDIT_WORST" -lt "$MSL" ]; then _eff=$AUDIT_WORST; fi
  case "$TAG" in
    30bi) _w=61000 ;;   # 30B-A3B bf16 weights, always resident
    *)    _w=8000  ;;   # 4B bf16 weights
  esac
  if   [ "$_eff" -le 24576 ]; then _act=6000
  elif [ "$_eff" -le 40960 ]; then _act=14000
  elif [ "$_eff" -le 57344 ]; then _act=18000
  else                             _act=24000
  fi
  NEED=$(( _w + _act ))
  echo "  [gate] $TAG: weights ${_w}MiB + activation floor ${_act}MiB at effective len ${_eff} -> need ${NEED}MiB/GPU"

  echo "  [gate] $TAG needs >= ${NEED}MiB free per GPU at cap $MSL"
  wait_for_gpus "$NEED" "$NUM_SHARDS" || { echo "!! GPUs never freed for $TAG (need ${NEED}MiB x $NUM_SHARDS) -- aborting"; exit 1; }
  # PIS is per student (the self arms); resolve it here and record what was consumed. A finished cell
  # is otherwise unauditable: rows.parquet has no hint column and there is no config sidecar.
  PIS=$(build_pis "$TAG") || exit 1
  echo "  [seq] --max-seq-len $MSL for $TAG"
  { echo "# resolved $(date -u +%FT%TZ)  ARMSET=$ARMSET  MODEL=$MP  MAX_SEQ_LEN=$MSL  TRAJ=$TRAJ"
    echo "$PIS" | tr ',' '\n'; } > "$OUT/pis_resolved.txt"
  # --resume IS NOT OPTIONAL ON A PREEMPTIBLE NODE. Without it analyze_pi_positions opens
  # done.shard{i}.jsonl and positions.shard{i}.jsonl with mode "w", so a restart TRUNCATES the very
  # ledger a restarted job just restored, leaves part_next at 0 so tokens.shard{i}.part0.parquet
  # is overwritten, and re-scores the whole pool. Measured 2026-09-20: the 4bt 211 cell was preempted
  # at 3,095/6,792 rollouts, restarted, and came back with done.shard0.jsonl cut
  # from 390 entries to 36 -- every hour of work destroyed, silently, while a periodic sync made it look
  # like state was being preserved. Everything else for resume was already wired up (the job's
  # _shards restore, the `resume_already_done` allowlist in the skip gate, do_merge's de-duplication
  # on (pi,row_idx,repetition_id,step,channel)); only the flag was missing.
  # Safe because a rollout killed mid-way has rows but no done-marker, so it is re-scored and its
  # partial rows are appended twice, then de-duplicated at merge. Verified the key is not degenerate
  # on tau2: row_idx is null for every record, but repetition_id is GLOBALLY unique (0..3112, 3,095
  # distinct over 8 shards, zero collisions), so (row_idx, repetition_id) still identifies a rollout.
  for S in $(seq 0 $((NUM_SHARDS-1))); do
    G=${_GPU[$S]}
    PYTHONPATH="$REPO_ROOT" $PY -m opsd.alfworld.analyze_pi_positions \
      --trajectories "$TRAJ" --input-format tau2 --benchmark tau2 \
      --pis "$PIS" --out-dir "$OUT" \
      --model-path "$MP" --device "cuda:$G" \
      --inject-style system-prepend --max-seq-len "$MSL" --max-steps 60 \
      --save-token-logprobs ${LIMIT:+--limit $LIMIT} \
      --num-shards "$NUM_SHARDS" --shard-index "$S" --resume \
      > "$OUT/shard$S.log" 2>&1 &
  done
  wait
  echo "--- shard results ($TAG) ---"
  grep -h "DONE:" "$OUT"/shard*.log || { echo "NO SHARD COMPLETED -- see $OUT/shard*.log"; exit 1; }
  # ⚠️ EVERY shard must finish, not just one. The earlier version of this guard only failed when ZERO
  # shards completed, so a 3-of-4 CUDA-OOM wipeout passed it: the run merged 25% of 4bi (34,776 rows vs
  # ~151k expected) and printed ALL DONE. "At least one succeeded" is not an invariant worth asserting.
  NDONE=$(grep -h -c "DONE:" "$OUT"/shard*.log | awk '{n+=$1} END{print n+0}')
  case "$NDONE" in ''|*[!0-9]*) echo "!! could not count DONE lines (got '$NDONE') -- refusing to merge" >&2; exit 1 ;; esac
  if [ "$NDONE" -ne "$NUM_SHARDS" ]; then
    echo "!! ONLY $NDONE/$NUM_SHARDS SHARDS COMPLETED for $TAG -- partial merge would be silent. Aborting."
    grep -l -E "OutOfMemoryError|Traceback" "$OUT"/shard*.log 2>/dev/null | sed 's/^/     failed: /'
    exit 1
  fi
  # And they must jointly cover every rollout -- MINUS the ones no scorer could ever mark done.
  # `no_assistant` and `no_tokens` `continue` in the OUTER rollout loop, so such a rollout never gets a
  # done-marker and the coverage sum is legitimately short. Measured 2026-09-20: 30bi on the 211 pool
  # logged exactly one `no_assistant` and the gate aborted on 6750 != 6751 after five hours of
  # scoring, with every shard healthy. Reconcile against those counts instead of demanding equality;
  # a genuine tiling failure (a dead shard, a bad partition) still fails because it does not come with
  # a matching structural-skip count.
  NSEEN=$(cat "$OUT"/_shards/done.shard*.jsonl 2>/dev/null | wc -l | awk '{print $1+0}')
  # `|| true`: grep exits 1 when the key is ABSENT, and under `set -eo pipefail` that killed the whole
  # run SILENTLY -- after every shard had finished, with no message and no merge. Measured 2026-09-22:
  # the tau2 matched-75 pool has zero no_assistant/no_tokens skips, so 2 shards x 1200 rollouts of GPU
  # work exited 1 at this line. A clean pool must not be fatal.
  NUNSCORABLE=$( (grep -h "DONE:" "$OUT"/shard*.log | grep -oE "'(no_assistant|no_tokens)': [0-9]+" \
                | grep -oE "[0-9]+$" || true) | awk '{n+=$1} END{print n+0}')
  if [ -z "$LIMIT" ] && [ $((NSEEN + NUNSCORABLE)) -ne "$NROLL" ]; then
    echo "!! COVERAGE $NSEEN done + $NUNSCORABLE unscorable != $NROLL rollouts for $TAG -- shards do not tile the pool. Aborting."; exit 1
  fi
  [ "$NUNSCORABLE" -gt 0 ] && echo "  [skips] $NUNSCORABLE rollout(s) unscorable (no assistant tokens after mask reconstruction) -- excluded from the pool"
  # skips MUST be empty; a non-empty dict means arms were scored on different rollouts.
  # SKIPS: empty is the invariant for every arm EXCEPT the checklists, which cover only 86 of the 103
  # retail-base tasks by construction (documented in the corpus README). That gap is known and is
  # handled at ANALYSIS time -- the 12 non-chk arms still cover all 103, so one cell yields both a
  # 12-arm/103-task view and a 14-arm/86-task view via the usual uid intersection. Any OTHER skip key
  # is the real failure this guards against (arms silently scored on different rollouts), so allow
  # `no_hint::chk_*` explicitly and still abort on anything else.
  # The allowlist covers three DIFFERENT things; do not collapse them.
  #   no_hint::chk_*        per (rollout,arm) -- checklists cover only part of the pool by construction
  #   too_long[_base]       per (rollout,arm) -- the --max-seq-len cap. EXPECTED when the cap is set
  #     no_im_end                              below the audited worst on purpose (30bi 211 at 49,152
  #                                            trades 9 rollout-arms for 5.4 GiB of headroom)
  #   no_assistant/no_tokens  per ROLLOUT -- structurally unscorable, reconciled by the coverage gate
  #   mask_reconstructed    NOT a skip at all -- a counter for "loss_mask rebuilt from markers"
  # All of them are handled at ANALYSIS time by the uid intersection, which drops an affected rollout
  # from EVERY arm so the arms stay on identical rollouts -- which is the invariant this gate exists
  # to protect. Anything outside the list is still fatal.
  BADSKIP=$(grep -h "DONE:" "$OUT"/shard*.log | grep -oE "skips=\{[^}]*\}" \
            | grep -oE "'[^']+':" | sed "s/^'//; s/':$//" \
            | grep -vE "^(no_hint::|resume_already_done$|too_long$|too_long_base$|no_im_end$|no_assistant$|no_tokens$|mask_reconstructed$|_parity_checked$)" | sort -u || true)
  # `no_hint::` is allowlisted for EVERY arm, not just chk_*. On the retail-base pool only the
  # checklists had gaps; on the tau2 matched-75 pool EVERY arm does, because a PI arm needs a
  # reference trajectory and only 57/75 tasks have a glm one, 56 a qwen37 one, 45-46 a checklist.
  # Verified 2026-09-22 on 30bi: for all 14 arms the skip count equals (75 - covered) x 32 EXACTLY
  # (e.g. glm_raw 18x32=576 observed 576; chk_glm 30x32=960 observed 960), i.e. these are structural
  # coverage gaps, not silently dropped hints. The "arms on identical rollouts" invariant is restored
  # at ANALYSIS time by the uid intersection. Reconcile the counts below rather than assuming.
  if [ -n "$BADSKIP" ]; then
    echo "!! UNEXPECTED SKIPS for $TAG -- arms are NOT on identical rollouts:"; echo "$BADSKIP" | sed 's/^/     /'
    exit 1
  fi
  # A cap-driven drop is only acceptable while it stays TINY. Allowlisting `too_long` without a ceiling
  # would let a badly chosen cap silently delete the upper tail of the longest arm, which is exactly
  # what voids the within-task contrast, so bound it at 1% of all (rollout, arm) pairs.
  NARMS=$(echo "$PIS" | tr ',' '\n' | grep -c .)
  NLONG=$( (grep -h "DONE:" "$OUT"/shard*.log | grep -oE "'(too_long|too_long_base|no_im_end)': [0-9]+" \
          | grep -oE "[0-9]+$" || true) | awk '{n+=$1} END{print n+0}')   # || true: see NUNSCORABLE
  if [ "$NLONG" -gt 0 ]; then
    LIMIT_LONG=$(( NROLL * NARMS / 100 ))
    echo "  [skips] $NLONG of $((NROLL * NARMS)) rollout-arms dropped by --max-seq-len $MSL (ceiling $LIMIT_LONG)"
    if [ "$NLONG" -gt "$LIMIT_LONG" ]; then
      echo "!! CAP $MSL DROPPED $NLONG rollout-arms (>1%) for $TAG -- the upper tail is being deleted unequally per arm. Aborting."
      exit 1
    fi
  fi
  NCHKSKIP=$( (grep -h "DONE:" "$OUT"/shard*.log | grep -oE "'no_hint::chk_[a-z]+': [0-9]+" | grep -oE "[0-9]+$" || true) | awk '{n+=$1} END{print n+0}')   # || true: see NUNSCORABLE
  [ -n "$NCHKSKIP" ] && echo "  [skips] $NCHKSKIP checklist rollout-arms uncovered (expected: chk covers 86/103 tasks)"
  NHINTSKIP=$( (grep -h "DONE:" "$OUT"/shard*.log | grep -oE "'no_hint::[a-z0-9_]+': [0-9]+" | grep -oE "[0-9]+$" || true) | awk '{n+=$1} END{print n+0}')
  [ "$NHINTSKIP" -gt 0 ] && echo "  [skips] $NHINTSKIP of $((NROLL * NARMS)) rollout-arms have no hint (arm does not cover that task) -- reconcile against (pool_tasks - arm_tasks) x trials"
  echo "--- merging $TAG ---"
  PYTHONPATH="$REPO_ROOT" $PY -m opsd.alfworld.analyze_pi_positions \
    --merge --out-dir "$OUT" --benchmark tau2 > "$OUT/merge.log" 2>&1
  tail -3 "$OUT/merge.log"
done
echo "=== ALL DONE -> $SCORE_ROOT ==="

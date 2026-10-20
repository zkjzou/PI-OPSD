# scripts/analysis

The scripts behind the PI pipeline, by stage (the stage table is in the top-level README). Every script
documents its inputs, outputs and environment variables in its header. Inputs default to the fetched dataset
under `$OPSD_DATA_ROOT`; outputs go to `./outputs/` unless a flag or variable says otherwise.

Shared: `pi_src_lib.py` (cell / label loading, D and Δd⁻_err estimators, PI-id mapping), `alf_study_common.py`
(ALFWorld study paths), `_tau2_env.sh` (τ² paths and helpers), `openai_batch.py` (OpenAI batch client for the
analyzer and the judge), `merge_fsdp_ckpt_to_hf.py` (FSDP checkpoint to a Hugging Face model).

## 1-2. Reference trajectories and PI construction

| Script | What it does |
|---|---|
| `build_appworld_pi_train90.sh` | AppWorld reference trajectories from a frontier model (train-90), with rescue rounds |
| `run_tau2_pi_gen.sh` | τ² reference trajectories from a frontier model, then the three trajectory hint formats |
| `build_tau2_retail_pi.py`, `build_tau2_pi_pool.py` | merge the two authors' τ² hints into the pool-scoped files and the matching training manifest |
| `rollouts_to_pi_refs.py` | reference trajectories from any won rollouts (e.g. a student's own) |
| `alfworld_rebuild_hints.py` | re-render ALFWorld trajectory hints (uncapped) from the reference trajectories |
| `alfworld_cap_many_24k.py`, `alfworld_cap_withthink_24k.py` | the 24k-capped ALFWorld trajectory renders used for training |
| `strip_alfworld_hint_preamble.py`, `sanity_filter_hints.py` | clean a hint file (leading `Task:` preamble; empty or malformed hints) |
| `build_appworld_self_hints.py` | AppWorld trajectory hints from won transcripts (reference or self) |
| `gen_alfworld_analyzer_hints.sh`, `regen_appworld_analyzer_hints.sh`, `gen_tau2_analyzer_hints.sh` | Skill / Summary hints written by an analyzer (GLM-5.2, Qwen3.7-Plus, or the student) |
| `gen_tau2_plain_summarizer.sh`, `measure_tau2_hint_leakage.py` | τ² Summary hints and their answer-leakage check |
| `gen_hints_local_models.py`, `serve_qwen3_4b_instruct_pool.sh` | analyzer calls against locally served students (self hints) |
| `fill_self_hints.sh`, `merge_self_glm_hints.py` | ALFWorld self hints, filled with GLM-5.2 hints where the student wrote none |
| `build_tau2_retailbase_self_hints.sh` | τ² self hints on the retail evaluation tasks |
| `build_checklist_hints.py`, `run_pi_checklist_hints.sh`, `run_tau2_checklist_hints.sh` | Checklist hints (a win contrasted with the student's failures) |

## 3-6. PI metrics (U, D, Δd⁻_err)

| Script | What it does |
|---|---|
| `pi_error_step_label.py`, `pi_error_step_label_tau2.py` | error-step labels from an LLM judge (ALFWorld / AppWorld; τ²) |
| `run_pi_positions_alf_new8_local.sh`, `run_appworld_pi_positions.sh`, `run_tau2_pi_positions.sh` | teacher-forced scoring of PIs on a no-hint corpus → `<cell>/rows.parquet` |
| `alf_pi_divergence_block.py` | ALFWorld D |
| `pi_error_step_balanced.py` | Δd⁻_err (ALFWorld; natural and balanced designs on every benchmark) |
| `build_appworld_traindev_pi_table.py`, `pi_error_step_appworld_traindev.py` | AppWorld D + U, and Δd⁻_err |
| `gen_tau2_matched132_k2_table.py` (+ helpers in `gen_tau2_median200_table.py`) | τ² U, D and Δd⁻_err on the matched-132 tasks (writes a LaTeX table; the only τ² metric script) |
| `run_alfworld_pi_utility.sh` | ALFWorld hinted utility rollouts (aggregate with `python -m opsd.alfworld.aggregate_utility_delta`) |
| `appworld_utility_both_k.py`, `build_appworld_nohint_baselines_from_passk.sh`, `write_appworld_baseline_provenance.py` | AppWorld U from `scripts/appworld/pi_utility.sh` output against the `scripts/appworld/passk.sh` no-hint corpus |

The no-hint corpora (stage 3) come from `opsd/alfworld/run_alfworld_generate_local.sh`, `scripts/appworld/passk.sh`
and `opsd/tau2_bench/run_tau2_passk.sh`; τ² utility rollouts from `opsd/tau2_bench/run_tau2_pi_utility.sh`.
Correlations with OPSD performance take `--targets <csv>` (`bench,student,pi,pass1` from your runs).

## ALFWorld-Passcode

| Script | What it does |
|---|---|
| `build_alfworld_passcode_hints.py` | the four Passcode PIs (key, raw+key, raw+wrong key, wrong raw) |
| `run_alf_passcode_utility.sh` | Passcode utility rollouts |
| `alf_passcode_report.py` | Passcode utility and metric report |

## Per-task PI selection

| Script | What it does |
|---|---|
| `run_alf_pi_multimodel.sh`, `run_alf_pi_topup.sh`, `make_train_chunks.py` | multi-author ALFWorld reference trajectories (the candidate pool) |
| `pi_multimodel_offline_prepare.py`, `pi_rem1400_prepare.py`, `pi_flash8_prepare.py`, `pi_self16_prepare.py`, `pi_self4_prepare.py`, `pi_glm8_build_hints.py` | build candidate hints and scoring chunks |
| `run_pi_chunk_score.sh`, `run_pi_chunks_free_gpus.sh`, `run_pi_author_offline200.sh`, `run_flash8_score_build.sh` | score the candidate chunks |
| `pi_glm8_report.py`, `pi_self4_report.py`, `pi_multimodel_offline_report.py`, `pi_errstep_percentile.py` | per-candidate d / Δd⁻_err estimators used by the builders |
| `pi_mix_select_build.py`, `pi_self16_select_build.py` | pick one hint per task (random / top-3 by d / top-3 by Δd⁻_err) → the selection arms' hint files |
| `build_selfwrong_hints.py` | hints from the student's own failed rollouts (the `selfwrong` arm) |

## Corrupted-PI controls

| Script | What it does |
|---|---|
| `build_pi_null_tiers.py` | mismatched-trajectory null: another task's hint, tiers n1–n3 (n3 = the `null_task` arm) |
| `build_pi_crossdomain_hints.py` | cross-domain null: length-matched τ² / AppWorld trajectories (pooled = the `null_xdomain` arm) |
| `build_selfwrong_hints.py` | the student's own failed rollout as the PI (the `selfwrong` arm) |

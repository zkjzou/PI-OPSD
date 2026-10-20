# Code adapted from SkyRL

SkyRL (Apache-2.0) is an external dependency of this repository, used unmodified and import-only: a
checkout of NovaSky-AI/SkyRL at
[`ecdaa9460e28`](https://github.com/NovaSky-AI/SkyRL/tree/ecdaa9460e281774638a9170d6724456f46c8356)
next to the repository (`../SkyRL-ecdaa946`), installed through `pyproject.toml`'s `[tool.uv.sources]`
([README, Quick start](README.md#-quick-start)). No SkyRL file is modified.
OPSD extends it from `opsd/skyrl_ext/` through SkyRL's extension points: subclasses,
mixins and the hooks of `BasePPOExp` ([skyrl_ext/README.md](opsd/skyrl_ext/README.md)). Where an extension
cannot simply call the upstream code, it contains a copy or close adaptation of it. Every such file starts
with one header line per upstream source:

```text
# Adapted from SkyRL (Apache-2.0), <upstream path> @ ecdaa946; modified for OPSD: <what changed>
```

This page lists those files. It is generated from the headers: `grep -rn '^# Adapted from SkyRL' opsd`
(run from the repository root) must list exactly the files below. Other `skyrl_ext` modules
(`config.py`, `exp.py`, `worker_selection.py`) subclass or call SkyRL without copying its code, so they
carry no header.

| Our file | Upstream source @ `ecdaa946` | What changed |
|---|---|---|
| [`opsd/skyrl_ext/evaluate.py`](opsd/skyrl_ext/evaluate.py) | [`skyrl/train/evaluate.py`](https://github.com/NovaSky-AI/SkyRL/blob/ecdaa9460e281774638a9170d6724456f46c8356/skyrl/train/evaluate.py) (`evaluate`, copied); [`skyrl/train/utils/trainer_utils.py`](https://github.com/NovaSky-AI/SkyRL/blob/ecdaa9460e281774638a9170d6724456f46c8356/skyrl/train/utils/trainer_utils.py) (`calculate_per_dataset_metrics`, `dump_per_dataset_eval_results`, copied); [`skyrl/train/generators/utils.py`](https://github.com/NovaSky-AI/SkyRL/blob/ecdaa9460e281774638a9170d6724456f46c8356/skyrl/train/generators/utils.py) (`get_metrics_from_generator_output`, wrapped); [`skyrl/train/trainer.py`](https://github.com/NovaSky-AI/SkyRL/blob/ecdaa9460e281774638a9170d6724456f46c8356/skyrl/train/trainer.py) (`RayPPOTrainer.eval`, replaced by `OPSDEvalMixin.eval`) | `evaluate`: per-data-source env classes, `eval/all/pass_at_1`, `eval/all/*` rollout metrics re-pooled over every eval batch, and the `SKYRL_EVAL_COMPACT` / `SKYRL_EVAL_NAMESPACE` post-processors. Per-dataset metrics: per-data-source `pass_at_1`, `generate/*` and `environment/*` metrics, and each rollout's `env_metrics` in every dumped eval row. Metrics helper: pass@n counts `env_metrics["success"]` when every rollout reports it, else `reward > 0`, plus `pass_at_1`. `RayPPOTrainer.eval` is routed to this `evaluate()`; step-wise eval stays upstream's. |
| [`opsd/skyrl_ext/main_generate.py`](opsd/skyrl_ext/main_generate.py) | [`skyrl/train/entrypoints/main_generate.py`](https://github.com/NovaSky-AI/SkyRL/blob/ecdaa9460e281774638a9170d6724456f46c8356/skyrl/train/entrypoints/main_generate.py) (`EvalOnlyEntrypoint.run`, the `eval_entrypoint` task and `main`) | Parses with `OPSDTrainConfig`, registers the OPSD envs in the eval task (`OPSDExpMixin`), runs `opsd.skyrl_ext.evaluate`'s `evaluate` / `evaluate_step_wise`, logs through `OPSDTracking`, and pins the eval task to the driver's node. |
| [`opsd/skyrl_ext/policy_worker.py`](opsd/skyrl_ext/policy_worker.py) | [`skyrl/backends/skyrl_train/workers/worker.py`](https://github.com/NovaSky-AI/SkyRL/blob/ecdaa9460e281774638a9170d6724456f46c8356/skyrl/backends/skyrl_train/workers/worker.py) (`PolicyWorkerBase._forward_backward_micro`) | The RL branch of `_forward_backward_micro` is copied into `OPSDFSDPPolicyWorkerBase` and adds the β teacher-KL and SDAR loss terms (`opsd/algorithms/distill_loss.py`) inside marked `# >>> OPSD` blocks; `forward_backward` splits the teacher log-probs the way the batch is split into micro-batches. The SFT branch is not copied (it falls through to upstream). |
| [`opsd/skyrl_ext/ray_runtime.py`](opsd/skyrl_ext/ray_runtime.py) | [`skyrl/train/utils/utils.py`](https://github.com/NovaSky-AI/SkyRL/blob/ecdaa9460e281774638a9170d6724456f46c8356/skyrl/train/utils/utils.py) (`initialize_ray`, copied; `prepare_runtime_environment`, wrapped) | `initialize_ray` is upstream's body, calling a `prepare_runtime_environment` that is upstream's plus the forwarding of the project's env-var knobs (`OPSD_*`, `ALFWORLD_*`, `APPWORLD_*`, `TAU2_*`, `SKYRL_*`, `PYTHONHASHSEED`, never credential-like names), and passing `SKYRL_RAY_NUM_CPUS` to `ray.init` as `num_cpus`. |
| [`opsd/skyrl_ext/tracking.py`](opsd/skyrl_ext/tracking.py) | [`skyrl/train/entrypoints/main_base.py`](https://github.com/NovaSky-AI/SkyRL/blob/ecdaa9460e281774638a9170d6724456f46c8356/skyrl/train/entrypoints/main_base.py) (`BasePPOExp.get_tracker`) | `get_tracker` builds `OPSDTracking`, a `Tracking` subclass whose `log()` uses `SKYRL_LOG_STEP_OVERRIDE` as the step when it is set to an integer. |
| [`opsd/skyrl_ext/trainer.py`](opsd/skyrl_ext/trainer.py) | [`skyrl/train/trainer.py`](https://github.com/NovaSky-AI/SkyRL/blob/ecdaa9460e281774638a9170d6724456f46c8356/skyrl/train/trainer.py) (`RayPPOTrainer.compute_advantages_and_returns`, `load_checkpoints`) | The `grpo/*` gradient-signal metrics re-derive the rows `compute_advantages_and_returns` scores (padding dropped; on the step-wise path only each trajectory's last step). `load_checkpoints` drains a resumed epoch whose restored dataloader iterator is already exhausted. Both overrides call `super()` first. |

## How the copies are kept honest

| Our file | Upstream-drift check | Behaviour check (regression goldens) |
|---|---|---|
| `evaluate.py` | [`test_eval_upstream_drift.py`](opsd/skyrl_ext/tests/test_eval_upstream_drift.py): sha256 of the source of every upstream function it copies, wraps or replaces | [`test_eval_fork_parity.py`](opsd/skyrl_ext/tests/test_eval_fork_parity.py), [`test_generator_metrics.py`](opsd/skyrl_ext/tests/test_generator_metrics.py) against [`fork_parity/eval`](opsd/tests/fixtures/fork_parity/eval/README.md) |
| `main_generate.py` | [`test_eval_upstream_drift.py`](opsd/skyrl_ext/tests/test_eval_upstream_drift.py): sha256 of upstream `EvalOnlyEntrypoint.run`, `eval_entrypoint` and `main` | none of its own; it runs the same `evaluate()` as the eval goldens |
| `policy_worker.py` | [`test_policy_worker_drift.py`](opsd/skyrl_ext/tests/test_policy_worker_drift.py): sha256 of upstream `_forward_backward_micro`, and the copy minus its `# >>> OPSD` blocks equals upstream's RL branch line for line | [`test_policy_worker_parity.py`](opsd/skyrl_ext/tests/test_policy_worker_parity.py) against [`fork_parity/worker`](opsd/tests/fixtures/fork_parity/worker/README.md) |
| `ray_runtime.py` | [`test_ray_runtime.py`](opsd/skyrl_ext/tests/test_ray_runtime.py): sha256 of upstream `initialize_ray`, and identical `ray.init` kwargs and side effects without the OPSD knobs | [`test_ray_runtime_fork_parity.py`](opsd/skyrl_ext/tests/test_ray_runtime_fork_parity.py) against [`fork_parity/runtime`](opsd/tests/fixtures/fork_parity/runtime/README.md) |
| `tracking.py` | [`test_trainer_tracking_upstream_drift.py`](opsd/skyrl_ext/tests/test_trainer_tracking_upstream_drift.py): sha256 of upstream `BasePPOExp.get_tracker` | [`test_tracking_fork_parity.py`](opsd/skyrl_ext/tests/test_tracking_fork_parity.py) against `fork_parity/runtime/tracking.json` |
| `trainer.py` | [`test_trainer_tracking_upstream_drift.py`](opsd/skyrl_ext/tests/test_trainer_tracking_upstream_drift.py): sha256 of upstream `RayPPOTrainer.compute_advantages_and_returns`, `load_checkpoints` and `train` (whose epoch arithmetic the resume drain mirrors); the parity tests run them through `super()` | [`test_trainer_grpo_signal.py`](opsd/skyrl_ext/tests/test_trainer_grpo_signal.py), [`test_trainer_resume.py`](opsd/skyrl_ext/tests/test_trainer_resume.py) against [`fork_parity/trainer`](opsd/tests/fixtures/fork_parity/trainer/README.md) |

Outside `skyrl_ext`, the OPSD self-distillation trainer (`SelfDistillationTrainer` in
[`opsd/trainer/main_opsd_alfworld.py`](opsd/trainer/main_opsd_alfworld.py), which the AppWorld and τ²
OPSD trainers inherit) replaces two `RayPPOTrainer` methods without calling `super()`:
`fwd_logprobs_values_reward` (upstream's body plus the hinted-teacher forward; it leaves out the critic
forward and the vision-input keys, which no recipe uses, and upstream's
`policy/rollout_train_logprobs_abs_diff_*` metrics) and `apply_reward_kl_penalty`. Both are
pinned by sha256 in [`test_trainer_tracking_upstream_drift.py`](opsd/skyrl_ext/tests/test_trainer_tracking_upstream_drift.py).
Its other `RayPPOTrainer` overrides (`convert_to_training_input`, `train_critic_and_policy`, `eval`) call
`super()`.

The regression goldens under `opsd/tests/fixtures/fork_parity/` are reference outputs recorded against SkyRL
at the same commit. Each fixture README says what was recorded and lists the intended differences, which the
tests assert explicitly.

## Bumping the SkyRL pin

The pin is copied into several files. The setup script and the CPU tests refuse most partial bumps loudly, but
one mistake is silent: `uv.lock` records skyrl's own dependency list, and `uv sync --frozen` (setup and CI)
never checks it, so a new SkyRL commit with a stale lock installs the old dependency set without an error.
Below, `NEW` is the new 40-hex commit and `<short>` its first 8 characters; run everything from the repository
root.

1. **A checkout at the new commit, in a new sibling directory**, so that the directory name stays true:
   `git clone https://github.com/NovaSky-AI/SkyRL ../SkyRL-<short> && git -C ../SkyRL-<short> checkout NEW`.
   (Checking out `NEW` inside `../SkyRL-ecdaa946` also works, with a misleading name; the steps below are the
   same, including the re-lock.)
2. **Every copy of the pin:**
   - `SKYRL_COMMIT` in `opsd/_repo.py` (`setup_train_env.sh` refuses any other checkout, and provenance
     records `matches_pin` against it);
   - both `[tool.uv.sources]` paths in `pyproject.toml` (`skyrl` → `../SkyRL-<short>`, `skyrl-gym` →
     `../SkyRL-<short>/skyrl-gym`) and the clone command in the comment above them;
   - `UPSTREAM_BASE` and the `../SkyRL-…` hints in `scripts/release_guard/check_release.py`, and `PIN` in
     `scripts/release_guard/tests/test_release_guard.py`;
   - the note on the sibling checkout in `.gitignore`.
3. **The mirrored `[tool.uv]` configuration.** uv applies only the root project's `[tool.uv.*]`, so
   `pyproject.toml` copies SkyRL's (see the comment above `[tool.uv]` in `pyproject.toml`). Re-copy it from the new
   commit's `pyproject.toml` (required-environments, conflicts, no-build-isolation-package,
   override-dependencies, extra-build-dependencies / -variables, the indexes and sources), keeping the
   differences the comment above `[tool.uv]` lists, and check the hand-written `constraint-dependencies` pins
   above the generated block against the new SkyRL `uv.lock`.
4. **Re-lock, always** (also after an in-place checkout): `uv lock`, then `uv lock --check`.
5. **The lock-drift check**:
   `python scripts/release_guard/check_lock_drift.py --ours uv.lock --skyrl ../SkyRL-<short>/uv.lock`
   ([`check_lock_drift.py`](scripts/release_guard/check_lock_drift.py), stdlib only, Python ≥ 3.11). It
   passes when every package of `uv.lock` is in SkyRL's lock at the same version, source and files, nothing is
   added beyond this project and the `tau2` extra's packages, and every `constraint-dependencies` pin is a
   version of SkyRL's lock. For each drifted package, add SkyRL's version to the generated block between
   `# >>> SkyRL uv.lock versions …` and `# <<< SkyRL uv.lock versions <<<` (`--suggest` prints the lines;
   drop pins it reports as no longer in SkyRL's lock), then repeat steps 4 and 5 until it reports no drift.
6. **The environment**: `bash scripts/setup/setup_train_env.sh --dev` (it keeps the other extras `.venv`
   has) verifies the new checkout and syncs the new lock.
7. **The adapted copies.** A commit that changes a pinned upstream function fails its drift test
   (`.venv/bin/python -m pytest opsd/skyrl_ext/tests opsd/tests/test_adapted_from_skyrl_inventory.py`).
   Re-sync the copy with the new upstream source, keep the changes its header lists, re-run the parity tests,
   then update the hash, the `@ <commit>` in the header and the links on this page. A failing pin of
   `RayPPOTrainer.fwd_logprobs_values_reward` or `apply_reward_kl_penalty` means re-checking
   `SelfDistillationTrainer`'s replacement against the new upstream method instead.
8. **Every other mention**: `git grep -n ecdaa946` lists the rest. Update the README install block ("Quick start"),
   `NOTICE` and the code comments that cite upstream line numbers (re-check those lines).
   The regression fixtures under `opsd/tests/fixtures/fork_parity/` record the commit they were captured at:
   leave them.
9. **The full CPU suite and the self-test**, not only `opsd/skyrl_ext/tests`: `.venv/bin/python -m pytest -q`
   and `bash scripts/common/selftest.sh`. `test_adapted_from_skyrl_inventory.py` checks that this page, the
   headers and the installed checkout agree with `SKYRL_COMMIT`; `test_release_guard.py` checks
   `UPSTREAM_BASE`; `test_check_lock_drift.py` runs the lock-drift check against the `[tool.uv.sources]`
   checkout.

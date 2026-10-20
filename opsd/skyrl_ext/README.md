# `opsd.skyrl_ext`: OPSD's extensions of SkyRL

SkyRL (an external checkout pinned at `ecdaa946`, installed import-only through the root `pyproject.toml`) runs unmodified. Where OPSD needs behaviour
SkyRL does not provide, this package supplies it through SkyRL's own extension points: subclasses of its
config dataclasses, of `RayPPOTrainer` and of the FSDP policy worker, the `BasePPOExp` hooks, and the
skyrl_gym registry (the envs themselves live in [`opsd.envs`](../envs/)). Importing the package has no side
effects. Outside this package, the OPSD self-distillation trainer (`opsd/trainer/main_opsd_alfworld.py`)
overrides `RayPPOTrainer` methods of its own; the two it replaces without `super()` are pinned below
(`trainer.py`). Files that adapt upstream code carry an `Adapted from SkyRL` header and are listed, with their
drift checks, in [ADAPTED_FROM_SKYRL.md](../../ADAPTED_FROM_SKYRL.md). The regression goldens the
tests compare against (reference outputs recorded against the same SkyRL commit) live in [`opsd/tests/fixtures/fork_parity/`](../tests/fixtures/fork_parity/), one folder per
area, each with a README of what was captured and which differences are intended. Run the tests with
`.venv/bin/python -m pytest opsd/skyrl_ext/tests` from the repository root (CPU only; no Ray cluster, GPU or network).

**`config.py`: the config tree.** `OPSDTrainConfig` is SkyRL's `SkyRLTrainConfig` with the fields the OPSD
recipes set and SkyRL lacks: `trainer.algorithm.overlong_buffer` (`OverlongBufferConfig`, the DAPO soft
overlong penalty) and `environment.skyrl_gym.alfworld` / `.appworld` (the env config dataclasses from
`opsd.envs`). It extends the tree the way SkyRL asks, with a subclass at every level on the path to a new
field (`OPSDTrainerConfig` → `OPSDAlgorithmConfig`, `OPSDEnvironmentConfig` → `OPSDSkyRLGymConfig`). Because
`SkyRLTrainConfig.from_cli_overrides` is a classmethod that builds `cls` level by level from the declared
field types, `OPSDTrainConfig.from_cli_overrides(argv)` parses and validates exactly like upstream
otherwise, and still rejects unknown keys. Each entry module exposes the class it parses with as
`CONFIG_CLS` (τ²'s `Tau2SkyRLConfig` is a further subclass). The module stays import-light and has no
`from __future__ import annotations`, because `cfg.trainer` is unpickled in every Ray actor and nested
configs are resolved from real annotations. Verified by [`tests/test_config.py`](tests/test_config.py)
(upstream's config is untouched, field order and defaults, node types, no string annotations, pickling by
reference, a light import in a fresh process) and by the recipe tests in `opsd/recipes/<bench>/tests/`,
which parse every arm's dry-run argv with its entry module's `CONFIG_CLS` and compare it with the goldens.

**`exp.py`: the experiment mixin.** `OPSDExpMixin` goes first in the bases of every OPSD experiment
(`class MyExp(OPSDExpMixin, BasePPOExp)`) and hooks SkyRL's run setup only through `BasePPOExp`'s own
methods. `__init__` refuses a config that is not an `OPSDTrainConfig` and calls
`opsd.envs.registry.register_opsd_envs()`; the Exp is built inside the entrypoint's Ray task, which is
exactly the process where the generator builds envs and `skyrl_gym.metrics.aggregate_for_environment` looks
them up, while the policy, reference and vLLM workers never touch the registry. `get_tracker` returns an
`OPSDTracking` (below). Every `get_trainer` a subclass defines is wrapped so that it raises `TypeError`
unless it returns an `OPSDRayPPOTrainer`, before `build_models` places the policy / ref / critic actors (the
tracker and the vLLM inference engines already exist by then). It lives in an importable module
because the entry modules run as `python -m` and Ray pickles their own classes by value. It copies no
upstream code. Verified by [`tests/test_exp.py`](tests/test_exp.py) (the config-type check, registration
before `BasePPOExp.__init__`, the tracker, and the trainer guard, including after a by-value cloudpickle
rebuild of the Exp class) and by [`opsd/tests/test_entrypoint_guards.py`](../tests/test_entrypoint_guards.py),
which scans every OPSD module statically (only `OPSDRayPPOTrainer` subclasses `RayPPOTrainer` directly,
every Exp lists `OPSDExpMixin` first, no upstream `initialize_ray`, every parse goes through `CONFIG_CLS`)
and checks every entry module and Exp dynamically (`CONFIG_CLS`, `initialize_ray`, the trainer each
`get_trainer` returns). Related checks: [`tests/test_eval_trainer_wiring.py`](tests/test_eval_trainer_wiring.py)
builds every release trainer and checks that its `eval()` reaches the OPSD evaluate, and
[`opsd/envs/tests/test_registry.py`](../envs/tests/test_registry.py) covers `register_opsd_envs()`
(idempotence, and the error when an id is already registered with another entry point).

**`trainer.py`: the trainer base.** `OPSDRayPPOTrainer(OPSDEvalMixin, OPSDWorkerMixin, RayPPOTrainer)` is the
one `RayPPOTrainer` subclass every OPSD trainer derives from; SkyRL's `train()` runs unmodified. Its
overrides call `super()` first: `compute_advantages_and_returns` adds the `grpo/*` gradient-signal
breakdown (`opsd.algorithms.grpo_signal`) for exactly the rows the advantage estimator consumed, and
`load_checkpoints` drains a restored dataloader iterator that is already exhausted (a checkpoint saved
right after an epoch's last batch), so the first resumed epoch is a full one instead of an empty pass.
`eval` and `build_models` come from the two mixins below. Verified against the goldens in
[`fork_parity/trainer`](../tests/fixtures/fork_parity/trainer/README.md):
[`tests/test_trainer_grpo_signal.py`](tests/test_trainer_grpo_signal.py) requires bit-identical `grpo/*`
metrics and advantages on the same batches (the other metrics within 1e-6), and
[`tests/test_trainer_resume.py`](tests/test_trainer_resume.py) drives upstream `train()` without models on a real torchdata `StatefulDataLoader` and requires the same
batches in the same order for fresh runs and for resumes from every kind of checkpoint, with the intended
differences asserted explicitly. [`tests/test_trainer_tracking_upstream_drift.py`](tests/test_trainer_tracking_upstream_drift.py)
pins upstream `compute_advantages_and_returns`, `load_checkpoints` and `train` (whose epoch arithmetic the
drain mirrors) by sha256, together with the two `RayPPOTrainer` methods the OPSD self-distillation trainer
(`opsd/trainer/main_opsd_alfworld.py`, `SelfDistillationTrainer`, which the AppWorld and τ² OPSD
trainers inherit) replaces without calling `super()`: `fwd_logprobs_values_reward` and
`apply_reward_kl_penalty`. Its other overrides (`convert_to_training_input`, `train_critic_and_policy`,
`eval`) call `super()`.

**`evaluate.py`: evaluation metrics.** `OPSDEvalMixin.eval` replaces `RayPPOTrainer.eval` and runs this
module's `evaluate()`, a copy of SkyRL's with the metrics the paper's tables and checkpoint pickers read:
pass@n from the env's own `env_metrics["success"]` when every rollout reports it (else `reward > 0`),
per-data-source `pass_at_1`, `generate/*` and `environment/*`, `eval/all/environment/*` re-pooled over all
eval batches (upstream sums the per-batch means), each rollout's `env_metrics` in the eval dumps, and the
`SKYRL_EVAL_COMPACT` / `SKYRL_EVAL_NAMESPACE` post-processors. Step-wise eval is delegated to upstream via
`super()`, with only the post-processors applied. Any OPSD code path that runs an eval imports `evaluate` /
`evaluate_step_wise` from here, never from `skyrl.train.evaluate`. Verified by
[`tests/test_eval_upstream_drift.py`](tests/test_eval_upstream_drift.py) (sha256 pins of every upstream
function it copies, wraps or replaces), [`tests/test_eval_fork_parity.py`](tests/test_eval_fork_parity.py)
and [`tests/test_generator_metrics.py`](tests/test_generator_metrics.py) against the 24 cases of
[`fork_parity/eval`](../tests/fixtures/fork_parity/eval/README.md) (returned metrics and every dumped
file byte for byte), and the mixin and wiring tests.

**`worker_selection.py`: choosing the policy worker.** SkyRL builds its workers inside Ray actors from the
classes `BasePPOExp._setup_trainer` passes to `RayPPOTrainer.build_models(PolicyWorker, CriticWorker,
RefWorker)`; that argument is the seam. `OPSDWorkerMixin.build_models` parses the loss knobs once on the
driver (`LossKnobs.from_env`, so a typo fails at startup), swaps in `OPSDFSDPPolicyWorker` only when
`OPSD_DISTILL_LOSS_COEF` or `OPSD_SDAR_LAMBDA` is non-zero (otherwise every actor class is upstream's),
refuses those knobs under `strategy=megatron` (the Megatron worker never sees the teacher tensor), clamps
`total_training_steps` to at least 1 around `build_models` on Megatron eval-only runs (Megatron's scheduler
asserts `lr_decay_steps > 0`), and pushes the knobs to the policy actors through SkyRL's `pass_through`
dispatch. It copies no upstream code. Verified by [`tests/test_worker_selection.py`](tests/test_worker_selection.py)
with a recording stand-in for `RayPPOTrainer.build_models`: which actor class reaches SkyRL, the Megatron
guard, the clamp and its restore on error, and the knob push.

**`policy_worker.py`: β and SDAR in the loss.** `OPSDFSDPPolicyWorkerBase` subclasses SkyRL's
`FSDPPolicyWorkerBase`. Upstream offers no hook inside `PolicyWorkerBase._forward_backward_micro` between
the KL term and `strategy.backward`, so the RL branch of that method is copied with the β teacher-KL and
SDAR terms (maths in `opsd.algorithms.distill_loss`) added in marked `# >>> OPSD` blocks. `forward_backward`
splits the trainer's `distill_log_probs` tensor exactly as the batch is chunked into micro-batches, since
SkyRL's `BatchIterator` drops keys it does not know. Verified by
[`tests/test_policy_worker_drift.py`](tests/test_policy_worker_drift.py) (sha256 pin of upstream's method;
the copy minus its OPSD blocks equals upstream's RL branch line for line) and
[`tests/test_policy_worker_parity.py`](tests/test_policy_worker_parity.py), which runs the worker on a CPU
stub model through the 32 cases of [`fork_parity/worker`](../tests/fixtures/fork_parity/worker/README.md)
(every loss passed to `strategy.backward`, the status dicts, the reduced metrics and the parameter
gradient; it matched bit for bit on the capture machine).

**`ray_runtime.py`: Ray driver setup.** The OPSD entrypoints call this module's `initialize_ray` instead of
SkyRL's. It is upstream's body, calling a `prepare_runtime_environment` that is upstream's plus the
forwarding of the project's knobs (`OPSD_*`, `ALFWORLD_*`, `APPWORLD_*`, `TAU2_*`, `SKYRL_*`,
`PYTHONHASHSEED` and a few W&B run-identity names; credential-like names are never forwarded) into the
job's runtime env, so tasks and actors on an existing Ray cluster see them; it also passes
`SKYRL_RAY_NUM_CPUS` to `ray.init` as `num_cpus`. Everything runs on the driver, and the module imports no
torch at import time, so the τ² launcher's key probe can import `collect_forwarded_env_vars` cheaply.
Verified by [`tests/test_ray_runtime.py`](tests/test_ray_runtime.py) (sha256 pin of upstream
`initialize_ray`; identical `ray.init` kwargs and side effects without the OPSD knobs),
[`tests/test_ray_runtime_env_forwarding.py`](tests/test_ray_runtime_env_forwarding.py), and
[`tests/test_ray_runtime_fork_parity.py`](tests/test_ray_runtime_fork_parity.py) against
[`fork_parity/runtime`](../tests/fixtures/fork_parity/runtime/README.md) (forwarded env dicts, warnings and
`ray.init` kwargs for 18 to 22 environ fixtures each).

**`main_generate.py`: generation-only evaluation.** `python -m opsd.skyrl_ext.main_generate <overrides>` is
SkyRL's `skyrl.train.entrypoints.main_generate` flow for OPSD configs, which upstream's cannot run (it parses
with `SkyRLTrainConfig`, registers no OPSD env and imports upstream `evaluate`). `main()` parses with
`CONFIG_CLS` (`OPSDTrainConfig`) and starts Ray through `ray_runtime.initialize_ray`; the eval task, pinned
to the driver's node where the launchers start the env servers, builds `OPSDEvalOnlyEntrypoint`
(`OPSDExpMixin` + upstream `EvalOnlyEntrypoint`), whose `run` is upstream's calling this package's
`evaluate` / `evaluate_step_wise` and logging through `OPSDTracking`. Its user is
`opsd/alfworld/run_alfworld_generate_local.sh`. It carries an `Adapted from SkyRL` header, and
[`tests/test_eval_upstream_drift.py`](tests/test_eval_upstream_drift.py) pins the upstream `run`,
`eval_entrypoint` and `main` it follows by sha256. [`tests/test_main_generate.py`](tests/test_main_generate.py)
checks the wiring with Ray and the model stubbed: `main()` parses with `CONFIG_CLS`, validates, starts Ray
through `ray_runtime` and pins the eval task to the driver's node, and `run` scores with this package's
`evaluate` / `evaluate_step_wise` and logs the result once. It has no regression golden of its own, but runs
the same `evaluate()` the eval goldens check.

**`tracking.py`: the W&B step override.** `OPSDTracking` is SkyRL's `Tracking` with a `log()` that uses
`SKYRL_LOG_STEP_OVERRIDE` as the step when it is set to an integer, so an evaluation-only process can log
into a resumed W&B run past its training tail (W&B drops steps behind the run's step pointer).
`get_tracker` has `BasePPOExp.get_tracker`'s signature and builds an `OPSDTracking` from the same config
fields; `OPSDExpMixin` binds it, and an Exp that needs its own tracker subclasses `OPSDTracking`. Upstream
`BasePPOExp.get_tracker` is pinned by sha256 in
[`tests/test_trainer_tracking_upstream_drift.py`](tests/test_trainer_tracking_upstream_drift.py). Verified
by [`tests/test_tracking.py`](tests/test_tracking.py) and
[`tests/test_tracking_fork_parity.py`](tests/test_tracking_fork_parity.py), which replays 13 override values
through a fake W&B backend and a fake non-W&B backend against `fork_parity/runtime/tracking.json`.

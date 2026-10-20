# Regression goldens: Ray runtime env + tracking step override

These goldens are reference outputs of the reference implementation (OPSD's runtime changes made directly in
`skyrl/train/utils/utils.py` and `skyrl/train/utils/tracking.py` of SkyRL `ecdaa946`).
The tests check that the code in `opsd.skyrl_ext.ray_runtime` and `opsd.skyrl_ext.tracking`, running
on top of unmodified SkyRL, behaves the same way:

- `opsd/skyrl_ext/tests/test_ray_runtime_fork_parity.py`
- `opsd/skyrl_ext/tests/test_tracking_fork_parity.py`

Every input is synthetic. The goldens contain no host paths, usernames, secrets or real environment values.
The temporary log directory is normalised to `<LOG_PATH>`.

## Files

| File | What was recorded (reference code) |
|---|---|
| `forwarding.json` | The forwarding constants. `collect_forwarded_env_vars(environ)` (ordered result plus the loguru warnings) for 18 environ fixtures. `is_secret_env_name(name)` for every name in those fixtures, plus edge cases. |
| `prepare_runtime_environment.json` | The `prepare_runtime_environment(SkyRLTrainConfig())` dict (in insertion order) and all INFO+ loguru records, for the same environs plus 2 cases with different module flags or config: LD_LIBRARY_PATH/PYTHONPATH export on, and megatron with no peer access and the legacy inference layer. |
| `initialize_ray.json` | For each of 22 cases (`SKYRL_RAY_NUM_CPUS` unset/valid/padded/`+4`/`1_6`/empty/0/-1/non-int/float, `RAY_ADDRESS` local/external/unset, a stale `SKYRL_LOG_FILE`, `SKYRL_DUMP_INFRA_LOG_TO_STDOUT` on/off), `initialize_ray(cfg)` records: the kwargs passed to `ray.init`, the error raised, `os.environ` changes, whether the log dir was created, how many times `sync_registries` was called, and the loguru records. |
| `tracking.json` | What a fake `wandb` backend and a fake non-wandb backend receive from `Tracking.log`, for 13 `SKYRL_LOG_STEP_OVERRIDE` values (unset, empty, whitespace, ints, `+5`, `1_000`, `-3`, non-ints) over 4 calls, including `step=None` and a call with `commit` omitted. |

## How it was captured

The capture script is not part of this repository. It ran CPU-only (`CUDA_VISIBLE_DEVICES=`) against the
reference `skyrl`. Nothing touched a cluster. These were stubbed:

- the `ray` module global of `skyrl.train.utils.utils` (`init` records its kwargs, `is_initialized` returns False)
- `peer_access_supported`
- `ppo_utils.sync_registries`
- `datetime.now`, fixed at 2026-01-02 03:04:05

The module flags `_SKYRL_USE_NEW_INFERENCE`, `SKYRL_LD_LIBRARY_PATH_EXPORT`, `SKYRL_PYTHONPATH_EXPORT` and
`SKYRL_DUMP_INFRA_LOG_TO_STDOUT` were set explicitly for each case. For the length of each call, `os.environ`
was replaced entirely by the fixture's environ. Two capture runs gave byte-identical files.

## Known, intended differences

The tests encode each of these explicitly:

1. **`SKYRL_WANDB_N_GENERATIONS` is no longer forwarded by name.** The W&B rollout viewer that read it was dropped.
   - It still reaches the runtime env through the `SKYRL_` prefix, so the env-var dicts are identical.
   - The reference logged `Exporting SKYRL_WANDB_N_GENERATIONS=... to ray runtime env`. That line is gone; the name
     now appears in the prefixed-forwarding summary line instead.
2. **The forwarding block runs after upstream's `prepare_runtime_environment` returns.** In the reference it ran in
   the middle of that function.
   - Values and warnings are unchanged.
   - The insertion order of runtime-env keys differs. Ray does not use that order.
   - Info lines come in a different order.
   - The summary line no longer lists `SKYRL_RAY_PG_TIMEOUT_IN_S`. Upstream has already copied it, with the same
     value, before the forwarding block runs.

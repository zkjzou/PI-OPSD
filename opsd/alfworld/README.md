# ALFWorld

OPSD and GRPO/DAPO training on [ALFWorld](https://github.com/alfworld/alfworld), the **text-only**
`AlfredTWEnv` (TextWorld + PDDL) embodied instruction-following benchmark. The policy acts in a ReAct
loop and each turn must commit one `<action>...</action>` that matches an admissible command. Reward is
`reward_scale * (10 * won - 0.1 * n_invalid)`: `10 * reward_scale` on a win at the end of the episode and
`-0.1 * reward_scale` per invalid action. `reward_scale` is set per arm by `ALFWORLD_REWARD_SCALE` in the
recipe: 1.0 for the `main`, `beta` and `null` tables, 0.1 for most other tables.

The environment is a `skyrl_gym` env (`env_class="alfworld"`) in `opsd/envs/alfworld/`, driven by
SkyRL's stock `SkyRLGymGenerator`. SkyRL is unmodified: the entrypoints register the env with skyrl_gym
(`opsd.envs.registry.register_opsd_envs()`), and its settings live under `environment.skyrl_gym.alfworld`
in the OPSD config tree (`opsd/skyrl_ext/config.py`). Its prompt templates are adapted from SDAR /
verl-agent (see the notice in `opsd/envs/alfworld/env.py`).

**Two environments.** `alfworld` needs `textworld`, `gymnasium==0.29.1` and a native PDDL/Inform7 stack
that conflict with the SkyRL + vLLM training environment. ALFWorld therefore runs in its **own venv** as a pool of
CPU-only FastAPI servers (`server/alfworld_server.py`, one game per server). The training env never
imports `alfworld`; it is an HTTP client that leases one server per rollout.

## 1. Setup (once)

```bash
bash scripts/setup/setup_alfworld.sh
```

This creates the ALFWorld venv (`$ALFWORLD_VENV`) with pinned versions (`alfworld==0.4.2`,
`textworld==1.7.0`, `gymnasium==0.29.1`, `stable-baselines3==2.6.0`) and downloads the game data into
`$ALFWORLD_DATA` (`json_2.1.1/{train,valid_seen,valid_unseen}` + `logic/`). The training side needs only
the training environment (`bash scripts/setup/setup_train_env.sh`).

Then build the SkyRL parquets. Each row stores the absolute path of its game under `$ALFWORLD_DATA`, so
rebuild them if the data moves:

```bash
python opsd/alfworld/alfworld_dataset.py --local_dir "$DATA_DIR" \
    --alfworld_data "$ALFWORLD_DATA" --splits train valid_seen valid_unseen
# -> train.parquet, validation.parquet (valid_seen, 140 games), validation_unseen.parquet (134 games)
```

## 2. Serve

The training launcher starts and stops its own server pool. To start one by hand, for standalone
generation or evaluation:

```bash
ALFWORLD_SERVER_URLS=$(NUM_ALFWORLD_SERVERS=16 ALFWORLD_BASE_PORT=9100 ALFWORLD_PID_DIR=./runs/pool/pids \
    bash opsd/alfworld/server/launch_alfworld_servers.sh | tail -1)
# ... later, stop exactly this pool:
kill $(cat ./runs/pool/pids/alfworld_serve.pids)
```

The script installs nothing. It fails if `$ALFWORLD_VENV` or `$ALFWORLD_DATA` is missing.

- Keep `environment.skyrl_gym.max_env_workers` at or above the number of concurrent trajectories, or
  the env thread pool deadlocks waiting for servers.
- Keep the pool's ports clear of vLLM's (the paper launchers use base port 9100).
- Each `/reset` pins about 35 MB of temp files until that server exits, so restart long-lived pools.

## 3. Run

Runs are named by arm id, `alfworld/<table>/<student>/<pi>[/<key>=<value>...][/seed=<n>]`, and resolved
from the recipes in `opsd/recipes/alfworld/`:

```bash
python -m opsd.recipes list --bench alfworld
DRY_RUN=1 bash scripts/alfworld/run.sh alfworld/main/4bi/glm_ao     # print the resolved command
bash scripts/alfworld/run.sh alfworld/main/4bi/glm_ao
```

Hint artifacts, task-id splits and reference trajectories come from the data bundle under
`$OPSD_DATA_ROOT/alfworld/`. 4B students use one 8×H100 80GB node, and the 30B student uses two nodes.
OPSD arms run `python -m opsd.trainer.main_opsd_alfworld`. GRPO/DAPO baselines run
`python -m opsd.alfworld.main_alfworld_dapo`, which runs SkyRL's `main_base` flow with the OPSD config
tree and trainer base, plus the DAPO soft overlong penalty (off unless
`trainer.algorithm.overlong_buffer.len > 0`).

## Files

| file | purpose |
|---|---|
| `main_alfworld_dapo.py` | GRPO/DAPO entrypoint |
| `alfworld_dataset.py` | builds the SkyRL parquets from `$ALFWORLD_DATA` |
| `server/` | env server (`alfworld_server.py`, `config_tw.yaml`) and pool launcher |
| `generate_alfworld_local.py` | generation-only rollouts against a running vLLM/OpenAI-compatible server; `--hints-path` / `--hint-placement` measure in-context PI utility |
| `run_alfworld_generate_local.sh` | generation-only eval on SkyRL-managed vLLM engines (`python -m opsd.skyrl_ext.main_generate`: SkyRL's generation-only entrypoint with the OPSD config, envs and eval metrics), written as SkyRL eval dumps |
| `generate_alfworld_litellm.py`, `run_alfworld_glm_litellm_local.sh` | reference trajectories from an API model behind an OpenAI-compatible endpoint (`--base-url`, `--api-key-env`); the wrapper's header documents the reference contract, including the admissible list shown after invalid actions |
| `build_pi_hint_formats.py` | trajectory hints from won references: `actions`, `raw`, `raw_no_think`, `actions_think` |
| `finalize_2k_hints.sh` | provenance: merges the two reference rounds into the 2400-task artifacts |
| `build_hint_eval_sets.py` | one eval parquet per hint set over shared task ids (`EVAL_HINT_SETS`) |
| `aggregate_utility_delta.py` | utility Δ table (with-hint minus no-hint success, paired per task) |
| `render_trajectories_html.py` | HTML viewer for eval and training-rollout dumps |
| `qwen3thinking_preserve.jinja2` | Qwen3 chat template that re-renders `<think>` for every assistant turn |

## What to watch

- `environment/num_invalid_actions`. A broken grammar or parser makes every action invalid and pins
  reward at the penalty.
- `generate/num_error_trajectories` ≈ 0: servers reachable and games loading.
- `environment/success` (won) and `environment/partial_success` (goal-condition rate, logged only).

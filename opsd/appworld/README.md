# AppWorld

OPSD and GRPO/DAPO training on [AppWorld](https://github.com/StonyBrookNLP/appworld): the policy writes
Python that calls simulated app APIs in a stateful sandbox and is scored by AppWorld's own evaluator
(binary task success). AppWorld is a `skyrl_gym` env (`env_class="appworld"`, in `opsd/envs/appworld/`)
driven by SkyRL's stock `SkyRLGymGenerator`, so multi-turn tokenization and per-turn loss masking come from
SkyRL, which stays unmodified: the entrypoints register the env with `opsd.envs.registry.register_opsd_envs()`.

| file | role |
|---|---|
| `main_appworld.py` | GRPO / DAPO entrypoint (`python -m opsd.appworld.main_appworld`): app-balanced sampler, DAPO overlong penalty, `AppWorld/*` W&B section, full-text rollout dumps |
| `opsd/trainer/main_opsd_appworld.py` | OPSD / RLSD / λ-mix / β entrypoint (`python -m opsd.trainer.main_opsd_appworld`) |
| `appworld_dataset.py` | builds the SkyRL parquets from AppWorld's splits (runs in the AppWorld venv) |
| `app_balanced_sampler.py` | per-app batch quotas (`APPWORLD_APP_BALANCED_SAMPLING=1`, mode `cover`) |
| `official_react_instructions.txt` | AppWorld's public ReAct code-agent prompt, byte-identical to upstream (Apache-2.0) |
| `generate_appworld_local.py` | rollouts against a running OpenAI-compatible endpoint (reference trajectories, offline pass@k, PI utility) |
| `generate_appworld_official_react.py` | optional: in-process port of the official ReAct agent (prompt-faithfulness check) |
| `opsd/envs/appworld/` | `env.py` (HTTP client + exclusive server pool), `scaffold.py` (action-parsing contract), `config.py` (`AppWorldEnvConfig`, the `environment.skyrl_gym.appworld` settings) |

## Two venvs

`appworld==0.1.3.post1` pins **pydantic < 2**; vLLM needs pydantic 2. They cannot share a venv, so:

- **Training venv** (`.venv` at the repository root, built by `bash scripts/setup/setup_train_env.sh`,
  or with `--extra megatron` for the 30B student): trainer, vLLM and `AppWorldEnv`. It never imports `appworld`; `AppWorldEnv` is a thin HTTP client.
- **AppWorld venv** (`$APPWORLD_VENV`, Python >= 3.11): the `appworld` package and data. It runs a pool of
  CPU-only `appworld serve environment` servers and builds the dataset parquets. Each rollout leases one
  server exclusively for its whole episode.

## 1. Setup (once)

```bash
export APPWORLD_ROOT=$HOME/appworld_root APPWORLD_VENV=$HOME/appworld-venv
export OPSD_DATA_ROOT=$HOME/.cache/opsd/data     # unpacked data bundle (PI hints), see the top-level docs
bash scripts/setup/setup_appworld.sh
```

This creates the AppWorld venv with pinned requirements, runs `appworld install` and
`appworld download data` inside `$APPWORLD_ROOT`, verifies and unpacks every encrypted AppWorld data bundle
present under `$OPSD_DATA_ROOT/appworld/` to `$OPSD_DATA_ROOT/appworld/plain/<bundle directory>/` (the batch-1
`appworld_opsd_pi.bundle` holds the PI hints and reference trajectories), and applies the bundled `b0a8eae`
ground-truth repair to `$APPWORLD_ROOT`.

**Why a bundle.** AppWorld's license requires anything derived from its protected data to be
redistributed encrypted. The PI hints, the reference trajectories and the `b0a8eae` repair are therefore
only in the bundle, never in this repo. Without the repair, `appworld_dataset.py` drops the three broken
`b0a8eae_*` train tasks (their evaluator compares seconds with minutes) and you train on 87 tasks. Every
paper arm trained on the repaired 90.

## 2. Data and servers

`scripts/appworld/run.sh` does both on every run (idempotent):

- builds `train.parquet` (90), `validation.parquet` (dev 57), `eval_normal.parquet` (test_normal 168)
  and `eval_challenge.parquet` (one task per test_challenge scenario, 139) with
  `appworld_dataset.py --split grpo_recipe --prompt_variant official`;
- starts `NUM_APPWORLD_SERVERS` supervised servers on `127.0.0.1:$APPWORLD_BASE_PORT...` from the
  AppWorld venv, waits until they accept connections, and stops only the servers it started.

The servers leak host memory across episodes. `APPWORLD_SERVER_RECYCLE_AFTER=N` restarts each one every
N episodes (the paper recipes use 25). The paper's 240 servers assume a large host (about 192 vCPUs and
2 TiB RAM). On smaller hosts, lower `NUM_APPWORLD_SERVERS`; rollouts then queue for a free server and run
in waves. Keep `MAX_ENV_WORKERS` >= `train_batch_size x n_samples_per_prompt` (18 x 8 = 144).

CPU-only smoke test of one server and the env (no GPU):

```bash
(cd "$APPWORLD_ROOT" && "$APPWORLD_VENV/bin/appworld" serve environment --port 9000 --no-show-usage) &
TID=$(cd "$APPWORLD_ROOT" && "$APPWORLD_VENV/bin/python" -c "from appworld import load_task_ids; print(load_task_ids('train')[0])")
.venv/bin/python - <<PY
from opsd.envs.appworld.env import AppWorldEnv
env = AppWorldEnv({"server_urls": ["http://127.0.0.1:9000"], "strict_parse": True},
                  {"reward_spec": {"ground_truth": "$TID"}, "max_turns": 5})
env.init([{"role": "user", "content": "task"}])
print(env.step("\`\`\`python\nprint(apis.api_docs.show_app_descriptions())\n\`\`\`")["observations"])
print(env.step("\`\`\`python\napis.supervisor.complete_task()\n\`\`\`"))   # done + reward
print(env.get_metrics()); env.close()
PY
```

## 3. Train

Runs are named by arm id (`<bench>/<table>/<student>/<pi>[/key=value...]`) and resolved from
`opsd/recipes/appworld/`:

```bash
python -m opsd.recipes list --bench appworld
DRY_RUN=1 bash scripts/appworld/run.sh appworld/grpo/4bi/none/seed=1   # print module, argv, env; no GPU
bash scripts/appworld/run.sh appworld/grpo/4bi/none/seed=1              # DAPO baseline
bash scripts/appworld/run.sh appworld/main/4bi/glm_aot                  # vanilla OPSD, GLM-5.2 PI
bash scripts/appworld/run.sh appworld/rlsd/4bi/glm_aot                  # RLSD
```

Override any recipe key on the command line (`KEY=VAL`). Outputs go to `$RUN_ROOT/$RUN_NAME`
(checkpoints, rollouts, `resolved.env`, `provenance.json`).

- **4B arms** (Qwen3-4B-Instruct-2507 / Thinking-2507): one node with 8x80 GB GPUs, FSDP2, colocated vLLM.
- **30B arms** (Qwen3-30B-A3B-Instruct-2507): Megatron backend. DAPO fits on one node; OPSD needs two.
  Run the launcher on the head node, and make sure that node has CPUs registered with Ray. The training
  loop is pinned to the driver's node (hard affinity) because the server pool listens on that node's
  `127.0.0.1` and server recycling scans its `/proc`. A head started with `--num-cpus=0` makes the task
  fail with `TaskUnschedulableError` instead of running elsewhere.

Shared protocol of the paper arms: train-90, strict ```` ```python ```` parsing with the `official`
scaffold, binary reward, at most 30 turns, 18 prompts x 8 samples per step. Validation runs on
test_normal-168 every 25 steps (n=4, temperature 0.7, top-p 0.8, top-k 20). The reported window metric
(best mean pass@1 over 3 consecutive evals) is selected on that same curve.

### Offline rollouts (PI metrics)

The AppWorld PI-metric corpora are plain `generate_appworld_local.py` runs against a served policy:
`scripts/appworld/passk.sh` builds the no-PI corpus (the paper's D uses 128 rollouts per task; it is
also the no-hint baseline for U), and `scripts/appworld/pi_utility.sh` runs the n=4 with-PI
condition for U (`--hints-path`, `--hint-inject-style system-prepend`).

Both scripts default to the paper's settings; keep them to regenerate the paper's corpora:

| setting | value |
|---|---|
| sampling | `onpolicy`: temperature 1.0, top_p 1.0, top_k -1, min_p 0 |
| samples per task | 128 without a PI (`passk.sh`); 4 with the PI (`pi_utility.sh`) |
| splits | train-90 + dev-57, pooled in the paper (`passk.sh` default; `pi_utility.sh SPLITS="train90 dev"` reads the dev hints from the decrypted `appworld_opsd_dev_hints.bundle`, evaluation-split material, or from `HINTS_DIR_dev`) |
| vLLM | `--max-model-len 49152`; TP1 for 4bi / 4bt, TP2 for 30bi |
| episode | official prompt + scaffold, strict parsing, binary reward, 30 turns, 50 interactions, 8192 tokens per turn, observations cut to 20000 characters |
| 4bt | `--preserve-thinking --enable-thinking` (as run) |

The original run-files defaulted to `SAMPLING_PROFILE=official` (0.7 / 0.8 / top-k 20) and 16 samples
per task. Those are not the paper settings.

Related analysis drivers:
- `scripts/analysis/build_appworld_nohint_baselines_from_passk.sh` converts the `passk.sh` transcripts
  into the U no-hint baselines.
- `scripts/analysis/appworld_utility_both_k.py` computes U from `pi_utility.sh` output and those baselines.

`generate_appworld_local.py` needs `--served-model-name`. If `$OPENAI_API_KEY` (or the variable named by
`--api-key-env`) is set, it is sent to every `--base-url`, including a local vLLM; pass `--api-key-env ''`
to send no key.

## 4. Metrics

`main_appworld` and `main_opsd_appworld` mirror AppWorld metrics into an `AppWorld/` W&B section; the
original keys stay:

- `AppWorld/train/{success,partial_completion,num_turns,completed_task_called,eval_unavailable,parse_errors}`
- `AppWorld/eval/<split>/{success,...,avg_score,pass_at_N}`, with `<split>` = `test_normal`,
  `test_challenge` or `all`. The headline curve is `eval/appworld_test_normal/pass_at_1`.

Health checks: `generate/num_error_trajectories` near 0 (servers reachable), `environment/eval_unavailable`
near 0, `environment/parse_errors` flat (a rise means format drift).

## License notes

- AppWorld code and data are not vendored. `official_react_instructions.txt` is upstream's public
  Apache-2.0 prompt.
- The LOOP prompt variant (`--prompt_variant loop`) is not redistributed because it is under Apple's
  sample-code license. Pass your own copy with `--instructions-path`.
- AppWorld-derived artifacts (hints, trajectories, the `b0a8eae` repair, PI-metric inputs) ship only in
  the encrypted bundles (see Setup).

# τ²-bench integration for SkyRL

Runs the [τ²-bench](https://github.com/sierra-research/tau2-bench) dual-control benchmark (telecom,
retail, airline) inside SkyRL: a standalone evaluation / pass@k harness (`baseline.py`), a GRPO/DAPO
entrypoint (`main.py`) and an on-policy self-distillation (OPSD) entrypoint (`distill/main_tau2_opsd.py`).
The paper's τ² results are all on **retail** (train pools `retail_*`, eval on `retail-base`, 114 tasks).

## Design in one paragraph

τ² already contains the entire episode loop — agent ↔ user ↔ environment, including **dual control**
(the user also holds tools and mutates env state; telecom only). We do **not** reimplement any of it and
we do **not** fork or patch τ². `tau2.agent.llm_agent.LLMAgent` calls a module-level
`generate(model, tools, messages, **llm_args)` that splats straight into `litellm.completion`, so
`llm_args` *is* the injection point: point `llm="openai/<served-model>"` at any OpenAI-compatible endpoint
via `api_base`/`api_key`. `episode.run_episode` builds env + agent + user + `Orchestrator` and runs τ²'s
own synchronous `run_simulation` (which runs **and** evaluates) on a worker thread.

## Layout

| file | role |
|------|------|
| `constants.py` | the pinned τ² revision, the pinned AReaL dataset revision, split manifest specs, sampling profiles, endpoint defaults |
| `source.py` | `ensure_checkout()` (git clone at the pin) and `bootstrap_tau2()` — imports τ² from a verified checkout with every litellm hazard neutralised |
| `manifest.py` | build/validate the `data/*.jsonl` manifests; string-safe uids (GRPO `:repeat-N` safe) |
| `agent.py` | `PolicySpec` / `UserSpec` → τ²'s stock `LLMAgent` / `UserSimulator` |
| `episode.py` | `run_episode()` — one episode → a normalised `Tau2EpisodeResult` with a status taxonomy |
| `metering.py` | per-leg (policy / sim / judge) token & USD accounting via a litellm success callback |
| `areal.py` | AReaL SEA-generated RL tasks (`inclusionAI/AReaL-tau2-data`) → native τ² `Task`s |
| `baseline.py` | standalone eval / pass@k harness: concurrency, resume, coverage pass@k, cost, PI-hint utility |
| `prepare_data.py` | `prepare` / `validate` / `audit` CLI (the `audit` subcommand is the judge safety gate) |
| `generator.py`, `main.py` | SkyRL generator + GRPO/DAPO entrypoint (`Tau2Config` under `tau2.*`) |
| `distill/` | OPSD generator / trainer / entrypoint |
| `build_tau2_pi_hints.py` | τ² trajectories → PI hint artifacts (`raw` / `raw_no_think` / `actions` / `actions_think`) |
| `aggregate_utility_delta.py`, `reward_replay.py` | paired with-hint vs no-hint Δ; replay of τ²'s shipped results through the evaluator |
| `serve_user_sim.sh`, `run_tau2_passk.sh`, `run_tau2_pi_utility.sh` | self-hosted user simulator; pass@k and PI-utility samplers |
| `data/*.jsonl` | public split manifests (task ids + digests only; task contents stay in the pinned checkout) |

## Setup (one-time)

`bash scripts/setup/setup_tau2.sh` automates these steps (training environment, τ² packages, the pinned
checkout and its manifests, the AReaL retail and airline pools; see its header). By hand, from the repository
root:

```bash
# 0. training environment (.venv: `uv sync --frozen` against the root uv.lock, with the external SkyRL
#    checkout installed editable; add --extra megatron for the 30B (MoE) student) and
# 1. τ²'s runtime deps that SkyRL does not already provide, the pyproject `tau2` extra: addict deepdiff
#    orderly-set cachebox toml, at the versions in uv.lock, so the locked litellm is left alone.
#    MUST precede step 2: every `tau2.utils` import needs addict + deepdiff + toml.
bash scripts/setup/setup_train_env.sh --tau2     # = uv sync --frozen --extra fsdp --extra tau2 (+ the extras
                                                 #   .venv has: dev, or megatron instead of fsdp)

# 2. pinned τ² checkout (git clone of sierra-research/tau2-bench at constants.TAU2_REVISION into
#    ~/.cache/skyrl/tau2-bench/<rev>) + manifests. τ² is never pip-installed: its litellm pin clashes
#    with SkyRL's, so <checkout>/src is put on sys.path instead.
python -m opsd.tau2_bench.prepare_data prepare --allow-judge
python -m opsd.tau2_bench.prepare_data audit     # inspect judge exposure per split

# 3. stage the AReaL pools (pinned to constants.AREAL_REVISION) so training never hits the HF hub.
#    The paper uses retail only; airline / telecom are needed only for those domains' *-areal-rl pools.
python -c "from opsd.tau2_bench.areal import stage_local; print(stage_local(domain='retail'))"
python -c "from opsd.tau2_bench.areal import stage_local; print(stage_local(domain='airline'))"
python -c "from opsd.tau2_bench.areal import stage_local; print(stage_local(domain='telecom'))"
```

## Endpoints: user simulator and NL judge

Both are reached through any OpenAI-compatible endpoint. Defaults:

| leg | model | base URL | key |
|-----|-------|----------|-----|
| user simulator | `gpt-5.6-luna`, T=0.0, `reasoning_effort=none`, `max_tokens=2048` | `$OPENAI_BASE_URL`, else `https://api.openai.com/v1` | `$OPENAI_API_KEY` |
| NL judge (retail only; inside the reward) | `openai/gpt-4.1` | same | same |

Override with `tau2.simulator_{model,base_url,api_key_env}` / `tau2.judge_{model,base_url,api_key_env}`
(training) or `--user-llm/--user-base-url/--user-api-key-env` / `--judge-*` (`baseline.py`). The
`*_api_key_env` settings name an environment variable; keys are never passed on the command line.
A self-hosted simulator (`serve_user_sim.sh`, plain `http://` URL) needs no key.

> **The simulator IS the environment.** Rewards, solve rates and pass@k are only comparable between runs
> that used the same simulator; re-measure the baseline before comparing across simulators.

## The three litellm hazards `bootstrap_tau2` neutralises

1. **10-connection cap.** Importing τ² sets a global `litellm.client_session` limited to 10 connections,
   throttling any concurrent sweep. We replace it with a wide `httpx.Client`.
2. **`drop_params=True`.** τ² sets this globally, which *silently drops* vLLM-only sampling params
   (`top_k`, `min_p`, `chat_template_kwargs`). `PolicySpec` nests them under `extra_body` — **and only
   when the endpoint is a self-hosted `http://` engine**, because a hosted gateway 400s on them.
3. **The reward function makes an outbound LLM call.** For retail, `EvaluationType.ALL` invokes an
   NL-assertion judge (default `gpt-4.1` at OpenAI) *inside* the reward.
   `bootstrap_tau2(judge_model=…, judge_llm_args=…)` redirects it to the configured judge endpoint, and an
   egress guard raises if anything tries to reach a host outside the resolved policy / simulator / judge
   URLs.

Measured judge usage at the pin (`prepare_data.py audit`): airline 0, telecom 0, **retail-base 40**,
**retail-train 29**. The AReaL pools are judge-free (`[DB, COMMUNICATE]` reward basis).

## Reward gotcha

τ² scores **0 unless the episode terminates on `AGENT_STOP` or `USER_STOP`**. A `MAX_STEPS` termination
is a hard zero regardless of task progress. `baseline.py` reports `rewardable_fraction` per domain so a
weak policy (low reward, high rewardable_fraction) is never confused with a broken integration (near-zero
rewardable_fraction).

## Evaluating a served policy

```bash
# 1. serve the policy — MUST enable native tool-calling, because τ²'s LLMAgent sends tools +
#    tool_choice="auto" (vLLM 400s "auto tool choice requires --enable-auto-tool-choice" otherwise).
vllm serve Qwen/Qwen3-4B-Instruct-2507 --enable-auto-tool-choice --tool-call-parser hermes --port 8300

# 2. run the baseline (user simulator + judge at $OPENAI_BASE_URL / $OPENAI_API_KEY by default)
python -m opsd.tau2_bench.baseline \
    --split retail-base --num-trials 4 --concurrency 32 --no-thinking \
    --model Qwen/Qwen3-4B-Instruct-2507 --policy-base-url http://127.0.0.1:8300/v1 \
    --output-dir runs/tau2_eval/q4bi_retail_base
```

`trajectories.jsonl` is append-and-resume keyed by `(domain, task_id, trial)`; re-running continues an
interrupted sweep. `summary.json` holds per-domain mean reward, **coverage** pass@k, τ²'s pass^k, the
termination histogram, the reward-component breakdown, and per-leg API cost.
`run_tau2_passk.sh` wraps the same harness with an N-replica local policy pool (pass@128 corpora), and
`run_tau2_pi_utility.sh` runs hint-guided vs no-hint arms (`--hints-path --require-hint`) for the
PI-utility measurement; both document their inputs in their headers.

## Training (GRPO / DAPO / OPSD)

Paper arms are launched through the recipe launcher, `bash scripts/tau2/run.sh <arm_id>` (list the
arms with `python -m opsd.recipes list --bench tau2`); the launcher sets each objective's hydra
knobs (e.g. the vanilla-OPSD combination documented in `distill/main_tau2_opsd.py`), so do not set them by
hand. The entrypoints are `python -m opsd.tau2_bench.main` (GRPO/DAPO, `Tau2DAPOTrainer`) and
`python -m opsd.tau2_bench.distill.main_tau2_opsd` (OPSD; per-task hints via
`OPSD_HINTS_PATH`). Both pin the SkyRL entrypoint task to the driver node.

## AReaL RL train pool (retail) — `areal.py`

The `retail-areal-rl` split is **AReaL's SEA-generated RL training set** (`inclusionAI/AReaL-tau2-data`,
HF, pinned to `constants.AREAL_REVISION`): **563 retail task specs**, each with its own DB snapshot (τ²'s
native retail-train has only 74, all sharing one DB — the AReaL set's per-task DBs are the
anti-overfitting lever). `areal.py` converts each record into a **native upstream τ² `Task`** — no fork:

- `evaluation_criteria` is a JSON string → `json.loads`; its omitted `reward_basis` then defaults to
  `[DB, COMMUNICATE]`, **identical to AReaL's fork default** — so we reproduce their scoring with no
  override (and, unlike τ²'s native retail, **no NL-judge**, so this pool is deterministic + judge-free).
- the per-task DB (`db_path`) is injected via `Task.initial_state.initialization_data`; the upstream
  orchestrator applies it to the agent env and the evaluator applies the *same* `initial_state` to the
  DB-target env, so the per-task DB is honoured on both sides of the DB check.

`load_tasks("retail", "retail-areal-rl")` returns the 563 converted tasks (verified: all 563 validate).
The curated `retail_*` training pools in `data/` (`retail_pi424`, `retail_tri352`, `retail_chk*`, …) are
subsets of this pool; `retail-base` stays the held-out eval so numbers remain leaderboard-comparable.

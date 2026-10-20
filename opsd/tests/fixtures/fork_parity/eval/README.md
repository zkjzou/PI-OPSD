# Regression goldens: eval

Reference outputs of the reference implementation (OPSD's eval changes made directly inside SkyRL
`ecdaa946`). Its eval code was `skyrl/train/evaluate.py` (`evaluate`), `skyrl/train/utils/trainer_utils.py`
(`calculate_per_dataset_metrics`, `dump_per_dataset_eval_results`) and `skyrl/train/generators/utils.py`
(`get_metrics_from_generator_output`). The copy in this repository is `opsd/skyrl_ext/evaluate.py`; the tests
are `opsd/skyrl_ext/tests/test_eval_fork_parity.py` and `test_generator_metrics.py`.

## What was captured

`<env_class>__n<N>__<env_metrics>.json`: 24 cases, each run through the reference `evaluate()` with a stub
generator, a stub tokenizer (`decode -> "<id><id>..."`) and a default `SkyRLTrainConfig`
(vllm backend, fixed eval sampling params, `dump_eval_results=True`):

| factor | values |
|---|---|
| `env_class` | `alfworld`, `appworld`, `tau2_bench` (data sources `alfworld` + `alfworld_valid_unseen`, `appworld` + `appworld_test_normal`, `retail` + `airline`) |
| `eval_n_samples_per_prompt` | 1, 4 |
| `env_metrics` | `success` (every row has `success`, which disagrees with `reward > 0` on some rows), `no_success`, `mixed` (every third row lacks `success`), `absent` (no `env_metrics` key; `global_step=None`, so the dump goes to `eval_only/`) |
| eval batches | 2 (3 + 2 prompts, both mixing the two data sources); each batch has `rollout_metrics` with `environment/*` and one custom key per concat heuristic branch |

Each file holds the inputs (`eval_batches`, `generator_outputs`), the generator inputs `evaluate()` built
(`generator_inputs`), every file it dumped under `export_path` as text (`dumps`; identical across the four
variants, which the capture asserted) and the returned dict for each post-processor variant (`metrics`):
`plain`, `compact` (`SKYRL_EVAL_COMPACT=1`), `namespace_test` (`SKYRL_EVAL_NAMESPACE=test`) and
`compact_namespace_test`.

`helpers.json`: the reference `get_metrics_from_generator_output` (24 inputs: scalar and token-level rewards;
`success` present / absent / on some rows / wrong length / `env_metrics=None`) and `calculate_per_dataset_metrics`
(12 inputs: `None` and `a/b` data sources, N in {1, 3}, with and without `concat_env_classes`).

The env ids resolved through the reference skyrl_gym registry (`alfworld`, `appworld`) and the production
metrics-only `tau2_bench` registration; all three aggregate with `default_aggregate_metrics`. All inputs are
synthetic and seeded (`random.Random(crc32(case name))`); re-running the capture reproduces the files
byte-for-byte. No host paths, usernames or real environment values appear in them.

These files are fixed reference data; the capture script is not part of this repository.

"""Shared stubs for the opsd.skyrl_ext.evaluate tests (no tests here).

Configs are built directly from ``SkyRLTrainConfig`` (``tests.train.*`` is not importable from this tree).
"""

import copy
import json
from contextlib import contextmanager
from pathlib import Path

from opsd._repo import fixtures_dir
from skyrl.train.config import SamplingParams, SkyRLTrainConfig
from skyrl.train.generators.base import GeneratorInterface, TrajectoryID

# Env ids the eval goldens use. All three aggregate env_metrics with skyrl_gym's default_aggregate_metrics
# (ALFWorld / AppWorld inherit BaseTextEnv.aggregate_metrics; tau2_bench registers a metrics-only env).
EVAL_ENV_IDS = ("alfworld", "appworld", "tau2_bench")
EVAL_ENV_KNOBS = ("SKYRL_EVAL_COMPACT", "SKYRL_EVAL_NAMESPACE")
# Post-processor variants captured per golden case (must match capture_eval.py).
EVAL_VARIANTS = {
    "plain": {},
    "compact": {"SKYRL_EVAL_COMPACT": "1"},
    "namespace_test": {"SKYRL_EVAL_NAMESPACE": "test"},
    "compact_namespace_test": {"SKYRL_EVAL_COMPACT": "1", "SKYRL_EVAL_NAMESPACE": "test"},
}


def fixture_dir() -> Path:
    """The eval goldens, ``opsd/tests/fixtures/fork_parity/eval``."""
    return fixtures_dir() / "fork_parity" / "eval"


def case_names():
    return sorted(p.stem for p in fixture_dir().glob("*__*.json"))


def load_case(name: str) -> dict:
    return json.loads((fixture_dir() / f"{name}.json").read_text())


class _DefaultMetricsEnv:
    @staticmethod
    def aggregate_metrics(metrics):
        from skyrl_gym.metrics import default_aggregate_metrics

        return default_aggregate_metrics(metrics)


@contextmanager
def registered_eval_envs():
    """Register metrics-only stand-ins for the eval env ids that are not registered yet; undo on exit."""
    from skyrl_gym.envs.registration import deregister, register, registry

    added = [env_id for env_id in EVAL_ENV_IDS if env_id not in registry]
    for env_id in added:
        register(id=env_id, entry_point=_DefaultMetricsEnv)
    try:
        yield
    finally:
        for env_id in added:
            deregister(env_id)


def make_cfg(env_class: str, n_samples_per_prompt: int, export_path, dump_eval_results: bool = True):
    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.backend = "vllm"
    cfg.generator.eval_sampling_params = SamplingParams(
        max_generate_length=64, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, logprobs=None, stop=["</answer>"]
    )
    cfg.generator.eval_n_samples_per_prompt = n_samples_per_prompt
    cfg.generator.step_wise_trajectories = False
    cfg.environment.env_class = env_class
    cfg.trainer.dump_eval_results = dump_eval_results
    cfg.trainer.export_path = str(export_path)
    cfg.trainer.log_example_interval = 1
    return cfg


class StubLoader:
    def __init__(self, batches):
        self._batches = batches

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


class StubGenerator(GeneratorInterface):
    """Returns the i-th stored output for the i-th generate() call."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.seen_inputs = []

    async def generate(self, input_batch):
        self.seen_inputs.append(input_batch)
        return self._outputs[len(self.seen_inputs) - 1]


class StubTokenizer:
    def decode(self, tokens):
        return "".join(f"<{t}>" for t in tokens)


def materialize_outputs(record: dict):
    outs = copy.deepcopy(record["generator_outputs"])
    for go in outs:
        if "trajectory_ids" in go:
            go["trajectory_ids"] = [TrajectoryID(**t) for t in go["trajectory_ids"]]
    return outs


def normalize_seen(seen_inputs):
    return [
        {
            "prompts": gi["prompts"],
            "env_classes": gi["env_classes"],
            "env_extras": gi["env_extras"],
            "sampling_params": gi["sampling_params"],
            "trajectory_ids": [[t.instance_id, t.repetition_id] for t in gi["trajectory_ids"]],
            "batch_metadata": {
                "global_step": gi["batch_metadata"].global_step,
                "training_phase": gi["batch_metadata"].training_phase,
            },
        }
        for gi in seen_inputs
    ]


def read_dumps(export_path) -> dict:
    """{path relative to export_path: file bytes} for every file evaluate() wrote."""
    root = Path(export_path)
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


async def run_golden_case(record: dict, export_path):
    """Run opsd evaluate() on a golden case's inputs; returns (metrics, generator)."""
    from opsd.skyrl_ext.evaluate import evaluate

    gen = StubGenerator(materialize_outputs(record))
    metrics = await evaluate(
        eval_dataloader=StubLoader(copy.deepcopy(record["eval_batches"])),
        generator=gen,
        cfg=make_cfg(record["env_class"], record["n_samples_per_prompt"], export_path),
        global_step=record["global_step"],
        tokenizer=StubTokenizer(),
    )
    return metrics, gen

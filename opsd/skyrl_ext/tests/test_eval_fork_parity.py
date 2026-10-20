"""Fork parity: opsd.skyrl_ext.evaluate.evaluate vs the fork's skyrl.train.evaluate.evaluate.

The goldens in opsd/tests/fixtures/fork_parity/eval/ were captured from the fork (see that folder's README.md). For
every case (env_class x N x env_metrics mode, 2 eval batches, 2 data sources) and every SKYRL_EVAL_COMPACT x
SKYRL_EVAL_NAMESPACE variant, the returned dict must be equal (same keys, same order, same values) and every
dumped file byte-equal. The remaining tests pin what downstream readers need from the same runs.
"""

import asyncio
import json
import re

import pytest

from opsd._repo import repo_root
from opsd.eval import window_metric
from opsd.skyrl_ext.tests.test_eval_support import (
    EVAL_ENV_KNOBS,
    EVAL_VARIANTS,
    case_names,
    load_case,
    normalize_seen,
    read_dumps,
    registered_eval_envs,
    run_golden_case,
)

CASES = case_names()
BENCH = {"alfworld": "alfworld", "appworld": "appworld", "tau2_bench": "tau2"}


@pytest.fixture
def eval_envs():
    with registered_eval_envs():
        yield


@pytest.fixture
def no_eval_knobs(monkeypatch):
    for knob in EVAL_ENV_KNOBS:
        monkeypatch.delenv(knob, raising=False)


def _aggregated(record, tmp_path):
    step = record["global_step"]
    d = tmp_path / "dumped_evals" / ("eval_only" if step is None else f"global_step_{step}_evals")
    return json.loads((d / "aggregated_results.jsonl").read_text())


def _final_test_metrics():
    """The metric names scripts/appworld/run.sh's FINAL_TEST checkpoint picker reads."""
    text = (repo_root(__file__) / "scripts" / "appworld" / "run.sh").read_text()
    names = re.findall(r'_metric="(eval/[^"]+)"', text)
    assert len(names) >= 2, names
    return names


def test_golden_matrix_is_complete():
    assert len(CASES) == 24
    for name in CASES:
        record = load_case(name)
        assert set(record["metrics"]) == set(EVAL_VARIANTS), name
        assert len(record["eval_batches"]) == 2, name


@pytest.mark.parametrize("variant", sorted(EVAL_VARIANTS))
@pytest.mark.parametrize("case_name", CASES)
def test_evaluate_matches_fork(case_name, variant, tmp_path, monkeypatch, eval_envs, no_eval_knobs):
    record = load_case(case_name)
    for knob, value in EVAL_VARIANTS[variant].items():
        monkeypatch.setenv(knob, value)

    metrics, gen = asyncio.run(run_golden_case(record, tmp_path))

    assert list(metrics.items()) == list(record["metrics"][variant].items())
    assert read_dumps(tmp_path) == {k: v.encode("utf-8") for k, v in record["dumps"].items()}
    assert normalize_seen(gen.seen_inputs) == record["generator_inputs"]


@pytest.mark.parametrize("case_name", [c for c in CASES if not c.endswith("__absent")])
def test_pooled_environment_metrics_are_count_weighted(case_name, tmp_path, eval_envs, no_eval_knobs):
    """eval/all/environment/* is the mean over every rollout of both batches (not the sum of batch means)."""
    record = load_case(case_name)
    metrics, _ = asyncio.run(run_golden_case(record, tmp_path))
    rows = [m for go in record["generator_outputs"] for m in go["env_metrics"]]

    for key in ("success", "num_turns"):
        vals = [float(m[key]) for m in rows if key in m]
        if not vals:
            assert f"eval/all/environment/{key}" not in metrics
            continue
        assert metrics[f"eval/all/environment/{key}"] == pytest.approx(sum(vals) / len(vals))
    if "eval/all/environment/success" in metrics:
        assert 0.0 <= metrics["eval/all/environment/success"] <= 1.0
        assert 0.0 <= _aggregated(record, tmp_path)["eval/all/environment/success"] <= 1.0


@pytest.mark.parametrize("case_name", CASES)
def test_downstream_reader_keys_present(case_name, tmp_path, eval_envs, no_eval_knobs):
    """Keys read by opsd.eval.window_metric (paper checkpoint selection) and the AppWorld FINAL_TEST picker."""
    record = load_case(case_name)
    metrics, _ = asyncio.run(run_golden_case(record, tmp_path))
    aggregated = _aggregated(record, tmp_path)
    bench = BENCH[record["env_class"]]
    n = record["n_samples_per_prompt"]
    has_success = record["env_metrics_mode"] in ("success", "mixed")

    # tau2's pass@1 leaf is environment/success, which exists only when the env reports success.
    if bench != "tau2" or has_success:
        for source in (aggregated, metrics):
            scores = window_metric.scores_from_metrics(source, bench)
            assert set(window_metric.PREFIXES[bench]) <= set(scores), (bench, sorted(scores))
            for split in window_metric.PREFIXES[bench]:
                assert scores[split].k == (n if n > 1 else None)

    if bench == "appworld" and has_success:
        for name in _final_test_metrics():
            assert name in aggregated, name


def test_calculate_per_dataset_metrics_matches_fork(eval_envs):
    """The fork's function on extra edge inputs: None / 'a/b' data sources, N in {1, 3}, with and without
    env classes (no environment/* without them), env_metrics with / without 'success' / absent."""
    from opsd.skyrl_ext.evaluate import calculate_per_dataset_metrics
    from opsd.skyrl_ext.tests.test_eval_support import fixture_dir

    cases = json.loads((fixture_dir() / "helpers.json").read_text())["calculate_per_dataset_metrics"]
    assert len(cases) == 12
    for case in cases:
        got = calculate_per_dataset_metrics(
            case["generator_output"],
            case["uids"],
            case["data_sources"],
            case["n_samples_per_prompt"],
            concat_env_classes=case["env_classes"],
        )
        assert list(got.items()) == list(case["result"].items())


@pytest.mark.parametrize("case_name", [c for c in CASES if c.endswith("__success")])
def test_namespace_renames_returned_dict_only(case_name, tmp_path, monkeypatch, eval_envs, no_eval_knobs):
    """FINAL_TEST (SKYRL_EVAL_NAMESPACE=test) logs test/*, but its dump keeps eval/* for the pickers."""
    monkeypatch.setenv("SKYRL_EVAL_NAMESPACE", "test")
    record = load_case(case_name)
    metrics, _ = asyncio.run(run_golden_case(record, tmp_path))
    aggregated = _aggregated(record, tmp_path)

    assert metrics and all(k.startswith("test/") for k in metrics)
    assert aggregated and all(k.startswith("eval/") for k in aggregated)
    assert {k[len("test/") :] for k in metrics} == {k[len("eval/") :] for k in aggregated}

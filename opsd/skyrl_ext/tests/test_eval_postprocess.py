"""SKYRL_EVAL_COMPACT / SKYRL_EVAL_NAMESPACE post-processors (untested in the fork)."""

import pytest

from opsd.skyrl_ext.evaluate import (
    compact_eval_metrics,
    namespace_eval_metrics,
    postprocess_eval_metrics,
)

METRICS = {
    "eval/alfworld/avg_score": 1.0,
    "eval/alfworld/pass_at_4": 1.0,
    "eval/alfworld/pass_at_1": 0.5,
    "eval/alfworld/mean_positive_reward": 5.0,
    "eval/alfworld/generate/avg_num_tokens": 12.0,
    "eval/alfworld/generate/max_num_tokens": 30,
    "eval/alfworld/environment/success": 0.5,
    "eval/all/avg_score": 1.0,
    "eval/all/environment/num_turns": 7.0,
    "eval/all/custom/infra_failures": 2.0,
    "timing/eval": 3.0,
}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SKYRL_EVAL_COMPACT", raising=False)
    monkeypatch.delenv("SKYRL_EVAL_NAMESPACE", raising=False)


def test_compact_off_by_default_returns_input():
    assert compact_eval_metrics(METRICS) is METRICS


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_compact_keeps_headline_metrics(monkeypatch, value):
    monkeypatch.setenv("SKYRL_EVAL_COMPACT", value)
    assert list(compact_eval_metrics(METRICS)) == [
        "eval/alfworld/avg_score",
        "eval/alfworld/pass_at_4",
        "eval/alfworld/pass_at_1",
        "eval/alfworld/generate/avg_num_tokens",
        "eval/alfworld/environment/success",
        "eval/all/avg_score",
        "eval/all/environment/num_turns",
    ]


@pytest.mark.parametrize("value", ["0", "false", "", "off"])
def test_compact_falsy_values(monkeypatch, value):
    monkeypatch.setenv("SKYRL_EVAL_COMPACT", value)
    assert compact_eval_metrics(METRICS) == METRICS


def test_namespace_renames_eval_prefix_only(monkeypatch):
    monkeypatch.setenv("SKYRL_EVAL_NAMESPACE", "test")
    out = namespace_eval_metrics(METRICS)
    assert out["test/alfworld/pass_at_4"] == 1.0
    assert out["timing/eval"] == 3.0
    assert not any(k.startswith("eval/") for k in out)
    assert METRICS["eval/alfworld/pass_at_4"] == 1.0  # input untouched


@pytest.mark.parametrize("ns", ["", "eval", "  "])
def test_namespace_noop_values(monkeypatch, ns):
    monkeypatch.setenv("SKYRL_EVAL_NAMESPACE", ns)
    out = namespace_eval_metrics(METRICS)
    assert out == METRICS and out is not METRICS


def test_pass_at_1_defaults_to_environment_success():
    """Only buckets without pass_at_1 get one; an existing pass_at_1 is never overwritten."""
    m = {"eval/a/environment/success": 0.25, "eval/b/pass_at_1": 0.9, "eval/b/environment/success": 0.1}
    assert namespace_eval_metrics(m) == {**m, "eval/a/pass_at_1": 0.25}


def test_postprocess_compacts_then_namespaces(monkeypatch):
    monkeypatch.setenv("SKYRL_EVAL_COMPACT", "1")
    monkeypatch.setenv("SKYRL_EVAL_NAMESPACE", "test")
    out = postprocess_eval_metrics(METRICS)
    assert "test/alfworld/environment/success" in out and "test/alfworld/mean_positive_reward" not in out
    assert "timing/eval" not in out  # compact drops keys outside eval/<bucket>/ too, as in the fork
    assert postprocess_eval_metrics(out) == out  # idempotent

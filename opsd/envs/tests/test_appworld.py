"""AppWorld env reward + metric computation tests (no live `appworld serve` needed).

    pytest opsd/envs/tests/test_appworld.py

AppWorldEnv talks raw HTTP to an `appworld serve` pool, so these tests construct the env
directly and monkeypatch ``_post`` to return canned ``/execute`` / ``/task_completed`` /
``/evaluate`` payloads. The focus is the parts that decide what W&B shows:
  - reward = success (binary) vs TGC fraction = passes/num_tests (partial),
  - get_metrics() keys/values (surfaced as environment/* after default_aggregate_metrics),
  - eval_unavailable flagging when /evaluate errors,
  - code extraction + empty-action handling in step().
"""

import pytest

from opsd.envs.appworld.env import AppWorldEnv, AppWorldEnvConfig, _extract_code
from skyrl_gym.metrics import default_aggregate_metrics


def make_env(reward_mode="partial", max_turns=40):
    cfg = AppWorldEnvConfig(server_urls=["http://x"], reward_mode=reward_mode)
    return AppWorldEnv(cfg, extras={"reward_spec": {"ground_truth": "t1"}, "max_turns": max_turns})


# --- canned /evaluate payloads (real schema: {success, num_tests, passes[], failures[]}) ---
EVAL_SOLVED = {"success": True, "num_tests": 4, "passes": ["a", "b", "c", "d"], "failures": []}
EVAL_PARTIAL = {"success": False, "num_tests": 4, "passes": ["a"], "failures": ["b", "c", "d"]}
EVAL_FAIL = {"success": False, "num_tests": 3, "passes": [], "failures": ["x", "y", "z"]}


def _finalize(env, eval_payload=None, raise_eval=False):
    if raise_eval:
        env._post = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("evaluate boom"))
    else:
        env._post = lambda path, payload: {"output": eval_payload}
    env._finalize_reward()


# =============================================================================
# REWARD MODE: partial (dense TGC) vs binary
# =============================================================================


@pytest.mark.parametrize(
    "reward_mode, eval_payload, expected_reward, expected_success, expected_partial",
    [
        ("partial", EVAL_SOLVED, 1.0, 1.0, 1.0),
        ("partial", EVAL_PARTIAL, 0.25, 0.0, 0.25),  # 1/4 tests -> reward == TGC
        ("partial", EVAL_FAIL, 0.0, 0.0, 0.0),
        ("binary", EVAL_SOLVED, 1.0, 1.0, 1.0),
        ("binary", EVAL_PARTIAL, 0.0, 0.0, 0.25),  # not fully solved -> 0 under binary; partial still reported
        ("binary", EVAL_FAIL, 0.0, 0.0, 0.0),
    ],
)
def test_reward_and_metrics(reward_mode, eval_payload, expected_reward, expected_success, expected_partial):
    env = make_env(reward_mode)
    env._completed = True
    env.turns = 7
    _finalize(env, eval_payload)
    assert env._reward == pytest.approx(expected_reward)
    m = env.get_metrics()
    assert m["success"] == pytest.approx(expected_success)
    assert m["partial_completion"] == pytest.approx(expected_partial)
    assert m["eval_unavailable"] == 0.0
    assert m["num_turns"] == 7.0
    # partial reward must equal partial_completion; binary reward must equal success
    if reward_mode == "partial":
        assert env._reward == pytest.approx(m["partial_completion"])
    else:
        assert env._reward == pytest.approx(m["success"])


def test_eval_unavailable_flag():
    """When /evaluate errors, success/partial fall back to 0 and eval_unavailable=1."""
    env = make_env("partial")
    env._completed = False
    env.turns = 40
    _finalize(env, raise_eval=True)
    m = env.get_metrics()
    assert m["eval_unavailable"] == 1.0
    assert m["success"] == 0.0 and m["partial_completion"] == 0.0
    assert env._reward == 0.0


def test_get_metrics_keys():
    """The six keys every AppWorld trajectory reports (become environment/* on W&B).

    `parse_errors` counts turns rejected by strict parsing (no fenced code block); it is the collapse
    signal for format drift, so it must stay in the metric set.
    """
    env = make_env("partial")
    assert set(env.get_metrics().keys()) == {
        "success",
        "partial_completion",
        "num_turns",
        "completed_task_called",
        "eval_unavailable",
        "parse_errors",
    }
    assert env.get_metrics()["parse_errors"] == 0.0


def test_strict_parse_counts_parse_errors_without_executing():
    """strict_parse: a fence-less turn is rejected (no /execute), nudged, and counted in parse_errors."""
    cfg = AppWorldEnvConfig(server_urls=["http://x"], reward_mode="binary", strict_parse=True)
    env = AppWorldEnv(cfg, extras={"reward_spec": {"ground_truth": "t1"}, "max_turns": 2})
    calls = []
    env._post = lambda path, payload: calls.append(path) or {"output": EVAL_FAIL}
    out = env.step("I will now call the api without a fence")
    assert out["done"] is False and out["reward"] == 0.0
    assert out["metadata"].get("parse_error") is True
    assert calls == []  # nothing executed
    assert env.get_metrics()["parse_errors"] == 1.0
    # last turn: still rejected, but the episode is finalized (only /evaluate is posted)
    out = env.step("still no fence")
    assert out["done"] is True and calls == ["/evaluate"]
    assert env.get_metrics()["parse_errors"] == 2.0


def test_aggregate_metrics_averages_across_trajectories():
    """default_aggregate_metrics (inherited) averages each key -> exactly the environment/* values."""
    metrics = []
    for payload, completed, turns in [(EVAL_SOLVED, True, 12), (EVAL_PARTIAL, True, 20), (EVAL_FAIL, False, 40)]:
        env = make_env("partial")
        env._completed = completed
        env.turns = turns
        _finalize(env, payload)
        metrics.append(env.get_metrics())
    agg = default_aggregate_metrics(metrics)
    assert agg["success"] == pytest.approx(1 / 3)  # 1 of 3 solved
    assert agg["partial_completion"] == pytest.approx((1.0 + 0.25 + 0.0) / 3)
    assert agg["completed_task_called"] == pytest.approx(2 / 3)
    assert agg["num_turns"] == pytest.approx((12 + 20 + 40) / 3)
    assert agg["eval_unavailable"] == 0.0


# =============================================================================
# CODE EXTRACTION + step() action handling
# =============================================================================


@pytest.mark.parametrize(
    "action, expected",
    [
        ("print(apis.api_docs.show_app_descriptions())", "print(apis.api_docs.show_app_descriptions())"),  # raw
        ("```python\nx = 1\n```", "x = 1"),  # fenced python
        ("```\ny = 2\n```", "y = 2"),  # bare fence
        ("some prose\n```python\nz = 3\n```\ntrailing", "z = 3"),  # fence amid prose
    ],
)
def test_extract_code(action, expected):
    assert _extract_code(action) == expected


def test_empty_action_nudges_then_finalizes_on_last_turn():
    env = make_env("partial", max_turns=2)
    # turn 1: empty action, not done -> nudge (no reward, not done)
    out = env.step("")
    assert out["done"] is False and out["reward"] == 0.0 and len(out["observations"]) == 1
    # turn 2: empty action, hits max_turns -> finalize (done). Mock /evaluate for finalize.
    env._post = lambda path, payload: {"output": EVAL_PARTIAL}
    out = env.step("")
    assert out["done"] is True
    assert out["reward"] == pytest.approx(0.25)  # partial TGC even on an empty terminal turn

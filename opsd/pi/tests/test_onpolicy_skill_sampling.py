"""What `onpolicy_skill` actually sends to the analyzer: sampling params + thinking.

WHY THIS FILE EXISTS. `onpolicy_skill` is the pi_type the AppWorld online-skill arms run on, and until now NO
test drove it -- the only analyzer fake in the suite (`test_online.py::_install_fake_analyzer`) pins a
fixed keyword signature and nothing exercised the constructor that calls it. So adding an argument to
`Analyzer.analyze` could go green here and still `TypeError` in production, where `_refine_one` SWALLOWS
the exception and returns "error": 144 refine_errors, zero hints, and with `OPSD_ONLINE_HINTS_STRICT`
unset the run trains on nothing. That is the same shape as an earlier silent no-hint failure, where an
arm trained 77 steps with no hints and no error anywhere.

So these tests assert the call CONTRACT (which kwargs reach the analyzer with which values), not just
that a hint comes back.

    .venv/bin/python -m pytest opsd/pi/tests/test_onpolicy_skill_sampling.py -q
"""

from __future__ import annotations

import asyncio

import pytest

from opsd.pi import build_constructor
from opsd.pi.base import TrajView
from opsd.pi.online import OnlineHintConfig, online_hint_cfg_from_env

_CANNED = (
    '{"success_analysis":"logged in first","mistake_analysis":"acted on too many entities",'
    '"golden_workflow":["login","filter","complete_task"]}'
)


class RecordingAnalyzer:
    """Captures every kwarg the constructor passes, and accepts anything (like the real signature)."""

    def __init__(self, reply=_CANNED):
        self.reply = reply
        self.calls = []

    async def analyze(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return self.reply


def _traj(success=True):
    return TrajView(
        task_id="fac291d_3",
        messages=[
            {"role": "user", "content": "Task: pay my share of the utility bill."},
            {"role": "assistant", "content": "<think>check the api docs</think>\napis.venmo.login(...)"},
            {"role": "user", "content": ">> logged in"},
        ],
        reward=1.0 if success else 0.0,
        success=success,
        num_turns=2,
    )


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def _build(**kw):
    az = RecordingAnalyzer()
    ctor = build_constructor("onpolicy_skill", az, **kw)
    return az, ctor


def test_online_path_reaches_the_analyzer_without_raising():
    """build_online falls through to build_offline for this pi_type; both must survive the kwargs."""
    az, ctor = _build(max_tokens=8192, temperature=None, reasoning_effort="high", prompt_env="appworld")
    rec = _run(ctor.build_online("fac291d_3", [_traj()], prior=None))
    assert rec is not None and rec.task, "no hint produced -- _refine_one would report 'no_hint'"
    assert len(az.calls) == 1


def test_temperature_none_is_forwarded_so_the_model_default_applies():
    """None must reach analyze() as None -- Analyzer.analyze then OMITS the param entirely.

    Sending 0.3 instead would silently impose a third, untested sampling config on a frontier analyzer
    that ships tuned defaults.
    """
    az, ctor = _build(max_tokens=8192, temperature=None, reasoning_effort="high")
    _run(ctor.build_offline("t", [_traj()]))
    assert az.calls[0]["temperature"] is None


def test_reasoning_effort_is_forwarded():
    az, ctor = _build(max_tokens=8192, temperature=None, reasoning_effort="high")
    _run(ctor.build_offline("t", [_traj()]))
    assert az.calls[0]["reasoning_effort"] == "high"
    assert az.calls[0]["max_tokens"] == 8192


def test_thinking_off_when_effort_is_none():
    az, ctor = _build(max_tokens=4096, temperature=0.3, reasoning_effort=None)
    _run(ctor.build_offline("t", [_traj()]))
    assert az.calls[0]["reasoning_effort"] is None
    assert az.calls[0]["temperature"] == 0.3, "offline CLI callers keep the historical 0.3 default"


def test_config_forwards_sampling_only_for_onpolicy_skill():
    """constructor_kwargs must NOT hand reasoning_effort to types whose signature rejects it."""
    kw = OnlineHintConfig(pi_type="onpolicy_skill", analyzer_max_tokens=8192).constructor_kwargs()
    assert kw["temperature"] is None and kw["reasoning_effort"] == "high"
    for other in ("contrastive", "reference", "prompt_skill", "group_skill"):
        kw_other = OnlineHintConfig(pi_type=other).constructor_kwargs()
        assert "reasoning_effort" not in kw_other, f"{other} would TypeError on reasoning_effort"
        assert "temperature" not in kw_other


def test_env_unset_temperature_means_model_default(monkeypatch):
    monkeypatch.delenv("OPSD_ONLINE_HINTS_ANALYZER_TEMPERATURE", raising=False)
    monkeypatch.delenv("OPSD_ONLINE_HINTS_ANALYZER_REASONING_EFFORT", raising=False)
    cfg = online_hint_cfg_from_env()
    assert cfg.analyzer_temperature is None
    assert cfg.analyzer_reasoning_effort == "high"


def test_env_explicit_zero_temperature_survives(monkeypatch):
    """0.0 is a DELIBERATE greedy setting; `or None` would silently turn it into 'model default'."""
    monkeypatch.setenv("OPSD_ONLINE_HINTS_ANALYZER_TEMPERATURE", "0.0")
    assert online_hint_cfg_from_env().analyzer_temperature == 0.0
    monkeypatch.setenv("OPSD_ONLINE_HINTS_ANALYZER_TEMPERATURE", "0.7")
    assert online_hint_cfg_from_env().analyzer_temperature == pytest.approx(0.7)


def test_env_can_turn_thinking_off(monkeypatch):
    # pi_type matters: constructor_kwargs only emits the sampling keys for onpolicy_skill, so the default
    # `contrastive` would omit them and make this assertion vacuous.
    monkeypatch.setenv("OPSD_ONLINE_HINTS_PI_TYPE", "onpolicy_skill")
    monkeypatch.setenv("OPSD_ONLINE_HINTS_ANALYZER_REASONING_EFFORT", "")
    cfg = online_hint_cfg_from_env()
    assert cfg.analyzer_reasoning_effort == ""
    assert cfg.constructor_kwargs()["reasoning_effort"] is None  # "" -> omit the param


def test_analyzer_sees_the_agents_reasoning_when_strip_think_is_off():
    """STRIP_THINK=0 is what lets the analyst read the agent's <think>; verify it reaches the prompt."""
    az, ctor = _build(max_tokens=8192, temperature=None, reasoning_effort="high", strip_think=False)
    _run(ctor.build_offline("t", [_traj()]))
    assert "check the api docs" in az.calls[0]["prompt"]

    az2, ctor2 = _build(max_tokens=8192, temperature=None, reasoning_effort="high", strip_think=True)
    _run(ctor2.build_offline("t", [_traj()]))
    assert "check the api docs" not in az2.calls[0]["prompt"]

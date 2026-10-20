"""Tests for the reference PI `trajectory` mode + `strip_think` toggle (pi/reference.py).

The online "self" PI arms render the student's own best successful rollout as a transcript, with the
<think> reasoning either stripped (trajectory-without-thinking) or kept (trajectory-with-thinking).
"""

import asyncio

from opsd.pi.base import TrajView
from opsd.pi.reference import ReferenceHintConstructor


def _traj():
    # One successful rollout: assistant turn has a <think> block + an <action>; plus an observation.
    return TrajView(
        task_id="t1",
        messages=[
            {"role": "user", "content": "Task: put a mug on the desk.\nObservation: you see a mug 1."},
            {
                "role": "assistant",
                "content": "<think>SECRET_REASONING the mug is on counter 1</think><action>take mug 1 from counter 1</action>",
            },
            {"role": "user", "content": "Observation: you pick up the mug 1."},
            {"role": "assistant", "content": "<think>now go to the desk</think><action>move mug 1 to desk 1</action>"},
        ],
        reward=1.0,
        success=True,
        num_turns=2,
    )


def _build(mode, strip_think):
    c = ReferenceHintConstructor(analyzer=None, mode=mode, strip_think=strip_think)
    rec = asyncio.run(c.build_offline("t1", [_traj()]))
    assert rec is not None and rec.task
    return rec


def test_trajectory_without_thinking_strips_think():
    rec = _build("trajectory", strip_think=True)
    assert "SECRET_REASONING" not in rec.task  # reasoning dropped
    assert "take mug 1 from counter 1" in rec.task  # action kept
    assert "move mug 1 to desk 1" in rec.task
    assert rec.meta.get("mode") == "trajectory" and rec.meta.get("strip_think") is True


def test_trajectory_with_thinking_keeps_think():
    rec = _build("trajectory", strip_think=False)
    assert "SECRET_REASONING" in rec.task  # reasoning kept
    assert "take mug 1 from counter 1" in rec.task
    assert rec.meta.get("strip_think") is False


def test_actions_mode_unchanged():
    # default actions mode still yields the ordered action list (no regression)
    rec = _build("actions", strip_think=True)
    assert "take mug 1 from counter 1" in rec.task and "move mug 1 to desk 1" in rec.task
    assert "SECRET_REASONING" not in rec.task
    assert rec.meta.get("mode") == "actions"

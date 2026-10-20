"""
Fork parity for ``opsd.skyrl_ext.tracking.OPSDTracking``: replays opsd/tests/fixtures/fork_parity/runtime/tracking.json,
captured from the forked SkyRL's ``Tracking.log`` (skyrl/train/utils/tracking.py), with fake backends.

pytest opsd/skyrl_ext/tests/test_tracking_fork_parity.py
"""

import json

import pytest

from opsd._repo import fixtures_dir
from opsd.skyrl_ext.tracking import OPSDTracking

FIXTURES = fixtures_dir() / "fork_parity" / "runtime"
TRACKING = json.loads((FIXTURES / "tracking.json").read_text())


class FakeWandb:
    def __init__(self):
        self.calls = []

    def log(self, data, step, commit=False):
        self.calls.append({"data": dict(data), "step": step, "commit": commit})

    def finish(self, exit_code=0):
        pass


class FakeBackend:
    def __init__(self):
        self.calls = []

    def log(self, data, step):
        self.calls.append({"data": dict(data), "step": step})

    def finish(self):
        pass


@pytest.mark.parametrize("case", TRACKING["cases"], ids=lambda c: c["name"])
def test_step_override_matches_the_fork(case, monkeypatch):
    tracker = OPSDTracking(project_name="synth-project", experiment_name="synth-run", backends=[], config=None)
    wandb, tensorboard = FakeWandb(), FakeBackend()
    tracker.logger["wandb"] = wandb
    tracker.logger["tensorboard"] = tensorboard
    monkeypatch.delenv("SKYRL_LOG_STEP_OVERRIDE", raising=False)
    if case["override"] is not None:
        monkeypatch.setenv("SKYRL_LOG_STEP_OVERRIDE", case["override"])

    for call in case["calls"]:
        kwargs = {"commit": call["commit"]} if "commit" in call else {}
        tracker.log(dict(call["data"]), step=call["step"], **kwargs)

    assert {"wandb": wandb.calls, "tensorboard": tensorboard.calls} == case["received"]

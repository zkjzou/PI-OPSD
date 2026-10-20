"""
Unit tests for ``opsd.skyrl_ext.tracking``: the SKYRL_LOG_STEP_OVERRIDE step override and the ``get_tracker``
drop-in for ``BasePPOExp.get_tracker``. No tracking backend is contacted.

pytest opsd/skyrl_ext/tests/test_tracking.py
"""

import inspect
from types import SimpleNamespace

from opsd.skyrl_ext import tracking
from opsd.skyrl_ext.tracking import OPSDTracking, get_tracker, log_step_override
from skyrl.train.utils.tracking import ConsoleLogger, Tracking


class RecordingBackend:
    def __init__(self):
        self.calls = []

    def log(self, data, step):
        self.calls.append((dict(data), step))

    def finish(self):
        pass


def _tracker(cls=OPSDTracking):
    tracker = cls(project_name="p", experiment_name="e", backends=[], config=None)
    backend = RecordingBackend()
    tracker.logger["tensorboard"] = backend
    return tracker, backend


def test_is_a_tracking_with_the_same_log_signature():
    assert issubclass(OPSDTracking, Tracking)
    assert inspect.signature(OPSDTracking.log) == inspect.signature(Tracking.log)


def test_unset_override_keeps_the_step(monkeypatch):
    monkeypatch.delenv("SKYRL_LOG_STEP_OVERRIDE", raising=False)
    tracker, backend = _tracker()
    tracker.log({"a": 1}, step=3)
    assert backend.calls == [({"a": 1}, 3)]


def test_override_is_read_on_every_call(monkeypatch):
    tracker, backend = _tracker()
    monkeypatch.setenv("SKYRL_LOG_STEP_OVERRIDE", "40")
    tracker.log({"a": 1}, step=3)
    monkeypatch.setenv("SKYRL_LOG_STEP_OVERRIDE", "not-an-int")
    tracker.log({"a": 2}, step=4)
    monkeypatch.delenv("SKYRL_LOG_STEP_OVERRIDE")
    tracker.log({"a": 3}, step=5)
    assert backend.calls == [({"a": 1}, 40), ({"a": 2}, 4), ({"a": 3}, 5)]


def test_log_step_override_helper(monkeypatch):
    monkeypatch.setenv("SKYRL_LOG_STEP_OVERRIDE", " 7 ")
    assert log_step_override(1) == 7
    monkeypatch.setenv("SKYRL_LOG_STEP_OVERRIDE", "")
    assert log_step_override(1) == 1


def test_cooperative_subclass_still_gets_the_override(monkeypatch):
    # e.g. AppWorldTracking: adds aliased keys, then calls super().log.
    class AliasingTracking(OPSDTracking):
        def log(self, data, step, commit=False):
            super().log({**data, "Alias/a": data["a"]}, step=step, commit=commit)

    monkeypatch.setenv("SKYRL_LOG_STEP_OVERRIDE", "99")
    tracker, backend = _tracker(AliasingTracking)
    tracker.log({"a": 1}, step=3)
    assert backend.calls == [({"a": 1, "Alias/a": 1}, 99)]


def test_get_tracker_builds_an_opsd_tracking_from_the_exp_cfg():
    cfg = SimpleNamespace(trainer=SimpleNamespace(project_name="proj", run_name="run", logger="console"))
    tracker = get_tracker(SimpleNamespace(cfg=cfg))
    assert type(tracker) is OPSDTracking
    assert list(tracker.logger) == ["console"] and isinstance(tracker.logger["console"], ConsoleLogger)


def test_get_tracker_binds_as_an_exp_method(monkeypatch):
    seen = {}

    class Recording(OPSDTracking):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.logger = {}

    monkeypatch.setattr(tracking, "OPSDTracking", Recording)
    cfg = SimpleNamespace(trainer=SimpleNamespace(project_name="proj", run_name="run", logger=["wandb"]))

    class Exp:
        get_tracker = tracking.get_tracker

        def __init__(self):
            self.cfg = cfg

    assert isinstance(Exp().get_tracker(), Recording)
    assert seen == {"project_name": "proj", "experiment_name": "run", "backends": ["wandb"], "config": cfg}
    assert list(inspect.signature(tracking.get_tracker).parameters) == ["self"]

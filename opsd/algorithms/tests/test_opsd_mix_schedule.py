"""Tests for the OPSD_MIX_LAMBDA step schedule (opsd/trainer/mix_schedule.py).

The properties worth pinning: an unset schedule is EXACTLY the old constant (so every existing lambda arm
is unchanged), the endpoints land where the knobs say, and the post-window value really is 1.0 when the
arm is meant to become plain DAPO.
"""

import importlib.util

import pytest

from opsd._repo import repo_root

REPO = repo_root(__file__)
MODULE = REPO / "opsd/trainer/mix_schedule.py"
TRAINER = REPO / "opsd/trainer/main_opsd_alfworld.py"
KNOBS = (
    "OPSD_MIX_LAMBDA",
    "OPSD_MIX_LAMBDA_FINAL",
    "OPSD_MIX_LAMBDA_DECAY_STEPS",
    "OPSD_MIX_LAMBDA_AFTER",
)


def _load():
    spec = importlib.util.spec_from_file_location("opsd_mix_schedule", MODULE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ms = _load()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)


def test_unset_is_pure_grpo():
    assert ms.mix_lambda(0) == 1.0
    assert ms.mix_lambda(150) == 1.0
    assert ms.mix_lambda_configured() is False


def test_constant_lambda_unchanged(monkeypatch):
    monkeypatch.setenv("OPSD_MIX_LAMBDA", "0.9")
    for step in (0, 1, 50, 100, 150):
        assert ms.mix_lambda(step) == pytest.approx(0.9)
    assert ms.mix_lambda_configured() is True


def test_ramp_then_plain_dapo(monkeypatch):
    monkeypatch.setenv("OPSD_MIX_LAMBDA", "0.5")
    monkeypatch.setenv("OPSD_MIX_LAMBDA_FINAL", "0.99")
    monkeypatch.setenv("OPSD_MIX_LAMBDA_DECAY_STEPS", "100")
    monkeypatch.setenv("OPSD_MIX_LAMBDA_AFTER", "1.0")
    assert ms.mix_lambda(0) == pytest.approx(0.5)
    assert ms.mix_lambda(1) == pytest.approx(0.5049)
    assert ms.mix_lambda(50) == pytest.approx(0.745)
    assert ms.mix_lambda(100) == pytest.approx(0.99)
    assert ms.mix_lambda(101) == 1.0
    assert ms.mix_lambda(150) == 1.0
    assert ms.mix_lambda_configured() is True


def test_after_defaults_to_final(monkeypatch):
    monkeypatch.setenv("OPSD_MIX_LAMBDA", "0.5")
    monkeypatch.setenv("OPSD_MIX_LAMBDA_FINAL", "0.9")
    monkeypatch.setenv("OPSD_MIX_LAMBDA_DECAY_STEPS", "10")
    assert ms.mix_lambda(10) == pytest.approx(0.9)
    assert ms.mix_lambda(500) == pytest.approx(0.9)


def test_ramp_down_from_one_still_counts_as_configured(monkeypatch):
    # starts at lam=1 (no mixing at step 0) but mixes later, so the teacher tensor must still be shipped
    monkeypatch.setenv("OPSD_MIX_LAMBDA", "1.0")
    monkeypatch.setenv("OPSD_MIX_LAMBDA_FINAL", "0.5")
    monkeypatch.setenv("OPSD_MIX_LAMBDA_DECAY_STEPS", "50")
    assert ms.mix_lambda(0) == 1.0
    assert ms.mix_lambda_configured() is True


@pytest.mark.parametrize("bad", ["-0.1", "1.5"])
def test_out_of_range_rejected(monkeypatch, bad):
    monkeypatch.setenv("OPSD_MIX_LAMBDA", "0.5")
    monkeypatch.setenv("OPSD_MIX_LAMBDA_DECAY_STEPS", "10")
    monkeypatch.setenv("OPSD_MIX_LAMBDA_FINAL", bad)
    with pytest.raises(ValueError, match="OPSD_MIX_LAMBDA_FINAL"):
        ms.mix_lambda(5)


def test_trainer_reads_the_schedule_not_the_raw_env():
    # the mix site must go through mix_lambda(global_step); a raw env read would silently ignore the schedule
    src = TRAINER.read_text()
    assert 'mix_lambda(int(getattr(self, "global_step", 0) or 0))' in src
    assert 'lam = float(os.environ.get("OPSD_MIX_LAMBDA"' not in src

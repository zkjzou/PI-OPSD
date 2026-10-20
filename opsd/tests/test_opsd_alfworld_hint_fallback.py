"""OPSD_NO_STATIC_FALLBACK: a task absent from the hints file gets NO hint, not the generic static one.

Only the hint lookup is exercised; the generator's __init__ needs a full SkyRL setup, so the three
attributes the lookup reads are set by hand.
"""

import pytest

mod = pytest.importorskip("opsd.trainer.main_opsd_alfworld")


def _gen(no_fallback: bool):
    g = object.__new__(mod.SelfDistillSkyRLGymGenerator)
    g._hints_by_task = {"trial_A": "reference for A"}
    g._static_hint = "STATIC"
    g._no_static_fallback = no_fallback
    return g


@pytest.mark.parametrize("no_fallback", [False, True])
def test_task_in_file_gets_its_hint(no_fallback):
    g = _gen(no_fallback)
    assert g._hint_for_task_id("trial_A") == "reference for A"
    assert g._hint_for_task_id("trial_A__suffix") == "reference for A"


def test_missing_task_falls_back_to_static_by_default():
    g = _gen(False)
    assert g._hint_for_task_id("trial_B") == "STATIC"
    assert g._hint_for_task_id(None) == "STATIC"


def test_missing_task_gets_no_hint_when_fallback_off():
    g = _gen(True)
    assert g._hint_for_task_id("trial_B") == ""
    assert g._hint_for_task_id(None) == ""

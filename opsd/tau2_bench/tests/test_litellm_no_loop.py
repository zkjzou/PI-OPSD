"""litellm must not create an event loop per sync completion inside τ² (the 2026-09-24 segfault site).

A λ=0.99 checklist-hint DAPO×OPSD arm died at step 42 with `Fatal Python error: Segmentation fault` inside
`asyncio.run`, called from `litellm/_service_logger.py:service_success_hook`, which litellm's
`track_llm_api_timing` runs after EVERY sync provider call. From a worker thread there is no loop, so the hook
creates one per call -- to run a coroutine that does nothing when `litellm.service_callback` is empty.
`source._disable_litellm_per_call_event_loop` short-circuits exactly that case.
"""

import asyncio
import sys
import threading
from pathlib import Path

import pytest

# The repository root: the first ancestor holding pyproject.toml, opsd/__init__.py and opsd/skyrl_ext/__init__.py
# (the predicate of opsd._repo; never a fixed parents[N]: the depth of this file changes when the package moves).
# Not found -> rely on PYTHONPATH as-is.
_REPO = next(
    (
        p
        for p in Path(__file__).resolve().parents
        if (p / "pyproject.toml").is_file()
        and (p / "opsd/__init__.py").is_file()
        and (p / "opsd/skyrl_ext/__init__.py").is_file()
    ),
    None,
)
if _REPO is not None:
    sys.path.insert(0, str(_REPO))

import litellm  # noqa: E402
from litellm._service_logger import ServiceLogging  # noqa: E402
from litellm.litellm_core_utils.logging_utils import track_llm_api_timing  # noqa: E402
from litellm.types.services import ServiceTypes  # noqa: E402

from opsd.tau2_bench import source  # noqa: E402


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    # the patch is process-global; put the class method (and the callback list) back after every test
    monkeypatch.setattr(ServiceLogging, "service_success_hook", ServiceLogging.service_success_hook)
    monkeypatch.setattr(litellm, "service_callback", [])


def _count_loops(monkeypatch):
    calls = []
    real = asyncio.run

    def counting_run(coro, *a, **k):
        calls.append(1)
        return real(coro, *a, **k)

    monkeypatch.setattr(asyncio, "run", counting_run)
    return calls


def _hook_from_thread():
    t = threading.Thread(
        target=lambda: ServiceLogging().service_success_hook(service=ServiceTypes.LITELLM, duration=0.1, call_type="t")
    )
    t.start()
    t.join()


def test_unpatched_hook_creates_a_loop_per_call_from_a_thread(monkeypatch):
    # documents the mechanism: this is what every τ² model call did before the patch
    calls = _count_loops(monkeypatch)
    _hook_from_thread()
    assert len(calls) == 1


def test_patched_hook_creates_no_loop_when_no_service_callback(monkeypatch):
    assert source._disable_litellm_per_call_event_loop() is True
    calls = _count_loops(monkeypatch)
    _hook_from_thread()
    assert calls == []


def test_patched_hook_still_runs_the_original_when_a_callback_is_configured(monkeypatch):
    source._disable_litellm_per_call_event_loop()
    monkeypatch.setattr(litellm, "service_callback", ["some_callback_nobody_matches"])
    calls = _count_loops(monkeypatch)
    _hook_from_thread()
    assert len(calls) == 1  # original behaviour, unchanged


def test_patch_is_idempotent():
    assert source._disable_litellm_per_call_event_loop() is True
    first = ServiceLogging.service_success_hook
    assert source._disable_litellm_per_call_event_loop() is True
    assert ServiceLogging.service_success_hook is first


def test_real_litellm_sync_wrapper_from_64_threads_creates_no_loop(monkeypatch):
    # end-to-end through litellm's own decorator, the exact wrapper in the crash stack
    source._disable_litellm_per_call_event_loop()
    calls = _count_loops(monkeypatch)

    @track_llm_api_timing()
    def fake_provider_completion(**kwargs):
        return "ok"

    out = []
    ts = [threading.Thread(target=lambda: out.extend(fake_provider_completion() for _ in range(50))) for _ in range(64)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(out) == 64 * 50 and calls == []


def test_bootstrap_installs_it():
    src = Path(source.__file__).read_text()
    i = src.index("def bootstrap_tau2(")
    assert "_disable_litellm_per_call_event_loop()" in src[i:], "bootstrap_tau2 must install the patch"

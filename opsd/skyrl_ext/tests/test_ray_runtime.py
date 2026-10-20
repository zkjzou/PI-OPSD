"""
Unit tests for ``opsd.skyrl_ext.ray_runtime.initialize_ray``: the optional SKYRL_RAY_NUM_CPUS cap and drift
guards against upstream ``skyrl.train.utils.utils.initialize_ray``, whose body it copies. Ray, the registries
and the peer-access probe are stubbed; no Ray cluster is started.

pytest opsd/skyrl_ext/tests/test_ray_runtime.py
"""

import hashlib
import inspect
import os
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace

import pytest

import skyrl.train.utils.utils as train_utils
from opsd.skyrl_ext import ray_runtime
from skyrl.backends.skyrl_train.utils import ppo_utils
from skyrl.train.config.config import SkyRLTrainConfig

# sha256 of inspect.getsource(skyrl.train.utils.utils.initialize_ray) at SkyRL ecdaa946.
UPSTREAM_INITIALIZE_RAY_SHA256 = "529cdef1b10519ab6a7f702c1db9fb2ad86eec6d08065cbed441525653094476"


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 2, 3, 4, 5)


class TestInitializeRay:
    @staticmethod
    def _stub_dependencies(monkeypatch, init_calls):
        fake_ray = SimpleNamespace(
            is_initialized=lambda: False,
            init=lambda **kwargs: init_calls.append(kwargs),
        )
        monkeypatch.setattr(ray_runtime, "ray", fake_ray)
        monkeypatch.setattr(
            ray_runtime,
            "prepare_runtime_environment",
            lambda _cfg: {"TEST_ENV": "1"},
        )
        monkeypatch.setattr(ppo_utils, "sync_registries", lambda: None)
        monkeypatch.setattr(ray_runtime, "SKYRL_DUMP_INFRA_LOG_TO_STDOUT", True)

    def test_optional_cpu_cap_is_forwarded_to_ray(self, monkeypatch):
        init_calls = []
        self._stub_dependencies(monkeypatch, init_calls)
        monkeypatch.setenv("SKYRL_RAY_NUM_CPUS", "16")

        ray_runtime.initialize_ray(SimpleNamespace())

        assert init_calls == [
            {
                "runtime_env": {"env_vars": {"TEST_ENV": "1"}},
                "log_to_driver": True,
                "num_cpus": 16,
            }
        ]

    @pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
    def test_invalid_cpu_cap_is_rejected(self, monkeypatch, value):
        init_calls = []
        self._stub_dependencies(monkeypatch, init_calls)
        monkeypatch.setenv("SKYRL_RAY_NUM_CPUS", value)

        with pytest.raises(ValueError, match="SKYRL_RAY_NUM_CPUS"):
            ray_runtime.initialize_ray(SimpleNamespace())

        assert init_calls == []

    @pytest.mark.parametrize("value", [None, ""])
    def test_unset_or_empty_cpu_cap_passes_upstream_kwargs_only(self, monkeypatch, value):
        init_calls = []
        self._stub_dependencies(monkeypatch, init_calls)
        monkeypatch.delenv("SKYRL_RAY_NUM_CPUS", raising=False)
        if value is not None:
            monkeypatch.setenv("SKYRL_RAY_NUM_CPUS", value)

        ray_runtime.initialize_ray(SimpleNamespace())

        assert init_calls == [{"runtime_env": {"env_vars": {"TEST_ENV": "1"}}, "log_to_driver": True}]

    @pytest.mark.parametrize("cpus", [None, "16"])
    def test_attach_to_existing_cluster_kwargs_do_not_depend_on_ray_address(self, monkeypatch, cpus):
        # Like the forked SkyRL, initialize_ray never reads RAY_ADDRESS: ray.init resolves it itself, and the
        # launchers unset SKYRL_RAY_NUM_CPUS in external mode (Ray refuses num_cpus when attaching).
        calls = {}
        for address in ("local", "192.0.2.10:6379"):
            init_calls = []
            self._stub_dependencies(monkeypatch, init_calls)
            monkeypatch.setenv("RAY_ADDRESS", address)
            monkeypatch.delenv("SKYRL_RAY_NUM_CPUS", raising=False)
            if cpus is not None:
                monkeypatch.setenv("SKYRL_RAY_NUM_CPUS", cpus)
            ray_runtime.initialize_ray(SimpleNamespace())
            calls[address] = init_calls
        assert calls["local"] == calls["192.0.2.10:6379"]
        assert "address" not in calls["local"][0]
        assert ("num_cpus" in calls["local"][0]) == (cpus is not None)

    def test_uses_upstream_prepare_runtime_environment_at_call_time(self, monkeypatch):
        # The wrapper extends whatever upstream returns; it does not re-implement it.
        monkeypatch.setattr(train_utils, "prepare_runtime_environment", lambda _cfg: {"UPSTREAM_ONLY": "1"})
        for name in list(os.environ):
            if (
                name.startswith(ray_runtime.RUNTIME_ENV_FORWARD_PREFIXES)
                or name in ray_runtime.RUNTIME_ENV_FORWARD_NAMES
                or name in ray_runtime.RUNTIME_ENV_EXPLICIT_NAMES
            ):
                monkeypatch.delenv(name)
        monkeypatch.setenv("OPSD_HINTS_PATH", "hints.json")
        monkeypatch.setenv("WANDB_RUN_ID", "run01")

        assert ray_runtime.prepare_runtime_environment(SimpleNamespace()) == {
            "UPSTREAM_ONLY": "1",
            "WANDB_RUN_ID": "run01",
            "OPSD_HINTS_PATH": "hints.json",
        }


def test_copied_upstream_initialize_ray_is_unchanged():
    """initialize_ray here is a copy of upstream's; a SkyRL bump that changes it must be re-synced by hand."""
    source = inspect.getsource(train_utils.initialize_ray)
    assert hashlib.sha256(source.encode()).hexdigest() == UPSTREAM_INITIALIZE_RAY_SHA256, (
        "skyrl.train.utils.utils.initialize_ray changed upstream: port the change into "
        "opsd.skyrl_ext.ray_runtime.initialize_ray, then update UPSTREAM_INITIALIZE_RAY_SHA256"
    )


@pytest.fixture
def isolated_environ():
    """Restore os.environ exactly after the test (initialize_ray sets SKYRL_LOG_FILE / RAY_BACKEND_LOG_LEVEL)."""
    saved = dict(os.environ)
    yield os.environ
    os.environ.clear()
    os.environ.update(saved)


@pytest.mark.parametrize("verbose", [True, False])
def test_matches_upstream_initialize_ray_without_opsd_knobs(monkeypatch, tmp_path, isolated_environ, verbose):
    """With nothing to forward and no CPU cap, ours and upstream's make the same ray.init call and the same
    os.environ / filesystem side effects."""
    for name in list(isolated_environ):
        if (
            name.startswith(ray_runtime.RUNTIME_ENV_FORWARD_PREFIXES)
            or name in ray_runtime.RUNTIME_ENV_FORWARD_NAMES
            or name in ray_runtime.RUNTIME_ENV_EXPLICIT_NAMES
        ):
            del isolated_environ[name]
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **_: True)
    for module in (train_utils, ray_runtime):
        monkeypatch.setattr(module, "SKYRL_DUMP_INFRA_LOG_TO_STDOUT", verbose)
        monkeypatch.setattr(module, "datetime", _FixedDatetime)

    results = {}
    for name, module in (("upstream", train_utils), ("opsd", ray_runtime)):
        init_calls, sync_calls = [], []
        monkeypatch.setattr(module, "ray", SimpleNamespace(init=lambda **kw: init_calls.append(kw)))
        monkeypatch.setattr(ppo_utils, "sync_registries", lambda: sync_calls.append(True))
        isolated_environ.pop("SKYRL_LOG_FILE", None)
        isolated_environ.pop("RAY_BACKEND_LOG_LEVEL", None)
        cfg = SkyRLTrainConfig()
        cfg.trainer.log_path = str(tmp_path / name)
        module.initialize_ray(cfg)
        results[name] = {
            "init_calls": init_calls,
            "sync_calls": len(sync_calls),
            "environ": {k: isolated_environ.get(k) for k in ("SKYRL_LOG_FILE", "RAY_BACKEND_LOG_LEVEL")},
            "log_dir": (tmp_path / name).is_dir(),
        }
    upstream_dir, opsd_dir = str((tmp_path / "upstream").resolve()), str((tmp_path / "opsd").resolve())
    assert repr(results["opsd"]) == repr(results["upstream"]).replace(upstream_dir, opsd_dir)
    assert results["opsd"]["log_dir"] is (not verbose)


def test_module_import_is_light():
    """scripts/tau2/lib.sh's key probe imports collect_forwarded_env_vars; that must not pull in torch."""
    code = "import sys; import opsd.skyrl_ext.ray_runtime; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=300)
    assert out.stdout.strip().splitlines()[-1] == "False"

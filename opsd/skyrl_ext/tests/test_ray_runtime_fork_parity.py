"""
Fork parity for ``opsd.skyrl_ext.ray_runtime``: replays the goldens in opsd/tests/fixtures/fork_parity/runtime/,
captured from the forked SkyRL (skyrl/train/utils/utils.py), against the wrapper over unmodified SkyRL.

For every synthetic environ fixture the wrapper must give the fork's runtime-env dict, ray.init kwargs, raised
errors, os.environ / log-dir side effects and warnings. Info log lines may differ only as documented in the
fixtures' README: their order (our lines follow upstream's), and SKYRL_WANDB_N_GENERATIONS, which is no longer
an explicitly forwarded name (it still reaches the runtime env through the SKYRL_ prefix).

pytest opsd/skyrl_ext/tests/test_ray_runtime_fork_parity.py
"""

import json
import os
import re
from collections import Counter
from datetime import datetime
from types import SimpleNamespace

import pytest
from loguru import logger

import skyrl.train.utils.utils as train_utils
from opsd._repo import fixtures_dir
from opsd.skyrl_ext import ray_runtime
from skyrl.backends.skyrl_train.utils import ppo_utils
from skyrl.train.config.config import SkyRLTrainConfig

FIXTURES = fixtures_dir() / "fork_parity" / "runtime"
FORWARDING = json.loads((FIXTURES / "forwarding.json").read_text())
PREPARE = json.loads((FIXTURES / "prepare_runtime_environment.json").read_text())
INIT = json.loads((FIXTURES / "initialize_ray.json").read_text())

# Names the fork forwarded explicitly but the wrapper does not (the W&B rollout viewer that read it is dropped).
DROPPED_EXPLICIT_NAMES = ("SKYRL_WANDB_N_GENERATIONS",)
_SUMMARY_RE = re.compile(r"^Exporting (\d+) prefixed env vars to ray runtime env: (.*)$")


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(*INIT["meta"]["fixed_now"])


@pytest.fixture
def replace_environ(monkeypatch):
    """Replace os.environ wholesale by a fixture's environ; the original is restored after the test."""
    saved = dict(os.environ)

    def _replace(environ):
        os.environ.clear()
        os.environ.update(environ)

    yield _replace
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def log_records():
    records = []
    sink = logger.add(
        lambda m: records.append({"level": m.record["level"].name, "message": m.record["message"]}), level="INFO"
    )
    yield records
    logger.remove(sink)


def _apply_module_flags(monkeypatch, flags):
    for name, value in {**PREPARE["meta"]["default_module_flags"], **flags}.items():
        monkeypatch.setattr(train_utils, name, value)


def _make_cfg(overrides):
    cfg = SkyRLTrainConfig()
    for dotted, value in overrides.items():
        obj = cfg
        *parents, leaf = dotted.split(".")
        for part in parents:
            obj = getattr(obj, part)
        setattr(obj, leaf, value)
    return cfg


def _split_logs(records):
    """(warnings in order, Counter of other info lines, set of names in the prefixed-forwarding summary line)."""
    warnings, infos, summary = [], Counter(), set()
    for record in records:
        if record["level"] == "WARNING":
            warnings.append(record["message"])
            continue
        match = _SUMMARY_RE.match(record["message"])
        if match:
            names = match.group(2).split(", ")
            assert int(match.group(1)) == len(names)
            summary |= set(names)
        else:
            infos[record["message"]] += 1
    return warnings, infos, summary


def _expected_logs(case):
    """The fork's log lines, adjusted for the two documented differences of the wrapper."""
    warnings, infos, summary = _split_logs(case["logs"])
    for name in DROPPED_EXPLICIT_NAMES:
        for line in [line for line in infos if line.startswith(f"Exporting {name}=")]:
            del infos[line]
            summary.add(name)  # now picked up by the prefix rule instead
    # The fork forwarded by prefix BEFORE upstream copied SKYRL_RAY_PG_TIMEOUT_IN_S, so it listed it as
    # forwarded; the wrapper runs after upstream, which already set it (to the same value).
    if case["environ"].get("SKYRL_RAY_PG_TIMEOUT_IN_S"):
        summary.discard("SKYRL_RAY_PG_TIMEOUT_IN_S")
    return warnings, infos, summary


def test_forwarding_constants_match_the_fork():
    constants = FORWARDING["constants"]
    assert list(ray_runtime.RUNTIME_ENV_FORWARD_PREFIXES) == constants["RUNTIME_ENV_FORWARD_PREFIXES"]
    assert sorted(ray_runtime.RUNTIME_ENV_FORWARD_NAMES) == constants["RUNTIME_ENV_FORWARD_NAMES"]
    assert list(ray_runtime.RUNTIME_ENV_SECRET_MARKERS) == constants["RUNTIME_ENV_SECRET_MARKERS"]
    assert sorted(ray_runtime.RUNTIME_ENV_MARKER_ALLOWLIST) == constants["RUNTIME_ENV_MARKER_ALLOWLIST"]
    assert list(ray_runtime.RUNTIME_ENV_KEY_POINTERS) == constants["RUNTIME_ENV_KEY_POINTERS"]


def test_is_secret_env_name_matches_the_fork():
    assert {name: ray_runtime.is_secret_env_name(name) for name in FORWARDING["is_secret_env_name"]} == (
        FORWARDING["is_secret_env_name"]
    )


@pytest.mark.parametrize("case", FORWARDING["collect_forwarded_env_vars"], ids=lambda c: c["name"])
def test_collect_forwarded_env_vars_matches_the_fork(case):
    warnings = []
    sink = logger.add(lambda m: warnings.append(m.record["message"]), level="WARNING")
    try:
        forwarded = ray_runtime.collect_forwarded_env_vars(dict(case["environ"]))
    finally:
        logger.remove(sink)
    assert list(forwarded.items()) == list(case["forwarded"].items())
    assert warnings == case["warnings"]


@pytest.mark.parametrize("case", PREPARE["cases"], ids=lambda c: c["name"])
def test_prepare_runtime_environment_matches_the_fork(case, monkeypatch, replace_environ, log_records):
    _apply_module_flags(monkeypatch, case["module_flags"])
    peer = case["peer_access_supported"]
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **_: peer)
    cfg = _make_cfg(case["cfg"])
    replace_environ(case["environ"])

    env_vars = ray_runtime.prepare_runtime_environment(cfg)

    assert env_vars == case["env_vars"]
    assert _split_logs(log_records) == _expected_logs(case)


@pytest.mark.parametrize("case", INIT["cases"], ids=lambda c: c["name"])
def test_initialize_ray_matches_the_fork(case, monkeypatch, tmp_path, replace_environ, log_records):
    init_calls, sync_calls = [], []
    log_path = (tmp_path / "skyrl-logs").resolve()
    _apply_module_flags(monkeypatch, {})
    monkeypatch.setattr(ray_runtime, "SKYRL_DUMP_INFRA_LOG_TO_STDOUT", case["verbose"])
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **_: True)
    monkeypatch.setattr(
        ray_runtime,
        "ray",
        SimpleNamespace(
            is_initialized=lambda: False, init=lambda **kwargs: init_calls.append(json.loads(json.dumps(kwargs)))
        ),
    )
    monkeypatch.setattr(ppo_utils, "sync_registries", lambda: sync_calls.append(True))
    monkeypatch.setattr(ray_runtime, "datetime", _FixedDatetime)
    cfg = _make_cfg({"trainer.log_path": str(log_path)})
    replace_environ(case["environ"])

    error = None
    try:
        ray_runtime.initialize_ray(cfg)
    except Exception as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "cause": type(exc.__cause__).__name__ if exc.__cause__ is not None else None,
        }
    after = dict(os.environ)

    def normalise(obj):
        return json.loads(json.dumps(obj).replace(json.dumps(str(log_path))[1:-1], "<LOG_PATH>"))

    before = case["environ"]
    environ_changes = {
        "set": {k: v for k, v in sorted(after.items()) if before.get(k) != v},
        "removed": sorted(k for k in before if k not in after),
    }
    assert normalise(init_calls) == case["ray_init_calls"]
    assert error == case["error"]
    assert normalise(environ_changes) == case["environ_changes"]
    assert log_path.is_dir() == case["log_dir_created"]
    assert len(sync_calls) == case["sync_registries_calls"]
    assert _split_logs(normalise(log_records)) == _expected_logs(case)


def test_goldens_cover_the_requested_fixtures():
    """Guard against a re-capture silently dropping a scenario."""
    prepare = {case["name"] for case in PREPARE["cases"]}
    init = {case["name"] for case in INIT["cases"]}
    assert {
        "prefix_vars",
        "credential_names",
        "allowlist_vars",
        "allowlist_vars_empty",
        "empty_values",
        "key_pointer_invalid_value",
    } <= prepare
    assert {
        "cpus_unset_verbose",
        "cpus_16_verbose",
        "cpus_not_an_integer_verbose",
        "cpus_zero_verbose",
        "cpus_invalid_nonverbose",
        "ray_address_external_verbose",
        "ray_address_external_with_cpus",
    } <= init
    assert any(case["error"] for case in INIT["cases"])
    assert any(case["warnings"] for case in FORWARDING["collect_forwarded_env_vars"])

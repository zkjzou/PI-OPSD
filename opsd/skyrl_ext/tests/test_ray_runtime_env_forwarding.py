"""
Unit tests for the prefix-based env-var forwarding into the Ray runtime env
(``collect_forwarded_env_vars`` / ``prepare_runtime_environment``). No Ray cluster is started.

pytest opsd/skyrl_ext/tests/test_ray_runtime_env_forwarding.py
"""

import os

import pytest

import skyrl.train.utils.utils as train_utils
from opsd.skyrl_ext import ray_runtime
from opsd.skyrl_ext.ray_runtime import collect_forwarded_env_vars, is_secret_env_name
from skyrl.train.config.config import SkyRLTrainConfig


@pytest.fixture
def clean_env(monkeypatch):
    """Drop every ambient variable the forwarder would pick up, so assertions are exact."""
    for name in list(os.environ):
        if name.startswith(ray_runtime.RUNTIME_ENV_FORWARD_PREFIXES) or name in ray_runtime.RUNTIME_ENV_FORWARD_NAMES:
            monkeypatch.delenv(name, raising=False)
    for name in ray_runtime.RUNTIME_ENV_EXPLICIT_NAMES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_forwards_prefixed_names_and_pythonhashseed():
    environ = {
        "OPSD_HINTS_PATH": "/data/hints.json",
        "ALFWORLD_BASE_PORT": "8200",
        "APPWORLD_ROOT": "/data/appworld",
        "TAU2_REVISION": "abc",
        "SKYRL_EVAL_COMPACT": "1",
        "PYTHONHASHSEED": "0",
        # not forwarded: no prefix, prefix not at the start, wrong case
        "HOME": "/home/user",
        "PATH": "/usr/bin",
        "XOPSD_FOO": "1",
        "opsd_lower": "1",
        "PYTHONPATH": "/somewhere",
    }
    assert collect_forwarded_env_vars(environ) == {
        "ALFWORLD_BASE_PORT": "8200",
        "APPWORLD_ROOT": "/data/appworld",
        "OPSD_HINTS_PATH": "/data/hints.json",
        "PYTHONHASHSEED": "0",
        "SKYRL_EVAL_COMPACT": "1",
        "TAU2_REVISION": "abc",
    }


@pytest.mark.parametrize(
    "name",
    [
        "OPSD_PI_ANALYZER_API_KEY",
        "OPSD_ONLINE_HINTS_ANALYZER_API_KEY",
        "TAU2_JUDGE_TOKEN",
        "APPWORLD_SECRET",
        "ALFWORLD_PASSWORD",
        "SKYRL_DATABASE_URL",
        "OPSD_AWS_CREDENTIALS",
        "opsd_api_key",
    ],
)
def test_secret_looking_names_are_never_forwarded(name):
    assert is_secret_env_name(name)
    assert collect_forwarded_env_vars({name: "sk-not-a-real-key"}) == {}


@pytest.mark.parametrize(
    "name",
    [
        "OPSD_TOKEN_CLIP",
        "OPSD_TOKEN_ADV",
        "OPSD_TOKEN_ADV_COEF",
        "OPSD_TOKEN_ADV_STD",
        "OPSD_TOKEN_ADV_ENV",
        "OPSD_HINT_KEY_FIELD",
        "OPSD_ONLINE_HINTS_KEY",
        "OPSD_ONLINE_HINTS_MAX_INPUT_TOKENS",
        "OPSD_ONLINE_HINTS_ANALYZER_MAX_TOKENS",
    ],
)
def test_allowlisted_hyperparameter_knobs_are_forwarded(name):
    assert not is_secret_env_name(name)
    assert collect_forwarded_env_vars({name: "0.5"}) == {name: "0.5"}


@pytest.fixture
def warnings_log():
    """Capture loguru WARNING+ messages emitted by the forwarder."""
    messages = []
    sink_id = ray_runtime.logger.add(lambda m: messages.append(str(m)), level="WARNING")
    yield messages
    ray_runtime.logger.remove(sink_id)


def test_analyzer_key_env_name_is_forwarded_but_not_the_key_it_names(warnings_log):
    # OPSD_PI_ANALYZER_API_KEY_ENV holds a variable NAME; without it a remote generator silently falls back
    # to OPENAI_API_KEY. The variable it names is never forwarded, whatever its name (see the next test).
    assert not is_secret_env_name("OPSD_PI_ANALYZER_API_KEY_ENV")
    environ = {
        "OPSD_PI_ANALYZER_API_KEY_ENV": "MY_PROVIDER_API_KEY",
        "MY_PROVIDER_API_KEY": "sk-not-a-real-key",
        "OPSD_PI_ANALYZER_API_KEY": "sk-not-a-real-key",
    }
    assert collect_forwarded_env_vars(environ) == {"OPSD_PI_ANALYZER_API_KEY_ENV": "MY_PROVIDER_API_KEY"}
    assert warnings_log == []


@pytest.mark.parametrize("target", ["OPSD_ANALYZER_AUTH", "TAU2_FW_CRED", "SKYRL_PROVIDER_PASS", "PYTHONHASHSEED"])
def test_prefixed_marker_free_key_target_is_withheld(target):
    # The name filter alone would forward these (forwarded prefix or name, no marker): only the pointer
    # lookup keeps the key they hold out of the Ray runtime env.
    assert not is_secret_env_name(target)
    environ = {
        "OPSD_PI_ANALYZER_API_KEY_ENV": target,
        target: "sk-not-a-real-key",
        "OPSD_HINTS_PATH": "/data/hints.json",
    }
    assert collect_forwarded_env_vars(environ) == {
        "OPSD_HINTS_PATH": "/data/hints.json",
        "OPSD_PI_ANALYZER_API_KEY_ENV": target,
    }


@pytest.mark.parametrize(
    "value",
    ["sk-not-a-real-key", "OPSD_ANALYZER AUTH", " OPSD_ANALYZER_AUTH", "OPSD_ANALYZER_AUTH\n", "1KEY", "MY-KEY"],
)
def test_non_identifier_key_pointer_is_withheld_and_value_never_logged(value, warnings_log):
    # A value that is not an env-var name is most likely the key itself, pasted into the pointer.
    environ = {
        "OPSD_PI_ANALYZER_API_KEY_ENV": value,
        "OPSD_ANALYZER_AUTH": "sk-not-a-real-key",
        "OPSD_HINTS_PATH": "/data/hints.json",
    }
    forwarded = collect_forwarded_env_vars(environ)
    assert "OPSD_PI_ANALYZER_API_KEY_ENV" not in forwarded
    # A whitespace-padded pointer still names OPSD_ANALYZER_AUTH once stripped; withhold that too.
    expected = {"OPSD_HINTS_PATH": "/data/hints.json"}
    if value.strip() != "OPSD_ANALYZER_AUTH":
        expected["OPSD_ANALYZER_AUTH"] = "sk-not-a-real-key"
    assert forwarded == expected
    assert len(warnings_log) == 1
    assert "OPSD_PI_ANALYZER_API_KEY_ENV" in warnings_log[0]
    assert value.strip() not in warnings_log[0]


def test_empty_key_pointer_is_forwarded_verbatim_without_warning(warnings_log):
    # "" carries no key and the analyzer treats it like unset, so the empty-stays-empty rule applies.
    assert collect_forwarded_env_vars({"OPSD_PI_ANALYZER_API_KEY_ENV": ""}) == {"OPSD_PI_ANALYZER_API_KEY_ENV": ""}
    assert warnings_log == []


def test_allowlist_is_exact_not_prefix():
    # A credential-looking name that merely starts like an allowlisted knob stays withheld.
    assert is_secret_env_name("OPSD_TOKEN_CLIP_API_KEY")
    assert is_secret_env_name("OPSD_TOKEN")
    assert is_secret_env_name("OPSD_PI_ANALYZER_API_KEY_ENV_VALUE")


def test_empty_values_are_forwarded_verbatim():
    assert collect_forwarded_env_vars({"OPSD_STRIP_THINK": ""}) == {"OPSD_STRIP_THINK": ""}


def test_prepare_runtime_environment_forwards_prefixed_vars(clean_env):
    monkeypatch = clean_env
    # peer_access_supported() would ray.init() on a GPU-less box; stub it so no Ray is started. Upstream's
    # prepare_runtime_environment looks it up as a module global, so it is stubbed on the upstream module.
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **_: True)
    monkeypatch.setenv("OPSD_HINTS_PATH", "/data/hints.json")
    monkeypatch.setenv("OPSD_TOKEN_CLIP", "0")
    monkeypatch.setenv("ALFWORLD_BASE_PORT", "8200")
    monkeypatch.setenv("APPWORLD_BASE_PORT", "9200")
    monkeypatch.setenv("TAU2_REVISION", "abc")
    monkeypatch.setenv("SKYRL_EVAL_NAMESPACE", "test")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("OPSD_PI_ANALYZER_API_KEY", "sk-not-a-real-key")
    monkeypatch.setenv("TAU2_SIM_TOKEN", "not-a-real-token")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    monkeypatch.setenv("OPSD_PI_ANALYZER_API_KEY_ENV", "OPSD_ANALYZER_AUTH")
    monkeypatch.setenv("OPSD_ANALYZER_AUTH", "sk-not-a-real-key")

    env_vars = ray_runtime.prepare_runtime_environment(SkyRLTrainConfig())

    for name, value in {
        "OPSD_HINTS_PATH": "/data/hints.json",
        "OPSD_TOKEN_CLIP": "0",
        "ALFWORLD_BASE_PORT": "8200",
        "APPWORLD_BASE_PORT": "9200",
        "TAU2_REVISION": "abc",
        "SKYRL_EVAL_NAMESPACE": "test",
        "PYTHONHASHSEED": "0",
        "OPSD_PI_ANALYZER_API_KEY_ENV": "OPSD_ANALYZER_AUTH",
    }.items():
        assert env_vars[name] == value
    for name in ("OPSD_PI_ANALYZER_API_KEY", "TAU2_SIM_TOKEN", "OPENAI_API_KEY", "OPSD_ANALYZER_AUTH"):
        assert name not in env_vars
    # Values computed by prepare_runtime_environment itself are kept (the underscore name is not a prefix match).
    assert env_vars["_SKYRL_USE_NEW_INFERENCE"] in ("0", "1")


def test_prepare_runtime_environment_without_prefixed_vars_adds_nothing(clean_env):
    monkeypatch = clean_env
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **_: True)

    env_vars = ray_runtime.prepare_runtime_environment(SkyRLTrainConfig())

    assert not [k for k in env_vars if k.startswith(ray_runtime.RUNTIME_ENV_FORWARD_PREFIXES)]
    assert "PYTHONHASHSEED" not in env_vars
    # ... and equals upstream's own result.
    assert env_vars == train_utils.prepare_runtime_environment(SkyRLTrainConfig())

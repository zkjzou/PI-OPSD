"""``register_opsd_envs()``: the only way the ``alfworld`` / ``appworld`` ids reach SkyRL's skyrl_gym registry.

SkyRL is unmodified, so its registry knows only upstream envs. The stock generator builds envs by id
(``skyrl_gym.make(env_class, env_config=getattr(cfg.environment.skyrl_gym, env_class, {}))``) and
``aggregate_for_environment`` resolves the same id for the env metrics, so a missing registration fails
the run and a stale one (an old skyrl_gym fork registering its own copy of the env) silently runs the
wrong env code.

Every check runs in a fresh interpreter: the registry is process-global, and in a real run the
registration happens in the Ray task process, not in the process that imported the modules.
"""

import dataclasses
import importlib.util
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from opsd.envs.alfworld.config import AlfworldEnvConfig
from opsd.envs.appworld.config import AppWorldEnvConfig
from opsd.envs.registry import OPSD_ENVS

# Top-level packages the config modules must not pull in: the config tree that imports them is unpickled
# in every Ray worker.
HEAVY = ("skyrl_gym", "skyrl", "requests", "omegaconf", "torch", "ray", "vllm")


def _package_root(name: str) -> str:
    return str(Path(importlib.util.find_spec(name).origin).parents[1])


def _run(code: str, tmp_path: Path, stdin: bytes = b"") -> str:
    """Run ``code`` in a fresh interpreter (from an unrelated cwd) and return its stdout."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    roots = [_package_root("opsd"), _package_root("skyrl_gym")]
    env["PYTHONPATH"] = os.pathsep.join(roots + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        input=stdin,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        timeout=300,
    )
    out, err = proc.stdout.decode(), proc.stderr.decode()
    assert proc.returncode == 0, f"subprocess failed (rc={proc.returncode})\nstdout:\n{out}\nstderr:\n{err}"
    return out


def test_ids_and_entry_points():
    assert OPSD_ENVS == {
        "alfworld": "opsd.envs.alfworld.env:AlfworldEnv",
        "appworld": "opsd.envs.appworld.env:AppWorldEnv",
    }


def test_importing_opsd_envs_has_no_side_effects(tmp_path):
    out = _run(
        f"""
        import sys

        before = set(sys.modules)
        import opsd.envs
        import opsd.envs.alfworld
        import opsd.envs.alfworld.config
        import opsd.envs.alfworld.passcode
        import opsd.envs.appworld
        import opsd.envs.appworld.config
        import opsd.envs.appworld.scaffold
        import opsd.envs.common
        import opsd.envs.common.xml_action
        import opsd.envs.registry

        heavy = sorted({{m.split(".")[0] for m in set(sys.modules) - before}} & set({HEAVY!r}))
        assert not heavy, f"light opsd.envs modules imported {{heavy}}"

        # Importing the env modules themselves (which do import skyrl_gym) must not register either.
        import opsd.envs.alfworld.env
        import opsd.envs.appworld.env
        from skyrl_gym.envs.registration import registry

        assert "alfworld" not in registry and "appworld" not in registry, sorted(registry)
        assert "gsm8k" in registry  # upstream's own registrations did run
        print("OK")
        """,
        tmp_path,
    )
    assert out.strip() == "OK"


def test_register_resolves_both_ids_and_is_idempotent(tmp_path):
    out = _run(
        """
        import skyrl_gym
        from skyrl_gym.envs.registration import load_env_creator, registry
        from skyrl_gym.metrics import aggregate_for_environment

        from opsd.envs.alfworld.config import AlfworldEnvConfig
        from opsd.envs.alfworld.env import AlfworldEnv
        from opsd.envs.appworld.config import AppWorldEnvConfig
        from opsd.envs.appworld.env import AppWorldEnv
        from opsd.envs.registry import OPSD_ENVS, register_opsd_envs

        upstream = {k: v.entry_point for k, v in registry.items()}
        assert not set(OPSD_ENVS) & set(upstream), sorted(upstream)

        register_opsd_envs()
        assert registry["alfworld"].entry_point == "opsd.envs.alfworld.env:AlfworldEnv"
        assert registry["appworld"].entry_point == "opsd.envs.appworld.env:AppWorldEnv"
        assert load_env_creator(registry["alfworld"].entry_point) is AlfworldEnv
        assert load_env_creator(registry["appworld"].entry_point) is AppWorldEnv
        for env_id in OPSD_ENVS:
            registry[env_id].to_json()  # string entry points, so the spec stays JSON-serialisable

        # What SkyRLGymGenerator does: make the env by id with the typed config from cfg.environment.skyrl_gym.
        alf = skyrl_gym.make(
            "alfworld", env_config=AlfworldEnvConfig(), extras={"reward_spec": {"ground_truth": "/g/traj_data.json"}}
        )
        app = skyrl_gym.make("appworld", env_config=AppWorldEnvConfig(), extras={"reward_spec": {"ground_truth": "t1"}})
        assert type(alf) is AlfworldEnv and type(app) is AppWorldEnv

        # What get_rollout_metrics / the eval loop do with the per-episode env metrics.
        assert aggregate_for_environment("alfworld", [{"success": 1.0}, {"success": 0.0}]) == {"success": 0.5}
        assert aggregate_for_environment("appworld", [{"success": True}, {"success": True}]) == {"success": 1.0}

        # Idempotent: the same spec objects survive repeated calls, nothing is added.
        specs = {k: registry[k] for k in OPSD_ENVS}
        n = len(registry)
        register_opsd_envs()
        register_opsd_envs()
        assert len(registry) == n
        assert all(registry[k] is specs[k] for k in OPSD_ENVS)
        # Upstream's own registrations are untouched.
        assert {k: registry[k].entry_point for k in upstream} == upstream
        print("OK")
        """,
        tmp_path,
    )
    assert out.strip() == "OK"


def test_register_is_thread_safe(tmp_path):
    out = _run(
        """
        import threading

        from skyrl_gym.envs.registration import registry

        from opsd.envs.registry import OPSD_ENVS, register_opsd_envs

        barrier = threading.Barrier(16)
        errors = []

        def worker():
            barrier.wait()
            try:
                register_opsd_envs()
            except Exception as e:  # collected and asserted below
                errors.append(repr(e))

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert {k: registry[k].entry_point for k in OPSD_ENVS} == OPSD_ENVS
        print("OK")
        """,
        tmp_path,
    )
    assert out.strip() == "OK"


@pytest.mark.parametrize(
    "env_id,stale",
    [
        # What the old OPSD fork of skyrl_gym registered at `import skyrl_gym`.
        ("alfworld", "'skyrl_gym.envs.alfworld.env:AlfworldEnv'"),
        ("appworld", "'skyrl_gym.envs.appworld.env:AppWorldEnv'"),
        # A callable entry point is a mismatch too, even if it is the right class.
        ("alfworld", "__import__('opsd.envs.alfworld.env', fromlist=['AlfworldEnv']).AlfworldEnv"),
    ],
    ids=["alfworld-fork-entry-point", "appworld-fork-entry-point", "alfworld-callable-entry-point"],
)
def test_mismatched_entry_point_raises(tmp_path, env_id, stale):
    out = _run(
        f"""
        import skyrl_gym
        from skyrl_gym.envs.registration import deregister, register, registry

        from opsd.envs.registry import OPSD_ENVS, register_opsd_envs

        stale = {stale}
        register(id={env_id!r}, entry_point=stale)
        try:
            register_opsd_envs()
        except RuntimeError as e:
            msg = str(e)
        else:
            raise AssertionError("a mismatched pre-existing entry point must raise")
        assert {env_id!r} in msg and repr(stale) in msg and repr(OPSD_ENVS[{env_id!r}]) in msg, msg
        assert skyrl_gym.__file__ in msg, msg  # names the skyrl_gym that is on sys.path
        assert registry[{env_id!r}].entry_point is stale  # never overwritten

        # The remedy the message names: drop the stale entry, then registration succeeds.
        deregister({env_id!r})
        register_opsd_envs()
        assert {{k: registry[k].entry_point for k in OPSD_ENVS}} == OPSD_ENVS
        print("OK")
        """,
        tmp_path,
    )
    assert out.strip() == "OK"


def test_env_configs_unpickle_light_in_a_fresh_process(tmp_path):
    """A Ray worker unpickling the config tree imports only the light config modules (by reference)."""
    from opsd.envs.appworld import env as appworld_env

    assert AlfworldEnvConfig.__module__ == "opsd.envs.alfworld.config"
    assert AppWorldEnvConfig.__module__ == "opsd.envs.appworld.config"
    assert appworld_env.AppWorldEnvConfig is AppWorldEnvConfig  # re-exported for existing importers

    payload = pickle.dumps(
        {
            "alfworld": AlfworldEnvConfig(server_urls=["http://127.0.0.1:1"], prompt_style="react"),
            "appworld": AppWorldEnvConfig(server_urls=["http://127.0.0.1:2"], reward_mode="partial"),
        }
    )
    out = _run(
        f"""
        import pickle
        import sys

        before = set(sys.modules)
        cfgs = pickle.loads(sys.stdin.buffer.read())
        new = set(sys.modules) - before
        assert {{"opsd.envs.alfworld.config", "opsd.envs.appworld.config"}} <= new, sorted(new)
        heavy = sorted({{m.split(".")[0] for m in new}} & set({HEAVY!r}))
        assert not heavy, f"unpickling the env configs imported {{heavy}}"
        assert cfgs["alfworld"].prompt_style == "react" and cfgs["appworld"].reward_mode == "partial"
        print("OK")
        """,
        tmp_path,
        stdin=payload,
    )
    assert out.strip() == "OK"


@pytest.mark.parametrize("cls", [AlfworldEnvConfig, AppWorldEnvConfig])
def test_env_config_annotations_are_real_types(cls):
    """SkyRL's build_nested_dataclass needs real class annotations (no ``from __future__ import annotations``)."""
    stringly = [f.name for f in dataclasses.fields(cls) if isinstance(f.type, str)]
    assert not stringly, f"{cls.__name__} has string annotations: {stringly}"

"""Register the OPSD envs (``alfworld``, ``appworld``) with SkyRL's public skyrl_gym registry.

Upstream SkyRL registers only its own envs (``skyrl_gym/envs/__init__.py``). The stock
``SkyRLGymGenerator`` builds every env by id (``skyrl_gym.make(env_class, env_config=...)``), and
``skyrl_gym.metrics.aggregate_for_environment`` looks the same id up to aggregate rollout / eval env
metrics, raising on an unknown id. So every process that builds an env or aggregates its metrics must
call :func:`register_opsd_envs` first. In a SkyRL run that is the ``skyrl_entrypoint`` Ray task (the
generator, its env threads and the eval loop all run there); the policy / ref / vLLM workers never touch
the registry.

The call is explicit, never an import side effect: importing ``opsd.envs`` (or this module) changes
nothing, so the config tree, which imports the env config dataclasses and is unpickled in every GPU
worker, stays free of skyrl_gym.

The ids must stay ``alfworld`` / ``appworld``: the released datasets write them into every row's
``env_class``, the launchers pass ``environment.env_class=<id>``, and the generator finds the env's
config at ``cfg.environment.skyrl_gym.<id>``. String entry points keep the env modules (``requests``,
``omegaconf``) unimported until an env is actually made.
"""

import threading
from typing import Dict

OPSD_ENVS: Dict[str, str] = {
    "alfworld": "opsd.envs.alfworld.env:AlfworldEnv",
    "appworld": "opsd.envs.appworld.env:AppWorldEnv",
}

_LOCK = threading.Lock()


def register_opsd_envs() -> None:
    """Register every id in ``OPSD_ENVS`` with skyrl_gym. Idempotent and thread-safe.

    An id that is already registered with the SAME entry point is left alone. An id registered with a
    DIFFERENT entry point raises ``RuntimeError``: upstream SkyRL never registers these ids, so that
    means a modified skyrl_gym (e.g. an old OPSD fork, which registered ``skyrl_gym.envs.alfworld.env``)
    is first on sys.path, and keeping its entry would silently run that fork's env code.
    """
    # Imported here so that importing this module stays free of skyrl_gym.
    from skyrl_gym.envs.registration import register, registry

    with _LOCK:
        for env_id, entry_point in OPSD_ENVS.items():
            spec = registry.get(env_id)
            if spec is None:
                register(id=env_id, entry_point=entry_point)
            elif spec.entry_point != entry_point:
                import skyrl_gym

                raise RuntimeError(
                    f"skyrl_gym env id {env_id!r} is already registered with entry point {spec.entry_point!r}, "
                    f"expected {entry_point!r}. The skyrl_gym on sys.path ({skyrl_gym.__file__}) is not the "
                    "pristine upstream one, or something else registered this id first. Use the skyrl-gym of the "
                    "pinned SkyRL checkout (the release venv), or deregister the id before calling "
                    "register_opsd_envs()."
                )

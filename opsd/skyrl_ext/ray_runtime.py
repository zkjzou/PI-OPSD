# Adapted from SkyRL (Apache-2.0), skyrl/train/utils/utils.py @ ecdaa946; modified for OPSD: initialize_ray() is
# upstream's body calling our prepare_runtime_environment() (upstream's + project env-var forwarding) and passing
# SKYRL_RAY_NUM_CPUS to ray.init as num_cpus.
"""Ray driver setup for the OPSD entrypoints.

Our entrypoints call :func:`initialize_ray` here instead of ``skyrl.train.utils.utils.initialize_ray``. It does
what upstream does, plus two things:

* Project knobs exported by the launchers (``OPSD_*``, ``ALFWORLD_*``, ``APPWORLD_*``, ``TAU2_*``, ``SKYRL_*``,
  ``PYTHONHASHSEED`` and a few W&B run-identity names) go into the job's runtime env. Tasks and actors on an
  EXISTING Ray cluster otherwise only see the environment their raylet captured at ``ray start``, so a knob
  exported on the driver alone silently falls back to its default there. In local mode (``RAY_ADDRESS=local``)
  the driver starts the raylet and workers inherit its environment anyway, so the forwarding is a no-op.
* ``SKYRL_RAY_NUM_CPUS`` caps the CPUs of a local Ray runtime on shared hosts (``ray.init(num_cpus=...)``).

Everything runs on the driver; nothing here is needed inside Ray workers. The module imports only the stdlib,
loguru, ray and ``skyrl.env_vars`` at import time (the torch-heavy upstream utils are imported on call), so the
``scripts/tau2/lib.sh`` key probe can import :func:`collect_forwarded_env_vars` cheaply.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import ray
from loguru import logger

from skyrl.env_vars import SKYRL_DUMP_INFRA_LOG_TO_STDOUT

if TYPE_CHECKING:
    from skyrl.train.config.config import SkyRLTrainConfig

# Env-var knobs forwarded wholesale into the Ray runtime env. They are read inside Ray tasks/actors
# (trainer, generator, envs, workers); a worker on another node otherwise only sees the environment its
# raylet was started with, so a knob exported on the driver alone would silently fall back to its default.
RUNTIME_ENV_FORWARD_PREFIXES = ("OPSD_", "ALFWORLD_", "APPWORLD_", "TAU2_", "SKYRL_")
RUNTIME_ENV_FORWARD_NAMES = frozenset({"PYTHONHASHSEED"})
# Never forward a variable whose NAME looks like a credential: runtime_env vars are visible in the Ray
# dashboard and logs. Credentials must already be in each node's environment when `ray start` runs.
RUNTIME_ENV_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "DATABASE_URL")
# Hyperparameter knobs whose names contain a marker but carry no credential (token counts, token-level
# advantage/clip, hint-lookup field names), plus the key-pointer names in RUNTIME_ENV_KEY_POINTERS. Exact
# names only, so any new marker-bearing name is withheld.
RUNTIME_ENV_MARKER_ALLOWLIST = frozenset(
    {
        "OPSD_TOKEN_ADV",
        "OPSD_TOKEN_ADV_COEF",
        "OPSD_TOKEN_ADV_STD",
        "OPSD_TOKEN_ADV_ENV",
        "OPSD_TOKEN_CLIP",
        "OPSD_HINT_KEY_FIELD",
        "OPSD_ONLINE_HINTS_KEY",
        "OPSD_ONLINE_HINTS_MAX_INPUT_TOKENS",
        "OPSD_ONLINE_HINTS_ANALYZER_MAX_TOKENS",
        "OPSD_PI_ANALYZER_API_KEY_ENV",
    }
)
# Variables whose VALUE is the NAME of another variable that holds a credential (the PI analyzer reads its
# key from the variable named by OPSD_PI_ANALYZER_API_KEY_ENV). The name filter alone cannot protect the
# variable they point at: a prefixed, marker-free target such as OPSD_ANALYZER_AUTH would pass it. So
# collect_forwarded_env_vars always withholds the target, and forwards the pointer only when its value is
# a valid env-var name (anything else is most likely the key itself, pasted in by mistake).
RUNTIME_ENV_KEY_POINTERS = ("OPSD_PI_ANALYZER_API_KEY_ENV",)
_ENV_VAR_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Knobs that are read INSIDE the ray entrypoint task (wandb.init in the tracker, the eval-metric
# post-processors, the tracker's step override) rather than in the launching process, forwarded by exact
# name whenever non-empty. Without them a chained final test on an existing cluster created a NEW W&B run
# instead of resuming the training run (WANDB_RUN_ID/WANDB_RESUME never arrived) and emitted no `test/*`
# metrics (SKYRL_EVAL_NAMESPACE never arrived). The SKYRL_* names are also covered by the prefix rule; listing
# them here only adds the "Exporting NAME=value" log line.
RUNTIME_ENV_EXPLICIT_NAMES = (
    "WANDB_RUN_ID",
    "WANDB_RESUME",
    "WANDB_RUN_GROUP",
    "SKYRL_EVAL_NAMESPACE",
    "SKYRL_LOG_STEP_OVERRIDE",
    "SKYRL_EVAL_COMPACT",
)

RAY_NUM_CPUS_ENV = "SKYRL_RAY_NUM_CPUS"


def is_secret_env_name(name: str) -> bool:
    """True if ``name`` looks like it carries a credential (see ``RUNTIME_ENV_SECRET_MARKERS``)."""
    upper = name.upper()
    if upper in RUNTIME_ENV_MARKER_ALLOWLIST:
        return False
    return any(marker in upper for marker in RUNTIME_ENV_SECRET_MARKERS)


def collect_forwarded_env_vars(environ=None) -> dict[str, str]:
    """Env vars to forward from the driver to Ray workers.

    Selects names starting with one of ``RUNTIME_ENV_FORWARD_PREFIXES`` or listed in
    ``RUNTIME_ENV_FORWARD_NAMES``, minus anything :func:`is_secret_env_name` flags. Values are forwarded
    verbatim (an empty string stays an empty string, so ``os.environ.get(k, default)`` behaves the same
    on every node).

    For each pointer in ``RUNTIME_ENV_KEY_POINTERS`` that is set, the variable it names is never
    forwarded, whatever its name. A non-empty pointer value that is not a valid env-var name is withheld
    too, with a warning that names the pointer but never prints its value.
    """
    environ = os.environ if environ is None else environ
    withheld = set()
    for pointer in RUNTIME_ENV_KEY_POINTERS:
        target = environ.get(pointer)
        if target is None:
            continue
        withheld.update((target, target.strip()))
        if target and not _ENV_VAR_NAME_RE.fullmatch(target):
            withheld.add(pointer)
            logger.warning(
                f"Not exporting {pointer} to the ray runtime env: its value is not a valid env-var name "
                "(value not shown). Set it to the NAME of the variable that holds the key."
            )
    return {
        name: value
        for name, value in sorted(environ.items())
        if (name.startswith(RUNTIME_ENV_FORWARD_PREFIXES) or name in RUNTIME_ENV_FORWARD_NAMES)
        and not is_secret_env_name(name)
        and name not in withheld
    }


def prepare_runtime_environment(cfg: "SkyRLTrainConfig") -> dict[str, str]:
    """Upstream ``prepare_runtime_environment(cfg)`` plus the explicit names and the prefixed project knobs.

    Values upstream computed itself take precedence over forwarded ones. Upstream sets no forwarded name
    except ``SKYRL_RAY_PG_TIMEOUT_IN_S``, which it copies from the same ``os.environ``, so the result equals
    the forked SkyRL's ``prepare_runtime_environment``.
    """
    from skyrl.train.utils import utils as skyrl_utils

    env_vars = skyrl_utils.prepare_runtime_environment(cfg)

    for var_name in RUNTIME_ENV_EXPLICIT_NAMES:
        if value := os.environ.get(var_name):
            logger.info(f"Exporting {var_name}={value} to ray runtime env")
            env_vars[var_name] = value

    forwarded = []
    for var_name, value in collect_forwarded_env_vars().items():
        if var_name not in env_vars:
            env_vars[var_name] = value
            forwarded.append(var_name)
    if forwarded:
        logger.info(f"Exporting {len(forwarded)} prefixed env vars to ray runtime env: {', '.join(forwarded)}")
    return env_vars


def ray_init_kwargs_from_env() -> dict[str, int]:
    """Extra ``ray.init`` kwargs from the environment: ``num_cpus`` from ``SKYRL_RAY_NUM_CPUS`` when set.

    Ray rejects ``num_cpus`` when attaching to an existing cluster, so launchers export this only for a
    local runtime (``scripts/tau2/run.sh`` unsets it in external mode).
    """
    ray_init_kwargs: dict[str, int] = {}
    if local_num_cpus := os.environ.get(RAY_NUM_CPUS_ENV):
        try:
            ray_init_kwargs["num_cpus"] = int(local_num_cpus)
        except ValueError as error:
            raise ValueError(f"{RAY_NUM_CPUS_ENV} must be an integer, got {local_num_cpus!r}") from error
        if ray_init_kwargs["num_cpus"] <= 0:
            raise ValueError(f"{RAY_NUM_CPUS_ENV} must be greater than zero")
        logger.info(f"Limiting the local Ray runtime to {ray_init_kwargs['num_cpus']} CPUs")
    return ray_init_kwargs


def initialize_ray(cfg: "SkyRLTrainConfig"):
    """Initialize Ray with the OPSD runtime environment (a copy of upstream ``initialize_ray``).

    Differences from upstream: the runtime env comes from :func:`prepare_runtime_environment` above, and
    :func:`ray_init_kwargs_from_env` adds ``num_cpus``.

    Args:
        cfg: Training config
    """
    from skyrl.backends.skyrl_train.utils.ppo_utils import sync_registries

    # When SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1, show all logs on stdout (no file redirect)
    verbose_logging = SKYRL_DUMP_INFRA_LOG_TO_STDOUT

    # Suppress Ray backend logs unless in verbose mode
    if not verbose_logging:
        os.environ["RAY_BACKEND_LOG_LEVEL"] = "fatal"

    env_vars = prepare_runtime_environment(cfg)

    # Set up log file for infrastructure logs (skip when dumping to stdout)
    if not verbose_logging:
        log_path = Path(cfg.trainer.log_path).resolve()
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        log_file = str(log_path / f"infra-{timestamp}.log")
        os.environ["SKYRL_LOG_FILE"] = log_file
        # Pass log file path to workers so they can redirect their output
        env_vars["SKYRL_LOG_FILE"] = log_file

    # Parsed only now, so an invalid value raises before ray.init but after the log-file setup above (the order
    # the fork-parity goldens pin).
    ray_init_kwargs = ray_init_kwargs_from_env()

    # log_to_driver=True allows training progress from skyrl_entrypoint to reach stdout.
    # Infrastructure logs (vLLM, workers) are redirected to log file via os.dup2 in their init.
    ray.init(runtime_env={"env_vars": env_vars}, log_to_driver=True, **ray_init_kwargs)

    if not verbose_logging:
        logger.info(f"Infrastructure logs will be written to: {log_file}")

    # create the named ray actors for the registries to make available to all workers
    sync_registries()

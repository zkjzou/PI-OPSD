"""Locate the repository root, the ``opsd`` package's own data and the SkyRL checkout it runs on.

The repository::

    <root>/pyproject.toml, uv.lock   the project; ``uv sync`` builds <root>/.venv
    <root>/opsd/                     the ``opsd`` package; recipes/ = the resolver and the recipe data
                                     (alfworld/ appworld/ tau2/ common/), tests/ = repo-level tests and fixtures
    <root>/scripts/                  launchers (alfworld/ appworld/ tau2/ common/), setup/, analysis/, release_guard/
    <root>/assets/                   docs/, results/ and the README's figures and headline tables

SkyRL is not part of the repository: it is an unmodified NovaSky-AI/SkyRL checkout at :data:`SKYRL_COMMIT`
(by default the sibling ``<root>/../SkyRL-ecdaa946``) that ``uv sync`` installs editable
(``[tool.uv.sources]`` in pyproject.toml). :func:`skyrl_root` finds it through the installed ``skyrl`` package,
never through a repository path.

Tests and scripts that read repository files (launchers, recipes, results) must not hard-code
``Path(__file__).parents[N]``. ``repo_root()`` walks up until it finds the directory holding ``pyproject.toml``,
``opsd/__init__.py`` and ``opsd/skyrl_ext/__init__.py``; the launchers (``scripts/<bench>/run.sh``) use the same
predicate. Data that ships inside the package (the recipes) is found from the package itself
(:func:`package_dir`, :func:`recipes_dir`).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional, Union

# The NovaSky-AI/SkyRL commit this repository is built and tested against (scripts/setup/setup_train_env.sh
# refuses any other; provenance.json records whether the installed checkout matches).
SKYRL_COMMIT = "ecdaa9460e281774638a9170d6724456f46c8356"

PACKAGE_DIR = "opsd"
ASSETS_DIR = "assets"


def _is_repo_root(path: Path) -> bool:
    return (
        (path / "pyproject.toml").is_file()
        and (path / PACKAGE_DIR / "__init__.py").is_file()
        and (path / PACKAGE_DIR / "skyrl_ext" / "__init__.py").is_file()
    )


def repo_root(start: Optional[Union[str, Path]] = None) -> Path:
    """Return the repository root: the nearest directory at or above ``start`` (default: this file) that
    contains ``pyproject.toml``, ``opsd/__init__.py`` and ``opsd/skyrl_ext/__init__.py``.

    Raises FileNotFoundError if no such directory exists (e.g. ``opsd`` was installed from a wheel instead of
    being used from a clone of the repository).
    """
    here = Path(start if start is not None else __file__).resolve()
    if not here.is_dir():
        here = here.parent
    for candidate in (here, *here.parents):
        if _is_repo_root(candidate):
            return candidate
    raise FileNotFoundError(
        f"no directory containing pyproject.toml, {PACKAGE_DIR}/__init__.py and {PACKAGE_DIR}/skyrl_ext/__init__.py"
        f" at or above {here} (run from a clone of the repository)"
    )


def package_dir() -> Path:
    """The ``opsd`` package directory (``<root>/opsd`` in a clone)."""
    return Path(__file__).resolve().parent


def recipes_dir() -> Path:
    """The shipped recipes tree, ``opsd/recipes`` (the resolver's default unless ``OPSD_RECIPES_DIR`` is set)."""
    return package_dir() / "recipes"


def fixtures_dir() -> Path:
    """The fixtures of the repo-level tests, ``opsd/tests/fixtures`` (fork-parity goldens). Named so that a test
    module importing it does not get a ``test*`` function that pytest would collect."""
    return package_dir() / "tests" / "fixtures"


def assets_dir(start: Optional[Union[str, Path]] = None) -> Path:
    """``<root>/assets``: the README's figures."""
    return repo_root(start) / ASSETS_DIR


def skyrl_root() -> Path:
    """The checkout of the installed ``skyrl`` package (read-only): the directory holding ``skyrl/``,
    ``skyrl-gym/``, ``pyproject.toml`` and ``uv.lock``, i.e. ``Path(skyrl.__file__).resolve().parents[1]``.

    Found without importing ``skyrl``. Raises FileNotFoundError when it is not installed (build the venv with
    ``bash scripts/setup/setup_train_env.sh``).
    """
    spec = importlib.util.find_spec("skyrl")
    if spec is None or not spec.origin or not spec.has_location:
        raise FileNotFoundError(
            "the skyrl package is not installed in this interpreter (build the venv with"
            " `bash scripts/setup/setup_train_env.sh`, which installs the SkyRL checkout at"
            f" {SKYRL_COMMIT[:8]} editable)"
        )
    return Path(spec.origin).resolve().parents[1]

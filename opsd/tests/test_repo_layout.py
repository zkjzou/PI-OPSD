"""The layout contract of ``opsd._repo``: the repository root, the package's own data and the external SkyRL.

Tests, scripts and the resolver locate files only through these helpers, so a layout change that breaks one of
them fails here first instead of as a missing file deep inside another test.
"""

from pathlib import Path

import pytest

import opsd
from opsd import _repo


def test_repo_root_is_found_from_anywhere_inside_the_repository(tmp_path):
    root = _repo.repo_root()
    assert (root / "pyproject.toml").is_file() and (root / "opsd" / "skyrl_ext" / "__init__.py").is_file()
    for start in (root, root / "opsd" / "recipes", root / "scripts" / "alfworld" / "run.sh", Path(__file__)):
        assert _repo.repo_root(start) == root, start
    with pytest.raises(FileNotFoundError):
        _repo.repo_root(tmp_path)


def test_package_data_is_found_from_the_package():
    pkg = Path(opsd.__file__).resolve().parent
    assert _repo.package_dir() == pkg == _repo.repo_root() / "opsd"
    assert _repo.recipes_dir() == pkg / "recipes"
    assert {"alfworld", "appworld", "tau2", "common"} <= {p.name for p in _repo.recipes_dir().iterdir()}
    assert (_repo.fixtures_dir() / "fork_parity").is_dir()


def test_root_holds_the_skyrl_attribution_inventory():
    assert (_repo.repo_root() / "ADAPTED_FROM_SKYRL.md").is_file()


def test_skyrl_is_the_installed_checkout_outside_the_repository():
    import skyrl

    root = _repo.skyrl_root()
    assert root == Path(skyrl.__file__).resolve().parents[1]
    assert (root / "skyrl" / "__init__.py").is_file()
    repo = _repo.repo_root()
    assert repo != root and repo not in root.parents, f"SkyRL ({root}) must be an external checkout, not in {repo}"

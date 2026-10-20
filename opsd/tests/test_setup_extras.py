"""The setup scripts keep the extras an existing ``.venv`` already has.

``uv sync`` removes every package of an extra it is not given, so a bare ``setup_tau2.sh`` after a
``uv sync --extra fsdp --extra dev`` used to uninstall pytest, and one after a Megatron sync switched the environment
back to FSDP. ``setup_train_env.sh`` now adds the extras ``.venv`` has (dev / tau2 / the Megatron backend) unless
``--exact`` is given or a backend is named, and ``setup_tau2.sh`` names a backend only when ``TAU2_UV_EXTRA`` is set.

Both scripts run here against a throw-away repository with stubs: a fake ``.venv/bin/python`` that reports a chosen
set of installed extras, and a fake ``uv`` that records its arguments (``--check`` mode only, so nothing is built).
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from opsd._repo import repo_root

ROOT = repo_root()
SETUP = ROOT / "scripts" / "setup"
PIN = "0123456789abcdef0123456789abcdef01234567"

FAKE_PYTHON = """#!/usr/bin/env bash
here="$(cd "$(dirname "$0")" && pwd)"
if [ "$#" = 1 ] && [ "$1" = - ]; then cat >/dev/null; cat "$here/../extras"; exit 0; fi  # installed-extras probe
if [ "${1:-}" = -c ]; then echo 3.12.0; exit 0; fi                                       # version
cat >/dev/null                                                                       # the import verification
"""
FAKE_UV = """#!/usr/bin/env bash
[ "${1:-}" = --version ] && { echo "uv 0.0.0"; exit 0; }
printf '%s\\n' "$*" >>"$UV_LOG"
"""


def _exe(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def _fake_repo(tmp_path: Path, extras: str | None) -> Path:
    repo = tmp_path / "repo"
    (repo / "opsd" / "skyrl_ext").mkdir(parents=True)
    (repo / "opsd" / "__init__.py").write_text("")
    (repo / "opsd" / "skyrl_ext" / "__init__.py").write_text("")
    (repo / "opsd" / "_repo.py").write_text(f'SKYRL_COMMIT = "{PIN}"\n')
    (repo / "uv.lock").write_text("")
    (repo / "pyproject.toml").write_text(
        "[tool.uv.sources]\n"
        'skyrl = { path = "../SkyRL-x", editable = true }\n'
        'skyrl-gym = { path = "../SkyRL-x/skyrl-gym", editable = true }\n'
    )
    (repo / "scripts" / "setup").mkdir(parents=True)
    for name in ("setup_train_env.sh", "setup_tau2.sh"):
        shutil.copy2(SETUP / name, repo / "scripts" / "setup" / name)
    if extras is not None:
        _exe(repo / ".venv" / "bin" / "python", FAKE_PYTHON)
        (repo / ".venv" / "extras").write_text(extras)
    return repo


def _env(tmp_path: Path, **extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPSD_", "TAU2_", "UV", "SKIP_UV"))}
    env.pop("SKYRL_DIR", None)
    env.update(UV=str(_exe(tmp_path / "bin" / "uv", FAKE_UV)), UV_LOG=str(tmp_path / "uv.log"), **extra)
    return env


def _check(tmp_path: Path, extras: str, *flags: str, **env) -> tuple[str, str]:
    """Run setup_train_env.sh --check FLAGS; return (the args of its `uv sync --check`, its stdout + stderr)."""
    repo = _fake_repo(tmp_path, extras)
    p = subprocess.run(
        ["bash", str(repo / "scripts/setup/setup_train_env.sh"), "--check", *flags],
        env=_env(tmp_path, **env),
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
    )
    log = (tmp_path / "uv.log").read_text().strip() if (tmp_path / "uv.log").exists() else ""
    return log, p.stdout + p.stderr


@pytest.mark.parametrize(
    "extras, flags, env, want, keeps",
    [
        ("", (), {}, "--extra fsdp", False),
        ("dev\n", ("--tau2",), {}, "--extra fsdp --extra tau2 --extra dev", True),
        ("tau2\ndev\n", (), {}, "--extra fsdp --extra tau2 --extra dev", True),
        ("megatron\ndev\n", ("--tau2",), {}, "--extra megatron --extra tau2 --extra dev", True),
        ("megatron\n", ("--extra", "fsdp"), {}, "--extra fsdp", False),  # an explicit backend switches
        ("megatron\n", (), {"OPSD_UV_EXTRA": "fsdp"}, "--extra fsdp", False),  # so does OPSD_UV_EXTRA
        ("megatron\ntau2\ndev\n", ("--exact",), {}, "--extra fsdp", False),
        (
            "megatron\ntau2\ndev\n",
            ("--exact", "--extra", "megatron", "--dev"),
            {},
            "--extra megatron --extra dev",
            False,
        ),
    ],
)
def test_setup_train_env_keeps_the_installed_extras(tmp_path, extras, flags, env, want, keeps):
    log, out = _check(tmp_path, extras, *flags, **env)
    assert log == f"sync --frozen {want} --check --offline", out
    assert ("keeping the extras" in out) == keeps, out


def test_setup_train_env_says_when_an_explicit_backend_replaces_megatron(tmp_path):
    _, out = _check(tmp_path, "megatron\n", "--extra", "fsdp")
    assert "replaces the Megatron stack" in out, out


def _tau2_args(tmp_path: Path, *flags: str, **env) -> tuple[str | None, subprocess.CompletedProcess]:
    """Run setup_tau2.sh with setup_train_env.sh stubbed out; return the stub's arguments (None: not called)."""
    repo = _fake_repo(tmp_path, "")
    args = tmp_path / "train_env.args"
    _exe(repo / "scripts/setup/setup_train_env.sh", f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >"{args}"\nexit 3\n')
    p = subprocess.run(
        ["bash", str(repo / "scripts/setup/setup_tau2.sh"), *flags],
        env=_env(tmp_path, **env),
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
    )
    return (args.read_text().strip() if args.exists() else None), p


@pytest.mark.parametrize(
    "flags, env, want",
    [
        ((), {}, "--tau2"),  # no backend named: setup_train_env.sh keeps the one .venv has
        (("--dev",), {}, "--tau2 --dev"),
        (("--dev",), {"TAU2_UV_EXTRA": "megatron"}, "--tau2 --extra megatron --dev"),
        (("--exact",), {"TAU2_UV_EXTRA": "fsdp"}, "--tau2 --extra fsdp --exact"),
    ],
)
def test_setup_tau2_names_a_backend_only_when_asked(tmp_path, flags, env, want):
    got, p = _tau2_args(tmp_path, *flags, **env)
    assert got == want, p.stderr
    assert p.returncode != 0 and "building the training environment failed" in p.stderr


def test_setup_tau2_rejects_an_unknown_backend(tmp_path):
    got, p = _tau2_args(tmp_path, TAU2_UV_EXTRA="cuda")
    assert got is None and p.returncode != 0 and "TAU2_UV_EXTRA=cuda" in p.stderr


def test_setup_train_env_resolves_the_skyrl_path_from_the_physical_repository(tmp_path):
    """A clone reached through a symlink: ../SkyRL-x is next to the REAL directory (where uv and git look)."""
    real = tmp_path / "real"
    real.mkdir()
    repo = _fake_repo(real, "")
    link = tmp_path / "elsewhere" / "repo"
    link.parent.mkdir()
    link.symlink_to(repo, target_is_directory=True)
    p = subprocess.run(
        ["bash", "scripts/setup/setup_train_env.sh", "--check"],
        env=_env(tmp_path),
        cwd=str(link),
        capture_output=True,
        text=True,
        timeout=60,
    )
    out = p.stdout + p.stderr
    assert f"no SkyRL checkout at {(real / 'SkyRL-x').resolve()}" in out, out

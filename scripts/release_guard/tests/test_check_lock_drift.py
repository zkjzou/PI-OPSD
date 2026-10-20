"""scripts/release_guard/check_lock_drift.py: uv.lock against SkyRL's own uv.lock at the pin.

Synthetic locks exercise every kind of drift; the last test runs the check on the real uv.lock, against the lock of
the SkyRL checkout that pyproject.toml's [tool.uv.sources] installs from (skipped when that checkout is absent).
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

CHECK = Path(__file__).resolve().parents[1] / "check_lock_drift.py"
ROOT = Path(__file__).resolve().parents[3]
PYPI = 'source = { registry = "https://pypi.org/simple" }'


def _load():
    spec = importlib.util.spec_from_file_location("opsd_check_lock_drift_under_test", CHECK)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


LD = _load()


def _pkg(name, version, source=PYPI, wheel=None):
    wheel = wheel or f"https://files.example/{name}-{version}-py3-none-any.whl"
    return f'[[package]]\nname = "{name}"\nversion = "{version}"\n{source}\nwheels = [{{ url = "{wheel}" }}]\n'


def _lock(path: Path, *pkgs: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("version = 1\nrevision = 3\n\n" + "\n".join(pkgs))
    return path


SKYRL_PKGS = (
    _pkg("requests", "2.32.0"),
    _pkg("torch", "2.11.0+cu128", 'source = { registry = "https://download.pytorch.org/whl/cu128" }'),
    _pkg("torch", "2.7.0"),
    _pkg("skyrl", "0.2.0", 'source = { editable = "." }'),
    _pkg("boto3", "1.0.0"),
)
OURS_PKGS = (
    _pkg("requests", "2.32.0"),
    _pkg("torch", "2.11.0+cu128", 'source = { registry = "https://download.pytorch.org/whl/cu128" }'),
    _pkg("skyrl", "0.2.0", 'source = { editable = "../SkyRL-x" }'),
    _pkg("opsd", "0.1.0", 'source = { editable = "." }'),
    _pkg("deepdiff", "9.1.0"),
)


def _run(tmp_path, ours=OURS_PKGS, skyrl=SKYRL_PKGS, pins=None, *extra):
    o = _lock(tmp_path / "repo" / "uv.lock", *ours)
    s = _lock(tmp_path / "SkyRL-x" / "uv.lock", *skyrl)
    args = ["--ours", str(o), "--skyrl", str(s), *extra]
    if pins is None:
        args.append("--no-pyproject")
    else:
        pp = tmp_path / "repo" / "pyproject.toml"
        pp.write_text("[tool.uv]\nconstraint-dependencies = [\n" + "".join(f'    "{p}",\n' for p in pins) + "]\n")
        args += ["--pyproject", str(pp)]
    p = subprocess.run([sys.executable, str(CHECK), *args], capture_output=True, text=True, timeout=60)
    return p.returncode, p.stdout + p.stderr


def test_no_drift(tmp_path):
    rc, out = _run(tmp_path, pins=["torch==2.11.0", "requests==2.32.0"])
    assert rc == 0, out
    assert "allowed addition: deepdiff" in out and "this project: opsd" in out and "boto3" in out


@pytest.mark.parametrize(
    "replace, needle",
    [
        (_pkg("requests", "2.31.0"), "version drift: requests 2.31.0"),
        (_pkg("requests", "2.32.0", 'source = { registry = "https://mirror.example/simple" }'), "source drift"),
        (_pkg("requests", "2.32.0", wheel="https://files.example/requests-rebuilt.whl"), "file drift"),
        (_pkg("skyrl", "0.3.0", 'source = { editable = "../SkyRL-x" }'), "version drift: skyrl 0.3.0"),
    ],
)
def test_drift_is_reported(tmp_path, replace, needle):
    name = replace.split('"')[1]
    ours = tuple(p for p in OURS_PKGS if f'name = "{name}"' not in p) + (replace,)
    rc, out = _run(tmp_path, ours=ours)
    assert rc == 1 and needle in out, out


def test_an_unexpected_package_is_drift_unless_allowed(tmp_path):
    ours = OURS_PKGS + (_pkg("leftpad", "1.0.0"),)
    rc, out = _run(tmp_path, ours=ours)
    assert rc == 1 and "only in ours: leftpad" in out, out
    rc, out = _run(tmp_path, ours, SKYRL_PKGS, None, "--allow", "leftpad")
    assert rc == 0, out


def test_constraint_pins_must_be_skyrl_versions(tmp_path):
    rc, out = _run(tmp_path, pins=["requests==2.31.0"])
    assert rc == 1 and "constraint requests==2.31.0" in out, out
    rc, out = _run(tmp_path, pins=["leftpad==1.0"])
    assert rc == 1 and "does not have" in out, out


def test_suggest_prints_skyrl_versions(tmp_path):
    ours = tuple(p for p in OURS_PKGS if 'name = "requests"' not in p) + (_pkg("requests", "2.31.0"),)
    rc, out = _run(tmp_path, ours, SKYRL_PKGS, None, "--suggest")
    assert rc == 1 and '"requests==2.32.0",' in out, out


def test_unreadable_input_exits_2(tmp_path):
    p = subprocess.run(
        [sys.executable, str(CHECK), "--ours", str(tmp_path / "nope.lock"), "--skyrl", str(tmp_path / "x.lock")],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 2, p.stdout + p.stderr


def test_default_skyrl_lock_comes_from_the_uv_sources_path(tmp_path):
    (tmp_path / "repo").mkdir()
    pp = tmp_path / "repo" / "pyproject.toml"
    pp.write_text('[tool.uv.sources]\nskyrl = { path = "../SkyRL-x", editable = true }\n')
    assert LD.skyrl_lock_from_sources(pp) == (tmp_path / "SkyRL-x" / "uv.lock").resolve()


def test_the_repository_lock_has_no_drift():
    skyrl = LD.skyrl_lock_from_sources(ROOT / "pyproject.toml")
    if skyrl is None or not skyrl.is_file():
        pytest.skip(f"no SkyRL checkout with a uv.lock at the [tool.uv.sources] path ({skyrl})")
    rc = LD.main(["--ours", str(ROOT / "uv.lock"), "--skyrl", str(skyrl)])
    assert rc == 0, "uv.lock drifted from SkyRL's lock: run scripts/release_guard/check_lock_drift.py --suggest"

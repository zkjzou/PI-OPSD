"""scripts/setup/appworld_bundles.py (step 3 of setup_appworld.sh) on fake bundles: no AppWorld, no network.

The real decryption is AppWorld's ``unpack_bundle``; here a zip stands in for it, so what is tested is everything
around it: which bundles are picked, the sha256 gate, the per-file check against the manifest's ``contents``, the
stamp that makes a re-run cheap, and that a failed unpack leaves the previous state untouched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from opsd import _repo

SCRIPT = _repo.repo_root() / "scripts" / "setup" / "appworld_bundles.py"
SETUP_SH = _repo.repo_root() / "scripts" / "setup" / "setup_appworld.sh"


@pytest.fixture(scope="module")
def AB():
    spec = importlib.util.spec_from_file_location("appworld_bundles", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def make_bundle(data_root: Path, name: str, files: dict, manifest_files: list, contents_override=None) -> Path:
    """A zip named appworld/<name>.bundle holding ``files`` ({path: bytes}) plus its manifest record."""
    aw = data_root / "appworld"
    aw.mkdir(parents=True, exist_ok=True)
    path = aw / f"{name}.bundle"
    with zipfile.ZipFile(path, "w") as z:
        for rel, data in files.items():
            z.writestr(rel, data)
    contents = [{"path": rel, "bytes": len(data), "sha256": _sha(data)} for rel, data in files.items()]
    manifest_files.append(
        {
            "path": f"appworld/{name}.bundle",
            "bytes": path.stat().st_size,
            "sha256": _sha(path.read_bytes()),
            "contents": contents_override if contents_override is not None else contents,
        }
    )
    (data_root / "manifest.json").write_text(json.dumps({"files": manifest_files}))
    return path


class ZipUnpacker:
    def __init__(self):
        self.calls = []

    def __call__(self, bundle, base, password, salt):
        assert (password, salt) == ("pw", "salt")
        self.calls.append(Path(bundle).name)
        with zipfile.ZipFile(bundle) as z:
            z.extractall(base)


XD = {
    "opsd_appworld_xdnull/LICENSE": b"notice",
    "opsd_appworld_xdnull/alfworld/glt_xdmix_tau2_appworld.json": b'{"trial_a": {"task": "hint"}}',
}


def test_inner_root_from_the_manifest_or_the_naming_rule(AB):
    assert AB.inner_root("appworld_opsd_xdnull", None) == "opsd_appworld_xdnull"
    assert AB.inner_root("appworld_opsd_corpora_dev", {"contents": []}) == "opsd_appworld_corpora_dev"
    rec = {"contents": [{"path": "somewhere/x.json"}, {"path": "somewhere/y/z.json"}]}
    assert AB.inner_root("appworld_opsd_new", rec) == "somewhere"


def test_every_dataset_bundle_is_pinned_and_flagged(AB):
    assert set(AB.PINNED) == {
        "appworld_opsd_pi",
        "appworld_opsd_metrics",
        "appworld_opsd_corpora_train90",
        "appworld_opsd_corpora_dev",
        "appworld_opsd_corpora_test_normal",
        "appworld_opsd_dev_hints",
        "appworld_opsd_xdnull",
        "appworld_opsd_test_normal_hints",
    }
    assert all(re.fullmatch(r"[0-9a-f]{64}", sha) for sha in AB.PINNED.values())
    assert re.fullmatch(r"[0-9a-f]{40}", AB.REVISION)
    # everything but the batch-1 (train-90) bundle and the train-90 corpora carries evaluation-split material
    assert set(AB.PINNED) - set(AB.EVAL_SPLIT) == {"appworld_opsd_pi", "appworld_opsd_corpora_train90"}
    assert set(AB.EVAL_SPLIT) <= set(AB.PINNED)
    assert "test_normal" in AB.EVAL_SPLIT["appworld_opsd_xdnull"]
    assert "test_normal" in AB.EVAL_SPLIT["appworld_opsd_test_normal_hints"]
    assert "labels" in AB.EVAL_SPLIT["appworld_opsd_metrics"]


def test_bundles_unpack_under_their_naming_rule_roots(AB, tmp_path):
    """Without a manifest record the inner root comes from the naming rule; every pinned bundle follows it."""
    assert AB.inner_root("appworld_opsd_test_normal_hints", None) == "opsd_appworld_test_normal_hints"
    assert AB.inner_root("appworld_opsd_metrics", None) == "opsd_appworld_metrics"
    # a plan over an empty data root lists every pinned bundle, each under its own plain directory
    data = tmp_path / "data"
    data.mkdir()
    plan = {b.name: b for b in AB.plan(data, None)}
    assert set(plan) == set(AB.PINNED)
    assert len({b.dir for b in plan.values()}) == len(plan)
    xn = plan["appworld_opsd_test_normal_hints"]
    assert xn.want == AB.PINNED["appworld_opsd_test_normal_hints"] and xn.inner == "opsd_appworld_test_normal_hints"


def _xdnull(AB, data):
    (b,) = [b for b in AB.plan(data, None) if b.name == "appworld_opsd_xdnull"]
    return b


def test_unpack_verify_stamp_then_skip(AB, tmp_path):
    data = tmp_path / "data"
    make_bundle(data, "appworld_opsd_xdnull", XD, [])
    un = ZipUnpacker()
    assert AB.setup(AB.plan(data, None), "pw", "salt", un) == 0
    plain = data / "appworld" / "plain"
    hint = plain / "opsd_appworld_xdnull" / "alfworld" / "glt_xdmix_tau2_appworld.json"
    assert hint.read_bytes() == XD["opsd_appworld_xdnull/alfworld/glt_xdmix_tau2_appworld.json"]
    stamp = json.loads((plain / ".opsd_appworld_xdnull.opsd_unpacked.json").read_text())
    assert stamp["verified_against_manifest"] is True and set(stamp["files"]) == set(XD)
    assert un.calls == ["appworld_opsd_xdnull.bundle"]
    assert not [p for p in plain.iterdir() if p.name.startswith(".unpack.")]  # scratch removed
    # re-run: nothing decrypted again; a changed size is noticed and repaired from the bundle
    assert AB.setup(AB.plan(data, None), "pw", "salt", un) == 0 and len(un.calls) == 1
    hint.write_bytes(b"tampered, longer than before")
    assert AB.state(_xdnull(AB, data), write=False) == "incomplete"
    assert AB.setup(AB.plan(data, None), "pw", "salt", un) == 0 and len(un.calls) == 2
    assert hint.read_bytes() == XD["opsd_appworld_xdnull/alfworld/glt_xdmix_tau2_appworld.json"]
    assert AB.check(AB.plan(data, None)) == 0


def test_a_directory_unpacked_without_a_stamp_is_verified_not_decrypted_again(AB, tmp_path):
    data = tmp_path / "data"
    make_bundle(data, "appworld_opsd_xdnull", XD, [])
    for rel, b in XD.items():  # as an earlier setup script left it: the files, no stamp
        p = data / "appworld" / "plain" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    un = ZipUnpacker()
    (bundle,) = [b for b in AB.plan(data, None) if b.name == "appworld_opsd_xdnull"]
    assert AB.state(bundle, write=False) == "verified" and not bundle.stamp.exists()  # check mode writes nothing
    assert AB.setup(AB.plan(data, None), "pw", "salt", un) == 0
    assert un.calls == [] and bundle.stamp.exists() and AB.quick_ok(bundle)


def test_a_bad_bundle_or_bad_contents_fail_and_leave_the_previous_state(AB, tmp_path):
    data = tmp_path / "data"
    files = []
    make_bundle(data, "appworld_opsd_xdnull", XD, files)
    un = ZipUnpacker()
    assert AB.setup(AB.plan(data, None), "pw", "salt", un) == 0
    good = (data / "appworld/plain/opsd_appworld_xdnull/alfworld/glt_xdmix_tau2_appworld.json").read_bytes()
    # a dev-hints bundle whose manifest contents do not match what it unpacks
    bad_contents = [{"path": "opsd_appworld_dev_hints/h.json", "bytes": 2, "sha256": "0" * 64}]
    make_bundle(data, "appworld_opsd_dev_hints", {"opsd_appworld_dev_hints/h.json": b"{}"}, files, bad_contents)
    # and a dev-corpus bundle whose bytes are not the ones the manifest lists
    p = make_bundle(data, "appworld_opsd_corpora_dev", {"opsd_appworld_corpora_dev/x.csv": b"a,b\n"}, files)
    p.write_bytes(p.read_bytes() + b"corrupt")
    assert AB.setup(AB.plan(data, None), "pw", "salt", un) == 1
    plain = data / "appworld" / "plain"
    assert not (plain / "opsd_appworld_dev_hints").exists() and not (plain / "opsd_appworld_corpora_dev").exists()
    assert "appworld_opsd_corpora_dev.bundle" not in un.calls  # refused before decryption
    assert (plain / "opsd_appworld_xdnull/alfworld/glt_xdmix_tau2_appworld.json").read_bytes() == good
    assert not [x for x in plain.iterdir() if x.name.startswith(".unpack.")]


def test_absent_bundles_are_skipped(AB, tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    un = ZipUnpacker()
    assert AB.setup(AB.plan(data, None), "pw", "salt", un) == 0 and un.calls == []
    err = capsys.readouterr().err
    assert "WARNING: no " in err and "appworld_opsd_pi.bundle" in err  # training needs the batch-1 bundle
    assert "not fetched (optional)" in err and "appworld_opsd_xdnull" in err
    assert not (data / "appworld" / "plain").exists()


def test_setup_appworld_check_reports_the_bundles_and_changes_nothing(tmp_path):
    data = tmp_path / "data"
    make_bundle(data, "appworld_opsd_xdnull", XD, [])
    # the check runs the helper with python3 when there is no AppWorld venv
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "python3").symlink_to(sys.executable)
    env = {
        "PATH": f"{shim}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(tmp_path / "home"),
        "OPSD_DATA_ROOT": str(data),
        "APPWORLD_VENV": str(tmp_path / "no-venv"),
        "APPWORLD_ROOT": str(tmp_path / "no-root"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    proc = subprocess.run(["bash", str(SETUP_SH), "--check"], env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "venv:    MISSING/BROKEN" in proc.stderr
    assert "bundle appworld_opsd_xdnull: present; plain: not unpacked" in proc.stderr
    assert "evaluation-split material" in proc.stderr
    assert "bundle appworld_opsd_pi: not fetched" in proc.stderr
    after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert after == before

#!/usr/bin/env python3
"""Verify and unpack the encrypted OPSD AppWorld bundles (step 3 of ``scripts/setup/setup_appworld.sh``).

    python scripts/setup/appworld_bundles.py check --data-root DIR [--pi-bundle FILE]
    python scripts/setup/appworld_bundles.py setup --data-root DIR [--pi-bundle FILE] --password P --salt S

Every ``appworld/appworld_opsd_*.bundle`` under ``--data-root`` ($OPSD_DATA_ROOT) is decrypted with AppWorld's own
``unpack_bundle`` into ``appworld/plain/<inner root>/``, the bundle's single top-level directory
(``appworld_opsd_xdnull.bundle`` -> ``plain/opsd_appworld_xdnull/``). ``--pi-bundle`` replaces the location of
``appworld_opsd_pi.bundle``.

* A bundle is decrypted only if its sha256 is the one ``<data-root>/manifest.json`` (written by
  ``python -m opsd.data.fetch``) lists for it, or, without a manifest record, the one pinned in ``PINNED`` for the
  dataset revision this release documents.
* Every unpacked file is checked against the sha256 of the manifest record's ``contents`` (and, for the PI
  bundle, against its own ``opsd_appworld_pi/CONTENTS.json``); a mismatch leaves the previous state untouched.
* A verified unpack leaves a stamp ``plain/.<inner root>.opsd_unpacked.json`` (bundle sha256, every file with its
  size and sha256). A later run that finds a stamp for the expected bundle, every stamped file at its size and
  the manifest's sha256 values equal to the stamped ones skips the bundle without hashing anything. An unpacked
  directory without a stamp (from an earlier version of the setup script) is re-verified file by file rather
  than decrypted again.

``check`` reports and changes nothing (exit 0); ``setup`` imports ``appworld``, so run it with the AppWorld venv's
python. Standard library only otherwise. AppWorld's license allows content derived from its task data to be
redistributed only in encrypted form: do not re-share the unpacked files in plain text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Dataset revision whose bundle sha256 values are pinned below (used when manifest.json has no record).
REVISION = "b9b1da0837c93ae981b60aaadbbbbbad2f0b222e"
PI = "appworld_opsd_pi"
PINNED = {
    PI: "d67909914474c9c233c247b55270e4deed18c3dd55f7fcc17372f0af4f2c826f",  # hints, reference trajectories, b0a8eae patch
    "appworld_opsd_metrics": "814c56215a34b5b2cb8c6e872e6b0cc9eda368af3ba5f39704e3f3897f57983c",  # error-step judge labels
    "appworld_opsd_corpora_train90": "04eba4ccf89c77f436d0379805d2a441e7ed93327289d37f91c124523020f6f5",
    "appworld_opsd_corpora_dev": "0f80284048de1afeed00e5775b92b26f733e4ed78f2ad1d1b1e547c3c8e4eae2",
    "appworld_opsd_corpora_test_normal": "8a980ee6b49616f9f3dc4cd28ecdf14d87a05aead6760afc16e8efaa555cd0e5",
    "appworld_opsd_dev_hints": "66370e960223ee349754fea99a04184d29d546a43a39f50eec4e85f48062360c",
    "appworld_opsd_test_normal_hints": "e70fc9302882e6d5f4c6c95a95b8abac743e05aff23570455c4f81332b19385d",
    "appworld_opsd_xdnull": "2b9512695bf79ece0cf0a009b04f6c2f6133d368ca214db0d1fb3ae990db6078",
}
# What each bundle holds that is keyed to an evaluation split (dataset card, "Flagged items"): never
# train a model you evaluate on that split with it; for reproducing the paper's controls / analysis only.
EVAL_SPLIT = {
    "appworld_opsd_metrics": "its dev and test_normal error-step labels",
    "appworld_opsd_corpora_dev": "AppWorld dev rollouts",
    "appworld_opsd_corpora_test_normal": "AppWorld test_normal rollouts",
    "appworld_opsd_dev_hints": "PI hints keyed to AppWorld dev",
    "appworld_opsd_xdnull": "ALFWorld cross-domain null hints whose text is AppWorld test_normal reference solutions",
    "appworld_opsd_test_normal_hints": "PI hints keyed to AppWorld test_normal",
}
STAMP_SUFFIX = ".opsd_unpacked.json"

Unpacker = Callable[[Path, Path, str, str], None]


def log(msg: str) -> None:
    print(f"[setup_appworld] {msg}", file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------- the plan
def load_manifest(path: Path) -> Dict[str, dict]:
    """bundle name -> its manifest record (``appworld/<name>.bundle``); {} without a readable manifest."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    records = data.get("files", []) if isinstance(data, dict) else data
    out = {}
    for rec in records:
        p = rec.get("path", "") if isinstance(rec, dict) else ""
        if p.startswith("appworld/") and p.endswith(".bundle") and p.count("/") == 1:
            out[p[len("appworld/") : -len(".bundle")]] = rec
    return out


def inner_root(name: str, rec: Optional[dict]) -> str:
    """The bundle's top-level directory: from the manifest's contents, else appworld_opsd_X -> opsd_appworld_X."""
    roots = {c["path"].split("/", 1)[0] for c in (rec or {}).get("contents") or [] if c.get("path")}
    if len(roots) == 1:
        return roots.pop()
    if name.startswith("appworld_opsd_"):
        return "opsd_appworld_" + name[len("appworld_opsd_") :]
    return name


class Bundle:
    def __init__(self, name: str, path: Path, rec: Optional[dict], plain: Path):
        self.name, self.path, self.rec, self.plain = name, path, rec, plain
        self.inner = inner_root(name, rec)
        self.want = (rec or {}).get("sha256") or PINNED.get(name)
        self.stamp = plain / f".{self.inner}{STAMP_SUFFIX}"

    @property
    def dir(self) -> Path:
        return self.plain / self.inner

    def expected(self, base: Optional[Path] = None) -> List[Tuple[str, Optional[int], str]]:
        """(path relative to the plain dir, bytes or None, sha256) of every file the manifest / CONTENTS.json name."""
        base = base or self.plain
        out = [
            (c["path"], c.get("bytes"), c["sha256"])
            for c in (self.rec or {}).get("contents") or []
            if c.get("path") and c.get("sha256")
        ]
        idx = base / self.inner / "CONTENTS.json"
        if self.name == PI and idx.is_file():
            data = json.loads(idx.read_text())
            for f in data["files"] if isinstance(data, dict) else data:
                if f.get("sha256"):
                    out.append((f"{self.inner}/{f['path']}", None, f["sha256"]))
        return out


def plan(data_root: Path, pi_bundle: Optional[Path]) -> List[Bundle]:
    aw = data_root / "appworld"
    plain = aw / "plain"
    manifest = load_manifest(data_root / "manifest.json")
    names = set(PINNED) | set(manifest) | {p.name[: -len(".bundle")] for p in aw.glob("appworld_opsd_*.bundle")}
    out = []
    for name in sorted(names, key=lambda n: (n != PI, n)):
        path = pi_bundle if (name == PI and pi_bundle) else aw / f"{name}.bundle"
        out.append(Bundle(name, path, manifest.get(name), plain))
    return out


# ---------------------------------------------------------------------------------------- verification
def scan(base: Path, inner: str) -> Dict[str, Tuple[int, str]]:
    """{path relative to base: (bytes, sha256)} of every file under base/inner."""
    out = {}
    root = base / inner
    for dirpath, _, files in os.walk(root):
        for fn in files:
            p = Path(dirpath) / fn
            out[str(p.relative_to(base))] = (p.stat().st_size, sha256_file(p))
    return out


def problems(expected: List[Tuple[str, Optional[int], str]], actual: Dict[str, Tuple[int, str]]) -> List[str]:
    bad = []
    for rel, size, sha in expected:
        if rel not in actual:
            bad.append(f"missing {rel}")
        elif actual[rel][1] != sha or (size is not None and actual[rel][0] != int(size)):
            bad.append(f"sha256 mismatch {rel}")
    return bad


def read_stamp(b: Bundle) -> Optional[dict]:
    try:
        return json.loads(b.stamp.read_text())
    except (OSError, ValueError):
        return None


def write_stamp(b: Bundle, actual: Dict[str, Tuple[int, str]], verified: bool) -> None:
    rec = {
        "bundle": b.name,
        "bundle_sha256": b.want,
        "inner_root": b.inner,
        "verified_against_manifest": verified,
        "files": {k: {"bytes": s, "sha256": h} for k, (s, h) in sorted(actual.items())},
    }
    tmp = b.stamp.with_name(b.stamp.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1) + "\n")
    os.replace(tmp, b.stamp)


def quick_ok(b: Bundle) -> bool:
    """A stamp for the expected bundle, every stamped file at its size, the manifest's sha256 values = the stamp's."""
    st = read_stamp(b)
    if not st or not b.dir.is_dir() or st.get("bundle_sha256") != b.want or not st.get("files"):
        return False
    files = st["files"]
    for rel, f in files.items():
        p = b.plain / rel
        if not p.is_file() or p.stat().st_size != f["bytes"]:
            return False
    return all(rel in files and files[rel]["sha256"] == sha for rel, _, sha in b.expected())


def state(b: Bundle, write: bool) -> str:
    """'ok' (stamp), 'verified' (no stamp, every expected file verified), 'unverifiable', 'incomplete', 'missing'."""
    if quick_ok(b):
        return "ok"
    if not b.dir.is_dir():
        return "missing"
    exp = b.expected()
    if not exp:
        return "unverifiable"
    actual = scan(b.plain, b.inner)
    bad = problems(exp, actual)
    for p in bad[:10]:
        log(f"    {b.name}: {p}")
    if bad:
        return "incomplete"
    if write and b.want:
        write_stamp(b, actual, verified=True)
    return "verified"


# --------------------------------------------------------------------------------------------- unpacking
def appworld_unpack(bundle: Path, base: Path, password: str, salt: str) -> None:
    from appworld.common.utils import unpack_bundle

    unpack_bundle(bundle_file_path=str(bundle), base_directory=str(base), password=password, salt=salt.encode())


def unpack(b: Bundle, password: str, salt: str, unpacker: Unpacker) -> None:
    """Decrypt into a scratch directory next to the target, verify, then swap it in. Raises RuntimeError."""
    got = sha256_file(b.path)
    if got != b.want:
        raise RuntimeError(
            f"sha256 of {b.path} is {got}, expected {b.want} (re-fetch it with python -m opsd.data.fetch)"
        )
    b.plain.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".unpack.", dir=b.plain))
    try:
        unpacker(b.path, tmp, password, salt)
        if not (tmp / b.inner).is_dir():
            raise RuntimeError(
                f"{b.path.name} did not contain {b.inner}/ (found {sorted(p.name for p in tmp.iterdir())})"
            )
        exp = b.expected(tmp)
        actual = scan(tmp, b.inner)
        bad = problems(exp, actual)
        if bad:
            raise RuntimeError(f"{b.path.name}: the unpacked files do not match the manifest: {bad[:10]}")
        if not exp:
            log(
                f"    WARNING {b.name}: no manifest record for it, so its files are not verified one by one "
                "(fetch with python -m opsd.data.fetch, which writes manifest.json)"
            )
        if b.stamp.exists():
            b.stamp.unlink()
        if b.dir.exists():
            shutil.rmtree(b.dir)
        os.replace(tmp / b.inner, b.dir)
        write_stamp(b, actual, verified=bool(exp))
        log(f"    {'verified' if exp else 'unpacked (unverified)'} {len(actual)} files -> {b.dir}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -------------------------------------------------------------------------------------------------- CLI
def _note(b: Bundle) -> str:
    return f"  [evaluation-split material: {EVAL_SPLIT[b.name]}]" if b.name in EVAL_SPLIT else ""


def check(bundles: List[Bundle]) -> int:
    for b in bundles:
        st = state(b, write=False)
        have = "present" if b.path.is_file() else "not fetched"
        plain = {
            "ok": f"OK ({b.dir})",
            "verified": f"OK, verified, no stamp yet ({b.dir})",
            "unverifiable": f"present, not verifiable without its manifest record ({b.dir})",
            "incomplete": f"INCOMPLETE ({b.dir})",
            "missing": "not unpacked",
        }[st]
        if st == "missing" and not b.path.is_file():
            log(f"bundle {b.name}: not fetched")
            continue
        log(f"bundle {b.name}: {have}; plain: {plain}{_note(b)}")
        if b.rec and b.name in PINNED and b.rec.get("sha256") != PINNED[b.name]:
            log(f"    NOTE: manifest.json lists another sha256 for it than dataset revision {REVISION[:8]}")
    return 0


def setup(bundles: List[Bundle], password: str, salt: str, unpacker: Unpacker = appworld_unpack) -> int:
    errors, absent = 0, []
    for b in bundles:
        present = b.path.is_file()
        st = state(b, write=True)
        if st in ("ok", "verified"):
            log(f"  {b.name}: already unpacked and verified ({b.dir}){_note(b)}")
            continue
        if not present:
            if st != "missing":
                why = "incomplete" if st == "incomplete" else "not verifiable (no stamp, no manifest record)"
                log(f"  WARNING {b.name}: {b.dir} is {why}, and {b.path} is not there to unpack it again")
            elif b.name == PI:
                log(f"  WARNING: no {b.path} (fetch it first: python -m opsd.data.fetch --repo <hf_dataset_id>")
                log("    --revision <commit_sha> --bench appworld, or --from-local <dir> --bench appworld).")
                log("    Without it there are no AppWorld PI hint files and no b0a8eae patch: only the DAPO /")
                log("    step-0 arms can run, on 87 train tasks (APPWORLD_ALLOW_UNPATCHED=1), which does not")
                log("    reproduce the paper.")
            else:
                absent.append(b.name)
            continue
        if not b.want:
            log(f"  WARNING {b.name}: no sha256 for {b.path.name} (not in manifest.json, not pinned): skipped")
            continue
        if b.rec and b.name in PINNED and b.rec["sha256"] != PINNED[b.name]:
            log(
                f"  NOTE {b.name}: manifest.json is from another dataset revision than {REVISION[:8]}; using its sha256"
            )
        log(f"  {b.name}: decrypting {b.path} -> {b.dir}{_note(b)}")
        try:
            unpack(b, password, salt, unpacker)
        except Exception as e:  # noqa: BLE001 - report every bundle, fail at the end
            log(f"  ERROR {b.name}: {e}")
            errors += 1
    if absent:
        log(f"  not fetched (optional): {', '.join(absent)}")
    return 1 if errors else 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("check", "setup"))
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--pi-bundle", type=Path, default=None, help="location of appworld_opsd_pi.bundle")
    ap.add_argument("--password", default=None)
    ap.add_argument("--salt", default=None)
    a = ap.parse_args(argv)
    bundles = plan(a.data_root, a.pi_bundle)
    if a.mode == "check":
        return check(bundles)
    if a.password is None or a.salt is None:
        ap.error("setup needs --password and --salt")
    return setup(bundles, a.password, a.salt)


if __name__ == "__main__":
    sys.exit(main())

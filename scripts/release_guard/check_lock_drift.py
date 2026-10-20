#!/usr/bin/env python3
"""Lock drift: is this repository's uv.lock the environment of SkyRL's own uv.lock at the pinned commit?

pyproject.toml mirrors SkyRL's [tool.uv] configuration, and its constraint-dependencies pin every package to its
version in SkyRL's uv.lock wherever this project's smaller resolution would otherwise pick another one, so that the
environment is the one the paper's runs used. This check holds uv.lock (OURS) to SkyRL's lock (SKYRL):

* every package of OURS is in SKYRL at the same version, from the same source (registry, git revision or URL) and
  with the same distribution files. Packages from a local path (this project, and skyrl / skyrl-gym from the SkyRL
  checkout) are compared by version only;
* OURS adds no package beyond this project itself and the allowed additions (default: the `tau2` extra's addict,
  cachebox, deepdiff and orderly-set; its toml is in SkyRL's lock too);
* every ``name==version`` in pyproject.toml's [tool.uv] constraint-dependencies is a version SKYRL has.

SKYRL may hold more (the extras this project never enables, other resolution forks of a package): that is listed,
not an error. Run it after `uv lock` whenever the SkyRL pin, pyproject.toml or uv.lock changes
(ADAPTED_FROM_SKYRL.md, "Bumping the SkyRL pin"); CI runs it after `uv lock --check`. From the
repository root::

    python scripts/release_guard/check_lock_drift.py      # uv.lock vs the lock of the [tool.uv.sources] checkout
    python scripts/release_guard/check_lock_drift.py --ours uv.lock --skyrl ../SkyRL-ecdaa946/uv.lock
    python scripts/release_guard/check_lock_drift.py --suggest   # + constraint lines for the drifted packages

Stdlib only; Python >= 3.11 (tomllib). Exit status: 0 = no drift, 1 = drift, 2 = an input could not be read.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    sys.exit("check_lock_drift.py needs Python >= 3.11 (tomllib): run it with .venv/bin/python")

ROOT = Path(__file__).resolve().parents[2]
# The `tau2` extra's packages that SkyRL's lock does not have (pyproject.toml [project.optional-dependencies]).
ALLOWED_ADDITIONS = ("addict", "cachebox", "deepdiff", "orderly-set")
LOCAL_KINDS = ("editable", "path", "directory", "virtual")
_PIN_RX = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;,]+)")


def normalize(name: str) -> str:
    """PEP 503 name: lower case, runs of - _ . as one -."""
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class Entry:
    version: str
    kind: str  # registry | git | url | editable | path | directory | virtual
    source: str
    files: frozenset

    @property
    def local(self) -> bool:
        return self.kind in LOCAL_KINDS

    def where(self) -> str:
        return f"{self.version} ({self.kind} {self.source})"


def _files(pkg: dict) -> frozenset:
    out = []
    for f in ([pkg["sdist"]] if pkg.get("sdist") else []) + list(pkg.get("wheels", [])):
        out.append(f.get("url") or f.get("path") or repr(sorted(f.items())))
    return frozenset(out)


def load_lock(path: Path) -> dict[str, list[Entry]]:
    """{normalized name: [Entry, ...]} for every [[package]] of a uv.lock."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[Entry]] = defaultdict(list)
    for pkg in data.get("package", []):
        src = pkg.get("source", {})
        kind = next((k for k in ("registry", "git", "url", *LOCAL_KINDS) if k in src), "unknown")
        source = str(src.get(kind, sorted(src.items())))
        out[normalize(pkg["name"])].append(Entry(str(pkg.get("version", "<dynamic>")), kind, source, _files(pkg)))
    return dict(out)


def constraint_pins(pyproject: Path) -> dict[str, str]:
    """{normalized name: version} of every `name==version` in [tool.uv] constraint-dependencies."""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    pins = {}
    for req in data.get("tool", {}).get("uv", {}).get("constraint-dependencies", []):
        m = _PIN_RX.match(req)
        if m:
            pins[normalize(m[1])] = m[2]
    return pins


def skyrl_lock_from_sources(pyproject: Path) -> Path | None:
    """<the [tool.uv.sources] skyrl path>/uv.lock, relative to pyproject.toml's directory (None: no such source)."""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    src = data.get("tool", {}).get("uv", {}).get("sources", {}).get("skyrl")
    if isinstance(src, dict) and src.get("path"):
        return (pyproject.parent / src["path"] / "uv.lock").resolve()
    return None


def _matches(pinned: str, version: str) -> bool:
    return version == pinned or version.split("+", 1)[0] == pinned  # torch==2.11.0 admits 2.11.0+cu128


def compare(
    ours: dict[str, list[Entry]],
    theirs: dict[str, list[Entry]],
    allowed: set[str],
    pins: dict[str, str] | None = None,
) -> tuple[list[str], list[str], dict[str, str]]:
    """(problems, notes, suggested pins {name: SkyRL's single version}); no problems = no drift."""
    problems: list[str] = []
    notes: list[str] = []
    suggest: dict[str, str] = {}
    for name in sorted(ours):
        mine = ours[name]
        if name not in theirs:
            if all(e.local and e.source == "." for e in mine):
                notes.append(f"this project: {name} {', '.join(e.version for e in mine)}")
            elif name in allowed:
                notes.append(f"allowed addition: {name} {', '.join(e.where() for e in mine)}")
            else:
                problems.append(f"only in ours: {name} {', '.join(e.where() for e in mine)}")
            continue
        sky = theirs[name]
        sky_versions = sorted({e.version for e in sky})
        for e in mine:
            same_version = [s for s in sky if s.version == e.version]
            if not same_version:
                problems.append(f"version drift: {name} {e.version} in ours, SkyRL has {sky_versions}")
                if len(sky_versions) == 1:
                    suggest[name] = sky_versions[0]
                continue
            if e.local:
                continue  # this checkout vs SkyRL's own tree: the path differs by construction
            same_source = [s for s in same_version if (s.kind, s.source) == (e.kind, e.source)]
            if not same_source:
                theirs_src = ", ".join(sorted({f"{s.kind} {s.source}" for s in same_version}))
                problems.append(
                    f"source drift: {name} {e.version} from {e.kind} {e.source} in ours, {theirs_src} in SkyRL's"
                )
            elif not any(s.files == e.files for s in same_source):
                problems.append(f"file drift: {name} {e.version} has other distribution files than in SkyRL's lock")
    for name, version in sorted((pins or {}).items()):
        if name not in theirs:
            problems.append(f"constraint {name}=={version} names a package SkyRL's lock does not have")
        elif not any(_matches(version, e.version) for e in theirs[name]):
            have = sorted({e.version for e in theirs[name]})
            problems.append(f"constraint {name}=={version}: SkyRL's lock has {have}")
    only_theirs = sorted(set(theirs) - set(ours))
    if only_theirs:
        notes.append(
            f"only in SkyRL's ({len(only_theirs)}; extras this project does not enable): {' '.join(only_theirs)}"
        )
    return problems, notes, suggest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", type=Path, default=ROOT / "uv.lock", help="this project's uv.lock (default: %(default)s)")
    ap.add_argument(
        "--skyrl",
        type=Path,
        default=None,
        help="SkyRL's uv.lock at the pin (default: <the [tool.uv.sources] skyrl path of the pyproject>/uv.lock)",
    )
    ap.add_argument(
        "--pyproject",
        type=Path,
        default=None,
        help="the pyproject.toml whose constraint-dependencies are checked (default: the one next to --ours)",
    )
    ap.add_argument("--no-pyproject", action="store_true", help="skip the constraint-dependencies check")
    ap.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="NAME",
        help=f"another package OURS may add (always allowed: {', '.join(ALLOWED_ADDITIONS)})",
    )
    ap.add_argument("--suggest", action="store_true", help="print constraint-dependencies lines for drifted packages")
    args = ap.parse_args(argv)

    pyproject = None if args.no_pyproject else (args.pyproject or args.ours.parent / "pyproject.toml")
    try:
        skyrl = args.skyrl
        if skyrl is None:
            src_pyproject = args.pyproject or args.ours.parent / "pyproject.toml"
            skyrl = skyrl_lock_from_sources(src_pyproject) if src_pyproject.is_file() else None
            if skyrl is None:
                print("check_lock_drift: no --skyrl and no [tool.uv.sources] skyrl path to find it", file=sys.stderr)
                return 2
        ours, theirs = load_lock(args.ours), load_lock(skyrl)
        pins = constraint_pins(pyproject) if pyproject is not None and pyproject.is_file() else None
    except (OSError, tomllib.TOMLDecodeError, KeyError) as e:
        print(f"check_lock_drift: cannot read the inputs: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    allowed = {normalize(n) for n in (*ALLOWED_ADDITIONS, *args.allow)}
    problems, notes, suggest = compare(ours, theirs, allowed, pins)
    print(f"ours:  {args.ours} ({len(ours)} packages)")
    print(f"SkyRL: {skyrl} ({len(theirs)} packages)")
    if pins is not None:
        print(f"constraints: {len(pins)} name==version pins in {pyproject}")
    for line in notes:
        print(f"  note  {line}")
    for line in problems:
        print(f"  DRIFT {line}")
    if args.suggest and suggest:
        print("suggested constraint-dependencies lines (SkyRL's version):")
        for name, version in sorted(suggest.items()):
            print(f'    "{name}=={version}",')
    print(f"RESULT: {'no drift' if not problems else f'DRIFT ({len(problems)})'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

"""Fetch (or verify) the OPSD data bundle into ``$OPSD_DATA_ROOT``.

    python -m opsd.data.fetch --repo <hf_dataset_id> --revision <commit_sha> [--bench alfworld,tau2]
    python -m opsd.data.fetch --from-local /path/to/unpacked/bundle [--bench appworld]
    python -m opsd.data.fetch --verify-only [--dest DIR]

The bundle root holds ``manifest.json``: every file with its ``path`` (relative to the root), ``sha256``
and ``bytes`` (plus descriptive fields such as ``benchmark``, ``kind``, ``status``). Only manifest files
are fetched, and every fetched file is verified against its sha256; a mismatch is an error (exit 1) and
the file is reported and moved aside (``<file>.sha256-mismatch``), never silently kept as good.
``--revision`` should be a commit sha so the data cannot change under a published result.

Selection: ``--bench`` (first path component: alfworld, appworld, tau2), ``--tier`` (A = training inputs:
splits, hints, reference trajectories, passcode; B = ``pi_metrics/`` metric inputs; C = optional ``corpora/``;
a manifest ``tier`` field wins), ``--include`` fnmatch patterns on the path.

Public repositories need no credentials (``huggingface_hub`` uses ``HF_TOKEN`` only if you have one).
The AppWorld files ship only as encrypted bundles (``appworld/appworld_opsd_*.bundle``, eight of them; the
manifest ``tier`` field places the PI bundle ``appworld_opsd_pi`` and the evaluation-split hint bundles
``appworld_opsd_dev_hints``, ``appworld_opsd_xdnull`` and ``appworld_opsd_test_normal_hints`` in A, the label bundle
``appworld_opsd_metrics`` in B and the corpora in C). After
fetching, ``scripts/setup/setup_appworld.sh`` verifies and decrypts every one present. The AppWorld arms need
only ``appworld_opsd_pi``; the ALFWorld ``null_xdomain`` arm needs ``appworld_opsd_xdnull``, which
``--bench alfworld`` does not select (``--include appworld/appworld_opsd_xdnull.bundle``).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Sequence

MANIFEST = "manifest.json"
EXTRA_FILES = ("README.md",)  # fetched alongside the manifest when present (never verified/required)
FETCH_RECORD = ".opsd_fetch.json"
TIERS = ("A", "B", "C")


class FetchError(Exception):
    pass


def default_dest() -> Path:
    env = os.environ.get("OPSD_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "opsd" / "data"


@dataclass
class Entry:
    path: str
    sha256: Optional[str]
    bytes: Optional[int]
    bench: str
    tier: str
    status: Optional[str]


def _safe_rel(p: str) -> str:
    pp = PurePosixPath(p)
    if pp.is_absolute() or ".." in pp.parts or not p or p.startswith("~"):
        raise FetchError(f"manifest path {p!r} is not a safe relative path")
    return str(pp)


def _tier(rec: dict, path: str) -> str:
    t = rec.get("tier")
    if t:
        return str(t).upper()
    parts = path.split("/")
    if "corpora" in parts:
        return "C"
    if "pi_metrics" in parts:
        return "B"
    return "A"


def parse_manifest(text: str) -> List[Entry]:
    """Accepts ``{"files": [...]}``, a bare list, or JSON lines; each record needs ``path``."""
    text = text.strip()
    if not text:
        raise FetchError("empty manifest")
    try:
        data = json.loads(text)
        records = data.get("files", data.get("entries")) if isinstance(data, dict) else data
    except json.JSONDecodeError:
        records = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    if not isinstance(records, list):
        raise FetchError("manifest must be a list of file records or {'files': [...]}")
    out, seen = [], set()
    for rec in records:
        if not isinstance(rec, dict) or "path" not in rec:
            raise FetchError(f"manifest record without a path: {rec!r}")
        path = _safe_rel(rec["path"])
        if path in seen:
            raise FetchError(f"manifest lists {path} twice")
        seen.add(path)
        sha = rec.get("sha256")
        if sha is not None and not re.fullmatch(r"[0-9a-f]{64}", str(sha)):
            raise FetchError(f"{path}: malformed sha256 {sha!r}")
        out.append(
            Entry(
                path,
                sha,
                rec.get("bytes"),
                rec.get("benchmark") or path.split("/", 1)[0],
                _tier(rec, path),
                rec.get("status"),
            )
        )
    return out


def select(
    entries: Sequence[Entry],
    benches: Optional[Sequence[str]] = None,
    tiers: Optional[Sequence[str]] = None,
    include: Optional[Sequence[str]] = None,
) -> List[Entry]:
    out = []
    for e in entries:
        if benches and e.bench not in benches:
            continue
        if tiers and e.tier not in tiers:
            continue
        if include and not any(fnmatch.fnmatch(e.path, pat) for pat in include):
            continue
        out.append(e)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root: Path, entries: Sequence[Entry]) -> Dict[str, List[str]]:
    """{"ok": [...], "missing": [...], "mismatch": [...], "unverified": [...]} for ``entries`` under ``root``."""
    res: Dict[str, List[str]] = {"ok": [], "missing": [], "mismatch": [], "unverified": []}
    for e in entries:
        p = root / e.path
        if not p.is_file():
            res["missing"].append(e.path)
        elif not e.sha256:
            res["unverified"].append(e.path)
        elif (e.bytes is not None and p.stat().st_size != int(e.bytes)) or sha256_file(p) != e.sha256:
            res["mismatch"].append(e.path)
        else:
            res["ok"].append(e.path)
    return res


# -------------------------------------------------------------------------------------------- sources
def fetch_local(src: Path, dest: Path, entries: Sequence[Entry], link: bool = False) -> None:
    for e in entries:
        s, d = src / e.path, dest / e.path
        if not s.is_file():
            continue  # reported as missing by verify()
        if d.resolve() == s.resolve():
            continue  # verifying in place
        if d.is_file() and e.sha256 and d.stat().st_size == s.stat().st_size and sha256_file(d) == e.sha256:
            continue  # already there and good
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.exists() or d.is_symlink():
            d.unlink()
        if link:
            os.link(s, d)
        else:
            shutil.copy2(s, d)


def _hf():
    try:
        import huggingface_hub
    except ImportError as e:  # pragma: no cover - environment dependent
        raise FetchError("--repo needs huggingface_hub (pip install huggingface_hub)") from e
    return huggingface_hub


def hf_manifest(repo: str, revision: str, cache_dir: Optional[str] = None) -> str:
    hub = _hf()
    p = hub.hf_hub_download(
        repo_id=repo, filename=MANIFEST, repo_type="dataset", revision=revision, cache_dir=cache_dir
    )
    return Path(p).read_text()


def fetch_hf(repo: str, revision: str, dest: Path, entries: Sequence[Entry], max_workers: int = 8) -> None:
    hub = _hf()
    patterns = [MANIFEST, *EXTRA_FILES, *[e.path for e in entries]]
    hub.snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dest),
        allow_patterns=patterns,
        max_workers=max_workers,
    )


# ------------------------------------------------------------------------------------------------ CLI
def _csv(s: Optional[str]) -> Optional[List[str]]:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m opsd.data.fetch",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="Hugging Face dataset id")
    src.add_argument("--from-local", type=Path, help="a local copy of the bundle (holding manifest.json)")
    src.add_argument("--verify-only", action="store_true", help="only verify what is already under --dest")
    ap.add_argument("--revision", help="dataset commit sha (required with --repo)")
    ap.add_argument("--dest", type=Path, default=None, help="default: $OPSD_DATA_ROOT or ~/.cache/opsd/data")
    ap.add_argument("--bench", help="comma list: alfworld,appworld,tau2 (default: all)")
    ap.add_argument("--tier", help="comma list of A,B,C (default: all)")
    ap.add_argument("--include", action="append", help="fnmatch pattern on the manifest path (repeatable)")
    ap.add_argument("--link", action="store_true", help="--from-local: hard-link instead of copying")
    ap.add_argument("--list", action="store_true", help="print the selected files and exit")
    ap.add_argument("--max-workers", type=int, default=8)
    a = ap.parse_args(argv)
    dest = (a.dest or default_dest()).expanduser().resolve()
    tiers = [t.upper() for t in _csv(a.tier) or []] or None
    if tiers and set(tiers) - set(TIERS):
        ap.error(f"--tier must be a subset of {','.join(TIERS)}")
    try:
        if a.repo:
            if not a.revision:
                raise FetchError("--repo needs --revision <commit sha> (pin the data version)")
            if not re.fullmatch(r"[0-9a-f]{40}", a.revision):
                print(
                    f"WARNING: --revision {a.revision!r} is not a commit sha; branch/tag contents can change",
                    file=sys.stderr,
                )
            manifest_text = hf_manifest(a.repo, a.revision)
        elif a.from_local:
            mp = a.from_local / MANIFEST
            if not mp.is_file():
                raise FetchError(f"{mp} not found")
            manifest_text = mp.read_text()
        else:
            mp = dest / MANIFEST
            if not mp.is_file():
                raise FetchError(f"{mp} not found (nothing fetched into {dest} yet?)")
            manifest_text = mp.read_text()
        entries = parse_manifest(manifest_text)
        sel = select(entries, _csv(a.bench), tiers, a.include)
        if not sel:
            raise FetchError("the selection matches no manifest entry")
        if a.list:
            for e in sel:
                print(f"{e.tier}\t{e.bench}\t{e.bytes if e.bytes is not None else '?'}\t{e.path}")
            return 0
        total = sum(int(e.bytes or 0) for e in sel)
        print(f"{len(sel)} file(s), {total / 1e9:.2f} GB -> {dest}", file=sys.stderr)
        dest.mkdir(parents=True, exist_ok=True)
        if a.repo:
            fetch_hf(a.repo, a.revision, dest, sel, a.max_workers)
        elif a.from_local:
            fetch_local(a.from_local, dest, sel, a.link)
            for name in (MANIFEST, *EXTRA_FILES):
                s = a.from_local / name
                if s.is_file() and (dest / name).resolve() != s.resolve():
                    shutil.copy2(s, dest / name)
        res = verify(dest, sel)
    except FetchError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    for kind in ("missing", "mismatch"):
        for p in res[kind]:
            print(f"{kind.upper()}: {p}", file=sys.stderr)
    if not a.verify_only and not (a.from_local and a.from_local.resolve() == dest):
        for p in res["mismatch"]:  # quarantine, so a later run can never pick up a corrupt file
            bad = dest / (p + ".sha256-mismatch")
            os.replace(dest / p, bad)
            print(f"  moved aside -> {bad}", file=sys.stderr)
    for p in res["unverified"]:
        print(f"WARNING: no sha256 in the manifest for {p} (not verified)", file=sys.stderr)
    in_progress = [e.path for e in sel if e.status and e.status != "final"]
    if in_progress:
        print(f"NOTE: {len(in_progress)} selected file(s) are marked status != final in the manifest", file=sys.stderr)
    if not a.verify_only:
        rec = {
            "source": a.repo or str(a.from_local),
            "revision": a.revision,
            "fetched_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bench": _csv(a.bench),
            "tier": tiers,
            "include": a.include,
            "files": len(sel),
            "ok": len(res["ok"]),
            "missing": res["missing"],
            "mismatch": res["mismatch"],
        }
        (dest / FETCH_RECORD).write_text(json.dumps(rec, indent=2) + "\n")
    print(
        f"verified {len(res['ok'])}/{len(sel)} file(s) under {dest}"
        + (
            f"; {len(res['missing'])} missing, {len(res['mismatch'])} sha256 mismatch"
            if res["missing"] or res["mismatch"]
            else ""
        )
    )
    return 1 if (res["missing"] or res["mismatch"]) else 0


if __name__ == "__main__":
    sys.exit(main())

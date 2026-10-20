"""ADAPTED_FROM_SKYRL.md lists exactly the files that carry an ``Adapted from SkyRL`` header.

Every file of this repository (``opsd/``, ``scripts/``) that copies or closely adapts upstream code starts with one
header line per upstream source::

    # Adapted from SkyRL (Apache-2.0), <upstream path> @ <short sha>; modified for OPSD: <what changed>

The attestation page is generated from those headers, so the two must agree: the same set of files, the same
upstream paths per file, every upstream path present in the installed SkyRL checkout, and every ``@ <sha>`` (and
every GitHub link on the page) at the commit the repository pins (``opsd._repo.SKYRL_COMMIT``).
"""

import re
import subprocess
from pathlib import Path

import pytest

from opsd._repo import SKYRL_COMMIT, package_dir, repo_root, skyrl_root

ROOT = repo_root()
PKG = package_dir()
SKYRL = skyrl_root()
DOC = repo_root() / "ADAPTED_FROM_SKYRL.md"
# The trees of this repository that hold Python code.
CODE_DIRS = (PKG, ROOT / "scripts")

HEADER_RX = re.compile(r"^# Adapted from SkyRL \(Apache-2\.0\), (?P<path>\S+) @ (?P<sha>[0-9a-f]{7,40});")
LINK_RX = re.compile(
    r"\[`(?P<shown>skyrl/[^`]+)`\]\(https://github\.com/NovaSky-AI/SkyRL/blob/(?P<sha>[0-9a-f]{40})/(?P<path>[^)]+)\)"
)
PIN_RX = re.compile(r"https://github\.com/NovaSky-AI/SkyRL/tree/(?P<sha>[0-9a-f]{40})")


def _headers():
    """{repo-relative file: [(upstream path, sha), ...]} for every .py under opsd/ or scripts/ with a header line."""
    out = {}
    for path in sorted(p for d in CODE_DIRS for p in d.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        found = [(i, m) for i, line in enumerate(lines) if (m := HEADER_RX.match(line))]
        if not found:
            continue
        rel = path.relative_to(ROOT).as_posix()
        # Headers belong to the leading comment block, before any code or docstring.
        first_code = next((i for i, line in enumerate(lines) if line.strip() and not line.startswith("#")), len(lines))
        late = [i + 1 for i, _ in found if i > first_code]
        assert not late, f"{rel}: 'Adapted from SkyRL' header after the first code line (lines {late})"
        out[rel] = [(m["path"], m["sha"]) for _, m in found]
    return out


def _doc_rows():
    """{our file: [(shown upstream path, link sha, link path), ...]} from the first table of the page."""
    rows = {}
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| [`opsd/"):
            continue
        ours = re.match(r"\| \[`(?P<f>opsd/[^`]+)`\]", line)["f"]
        upstream_cell = line.split("|")[2]
        rows[ours] = [(m["shown"], m["sha"], m["path"]) for m in LINK_RX.finditer(upstream_cell)]
    return rows


def _pinned_sha():
    m = PIN_RX.search(DOC.read_text(encoding="utf-8"))
    assert m, f"{DOC} names no SkyRL tree/<sha> pin"
    return m["sha"]


def test_the_page_lists_exactly_the_files_with_a_header():
    headers, rows = _headers(), _doc_rows()
    assert headers, "no 'Adapted from SkyRL' header found under opsd/ or scripts/"
    assert set(rows) == set(headers), (
        f"only in ADAPTED_FROM_SKYRL.md: {sorted(set(rows) - set(headers))}; "
        f"only carrying a header: {sorted(set(headers) - set(rows))}"
    )


def test_each_row_names_the_upstream_sources_of_its_headers():
    headers, rows = _headers(), _doc_rows()
    for ours, links in rows.items():
        assert links, f"{ours}: the row links no upstream source"
        for shown, _, path in links:
            assert shown == path, f"{ours}: link text {shown} points at {path}"
        assert sorted({p for p, _, _ in links}) == sorted({p for p, _ in headers.get(ours, [])}), ours


def test_every_upstream_path_exists_in_the_skyrl_checkout():
    for ours, entries in _headers().items():
        for path, _ in entries:
            assert (SKYRL / path).is_file(), f"{ours}: header names {path}, which is not in the SkyRL checkout {SKYRL}"


def test_every_sha_is_the_skyrl_pin():
    pin = _pinned_sha()
    assert pin == SKYRL_COMMIT, f"ADAPTED_FROM_SKYRL.md pins SkyRL at {pin}, opsd._repo.SKYRL_COMMIT is {SKYRL_COMMIT}"
    for ours, entries in _headers().items():
        for path, sha in entries:
            assert pin.startswith(sha), f"{ours}: header for {path} is @ {sha}, the page pins {pin}"
    for ours, links in _doc_rows().items():
        for _, sha, path in links:
            assert sha == pin, f"{ours}: the link to {path} is at {sha}, the page pins {pin}"
    # The checkout the upstream paths were checked against is the pinned one (when it is a git checkout).
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=SKYRL, capture_output=True, text=True, check=True
        ).stdout.strip()
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=SKYRL, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        top = head = None
    if top is None or Path(top).resolve() != SKYRL:
        pytest.skip(f"the installed SkyRL ({SKYRL}) is not a git checkout")
    assert head == pin, f"the installed SkyRL checkout {SKYRL} is at {head}, ADAPTED_FROM_SKYRL.md pins {pin}"


def test_every_file_on_the_page_has_a_drift_and_parity_row():
    text = DOC.read_text(encoding="utf-8")
    for ours in _doc_rows():
        name = ours.rsplit("/", 1)[1]
        assert re.search(
            rf"^\| `{re.escape(name)}` \|", text, re.M
        ), f"{name}: no row in 'How the copies are kept honest'"


def test_header_free_skyrl_ext_modules_are_named_on_the_page():
    """The page names the skyrl_ext modules that extend SkyRL without copying it; every other module has a header."""
    m = re.search(r"Other `skyrl_ext` modules\s*\(([^)]*)\)", DOC.read_text(encoding="utf-8"))
    assert m, "the page no longer names the header-free skyrl_ext modules"
    named = set(re.findall(r"`([a-z_]+\.py)`", m.group(1)))
    ext = PKG / "skyrl_ext"
    modules = {p.name for p in ext.glob("*.py") if p.name != "__init__.py"}
    with_header = {rel.rsplit("/", 1)[1] for rel in _headers() if rel.startswith("opsd/skyrl_ext/")}
    assert named == modules - with_header, (named, sorted(modules - with_header))

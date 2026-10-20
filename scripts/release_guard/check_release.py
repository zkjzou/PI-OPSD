#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Release guard: block secrets, host-local paths, private infrastructure references and stray
artifacts from entering this repository, and keep SkyRL out of it.

Layout: every tracked file is ours. SkyRL (NovaSky-AI/SkyRL@ecdaa946) is an external, import-only checkout next
to the repository, installed through pyproject.toml's [tool.uv.sources]; nothing of it is tracked here. There is
no upstream base to diff against, so every tracked file is scanned in full with every rule. Every git mode also
checks the layout: no submodule (gitlink), no .gitmodules, and no tracked path under SkyRL/ or SkyRL-*/ or in a
vendored upstream package directory (skyrl/, skyrl_gym/, skyrl-gym/, ...). History mode allows one exception:
the commits this branch inherits from the earlier submodule layout carry a SkyRL gitlink, which must point at
UPSTREAM_BASE with a .gitmodules declaring exactly that submodule, and once a commit has dropped it no
descendant may bring it back.

Stdlib only. Run from the repository root. Typical uses::

    # whole tree (tracked files) + the layout of HEAD, the index and the worktree
    python scripts/release_guard/check_release.py
    # staged files (pre-commit) + the layout of the index
    python scripts/release_guard/check_release.py --staged
    # the layout of every commit, every blob and every commit message of a range A..B, or of the whole
    # history of one rev (an orphan branch has no base)
    python scripts/release_guard/check_release.py --history-only --all-commits HEAD --commits HEAD
    # a data directory (json/jsonl/csv/md/txt)
    python scripts/release_guard/check_release.py --tree path/to/bundle --no-git

Extra, maintainer-only pattern files can be supplied with ``--extra-denylist FILE`` (one regex per line,
``WARN:`` prefix for warnings, ``PATH:`` prefix for a forbidden path glob) and ``--harvested FILE`` (exact
tokens). They are intentionally not part of
this repository. ``--known-secrets-file ENVFILE`` scans for the VALUES of KEY=VALUE lines without ever
printing them.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

UPSTREAM_BASE = "ecdaa9460e281774638a9170d6724456f46c8356"  # the NovaSky-AI/SkyRL commit the install pins
# The submodule of the earlier layout: only commits inherited from it may carry this gitlink (history mode).
LEGACY_GITLINK = "SkyRL"
LEGACY_URL = "https://github.com/NovaSky-AI/SkyRL.git"
# Upstream checkouts and vendored copies, at any depth: a directory named SkyRL or SkyRL-<anything> (any case,
# which covers upstream's skyrl/ package and its skyrl-gym/, skyrl-train/, ... projects) or one of upstream's
# other package directories. Our own opsd/skyrl_ext/ does not match.
UPSTREAM_DIR_RX = re.compile(r"skyrl(?:-.*)?", re.I)
UPSTREAM_PKG_DIRS = {"skyrl_gym", "skyrl_train", "skyrl_agent", "skyrl_tx"}
OURS_DIR = "opsd"  # the package; FORBIDDEN_GLOBS name root paths and are matched at the root and in here
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------------------------------
# Pattern tiers. scope "all": secret values, scanned on everything (incl. tracked upstream files, which are
# a layout FAIL anyway). scope "diff": host/infra rules, scanned on everything we own: every tracked file
# (whole files; there is no upstream base), every --only path and every blob of a history range.
# ---------------------------------------------------------------------------------------------------
@dataclass
class Rule:
    rid: str
    rx: re.Pattern
    scope: str  # "all" | "diff"
    level: str  # "FAIL" | "WARN"
    secret: bool = False


def _r(rid, pat, scope="diff", level="FAIL", secret=False, flags=0):
    return Rule(rid, re.compile(pat, flags), scope, level, secret)


RULES = [
    # S1 secret values
    _r(
        "S1.openai_anthropic",
        r"(?<![A-Za-z0-9_-])sk-(?:proj-|ant-(?:api\d{2}-)?)?[A-Za-z0-9_\-]{20,}",
        "all",
        secret=True,
    ),
    _r("S1.hf_token", r"\bhf_[A-Za-z0-9]{30,}\b", "all", secret=True),
    _r("S1.aws_key_id", r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[0-9A-Z]{16}\b", "all", secret=True),
    _r("S1.aws_secret", r"aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+]{40}", "all", secret=True, flags=re.I),
    _r("S1.private_key", r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY", "all", secret=True),
    _r("S1.github", r"\bgh[pousr]_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{60,}\b", "all", secret=True),
    _r("S1.slack", r"\bxox[abprs]-[A-Za-z0-9-]{10,}", "all", secret=True),
    _r("S1.google", r"\bAIza[0-9A-Za-z_\-]{35}\b", "all", secret=True),
    _r("S1.fireworks", r"\bfw_[A-Za-z0-9]{20,}\b", "all", secret=True),
    _r("S1.replicate", r"\br8_[A-Za-z0-9]{30,}\b", "all", secret=True),
    _r(
        "S1.wandb",
        r"wandb_api_key\s*[=:]\s*['\"]?[0-9a-f]{40}\b|\bwandb_v1_[A-Za-z0-9_]{20,}",
        "all",
        secret=True,
        flags=re.I,
    ),
    _r("S1.bearer", r"authorization:\s*bearer\s+[A-Za-z0-9._\-]{16,}", "all", secret=True, flags=re.I),
    _r("S1.url_creds", r"https?://[^/\s:@'\"<>]+:[^/\s:@'\"<>]+@[^\s'\"<>]+", "all", secret=True),
    _r("S1.presigned", r"X-Amz-Signature=|X-Amz-Credential=", "all", secret=True),
    _r(
        "S1w.assignment",
        r"\b(?:api[_-]?key|secret|token|passw(?:or)?d)\b['\"]?\s*[:=]\s*['\"][A-Za-z0-9_\-./+=]{16,}['\"]",
        "all",
        "WARN",
        secret=True,
        flags=re.I,
    ),
    # S2 reading secret files / credential harvesting
    _r("S2.secret_files", r"\.env\.vars\b|\.zshrc\b|\.bashrc\b|\.bash_profile\b|\.aws/credentials|\.netrc\b"),
    # S3 host-local paths and cloud storage
    _r("S3.host_paths", r"/mnt/(?:efs|nvme)\b|/opt/dlami\b|/home/ray\b"),
    _r(
        "S3.object_store",
        r"\bs3://|\.s3[.-][a-z0-9-]*\.?amazonaws\.com|\.dkr\.ecr\.|codeartifact|\bAWS_PROFILE\b"
        r"|\baws\s+s3\s+(?:cp|sync|ls|mv|rm)\b|\bs5cmd\b",
        flags=re.I,
    ),
    _r("S3.cloud_hosts", r"\bip-\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}\b|\.(?:ec2|compute)\.internal\b"),
    # S4 private networking (use RFC 5737 / *.example placeholders instead)
    _r(
        "S4.private_ip",
        r"\b10\.(?:\d{1,3}\.){2}\d{1,3}\b|\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"
        r"|\b192\.168\.\d{1,3}\.\d{1,3}\b",
    ),
    _r("S4.internal_dns", r"\b[a-z0-9-]+\.(?:internal|corp|svc\.cluster\.local)\b", level="WARN"),
    # S5 cluster / orchestration / observability
    _r(
        "S5.orchestration",
        r"\b(?:gke|eks|jobset|kueue|datadog)\b|\bDD_(?:SERVICE|ENV|API_KEY|AGENT_HOST)\b|\bJOB_COMPLETION_INDEX\b",
        flags=re.I,
    ),
    _r("S5.instance_types", r"\bFI_EFA_|aws-ofi-nccl"),
    _r("S5.job_ids", r"\bjob_[a-z0-9]{20}\b"),
    _r("S5.wandb_run_links", r"wandb\.ai/[^\s)]+/runs/"),
    # S6 identity (manual review)
    _r("S6.email", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", level="WARN"),
    # S7 cost statements in docs
    _r(
        "S7.cost",
        r"(?:cost|spend|spent|budget|price).{0,40}\$\s?\d|\$\s?\d[\d,]*(?:\.\d+)?.{0,40}(?:cost|spend|budget)",
        level="WARN",
        flags=re.I,
    ),
]

FORBIDDEN_GLOBS = [
    "*.env.vars",
    "*.pids",
]
EXTRA_GLOBS: list[str] = []  # more forbidden path globs, from "PATH:" lines of --extra-denylist files
FORBIDDEN_EXT = {
    ".parquet",
    ".pt",
    ".pth",
    ".safetensors",
    ".bin",
    ".ckpt",
    ".npz",
    ".pkl",
    ".zip",
    ".tar",
    ".tgz",
    ".gz",
    ".zst",
    ".bundle",
    ".db",
    ".sqlite",
    ".log",
    ".tw-pddl",
    ".pyc",
    ".pyo",
}
TEXT_EXT = {
    ".py",
    ".sh",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".cfg",
    ".ini",
    ".json",
    ".jsonl",
    ".csv",
    ".env",
    ".jinja",
    ".jinja2",
    ".j2",
    ".tex",
    ".rst",
    ".ts",
    ".tsx",
    ".js",
    ".mjs",
    ".html",
    ".lock",  # uv.lock is ours in this layout: an absolute source path or a private index would show up here
    ".svg",  # assets/ diagrams are XML text
    "",
}
SIZE_WARN, SIZE_FAIL = 1 << 20, 5 << 20


def git(*args, cwd=None, check=True) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True).stdout


def git_run(*args, cwd=None, data: bytes | None = None) -> subprocess.CompletedProcess:
    """Unchecked git call with bytes in/out (return code and stderr are the caller's business)."""
    return subprocess.run(["git", *args], cwd=cwd, input=data, capture_output=True)


def git_paths(*args, cwd=None) -> list[str]:
    """Path listing with -z, so unusual file names are neither quoted nor split."""
    return [p for p in git(*args, "-z", cwd=cwd).split("\0") if p]


def load_allowlist() -> list[tuple[str, str]]:
    path = HERE / "allowlist.txt"
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                out.append((parts[0], parts[1]))
    return out


def allowed(allow, path: str, rid: str) -> bool:
    """`path` is a repository path, or path@sha for a blob of a history range. A glob that contains "@" matches
    only such blob labels (an exemption for older commits alone); every other glob matches the path part."""
    plain = path.split("@", 1)[0]
    return any(fnmatch.fnmatch(path if "@" in g else plain, g) and (r == "*" or rid.startswith(r)) for g, r in allow)


def load_extra(path: str | None) -> list[Rule]:
    rules = []
    if path:
        for i, line in enumerate(Path(path).read_text().splitlines()):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#") or line.startswith("PATH:"):
                continue
            level = "FAIL"
            if line.startswith("WARN:"):
                level, line = "WARN", line[5:].strip()
            rules.append(Rule(f"X{i}", re.compile(line), "diff", level))
    return rules


def load_extra_globs(path: str | None) -> list[str]:
    """Forbidden path globs from the "PATH: <glob>" lines of an extra denylist file."""
    if not path:
        return []
    return [line[5:].strip() for line in Path(path).read_text().splitlines() if line.startswith("PATH:")]


def load_harvested(path: str | None) -> set[str]:
    if not path:
        return set()
    return {t.strip() for t in Path(path).read_text().splitlines() if t.strip() and not t.startswith("#")}


def load_known_secret_values(path: str | None) -> list[tuple[str, str]]:
    vals = []
    if path and os.path.exists(path):
        for line in open(path, errors="ignore"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                v = v.strip().strip("'\"")
                if len(v) >= 12:
                    vals.append((k.replace("export", "").strip(), v))
    return vals


class Report:
    def __init__(self):
        self.fail = 0
        self.warn = 0

    def emit(self, level, path, line, rid, snippet, secret=False):
        if level == "FAIL":
            self.fail += 1
        else:
            self.warn += 1
        shown = "<masked>" if secret else snippet.strip()[:120]
        print(f"{level} {path}:{line} {rid} {shown}")


def scan_text(rep, allow, path, text, rules, harvested, known, added_lines=None):
    lines = text.splitlines()
    for i, ln in enumerate(lines, 1):
        in_diff = added_lines is None or i in added_lines
        for rule in rules:
            if rule.scope == "diff" and not in_diff:
                continue
            m = rule.rx.search(ln)
            if m and not allowed(allow, path, rule.rid):
                rep.emit(rule.level, path, i, rule.rid, ln, rule.secret)
        if in_diff and harvested:
            for tok in re.findall(r"\b[0-9a-f]{7,40}\b|[\w.-]+", ln):
                if tok in harvested and not allowed(allow, path, "H"):
                    rep.emit("FAIL", path, i, "H.harvested", "<internal identifier>", True)
    for k, v in known:
        if v in text:
            rep.emit("FAIL", path, 0, f"K.known_secret({k})", "", True)


def upstream_dir(p: str) -> str | None:
    """The outermost directory of path `p` that is an upstream checkout or a vendored upstream package."""
    parts = p.split("/")
    for i, part in enumerate(parts[:-1]):
        if UPSTREAM_DIR_RX.fullmatch(part) or part in UPSTREAM_PKG_DIRS:
            return "/".join(parts[: i + 1])
    return None


def is_upstream(p: str) -> bool:
    return upstream_dir(p) is not None


def forbidden_path(p: str) -> bool:
    # the globs name root paths; they are also matched under the package (opsd/a.env.vars is as bad as a.env.vars)
    cands = [p, p[len(OURS_DIR) + 1 :]] if p.startswith(OURS_DIR + "/") else [p]
    return any(fnmatch.fnmatch(c, g) for c in cands for g in FORBIDDEN_GLOBS + EXTRA_GLOBS)


def path_rules(rep, allow, p, size=None, label=None):
    """Name- and size-based rules for one path (a tracked file, or a blob of a history range)."""
    label = label or p
    if forbidden_path(p) and not allowed(allow, p, "F"):
        rep.emit("FAIL", label, 0, "F.forbidden_path", "")
    ext = Path(p).suffix.lower()
    if ext in FORBIDDEN_EXT and not allowed(allow, p, "F"):
        rep.emit("FAIL", label, 0, "F.forbidden_ext", ext)
    if Path(p).name.startswith(".env") and not allowed(allow, p, "F"):
        rep.emit("FAIL", label, 0, "F.env_file", "")
    if size is not None:
        if size > SIZE_FAIL and not allowed(allow, p, "Z"):
            rep.emit("FAIL", label, 0, "Z.size", f"{size} bytes")
        elif size > SIZE_WARN and not allowed(allow, p, "Z"):
            rep.emit("WARN", label, 0, "Z.size", f"{size} bytes")


def structural(rep, allow, paths, root: Path, check_size=True):
    for p in paths:
        fp = root / p
        if fp.is_symlink():
            path_rules(rep, allow, p)
            tgt = os.readlink(fp)
            if os.path.isabs(tgt) or ".." in Path(tgt).parts:
                rep.emit("FAIL", p, 0, "F.symlink", tgt)
        else:
            path_rules(rep, allow, p, fp.stat().st_size if check_size and fp.is_file() else None)


def scan_files(rep, allow, root: Path, files, rules, harvested, known):
    """Full scan of tracked files. Ours get every rule; a vendored upstream file (already a layout FAIL) gets
    the secret tier only."""
    structural(rep, allow, [p for p in files if not is_upstream(p)], root)
    secret_tier = [r for r in rules if r.scope == "all"]
    for p in files:
        if Path(p).suffix.lower() not in TEXT_EXT or not (root / p).is_file():
            continue
        text = (root / p).read_text(errors="ignore")
        if is_upstream(p):
            scan_text(rep, allow, p, text, secret_tier, set(), known, set())
        else:
            scan_text(rep, allow, p, text, rules, harvested, known)


# ---------------------------------------------------------------------------------------------------
# The layout: SkyRL stays outside (U.* findings are never allowlisted)
# ---------------------------------------------------------------------------------------------------
def tree_entries(root: Path, where: str) -> list[tuple[str, str, str]]:
    """(mode, sha, path) of every entry of a commit (`where` = a rev) or of the index (`where` = ":"). A gitlink
    is one entry with mode 160000; it is not descended into."""
    out = []
    if where == ":":
        recs = git("ls-files", "-s", "-z", cwd=root).split("\0")
    else:
        recs = git("ls-tree", "-r", "-z", where, cwd=root).split("\0")
    for rec in filter(None, recs):
        meta, path = rec.split("\t", 1)
        f = meta.split()
        out.append((f[0], f[1] if where == ":" else f[2], path))
    return out


def gitmodules_problems(listing: str | None) -> list[str]:
    """Problems in a `git config --list` dump of the .gitmodules of an inherited submodule-layout commit (None:
    missing or unreadable)."""
    if listing is None:
        return [f".gitmodules is missing or unreadable (a commit with the {LEGACY_GITLINK} gitlink needs it)"]
    found = {}
    for line in listing.splitlines():
        key, _, val = line.partition("=")
        sect, _, rest = key.partition(".")
        name, _, field = rest.rpartition(".")
        found[(sect, name, field)] = val
    want = {("submodule", LEGACY_GITLINK, "path"): LEGACY_GITLINK, ("submodule", LEGACY_GITLINK, "url"): LEGACY_URL}
    if found == want:
        return []
    shown = ", ".join(f"{'.'.join(k)}={v}" for k, v in sorted(found.items())) or "nothing"
    return [f"must declare exactly submodule {LEGACY_GITLINK} (path {LEGACY_GITLINK}, url {LEGACY_URL}); found {shown}"]


def has_legacy_gitlink(entries) -> bool:
    return any(mode == "160000" and p == LEGACY_GITLINK for mode, _, p in entries)


def layout_problems(root: Path, where: str, entries, legacy=False) -> list[tuple[str, str, str]]:
    """(label, rule id, message) for each way `where` (a commit, or ":" for the index, with its tree_entries)
    breaks the layout. legacy: `where` is a commit inherited from the submodule layout, whose SkyRL gitlink
    must sit at the pin with a matching .gitmodules; anywhere else gitlinks and .gitmodules are not allowed."""
    label = "index" if where == ":" else f"commit:{where[:10]}"
    out = []
    for mode, sha, p in entries:
        if mode != "160000":
            continue
        if not legacy:
            msg = f"submodule at {sha}: SkyRL is an external checkout installed via pyproject [tool.uv.sources]"
        elif p != LEGACY_GITLINK:
            msg = f"submodule at {sha}: the only gitlink an inherited commit may carry is {LEGACY_GITLINK}"
        elif sha != UPSTREAM_BASE:
            msg = f"the inherited {LEGACY_GITLINK} gitlink must point at {UPSTREAM_BASE}; found {sha}"
        else:
            continue
        out.append((f"{label}:{p}", "U.gitlink", msg))
    if legacy:
        cp = git_run("config", "--blob", f"{where}:.gitmodules", "--list", cwd=root)
        listing = cp.stdout.decode(errors="replace") if cp.returncode == 0 else None
        out += [(f"{label}:.gitmodules", "U.gitmodules", m) for m in gitmodules_problems(listing)]
    elif any(p == ".gitmodules" for _, _, p in entries):
        out.append((f"{label}:.gitmodules", "U.gitmodules", "must not exist: the repository has no submodules"))
    vendored = {}
    for mode, _, p in entries:
        d = upstream_dir(p) if mode != "160000" else None
        if d:
            vendored.setdefault(d, []).append(p)
    for d, paths in vendored.items():
        out.append(
            (
                f"{label}:{d}/",
                "U.upstream_path",
                f"{len(paths)} tracked path(s), e.g. {paths[0]}: SkyRL is an external import-only checkout "
                "(../SkyRL-ecdaa946); never vendor it, put changes in opsd/",
            )
        )
    return out


def check_layout(rep, root: Path, where: str):
    """HEAD (or another commit) or the index (":") must have no gitlink, no .gitmodules and no upstream files."""
    for label, rid, msg in layout_problems(root, where, tree_entries(root, where)):
        rep.emit("FAIL", label, 0, rid, msg)


def check_worktree_layout(rep, root: Path):
    """On disk: no .gitmodules (FAIL), and no upstream checkout inside the repository (WARN: untracked, but the
    contract puts it next to the repository, at ../SkyRL-ecdaa946)."""
    if (root / ".gitmodules").exists():
        rep.emit("FAIL", "worktree:.gitmodules", 0, "U.gitmodules", "must not exist: the repository has no submodules")
    for d in sorted(root.iterdir()):
        if d.is_dir() and UPSTREAM_DIR_RX.fullmatch(d.name) and not git_paths("ls-files", "--", d.name, cwd=root):
            rep.emit("WARN", f"worktree:{d.name}/", 0, "U.upstream_checkout", "keep SkyRL outside: ../SkyRL-ecdaa946")


def check_history_layout(rep, root: Path, rng: str) -> bool:
    """The layout of every commit of `rng`. A commit with the SkyRL gitlink is inherited from the submodule
    layout: the gitlink must be at the pin, and every parent must carry it too (once dropped, it stays out).
    Every other commit must be free of gitlinks, .gitmodules and upstream files. False: bad range."""
    cp = git_run("rev-list", "--parents", rng, cwd=root)
    if cp.returncode:
        rep.emit("FAIL", f"range:{rng}", 0, "G.range", cp.stderr.decode(errors="replace").strip()[:200])
        return False
    legacy = {}

    def is_legacy(sha: str) -> bool:  # parents outside the range are looked up on demand
        if sha not in legacy:
            legacy[sha] = git("ls-tree", sha, "--", LEGACY_GITLINK, cwd=root).startswith("160000 ")
        return legacy[sha]

    for line in cp.stdout.decode().splitlines():
        sha, *parents = line.split()
        entries = tree_entries(root, sha)
        legacy[sha] = has_legacy_gitlink(entries)
        for label, rid, msg in layout_problems(root, sha, entries, legacy[sha]):
            rep.emit("FAIL", label, 0, rid, msg)
        dropped = [p for p in parents if not is_legacy(p)] if legacy[sha] else []
        if dropped:
            rep.emit(
                "FAIL",
                f"commit:{sha[:10]}:{LEGACY_GITLINK}",
                0,
                "U.gitlink_reintroduced",
                f"parent {dropped[0][:10]} has no {LEGACY_GITLINK} gitlink: only the inherited submodule-layout "
                "commits may carry it",
            )
    return True


def cat_objects(root: Path, shas: list[str]) -> dict[str, tuple[str, bytes]]:
    """{sha: (type, content)} for many objects, read with one `git cat-file --batch` call."""
    if not shas:
        return {}
    data = git_run("cat-file", "--batch", cwd=root, data=("\n".join(shas) + "\n").encode()).stdout
    out, pos = {}, 0
    while pos < len(data):
        eol = data.index(b"\n", pos)
        head = data[pos:eol].decode().split()
        pos = eol + 1
        if len(head) != 3:  # "<sha> missing"
            continue
        size = int(head[2])
        out[head[0]] = (head[1], data[pos : pos + size])
        pos += size + 1
    return out


def scan_history(rep, allow, root: Path, rng: str, rules, harvested, known):
    """--all-commits: the layout of EVERY commit in the range, then every blob the range introduces, scanned in
    full. A single rev (e.g. HEAD) means its whole history, which is the whole branch for an orphan."""
    if not check_history_layout(rep, root, rng):
        return
    objs = []
    for line in git("rev-list", "--objects", rng, cwd=root).splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1]:
            objs.append((parts[0], parts[1]))
    content = cat_objects(root, list(dict.fromkeys(sha for sha, _ in objs)))
    secret_tier = [r for r in rules if r.scope == "all"]
    seen = set()
    for sha, path in objs:
        if sha in seen or content.get(sha, ("", b""))[0] != "blob":
            continue
        seen.add(sha)
        label, blob = f"{path}@{sha[:10]}", content[sha][1]
        if is_upstream(path):  # before the suffix filter, so binary upstream files are caught too
            rep.emit("FAIL", label, 0, "U.upstream_blob", f"a commit tracks a file under {upstream_dir(path)}/")
        path_rules(rep, allow, path, len(blob), label)
        if Path(path).suffix.lower() not in TEXT_EXT:
            continue
        text = blob.decode(errors="ignore")
        if is_upstream(path):
            scan_text(rep, allow, label, text, secret_tier, set(), known, set())
        else:
            scan_text(rep, allow, label, text, rules, harvested, known)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default=".", help="repository (or plain directory with --no-git)")
    ap.add_argument("--no-git", action="store_true")
    ap.add_argument("--diff-base", default=None, help=argparse.SUPPRESS)  # removed: rejected with a hint below
    ap.add_argument("--staged", action="store_true")
    ap.add_argument("--commits", default=None, help="A..B or REV: scan commit messages and identities")
    ap.add_argument(
        "--all-commits",
        default=None,
        help="A..B or REV: check the layout of every commit (no SkyRL inside; an inherited gitlink only at the "
        "pin) and scan every blob introduced",
    )
    ap.add_argument("--history-only", action="store_true", help="with --all-commits/--commits: skip the tree scan")
    ap.add_argument("--extra-denylist", default=None)
    ap.add_argument("--harvested", default=None)
    ap.add_argument("--known-secrets-file", default=None)
    ap.add_argument("--allowed-identities", default=None, help="file: one 'Name <email>' per line")
    ap.add_argument("--only", nargs="+", default=None, help="scan exactly these paths (relative to --tree), fully")
    a = ap.parse_args()
    if a.diff_base:
        ap.error(
            "--diff-base was removed: SkyRL is an external checkout and the branch has no upstream base, so "
            "every tracked file is scanned in full; drop the flag"
        )
    if a.history_only and not (a.all_commits or a.commits):
        ap.error("--history-only needs --all-commits and/or --commits")

    root = Path(a.tree).resolve()
    allow = load_allowlist()
    rules = RULES + load_extra(a.extra_denylist)
    EXTRA_GLOBS.extend(load_extra_globs(a.extra_denylist))
    harvested = load_harvested(a.harvested)
    known = load_known_secret_values(a.known_secrets_file)
    rep = Report()

    if a.only:
        structural(rep, allow, a.only, root)
        for p in a.only:
            if Path(p).suffix.lower() in TEXT_EXT and (root / p).is_file():
                scan_text(rep, allow, p, (root / p).read_text(errors="ignore"), rules, harvested, known)
    elif a.no_git:
        files = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()]
        structural(rep, allow, [], root)
        for p in files:
            if Path(p).suffix.lower() in TEXT_EXT:
                scan_text(rep, allow, p, (root / p).read_text(errors="ignore"), rules, harvested, known)
    else:
        top = git_run("rev-parse", "--show-toplevel", cwd=root)
        if top.returncode:
            ap.error(f"{root} is not inside a git repository (use --no-git for a plain directory)")
        root = Path(top.stdout.decode().strip())
        links = {p for mode, _, p in tree_entries(root, ":") if mode == "160000"}  # reported by check_layout
        if a.staged:
            check_layout(rep, root, ":")
            files = git_paths("diff", "--cached", "--name-only", "--no-renames", "--diff-filter=ACMR", cwd=root)
            scan_files(rep, allow, root, [f for f in files if f not in links], rules, harvested, known)
        elif not a.history_only:
            if git_run("rev-parse", "-q", "--verify", "HEAD^{commit}", cwd=root).returncode == 0:
                check_layout(rep, root, "HEAD")
            check_layout(rep, root, ":")
            check_worktree_layout(rep, root)
            files = git_paths("ls-files", cwd=root)
            scan_files(rep, allow, root, [f for f in files if f not in links], rules, harvested, known)
        if a.all_commits:
            scan_history(rep, allow, root, a.all_commits, rules, harvested, known)
        if a.commits:
            ids = {line.strip() for line in open(a.allowed_identities)} if a.allowed_identities else None
            log = git("log", "--format=%H%x00%an <%ae>%x00%cn <%ce>%x00%B%x1e", a.commits, cwd=root)
            for rec in log.split("\x1e"):
                if not rec.strip():
                    continue
                sha, author, committer, body = rec.strip().split("\x00", 3)
                if ids is not None:
                    for who in (author, committer):
                        if who not in ids:
                            rep.emit("FAIL", f"commit:{sha[:10]}", 0, "C.identity", who)
                body_rules = [r for r in rules if not r.rid.startswith("S6")]
                scan_text(rep, allow, f"commit:{sha[:10]}", body, body_rules, harvested, known)
    print(f"release-guard: {rep.fail} FAIL, {rep.warn} WARN")
    return 1 if rep.fail else 0


if __name__ == "__main__":
    sys.exit(main())

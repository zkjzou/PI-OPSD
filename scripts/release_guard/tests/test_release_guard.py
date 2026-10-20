"""Release guard: rules and verdicts against reference outputs (golden) and the layout rules (SkyRL stays outside).

The layout tests build throwaway git repositories under tmp_path with plumbing commands and run the guard as a
subprocess against them; no real repository is touched. Strings that would trip the guard or gitleaks are
assembled at runtime.
"""

import base64
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1] / "check_release.py"
ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "opsd" / "tests" / "fixtures" / "fork_parity" / "guard" / "guard_verdicts.json"
PIN = "ecdaa9460e281774638a9170d6724456f46c8356"
URL = "https://github.com/NovaSky-AI/SkyRL.git"
OTHER_SHA = "1" * 40
GITMODULES = f'[submodule "SkyRL"]\n\tpath = SkyRL\n\turl = {URL}\n'
HOST_PATH = "/mnt/" + "efs/example/data"
FAKE_TOKEN = "hf_" + "Ab3x" * 9
PRIVATE_IP = "10." + "1.2.3"
OLD_GUARD = "opsd/tools/release_guard/check_release.py"  # the guard's location in the oldest commits


def _load_guard():
    spec = importlib.util.spec_from_file_location("opsd_release_guard_under_test", GUARD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


guard = _load_guard()


def _b64(s: str) -> str:
    return base64.b64decode(s).decode()


class Collect:
    def __init__(self):
        self.fail = self.warn = 0
        self.hits = []

    def emit(self, level, path, line, rid, snippet, secret=False):
        self.hits.append([level, rid])


# ---------------------------------------------------------------------------------------------------
# Rules and verdicts vs the reference outputs
# ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text())


def _to_new_layout(p: str) -> str:
    """Reference-corpus path -> path in this layout (opsd/ package, scripts/, assets/ at the root)."""
    for old, new in (
        ("tools/release_guard/", "scripts/release_guard/"),
        ("docs/opsd/", "assets/docs/"),
        ("tests/opsd/", "opsd/tests/"),
        ("recipes/", "opsd/recipes/"),
        ("requirements/", "scripts/setup/requirements/"),
        ("results/", "assets/results/"),
    ):
        if p.startswith(old):
            return new + p[len(old) :]
    return p  # opsd/ (the package), scripts/ and root files (.gitignore, README.md, ...) keep their path


def test_rule_table_unchanged(golden):
    got = [
        {
            "rid": r.rid,
            "pattern_sha256": hashlib.sha256(r.rx.pattern.encode()).hexdigest(),
            "flags": int(r.rx.flags),
            "scope": r.scope,
            "level": r.level,
            "secret": r.secret,
        }
        for r in guard.RULES
    ]
    assert got == golden["rules"]


def test_tables(golden):
    t = golden["tables"]
    assert guard.FORBIDDEN_GLOBS == [_b64(x) for x in t["forbidden_globs_b64"]]
    assert sorted(guard.FORBIDDEN_EXT) == sorted(set(t["forbidden_ext"]) | {".pyc", ".pyo"})  # .pyc/.pyo are new
    # .lock and .svg are new: uv.lock and the assets/ diagrams are ours in the recipe layout
    assert sorted(guard.TEXT_EXT) == sorted(set(t["text_ext"]) | {".lock", ".svg"})
    assert (guard.SIZE_WARN, guard.SIZE_FAIL) == (t["size_warn"], t["size_fail"])


def test_scan_text_verdicts(golden):
    allow = guard.load_allowlist()
    known = [(k, _b64(v)) for k, v in golden["known_b64"]]
    harvested = {_b64(x) for x in golden["harvested_b64"]}
    for case in golden["scan_cases"]:
        line = _b64(case["line_b64"])
        for mode, added in (("full", None), ("secret_tier", set())):
            rep = Collect()
            guard.scan_text(rep, allow, "sample.txt", line, guard.RULES, harvested, known, added)
            assert rep.hits == case[mode], (mode, line)


def test_extra_denylist(golden, tmp_path):
    dl = tmp_path / "denylist.txt"
    dl.write_text(_b64(golden["extra_denylist"]["text_b64"]))
    got = [
        {
            "rid": r.rid,
            "pattern_b64": base64.b64encode(r.rx.pattern.encode()).decode(),
            "scope": r.scope,
            "level": r.level,
        }
        for r in guard.load_extra(str(dl))
    ]
    assert got == golden["extra_denylist"]["rules"]


def test_path_rules_at_root_and_under_opsd(golden, tmp_path):
    allow = guard.load_allowlist()
    for case in golden["path_cases"]:
        p, want = _b64(case["path_b64"]), case["hits"]
        if p.endswith((".pyc", ".pyo")):
            want = want + [["FAIL", "F.forbidden_ext"]]  # intended: byte-code is never committed
        for cand in (p, "opsd/" + p, _to_new_layout(p)):
            rep = Collect()
            guard.structural(rep, allow, [cand], tmp_path)
            assert rep.hits == want, cand


def test_allowlist_follows_the_move(golden):
    allow = guard.load_allowlist()
    for case in golden["allow_cases"]:
        p = _b64(case["path_b64"])
        assert guard.allowed(allow, _to_new_layout(p), case["rid"]) is case["allowed"], p
    # earlier locations of the guard are no longer exempt in the tree ...
    assert not guard.allowed(allow, "tools/release_guard/check_release.py", "S5.orchestration")
    assert not guard.allowed(allow, OLD_GUARD, "S5.orchestration")
    # ... but its blobs in the inherited submodule-layout commits are (history mode labels them path@sha)
    assert guard.allowed(allow, OLD_GUARD + "@0123456789", "S5.orchestration")
    assert not guard.allowed(allow, "opsd/tools/other.py@0123456789", "S5.orchestration")


# ---------------------------------------------------------------------------------------------------
# The layout (SkyRL stays outside the repository), on throwaway repositories
# ---------------------------------------------------------------------------------------------------
class Repo:
    def __init__(self, root: Path, env: dict):
        self.root, self.env = root, env
        root.mkdir(exist_ok=True)
        self.git("init", "-q", "-b", "main")

    def git(self, *args, check=True) -> str:
        cp = subprocess.run(["git", *args], cwd=self.root, env=self.env, capture_output=True, text=True)
        if check and cp.returncode:
            raise AssertionError(f"git {args}: {cp.stderr}")
        return cp.stdout

    def write(self, rel: str, text: str, stage=True):
        fp = self.root / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(text)
        if stage:
            self.git("update-index", "--add", "--", rel)

    def gitlink(self, sha=PIN, path="SkyRL"):
        (self.root / path).mkdir(parents=True, exist_ok=True)  # an uninitialised submodule is an empty directory
        self.git("update-index", "--add", "--cacheinfo", f"160000,{sha},{path}")

    def drop(self, rel: str):
        self.git("update-index", "--force-remove", "--", rel)
        fp = self.root / rel
        if fp.is_file():
            fp.unlink()
        elif fp.is_dir() and not any(fp.iterdir()):
            fp.rmdir()

    def legacy(self, sha=PIN, gitmodules=GITMODULES):
        """Stage the submodule layout the branch inherits: the SkyRL gitlink and its .gitmodules."""
        self.write(".gitmodules", gitmodules)
        self.gitlink(sha)

    def recipe(self):
        """Stage the move to the recipe layout: SkyRL becomes an external checkout."""
        self.drop("SkyRL")
        self.drop(".gitmodules")

    def commit(self, msg="change") -> str:
        tree = self.git("write-tree").strip()
        parent = self.git("rev-parse", "-q", "--verify", "HEAD", check=False).strip()
        sha = self.git("commit-tree", tree, *(["-p", parent] if parent else []), "-m", msg).strip()
        self.git("update-ref", "HEAD", sha)
        return sha

    def guard(self, *args):
        cp = subprocess.run(
            [sys.executable, str(GUARD), "--tree", str(self.root), *args], env=self.env, capture_output=True, text=True
        )
        return cp.returncode, cp.stdout + cp.stderr

    def history(self, rng="HEAD"):
        return self.guard("--history-only", "--all-commits", rng)


@pytest.fixture
def git_env(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        HOME=str(tmp_path),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=str(tmp_path / "gitconfig"),
        GIT_AUTHOR_NAME="Test",
        GIT_AUTHOR_EMAIL="test@example.com",
        GIT_COMMITTER_NAME="Test",
        GIT_COMMITTER_EMAIL="test@example.com",
        GIT_AUTHOR_DATE="2026-01-01T00:00:00Z",
        GIT_COMMITTER_DATE="2026-01-01T00:00:00Z",
    )
    return env


@pytest.fixture
def repo(tmp_path, git_env):
    """The recipe layout, no history before it."""
    r = Repo(tmp_path / "repo", git_env)
    r.write("README.md", "# example\n")
    r.write("pyproject.toml", '[tool.uv.sources]\nskyrl = { path = "../SkyRL-ecdaa946", editable = true }\n')
    r.write("opsd/__init__.py", "")
    r.write("opsd/skyrl_ext/__init__.py", "")  # ours, despite the name
    r.commit("init")
    return r


@pytest.fixture
def lineage(tmp_path, git_env):
    """The real branch's shape: submodule-layout commits (SkyRL gitlink at the pin), then the recipe layout."""
    r = Repo(tmp_path / "repo", git_env)
    r.write("README.md", "# example\n")
    r.write("opsd/opsd/__init__.py", "")
    r.legacy()
    r.commit("submodule layout")
    r.write("opsd/opsd/x.py", "x = 1\n")
    r.commit("more")
    r.recipe()
    r.drop("opsd/opsd/__init__.py")
    r.drop("opsd/opsd/x.py")
    r.write("opsd/__init__.py", "")
    r.commit("recipe layout")
    return r


def _rids(out: str) -> list[str]:
    return [ln.split()[2] for ln in out.splitlines() if ln.startswith(("FAIL ", "WARN "))]


def test_clean_repo_passes(repo):
    rc, out = repo.guard()
    assert rc == 0 and " 0 FAIL, 0 WARN" in out, out
    rc, out = repo.guard("--staged")
    assert rc == 0, out
    rc, out = repo.guard("--history-only", "--all-commits", "HEAD", "--commits", "HEAD")
    assert rc == 0 and " 0 FAIL" in out, out


@pytest.mark.parametrize("path", ["SkyRL", "third_party/SkyRL", "vendor/tau2"])
def test_any_submodule_fails(repo, path):
    repo.gitlink(PIN, path)  # even the pinned upstream commit: SkyRL is an external checkout now
    rc, out = repo.guard("--staged")
    assert rc == 1 and _rids(out) == ["U.gitlink"], out
    repo.commit("add a submodule")
    rc, out = repo.guard()
    assert rc == 1 and _rids(out) == ["U.gitlink", "U.gitlink"], out  # HEAD and index
    rc, out = repo.history()
    # in history a SkyRL gitlink reads as the inherited one coming back (without its .gitmodules)
    want = ["U.gitmodules", "U.gitlink_reintroduced"] if path == "SkyRL" else ["U.gitlink"]
    assert rc == 1 and _rids(out) == want, out


@pytest.mark.parametrize("text", [GITMODULES, "", "not [valid"])
def test_gitmodules_fails(repo, text):
    repo.write(".gitmodules", text)
    rc, out = repo.guard("--staged")
    assert rc == 1 and _rids(out) == ["U.gitmodules"], out
    rc, out = repo.guard()
    assert rc == 1 and _rids(out) == ["U.gitmodules", "U.gitmodules"], out  # index + worktree (HEAD is fine)
    repo.commit("add .gitmodules")
    rc, out = repo.history()
    assert rc == 1 and _rids(out) == ["U.gitmodules"], out


def test_untracked_gitmodules_fails_the_tree_mode(repo):
    repo.write(".gitmodules", GITMODULES, stage=False)
    rc, out = repo.guard()
    assert rc == 1 and _rids(out) == ["U.gitmodules"] and "worktree:.gitmodules" in out, out
    rc, out = repo.guard("--staged")
    assert rc == 0, out


@pytest.mark.parametrize(
    "rel",
    [
        "SkyRL/skyrl/train/trainer.py",
        "SkyRL-ecdaa946/README.md",
        "skyrl-ecdaa946/pyproject.toml",
        "skyrl/train/trainer.py",
        "skyrl-gym/skyrl_gym/envs/base.py",
        "third_party/SkyRL/LICENSE",
        "opsd/vendor/skyrl_gym/core.py",
        "opsd/_vendored/skyrl/__init__.py",
        "opsd/skyrl_train/utils.py",
    ],
)
def test_vendored_upstream_files_fail(repo, rel):
    repo.write(rel, "print('patched')\n")
    rc, out = repo.guard("--staged")
    assert rc == 1 and _rids(out) == ["U.upstream_path"], out
    repo.commit("vendor")
    rc, out = repo.guard()
    assert rc == 1 and _rids(out) == ["U.upstream_path", "U.upstream_path"], out  # HEAD and index
    rc, out = repo.history()
    assert rc == 1 and sorted(_rids(out)) == ["U.upstream_blob", "U.upstream_path"], out


def test_vendored_files_are_grouped_and_get_the_secret_tier(repo):
    repo.write("SkyRL/a.py", f"data under {HOST_PATH}\n")  # host rules are for our files only ...
    repo.write("SkyRL/b.py", f"token {FAKE_TOKEN}\n")  # ... secrets are reported everywhere
    rc, out = repo.guard("--staged")
    assert rc == 1 and _rids(out) == ["U.upstream_path", "S1.hf_token"] and "2 tracked path(s)" in out, out


@pytest.mark.parametrize(
    "rel",
    ["opsd/skyrl_ext/config.py", "opsd/pi/adapters/skyrl_eval_dump.py", "ADAPTED_FROM_SKYRL.md", "x/skyrl"],
)
def test_our_skyrl_named_paths_pass(repo, rel):
    repo.write(rel, "x = 1\n")
    rc, out = repo.guard("--staged")
    assert rc == 0, out


def test_checkout_inside_the_repository_warns(repo):
    (repo.root / "SkyRL-ecdaa946").mkdir()
    (repo.root / "SkyRL-ecdaa946" / "README.md").write_text("upstream\n")
    rc, out = repo.guard()
    assert rc == 0 and _rids(out) == ["U.upstream_checkout"], out


def test_inherited_submodule_lineage_passes(lineage):
    rc, out = lineage.guard()
    assert rc == 0 and " 0 FAIL" in out, out
    rc, out = lineage.guard("--history-only", "--all-commits", "HEAD", "--commits", "HEAD")
    assert rc == 0 and " 0 FAIL" in out, out


def test_inherited_gitlink_must_be_at_the_pin(tmp_path, git_env):
    r = Repo(tmp_path / "repo", git_env)
    r.write("README.md", "# example\n")
    r.legacy()
    c1 = r.commit("submodule layout")
    r.gitlink(OTHER_SHA)
    c2 = r.commit("transient bump")
    r.gitlink(PIN)
    r.commit("restore")
    r.recipe()
    r.commit("recipe layout")
    rc, out = r.guard()
    assert rc == 0, out  # the tip is fine ...
    rc, out = r.history()
    assert rc == 1 and _rids(out) == ["U.gitlink"] and f"commit:{c2[:10]}" in out, out  # ... its history is not
    rc, out = r.history(f"{c1}..HEAD")
    assert rc == 1 and _rids(out) == ["U.gitlink"], out
    rc, out = r.history(f"{c2}..HEAD")
    assert rc == 0, out


@pytest.mark.parametrize(
    "gitmodules",
    [
        GITMODULES.replace(URL, "https://github.com/example/SkyRL.git"),
        GITMODULES + '[submodule "extra"]\n\tpath = extra\n\turl = https://example.com/x.git\n',
        GITMODULES + "\tshallow = true\n",
        "",
        None,
    ],
)
def test_inherited_gitmodules_must_declare_exactly_the_pin(tmp_path, git_env, gitmodules):
    r = Repo(tmp_path / "repo", git_env)
    r.write("README.md", "# example\n")
    r.legacy()
    if gitmodules is None:
        r.drop(".gitmodules")
    else:
        r.write(".gitmodules", gitmodules)
    r.commit("submodule layout")
    r.recipe()
    r.commit("recipe layout")
    rc, out = r.history()
    assert rc == 1 and _rids(out) == ["U.gitmodules"], out


def test_a_gitlink_may_not_come_back(lineage):
    lineage.legacy()
    back = lineage.commit("bring the submodule back")
    rc, out = lineage.guard()
    assert rc == 1 and _rids(out) == ["U.gitlink", "U.gitmodules"] * 2 + ["U.gitmodules"], out  # HEAD, index, disk
    lineage.recipe()
    lineage.commit("and drop it again")
    rc, out = lineage.guard()
    assert rc == 0, out  # the tip is fine ...
    rc, out = lineage.history()
    assert rc == 1 and _rids(out) == ["U.gitlink_reintroduced"] and f"commit:{back[:10]}" in out, out  # ... not so
    rc, out = lineage.history(f"{back}..HEAD")
    assert rc == 0, out


def test_dropping_only_the_gitlink_keeps_gitmodules_out(lineage):
    lineage.legacy()
    lineage.commit("back")
    lineage.drop("SkyRL")
    lineage.commit("half-way: .gitmodules left behind")
    rc, out = lineage.history("HEAD~1..HEAD")
    assert rc == 1 and _rids(out) == ["U.gitmodules"], out


def test_a_bad_range_fails(repo):
    rc, out = repo.history("no-such-rev")
    assert rc == 1 and _rids(out) == ["G.range"], out


def test_history_scans_blobs_that_were_removed_again(repo):
    repo.write("opsd/notes.md", f"data under {HOST_PATH}\n")
    repo.commit("add")
    repo.drop("opsd/notes.md")
    repo.commit("remove")
    rc, out = repo.guard()
    assert rc == 0, out
    rc, out = repo.history()
    assert rc == 1 and _rids(out) == ["S3.host_paths"] and "opsd/notes.md@" in out, out


def test_history_exempts_only_the_old_guard_blobs(lineage):
    patterns = f"rx = r'{HOST_PATH}'\n"
    lineage.write(OLD_GUARD, patterns)
    lineage.commit("the guard at its old location")
    rc, out = lineage.guard()
    assert rc == 1 and _rids(out) == ["S3.host_paths"], out  # in the tree, the old location is not exempt
    lineage.drop(OLD_GUARD)
    lineage.write("scripts/release_guard/check_release.py", patterns + "# moved\n")
    lineage.commit("moved")
    rc, out = lineage.guard()
    assert rc == 0, out
    rc, out = lineage.history()
    assert rc == 0, out  # its blob at the old location, in history: exempt


def test_root_files_get_the_full_scan(repo):
    # before the split README.md was a modified upstream file and only its added lines were scanned
    repo.write("README.md", f"# example\nserved from {PRIVATE_IP}\n")
    rc, out = repo.guard()
    assert rc == 1 and _rids(out) == ["S4.private_ip"], out


def test_uv_lock_and_svg_are_scanned(repo):
    repo.write("uv.lock", f'version = 1\n[[package]]\nname = "x"\nversion = "{PRIVATE_IP}"\n')
    rc, out = repo.guard("--staged")
    assert rc == 0, out  # package versions that read as private IPs are allowlisted ...
    repo.write("uv.lock", f'[[package]]\nname = "skyrl"\nsource = {{ editable = "{HOST_PATH}" }}\n')
    repo.write("assets/diagram.svg", f"<svg><!-- rendered from {HOST_PATH} --></svg>\n")
    rc, out = repo.guard("--staged")
    assert rc == 1 and _rids(out) == ["S3.host_paths", "S3.host_paths"], out  # ... an absolute source path is not


def test_forbidden_paths_at_the_root_and_under_opsd(repo):
    for rel in ("x.pids", "opsd/x.pids", "opsd/tests/x.pids", "opsd/a.env.vars"):
        repo.write(rel, "x = 1\n")
    repo.write("opsd/m.pyc", "x")
    rc, out = repo.guard("--staged")
    assert rc == 1 and sorted(_rids(out)) == ["F.forbidden_ext"] + ["F.forbidden_path"] * 4, out


def test_extra_denylist_path_globs(repo, tmp_path):
    dl = tmp_path / "denylist.txt"
    dl.write_text("example[-_]internal\\d+\nPATH: private_dir/*\nPATH: *_private.sh\n")
    assert guard.load_extra_globs(str(dl)) == ["private_dir/*", "*_private.sh"]
    assert [r.rx.pattern for r in guard.load_extra(str(dl))] == ["example[-_]internal\\d+"]
    for rel in ("private_dir/x.py", "opsd/private_dir/x.py", "scripts/run_private.sh"):
        repo.write(rel, "x = 1\n")
    rc, out = repo.guard("--staged")
    assert rc == 0, out  # without the extra file these paths are fine ...
    rc, out = repo.guard("--staged", "--extra-denylist", str(dl))
    assert rc == 1 and _rids(out) == ["F.forbidden_path"] * 3, out  # ... with it they are forbidden


def test_diff_base_is_rejected(repo):
    rc, out = repo.guard("--diff-base", PIN)
    assert rc == 2 and "--diff-base was removed" in out, out


def test_upstream_base_is_the_install_pin():
    # one pin: the guard (stdlib only, so it cannot import opsd) must agree with opsd/_repo.py, which the setup
    # script and the provenance record read
    m = re.search(r'^SKYRL_COMMIT = "([0-9a-f]{40})"$', (ROOT / "opsd" / "_repo.py").read_text(), re.M)
    assert m and guard.UPSTREAM_BASE == m.group(1) == PIN, (guard.UPSTREAM_BASE, m and m.group(1))

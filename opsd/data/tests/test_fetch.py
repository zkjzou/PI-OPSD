"""fetch.py on a tiny fake bundle: selection, sha256 verification, local copy and a stubbed HF download."""

import hashlib
import json
import sys
import types

import pytest

from opsd.data import fetch as F

FILES = {
    "alfworld/splits/task_ids_train1600.json": b'["a", "b"]',
    "alfworld/hints/glm-5.2/hints_actions.json": b'{"a": "go"}',
    "alfworld/pi_metrics/scores/rows.parquet": b"PAR1fake",
    "tau2/splits/retail_pi424.jsonl": b'{"id": 1}\n',
    "appworld/appworld_opsd_pi.bundle": b"\x00encrypted",
}


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    recs = []
    for rel, data in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        recs.append(
            {
                "path": rel,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "benchmark": rel.split("/")[0],
                "status": "final",
            }
        )
    (root / F.MANIFEST).write_text(json.dumps({"files": recs}))
    (root / "README.md").write_text("card")
    return root


def test_parse_manifest_forms_and_tiers():
    recs = [
        {"path": "alfworld/hints/x.json"},
        {"path": "tau2/pi_metrics/a.parquet"},
        {"path": "tau2/corpora/c.jsonl"},
        {"path": "appworld/b.bundle", "tier": "b"},
    ]
    for text in (json.dumps(recs), json.dumps({"files": recs}), "\n".join(json.dumps(r) for r in recs)):
        es = F.parse_manifest(text)
        assert [e.tier for e in es] == ["A", "B", "C", "B"] and es[0].bench == "alfworld"


def test_appworld_bundles_select_by_their_manifest_tier():
    """The eight AppWorld bundles of the dataset, with the tiers the manifest gives them (appworld_opsd_pi has
    none and falls back to A by its path): --bench / --tier / --include pick them as the docs say."""
    tiers = {
        "appworld_opsd_pi": None,
        "appworld_opsd_metrics": "B",
        "appworld_opsd_corpora_train90": "C",
        "appworld_opsd_corpora_dev": "C",
        "appworld_opsd_corpora_test_normal": "C",
        "appworld_opsd_dev_hints": "A",
        "appworld_opsd_xdnull": "A",
        "appworld_opsd_test_normal_hints": "A",
    }
    recs = [
        {"path": f"appworld/{n}.bundle", "benchmark": "appworld", **({"tier": t} if t else {})}
        for n, t in tiers.items()
    ]
    recs += [{"path": "alfworld/hints/nulls/glt_n3_from_glm.json", "benchmark": "alfworld"}]
    es = F.parse_manifest(json.dumps({"files": recs}))

    def names(sel):
        return sorted(e.path.split("/")[-1][: -len(".bundle")] for e in sel if e.path.endswith(".bundle"))

    assert names(F.select(es, ["appworld"])) == sorted(tiers)
    assert names(F.select(es, ["appworld"], ["A"])) == [
        "appworld_opsd_dev_hints",
        "appworld_opsd_pi",
        "appworld_opsd_test_normal_hints",
        "appworld_opsd_xdnull",
    ]
    assert names(F.select(es, ["appworld"], ["B"])) == ["appworld_opsd_metrics"]
    assert names(F.select(es, ["appworld"], ["C"])) == [
        "appworld_opsd_corpora_dev",
        "appworld_opsd_corpora_test_normal",
        "appworld_opsd_corpora_train90",
    ]
    assert names(F.select(es, ["alfworld"])) == []  # the ALFWorld null_xdomain hint ships in an AppWorld bundle
    xd = ["appworld/appworld_opsd_xdnull.bundle"]
    assert (
        names(F.select(es, ["appworld"], None, xd)) == names(F.select(es, None, None, xd)) == ["appworld_opsd_xdnull"]
    )
    # an exact --include picks that bundle alone, not a sibling whose name it prefixes
    sib = [
        {"path": f"appworld/{n}.bundle", "benchmark": "appworld"}
        for n in ("appworld_opsd_xdnull", "appworld_opsd_xdnull_x")
    ]
    es2 = F.parse_manifest(json.dumps({"files": sib}))
    assert names(F.select(es2, None, None, xd)) == ["appworld_opsd_xdnull"]
    assert names(F.select(es2, None, None, ["appworld/appworld_opsd_xdnull_x.bundle"])) == ["appworld_opsd_xdnull_x"]


@pytest.mark.parametrize(
    "bad",
    [
        [{"path": "../x"}],
        [{"path": "/abs"}],
        [{"path": "a"}, {"path": "a"}],
        [{"nopath": 1}],
        [{"path": "a", "sha256": "xyz"}],
    ],
)
def test_parse_manifest_rejects(bad):
    with pytest.raises(F.FetchError):
        F.parse_manifest(json.dumps(bad))


def test_from_local_copies_and_verifies(bundle, tmp_path):
    dest = tmp_path / "data"
    assert F.main(["--from-local", str(bundle), "--dest", str(dest)]) == 0
    for rel, data in FILES.items():
        assert (dest / rel).read_bytes() == data
    assert (dest / F.MANIFEST).is_file() and (dest / "README.md").is_file()
    rec = json.loads((dest / F.FETCH_RECORD).read_text())
    assert rec["ok"] == len(FILES) and rec["mismatch"] == []
    assert F.main(["--verify-only", "--dest", str(dest)]) == 0


def test_bench_tier_and_include_selection(bundle, tmp_path, capsys):
    dest = tmp_path / "d1"
    assert F.main(["--from-local", str(bundle), "--dest", str(dest), "--bench", "alfworld", "--tier", "A"]) == 0
    got = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file() and p.parent != dest)
    assert got == ["alfworld/hints/glm-5.2/hints_actions.json", "alfworld/splits/task_ids_train1600.json"]
    assert F.main(["--from-local", str(bundle), "--dest", str(dest), "--include", "tau2/*", "--list"]) == 0
    assert capsys.readouterr().out.strip().endswith("tau2/splits/retail_pi424.jsonl")
    assert F.main(["--from-local", str(bundle), "--dest", str(dest), "--bench", "nope"]) == 2


def test_sha_mismatch_and_missing_fail(bundle, tmp_path, capsys):
    (bundle / "tau2/splits/retail_pi424.jsonl").write_bytes(b"tampered")
    (bundle / "appworld/appworld_opsd_pi.bundle").unlink()
    dest = tmp_path / "data"
    assert F.main(["--from-local", str(bundle), "--dest", str(dest)]) == 1
    err = capsys.readouterr().err
    assert "MISMATCH: tau2/splits/retail_pi424.jsonl" in err and "MISSING: appworld/appworld_opsd_pi.bundle" in err
    assert not (dest / "tau2/splits/retail_pi424.jsonl").exists()
    assert (dest / "tau2/splits/retail_pi424.jsonl.sha256-mismatch").read_bytes() == b"tampered"


def test_verify_in_place(bundle):
    assert F.main(["--from-local", str(bundle), "--dest", str(bundle)]) == 0
    (bundle / "alfworld/hints/glm-5.2/hints_actions.json").write_bytes(b'{"a": "stop"}')
    assert F.main(["--verify-only", "--dest", str(bundle)]) == 1


def test_repo_mode_uses_pinned_revision_and_allow_patterns(bundle, tmp_path, monkeypatch):
    calls = {}

    def hf_hub_download(repo_id, filename, repo_type=None, revision=None, cache_dir=None):
        calls["manifest"] = (repo_id, filename, repo_type, revision)
        return str(bundle / filename)

    def snapshot_download(repo_id, repo_type=None, revision=None, local_dir=None, allow_patterns=None, max_workers=8):
        calls["snapshot"] = (repo_id, repo_type, revision, sorted(allow_patterns))
        for pat in allow_patterns:
            src = bundle / pat
            if src.is_file():
                dst = tmp_path / "hf" / pat
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())
        return local_dir

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=hf_hub_download, snapshot_download=snapshot_download),
    )
    sha = "a" * 40
    assert (
        F.main(["--repo", "org/opsd-data", "--revision", sha, "--dest", str(tmp_path / "hf"), "--bench", "tau2"]) == 0
    )
    assert calls["manifest"] == ("org/opsd-data", "manifest.json", "dataset", sha)
    assert calls["snapshot"][:3] == ("org/opsd-data", "dataset", sha)
    assert calls["snapshot"][3] == sorted(["manifest.json", "README.md", "tau2/splits/retail_pi424.jsonl"])
    assert F.main(["--repo", "org/opsd-data", "--dest", str(tmp_path / "hf")]) == 2  # no revision


def test_default_dest_follows_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPSD_DATA_ROOT", str(tmp_path / "x"))
    assert F.default_dest() == tmp_path / "x"
    monkeypatch.delenv("OPSD_DATA_ROOT")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "c"))
    assert F.default_dest() == tmp_path / "c" / "opsd" / "data"

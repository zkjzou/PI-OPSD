"""Unit tests for the recipe resolver, on the synthetic ``fixtures/recipes`` tree (bench ``toy``)."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from opsd import _repo
from opsd.recipes import resolve as R

FIX = Path(__file__).resolve().parent / "fixtures" / "recipes"
ENV = {"HOME": "/home/u", "OPSD_DATA_ROOT": "/data"}


def _repo_root() -> Path:
    """The repository root (holds pyproject.toml and the ``opsd`` package)."""
    return R.repo_root(Path(__file__))


def res(arm, overlays=(), overrides=None, environ=None, **kw):
    return R.resolve(arm, overlays, overrides or {}, FIX, environ=ENV if environ is None else environ, **kw)


@pytest.fixture
def tree(tmp_path):
    """A writable copy of the fixture tree for negative tests."""
    dst = tmp_path / "recipes"
    shutil.copytree(FIX, dst)
    return dst


# ------------------------------------------------------------------------------------------ arm ids
def test_parse_arm_id_round_trip():
    for s in [
        "toy/main/4bi/glm_a",
        "alfworld/opsd_grpo/4bi/glm_ao/lambda=0.99/seed=2",
        "appworld/beta/4bi/glm_ao/beta=1e-3/teacher=live",
        "tau2/grpo/4bi/none/seed=1",
    ]:
        assert str(R.parse_arm_id(s)) == s
    a = R.parse_arm_id("appworld/rlsd/4bi/glm_aot/lambda=0.5/decay=0")
    assert a.extras == (("lambda", "0.5"), ("decay", "0")) and a.seed is None
    assert R.parse_arm_id("x/t/s/p/seed=3").seed == "3"


@pytest.mark.parametrize(
    "bad",
    [
        "toy/main/4bi",
        "toy/main/4bi/glm_a/lambda",
        "toy/main/4bi/glm_a/seed=1/lambda=0.5",
        "toy/main/4bi/glm_a/a=1/a=2",
        "toy/ma in/4bi/glm_a",
        "toy/main/4bi/glm_a/x=a/b",
    ],
)
def test_parse_arm_id_rejects(bad):
    with pytest.raises(R.RecipeError):
        R.parse_arm_id(bad)


def test_run_name_is_path_and_hydra_safe():
    assert R.parse_arm_id("a/b/c/d/lambda=0.5/seed=1").run_name == "a__b__c__d__lambda-0.5__seed-1"


# --------------------------------------------------------------------------------------- generation
def test_tables_generate_expected_arm_ids():
    b = R.load_bench("toy", FIX)
    assert list(b.arms) == [
        "toy/main/4bi/glm_a",
        "toy/main/4bi/glm_ao",
        "toy/main/4bi/self_summ",
        "toy/main/4bt/glm_a",
        "toy/main/4bt/self_summ",
        "toy/opsd_grpo/4bi/glm_ao/lambda=0.5/seed=1",
        "toy/opsd_grpo/4bi/glm_ao/lambda=0.5/seed=2",
        "toy/opsd_grpo/4bi/glm_ao/lambda=0.99/seed=1",
        "toy/opsd_grpo/4bi/glm_ao/lambda=0.99/seed=2",
        "toy/grpo/4bi/none/seed=1",
        "toy/grpo/4bi/none/seed=2",
        "toy/teacher/4bi/glm_a/teacher=ema",
        "toy/teacher/4bi/glm_a/teacher=h10",
        "toy/beta/4bi/glm_a/beta=1e-4",
        "toy/beta/4bi/glm_a/beta=0.50",
        "toy/future/4bi/glm_a/lambda=0.7",
        "toy/selection/4bi/glm_a/seed=1",
    ]
    # glm_ao has no 4bt hint -> skipped (reported by validate), not an error
    assert any(s.startswith("toy/main/4bt/glm_ao ") for s in b.skipped)
    assert R.list_benches(FIX) == ["toy"]


def test_yaml_scalars_keep_their_spelling():
    """1e-4 and 0.50 must not become floats: arm ids and env values are compared as text."""
    e = res("toy/beta/4bi/glm_a/beta=0.50").env
    assert e["OPSD_DISTILL_LOSS_COEF"] == "0.50"
    assert res("toy/beta/4bi/glm_a/beta=1e-4").env["OPSD_DISTILL_LOSS_COEF"] == "1e-4"
    assert res("toy/opsd_grpo/4bi/glm_ao/lambda=0.5/seed=1").env["TRAINER_SEED"] == "42"


# ------------------------------------------------------------------------------------ merge order
def test_merge_order_and_provenance():
    r = res("toy/main/4bt/glm_a", overlays=["smoke"], overrides={"MAX_TURNS": "3"})
    chain = r.provenance["MAX_TURNS"]
    assert [n for n, _ in chain] == ["toy/base.env", "toy/students/4bt.env", "overlay:common/overlays/smoke.env", "cli"]
    assert [v for _, v in chain] == ["30", "25", "6", "3"]
    assert r.env["MAX_TURNS"] == "3"
    # bench overlay refines the common overlay
    assert r.env["NUM_TOY_SERVERS"] == "8" and r.env["NUM_GPUS_PER_NODE"] == "4"
    assert r.layers[:3] == ["toy/base.env", "toy/students/4bt.env", "toy/objectives/opsd.env"]


def test_objective_then_extra_then_seed():
    r = res("toy/opsd_grpo/4bi/glm_ao/lambda=0.99/seed=2")
    assert r.env["OPSD_MIX_LAMBDA"] == "0.99"
    assert [n for n, _ in r.provenance["OPSD_MIX_LAMBDA"]] == [
        "toy/objectives/opsd_grpo.env",
        "toy/arms.yaml[2:opsd_grpo].extra.lambda=0.99",
    ]
    assert r.env["TRAINER_SEED"] == "43" and r.env["EPOCHS"] == "2"
    assert r.env["USE_KL_IN_REWARD"] == "false"


def test_pi_layer_hint_cap_sha_and_expansion():
    r = res("toy/opsd_grpo/4bi/glm_ao/lambda=0.5/seed=1")
    assert r.env["OPSD_HINTS_PATH"] == "/data/toy/hints/glm/hints_traj_cap24k.json"
    assert r.env["OPSD_REF_SEQ_CAP"] == "65536"
    assert r.env["OPSD_HINTS_CONTENT_SHA256"] == "0" * 64
    assert r.env["TASK_IDS_PATH"] == "/data/toy/splits/train.json"
    assert r.env["REASONING_PREFIX"] == "THOUGHTS: " and r.env["STOP_TOKENS"] == "[]"
    assert r.env["EVAL_N_SAMPLES"] == "4" and r.env["EMPTY_VALUE"] == ""


def test_cli_hint_override_clears_the_declared_sha():
    arm = "toy/opsd_grpo/4bi/glm_ao/lambda=0.5/seed=1"
    r = res(arm, overrides={"OPSD_HINTS_PATH": "/mine/hints.json"})
    assert r.env["OPSD_HINTS_PATH"] == "/mine/hints.json" and r.env["OPSD_HINTS_CONTENT_SHA256"] == ""
    assert r.provenance["OPSD_HINTS_CONTENT_SHA256"][-1][0].startswith("cleared:")
    assert any("OPSD_HINTS_CONTENT_SHA256" in w and "cleared" in w for w in r.warnings)
    # an explicit sha on the command line is kept (and checked by preflight)
    r2 = res(arm, overrides={"OPSD_HINTS_PATH": "/mine/hints.json", "OPSD_HINTS_CONTENT_SHA256": "a" * 64})
    assert r2.env["OPSD_HINTS_CONTENT_SHA256"] == "a" * 64 and not r2.warnings
    # no override -> the published hash stays; a pi without a declared hash gets no warning
    assert res(arm).env["OPSD_HINTS_CONTENT_SHA256"] == "0" * 64
    r3 = res("toy/main/4bi/glm_a", overrides={"OPSD_HINTS_PATH": "/mine/hints.json"})
    assert "OPSD_HINTS_CONTENT_SHA256" not in r3.env and not r3.warnings


def test_relative_locations_resolve_against_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = res("toy/main/4bi/glm_a", environ={"HOME": "/h", "RUN_ROOT": "./myruns", "OPSD_DATA_ROOT": "data"})
    assert r.env["RUN_DIR"] == str(tmp_path / "myruns" / "toy__main__4bi__glm_a")
    assert r.env["OPSD_HINTS_PATH"] == str(tmp_path / "data" / "toy" / "hints" / "glm" / "hints_actions.json")
    # relative input paths given as KEY=VAL too (a launcher may cd before running the trainer)
    r2 = res("toy/main/4bi/glm_a", overrides={"OPSD_HINTS_PATH": "my/h.json", "TASK_IDS_PATH": "${OPSD_DATA_ROOT}/t"})
    assert r2.env["OPSD_HINTS_PATH"] == str(tmp_path / "my" / "h.json")
    assert r2.env["TASK_IDS_PATH"] == "/data/t"
    assert res("toy/main/4bi/glm_a", overrides={"MAX_TURNS": "3"}).env["MAX_TURNS"] == "3"
    # checkpoint / manifest / template / output paths given relative to the launch directory
    rel = {
        "RESUME_PATH": "runs/x/ckpts/global_step_5",
        "EVAL_CKPT_DIR": "runs/x/ckpts",
        "TAU2_MANIFEST_DIR": "manifests",
        "PRESERVE_TEMPLATE": "t.jinja2",
        "OPSD_ROLLOUT_DUMP_DIR": "dumps",
        "OPSD_EMA_DIR": "~/ema",
    }
    e3 = res("toy/main/4bi/glm_a", overrides=rel).env
    assert {k: e3[k] for k in rel} == {
        **{k: str(tmp_path / v) for k, v in rel.items() if not v.startswith("~")},
        "OPSD_EMA_DIR": os.path.expanduser("~/ema"),
    }
    # null / empty / URL values are left alone
    keep = {"RESUME_PATH": "null", "EVAL_CKPT_DIR": "", "OPSD_EMA_DIR": "gs://b/e"}
    e4 = res("toy/main/4bi/glm_a", overrides=keep).env
    assert (e4["RESUME_PATH"], e4["EVAL_CKPT_DIR"], e4["OPSD_EMA_DIR"]) == ("null", "", "gs://b/e")
    # MODEL_PATH: a local dir becomes absolute, a Hugging Face id (no such path here) is kept
    (tmp_path / "exports" / "policy").mkdir(parents=True)
    assert res("toy/main/4bi/glm_a", overrides={"MODEL_PATH": "exports/policy"}).env["MODEL_PATH"] == str(
        tmp_path / "exports" / "policy"
    )
    assert res("toy/main/4bi/glm_a", overrides={"MODEL_PATH": "Qwen/Qwen3-4B"}).env["MODEL_PATH"] == "Qwen/Qwen3-4B"


def test_pi_default_entry_student_override_and_templates():
    a = res("toy/main/4bi/self_summ").env
    assert a["OPSD_HINTS_PATH"] == "/data/toy/hints/self/summ_4bi.json"
    assert a["OPSD_PI_SOURCE"] == "self"
    b = res("toy/main/4bt/self_summ")
    assert b.env["OPSD_HINTS_PATH"] == "/data/toy/hints/self/summ_4bt_special.json"
    assert b.env["MAX_TURNS"] == "20" and b.provenance["MAX_TURNS"][-1][0] == "toy/pis.yaml:self_summ[4bt]"


def test_seed_env_dict_overrides_pi_hint_and_templates_seed():
    r = res("toy/selection/4bi/glm_a/seed=1")
    assert r.env["OPSD_HINTS_PATH"] == "/data/toy/hints/sel/seed1_4bi.json"
    assert r.env["TRAINER_SEED"] == "42"


def test_value_map_and_upper_case_extra_alias():
    h10 = res("toy/teacher/4bi/glm_a/teacher=h10").env
    assert h10["OPSD_TEACHER"] == "hard" and h10["OPSD_TEACHER_SYNC_INTERVAL"] == "10"
    ema = res("toy/teacher/4bi/glm_a/teacher=ema").env
    assert ema["OPSD_TEACHER"] == "ema" and "OPSD_TEACHER_SYNC_INTERVAL" not in ema


def test_pi_none_has_no_hint_and_seed_list_sets_no_seed():
    r = res("toy/grpo/4bi/none/seed=2")
    assert "OPSD_HINTS_PATH" not in r.env and "TRAINER_SEED" not in r.env
    assert r.env["ADVANTAGE_ESTIMATOR"] == "grpo"


def test_locations_and_derived_keys(tmp_path):
    r = res("toy/main/4bi/glm_a", environ={"HOME": "/home/u", "RUN_ROOT": str(tmp_path)})
    assert r.env["OPSD_DATA_ROOT"] == "/home/u/.cache/opsd/data"
    assert r.env["RUN_DIR"] == str(tmp_path / "toy__main__4bi__glm_a")
    assert r.env["CKPT_PATH"] == str(tmp_path / "toy__main__4bi__glm_a" / "ckpts")
    assert r.env["OPSD_ARM_ID"] == "toy/main/4bi/glm_a" and r.env["OPSD_BENCH"] == "toy"
    r2 = res(
        "toy/main/4bi/glm_a",
        overrides={"RUN_NAME": "mine", "OPSD_DATA_ROOT": "/d2"},
        environ={"HOME": "/h", "XDG_CACHE_HOME": "/x", "RUN_ROOT": str(tmp_path)},
    )
    assert r2.env["RUN_DIR"] == str(tmp_path / "mine") and r2.env["OPSD_HINTS_PATH"].startswith("/d2/")
    assert res("toy/main/4bi/glm_a", environ={"XDG_CACHE_HOME": "/x"}).env["OPSD_DATA_ROOT"] == "/x/opsd/data"


# ------------------------------------------------------------------------------------------- status
def test_planned_is_refused_and_in_progress_warns():
    with pytest.raises(R.RecipeError, match="planned"):
        res("toy/future/4bi/glm_a/lambda=0.7")
    assert res("toy/future/4bi/glm_a/lambda=0.7", allow_planned=True).env["OPSD_MIX_LAMBDA"] == "0.7"
    assert any("in_progress" in w for w in res("toy/grpo/4bi/none/seed=1").warnings)
    assert not res("toy/main/4bi/glm_a").warnings


def test_unknown_arm_suggests_and_explains_skips():
    with pytest.raises(R.RecipeError, match="closest"):
        res("toy/opsd_grpo/4bi/glm_ao/lambda=0.9/seed=1")
    with pytest.raises(R.RecipeError, match="no 4bt entry"):
        res("toy/main/4bt/glm_ao")
    with pytest.raises(R.RecipeError, match="no recipes for bench"):
        res("nope/main/4bi/glm_a")
    with pytest.raises(R.RecipeError, match="unknown overlay"):
        res("toy/main/4bi/glm_a", overlays=["nope"])


def test_empty_overlay_and_overlay_by_path(tmp_path):
    assert res("toy/main/4bi/glm_a", overlays=["empty"]).env == res("toy/main/4bi/glm_a").env
    p = tmp_path / "mine.env"
    p.write_text("LR=3e-6\n")
    assert res("toy/main/4bi/glm_a", overlays=[str(p)]).env["LR"] == "3e-6"


def test_overlay_runs_get_their_own_run_dir(tmp_path):
    """A smoke (or any shaping) overlay must not share -- and later resume into -- the full run's RUN_DIR."""
    environ = {"HOME": "/h", "OPSD_DATA_ROOT": "/data", "RUN_ROOT": str(tmp_path)}
    full = res("toy/main/4bi/glm_a", environ=environ).env
    smoke = res("toy/main/4bi/glm_a", overlays=["smoke"], environ=environ).env
    assert full["RUN_NAME"] == "toy__main__4bi__glm_a"
    assert smoke["RUN_NAME"] == "toy__main__4bi__glm_a__smoke"
    assert smoke["RUN_DIR"] == str(tmp_path / "toy__main__4bi__glm_a__smoke") != full["RUN_DIR"]
    assert smoke["CKPT_PATH"] == str(tmp_path / "toy__main__4bi__glm_a__smoke" / "ckpts")  # ${RUN_DIR} follows
    # a no-op overlay changes nothing, so it keeps the arm's directory
    assert res("toy/main/4bi/glm_a", overlays=["empty"], environ=environ).env["RUN_DIR"] == full["RUN_DIR"]
    # every shaping overlay, in order, once; an overlay given as a .env path contributes its file stem
    p = tmp_path / "my lr.env"
    p.write_text("LR=3e-6\n")
    two = res("toy/main/4bi/glm_a", overlays=["smoke", "empty", str(p), "smoke"], environ=environ).env
    assert two["RUN_NAME"] == "toy__main__4bi__glm_a__smoke__my-lr"
    # an explicit RUN_NAME (command line or environment) or RUN_DIR wins over the derived name
    assert res("toy/main/4bi/glm_a", ["smoke"], {"RUN_NAME": "mine"}, environ=environ).env["RUN_NAME"] == "mine"
    assert res("toy/main/4bi/glm_a", ["smoke"], environ=dict(environ, RUN_NAME="amb")).env["RUN_NAME"] == "amb"
    d = res("toy/main/4bi/glm_a", ["smoke"], {"RUN_DIR": str(tmp_path / "x")}, environ=environ).env
    assert d["RUN_DIR"] == str(tmp_path / "x")
    assert R.overlay_run_name_suffix([]) == "" and R.overlay_run_name_suffix(["a/b/c.env", "c"]) == "__c"


# --------------------------------------------------------------------------------- negative inputs
@pytest.mark.parametrize(
    "line,match",
    [
        ("FOO=${HOME}/x", r"\$HOME is not allowed"),
        ("FOO=$(whoami)", "command substitution"),
        ("FOO=${X:-1}", "unsupported shell expansion"),
        ("FOO=1  # comment", "inline comments"),
        ("not a line", "expected KEY=VALUE"),
        ("LR=2", "duplicate key LR"),
    ],
)
def test_bad_env_lines(tree, line, match):
    with open(tree / "toy" / "base.env", "a") as f:
        f.write(line + "\n")
    with pytest.raises(R.RecipeError, match=match):
        R.resolve("toy/main/4bi/glm_a", (), {}, tree, environ=ENV)


def test_bad_yaml_tables(tree):
    arms = tree / "toy" / "arms.yaml"
    text = arms.read_text()
    arms.write_text(text.replace("pis: [glm_ao]", "pis: [glm_aoo]"))
    with pytest.raises(R.RecipeError, match="did you mean"):
        R.load_bench("toy", tree)
    arms.write_text(text.replace("    status: in_progress", "    status: running"))
    with pytest.raises(R.RecipeError, match="status must be"):
        R.load_bench("toy", tree)
    arms.write_text(text + "  - table: main\n    students: [4bi]\n    pis: [glm_a]\n    objective: opsd\n")
    with pytest.raises(R.RecipeError, match="generated twice"):
        R.load_bench("toy", tree)
    arms.write_text(text.replace("seed_var: TRAINER_SEED", "seed_var: TRAINER_SEED\nseed_var: SEED"))
    with pytest.raises(R.RecipeError, match="duplicate key"):
        R.load_bench("toy", tree)
    arms.write_text(
        text.replace("  beta: OPSD_DISTILL_LOSS_COEF\n", "").replace(
            "OPSD_DISTILL_LOSS_COEF: [1e-4, 5e-4, 0.50]", "beta: [1e-4]"
        )
    )
    with pytest.raises(R.RecipeError, match="no env var for arm-id key"):
        R.load_bench("toy", tree)


def test_yaml_merge_keys_override_but_duplicates_fail(tmp_path):
    f = tmp_path / "m.yaml"
    f.write_text("a: &a {X: '1', Y: '2'}\nb: &b {Y: '9', Z: '3'}\nc:\n  <<: [*a, *b]\n  X: '5'\n")
    assert R.load_yaml(f)["c"] == {"X": "5", "Y": "2", "Z": "3"}  # own key wins; earlier alias wins
    f.write_text("c:\n  <<: {X: '1'}\n  X: '5'\n  X: '6'\n")
    with pytest.raises(R.RecipeError, match="duplicate key 'X' at line 4"):
        R.load_yaml(f)


def test_missing_student_or_objective_file(tree):
    (tree / "toy" / "objectives" / "grpo.env").unlink()
    with pytest.raises(R.RecipeError, match="objective 'grpo'"):
        R.resolve("toy/grpo/4bi/none/seed=1", (), {}, tree, environ=ENV)


def test_pis_entry_for_unknown_student_is_an_error(tree):
    p = tree / "toy" / "pis.yaml"
    p.write_text(p.read_text() + "  4bx: {hint: x.json}\n")
    with pytest.raises(R.RecipeError, match="students/4bx.env"):
        R.load_bench("toy", tree)


# ------------------------------------------------------------------------------------------ hermetic
def test_hermetic_violations():
    r = res("toy/main/4bi/glm_a")
    amb = {
        "OPSD_STRAY": "1",
        "LR": "9",
        "MAX_TURNS": "30",
        "OPSD_PREFLIGHT_MIN_FREE_GB": "1",
        "OPSD_PREFLIGHT": "0",
        "OPSD_RESOLVED_ENV": "/x",
        "OPSD_DATA_ROOT": "/elsewhere",
        "PATH": "/bin",
        "OPSD_ARM_ID": "other",
    }
    bad = R.hermetic_violations(r, {}, amb)
    assert len(bad) == 2 and "LR=" in bad[0] and "OPSD_STRAY" in bad[1]  # sorted by name
    assert R.hermetic_violations(r, {"OPSD_STRAY": "1", "LR": "9"}, amb) == []


# ----------------------------------------------------------------------------------------- formats
def test_format_env_is_sorted_sourceable_and_round_trips(tmp_path):
    r = res("toy/main/4bi/glm_a")
    text = R.format_env(r.env)
    keys = [ln.split("=", 1)[0] for ln in text.splitlines()]
    assert keys == sorted(keys)
    f = tmp_path / "resolved.env"
    f.write_text(text)
    assert R.read_resolved_env(f) == r.env
    out = subprocess.run(
        [
            "bash",
            "-c",
            'set -a; . "$1"; set +a; printf "%s|%s|%s" "$REASONING_PREFIX" ' '"$STOP_TOKENS" "$OPSD_ARM_ID"',
            "_",
            str(f),
        ],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    )
    assert out.stdout == "THOUGHTS: |[]|toy/main/4bi/glm_a"


def _cli(*args, env_extra=None):
    root = _repo_root()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": "/home/u",
        "OPSD_DATA_ROOT": "/data",
        # The launchers' convention: the repository root first, so `opsd` is the package (skyrl_gym comes
        # from the SkyRL install, never from PYTHONPATH).
        "PYTHONPATH": str(root),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", "opsd.recipes", "--recipes-dir", str(FIX), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(root),
    )


def test_cli_resolve_list_show_validate(tmp_path):
    out = _cli(
        "resolve",
        "toy/teacher/4bi/glm_a/teacher=h10",
        "LR=5e-6",
        "--overlay",
        "smoke",
        "MAX_TURNS=2",
        "--out",
        str(tmp_path / "r.env"),
    )
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert "LR=5e-6" in lines and "MAX_TURNS=2" in lines and "NUM_GPUS_PER_NODE=4" in lines
    assert lines == sorted(lines) and (tmp_path / "r.env").read_text() == out.stdout
    assert _cli("list", "--table", "grpo").stdout.split() == ["toy/grpo/4bi/none/seed=1", "toy/grpo/4bi/none/seed=2"]
    assert len(_cli("list", "--status", "planned").stdout.split()) == 1
    show = _cli("show", "toy/main/4bt/glm_a", "--json")
    assert json.loads(show.stdout)["provenance"]["MAX_TURNS"][-1] == ["toy/students/4bt.env", "25"]
    v = _cli("validate")
    assert v.returncode == 0 and "skipped toy/main/4bt/glm_ao" in v.stdout
    bad = _cli("resolve", "toy/future/4bi/glm_a/lambda=0.7")
    assert bad.returncode == 2 and "planned" in bad.stderr
    herm = _cli("resolve", "toy/main/4bi/glm_a", "--hermetic", env_extra={"OPSD_STRAY": "1"})
    assert herm.returncode == 3 and "OPSD_STRAY" in herm.stderr
    assert (
        _cli("resolve", "toy/main/4bi/glm_a", "--hermetic", "OPSD_STRAY=1", env_extra={"OPSD_STRAY": "1"}).returncode
        == 0
    )
    assert _cli("resolve", "toy/main/4bi/glm_a", "--bogus").returncode != 0


def test_provenance_records_hint_hashes(tmp_path):
    hint = tmp_path / "data" / "toy" / "hints" / "glm" / "hints_actions.json"
    hint.parent.mkdir(parents=True)
    hint.write_text(
        json.dumps(
            {
                "__schema__": "opsd_hints/v2",
                "t1": "do x",
                "t2": {"task": "do y", "meta": {"a": 1}},
                "t3": {"task": "", "steps": {}},
            }
        )
    )
    expect = R.hint_content_sha256(str(hint))
    assert (
        expect
        == __import__("hashlib")
        .sha256(json.dumps({"t1": "do x", "t2": "do y"}, sort_keys=True, ensure_ascii=False).encode())
        .hexdigest()
    )
    r = res("toy/main/4bi/glm_a", environ={"HOME": "/h", "OPSD_DATA_ROOT": str(tmp_path / "data")})
    env = dict(r.env, OPSD_HINTS_CONTENT_SHA256=expect)
    f = tmp_path / "resolved.env"
    f.write_text(R.format_env(env))
    out = tmp_path / "prov.json"
    rc = R.main(["provenance", str(f), "--out", str(out), "--no-gpu-query", "--strict"])
    rec = json.loads(out.read_text())
    assert rc == 0 and rec["arm_id"] == "toy/main/4bi/glm_a"
    h = rec["hints"]["OPSD_HINTS_PATH"]
    assert h["content_sha256"] == expect and h["content_sha256_matches"] is True
    assert rec["resolved_env"]["sha256"] and "git" in rec and "PYTHONHASHSEED" in rec["runtime_env"]
    # SkyRL is external: the record names the checkout the interpreter imports and the repository's pin.
    assert rec["skyrl"]["pinned_sha"] == _repo.SKYRL_COMMIT and set(rec["skyrl"]) == {
        "path",
        "sha",
        "pinned_sha",
        "matches_pin",
        "dirty",
    }
    f.write_text(R.format_env(dict(env, OPSD_HINTS_CONTENT_SHA256="f" * 64)))
    assert R.main(["provenance", str(f), "--out", str(out), "--no-gpu-query", "--strict"]) == 4


def test_provenance_records_the_runtime_env(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "0")
    monkeypatch.setenv("NCCL_FAKE_API_KEY", "x")  # secret-looking names are never recorded
    monkeypatch.setenv("RAY_ADDRESS", "192.0.2.10:6379")
    monkeypatch.delenv("LOGGER", raising=False)
    rec = R.runtime_env_record()
    assert rec["PYTHONHASHSEED"] == "random" and rec["NCCL_CUMEM_ENABLE"] == "0"
    assert "NCCL_FAKE_API_KEY" not in rec and rec["LOGGER"] is None
    assert rec["ray_mode"] == "external" and "192.0.2.10" not in json.dumps(rec)
    assert R.runtime_env_record({"RAY_ADDRESS": "local"})["ray_mode"] == "local"
    assert R.runtime_env_record({})["ray_mode"] == "local"


def test_skyrl_checkout_is_the_imported_git_checkout(tmp_path, monkeypatch):
    """provenance.json's "skyrl" comes from the checkout the interpreter imports (never a repository path); a
    skyrl that is not the top of a git checkout (a wheel, a tarball) records no commit."""
    plain = (tmp_path / "SkyRL-ecdaa946").resolve()
    (plain / "skyrl").mkdir(parents=True)
    monkeypatch.setattr(R._repo, "skyrl_root", lambda: plain)
    assert R.skyrl_checkout() is None
    subprocess.run(["git", "init", "-q", str(plain)], check=True, capture_output=True)
    assert R.skyrl_checkout() == plain
    monkeypatch.setattr(R._repo, "skyrl_root", lambda: plain / "skyrl")  # inside a checkout, not its top
    assert R.skyrl_checkout() is None


# --------------------------------------------------------------------------------- shipped recipes
def test_shipped_recipe_trees_load_if_present():
    """Every bench under opsd/recipes/ must load and expand (resolution is checked by `validate`)."""
    shipped = _repo.recipes_dir()
    benches = R.list_benches(shipped)
    if not benches:
        pytest.skip("no bench recipes in this tree yet")
    for b in benches:
        assert R.load_bench(b, shipped).arms, b

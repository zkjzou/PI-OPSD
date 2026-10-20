"""AppWorld recipes + launcher vs. the configurations the paper runs actually used.

Fixtures (``golden/``, sanitized copies of what the original runs recorded):
  * ``job_env_<g>.env``      the variables the original launcher passed to the job
  * ``wandb_config_<g>.json`` the trainer's resolved config as logged by the run (the vanilla-OPSD one is the
                             chained final-test pass; its training values are given in the file)

Tests:
  * every arm resolves;
    the main table covers the 16 PIs for every student;
  * job-env parity: each golden variable equals the resolved recipe (hint files mapped to the data bundle;
    infrastructure-only variables are listed in ``INFRA_ENV`` with the reason);
  * the OPSD knobs every OPSD-family arm inherited from the original OPSD run-file (not in any golden);
  * config parity: ``run.sh`` under DRY_RUN=1 produces a hydra command whose parsed config equals the
    golden config on every logged key (infrastructure keys in ``INFRA_CFG``; keys the current schema no
    longer has must be in ``LEGACY_CFG``);
  * DRY_RUN config parse: every arm's command parses into the trainer config and passes validate_cfg;
  * launcher behaviour: smoke-overlay isolation, the disk floor, guards; the offline corpus scripts pin the
    paper's corpus settings.
No GPU, no Ray, no servers: the launchers are only run with DRY_RUN=1.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from opsd import _repo
from opsd.recipes import resolve as R

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"


# The repository root (scripts/, opsd/recipes/).
ROOT = R.repo_root(HERE)
RECIPES = _repo.recipes_dir()
RUN_SH = ROOT / "scripts/appworld/run.sh"
PASSK_SH = ROOT / "scripts/appworld/passk.sh"
PI_UTILITY_SH = ROOT / "scripts/appworld/pi_utility.sh"
DATA_ROOT = "/opsd-data"  # a fixed fake OPSD_DATA_ROOT: nothing is read from it

GOLDEN_ARMS = {
    "dapo": "appworld/grpo/4bi/none/seed=1",
    "dapo_r2": "appworld/grpo/4bi/none/seed=2",
    "dapo_r4": "appworld/grpo/4bi/none/seed=3",
    "opsd": "appworld/main/4bi/glm_aot",
    "lambda": "appworld/opsd_grpo/4bi/glm_aot/lambda=0.99/seed=1",
    "beta": "appworld/beta/4bi/glm_ao/beta=0.1",
    "rlsd": "appworld/rlsd/4bi/glm_aot",
    "lamsched": "appworld/lamsched/4bi/glm_aot",
}

# original hint object keys (below the AppWorld prefix) -> bundle paths (relative to $OPSD_DATA_ROOT)
_B = "appworld/plain/opsd_appworld_pi/hints/"
HINT_SOURCES = {
    "pi_train90/hints_glm52/hints_raw.json": _B + "trajectory/glm-5.2/hints_raw.json",
    "pi_train90/hints_glm52/hints_raw_no_think.json": _B + "trajectory/glm-5.2/hints_raw_no_think.json",
    "pi_train90/hints_glm52/hints_actions.json": _B + "trajectory/glm-5.2/hints_actions.json",
    "pi_train90/hints_glm52/hints_actions_think.json": _B + "trajectory/glm-5.2/hints_actions_think.json",
}

# Job variables that are deliberately not part of the recipes (reason).
INFRA_ENV = {
    "MODEL_NAME": "display name only (no consumer)",
    "DATA_RECIPE": "the launcher always builds the traindev_evalnc layout (train / dev / test_normal / test_challenge)",
    "DUMP_ROLLOUTS": "no consumer in the code",
    "RESTORE_CKPT_FROM_S3": "object-store resume; replaced by RESUME_MODE=latest on a local RUN_DIR",
    "CKPT_S3_KEEP_EVERY": "object-store checkpoint pruning",
    "CKPT_S3_KEEP_EVAL": "object-store checkpoint pruning",
    "OPSD_HINTS_SOURCE": "checked separately against the bundle path (test_job_env_hint_file)",
}

# Trainer-config keys that legitimately differ (reason).
INFRA_CFG = {
    "trainer.logger": "wandb when WANDB_API_KEY is set, else console",
    "trainer.resume_mode": "the launcher resumes from RUN_DIR (latest); the runs resumed from object storage",
    "generator.inference_engine.http_endpoint_port": "host port",
}

# Logged keys that the current trainer-config schema no longer has: an older trainer's algorithm options, and
# the forked SkyRL's trainer.update_ref_every_n_steps (every arm ran it at 0 = off; the release drops the fork
# and its option). All inert for these arms. Any OTHER logged key missing from the parsed config is a dropped
# setting.
LEGACY_CFG = {
    "trainer.algorithm.distill_only",
    "trainer.algorithm.dynamic_teacher",
    "trainer.algorithm.name",
    "trainer.algorithm.skill_kl_coef",
    "trainer.algorithm.skill_prompt_path",
    "trainer.update_ref_every_n_steps",
}

# The OPSD knobs every 4B OPSD-family arm got from the original OPSD run-file's defaults (they are in no
# golden job env). The 30bi OPSD arms ran through the DAPO launcher: no teacher cap of their own, no skipping.
OPSD_RUNFILE_KNOBS = {
    "OPSD_REF_SEQ_CAP": "98304",
    "OPSD_SKIP_NO_HINT": "1",
    "OPSD_TOKEN_CLIP": "0",
    "OPSD_LENGTH_NORM": "0",
    "OPSD_DISTILL_SIDE": "both",
    "OPSD_ENV_COEF": "0",
    "OPSD_DISTILL_COEF": "1.0",
}
PIS16 = (
    "glm_a glm_aat glm_ao glm_aot qwen_a qwen_aat qwen_ao qwen_aot glm_skill qwen_skill glm_summ qwen_summ "
    "glm_chk qwen_chk self_skill self_summ"
).split()


def _num(v):
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("null", "none", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return s


def _same(a, b) -> bool:
    if isinstance(a, list) or isinstance(b, list):
        return isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(map(_same, a, b))
    return _num(a) == _num(b)


def _env_file(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            k, _, v = line.partition("=")
            out[k] = v
    return out


def _resolve(arm: str, overrides=None) -> dict:
    env = {"HOME": "/home/user", "OPSD_DATA_ROOT": DATA_ROOT, "RUN_ROOT": "/runs"}
    return R.resolve(arm, overrides=overrides or {}, recipes_dir=RECIPES, environ=env).env


# ------------------------------------------------------------------------------------------- recipe tree
def test_all_arms_resolve():
    bench = R.load_bench("appworld", RECIPES)
    assert not bench.skipped, bench.skipped
    for arm in bench.arms:
        env = _resolve(arm) if bench.arms[arm].status != "planned" else None
        assert env is not None and env["OPSD_BENCH"] == "appworld"
        if not arm.split("/")[3] == "none":
            assert env["OPSD_HINTS_PATH"].startswith(DATA_ROOT + "/appworld/plain/opsd_appworld_pi/hints/"), arm
            assert len(env["OPSD_HINTS_CONTENT_SHA256"]) == 64, arm


def test_dapo_family_objectives_share_the_dapo_recipe():
    """lambda / beta / RLSD / lamsched must differ from the DAPO baseline only in the OPSD keys."""
    obj = {p.stem: _env_file(p) for p in (RECIPES / "appworld" / "objectives").glob("*.env")}
    dapo_keys = (
        "ADVANTAGE_ESTIMATOR",
        "POLICY_LOSS_TYPE",
        "GRPO_NORM_BY_STD",
        "EPS_CLIP_HIGH",
        "LR",
        "OVERLONG_BUFFER_LEN",
    )
    for name in ("opsd_grpo", "beta", "rlsd", "lamsched"):
        for k in dapo_keys:
            assert obj[name][k] == obj["grpo"][k], (name, k)
        assert obj[name]["USE_KL_LOSS"] == "true" and obj[name]["KL_LOSS_COEF"] == "0.0", name
        assert obj[name]["USE_KL_IN_REWARD"] == "false", name


def test_main_table_has_the_16_pis_for_every_student():
    bench = R.load_bench("appworld", RECIPES)
    for student in ("4bi", "4bt", "30bi"):
        got = {a.split("/")[3] for a in bench.arms if a.startswith(f"appworld/main/{student}/")}
        assert got == set(PIS16), (student, sorted(set(PIS16) ^ got))
        for author, d in (("glm", "glm-5.2"), ("qwen", "qwen3.7-plus")):
            env = _resolve(f"appworld/main/{student}/{author}_aat")
            assert env["OPSD_HINTS_PATH"].endswith(f"/hints/trajectory/{d}/hints_actions_think.json"), (student, author)
    # Action+Thoughts arms ran with exactly the recipe of their student's analyzer-PI arms (W&B configs of the
    # glm_aat / qwen_aat runs differ only in run name, host and paths).
    for student, twin in (("4bi", "glm_skill"), ("4bt", "glm_skill"), ("30bi", "glm_skill")):
        ref = _resolve(f"appworld/main/{student}/{twin}")
        for pi in ("glm_aat", "qwen_aat"):
            env = _resolve(f"appworld/main/{student}/{pi}")
            diff = {k for k in set(env) | set(ref) if env.get(k) != ref.get(k)}
            assert diff <= {
                "OPSD_ARM_ID",
                "RUN_NAME",
                "RUN_DIR",
                "OPSD_ROLLOUT_DUMP_DIR",
                "OPSD_HINTS_PATH",
                "OPSD_HINTS_CONTENT_SHA256",
            }, (student, pi, sorted(diff))


def test_opsd_runfile_knobs():
    arms = [
        "appworld/main/4bi/glm_aot",
        "appworld/main/4bt/glm_skill",
        "appworld/null/4bi/null_task",
        "appworld/opsd_grpo/4bi/glm_chk/lambda=0.9",
        "appworld/beta/4bi/glm_ao/beta=0.1",
        "appworld/rlsd/4bi/glm_aot",
        "appworld/lamsched/4bi/glm_aot",
    ]
    for arm in arms:
        env = _resolve(arm)
        got = {k: env.get(k) for k in OPSD_RUNFILE_KNOBS}
        assert got == OPSD_RUNFILE_KNOBS, (arm, got)
    for pi in ("glm_aot", "glm_skill", "qwen_aat"):
        env = _resolve(f"appworld/main/30bi/{pi}")
        assert (env["OPSD_REF_SEQ_CAP"], env["OPSD_SKIP_NO_HINT"]) == ("", "0"), pi
        assert {k: env[k] for k in OPSD_RUNFILE_KNOBS if k not in ("OPSD_REF_SEQ_CAP", "OPSD_SKIP_NO_HINT")} == {
            k: v for k, v in OPSD_RUNFILE_KNOBS.items() if k not in ("OPSD_REF_SEQ_CAP", "OPSD_SKIP_NO_HINT")
        }, pi


def test_arm_specific_settings():
    a = _resolve("appworld/main/30bi/glm_a")
    assert (a["MAX_MODEL_LEN"], a["MAX_GENERATE_LENGTH"], a["NUM_NODES"], a["MEGATRON_EP"]) == (
        "49152",
        "8192",
        "2",
        "16",
    )
    b = _resolve("appworld/main/30bi/glm_skill")
    assert (b["MAX_MODEL_LEN"], b["MAX_GENERATE_LENGTH"], b["LR"], b["OPSD_REF_SEQ_CAP"]) == (
        "24576",
        "4096",
        "3.0e-6",
        "",
    )
    assert b["EVAL_BEFORE_TRAIN"] == "true" and _resolve("appworld/main/30bi/glm_aot")["EVAL_BEFORE_TRAIN"] == "false"
    assert _resolve("appworld/grpo/30bi/none")["NUM_NODES"] == "1"
    assert _resolve("appworld/main/4bi/glm_a")["VAL_ON_TEST"] == "both"
    assert _resolve("appworld/main/4bi/glm_skill")["VAL_ON_TEST"] == "1"
    assert _resolve("appworld/main/4bt/glm_chk")["OPSD_HINTS_PATH"].endswith("student=4bt/hints_checklist_4bt.json")
    assert _resolve("appworld/main/30bi/self_summ")["OPSD_HINTS_PATH"].endswith("summary/30bi_from_glm52_train90.json")
    h10 = _resolve("appworld/teacher/4bi/glm_aot/teacher=h10")
    assert (h10["OPSD_TEACHER"], h10["OPSD_TEACHER_ALPHA"], h10["OPSD_EMA_INTERVAL"]) == ("ema", "1.0", "10")
    c = _resolve("appworld/rlsd/4bi/glm_aot/lambda=0.99/decay=0")
    assert (c["OPSD_RLSD_LAMBDA"], c["OPSD_RLSD_DECAY_STEPS"], c["OPSD_MIX_LAMBDA"]) == ("0.99", "0", "1.0")
    assert _resolve("appworld/beta/4bi/glm_ao/beta=5e-4/teacher=live")["OPSD_TEACHER"] == "live"
    assert _resolve("appworld/teacher/4bi/glm_aot/teacher=frozen") | {
        "OPSD_ARM_ID": "",
        "RUN_NAME": "",
        "RUN_DIR": "",
        "OPSD_ROLLOUT_DUMP_DIR": "",
    } == _resolve("appworld/lamsched/4bi/glm_aot") | {
        "OPSD_ARM_ID": "",
        "RUN_NAME": "",
        "RUN_DIR": "",
        "OPSD_ROLLOUT_DUMP_DIR": "",
    }


# --------------------------------------------------------------------------------------- job-env parity
@pytest.mark.parametrize("golden", sorted(GOLDEN_ARMS))
def test_job_env_parity(golden):
    job = _env_file(GOLDEN / f"job_env_{golden}.env")
    env = _resolve(GOLDEN_ARMS[golden])
    bad = []
    for k, v in sorted(job.items()):
        if k in INFRA_ENV:
            continue
        if k not in env:
            bad.append(f"{k}={v!r} is not set by the recipe")
        elif not _same(env[k], v):
            bad.append(f"{k}: recipe {env[k]!r} != as-run {v!r}")
    assert not bad, f"{golden} ({GOLDEN_ARMS[golden]}):\n  " + "\n  ".join(bad)


@pytest.mark.parametrize("golden", sorted(GOLDEN_ARMS))
def test_job_env_hint_file(golden):
    job = _env_file(GOLDEN / f"job_env_{golden}.env")
    env = _resolve(GOLDEN_ARMS[golden])
    if "OPSD_HINTS_SOURCE" not in job:
        assert "OPSD_HINTS_PATH" not in env
        return
    assert env["OPSD_HINTS_PATH"] == f"{DATA_ROOT}/{HINT_SOURCES[job['OPSD_HINTS_SOURCE']]}"


# --------------------------------------------------------------------------------- launcher (DRY_RUN) argv
def _bash_ok() -> bool:
    return shutil.which("bash") is not None


def _python_bin() -> str:
    import sys

    return os.environ.get("OPSD_PYTHON") or sys.executable


def dry_run(arm: str, tmp: Path, *extra: str):
    """-> (entry module, argv) of `run.sh <arm>` under DRY_RUN=1, in a clean environment."""
    tmp.mkdir(parents=True, exist_ok=True)
    argv_file = tmp / "argv.bin"
    env = {
        "HOME": str(tmp / "home"),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "TMPDIR": str(tmp),
        "OPSD_PYTHON": _python_bin(),
        "OPSD_DATA_ROOT": DATA_ROOT,
        "RUN_ROOT": str(tmp / "runs"),
        "DRY_RUN": "1",
        "DRY_RUN_ARGV_FILE": str(argv_file),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    p = subprocess.run(["bash", str(RUN_SH), arm, *extra], env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, f"run.sh {arm} exited {p.returncode}:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
    parts = argv_file.read_bytes().split(b"\0")[:-1]
    assert not (tmp / "runs").exists(), "DRY_RUN must not create RUN_DIR"
    assert not list(tmp.glob("opsd_resolved.*")), "DRY_RUN left its temporary resolved env behind"
    return parts[0].decode(), [x.decode() for x in parts[1:]]


def _argv_dict(argv) -> dict:
    out = {}
    for a in argv:
        k, sep, v = a.partition("=")
        assert sep, a
        out[k] = v
    return out


def _config_cls(module: str):
    """The config class the entry module parses with (``<module>.CONFIG_CLS``, i.e. ``OPSDTrainConfig``)."""
    import importlib

    try:
        entry = importlib.import_module(module)
    except Exception as e:  # noqa: BLE001 - the trainer stack is optional for this test module
        pytest.skip(f"entry module {module} not importable here: {e}")
    return entry.CONFIG_CLS


def _flatten(d, pre="") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{pre}.{k}" if pre else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _cfg_value(key, value, cfg):
    """The config value as the run logged it: validate_cfg (run by the entrypoint before logging) rewrites
    two keys, which is mirrored here; paths and the server list are reduced like the fixtures."""
    if key in ("data.train_data", "data.val_data"):
        return [os.path.basename(x) for x in value]
    if key == "environment.skyrl_gym.appworld.server_urls":
        return {"count": len(value)}
    if key == "trainer.strategy" and value == "fsdp2":
        return "fsdp"
    if key == "generator.inference_engine.override_existing_update_group" and value == "auto":
        return "disable" if cfg["generator.inference_engine.run_engines_locally"] else "enable"
    return value


@pytest.mark.skipif(not _bash_ok(), reason="bash not available")
@pytest.mark.parametrize("golden", ["dapo", "dapo_r2", "dapo_r4", "opsd", "lambda", "beta", "rlsd"])
def test_config_parity(golden, tmp_path):
    fixture = json.loads((GOLDEN / f"wandb_config_{golden}.json").read_text())
    want = dict(fixture["config"])
    if "final_test_segment" in fixture:
        want.update(fixture["final_test_segment"]["training_values"])
    module, argv = dry_run(GOLDEN_ARMS[golden], tmp_path)
    expected_module = "opsd.appworld.main_appworld" if golden.startswith("dapo") else "opsd.trainer.main_opsd_appworld"
    assert module == expected_module
    cfg = _flatten(dataclasses.asdict(_config_cls(module).from_cli_overrides(argv)))
    passed = _argv_dict(argv)
    bad, checked = [], 0
    absent = {k for k in want if k not in cfg}
    assert absent <= LEGACY_CFG, f"{golden}: logged keys missing from the parsed config: {sorted(absent - LEGACY_CFG)}"
    for k, v in sorted(want.items()):
        if k in INFRA_CFG or k in absent:
            continue
        got = _cfg_value(k, cfg[k], cfg)
        checked += 1
        if not _same(got, v) and got != v:
            src = "passed by run.sh" if k in passed else "config default"
            bad.append(f"{k}: {got!r} ({src}) != as-run {v!r}")
    assert checked > 250, f"only {checked} keys compared -- fixture/schema mismatch?"
    assert not bad, f"{golden} ({GOLDEN_ARMS[golden]}), {len(bad)} of {checked} keys differ:\n  " + "\n  ".join(bad)


# ----------------------------------------------------------------------------------- DRY_RUN config parse
@pytest.mark.skipif(not _bash_ok(), reason="bash not available")
def test_dry_run_every_arm_parses(tmp_path):
    # Importing the entry modules registers OPSD's no_op advantage estimator and policy losses, which
    # validate_cfg (run by both entrypoints before training) checks for.
    from skyrl.train.utils import validate_cfg

    bench = R.load_bench("appworld", RECIPES)
    arms = [a for a, arm in bench.arms.items() if arm.status != "planned"]
    failures = []
    for i, arm in enumerate(sorted(arms)):
        module, argv = dry_run(arm, tmp_path / f"a{i}")
        cls = _config_cls(module)
        try:
            cfg = cls.from_cli_overrides(argv)
            validate_cfg(cls.from_cli_overrides(argv))
        except Exception as e:  # noqa: BLE001
            failures.append(f"{arm}: {type(e).__name__}: {e}")
            continue
        env = _resolve(arm)
        t = cfg.trainer
        assert t.train_batch_size == int(env["TRAIN_BATCH_SIZE"]) and t.seed == 42, arm
        assert t.policy.optimizer_config.lr == float(env["LR"]), arm
        assert cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length == int(
            env["MAX_MODEL_LEN"]
        ), arm
        assert len(cfg.environment.skyrl_gym.appworld.server_urls) == int(env["NUM_APPWORLD_SERVERS"]), arm
        if env["STRATEGY"] == "megatron":
            assert t.policy.megatron_config.expert_model_parallel_size == int(env["MEGATRON_EP"]), arm
            assert t.ref.megatron_config.expert_model_parallel_size == int(env["MEGATRON_EP"]), arm
        if env.get("EVAL_ONLY") == "1":
            assert t.epochs == 0 and t.eval_before_train and t.ckpt_interval == -1, arm
    assert not failures, "\n".join(failures)


@pytest.mark.skipif(not _bash_ok(), reason="bash not available")
def test_smoke_overlay_dry_run(tmp_path):
    module, argv = dry_run("appworld/main/4bi/glm_chk", tmp_path, "--overlay", "smoke_4gpu")
    a = _argv_dict(argv)
    assert a["trainer.train_batch_size"] == "4" and a["generator.n_samples_per_prompt"] == "2"
    assert a["environment.skyrl_gym.max_env_workers"] == "16"
    assert a["environment.skyrl_gym.appworld.server_urls"].startswith('["http://127.0.0.1:21300"')
    assert ".head8.parquet" in a["data.train_data"] and a["trainer.placement.policy_num_gpus_per_node"] == "4"
    # outside Ray's worker-port range (10002-19999) and the full run's 8000
    assert a["generator.inference_engine.http_endpoint_port"] == "8100"
    # a smoke never shares (and so never resumes into, or FINAL_TEST-ranks) the full run's RUN_DIR
    _, full = dry_run("appworld/main/4bi/glm_chk", tmp_path / "full")
    smoke_ckpt, full_ckpt = a["trainer.ckpt_path"], _argv_dict(full)["trainer.ckpt_path"]
    assert os.path.basename(os.path.dirname(full_ckpt)) == "appworld__main__4bi__glm_chk"
    assert os.path.dirname(smoke_ckpt) != os.path.dirname(full_ckpt), smoke_ckpt


@pytest.mark.skipif(not _bash_ok(), reason="bash not available")
def test_dry_run_guards(tmp_path):
    env = {
        "HOME": str(tmp_path),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "TMPDIR": str(tmp_path),
        "OPSD_PYTHON": _python_bin(),
        "OPSD_DATA_ROOT": DATA_ROOT,
        "RUN_ROOT": str(tmp_path / "runs"),
        "DRY_RUN": "1",
    }

    def run(*args, **extra):
        return subprocess.run(
            ["bash", str(RUN_SH), *args], env={**env, **extra}, capture_output=True, text=True, timeout=300
        )

    p = run("appworld/main/4bi/glm_aot", "MAX_ENV_WORKERS=100")
    assert p.returncode != 0 and "deadlocks" in p.stderr
    p = run("appworld/main/4bi/glm_aot", "USE_KL_LOSS=true")
    assert p.returncode != 0 and "mutually exclusive" in p.stderr
    p = run("appworld/main/4bi/glm_aot", OPSD_TEACHER="live")  # ambient recipe key: hermetic violation
    assert p.returncode != 0 and "HERMETIC" in p.stderr
    p = run("appworld/main/30bi/glm_aot", "--overlay", "smoke_4gpu")
    assert p.returncode != 0
    p = run("appworld/main/4bi/glm_aot", "RESUME_PATH=/ckpts/global_step_50")
    assert p.returncode == 0 and "trainer.resume_mode=from_path" in p.stdout


def _dry_stdout(script: Path, tmp: Path, *args: str, **extra: str) -> str:
    tmp.mkdir(parents=True, exist_ok=True)
    env = {
        "HOME": str(tmp / "home"),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "TMPDIR": str(tmp),
        "OPSD_PYTHON": _python_bin(),
        "OPSD_DATA_ROOT": DATA_ROOT,
        "RUN_ROOT": str(tmp / "runs"),
        "DRY_RUN": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        **extra,
    }
    p = subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, f"{script.name} {args} exited {p.returncode}:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
    return p.stdout + p.stderr


@pytest.mark.skipif(not _bash_ok(), reason="bash not available")
def test_disk_floor_follows_the_checkpoint_footprint(tmp_path):
    """The preflight's free-space floor = the run's peak checkpoint footprint + 50 GB (see run.sh's header)."""

    def floor(*args, **extra):
        out = _dry_stdout(RUN_SH, tmp_path / f"d{len(list(tmp_path.iterdir()))}", *args, **extra)
        line = next(x for x in out.splitlines() if x.startswith("[appworld] disk:"))
        return line.split("preflight floor ")[1].split()[0]

    assert floor("appworld/main/4bi/glm_aot") == str(30 * 45 + 50)  # 30 checkpoints (every epoch end), all kept
    assert floor("appworld/grpo/4bi/none/seed=1") == str(30 * 45 + 50)
    assert floor("appworld/main/30bi/glm_skill") == str(4 * 400 + 50)  # keep 3, peak 4
    assert floor("appworld/step0/4bt/none") == "50"  # evaluation only: no checkpoint
    assert floor("appworld/main/4bi/glm_aot", "EVAL_ONLY=1") == "50"
    assert floor("appworld/main/4bi/glm_aot", "MAX_CKPTS_TO_KEEP=2") == str(3 * 45 + 50)
    assert floor("appworld/main/4bi/glm_aot", OPSD_PREFLIGHT_MIN_FREE_GB="500") == "500"  # the user's value wins
    out = _dry_stdout(RUN_SH, tmp_path / "w", "appworld/main/4bi/glm_aot", OPSD_PREFLIGHT_MIN_FREE_GB="100")
    assert "below this run's checkpoint footprint" in out


def _gen_cmds(out: str) -> dict:
    """DRY_RUN lines of passk.sh / pi_utility.sh -> {label: {--flag: value or True}}."""
    import shlex

    cmds = {}
    for line in out.splitlines():
        if not line.startswith("DRY_RUN ["):
            continue
        label, _, cmd = line[len("DRY_RUN [") :].partition("]:")
        toks, flags, i = shlex.split(cmd)[2:], {}, 0
        while i < len(toks):
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                flags[toks[i]] = toks[i + 1]
                i += 2
            else:
                flags[toks[i]] = True
                i += 1
        cmds[label] = flags
    return cmds


@pytest.mark.skipif(not _bash_ok(), reason="bash not available")
def test_offline_corpus_scripts_pin_the_paper_settings(tmp_path):
    """passk.sh / pi_utility.sh defaults = the paper's AppWorld corpus settings (onpolicy 1.0 / 1.0 / -1 / min_p 0,
    N=128 no-PI on train-90 + dev, N=4 with the PI, 49152 context, strict parsing, binary reward, 30 turns, 8192
    tokens per turn, 20000-char observations; 4bt with its reasoning preserved; 30bi served at TP2)."""
    common = {
        "--temperature": "1.0",
        "--top-p": "1.0",
        "--top-k": "-1",
        "--min-p": "0.0",
        "--max-turns": "30",
        "--max-interactions": "50",
        "--max-generate-length": "8192",
        "--repl-output-max-chars": "20000",
        "--reward-mode": "binary",
        "--strict-parse": True,
    }
    for student in ("4bi", "4bt", "30bi"):
        out = _dry_stdout(PASSK_SH, tmp_path / f"p{student}", student)
        assert "max_model_len=49152" in out and f"vLLM TP={2 if student == '30bi' else 1}" in out, student
        cmds = _gen_cmds(out)
        assert sorted(cmds) == ["dev", "train90"], (student, sorted(cmds))
        for split, flags in cmds.items():
            assert flags["--n-samples"] == "128" and "--hints-path" not in flags, (student, split)
            assert {k: flags.get(k) for k in common} == common, (student, split)
            thinking = student == "4bt"
            assert ("--preserve-thinking" in flags, "--enable-thinking" in flags) == (thinking, thinking), student
        assert cmds["train90"]["--data-file"].endswith("/train.parquet")
        assert cmds["dev"]["--data-file"].endswith("/validation.parquet")
        out = _dry_stdout(PI_UTILITY_SH, tmp_path / f"u{student}", student)
        cmds = _gen_cmds(out)
        assert len(cmds) == 14, (student, sorted(cmds))  # 6 trajectory + 8 analyzer / checklist cells
        for cell, flags in cmds.items():
            assert flags["--n-samples"] == "4" and flags["--hint-inject-style"] == "system-prepend", cell
            assert flags["--hints-path"].startswith(DATA_ROOT + "/appworld/plain/opsd_appworld_pi/hints/"), cell
            assert {k: flags.get(k) for k in common} == common, (student, cell)
            assert ("--preserve-thinking" in flags) == (student == "4bt"), (student, cell)


# ------------------------------------------------------------------------- stale / foreign server detection
_LISTENER = (
    "import socket, sys, time\n"
    "s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
    "s.bind(('127.0.0.1', int(sys.argv[1]))); s.listen(); time.sleep(120)\n"
)


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(not _bash_ok() or not Path("/proc/net/tcp").exists(), reason="needs bash and Linux /proc")
def test_port_owner_check_rejects_a_foreign_listener(tmp_path):
    """A port served by a process this run did not start must fail the readiness check, never be reused."""
    import sys
    import time

    ours, foreign = _free_port(), _free_port()
    other = subprocess.Popen([sys.executable, "-c", _LISTENER, str(foreign)])
    try:
        script = f"""
            source "{ROOT}/scripts/common/bootstrap.sh"
            source "{ROOT}/scripts/appworld/servers.sh"
            export RUN_DIR="{tmp_path}"
            opsd_spawn appworld-server-{ours} "{sys.executable}" -c "$LISTENER" {ours} >/dev/null 2>&1
            opsd_wait_ports 127.0.0.1 {ours} 1 30 || exit 10
            opsd_wait_ports 127.0.0.1 {foreign} 1 30 || exit 11
            aw_check_port_owners {ours} 1 || exit 12
            if aw_check_port_owners {foreign} 1; then exit 13; fi
            opsd_cleanup_own 'appworld-server-*'
            exit 0
        """
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(tmp_path),
            "LISTENER": _LISTENER,
            "OPSD_PYTHON": sys.executable,
            "OPSD_CLEANUP_GRACE_S": "2",
        }
        p = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=120)
        assert p.returncode == 0, f"rc={p.returncode}\n{p.stdout}\n{p.stderr}"
        assert "did not start" in p.stderr
        time.sleep(0.5)
        assert not list((tmp_path / "pids").glob("*.pid")), "our listener was not cleaned up"
    finally:
        other.kill()
        other.wait()

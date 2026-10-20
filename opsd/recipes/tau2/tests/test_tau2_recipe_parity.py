"""Golden parity of the tau2 recipes (opsd/recipes/tau2/) and launcher (scripts/tau2/run.sh).

Two kinds of sanitized fixtures live in ``golden/`` (one file per arm, named after its default run name):

* ``<arm>.env``  -- the stored job environment of the run (cluster, object-store, endpoint and credential-name
  keys removed; hint objects mapped to their data-bundle path; per-arm values of multi-arm jobs applied).
  The resolved recipe must contain every key with the same value.
* ``<arm>.hydra.json`` -- the run's resolved trainer config, restricted to what defines the experiment
  (algorithm, lengths, batch, schedule, sampling, placement, Megatron parallelism, simulator settings;
  locations, endpoints and key-variable names are not included). ``DRY_RUN=1 run.sh <arm>`` must produce an
  argv that parses (``Tau2SkyRLConfig.from_cli_overrides``) and validates (``validate_cfg``) to the same
  values -- i.e. the launcher's OPSD / mix / RLSD forcing is checked too.

Everything is CPU-only: no GPU, Ray, server or network. The config checks import the trainer stack (slow
the first time). Hint files are checked only when ``$OPSD_DATA_ROOT`` holds the tau2 data bundle.
"""

from __future__ import annotations

import concurrent.futures
import csv
import dataclasses
import enum
import importlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    """The repository root: the directory holding ``pyproject.toml``, ``opsd/__init__.py`` and
    ``opsd/skyrl_ext/__init__.py`` (the predicate of ``opsd._repo``, repeated here so the bootstrap below can run
    before ``opsd`` is importable)."""
    for d in Path(__file__).resolve().parents:
        if (
            (d / "pyproject.toml").is_file()
            and (d / "opsd/__init__.py").is_file()
            and (d / "opsd/skyrl_ext/__init__.py").is_file()
        ):
            return d
    raise RuntimeError("repository root (pyproject.toml + opsd/skyrl_ext/) not found above this test")


REPO = _repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from opsd.recipes import resolve as R  # noqa: E402

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"
RECIPES = REPO / "opsd/recipes/"
RUN_SH = REPO / "scripts/tau2/run.sh"
LIB_SH = REPO / "scripts/tau2/lib.sh"
SETUP_SH = REPO / "scripts/setup/setup_tau2.sh"
BOOTSTRAP_SH = REPO / "scripts/common/bootstrap.sh"
TAU2_DATA = REPO / "opsd/tau2_bench/data"
DATA_ROOT = "/opsd-test-data"

ENV_FIXTURES = sorted(GOLDEN.glob("*.env"))
HYDRA_FIXTURES = sorted(GOLDEN.glob("*.hydra.json"))
# Trainer-config fields the runs recorded that the released trainer no longer has (they were only used by
# code that is not released; none of the tau2 runs set a non-default value).
REMOVED_CONFIG_FIELDS = {
    "trainer.algorithm.name",
    "trainer.algorithm.distill_only",
    "trainer.algorithm.dynamic_teacher",
    "trainer.algorithm.skill_kl_coef",
    "trainer.algorithm.skill_prompt_path",
}
OPSD_MODULE = "opsd.tau2_bench.distill.main_tau2_opsd"
GRPO_MODULE = "opsd.tau2_bench.main"
# The 16 PIs of the main table: the paper's 14 main-table rows plus Action+Thoughts (glm_aat, qwen_aat).
AAT_PIS = {"glm_aat", "qwen_aat"}
PAPER_PIS = {
    "glm_a",
    "glm_ao",
    "glm_aot",
    "qwen_a",
    "qwen_ao",
    "qwen_aot",
    "glm_skill",
    "qwen_skill",
    "self_skill",
    "glm_summ",
    "qwen_summ",
    "self_summ",
    "glm_chk",
    "qwen_chk",
}
STUDENTS = ("4bi", "4bt", "30bi")


# ------------------------------------------------------------------------------------------------ helpers
def _bench():
    return R.load_bench("tau2", RECIPES)


def _read_env_fixture(path: Path):
    arm, env = None, {}
    for line in path.read_text().splitlines():
        m = re.match(r"#\s*arm:\s*(\S+)", line)
        if m:
            arm = m.group(1)
            continue
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        env[k] = v
    assert arm, f"{path.name}: missing '# arm:' header"
    return arm, env


def _same_scalar(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=0.0)
    except ValueError:
        return False


def _clean_env(tmp: Path, **extra) -> dict:
    """A minimal, hermetic environment for run.sh (no ambient OPSD_* / API keys / Ray address)."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp / "home"),
        "TMPDIR": str(tmp),
        "OPSD_PYTHON": sys.executable,
        "OPSD_DATA_ROOT": DATA_ROOT,
        "RUN_ROOT": str(tmp / "runs"),
    }
    env.update({k: str(v) for k, v in extra.items()})
    return env


@dataclasses.dataclass
class DryRun:
    module: str
    argv: list
    output: str


def _dry_run(arm: str, tmp: Path, *args: str, **extra) -> DryRun:
    argv_file = tmp / (re.sub(r"[^A-Za-z0-9_.-]", "_", arm + "_".join(args)) + ".argv")
    env = _clean_env(tmp, DRY_RUN=1, DRY_RUN_ARGV_FILE=argv_file, **extra)
    p = subprocess.run(["bash", str(RUN_SH), arm, *args], env=env, capture_output=True, text=True, timeout=600)
    out = p.stdout + p.stderr
    assert p.returncode == 0, f"DRY_RUN {arm} {args} failed (rc={p.returncode}):\n{out[-4000:]}"
    parts = argv_file.read_bytes().split(b"\0")[:-1]
    return DryRun(parts[0].decode(), [x.decode() for x in parts[1:]], out)


def _argv_dict(dr: DryRun) -> dict:
    return dict(a.split("=", 1) for a in dr.argv)


def _flatten(obj, prefix: str = "") -> dict:
    if dataclasses.is_dataclass(obj):
        items = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    elif isinstance(obj, dict):
        items = obj
    else:
        return {prefix: obj}
    out = {}
    for k, v in items.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if dataclasses.is_dataclass(v) or (isinstance(v, dict) and v):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _norm(key: str, v):
    if isinstance(v, enum.Enum):
        v = v.value
    if isinstance(v, tuple):
        v = list(v)
    if key in ("data.train_data", "data.val_data") and isinstance(v, list):
        v = [os.path.basename(str(x)) for x in v]
    if key == "generator.chat_template.name_or_path" and isinstance(v, str):
        v = os.path.basename(v)
    return json.loads(json.dumps(v, default=str))


def _eq(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    return a == b


# ------------------------------------------------------------------------------------------ module fixtures
@pytest.fixture(scope="module")
def dry_runs(tmp_path_factory):
    """DRY_RUN of every arm that has a trainer-config fixture (run in parallel; ~1 s each)."""
    tmp = tmp_path_factory.mktemp("tau2_dry")
    arms = [json.loads(p.read_text())["arm_id"] for p in HYDRA_FIXTURES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        return dict(zip(arms, ex.map(lambda a: _dry_run(a, tmp), arms)))


@pytest.fixture(scope="module")
def tau2_cfg():
    """(Tau2SkyRLConfig, validate_cfg); importing the trainer stack is slow the first time."""
    main = pytest.importorskip("opsd.tau2_bench.main")
    from skyrl.train.utils import validate_cfg

    return main.Tau2SkyRLConfig, validate_cfg


def _parse(dr: DryRun, tau2_cfg):
    cls, validate_cfg = tau2_cfg
    entry = importlib.import_module(dr.module)  # registers the OPSD estimator / loss, exactly as `python -m` does
    # Parse with the class the entry module itself parses with (both tau2 entrypoints use Tau2SkyRLConfig).
    assert entry.CONFIG_CLS is cls, (dr.module, entry.CONFIG_CLS)
    cfg = entry.CONFIG_CLS.from_cli_overrides(dr.argv)
    validate_cfg(cfg)
    return cfg


# --------------------------------------------------------------------------------------------- the recipes
def test_recipes_load():
    bench = _bench()
    assert not bench.skipped, f"(student, pi) pairs without a hint: {bench.skipped}"
    arms = {k: a for k, a in bench.arms.items()}
    assert len(arms) == 69
    planned = sorted(k for k, a in arms.items() if a.status == "planned")
    assert planned == ["tau2/lamsched/4bi/glm_aot"]


def test_every_runnable_arm_has_a_trainer_config_fixture():
    fixtures = {json.loads(p.read_text())["arm_id"] for p in HYDRA_FIXTURES}
    runnable = {k for k, a in _bench().arms.items() if a.status == "done"}
    # 48 main (16 PIs x 3) + 3 DAPO seeds + 1 tri352 DAPO + 2 self-hosted-simulator DAPO + 12 mix + 2 RLSD
    assert len(runnable) == 68
    assert runnable == fixtures, sorted(runnable ^ fixtures)


def test_main_table_has_all_16_pis_for_every_student():
    """Every student trains all 16 PIs; the paper's 14 main-table PIs point at that table,
    Action+Thoughts (not in the paper) does not."""
    bench = _bench()
    assert len(AAT_PIS | PAPER_PIS) == 16
    for student in STUDENTS:
        main = {k.split("/")[3]: a for k, a in bench.arms.items() if k.startswith(f"tau2/main/{student}/")}
        assert set(main) == PAPER_PIS | AAT_PIS, f"{student}: {sorted(set(main) ^ (PAPER_PIS | AAT_PIS))}"
        assert all(a.status == "done" for a in main.values()), student
        assert {p for p, a in main.items() if a.entry.paper == "tab:opsd_results"} == PAPER_PIS, student
        assert all(main[p].entry.paper is None for p in AAT_PIS), student


@pytest.mark.parametrize("path", ENV_FIXTURES, ids=lambda p: p.name[: -len(".env")])
def test_resolved_recipe_contains_the_stored_run_environment(path):
    arm, golden = _read_env_fixture(path)
    res = R.resolve(arm, recipes_dir=RECIPES, environ={"HOME": "/h", "OPSD_DATA_ROOT": DATA_ROOT}, allow_planned=True)
    bad = []
    for k, v in sorted(golden.items()):
        want = v.replace("${OPSD_DATA_ROOT}", DATA_ROOT)
        got = res.env.get(k)
        if got is None or not _same_scalar(got, want):
            bad.append(f"{k}: recipe={got!r} run={want!r}")
    assert not bad, f"{arm}: the recipe differs from what the run used:\n  " + "\n  ".join(bad)


@pytest.mark.parametrize(
    "arm, cap, require_hint, pool",
    [
        ("tau2/main/4bi/glm_a", "80000", "1", "retail_tri352.jsonl"),
        ("tau2/main/4bi/glm_skill", "48000", "1", "retail_tri352.jsonl"),
        ("tau2/main/4bi/glm_chk", "48000", "1", "retail_chk4bi323.jsonl"),
        ("tau2/main/4bi/qwen_aat", "80000", "1", "retail_tri352.jsonl"),
        # 4bt trajectory forms: glm ran at 80000, qwen at 100000
        ("tau2/main/4bt/glm_a", "80000", "1", "retail_tri352.jsonl"),
        ("tau2/main/4bt/glm_ao", "80000", "1", "retail_tri352.jsonl"),
        ("tau2/main/4bt/glm_aot", "80000", "1", "retail_tri352.jsonl"),
        ("tau2/main/4bt/glm_aat", "80000", "1", "retail_tri352.jsonl"),
        ("tau2/main/4bt/qwen_a", "100000", "1", "retail_tri352.jsonl"),
        ("tau2/main/4bt/qwen_aat", "100000", "1", "retail_tri352.jsonl"),
        ("tau2/main/4bt/qwen_chk", "64000", "1", "retail_chk4bt344.jsonl"),
        # glm_skill / glm_chk on 4bt ran at 48000; 64000 is kept on purpose (see pis.yaml)
        ("tau2/main/4bt/glm_skill", "64000", "1", "retail_tri352.jsonl"),
        ("tau2/main/4bt/glm_chk", "64000", "1", "retail_chk4bt347.jsonl"),
        ("tau2/main/30bi/glm_aot", "56000", "1", "retail_tri352.jsonl"),
        ("tau2/main/30bi/qwen_aot", "56000", "0", "retail_tri352.jsonl"),
        ("tau2/main/30bi/qwen_aat", "80000", "1", "retail_tri352.jsonl"),
        ("tau2/main/30bi/qwen_skill", "80000", "1", "retail_tri352.jsonl"),
        ("tau2/main/30bi/qwen_chk", "80000", "1", "retail_chk30bi348.jsonl"),
        ("tau2/opsd_grpo/4bi/glm_aot/lambda=0.99", "56000", "1", "retail_pi424.jsonl"),
        ("tau2/opsd_grpo/4bi/glm_chk/lambda=0.5", "48000", "1", "retail_pi424.jsonl"),
        ("tau2/rlsd/4bi/glm_aot/pool=tri352", "56000", "1", "retail_tri352.jsonl"),
    ],
)
def test_ref_seq_cap_require_hint_and_pool_as_run(arm, cap, require_hint, pool):
    env = R.resolve(arm, recipes_dir=RECIPES, environ={"HOME": "/h", "OPSD_DATA_ROOT": DATA_ROOT}).env
    assert env["OPSD_REF_SEQ_CAP"] == cap
    assert env["OPSD_REQUIRE_HINT"] == require_hint
    assert env["TRAIN_DATA"] == pool
    assert (TAU2_DATA / pool).is_file()
    assert env["OPSD_HINTS_PATH"].startswith(DATA_ROOT + "/tau2/hints/")
    assert re.fullmatch(r"[0-9a-f]{64}", env["OPSD_HINTS_CONTENT_SHA256"])


def test_every_training_pool_manifest_is_shipped():
    for arm in _bench().arms:
        env = R.resolve(arm, recipes_dir=RECIPES, environ={"HOME": "/h"}, allow_planned=True).env
        for key in ("TRAIN_DATA", "VAL_DATA"):
            assert (TAU2_DATA / env[key]).is_file(), f"{arm}: {key}={env[key]} not under {TAU2_DATA}"


def test_hint_files_match_the_bundle_and_cover_the_pool():
    """Needs the data bundle: every hint file exists, has its declared content sha256 and covers every task of
    the arm's training pool (OPSD_REQUIRE_HINT=1 fails a run on the first uncovered task)."""
    root = os.environ.get("OPSD_DATA_ROOT")
    if not root or not (Path(root) / "tau2" / "hints").is_dir():
        pytest.skip("OPSD_DATA_ROOT does not hold the tau2 data bundle")
    from opsd.pi.artifact import load_task_hints

    seen = {}
    for arm in _bench().arms:
        env = R.resolve(
            arm, recipes_dir=RECIPES, environ={"HOME": "/h", "OPSD_DATA_ROOT": root}, allow_planned=True
        ).env
        hp = env.get("OPSD_HINTS_PATH")
        if not hp:
            continue
        assert Path(hp).is_file(), f"{arm}: missing {hp}"
        if hp not in seen:
            assert R.hint_content_sha256(hp) == env["OPSD_HINTS_CONTENT_SHA256"], f"{arm}: content sha256 of {hp}"
            seen[hp] = {k for k, v in load_task_hints(hp).items() if str(v or "").strip()}
        ids = [json.loads(line)["task_id"] for line in open(TAU2_DATA / env["TRAIN_DATA"]) if line.strip()]
        uncovered = [t for t in ids if t not in seen[hp]]
        if env.get("OPSD_REQUIRE_HINT") == "1":
            assert not uncovered, f"{arm}: {len(uncovered)} of {len(ids)} pool tasks have no hint in {hp}"


# --------------------------------------------------------------------------------------------- the launcher
def test_launcher_scripts_parse():
    for script in (RUN_SH, LIB_SH, SETUP_SH):
        p = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert p.returncode == 0, f"{script}: {p.stderr}"


def test_pinned_tau2_revision_matches_the_code():
    from opsd.tau2_bench.constants import TAU2_REVISION

    for script in (RUN_SH, SETUP_SH):
        pins = set(re.findall(r"tau2-bench/([0-9a-f]{40})|TAU2_REVISION=([0-9a-f]{40})", script.read_text()))
        pins = {a or b for a, b in pins}
        assert pins == {TAU2_REVISION}, f"{script.name} pins {pins}, constants.py pins {TAU2_REVISION}"


@pytest.mark.parametrize("path", HYDRA_FIXTURES, ids=lambda p: p.name[: -len(".hydra.json")])
def test_dry_run_config_matches_the_run(path, dry_runs, tau2_cfg):
    fx = json.loads(path.read_text())
    cfg = _parse(dry_runs[fx["arm_id"]], tau2_cfg)
    flat = {k: _norm(k, v) for k, v in _flatten(cfg).items()}
    bad = []
    for key, want in sorted(fx["config"].items()):
        if key in REMOVED_CONFIG_FIELDS:
            continue
        if key not in flat:
            bad.append(f"{key}: not in the trainer config (run had {want!r})")
        elif not _eq(flat[key], _norm(key, want)):
            bad.append(f"{key}: launcher={flat[key]!r} run={want!r}")
    assert not bad, f"{fx['arm_id']}: the launcher's config differs from the run:\n  " + "\n  ".join(bad)


def test_entry_module_follows_the_objective(dry_runs):
    for arm, dr in dry_runs.items():
        table = arm.split("/")[1]
        want = GRPO_MODULE if table == "grpo" else OPSD_MODULE
        assert dr.module == want, f"{arm}: entry module {dr.module}"
        if table == "main":  # vanilla OPSD: the launcher FORCES the pure-distillation recipe
            assert "[opsd] FORCING USE_KL_IN_REWARD=true" in dr.output
            assert "[opsd] FORCING MAX_GRAD_NORM=0.5" in dr.output
        elif table in ("opsd_grpo", "rlsd"):
            assert "[opsd] FORCING ADV_ESTIMATOR" not in dr.output  # grpo already
            assert ("MIX MODE" in dr.output) == (table == "opsd_grpo")
            assert ("RLSD MODE" in dr.output) == (table == "rlsd")


def test_dry_run_writes_nothing_and_prints_the_command(tmp_path):
    dr = _dry_run("tau2/main/4bi/glm_aot", tmp_path)
    assert not (tmp_path / "runs").exists(), "DRY_RUN must not create RUN_DIR"
    assert f"-m {OPSD_MODULE}" in dr.output
    assert "OPSD_HINTS_PATH=" in dr.output and "# resolved environment" in dr.output
    args = _argv_dict(dr)
    assert args["trainer.ckpt_path"] == str(tmp_path / "runs" / "tau2__main__4bi__glm_aot" / "ckpts")
    assert args["trainer.resume_mode"] == "latest"
    assert args["trainer.logger"] == "console"  # no WANDB_API_KEY in the environment
    assert "tau2.simulator_base_url" not in args  # resolves to $OPENAI_BASE_URL / the public API in the driver


def test_smoke_overlay_dry_run(tmp_path, tau2_cfg):
    dr = _dry_run("tau2/main/4bi/glm_chk", tmp_path, "--overlay", "smoke_4gpu")
    a = _argv_dict(dr)
    run_dir = tmp_path / "runs" / "tau2__main__4bi__glm_chk__smoke_4gpu"  # an overlay suffixes the run name
    assert a["data.train_data"] == f"['{run_dir}/data/retail_chk4bi323.first8.jsonl']"
    assert a["data.val_data"] == f"['{run_dir}/data/retail_base.first8.jsonl']"
    assert a["generator.inference_engine.num_engines"] == "4"
    assert a["trainer.placement.policy_num_gpus_per_node"] == "4"
    assert a["tau2.max_steps"] == a["generator.max_turns"] == "60"  # a turn cap, not the common overlay's 2
    assert (a["trainer.train_batch_size"], a["generator.n_samples_per_prompt"]) == ("4", "2")
    assert (a["trainer.epochs"], a["tau2.rollout_batches"]) == ("1", "2")
    assert a["generator.eval_n_samples_per_prompt"] == "1" and a["trainer.eval_batch_size"] == "8"
    _parse(dr, tau2_cfg)  # colocation arithmetic: 4 GPUs == 4 engines x TP 1


def test_cli_override_and_hermetic_check(tmp_path):
    dr = _dry_run("tau2/main/4bi/glm_aot", tmp_path, "OPSD_REF_SEQ_CAP=90000", "MAX_CONCURRENCY=32")
    assert "OPSD_REF_SEQ_CAP=90000" in dr.output
    assert _argv_dict(dr)["tau2.max_concurrency"] == "32"
    env = _clean_env(tmp_path, DRY_RUN=1, OPSD_MIX_LAMBDA="0.5")
    p = subprocess.run(["bash", str(RUN_SH), "tau2/main/4bi/glm_aot"], env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "HERMETIC" in p.stderr


def test_ambient_result_knobs_must_come_through_the_command_line(tmp_path):
    """Launcher knobs that no recipe sets (e.g. ADV_ESTIMATOR) change results; an ambient value is refused, a
    command-line value is accepted and recorded in resolved.env."""
    env = _clean_env(tmp_path, DRY_RUN=1, ADV_ESTIMATOR="rloo")
    p = subprocess.run(["bash", str(RUN_SH), "tau2/grpo/4bi/none/seed=1"], env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "ambient ADV_ESTIMATOR=rloo is not part of the recipe" in p.stderr
    dr = _dry_run("tau2/grpo/4bi/none/seed=1", tmp_path, "ADV_ESTIMATOR=rloo")
    assert _argv_dict(dr)["trainer.algorithm.advantage_estimator"] == "rloo"
    assert "ADV_ESTIMATOR=rloo" in dr.output  # printed from resolved.env


def test_manifest_dir_and_train_nodes_must_come_through_the_command_line(tmp_path):
    """TAU2_MANIFEST_DIR replaces every manifest and NUM_TRAIN_NODES the trainer placement: neither may come
    in from the environment unrecorded."""
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for name in ("retail_tri352.jsonl", "retail_base.jsonl"):
        rows = (TAU2_DATA / name).read_text().splitlines()[:3]
        (manifests / name).write_text("\n".join(rows) + "\n")
    for knob, value in (("TAU2_MANIFEST_DIR", manifests), ("NUM_TRAIN_NODES", 2)):
        env = _clean_env(tmp_path, DRY_RUN=1, **{knob: value})
        p = subprocess.run(["bash", str(RUN_SH), "tau2/main/4bi/glm_aot"], env=env, capture_output=True, text=True)
        assert p.returncode != 0 and f"ambient {knob}=" in p.stderr, (knob, p.stderr[-2000:])
    dr = _dry_run("tau2/main/4bi/glm_aot", tmp_path, f"TAU2_MANIFEST_DIR={manifests}")
    assert _argv_dict(dr)["data.train_data"] == f"['{manifests}/retail_tri352.jsonl']"
    assert f"TAU2_MANIFEST_DIR={manifests}" in dr.output  # recorded in resolved.env


def _fake_ready_host(tmp: Path) -> dict:
    """A tau2 checkout stub and hint files, so a non-dry run gets as far as its endpoint checks."""
    src = tmp / "tau2-src"
    (src / "src" / "tau2").mkdir(parents=True)
    data = tmp / "data"
    for rel in (
        "tau2/hints/glm-5.2/pool424/hints_raw.json",
        "tau2/hints/checklist/pi_tau2_chk_tri352/4bi/glm/hints_checklist.json",
    ):
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        (data / rel).write_text("{}")
    return {"TAU2_SOURCE_DIR": src, "OPSD_DATA_ROOT": data}


def test_real_run_refuses_to_start_without_the_api_key(tmp_path):
    env = _clean_env(tmp_path, **_fake_ready_host(tmp_path))
    p = subprocess.run(
        ["bash", str(RUN_SH), "tau2/main/4bi/glm_aot"], env=env, capture_output=True, text=True, timeout=300
    )
    assert p.returncode != 0
    assert "set OPENAI_API_KEY" in p.stderr and "missing API key" in p.stderr
    assert "[preflight]" not in p.stderr  # stopped before touching GPUs, ports or servers


def _stub_python(tmp: Path) -> Path:
    """A python that runs everything except the tau2 trainer, which it replaces by an echo + exit 7."""
    stub = tmp / "python"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = -m ] && [ "${2#opsd.tau2_bench.}" != "$2" ]; then\n'
        '  echo "STUB $2 nargs=$(($# - 2)) hints=$OPSD_HINTS_PATH ray=$RAY_ADDRESS"; exit 7\n'
        "fi\n"
        f'exec "{sys.executable}" "$@"\n'
    )
    stub.chmod(0o755)
    return stub


def test_real_run_path_with_a_stub_trainer(tmp_path):
    """The non-dry path up to the trainer, with the trainer replaced by a stub (preflight skipped: no GPUs).
    Checks the run layout, provenance, the captured trainer log and that the trainer's exit code is returned."""
    env = _clean_env(tmp_path, **_fake_ready_host(tmp_path), OPENAI_API_KEY="test-key-not-real", OPSD_PREFLIGHT=0)
    env["OPSD_PYTHON"] = str(_stub_python(tmp_path))
    arm = "tau2/opsd_grpo/4bi/glm_aot/lambda=0.99"
    p = subprocess.run(
        ["bash", str(RUN_SH), arm, "OPSD_HINTS_CONTENT_SHA256="], env=env, capture_output=True, text=True, timeout=300
    )
    assert p.returncode == 7, p.stdout + p.stderr
    run_dir = tmp_path / "runs" / "tau2__opsd_grpo__4bi__glm_aot__lambda-0.99"
    assert f"STUB {OPSD_MODULE}" in p.stdout and "ray=local" in p.stdout
    for sub in ("ckpts", "exports", "trajectories", "logs", "resolved.env", "provenance.json", "command.sh"):
        assert (run_dir / sub).exists(), sub
    prov = json.loads((run_dir / "provenance.json").read_text())
    assert prov["arm_id"] == arm and prov["resolved_env"]["sha256"]
    (log,) = list((run_dir / "logs").glob("trainer-*.log"))
    assert "STUB" in log.read_text()
    assert not list((run_dir / "pids").glob("*.pid")), "tracked PIDs must be cleaned up on exit"


def _stub_python_with_ray_probe(tmp: Path, missing: str) -> Path:
    """_stub_python, plus a stand-in for the Ray API-key probe of lib.sh that reports MISSING as the names a
    Ray task on the driver node cannot see (no Ray cluster is started). The names it was asked about are
    appended to <tmp>/probe_args."""
    stub = _stub_python(tmp)
    body = stub.read_text().split("\n", 1)[1]
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = -c ]; then case "$2" in *"# tau2_ray_key_probe"*)\n'
        f'  echo "${{*:3}}" >>"{tmp}/probe_args"; echo "TAU2_KEY_PROBE headnode {missing}"; exit 0 ;; esac; fi\n' + body
    )
    return stub


def test_external_ray_checks_the_api_key_where_the_generator_runs(tmp_path):
    """On an existing Ray cluster the simulator / judge keys are read inside a Ray task on the driver node,
    whose environment comes from `ray start`: the launcher must verify them there, not only in its shell."""
    base = dict(_fake_ready_host(tmp_path), OPENAI_API_KEY="test-key-not-real", OPSD_PREFLIGHT=0)
    base["RAY_ADDRESS"] = "head.example:6379"
    arm = "tau2/main/4bi/glm_aot"
    # the head was started without the key: refuse, with the fix, before anything runs
    env = _clean_env(tmp_path, **base)
    env["OPSD_PYTHON"] = str(_stub_python_with_ray_probe(tmp_path, "OPENAI_API_KEY"))
    p = subprocess.run(
        ["bash", str(RUN_SH), arm, "OPSD_HINTS_CONTENT_SHA256="], env=env, capture_output=True, text=True
    )
    assert p.returncode != 0, p.stdout + p.stderr
    assert (tmp_path / "probe_args").read_text() == "OPENAI_API_KEY\n"  # each needed key name, once
    assert "do not see: OPENAI_API_KEY" in p.stderr
    assert "export OPENAI_API_KEY before `scripts/common/ray_cluster.sh head`" in p.stderr
    assert "STUB" not in p.stdout
    # the head has the key: the run proceeds to the trainer
    env["OPSD_PYTHON"] = str(_stub_python_with_ray_probe(tmp_path, ""))
    p = subprocess.run(
        ["bash", str(RUN_SH), arm, "OPSD_HINTS_CONTENT_SHA256="], env=env, capture_output=True, text=True
    )
    assert p.returncode == 7 and "ray=head.example:6379" in p.stdout, p.stdout + p.stderr
    # a single-node run never probes (its Ray cluster is started by the driver itself)
    env.pop("RAY_ADDRESS")
    (tmp_path / "probe_args").unlink()
    p = subprocess.run(
        ["bash", str(RUN_SH), arm, "OPSD_HINTS_CONTENT_SHA256="], env=env, capture_output=True, text=True
    )
    assert p.returncode == 7 and not (tmp_path / "probe_args").exists(), p.stdout + p.stderr
    # DRY_RUN never contacts the cluster; it says what a real run checks
    dr = _dry_run("tau2/main/30bi/glm_aot", tmp_path, RAY_ADDRESS="head.example:6379")
    assert "a real run first checks that Ray tasks on this node see: OPENAI_API_KEY" in dr.output


def test_ray_key_probe_reports_what_is_missing(tmp_path):
    ok = _stub_python_with_ray_probe(tmp_path, "")
    p = _lib(tmp_path, "tau2_check_keys_on_ray_driver_node OPENAI_API_KEY", OPSD_PYTHON=ok, RAY_ADDRESS="h:1")
    assert p.returncode == 0, p.stderr
    (tmp_path / "b").mkdir()
    bad = _stub_python_with_ray_probe(tmp_path / "b", "OPENAI_API_KEY MY_JUDGE_KEY")
    p = _lib(
        tmp_path, "tau2_check_keys_on_ray_driver_node OPENAI_API_KEY MY_JUDGE_KEY", OPSD_PYTHON=bad, RAY_ADDRESS="h:1"
    )
    assert p.returncode == 1 and "(headnode) do not see: OPENAI_API_KEY MY_JUDGE_KEY" in p.stderr
    assert _lib(tmp_path, "tau2_check_keys_on_ray_driver_node", OPSD_PYTHON=bad).returncode == 0  # nothing to check


def test_smoke_overlay_truncates_the_task_lists(tmp_path):
    env = _clean_env(tmp_path, **_fake_ready_host(tmp_path), OPENAI_API_KEY="test-key-not-real", OPSD_PREFLIGHT=0)
    env["OPSD_PYTHON"] = str(_stub_python(tmp_path))
    p = subprocess.run(
        ["bash", str(RUN_SH), "tau2/main/4bi/glm_chk", "--overlay", "smoke_4gpu", "OPSD_HINTS_CONTENT_SHA256="],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert p.returncode == 7, p.stdout + p.stderr
    data = tmp_path / "runs" / "tau2__main__4bi__glm_chk__smoke_4gpu" / "data"
    train, val = data / "retail_chk4bi323.first8.jsonl", data / "retail_base.first8.jsonl"
    assert train.read_text().splitlines() == (TAU2_DATA / "retail_chk4bi323.jsonl").read_text().splitlines()[:8]
    assert len(val.read_text().splitlines()) == 8


def test_self_hosted_simulator_arm(tmp_path):
    env = _clean_env(tmp_path, **_fake_ready_host(tmp_path), OPENAI_API_KEY="test-key-not-real")
    p = subprocess.run(["bash", str(RUN_SH), "tau2/grpo/4bi/none/sim=qwen35"], env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "SIMULATOR_BASE_URL" in p.stderr
    dr = _dry_run(
        "tau2/grpo/4bi/none/sim=qwen35", tmp_path, SIM_SERVE_LOCAL=1, SIM_CUDA_VISIBLE_DEVICES="8,9,10,11,12,13,14,15"
    )
    a = _argv_dict(dr)
    assert a["tau2.simulator_model"] == "Qwen/Qwen3.5-35B-A3B"
    assert a["tau2.simulator_base_url"] == ",".join(f"http://127.0.0.1:{8000 + i}/v1" for i in range(4))
    assert a["tau2.eval_simulator_model"] == "gpt-5.6-luna"  # evaluation keeps the remote yardstick
    assert a["data.train_data"].endswith("retail_areal_instruct_drop8of8.jsonl']")
    assert "would start: 4 x serve_user_sim.sh" in dr.output
    # the local simulator needs no key; the remote evaluation simulator and the judge still do
    assert "simulated user" not in dr.output and "evaluation simulator (gpt-5.6-luna" in dr.output


def test_multinode_arm_needs_a_ray_cluster(tmp_path):
    env = _clean_env(tmp_path, **_fake_ready_host(tmp_path), OPENAI_API_KEY="test-key-not-real")
    p = subprocess.run(["bash", str(RUN_SH), "tau2/main/30bi/glm_aot"], env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "multi-node Ray cluster" in p.stderr


def test_eval_only_is_refused(tmp_path):
    env = _clean_env(tmp_path, DRY_RUN=1, EVAL_ONLY=1)
    p = subprocess.run(["bash", str(RUN_SH), "tau2/main/4bi/glm_aot"], env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "EVAL_ONLY is not supported" in p.stderr


# ------------------------------------------------------------------------------------- resume helpers (lib.sh)
def _lib(tmp: Path, snippet: str, **extra) -> subprocess.CompletedProcess:
    env = _clean_env(tmp, OPSD_BOOTSTRAP_AUTO=0, **extra)
    script = f'source "{BOOTSTRAP_SH}"; source "{LIB_SH}"; {snippet}'
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=300)


def test_resume_checkpoint_lookup(tmp_path):
    ck = tmp_path / "ckpts"
    ck.mkdir()
    assert _lib(tmp_path, f'tau2_resume_checkpoint "{ck}"').stdout == ""
    (ck / "latest_ckpt_global_step.txt").write_text("100\n")
    assert _lib(tmp_path, f'tau2_resume_checkpoint "{ck}"').stdout.strip() == f"{ck}/global_step_100"
    p = _lib(tmp_path, f'tau2_resume_checkpoint "{ck}"', RESUME_MODE="from_path", RESUME_PATH="/x/global_step_7")
    assert p.stdout.strip() == "/x/global_step_7"


def test_epoch_boundary_dataloader_reset_is_opt_in(tmp_path):
    torch = pytest.importorskip("torch")
    ck = tmp_path / "global_step_100"
    ck.mkdir()
    torch.save({"_num_yielded": 100, "_iterator_finished": False}, ck / "data.pt")
    _lib(tmp_path, f'tau2_maybe_reset_dataloader "{ck}"', ROLLOUT_BATCHES=100)  # default: never touched
    assert (ck / "data.pt").is_file()
    torch.save({"_num_yielded": 40}, ck / "data.pt")
    _lib(tmp_path, f'tau2_maybe_reset_dataloader "{ck}"', ROLLOUT_BATCHES=100, RESUME_RESET_DATALOADER="auto")
    assert (ck / "data.pt").is_file(), "a mid-epoch iterator must be kept"
    torch.save({"_num_yielded": 100}, ck / "data.pt")
    p = _lib(tmp_path, f'tau2_maybe_reset_dataloader "{ck}"', ROLLOUT_BATCHES=100, RESUME_RESET_DATALOADER="auto")
    assert not (ck / "data.pt").exists(), p.stderr
    assert len(list(ck.glob("data.pt.exhausted-*"))) == 1, "set aside, never deleted"

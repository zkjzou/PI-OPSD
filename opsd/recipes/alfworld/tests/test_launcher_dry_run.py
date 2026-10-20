"""DRY_RUN of the ALFWorld launcher (scripts/alfworld/run.sh), one arm per table, CPU only.

For each arm the launcher resolves the recipe, applies its guards and prints the trainer module, the full
hydra argv and the resolved environment, without starting servers or touching a GPU. The argv is then parsed
exactly the way the entry points do it (import the entry module, ``<module>.CONFIG_CLS.from_cli_overrides``,
``validate_cfg``), and the parsed config is checked against the resolved recipe.
"""

from __future__ import annotations

import ast
import importlib
import os
import re
import shlex
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from opsd import _repo
from opsd.recipes import resolve as R

# The repository root (scripts/, opsd/recipes/).
ROOT = R.repo_root()
RUN_SH = ROOT / "scripts/alfworld/run.sh"
OPSD = "opsd.trainer.main_opsd_alfworld"
DAPO = "opsd.alfworld.main_alfworld_dapo"


def _rescore_args(arm_id: str) -> Tuple[str, ...]:
    """The arm's checkpoint re-score of record (`rescore:` in arms.yaml) as KEY=VAL launcher arguments."""
    meta = R.load_bench("alfworld").arms[arm_id].entry.meta
    return tuple(f"{k}={v}" for k, v in meta["rescore"].items())


# One arm per table (and per student), plus the alternative modes of the launcher.
ARMS = {
    "step0_30bi": ("alfworld/step0/30bi/none",),
    "main_4bt_guide": ("alfworld/main/4bt/qwen_aot",),
    "main_4bi_campaign": ("alfworld/main/4bi/glm_chk",),
    "null": ("alfworld/null/4bi/null_task",),
    "null_xdomain": ("alfworld/null/4bi/null_xdomain",),
    "opsd_grpo": ("alfworld/opsd_grpo/4bi/glm_skill/lambda=0.999",),
    "beta": ("alfworld/beta/4bi/glm_ao/beta=5e-4",),
    "teacher": ("alfworld/teacher/4bi/glm_ao/teacher=h10",),
    "rlsd": ("alfworld/rlsd/4bi/glm_chk/decay=100",),
    "grpo": ("alfworld/grpo/4bi/none/seed=2",),
    "lamsched": ("alfworld/lamsched/4bi/glm_ao",),
    "passcode_utility": ("alfworld/passcode/4bi/none/eval=utility/placement=system",),
    "passcode": ("alfworld/passcode/4bi/pc_rawkey",),
    "selection": ("alfworld/selection/4bi/sa16/select=top3e/seed=3",),
    "selfwrong": ("alfworld/selfwrong/4bi/self_aot",),
    "main_30bi": ("alfworld/main/30bi/glm_ao",),
    "dapo_entry": ("alfworld/grpo/4bi/none/seed=1", "ENTRY=dapo", "OVERLONG_BUFFER_LEN=7000"),
    "ckpt_rescore": ("alfworld/main/4bi/glm_ao", "EVAL_CKPT_STEPS=25+50"),
    "rescore_4bt": ("alfworld/main/4bt/qwen_ao", *_rescore_args("alfworld/main/4bt/qwen_ao")),
    "rescore_30bi": ("alfworld/main/30bi/glm_aot", *_rescore_args("alfworld/main/30bi/glm_aot")),
    "smoke": ("alfworld/main/4bi/glm_ao", "--overlay", "smoke_4gpu"),
}


def launcher_env(tmp: Path) -> Dict[str, str]:
    """A clean environment: no ambient OPSD_* / recipe keys (the launcher is hermetic), no tracker key."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp / "home"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "TMPDIR": str(tmp),
        "OPSD_PYTHON": sys.executable,
        "PYTHONDONTWRITEBYTECODE": "1",  # the launcher's python steps write no bytecode into the tree
        "OPSD_DATA_ROOT": str(tmp / "data"),
        "RUN_ROOT": str(tmp / "runs"),
        "DRY_RUN": "1",
    }


def run_dry(tmp: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(RUN_SH), *args], env=launcher_env(tmp), cwd=str(tmp), capture_output=True, text=True, timeout=300
    )


def parse_dry(stdout: str) -> List[Tuple[str, List[str]]]:
    """[(module, argv), ...] -- one block per trainer invocation -- from opsd_dry_run_print's output."""
    blocks, lines, i = [], stdout.splitlines(), 0
    while i < len(lines):
        if lines[i].startswith("DRY_RUN: "):
            module = lines[i].split(" -m ", 1)[1].rstrip(" \\").strip()
            argv, i = [], i + 1
            while i < len(lines) and lines[i].strip():
                argv += shlex.split(lines[i].rstrip().removesuffix(" \\"))
                i += 1
            blocks.append((module, argv))
        i += 1
    return blocks


def parse_env(stdout: str) -> Dict[str, str]:
    """The resolved environment printed after the first argv block."""
    start = stdout.index("# resolved environment")
    env = {}
    for line in stdout[start:].splitlines()[1:]:
        if line.startswith("DRY_RUN: "):
            break
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", line):
            continue
        k, _, v = line.partition("=")
        parts = shlex.split(v) if v else [""]
        env[k] = parts[0] if parts else ""
    return env


@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("alfworld_dry")
    out = {}
    for name, args in ARMS.items():
        proc = run_dry(tmp, *args)
        assert proc.returncode == 0, f"{name}: rc={proc.returncode}\n{proc.stderr[-3000:]}"
        out[name] = (parse_dry(proc.stdout), parse_env(proc.stdout), proc.stderr)
    assert not (tmp / "runs").exists(), "DRY_RUN created the run directory"
    return out


@lru_cache(maxsize=None)
def _entry(module: str):
    return importlib.import_module(module)  # registers the entry's advantage estimators (e.g. no_op)


def parse_cfg(module: str, argv: List[str]):
    from skyrl.train.utils import validate_cfg

    cfg = _entry(module).CONFIG_CLS.from_cli_overrides(argv)
    validate_cfg(cfg)
    return cfg


def _b(v: str) -> bool:
    return v.strip().lower() == "true"


@pytest.mark.parametrize("name", list(ARMS))
def test_dry_run_argv_parses_and_matches_the_recipe(dry, name):
    blocks, env, _ = dry[name]
    assert blocks, f"{name}: no DRY_RUN block"
    for module, argv in blocks:
        assert module == (DAPO if env["ENTRY"] == "dapo" else OPSD)
        cfg = parse_cfg(module, argv)
        t, g, a = cfg.trainer, cfg.generator, cfg.environment.skyrl_gym.alfworld
        assert t.policy.model.path == env["MODEL_PATH"]
        assert t.strategy == {"fsdp2": "fsdp"}.get(env["STRATEGY"], env["STRATEGY"])  # validate_cfg's rename
        assert t.placement.policy_num_nodes == int(env["NUM_NODES"])
        assert g.inference_engine.num_engines * g.inference_engine.tensor_parallel_size == (
            int(env["NUM_GPUS_PER_NODE"]) * int(env["NUM_NODES"])
        )
        assert t.algorithm.loss_reduction == env["LOSS_REDUCTION"]
        assert t.algorithm.eps_clip_high == float(env["EPS_CLIP_HIGH"])
        assert t.policy.optimizer_config.lr == float(env["LR"])
        assert t.seed == int(env["TRAINER_SEED"])
        assert g.max_input_length == int(env["MAX_INPUT_LENGTH"])
        assert g.sampling_params.top_k == int(env["TOP_K"])
        stop = ast.literal_eval(env["STOP_TOKENS"])
        assert list(g.sampling_params.stop or []) == stop and list(g.eval_sampling_params.stop or []) == stop
        # the pre-quoted label must reach the env as a STRING (unquoted it parses as a YAML mapping)
        assert a.reasoning_prefix == "THOUGHTS:"
        assert a.prompt_style == "react" and a.show_admissible is False
        assert a.invalid_action_feedback_guide is _b(env["INVALID_ACTION_FEEDBACK_GUIDE"])
        assert a.reward_scale == float(env["ALFWORLD_REWARD_SCALE"])
        assert a.passcode_mode is _b(env["ALFWORLD_PASSCODE"])
        assert a.hint_placement == env["ALFWORLD_HINT_PLACEMENT"]
        assert len(a.server_urls) == int(env["NUM_ALFWORLD_SERVERS"])
        if module == OPSD:
            assert t.algorithm.advantage_estimator == env["ADVANTAGE_ESTIMATOR"]
            assert t.algorithm.policy_loss_type == env["POLICY_LOSS_TYPE"]
            assert t.algorithm.use_kl_loss is _b(env["USE_KL_LOSS"])
            assert t.ref.model.path == env["MODEL_PATH"]
            assert a.enable_thinking is _b(env["ENABLE_THINKING"])
        # the env-worker deadlock guard: concurrent eval trajectories never exceed the env workers
        assert t.eval_batch_size * g.eval_n_samples_per_prompt <= cfg.environment.skyrl_gym.max_env_workers


def _cfg(dry, name, block=0):
    module, argv = dry[name][0][block]
    return parse_cfg(module, argv), argv, dry[name][1]


def test_mode_specific_settings(dry):
    cfg, argv, env = _cfg(dry, "step0_30bi")
    assert cfg.trainer.epochs == 0 and cfg.trainer.eval_before_train is True and cfg.trainer.ckpt_interval == -1
    assert cfg.trainer.policy.megatron_config.expert_model_parallel_size == 8
    assert cfg.trainer.ref.megatron_config.tensor_model_parallel_size == 4
    assert len(cfg.data.val_data) == 2 and cfg.generator.eval_n_samples_per_prompt == 4

    cfg, argv, env = _cfg(dry, "main_4bt_guide")
    assert cfg.environment.skyrl_gym.alfworld.invalid_action_feedback_guide is True
    assert cfg.environment.skyrl_gym.alfworld.think_tags_in_prompt is False  # as these arms TRAINED
    assert cfg.environment.skyrl_gym.alfworld.strict_action_outside_think is True
    assert cfg.trainer.ckpt_interval == 10 and cfg.data.val_data[0].endswith("/data/dev.parquet")

    cfg, argv, env = _cfg(dry, "opsd_grpo")
    assert cfg.trainer.eval_before_train is False and cfg.trainer.algorithm.use_kl_in_reward is False
    assert cfg.trainer.resume_mode is None and cfg.trainer.epochs == 2

    cfg, argv, env = _cfg(dry, "passcode_utility")
    assert cfg.trainer.epochs == 0
    names = [Path(p).name for p in cfg.data.val_data]
    assert names == [f"eval_hint_{s}.parquet" for s in ("nohint", "pc_key", "pc_rawkey", "pc_rawwrong", "pc_wrongraw")]
    assert cfg.environment.skyrl_gym.alfworld.hint_placement == "system"

    cfg, argv, env = _cfg(dry, "selection")
    assert env["OPSD_HINTS_PATH"].endswith("/alfworld/selection/hints_self16_s2/hints_sa16_top3e.json")
    assert cfg.trainer.seed == 44 and cfg.trainer.epochs == 4 and cfg.trainer.eval_interval == 20

    cfg, argv, env = _cfg(dry, "dapo_entry")
    assert cfg.trainer.algorithm.overlong_buffer.len == 7000

    blocks, env = dry["ckpt_rescore"][0], dry["ckpt_rescore"][1]
    assert len(blocks) == 2
    train_dir = Path(env["RUN_ROOT"]) / "alfworld__main__4bi__glm_ao"
    for (module, argv), step in zip(blocks, (25, 50)):
        cfg = parse_cfg(module, argv)
        assert cfg.trainer.resume_mode == "from_path"
        assert cfg.trainer.resume_path == str(train_dir / "ckpts" / f"global_step_{step}")
        assert cfg.trainer.epochs == 0 and cfg.generator.eval_n_samples_per_prompt == 4
        assert [Path(p).name for p in cfg.data.val_data] == ["validation.parquet", "validation_unseen.parquet"]
        _assert_own_rescore_dir(cfg, env, train_dir)

    cfg, argv, env = _cfg(dry, "smoke")
    assert cfg.trainer.placement.policy_num_gpus_per_node == 4 and cfg.generator.max_turns == 6
    assert cfg.data.train_data[0].endswith("/data/train.parquet")


def _assert_own_rescore_dir(cfg, env, train_dir: Path, suffix: str = "__ckpteval") -> None:
    """A re-score runs in <training RUN_DIR><suffix> and writes nothing under the training run."""
    own = Path(str(train_dir) + suffix)
    assert env["RUN_DIR"] == str(own) and env["RUN_NAME"] == own.name
    assert env["EVAL_CKPT_DIR"] == str(train_dir / "ckpts")
    assert cfg.trainer.run_name == own.name
    for p in (cfg.trainer.ckpt_path, cfg.trainer.export_path):
        assert Path(p).parent == own, p
    assert cfg.trainer.ckpt_interval == -1


def test_rescore_of_record_4bt_scores_under_the_recorded_prompt(dry):
    """The six 4bt trajectory cells: trained WITH the guide and WITHOUT the think tags in the prompt, scored by
    a re-score WITHOUT the guide and WITH the tags named. The recipe's `rescore:` must reproduce the latter."""
    blocks, env, _ = dry["rescore_4bt"]
    assert (env["INVALID_ACTION_FEEDBACK_GUIDE"], env["ALFWORLD_THINK_TAGS_IN_PROMPT"]) == ("false", "true")
    train_dir = Path(env["RUN_ROOT"]) / "alfworld__main__4bt__qwen_ao"
    assert len(blocks) == 4
    for (module, argv), step in zip(blocks, (20, 50, 80, 100)):  # these three checkpointed every 10 steps
        cfg = parse_cfg(module, argv)
        a = cfg.environment.skyrl_gym.alfworld
        assert a.think_tags_in_prompt is True and a.invalid_action_feedback_guide is False
        assert a.strict_action_outside_think is True and a.reasoning_style == "tag" and a.enable_thinking is True
        assert list(cfg.generator.eval_sampling_params.stop or []) == []
        assert (cfg.generator.eval_sampling_params.temperature, cfg.generator.eval_sampling_params.top_p) == (0.6, 0.95)
        assert cfg.generator.eval_n_samples_per_prompt == 4 and cfg.trainer.epochs == 0
        assert cfg.trainer.resume_path == str(train_dir / "ckpts" / f"global_step_{step}")
        assert [Path(p).name for p in cfg.data.val_data] == ["validation.parquet", "validation_unseen.parquet"]
        _assert_own_rescore_dir(cfg, env, train_dir)


def test_rescore_of_record_instruct_stops_at_the_first_action_close(dry):
    blocks, env, _ = dry["rescore_30bi"]
    assert len(blocks) == 4
    for module, argv in blocks:
        cfg = parse_cfg(module, argv)
        assert list(cfg.generator.eval_sampling_params.stop) == ["</action>"]
        assert cfg.environment.skyrl_gym.alfworld.think_tags_in_prompt is True  # the Instruct contract, as trained
        assert cfg.environment.skyrl_gym.alfworld.invalid_action_feedback_guide is False
        assert cfg.trainer.strategy == "megatron" and cfg.trainer.placement.policy_num_nodes == 2


def test_rescore_naming_follows_the_training_run(tmp_path):
    """RUN_NAME names the TRAINING run; the re-score derives its own name, directory and checkpoint source."""
    proc = run_dry(tmp_path, "alfworld/main/4bi/glm_ao", "RUN_NAME=mytrain", "EVAL_CKPT_STEPS=25", "EVAL_CKPT_TAG=v2")
    assert proc.returncode == 0, proc.stderr[-2000:]
    ((module, argv),) = parse_dry(proc.stdout)
    cfg, env = parse_cfg(module, argv), parse_env(proc.stdout)
    train_dir = tmp_path / "runs" / "mytrain"
    assert cfg.trainer.resume_path == str(train_dir / "ckpts" / "global_step_25")
    _assert_own_rescore_dir(cfg, env, train_dir, suffix="__ckpteval_v2")

    proc = run_dry(tmp_path, "alfworld/main/4bi/glm_ao", "EVAL_CKPT_STEPS=25", "EVAL_CKPT_DIR=/elsewhere/ckpts")
    assert proc.returncode == 0, proc.stderr[-2000:]
    cfg = parse_cfg(*parse_dry(proc.stdout)[0])
    assert cfg.trainer.resume_path == "/elsewhere/ckpts/global_step_25"
    assert cfg.trainer.run_name == "alfworld__main__4bi__glm_ao__ckpteval"

    own = tmp_path / "runs" / "alfworld__main__4bi__glm_ao__ckpteval" / "ckpts"
    proc = run_dry(tmp_path, "alfworld/main/4bi/glm_ao", "EVAL_CKPT_STEPS=25", f"EVAL_CKPT_DIR={own}")
    assert proc.returncode != 0 and "own checkpoint directory" in proc.stderr
    assert not (tmp_path / "runs").exists()


def test_relative_locations_are_relative_to_the_callers_directory(tmp_path):
    """RUN_ROOT, OPSD_DATA_ROOT and KEY=VAL input paths given relative mean "relative to where you launched"
    (the resolution runs before the launcher changes to the repository root)."""
    env = launcher_env(tmp_path)
    env.update(RUN_ROOT="rel_runs", OPSD_DATA_ROOT="rel_data")
    proc = subprocess.run(
        ["bash", str(RUN_SH), "alfworld/main/4bi/glm_ao", "EVAL_CKPT_STEPS=25", "TASK_IDS_PATH=ids.json"],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    res = parse_env(proc.stdout)
    assert res["RUN_DIR"] == str(tmp_path / "rel_runs" / "alfworld__main__4bi__glm_ao__ckpteval")
    assert res["EVAL_CKPT_DIR"] == str(tmp_path / "rel_runs" / "alfworld__main__4bi__glm_ao" / "ckpts")
    assert res["OPSD_HINTS_PATH"].startswith(str(tmp_path / "rel_data") + "/")
    assert res["TASK_IDS_PATH"] == str(tmp_path / "ids.json")


def test_null_xdomain_reads_its_hint_from_the_decrypted_xdnull_bundle(tmp_path):
    """Its hint file ships encrypted in appworld/appworld_opsd_xdnull.bundle: the dry run reads the decrypted copy
    (content-pinned) without a warning, and a real run without it refuses (before the host checks, and also with
    OPSD_PREFLIGHT=0) with how to fetch and decrypt that bundle rather than a generic missing-file error."""
    proc = run_dry(tmp_path, "alfworld/null/4bi/null_xdomain")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "not in the public data bundle" not in proc.stderr and "OPSD_HINTS_NOT_BUNDLED" not in proc.stdout
    env = parse_env(proc.stdout)
    assert env["OPSD_HINTS_PATH"] == str(
        tmp_path / "data/appworld/plain/opsd_appworld_xdnull/alfworld/glt_xdmix_tau2_appworld.json"
    )
    assert env["OPSD_HINTS_CONTENT_SHA256"] == "e6a41b2d732fc4b1d0f5f04525b08a4536d7f6241f6268403445a01e4b409f3e"
    for extra in ({}, {"OPSD_PREFLIGHT": "0"}):
        env = launcher_env(tmp_path)
        del env["DRY_RUN"]
        env.update(extra)
        proc = subprocess.run(
            ["bash", str(RUN_SH), "alfworld/null/4bi/null_xdomain"],
            env=env,
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode != 0
        assert "hint file not found" in proc.stderr and "static hint" in proc.stderr, proc.stderr
        assert "--include appworld/appworld_opsd_xdnull.bundle" in proc.stderr, proc.stderr
        assert "setup_appworld.sh" in proc.stderr, proc.stderr
        assert "preflight" not in proc.stderr.lower()


def test_a_missing_hint_file_is_refused_even_without_preflight(tmp_path):
    """The trainer silently substitutes its generic static hint for a missing hint file (a different PI)."""
    env = launcher_env(tmp_path)
    del env["DRY_RUN"]
    env["OPSD_PREFLIGHT"] = "0"
    proc = subprocess.run(
        ["bash", str(RUN_SH), "alfworld/main/4bi/glm_ao"],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode != 0 and "hint file not found" in proc.stderr and "static hint" in proc.stderr


def test_planned_arms_are_refused(tmp_path):
    """No shipped ALFWorld arm is planned any more (the f8 selection runs finished), so mark one planned in a copy
    of the recipes tree: the launcher refuses it, and passes --allow-planned through to the resolver."""
    import shutil

    recipes = tmp_path / "recipes"
    for sub in ("alfworld", "common"):
        shutil.copytree(_repo.recipes_dir() / sub, recipes / sub, ignore=shutil.ignore_patterns("tests", "__pycache__"))
    arms = recipes / "alfworld" / "arms.yaml"
    text = arms.read_text()
    block = "    pis: [f8]\n    objective: opsd\n    extra:\n      select: [rand8]\n"
    assert text.count(block) == 1
    arms.write_text(text.replace(block, block + "    status: planned\n"))
    env = {**launcher_env(tmp_path), "OPSD_RECIPES_DIR": str(recipes)}
    arm = "alfworld/selection/4bi/f8/select=rand8/seed=1"
    proc = subprocess.run(
        ["bash", str(RUN_SH), arm], env=env, cwd=str(tmp_path), capture_output=True, text=True, timeout=300
    )
    assert proc.returncode != 0 and "planned" in proc.stderr
    proc = subprocess.run(
        ["bash", str(RUN_SH), arm, "--allow-planned"],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    # with the shipped recipes the same arm is done and runs without the flag
    proc = run_dry(tmp_path, arm)
    assert proc.returncode == 0 and "planned" not in proc.stderr, proc.stderr[-2000:]


def test_hermetic_ambient_recipe_keys_are_rejected(tmp_path):
    env = launcher_env(tmp_path)
    env["LR"] = "3e-7"  # a recipe key in the ambient environment must come in as KEY=VAL instead
    proc = subprocess.run(
        ["bash", str(RUN_SH), "alfworld/main/4bi/glm_ao"],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode != 0 and "LR" in proc.stderr


# ------------------------------------------------------------------- the real path, up to the server pool
def _toy_parquets(pq_dir: Path) -> None:
    import pandas as pd

    pq_dir.mkdir(parents=True)

    def rows(ids, source):
        return pd.DataFrame(
            {
                "data_source": [source] * len(ids),
                "prompt": [[{"role": "user", "content": f"task {i}"}] for i in ids],
                "env_class": ["alfworld"] * len(ids),
                "extra_info": [{"task_id": i, "game_file": f"/games/{i}/game.tw-pddl"} for i in ids],
            }
        )

    rows(["t1", "t2", "t3", "t4"], "alfworld").to_parquet(pq_dir / "train.parquet", index=False)
    rows(["s1", "s2"], "alfworld").to_parquet(pq_dir / "validation.parquet", index=False)
    rows(["u1"], "alfworld_valid_unseen").to_parquet(pq_dir / "validation_unseen.parquet", index=False)


def _toy_inputs(tmp_path: Path) -> Dict[str, str]:
    """Task-id splits, the glm_ao hint file and toy parquets; returns the real-run launcher environment."""
    import json

    data = tmp_path / "data" / "alfworld"
    (data / "splits").mkdir(parents=True)
    (data / "splits" / "task_ids_train1600.json").write_text(json.dumps(["t1", "t2"]))
    (data / "splits" / "task_ids_dev200.json").write_text(json.dumps(["t3"]))
    (data / "hints" / "glm-5.2").mkdir(parents=True)
    (data / "hints" / "glm-5.2" / "hints_traj_nothink_cap24k.json").write_text(json.dumps({"t1": "go"}))
    _toy_parquets(tmp_path / "pq")
    env = launcher_env(tmp_path)
    del env["DRY_RUN"]
    env.update(
        OPSD_PREFLIGHT="0",
        ALFWORLD_VENV=str(tmp_path / "no-venv"),
        ALFWORLD_DATA=str(tmp_path / "game-data"),
        ALFWORLD_PARQUET_DIR=str(tmp_path / "pq"),
    )
    return env


def test_real_run_prepares_data_then_stops_without_a_server_venv(tmp_path):
    """Everything before the server pool runs for real on CPU (recipe, provenance, per-run parquets with the
    dev split and its leakage check); the pool then refuses to start from a missing venv, so no process is
    ever spawned. Preflight is skipped (OPSD_PREFLIGHT=0) because this host's GPUs are not the point."""
    import pandas as pd

    env = _toy_inputs(tmp_path)
    run_dir = tmp_path / "runs" / "alfworld__main__4bi__glm_ao"
    (run_dir / "ckpts" / "global_step_25").mkdir(parents=True)  # an earlier launch of this RESUME_MODE=null arm
    proc = subprocess.run(
        ["bash", str(RUN_SH), "alfworld/main/4bi/glm_ao"],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode != 0
    assert "no ALFWorld venv python" in proc.stderr, proc.stderr[-2000:]
    assert "starts again from step 0" in proc.stderr and "RESUME_MODE=latest" in proc.stderr
    assert (run_dir / "resolved.env").is_file() and (run_dir / "provenance.json").is_file()
    train = pd.read_parquet(run_dir / "data" / "train.parquet")
    dev = pd.read_parquet(run_dir / "data" / "dev.parquet")
    assert sorted(train["extra_info"].map(lambda e: e["task_id"])) == ["t1", "t2"]
    assert list(dev["extra_info"].map(lambda e: e["task_id"])) == ["t3"]
    assert set(dev["data_source"]) == {"alfworld_dev"}
    assert not list((run_dir / "pids").glob("*.pid")) if (run_dir / "pids").exists() else True


def test_real_rescore_never_writes_into_the_training_run(tmp_path):
    """A re-score (EVAL_CKPT_STEPS) records its own resolved.env / provenance.json next to, not over, the
    training run's: the training record keeps describing what trained."""
    env = _toy_inputs(tmp_path)
    train_dir = tmp_path / "runs" / "alfworld__main__4bt__glm_a"
    (train_dir / "ckpts" / "global_step_25" / "policy").mkdir(parents=True)
    for name in ("resolved.env", "provenance.json"):
        (train_dir / name).write_text(f"training record: {name}\n")
    meta = R.load_bench("alfworld").arms["alfworld/main/4bt/glm_a"].entry.meta["rescore"]
    args = [f"{k}={v}" for k, v in meta.items()]
    proc = subprocess.run(
        ["bash", str(RUN_SH), "alfworld/main/4bt/glm_a", *args],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode != 0 and "no ALFWorld venv python" in proc.stderr, proc.stderr[-2000:]
    for name in ("resolved.env", "provenance.json"):
        assert (train_dir / name).read_text() == f"training record: {name}\n"
    own = tmp_path / "runs" / "alfworld__main__4bt__glm_a__ckpteval"
    assert (own / "provenance.json").is_file()
    resolved = R.read_resolved_env(own / "resolved.env")
    assert resolved["RUN_DIR"] == str(own) and resolved["EVAL_CKPT_DIR"] == str(train_dir / "ckpts")
    assert (resolved["INVALID_ACTION_FEEDBACK_GUIDE"], resolved["ALFWORLD_THINK_TAGS_IN_PROMPT"]) == ("false", "true")
    assert sorted(p.name for p in train_dir.iterdir()) == ["ckpts", "provenance.json", "resolved.env"]


def test_prepare_data_refuses_dev_train_leakage(tmp_path):
    pq_dir = tmp_path / "pq"
    _toy_parquets(pq_dir)
    (tmp_path / "train.json").write_text('["t1", "t2"]')
    (tmp_path / "dev.json").write_text('["t2"]')
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/alfworld/prepare_data.py"),
            "--parquet-dir",
            str(pq_dir),
            "--out-dir",
            str(tmp_path / "out"),
            "--task-ids",
            str(tmp_path / "train.json"),
            "--dev",
            "--dev-task-ids",
            str(tmp_path / "dev.json"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode != 0 and "LEAKAGE" in proc.stderr


def test_prepare_data_smoke_truncation(tmp_path):
    import json

    pq_dir = tmp_path / "pq"
    _toy_parquets(pq_dir)
    (tmp_path / "train.json").write_text('["t4", "t1", "t2"]')
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/alfworld/prepare_data.py"),
            "--parquet-dir",
            str(pq_dir),
            "--out-dir",
            str(tmp_path / "out"),
            "--task-ids",
            str(tmp_path / "train.json"),
            "--max-train-tasks",
            "2",
            "--max-eval-tasks",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    written = json.loads(proc.stdout.strip().splitlines()[-1])
    assert set(written) == {"train", "validation", "validation_unseen"}
    import pandas as pd

    assert sorted(pd.read_parquet(written["train"])["extra_info"].map(lambda e: e["task_id"])) == ["t1", "t4"]
    assert len(pd.read_parquet(written["validation"])) == 1

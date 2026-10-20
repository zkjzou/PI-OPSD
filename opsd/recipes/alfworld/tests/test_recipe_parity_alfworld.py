"""Golden parity for the ALFWorld recipes: every launch configuration of record resolves from the recipes.

``golden/*.env`` holds one sanitized launch configuration per ALFWorld arm that was run (134 arms): the
environment the launch wrapper handed to the run-file, with object-store URIs mapped to
``${OPSD_DATA_ROOT}`` paths and run-name / tracking-project / host-path keys removed. For every one of them,
``resolve(<arm id>)`` must contain every golden key with an equal value, except

* ``INFRA_KEYS`` -- machine-local settings the launcher defaults itself (ports);
* ``AS_RAN`` -- the few keys where the recorded launch configuration is NOT what the run actually used. Each
  entry states the evidence (the run's logged configuration or its rollout dumps); the recipe encodes the
  run-time value, and ``test_as_ran_entries_are_real_differences`` keeps the list from going stale.

Golden files are named after the arm id in the launch records; ``public_arm_id`` maps the handful whose
public id (the recipes) differs.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict

import pytest

from opsd import _repo
from opsd.recipes import resolve as R

HERE = Path(__file__).resolve().parent
GOLDEN_DIR = HERE / "golden"
RECIPES = _repo.recipes_dir()
BENCH = "alfworld"
DATA = "/opsd-data"
ENVIRON = {"OPSD_DATA_ROOT": DATA, "RUN_ROOT": "/runs", "HOME": "/nonexistent"}

# Machine-local: the launcher defaults the server base port (9100) and it never affects results.
INFRA_KEYS = {"ALFWORLD_BASE_PORT"}

_TRAJ_PIS = ("glm_a", "glm_ao", "glm_aot", "qwen_a", "qwen_ao", "qwen_aot")
_LAMBDA_GRID = [
    f"alfworld/opsd_grpo/4bi/{pi}/lambda={lam}"
    for pi in ("glm_ao", "glm_skill", "glm_chk")
    for lam in ("0.1", "0.5", "0.9", "0.99", "0.999")
]


def _as_ran() -> Dict[str, Dict[str, tuple]]:
    out: Dict[str, Dict[str, tuple]] = {}

    def put(arm, key, value, why):
        out.setdefault(arm, {})[key] = (value, why)

    for student in ("4bi", "4bt"):
        for pi in _TRAJ_PIS:
            put(
                f"alfworld/main/{student}/{pi}",
                "KL_LOSS_COEF",
                "0.05",
                "the launch wrapper of that date did not forward KL_LOSS_COEF, so the run-file default applied "
                "(logged config); inert because USE_KL_LOSS=false",
            )
    for pi in _TRAJ_PIS:
        put(
            f"alfworld/main/4bt/{pi}",
            "INVALID_ACTION_FEEDBACK_GUIDE",
            "true",
            "launched before the guide became optional: the logged env config has no such field and every "
            "invalid-action observation in the training rollouts carries the full action guide",
        )
    for pi in ("qwen_a", "qwen_ao", "qwen_aot"):
        put(f"alfworld/main/4bt/{pi}", "CKPT_INTERVAL", "10", "checkpoints exist every 10 steps")
    for arm in _LAMBDA_GRID:
        put(arm, "EVAL_N_SAMPLES", "1", "logged config: the grid validated at N=1")
        put(arm, "RESUME_MODE", "null", "logged config: no autoresume")
        if arm.endswith("lambda=0.999"):
            put(arm, "EVAL_BEFORE_TRAIN", "false", "logged config: no step-0 evaluation")
    return out


AS_RAN = _as_ran()


def public_arm_id(golden_id: str) -> str:
    """Arm id in the launch records -> public arm id (the recipes)."""
    a = golden_id
    m = re.fullmatch(r"(alfworld/selection/4bi/\w+)/sel=(\w+)/seed=(\d+)", a)
    if m:  # selection: `select=` and 1-based replicate index
        return f"{m.group(1)}/select={m.group(2)}/seed={int(m.group(3)) + 1}"
    m = re.fullmatch(r"(alfworld/selection/4bi/mix_\w+)/seed=(\d+)", a)
    if m:
        return f"{m.group(1)}/seed={int(m.group(2)) + 1}"
    m = re.fullmatch(r"alfworld/beta/4bi/glm_ao/(beta=[\w.+-]+)/reward_scale=1\.0", a)
    if m:  # every beta arm ran at reward scale 1.0; the public id leaves it implicit
        return f"alfworld/beta/4bi/glm_ao/{m.group(1)}"
    if a == "alfworld/opsd_grpo/4bi/glm_ao/lambda=0.5/reward_scale=1.0":
        return "alfworld/beta/4bi/glm_ao/lambda=0.5"  # the lambda comparator of the beta sweep
    m = re.fullmatch(r"(alfworld/rlsd/4bi/\w+)/decay=50", a)
    if m:  # decay 50 is RLSD's default schedule
        return m.group(1)
    if a == "alfworld/selfwrong/4bi/self_wrong_raw":
        return "alfworld/selfwrong/4bi/self_aot"
    return a


def _golden_files():
    return sorted(GOLDEN_DIR.glob("*.env"))


def _golden_id(path: Path) -> str:
    first = path.read_text().splitlines()[0]
    m = re.fullmatch(r"# Launch configuration of (\S+)", first)
    assert m, f"{path.name}: first line must name the arm"
    return m.group(1)


def _golden_env(path: Path) -> Dict[str, str]:
    return {k: v.replace("${OPSD_DATA_ROOT}", DATA) for k, (v, _) in R.parse_env_file(path).items()}


@pytest.fixture(scope="module")
def bench():
    return R.load_bench(BENCH, RECIPES)


# ------------------------------------------------------------------------------------------- fixtures
def test_fixtures_are_complete_and_sanitized():
    files = _golden_files()
    assert len(files) == 134, f"expected the 134 launch configurations of record, found {len(files)}"
    for p in files:
        gid = _golden_id(p)
        assert p.name == gid.replace("/", "__") + ".env", p.name
        text = p.read_text()
        for bad in ("://", "/mnt/", "/home/", "_S3="):  # no URIs of any scheme, no host paths
            assert bad not in text, f"{p.name} is not sanitized ({bad!r})"
        assert not re.search(r"^(RUN_NAME|PROJECT_NAME|WANDB_\w*|OPSD_EMA_DIR)=", text, re.M), p.name


def test_golden_table_sizes():
    counts = Counter(_golden_id(p).split("/")[1] for p in _golden_files())
    assert counts == {
        "main": 48,
        "selection": 39,
        "opsd_grpo": 19,
        "rlsd": 7,
        "passcode": 6,
        "teacher": 5,
        "beta": 3,
        "grpo": 3,
        "null": 2,
        "lamsched": 1,
        "selfwrong": 1,
    }


def test_recipes_generate_exactly_the_goldens_plus_step0(bench):
    from_goldens = {public_arm_id(_golden_id(p)) for p in _golden_files()}
    step0 = {f"alfworld/step0/{s}/none" for s in ("4bi", "4bt", "30bi")}
    assert len(from_goldens) == 134, "two launch records map to the same public arm id"
    assert set(bench.arms) == from_goldens | step0
    assert not bench.skipped, bench.skipped


# --------------------------------------------------------------------------------------------- parity
@pytest.mark.parametrize("path", _golden_files(), ids=lambda p: p.stem)
def test_golden_parity(path):
    gid = _golden_id(path)
    golden = _golden_env(path)
    for key, (value, _why) in AS_RAN.get(gid, {}).items():
        golden[key] = value
    env = R.resolve(public_arm_id(gid), environ=ENVIRON, allow_planned=True, recipes_dir=RECIPES).env
    diffs = {
        k: (v, env.get(k, "<missing>")) for k, v in sorted(golden.items()) if k not in INFRA_KEYS and env.get(k) != v
    }
    assert not diffs, f"{gid}: golden != recipe for {diffs}"


def test_as_ran_entries_are_real_differences():
    """Every AS_RAN override must name an existing golden key whose recorded value it changes."""
    by_id = {_golden_id(p): p for p in _golden_files()}
    for gid, keys in AS_RAN.items():
        assert gid in by_id, f"AS_RAN names an unknown arm {gid}"
        golden = _golden_env(by_id[gid])
        for key, (value, why) in keys.items():
            assert key in golden, f"{gid}: {key} is not in the golden"
            assert golden[key] != value, f"{gid}: {key}={value} already matches the golden; drop the entry"
            assert why


# ------------------------------------------------------------------------------------- recipe content
_SELECTION_POOLS = {
    "sw8": ("rand", "top3d", "top3e"),
    "sa16": ("rand", "top3d", "top3e"),
    "f8": ("rand8", "top3d", "top3e"),
}


def test_statuses(bench):
    """Every per-task-selection run finished (all 39 runs complete); the only arms still not done
    are the passcode trainer-based utility evaluations."""
    status = {a: arm.status for a, arm in bench.arms.items()}
    assert sorted(a for a, s in status.items() if s == "planned") == []
    assert sorted(a for a, s in status.items() if s == "in_progress") == sorted(
        f"alfworld/passcode/4bi/none/eval=utility/placement={p}" for p in ("user", "system")
    )
    selection = sorted(a for a in bench.arms if a.split("/")[1] == "selection")
    assert selection == sorted(
        [f"alfworld/selection/4bi/{pi}/seed={n}" for pi in ("mix_rand8", "mix_top3d", "mix_top3e") for n in range(1, 5)]
        + [
            f"alfworld/selection/4bi/{pool}/select={s}/seed={n}"
            for pool, rules in _SELECTION_POOLS.items()
            for s in rules
            for n in (1, 2, 3)
        ]
    )
    # the f8 arms resolve without --allow-planned now
    env = R.resolve("alfworld/selection/4bi/f8/select=rand8/seed=1", environ=ENVIRON, recipes_dir=RECIPES).env
    assert env["OPSD_HINTS_PATH"] == f"{DATA}/alfworld/selection/hints_flash8/hints_mix_random8.json"


def test_the_main_table_has_all_16_pis_for_every_student(bench):
    pis16 = [
        "glm_a",
        "glm_aat",
        "glm_ao",
        "glm_aot",
        "qwen_a",
        "qwen_aat",
        "qwen_ao",
        "qwen_aot",
        "glm_skill",
        "qwen_skill",
        "glm_summ",
        "qwen_summ",
        "glm_chk",
        "qwen_chk",
        "self_skill",
        "self_summ",
    ]
    assert set(pis16) <= set(bench.pis)
    main = {a: arm for a, arm in bench.arms.items() if a.split("/")[1] == "main"}
    assert sorted(main) == sorted(f"alfworld/main/{s}/{pi}" for s in ("4bi", "4bt", "30bi") for pi in pis16)
    assert all(arm.status == "done" for arm in main.values())
    # The two Action+Thoughts PIs are not rows of the paper's 14-row table; their numbers are the training runs'.
    assert {a for a, arm in main.items() if arm.entry.paper is None} == {
        f"alfworld/main/{s}/{pi}" for s in ("4bi", "4bt", "30bi") for pi in ("glm_aat", "qwen_aat")
    }
    for s in ("4bi", "30bi"):
        assert (
            R.resolve(f"alfworld/main/{s}/qwen_aat", environ=ENVIRON, recipes_dir=RECIPES)
            .env["OPSD_HINTS_PATH"]
            .endswith("/qwen3.7-plus/hints_traj_actthoughts_cap24k.json")
        )
    assert (
        R.resolve("alfworld/main/4bt/glm_aat", environ=ENVIRON, recipes_dir=RECIPES)
        .env["OPSD_HINTS_PATH"]
        .endswith("/glm-5.2/hints_traj_actthink_cap24k.json")
    )


def test_every_bundled_hint_is_content_pinned(bench):
    """Preflight can only verify a hint file whose recipe carries its content hash: every arm that reads a hint
    carries one."""
    unpinned = []
    for a in bench.arms:
        env = R.resolve(a, environ=ENVIRON, allow_planned=True, recipes_dir=RECIPES).env
        if env.get("OPSD_HINTS_PATH") and not env.get("OPSD_HINTS_CONTENT_SHA256"):
            unpinned.append(a)
    assert unpinned == []
    # replicate n of a self pool reads draw n-1 of that pool and rule, each with its own hash
    shas = set()
    for pi in ("sw8", "sa16"):
        for sel in ("rand", "top3d", "top3e"):
            for n, d in ((1, "hints_self16"), (2, "hints_self16_s1"), (3, "hints_self16_s2")):
                env = R.resolve(
                    f"alfworld/selection/4bi/{pi}/select={sel}/seed={n}", environ=ENVIRON, recipes_dir=RECIPES
                ).env
                assert env["OPSD_HINTS_PATH"] == f"{DATA}/alfworld/selection/{d}/hints_{pi}_{sel}.json"
                shas.add(env["OPSD_HINTS_CONTENT_SHA256"])
    assert len(shas) == 18
    # f8 likewise: replicate n reads draw n-1 of the GLM-5.3-flash pool (hints_flash8{,_s1,_s2}), rule file names
    # as in the mix pool, each pinned by its content_sha256.
    f8 = {}
    for sel, fn in (("rand8", "random8"), ("top3d", "top3_d"), ("top3e", "top3_derr")):
        for n, d in ((1, "hints_flash8"), (2, "hints_flash8_s1"), (3, "hints_flash8_s2")):
            env = R.resolve(
                f"alfworld/selection/4bi/f8/select={sel}/seed={n}", environ=ENVIRON, recipes_dir=RECIPES
            ).env
            assert env["OPSD_HINTS_PATH"] == f"{DATA}/alfworld/selection/{d}/hints_mix_{fn}.json"
            assert env["TRAINER_SEED"] == str(41 + n)
            f8[(sel, n)] = env["OPSD_HINTS_CONTENT_SHA256"]
    assert f8 == F8_SHA
    assert not set(f8.values()) & shas


# content_sha256 of the f8 hint files (dataset revision 60993553, alfworld/selection/hints_flash8*/; verified with
# opsd.recipes.resolve.hint_content_sha256 on the files themselves).
F8_SHA = {
    ("rand8", 1): "7a47fdf9787b46e047e9aa2ae1120ac87949b7e80db1c1db6ea5388559a6f379",
    ("rand8", 2): "c4ca79168fd7b12d34777348fc824f419374b118080699acbff84424053b613f",
    ("rand8", 3): "d75e0f3f85ab7fe3c38e7a21f78e49e9f0ac7272143580b3887029c5ef56633b",
    ("top3d", 1): "05194712f9bd559d4c620efc99f15d29e4e49d52dc4581773334ab40236a5f6e",
    ("top3d", 2): "84244e9f6a3d18eb49b7bc2420371866dd9a93270598884e54cfc07d4e7770f5",
    ("top3d", 3): "6e0097c666f9d1777b348b66f312dc1b8358f102998ec1271b6f1c1fc8669e92",
    ("top3e", 1): "25f5ac69b8ae500e0d5cd47ee99cd1432907c8451cd2d89a549c081a84b17907",
    ("top3e", 2): "bc203f28b9261242c483018034f442bfcf25fc91d1125d7bca928743de291cdf",
    ("top3e", 3): "b3e109ebb0066e6c5b93b8ddfe22fb8e981c8eae5532a6fc952f045e6bccac09",
}


# The cross-domain null's hint file ships encrypted in the dataset's appworld/appworld_opsd_xdnull.bundle (batch 3b);
# scripts/setup/setup_appworld.sh decrypts it under appworld/plain/. content_sha256 as published with the bundle.
XDNULL_HINT = f"{DATA}/appworld/plain/opsd_appworld_xdnull/alfworld/glt_xdmix_tau2_appworld.json"
XDNULL_SHA = "e6a41b2d732fc4b1d0f5f04525b08a4536d7f6241f6268403445a01e4b409f3e"


def test_null_xdomain_reads_the_decrypted_xdnull_bundle(bench):
    """The arm's hint file is in the data set now (encrypted, evaluation-split material): the recipe points at its
    decrypted location and pins its content, and no arm carries the old not-bundled refusal any more."""
    env = R.resolve("alfworld/null/4bi/null_xdomain", environ=ENVIRON, recipes_dir=RECIPES).env
    assert env["OPSD_HINTS_PATH"] == XDNULL_HINT
    assert env["OPSD_HINTS_CONTENT_SHA256"] == XDNULL_SHA
    assert env["OPSD_REF_SEQ_CAP"] == "65536"
    for a in bench.arms:
        env = R.resolve(a, environ=ENVIRON, allow_planned=True, recipes_dir=RECIPES).env
        assert "OPSD_HINTS_NOT_BUNDLED" not in env, a
    # its AppWorld donor text is test_normal reference solutions: the recipe says so where users read it
    note = bench.arms["alfworld/null/4bi/null_xdomain"].entry.meta["note"]
    assert "appworld_opsd_xdnull.bundle" in note and "test_normal" in note and "EVALUATION-SPLIT" in note


# ------------------------------------------------------------------------- checkpoint re-scores of record
_THINK_GUIDE_RESCORE = {"INVALID_ACTION_FEEDBACK_GUIDE": "false", "ALFWORLD_THINK_TAGS_IN_PROMPT": "true"}


def _rescore_of_record() -> Dict[str, Dict[str, str]]:
    """The KEY=VAL arguments of the checkpoint re-score behind each re-scored arm's reported numbers."""
    out: Dict[str, Dict[str, str]] = {}
    for pi in _TRAJ_PIS:
        # Thinking student: the re-score ran WITHOUT the guide its training had, and WITH the think tags named
        # in the prompt (its training prompt did not name them). The qwen three checkpointed every 10 steps.
        steps = "20+50+80+100" if pi.startswith("qwen") else "25+50+75+100"
        out[f"alfworld/main/4bt/{pi}"] = {"EVAL_CKPT_STEPS": steps, **_THINK_GUIDE_RESCORE}
        # Instruct students: the re-score stopped evaluation turns at the first </action>
        for s in ("4bi", "30bi"):
            out[f"alfworld/main/{s}/{pi}"] = {"EVAL_CKPT_STEPS": "25+50+75+100", "STOP_TOKENS": "['</action>']"}
    for x in ("beta=0.5", "beta=5e-4", "beta=1e-4", "lambda=0.5"):
        out[f"alfworld/beta/4bi/glm_ao/{x}"] = {"EVAL_CKPT_STEPS": "100"}
    return out


def test_rescore_overrides_of_record(bench):
    got = {
        a: {k: str(v) for k, v in arm.entry.meta["rescore"].items()}
        for a, arm in bench.arms.items()
        if arm.entry.meta.get("rescore")
    }
    assert got == _rescore_of_record()


def test_4bt_rescore_flips_exactly_the_guide_and_the_think_tags(bench):
    """The six 4bt trajectory cells were scored under a different prompt than they trained under: the
    re-score turns the guide off and names the think tags. Pin that, and that nothing else changes."""
    for pi in _TRAJ_PIS:
        arm = f"alfworld/main/4bt/{pi}"
        rescore = {k: str(v) for k, v in bench.arms[arm].entry.meta["rescore"].items()}
        train = R.resolve(arm, environ=ENVIRON, recipes_dir=RECIPES).env
        score = R.resolve(arm, overrides=rescore, environ=ENVIRON, recipes_dir=RECIPES).env
        assert (train["INVALID_ACTION_FEEDBACK_GUIDE"], train["ALFWORLD_THINK_TAGS_IN_PROMPT"]) == ("true", "false")
        assert (score["INVALID_ACTION_FEEDBACK_GUIDE"], score["ALFWORLD_THINK_TAGS_IN_PROMPT"]) == ("false", "true")
        changed = {k for k in set(train) | set(score) if train.get(k) != score.get(k)}
        assert changed == {"INVALID_ACTION_FEEDBACK_GUIDE", "ALFWORLD_THINK_TAGS_IN_PROMPT", "EVAL_CKPT_STEPS"}
        assert score["STOP_TOKENS"] == "[]" and score["STRICT_ACTION_OUTSIDE_THINK"] == "true"
        note = bench.arms[arm].entry.meta["note"]
        assert "CAVEAT" in note and "ALFWORLD_THINK_TAGS_IN_PROMPT=true" in note
    # every other 4bt arm was scored in training, without the tags in the prompt
    for a in bench.arms:
        if a.startswith("alfworld/main/4bt/") and a.split("/")[3] not in _TRAJ_PIS:
            assert not bench.arms[a].entry.meta.get("rescore"), a
            assert R.resolve(a, environ=ENVIRON, recipes_dir=RECIPES).env["ALFWORLD_THINK_TAGS_IN_PROMPT"] == "false"


def test_invalid_action_guide_only_on_the_six_early_4bt_arms(bench):
    on = sorted(
        a
        for a in bench.arms
        if R.resolve(a, environ=ENVIRON, allow_planned=True, recipes_dir=RECIPES).env["INVALID_ACTION_FEEDBACK_GUIDE"]
        == "true"
    )
    assert on == sorted(f"alfworld/main/4bt/{pi}" for pi in _TRAJ_PIS)


# Keys run.sh reads without a usable default: every arm must resolve all of them.
LAUNCHER_KEYS = {
    "ENTRY",
    "MODEL_PATH",
    "STRATEGY",
    "NUM_NODES",
    "NUM_GPUS_PER_NODE",
    "INFERENCE_ENGINE_TP_SIZE",
    "GPU_MEM_UTIL",
    "MAX_INPUT_LENGTH",
    "MAX_GENERATE_LENGTH",
    "MAX_MODEL_LEN",
    "MAX_PROMPT_LENGTH",
    "MAX_TURNS",
    "ALFWORLD_MAX_STEPS",
    "ENABLE_THINKING",
    "PROMPT_STYLE",
    "SHOW_ADMISSIBLE",
    "INVALID_ACTION_FEEDBACK",
    "INVALID_ACTION_SHOW_ADMISSIBLE",
    "INVALID_ACTION_FEEDBACK_GUIDE",
    "HISTORY_LENGTH",
    "ADVANTAGE_ESTIMATOR",
    "POLICY_LOSS_TYPE",
    "LOSS_REDUCTION",
    "GRPO_NORM_BY_STD",
    "EPS_CLIP_LOW",
    "EPS_CLIP_HIGH",
    "USE_KL_IN_REWARD",
    "USE_KL_LOSS",
    "KL_LOSS_COEF",
    "LR",
    "MAX_GRAD_NORM",
    "USE_SAMPLE_PACKING",
    "LANGUAGE_MODEL_ONLY",
    "CPU_OFFLOAD",
    "TRAIN_TEMPERATURE",
    "TOP_P",
    "TOP_K",
    "MIN_P",
    "EVAL_TEMPERATURE",
    "EVAL_TOP_P",
    "EVAL_TOP_K",
    "STOP_TOKENS",
    "N_SAMPLES",
    "TRAIN_BATCH_SIZE",
    "MICRO_BATCH",
    "EVAL_BATCH_SIZE",
    "EVAL_N_SAMPLES",
    "EVAL_INTERVAL",
    "EVAL_BEFORE_TRAIN",
    "EVAL_ONLY",
    "TRAINER_EPOCHS",
    "CKPT_INTERVAL",
    "MAX_CKPTS_TO_KEEP",
    "RESUME_MODE",
    "RESUME_PATH",
    "TRAINER_SEED",
    "NUM_ALFWORLD_SERVERS",
    "MAX_ENV_WORKERS",
    "APPLY_OVERLONG_FILTERING",
    "ALFWORLD_REWARD_SCALE",
    "ALFWORLD_REASONING_PREFIX",
    "ALFWORLD_REASONING_STYLE",
    "ALFWORLD_THINK_TAGS_IN_PROMPT",
    "TASK_IDS_PATH",
    "OPSD_TEACHER",
    "OPSD_MIX_LAMBDA",
    "OPSD_RLSD",
    "OPSD_TOKEN_ADV",
    "OPSD_ENV_COEF",
    "OPSD_ROLLOUT_DUMP_DIR",
}


def test_every_arm_resolves_every_launcher_key(bench):
    for a in bench.arms:
        env = R.resolve(a, environ=ENVIRON, allow_planned=True, recipes_dir=RECIPES).env
        missing = sorted(LAUNCHER_KEYS - set(env))
        assert not missing, f"{a}: missing {missing}"
        if a.split("/")[3] != "none" or a.split("/")[1] == "grpo":
            hint = env["OPSD_HINTS_PATH"]
            # every ALFWorld hint is a plain dataset file, except the cross-domain null's (decrypted AppWorld bundle)
            want = XDNULL_HINT if a == "alfworld/null/4bi/null_xdomain" else f"{DATA}/alfworld/"
            assert hint.startswith(want) and hint.endswith(".json"), (a, hint)
            assert "{" not in hint, f"{a}: unexpanded placeholder in {hint}"
        sha = env.get("OPSD_HINTS_CONTENT_SHA256")
        assert sha is None or re.fullmatch(r"[0-9a-f]{64}", sha), (a, sha)


def test_student_contracts(bench):
    env = {
        s: R.resolve(f"alfworld/step0/{s}/none", environ=ENVIRON, recipes_dir=RECIPES).env
        for s in ("4bi", "4bt", "30bi")
    }
    for s, e in env.items():
        assert int(e["MAX_INPUT_LENGTH"]) + int(e["MAX_GENERATE_LENGTH"]) <= int(e["MAX_MODEL_LEN"]), s
        assert e["ENABLE_THINKING"] == e["ALFWORLD_ENABLE_THINKING"], s
    assert env["4bt"]["STRICT_ACTION_OUTSIDE_THINK"] == "true" and env["4bt"]["ALFWORLD_REASONING_STYLE"] == "tag"
    assert env["30bi"]["STRATEGY"] == "megatron" and env["30bi"]["NUM_NODES"] == "2"
    assert env["4bi"]["STRATEGY"] == env["4bt"]["STRATEGY"] == "fsdp2"

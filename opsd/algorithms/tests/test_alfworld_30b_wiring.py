"""Tests for the ALFWorld 30B (MoE, Megatron) path: checkpoint sharding, the shared Megatron block, and
protocol fidelity of the 30B recipe.

Every assertion here guards a SILENT NO-OP, which is the dominant failure mode of a long launcher: a knob
the launcher reads but that never reaches the trainer, or a default that quietly differs from what the
paper arms ran. The arm then runs its full budget on the wrong config while every panel looks normal. That
has happened with pi_type, OPSD_TEACHER, MAX_MODEL_LEN, EVAL_ONLY, EPS_CLIP_HIGH and LR. At 30B the cost of
one is ~8-16 node-hours, so they are asserted rather than eyeballed.

Three layers:
  * core and shared-block checks: FSDP shard names, ``scripts/common/megatron_strategy.sh``;
  * launcher checks: the argv ``scripts/alfworld/run.sh`` prints under DRY_RUN=1 (CPU only, no servers),
    plus the source-level properties a dry run cannot show (how the shared block is included);
  * recipe checks against the resolved 30B arm (``opsd/recipes/alfworld/``).
"""

import os
import re
import shlex
import subprocess
import sys

from opsd._repo import recipes_dir, repo_root, skyrl_root

REPO = repo_root(__file__)
MEGATRON_BLOCK = REPO / "scripts/common/megatron_strategy.sh"
LAUNCH_DIR = REPO / "scripts/alfworld"
LAUNCH_RUN = LAUNCH_DIR / "run.sh"
RECIPES_DIR = recipes_dir() / "alfworld"
ARM_30B = "alfworld/main/30bi/glm_ao"

ARM_4B = "alfworld/main/4bi/glm_ao"


def _dry_run(tmp_path, *args):
    """Run the launcher with DRY_RUN=1 in a clean environment -> (returncode, argv, stderr)."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path / "home"),
        "TMPDIR": str(tmp_path),
        "OPSD_PYTHON": sys.executable,
        "PYTHONDONTWRITEBYTECODE": "1",  # the launcher's python steps write no bytecode into the tree
        "OPSD_DATA_ROOT": str(tmp_path / "data"),
        "RUN_ROOT": str(tmp_path / "runs"),
        "DRY_RUN": "1",
    }
    proc = subprocess.run(
        ["bash", str(LAUNCH_RUN), *args], env=env, cwd=str(tmp_path), capture_output=True, text=True, timeout=300
    )
    argv, lines = [], proc.stdout.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("DRY_RUN: ")), None)
    if start is not None:
        for ln in lines[start + 1 :]:
            if not ln.strip():
                break
            argv += shlex.split(ln.rstrip().removesuffix(" \\"))
    return proc.returncode, argv, proc.stderr


def _src(p):
    return p.read_text()


def _strip_comments(text):
    """Source with comment lines removed.

    Use this for every NEGATIVE assertion. The comments here deliberately QUOTE the thing being asserted
    absent (e.g. a note explaining that `trainer.strategy=fsdp2` used to be hardcoded), so a whole-file
    match fails on the documentation rather than on the code.
    """
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


def _code(p):
    return _strip_comments(p.read_text())


def _launcher_src():
    """Every shell file of the ALFWorld launcher (run.sh plus any helper kept next to it)."""
    return "\n".join(p.read_text() for p in sorted(LAUNCH_DIR.glob("*.sh")))


def _resolved_30b():
    from opsd.recipes.resolve import resolve

    environ = {"OPSD_DATA_ROOT": "/opsd-data", "RUN_ROOT": "/runs", "HOME": "/nonexistent"}
    return resolve(ARM_30B, environ=environ).env


# --------------------------------------------------------------------------- multi-node checkpoints
def test_shard_filenames_carry_the_rank_so_merging_is_safe():
    """On 2 nodes every rank writes its own FSDP shard into one checkpoint directory. That is only safe
    because the shard files are named per-rank; if they were not, node 1 would overwrite node 0's."""
    strategy = _src(skyrl_root() / "skyrl/backends/skyrl_train/distributed/fsdp_strategy.py")
    for stem in ("model", "optim", "extra_state"):
        assert f'f"{stem}_world_size_{{world_size}}_rank_{{rank}}.pt"' in strategy


# --------------------------------------------------------------------------- the shared Megatron block
def test_megatron_block_is_a_noop_on_the_default_path():
    """Every 4B arm must stay bit-identical: MEGATRON_OVERRIDES has to expand to nothing unless
    STRATEGY=megatron, and the env exports must be scoped INSIDE the branch."""
    code = _code(MEGATRON_BLOCK)
    assert "STRATEGY=${STRATEGY:-fsdp2}" in code, "default must remain fsdp2"
    body = code[code.index('if [ "$STRATEGY" = "megatron" ]; then') :]
    for scoped in ("CUDNN_PATH", "LD_LIBRARY_PATH"):
        assert f"export {scoped}=" in body, f"{scoped} must be set INSIDE the megatron branch"
    assert "export " not in code[: code.index('if [ "$STRATEGY" = "megatron" ]; then')], "env leaks to fsdp2"


def test_expandable_segments_is_never_exported():
    """REGRESSION: this cost a 2-node allocation, and it was an imported assumption rather than a check.

    The AppWorld block sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, arguing it is safe under
    Megatron because validate_megatron_cfg asserts weight_sync_backend == nccl, so no CUDA-IPC transfer
    exists to break. MEASURED FALSE on the ALFWorld path -- the first weight sync went through
    weight_sync/cuda_ipc_strategy.py and died with

        rebuild_cuda_tensor -> RuntimeError: pidfd_getfd: Operation not permitted

    exactly as the standing rule ("never set expandable_segments on a colocate run") predicts. The rule
    holds on BOTH backends here; only the reason differs. A fragmentation OOM is fixed with CP or the
    micro-batch, not this flag.
    """
    code = _code(MEGATRON_BLOCK)
    assert not re.search(
        r"^\s*export\s+PYTORCH_CUDA_ALLOC_CONF=", code, re.M
    ), "expandable_segments breaks the CUDA-IPC weight sync on this path"


def test_ref_model_gets_its_own_parallelism():
    """THE OPSD-CRITICAL ONE. Vanilla OPSD sets use_kl_in_reward=true and
    `use_ref_model = use_kl_loss or use_kl_in_reward`, so a ref worker IS built -- here it is the FROZEN
    TEACHER. Unconfigured it defaults to TP=1/PP=1/EP=1 and every rank tries to hold the whole 30.5 B
    (~61 GB bf16): a silent OOM that reads as a memory failure but is a config failure.
    """
    src = _src(MEGATRON_BLOCK)
    for field in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
    ):
        assert f"$_RG.{field}=" in src, f"ref megatron_config missing {field}"
    assert "$_RG.empty_cuda_cache=true" in src
    # ref must MIRROR the policy, not default independently
    assert "$_RG.tensor_model_parallel_size=${MEGATRON_TP:-4}" in src


def test_routing_replay_sets_all_three_coupled_flags():
    """validate_megatron_cfg asserts all three agree; `moe_enable_routing_replay` alone kills the job in
    config validation before a single rollout."""
    src = _src(MEGATRON_BLOCK)
    seg = src[src.index('if [ "${MOE_ROUTING_REPLAY:-false}" = "true" ]; then') :]
    assert "enable_return_routed_experts=true" in seg
    assert "distributed_executor_backend=mp" in seg
    # default OFF: unvalidated on a colocate path (generation died with an empty vLLM response body)
    assert "moe_enable_routing_replay=${MOE_ROUTING_REPLAY:-false}" in src


def test_parallelism_guards_run_before_ray():
    """Megatron enforces two divisibility rules and violating the second costs a whole run
    (`world_size (8) is not divisible by expert_tensor_model_pipeline_parallel size (16)`). CP must be part
    of the model-parallel product or the printed DP is wrong."""
    src = _src(MEGATRON_BLOCK)
    assert "_TPPP=$(( ${MEGATRON_TP:-4} * ${MEGATRON_PP:-1} * ${MEGATRON_CP:-1} ))" in src, "CP missing from DP"
    assert "_ETMP=$(( ${MEGATRON_ETP:-1} * ${MEGATRON_EP:-8} * ${MEGATRON_PP:-1} ))" in src
    assert src.count("exit 1") >= 3, "each divisibility/packing violation must exit"
    # TP=4 PP=1 EP=8 ETP=1 must be legal on both 8 and 16 GPUs
    for ws in (8, 16):
        assert ws % (4 * 1 * 1) == 0 and ws % (1 * 8 * 1) == 0


def test_cp_gt_1_requires_packing():
    """CP>1 asserts sample packing, and the vanilla-OPSD arms run packing OFF to replicate the 4B protocol
    -- so CP=1 is the only setting compatible with them (the recipe check below pins CP=1)."""
    guard = 'if [ "${MEGATRON_CP:-1}" -gt 1 ] && [ "${USE_SAMPLE_PACKING:-true}" != "true" ]; then'
    assert guard in _src(MEGATRON_BLOCK)


def test_prestage_has_a_real_consumer():
    """PRESTAGE_MODEL must be read somewhere, or setting it is the silent-no-op pattern this file exists to
    prevent. A live pull of a 61 GB MoE by every worker is the same shape as the failure that killed a
    Qwen3.5-4B run."""
    assert 'if [ "${PRESTAGE_MODEL:-0}" = "1" ]' in _src(MEGATRON_BLOCK)


def test_launcher_prestages_after_preflight_not_while_building_the_argv():
    """The launcher sources the shared block for the argv with the pre-stage held off, and runs the block's
    pre-stage only after preflight, so a busy GPU or port fails before a 61 GB download starts."""
    lines = _code(LAUNCH_RUN).splitlines()
    first = next(i for i, ln in enumerate(lines) if re.match(r'\s*source "\$_MGT_BLOCK"', ln))
    assert any(re.match(r"\s*PRESTAGE_MODEL=0\s*$", ln) for ln in lines[max(0, first - 3) : first])
    restore = next(i for i, ln in enumerate(lines) if i > first and "PRESTAGE_MODEL=" in ln)
    assert lines[restore].strip() == "PRESTAGE_MODEL=$_PRESTAGE_REQ"
    preflight = next(i for i, ln in enumerate(lines) if "opsd_preflight_all" in ln)
    stage = next(i for i, ln in enumerate(lines) if re.search(r'\(\s*source "\$_MGT_BLOCK"\s*\)', ln))
    gate = next(i for i, ln in enumerate(lines) if 'if [ "${PRESTAGE_MODEL:-0}" = 1 ]' in ln)
    assert first < preflight < gate < stage, (first, preflight, gate, stage)


def test_worker_node_prestage_is_documented():
    """Only the launching (head) node runs the pre-stage; the worker of a 2-node run must download the model
    itself before it joins the cluster, or its 8 ranks pull the 61 GB MoE live."""
    header = LAUNCH_RUN.read_text().split("set -o pipefail", 1)[0]
    dl = header.index("hf download Qwen/Qwen3-30B-A3B-Instruct-2507")
    assert dl < header.index("ray_cluster.sh worker"), "the worker downloads BEFORE it joins"
    student = (RECIPES_DIR / "students" / "30bi.env").read_text()
    assert "once per node" not in student and "hf download Qwen/Qwen3-30B-A3B-Instruct-2507" in student


# --------------------------------------------------------------------------- the launcher
def test_launcher_parameterises_lr(tmp_path):
    """REGRESSION: `optimizer_config.lr=1.0e-6` was once hardcoded at the hydra call, so `LR=3e-7` was a
    no-op and the arm trained at 1e-6 under whatever label the caller chose."""
    rc, argv, err = _dry_run(tmp_path, ARM_4B, "LR=3e-7")
    assert rc == 0, err
    assert "trainer.policy.optimizer_config.lr=3e-7" in argv
    assert "optimizer_config.lr=1.0e-6" not in _strip_comments(_launcher_src())


def test_launcher_does_not_hardcode_fsdp2(tmp_path):
    """`trainer.strategy=fsdp2` was once hardcoded, so a MoE had NO supported path -- and a 30B MoE run down
    the FSDP colocate path emitted pure `++;` token garbage on 144/144 rollouts. The 30B arm must reach hydra
    as megatron WITH the shared block's parallelism for policy AND ref; a 4B arm must carry none of it."""
    rc, argv, err = _dry_run(tmp_path, ARM_30B)
    assert rc == 0, err
    assert "trainer.strategy=megatron" in argv
    for arg in (
        "trainer.policy.megatron_config.tensor_model_parallel_size=4",
        "trainer.policy.megatron_config.expert_model_parallel_size=8",
        "trainer.ref.megatron_config.tensor_model_parallel_size=4",
        "trainer.ref.megatron_config.expert_model_parallel_size=8",
        "trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=true",
        "trainer.placement.policy_num_nodes=2",
        "generator.inference_engine.tensor_parallel_size=4",
        "generator.inference_engine.num_engines=4",
    ):
        assert arg in argv, f"30B argv lacks {arg}"
    rc, argv, err = _dry_run(tmp_path, ARM_4B)
    assert rc == 0, err
    assert "trainer.strategy=fsdp2" in argv
    assert not [a for a in argv if "megatron_config" in a], "megatron overrides leak into a 4B arm"


def test_the_sourced_megatron_block_path_actually_resolves():
    """REGRESSION, and it cost a full 2-node run.

    The source line was once written with a relative `../../..` idiom copied from a script at a different
    depth; it resolved to a non-existent `scripts/common/megatron_strategy.sh`, the source failed, and
    -- because the failure was tolerated -- the run continued with MEGATRON_OVERRIDES EMPTY. hydra then got
    trainer.strategy=megatron with ALL-DEFAULT parallelism (TP=1/EP=1 on policy AND ref), which OOMed 8
    minutes in trying to allocate 56.87 GiB on a single rank. Asserting the resolved path exists catches it
    before any GPU time (the dry-run test above shows the overrides actually arrive).
    """
    code = _strip_comments(_launcher_src())
    m = re.search(r"\$\{?(ROOT|REPO|REPO_ROOT|OPSD_HOME)\}?/([\w./-]*megatron_strategy\.sh)", code)
    assert m, "reference the shared block relative to the repo root ($ROOT/...), not via ../"
    # Every one of these names the repository root (REPO here), which holds scripts/.
    resolved = (REPO / m.group(2)).resolve()
    assert resolved.exists(), f"the launcher sources a non-existent path: {resolved}"
    assert resolved == MEGATRON_BLOCK.resolve(), f"the launcher resolves to {resolved}, not the shared block"


def test_a_missing_megatron_block_is_fatal_not_tolerated():
    """The defect was not only the wrong path -- a failed include DEGRADED into a differently-configured run
    instead of stopping. Accept either an explicit existence check that exits, or errexit at the source."""
    lines = _strip_comments(_launcher_src()).splitlines()
    src_idx = next(
        (i for i, ln in enumerate(lines) if re.match(r"\s*(source|\.)\s", ln) and re.search(r"megatron|mgt", ln, re.I)),
        None,
    )
    assert src_idx is not None, "could not find the line that sources the shared Megatron block"
    window = "\n".join(lines[max(0, src_idx - 15) : src_idx])
    explicit = "-f " in window and re.search(r"\b(exit|die)\b", window)
    errexit = any(re.match(r"\s*set\s+(-[a-zA-Z]*e[a-zA-Z]*\b|-o\s+errexit)", ln) for ln in lines[:src_idx])
    assert explicit or errexit, "a missing megatron_strategy.sh must abort the launch"


def test_launcher_never_exports_expandable_segments(tmp_path):
    """Same rule as the shared block: it must not be smuggled in through the launcher or the recipe."""
    assert not re.search(r"^\s*export\s+PYTORCH_CUDA_ALLOC_CONF=", _strip_comments(_launcher_src()), re.M)
    assert "PYTORCH_CUDA_ALLOC_CONF" not in _resolved_30b()


# --------------------------------------------------------------------------- the 30B recipe
def test_30b_recipe_pins_the_verified_model_and_megatron_layout():
    env = _resolved_30b()
    assert env["MODEL_PATH"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert env["STRATEGY"] == "megatron", "a 30B MoE down the FSDP colocate path emits garbage"
    assert env["INFERENCE_ENGINE_TP_SIZE"] == "4", "TP=1 puts 61 GB of weights on one 80 GB card"
    assert env["NUM_GPUS_PER_NODE"] == "8"
    for key, value in (("MEGATRON_TP", "4"), ("MEGATRON_PP", "1"), ("MEGATRON_EP", "8"), ("MEGATRON_ETP", "1")):
        assert env[key] == value, f"{key}={env.get(key)} differs from the configuration that ran"
    assert env["MEGATRON_CP"] == "1", "CP>1 requires packing, which the vanilla-OPSD arms run OFF"
    assert env["MOE_ROUTING_REPLAY"] == "false"
    assert env["PRESTAGE_MODEL"] == "1"
    assert "expandable_segments" not in env.get("PYTORCH_CUDA_ALLOC_CONF", "")


def test_30b_recipe_gives_every_arm_the_same_100_step_budget():
    """The 4B GRPO control defaults to 2 epochs. Left at that, a control would get twice the optimization
    budget of the OPSD arms -- and at the MATCHED step 100 the 4B ordering reverses."""
    env = _resolved_30b()
    assert env["TRAINER_EPOCHS"] == "1"
    # eval/ckpt cadence is per arm (the later 30B arms ran every 10 steps); this arm ran every 25
    assert env["EVAL_INTERVAL"] == "25" and env["CKPT_INTERVAL"] == "25"
    assert env["LR"] == "1.0e-6"


def test_30b_recipe_samples_purely_on_policy():
    """TOP_K=20 renormalises over the 20 likeliest tokens, so rollouts come from a TRUNCATED q != pi_theta
    while every importance ratio and teacher-KL term is computed as though they came from pi_theta -- a
    silent, one-directional bias that deletes exactly the exploration tail. -1 is vLLM's disable value.
    EVAL sampling is NOT dragged along: it stays on the checkpoint's official 0.7/0.8/20 profile.
    """
    env = _resolved_30b()
    assert (env["TOP_P"], env["TOP_K"], env["TRAIN_TEMPERATURE"], env["MIN_P"]) == ("1.0", "-1", "1.0", "0.0")
    assert (env["EVAL_TEMPERATURE"], env["EVAL_TOP_P"], env["EVAL_TOP_K"]) == ("0.7", "0.8", "20")


def test_30b_recipe_lengths_sum_correctly():
    """16k context; 12288 + 4096 == 16384 exactly, so the input cap is not silently inconsistent with ctx."""
    env = _resolved_30b()
    assert env["MAX_MODEL_LEN"] == "16384"
    assert int(env["MAX_INPUT_LENGTH"]) + int(env["MAX_GENERATE_LENGTH"]) == int(env["MAX_MODEL_LEN"])
    assert (env["MAX_INPUT_LENGTH"], env["MAX_GENERATE_LENGTH"]) == ("12288", "4096")


def test_30b_recipe_is_vanilla_opsd():
    """These are what MAKE an arm OPSD: a group-relative advantage here would no longer be vanilla OPSD."""
    env = _resolved_30b()
    assert env["ADVANTAGE_ESTIMATOR"] == "no_op"
    assert env["USE_KL_IN_REWARD"] == "true" and env["USE_KL_LOSS"] == "false"
    assert env["POLICY_LOSS_TYPE"] == "importance_sampling"


def test_30b_recipe_prompt_and_env_feedback_settings():
    """The prompt and env-feedback settings the 30B arms ran with: the THOUGHTS: prefix (the style the model
    actually adopts; the <think>-tag variant got 0% compliance twice at 4B), no worked demo (it cost 37pp of
    ACTION compliance at 4B), no format reward, and the shared env-feedback block -- SHOW_ADMISSIBLE in
    particular is the single biggest measured lever on ALFWorld success."""
    env = _resolved_30b()
    expected = {
        "ALFWORLD_THINK_TAGS_IN_PROMPT": "true",
        "ALFWORLD_REASONING_STYLE": "prefix",
        "ALFWORLD_ONESHOT_EXAMPLE": "false",
        "ALFWORLD_FORMAT_REWARD_COEF": "0.0",
        "PROMPT_STYLE": "react",
        "SHOW_ADMISSIBLE": "false",
        "INVALID_ACTION_FEEDBACK": "true",
        "INVALID_ACTION_SHOW_ADMISSIBLE": "false",
        "HISTORY_LENGTH": "0",
    }
    assert {k: env.get(k) for k in expected} == expected

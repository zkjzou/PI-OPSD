"""Tests for DAPO Soft Overlong Punishment.

Two layers, because the bug that motivated these tests was NOT in the arithmetic:

* the arithmetic of ``apply_soft_overlong_punishment`` (fast, no heavy imports), and
* the WIRING -- that the config paths the trainer reads actually resolve, and that the penalty lives on
  an object that owns the full config.

The original implementation put the penalty in ``SkyRLGymGenerator`` and read
``self.cfg.trainer.algorithm.overlong_buffer``. The generator has no ``self.cfg`` -- ``get_generator``
hands it only ``cfg.generator`` and ``cfg.environment.skyrl_gym`` -- so it raised AttributeError on the
first generate call and killed both DAPO arms (each run was created and logged zero rows).
The check that missed it called the method on an object whose ``cfg`` had been set by hand, which
exercised only the maths. Hence the wiring tests below pin the seam, not the formula. They drive SkyRL's
own (unmodified) ``BasePPOExp.get_generator`` and ``RayPPOTrainer.__init__`` / ``train()`` / ``generate()`` /
``eval()`` with the DAPO entrypoint's trainer and stubbed model stages, so a SkyRL change that moves a seam
fails here.
"""

import asyncio
import importlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from opsd.algorithms.overlong_buffer import apply_soft_overlong_punishment
from opsd.skyrl_ext.config import OPSDTrainConfig, OverlongBufferConfig

OPSD_ENTRY = "opsd.trainer.main_opsd_alfworld"
DAPO_ENTRY = "opsd.alfworld.main_alfworld_dapo"


def _launcher() -> Path:
    """The ALFWorld launcher that picks the trainer entrypoint; checked on what it prints under DRY_RUN=1."""
    here = Path(__file__).resolve()
    for d in here.parents:
        if (d / "scripts" / "alfworld" / "run.sh").is_file():
            return d / "scripts" / "alfworld" / "run.sh"
    raise FileNotFoundError(f"no scripts/alfworld/run.sh above {here}")


def _dry_run(tmp_path, *args):
    """Run the launcher with DRY_RUN=1 in a clean environment -> (returncode, module, argv, stderr)."""
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
        ["bash", str(_launcher()), *args], env=env, cwd=str(tmp_path), capture_output=True, text=True, timeout=300
    )
    module, argv, lines = None, [], proc.stdout.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("DRY_RUN: ")), None)
    if start is not None:
        module = lines[start].split(" -m ", 1)[1].rstrip(" \\").strip()
        for ln in lines[start + 1 :]:
            if not ln.strip():
                break
            argv += shlex.split(ln.rstrip().removesuffix(" \\"))
    return proc.returncode, module, argv, proc.stderr


# 24000 + 8000, the alfworld DAPO arms' budget; LEN=7000 puts the onset at 25,000.
BUDGET = 32000
BUFFER = 7000
ONSET = BUDGET - BUFFER


# ---------------------------------------------------------------------------
# arithmetic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "length,expected",
    [
        (0, 10.0),  # empty
        (ONSET - 1, 10.0),  # just below the onset: untouched
        (ONSET, 10.0),  # AT the onset: untouched (strict >)
        (ONSET + BUFFER // 2, 10.0 - 0.5),  # halfway up the ramp
        (BUDGET, 10.0 - 1.0),  # at the budget: full penalty_factor
        (BUDGET + 1, 10.0 - 1.0),  # past it: SATURATES at -penalty_factor (paper eq.13 / verl), NOT zeroed
        (BUDGET * 2, 10.0 - 1.0),
    ],
)
def test_ramp_shape(length, expected):
    rewards, _ = apply_soft_overlong_punishment([10.0], [length], BUDGET, BUFFER, penalty_factor=1.0)
    assert rewards[0] == pytest.approx(expected)


def test_penalty_is_additive_never_a_wholesale_zeroing():
    """REGRESSION vs SkyRL's bundled example, which does `rewards[i] = 0.0` past the budget.

    The paper (arXiv:2503.14476 eq.13) and verl both ADD a bounded term that saturates at
    -penalty_factor. Zeroing would make a trajectory at the budget worth 9.0 and one token longer worth
    0.0 -- a 9-unit cliff, in a scheme whose entire purpose is to replace a cliff with a ramp.
    """
    at, past = (
        apply_soft_overlong_punishment([10.0], [BUDGET], BUDGET, BUFFER)[0][0],
        apply_soft_overlong_punishment([10.0], [BUDGET + 5000], BUDGET, BUFFER)[0][0],
    )
    assert at == pytest.approx(9.0)
    assert past == pytest.approx(9.0), "penalty must saturate, not zero the reward"
    assert abs(at - past) < 1e-9, "no discontinuity at the budget"


def test_disabled_at_zero_buffer_len():
    """len=0 is the documented 'off', and must not divide by zero."""
    rewards, stats = apply_soft_overlong_punishment([10.0, 10.0], [0, 10**9], BUDGET, 0)
    assert rewards == [10.0, 10.0]
    assert stats["overlong_buffer/frac_penalized"] == 0.0
    assert stats["overlong_buffer/frac_saturated"] == 0.0


def test_negative_buffer_len_is_also_off():
    rewards, _ = apply_soft_overlong_punishment([10.0], [10**9], BUDGET, -1)
    assert rewards == [10.0]


def test_penalty_factor_scales_the_ramp():
    r1, _ = apply_soft_overlong_punishment([10.0], [BUDGET], BUDGET, BUFFER, penalty_factor=1.0)
    r5, _ = apply_soft_overlong_punishment([10.0], [BUDGET], BUDGET, BUFFER, penalty_factor=5.0)
    assert r1[0] == pytest.approx(9.0)
    assert r5[0] == pytest.approx(5.0)


def test_default_reward_range_reproduces_the_paper():
    """Default reward_range is DAPO's own +/-1 span, so the published factor 1.0 costs exactly 1.0 --
    i.e. a correct answer (+1) is driven to neutral, as in eq.13."""
    rewards, _ = apply_soft_overlong_punishment([1.0], [BUDGET], BUDGET, BUFFER, penalty_factor=1.0)
    assert rewards[0] == pytest.approx(0.0)


def test_penalty_is_proportional_to_the_env_reward_range():
    """THE SCALE TRAP. penalty_factor is defined on DAPO's +/-1 span; copying the published 1.0 onto
    ALFWorld's 10.0*won would bite ~5x weaker in proportional terms. reward_range rescales it, so the
    same factor removes the same FRACTION of the reward range on either env."""
    dapo, _ = apply_soft_overlong_punishment([1.0], [BUDGET], BUDGET, BUFFER, penalty_factor=1.0, reward_range=2.0)
    alf, _ = apply_soft_overlong_punishment([10.0], [BUDGET], BUDGET, BUFFER, penalty_factor=1.0, reward_range=10.0)
    assert dapo[0] == pytest.approx(0.0), "DAPO: +1 correct -> 0"
    assert alf[0] == pytest.approx(5.0), "ALFWorld: 10.0 success -> 5.0, i.e. half the range, same as DAPO"
    # the penalty is the same FRACTION of each env's span
    assert (1.0 - dapo[0]) / 2.0 == pytest.approx((10.0 - alf[0]) / 10.0)


def test_reward_range_default_is_backwards_compatible():
    """Omitting reward_range must behave exactly as the raw-units version did on a +/-1 env."""
    explicit, _ = apply_soft_overlong_punishment([1.0], [BUDGET], BUDGET, BUFFER, penalty_factor=1.0, reward_range=2.0)
    implicit, _ = apply_soft_overlong_punishment([1.0], [BUDGET], BUDGET, BUFFER, penalty_factor=1.0)
    assert explicit == implicit


def test_effective_penalty_is_reported():
    """The rescaled magnitude must be visible, else a mis-set reward_range is invisible in the logs."""
    _, stats = apply_soft_overlong_punishment([10.0], [BUDGET], BUDGET, BUFFER, penalty_factor=1.0, reward_range=10.0)
    assert stats["overlong_buffer/effective_penalty"] == pytest.approx(5.0)


def test_zero_reward_trajectory_goes_negative():
    """A failed-but-overlong trajectory must be worse than a failed-and-short one, else the penalty
    cannot separate them and contributes no gradient in a group where nobody succeeded."""
    rewards, _ = apply_soft_overlong_punishment([0.0, 0.0], [100, BUDGET], BUDGET, BUFFER)
    assert rewards[0] == 0.0
    assert rewards[1] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# per-token / per-turn reward lists (multi-turn ALFWorld can carry these)
# ---------------------------------------------------------------------------
def test_list_reward_penalty_lands_on_last_element():
    rewards, _ = apply_soft_overlong_punishment([[0.0, 0.0, 10.0]], [BUDGET], BUDGET, BUFFER)
    assert rewards[0] == pytest.approx([0.0, 0.0, 9.0])


def test_list_reward_saturates_on_last_element_past_budget():
    """Past the budget the list case saturates on the terminal element, matching the scalar case --
    the other elements are untouched (they are not the terminal env reward)."""
    rewards, _ = apply_soft_overlong_punishment([[1.0, 2.0, 10.0]], [BUDGET + 1], BUDGET, BUFFER)
    assert rewards[0] == pytest.approx([1.0, 2.0, 9.0])


def test_empty_list_reward_does_not_raise():
    rewards, _ = apply_soft_overlong_punishment([[]], [BUDGET], BUDGET, BUFFER)
    assert rewards[0] == []


def test_mixed_scalar_and_list_rewards():
    rewards, _ = apply_soft_overlong_punishment([10.0, [0.0, 10.0]], [BUDGET, BUDGET], BUDGET, BUFFER)
    assert rewards[0] == pytest.approx(9.0)
    assert rewards[1] == pytest.approx([0.0, 9.0])


# ---------------------------------------------------------------------------
# stats -- "did it fire" is not "did it do anything"
# ---------------------------------------------------------------------------
def test_stats_report_what_fired():
    lengths = [100, ONSET + 1000, BUDGET + 1, BUDGET + 1]
    _, stats = apply_soft_overlong_punishment([10.0] * 4, lengths, BUDGET, BUFFER)
    # frac_penalized counts everything past the onset, INCLUDING the saturated ones (3 of 4);
    # frac_saturated is the subset pinned at -penalty_factor (2 of 4).
    assert stats["overlong_buffer/frac_penalized"] == pytest.approx(0.75)
    assert stats["overlong_buffer/frac_saturated"] == pytest.approx(0.5)
    assert stats["overlong_buffer/onset_tokens"] == ONSET


def test_empty_batch_does_not_divide_by_zero():
    rewards, stats = apply_soft_overlong_punishment([], [], BUDGET, BUFFER)
    assert rewards == []
    assert stats["overlong_buffer/frac_penalized"] == 0.0


def test_onset_floors_at_zero_when_buffer_exceeds_budget():
    _, stats = apply_soft_overlong_punishment([10.0], [1], 1000, 5000)
    assert stats["overlong_buffer/onset_tokens"] == 0.0


# ---------------------------------------------------------------------------
# WIRING -- the layer the original bug lived in
# ---------------------------------------------------------------------------
# One training run of the DAPO entrypoint's trainer through SkyRL's own train(): 2 steps of 1 prompt x 2
# samples, eval before training and after every step, every rollout an ALFWorld success (10.0). The trajectory
# budget is max_input_length + max_generate_length = 40 with a 10-token buffer, so a 40-token rollout takes the
# full penalty (10.0 -> 5.0 at reward_range=10) and a 5-token one none. Training step 1 rolls out 40 tokens,
# step 2 5 tokens; eval always rolls out 40, so an eval that paid the penalty would score 5.0, not 10.0.
WIRING_BUDGET = (20, 20)  # generator.max_input_length, generator.sampling_params.max_generate_length
WIRING_BUFFER = 10
TRAIN_LENGTHS = {1: 40, 2: 5}  # global_step -> rollout length
EVAL_LENGTH = 40


def _wiring_cfg(tmp_path):
    return OPSDTrainConfig.from_cli_overrides(
        [
            f"trainer.algorithm.overlong_buffer.len={WIRING_BUFFER}",
            "trainer.algorithm.overlong_buffer.penalty_factor=1.0",
            "trainer.algorithm.overlong_buffer.reward_range=10.0",
            f"generator.max_input_length={WIRING_BUDGET[0]}",
            f"generator.sampling_params.max_generate_length={WIRING_BUDGET[1]}",
            "generator.n_samples_per_prompt=2",
            "generator.eval_n_samples_per_prompt=2",
            "trainer.policy.model.path=stub/model",
            "trainer.epochs=1",
            "trainer.eval_interval=1",
            "trainer.eval_before_train=true",
            "trainer.ckpt_interval=0",
            "trainer.hf_save_interval=-1",
            "trainer.resume_mode=none",
            "trainer.log_example_interval=0",
            "trainer.dump_eval_results=false",
            "trainer.logger=console",
            "generator.inference_engine.enable_ray_prometheus_stats=false",  # its scraper needs a Ray cluster
            "trainer.placement.colocate_all=false",  # no inference engine to sleep / wake
            f"trainer.export_path={tmp_path / 'exports'}",
            f"trainer.ckpt_path={tmp_path / 'ckpts'}",
            "environment.env_class=alfworld",
            "environment.skyrl_gym.max_env_workers=0",
        ]
    )


class _StubTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, return_dict=False, **kw):
        return [1] * len(messages) + [2] + ([3] if add_generation_prompt else [])

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class _Generator:
    """Stands in for the rollout engine: every rollout is a success (10.0) of a fixed length per phase/step."""

    def __init__(self):
        self.calls = []

    async def generate(self, input_batch):
        meta = input_batch["batch_metadata"]
        self.calls.append((meta.training_phase, meta.global_step))
        length = EVAL_LENGTH if meta.training_phase == "eval" else TRAIN_LENGTHS[meta.global_step]
        n = len(input_batch["prompts"])
        return {
            "prompt_token_ids": [[1, 2] for _ in range(n)],
            "response_ids": [[5] * length for _ in range(n)],
            "rewards": [10.0] * n,
            "loss_masks": [[1] * length for _ in range(n)],
            "stop_reasons": ["stop"] * n,
            "rollout_metrics": {"generate/probe_len": float(length)},
            "rollout_logprobs": None,
        }


class _Tracker:
    def __init__(self):
        self.logs = []
        self.finished = False

    def log(self, data, step, commit=False):
        self.logs.append((step, dict(data)))

    def finish(self):
        self.finished = True


class _Dispatch:
    async def save_weights_for_sampler(self):
        pass

    def get_lcm_dp_size(self):
        return 1


class _Batch(dict):
    def __init__(self):
        super().__init__(rewards=None)
        self.metadata = {"uids": []}


def _prompt(uid):
    return {
        "prompt": [{"role": "user", "content": uid}],
        "env_class": None,
        "env_extras": {"data_source": "ds"},
        "uid": uid,
    }


@pytest.fixture(scope="module")
def dapo_entry():
    return importlib.import_module(DAPO_ENTRY)


def _trainer_kwargs(cfg, **kw):
    return dict(
        cfg=cfg,
        tracker=kw.get("tracker"),
        tokenizer=_StubTokenizer(),
        train_dataset=None,
        eval_dataset=None,
        inference_engine_client=None,
        generator=kw.get("generator"),
        colocate_pg=None,
    )


@pytest.fixture(scope="module")
def dapo_run(dapo_entry, tmp_path_factory):
    """Run the DAPO entrypoint's trainer through RayPPOTrainer.train() once; the model stages are stubbed."""
    import skyrl.train.trainer as upstream_trainer

    cfg = _wiring_cfg(tmp_path_factory.mktemp("dapo_run"))
    exp = object.__new__(dapo_entry.ALFWorldDAPOExp)
    exp.cfg = cfg
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(upstream_trainer, "configure_ray_worker_logging", lambda: None)  # global loguru setup
        trainer_cls = type(exp.get_trainer(**_trainer_kwargs(cfg)))

        class _StubbedStages(trainer_cls):
            """The entrypoint's trainer with the GPU stages of train() stubbed. train(), generate(), eval() and
            the postprocess_generator_output chain are the real ones."""

            def init_weight_sync_state(self):
                pass

            def postprocess_generator_output(self, generator_output, uids):
                self.postprocess_saw.append(set(generator_output))
                return super().postprocess_generator_output(generator_output, uids)

            def convert_to_training_input(self, generator_output, uids):
                self.trained_rewards.append([list(r) for r in generator_output["rewards"]])
                return _Batch()

            def fwd_logprobs_values_reward(self, data):
                return data

            def compute_advantages_and_returns(self, data):
                return data

            def train_critic_and_policy(self, data):
                return {}

        tracker, generator = _Tracker(), _Generator()
        trainer = _StubbedStages(**_trainer_kwargs(cfg, tracker=tracker, generator=generator))
        trainer.train_dataloader = [[_prompt("t0")], [_prompt("t1")]]
        trainer.total_training_steps = 2
        trainer.eval_dataloader = [[_prompt("e0")]]
        trainer.dispatch = _Dispatch()
        trainer.postprocess_saw, trainer.trained_rewards = [], []
        asyncio.run(trainer.train())
    return cfg, trainer_cls, trainer, tracker, generator


def test_generator_cannot_reach_trainer_config(dapo_entry, tmp_path):
    """REGRESSION: the penalty must not be reintroduced into the generator.

    The DAPO entrypoint's ``get_generator`` builds a ``SkyRLGymGenerator`` from ``cfg.generator`` and
    ``cfg.environment.skyrl_gym`` only, so any ``self.cfg`` there is an AttributeError at rollout time --
    which no config validation and no metric can catch.
    """
    from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator

    cfg = _wiring_cfg(tmp_path)
    exp = object.__new__(dapo_entry.ALFWorldDAPOExp)
    exp.cfg = cfg
    gen = exp.get_generator(cfg, _StubTokenizer(), None)
    assert isinstance(gen, SkyRLGymGenerator)
    assert gen.generator_cfg is cfg.generator and gen.skyrl_gym_cfg is cfg.environment.skyrl_gym
    full = (cfg, cfg.trainer, cfg.trainer.algorithm, cfg.trainer.algorithm.overlong_buffer)
    assert not [k for k, v in vars(gen).items() if any(v is c for c in full)], "the generator holds trainer config"
    with pytest.raises(AttributeError):
        gen.cfg


def test_trainer_owns_full_config(dapo_run):
    """The seam this feature depends on: RayPPOTrainer.__init__ binds the whole config as self.cfg."""
    cfg, _, trainer, _, _ = dapo_run
    assert trainer.cfg is cfg
    assert type(trainer.cfg.trainer.algorithm.overlong_buffer) is OverlongBufferConfig
    assert trainer.cfg.trainer.algorithm.overlong_buffer.len == WIRING_BUFFER


def test_penalty_applies_to_training_only_not_eval(dapo_run):
    """postprocess_generator_output is called from train() only; eval() must not pay the penalty, or the
    reported success rate stops being comparable with every other arm."""
    _, _, trainer, tracker, generator = dapo_run
    assert generator.calls == [("eval", 0), ("train", 1), ("eval", 1), ("train", 2), ("eval", 2)]
    assert len(trainer.postprocess_saw) == 2, "postprocess must run once per training step and never for eval"
    # what train() trained on: per-token rewards with the penalty on the terminal token
    assert trainer.trained_rewards == [[[0.0] * 39 + [5.0]] * 2, [[0.0] * 4 + [10.0]] * 2]
    # every eval rollout sat at the budget, yet scored the unpenalised 10.0
    steps = [step for step, _ in tracker.logs]
    assert steps == [0, 1, 2]
    for _, logged in tracker.logs:
        assert logged["eval/all/avg_score"] == pytest.approx(10.0)
    assert not [k for k in tracker.logs[0][1] if k.startswith("overlong_buffer/")]


def test_stats_go_to_all_metrics_not_rollout_metrics(dapo_run):
    """REGRESSION: ``rollout_metrics`` is consumed and POPPED inside generate(), which runs earlier in the
    same step than postprocess_generator_output -- so writing the penalty stats there is a silent no-op.

    Caught only by noticing the keys were absent in W&B at step 129 while the penalty itself worked. The
    original code guarded with ``if isinstance(metrics, dict)``, which turned the wiring error into
    silence instead of a KeyError.
    """
    _, _, trainer, tracker, _ = dapo_run
    assert all("rollout_metrics" not in keys for keys in trainer.postprocess_saw), "generate() no longer pops it"
    by_step = dict(tracker.logs)
    # generate() routes the generator's rollout_metrics into self.all_metrics ...
    assert by_step[1]["generate/probe_len"] == 40.0 and by_step[2]["generate/probe_len"] == 5.0
    # ... and the penalty stats written to self.all_metrics by postprocess reach the same step's log
    for step in (1, 2):
        assert {"overlong_buffer/frac_penalized", "overlong_buffer/effective_penalty"} <= set(by_step[step])
    assert by_step[1]["overlong_buffer/effective_penalty"] == pytest.approx(5.0)


def test_all_metrics_is_flushed_after_postprocess_runs(dapo_run):
    """self.all_metrics must still be live when the override writes to it: flushed at the end of the step,
    then cleared -- not before postprocess."""
    _, _, trainer, tracker, _ = dapo_run
    by_step = dict(tracker.logs)
    # each step logs ITS OWN stats (step 1: every rollout past the onset; step 2: none) -- nothing stale
    assert by_step[1]["overlong_buffer/frac_penalized"] == 1.0
    assert by_step[2]["overlong_buffer/frac_penalized"] == 0.0
    # upstream's reward metrics are computed after the override, i.e. on the penalised rewards
    assert by_step[1]["reward/avg_raw_reward"] == pytest.approx(5.0)
    assert by_step[2]["reward/avg_raw_reward"] == pytest.approx(10.0)
    assert trainer.all_metrics == {} and tracker.finished


def test_config_paths_the_trainer_reads_all_resolve():
    """Every dotted path the override touches must exist on a real default config."""
    cfg = OPSDTrainConfig()
    assert int(cfg.trainer.algorithm.overlong_buffer.len) == 0, "default must be OFF"
    assert float(cfg.trainer.algorithm.overlong_buffer.penalty_factor) == 1.0
    assert isinstance(int(cfg.generator.max_input_length), int)
    assert isinstance(int(cfg.generator.sampling_params.max_generate_length), int)


# ---------------------------------------------------------------------------
# the launcher must not hand the knob to a trainer that ignores it
# ---------------------------------------------------------------------------
def test_run_script_runs_the_dapo_entrypoint_with_the_buffer(tmp_path):
    rc, module, argv, err = _dry_run(
        tmp_path, "alfworld/grpo/4bi/none/seed=1", "ENTRY=dapo", "OVERLONG_BUFFER_LEN=7000"
    )
    assert rc == 0, err
    assert module == DAPO_ENTRY
    assert "trainer.algorithm.overlong_buffer.len=7000" in argv
    assert "trainer.algorithm.overlong_buffer.penalty_factor=1.0" in argv
    # the ALFWorld reward span, so the published factor keeps DAPO's proportional pressure
    assert "trainer.algorithm.overlong_buffer.reward_range=10.0" in argv


def test_run_script_refuses_the_buffer_on_the_opsd_trainer(tmp_path):
    """The OPSD trainer reads overlong_buffer.* into its config and never applies it, so a buffer on the OPSD
    entry would be the silent no-op this file exists to prevent: the launcher refuses before any GPU time."""
    rc, module, argv, err = _dry_run(tmp_path, "alfworld/grpo/4bi/none/seed=1", "OVERLONG_BUFFER_LEN=7000")
    assert rc != 0 and module is None
    assert "ENTRY=dapo" in err


def test_paper_arms_run_the_opsd_entrypoint_without_the_buffer(tmp_path):
    rc, module, argv, err = _dry_run(tmp_path, "alfworld/grpo/4bi/none/seed=1")
    assert rc == 0, err
    assert module == OPSD_ENTRY
    assert not [a for a in argv if "overlong_buffer" in a]

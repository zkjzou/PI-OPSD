"""Tests for the two env-anchored OPSD mechanisms ported from the AppWorld sweep.

  beta   (OPSD_DISTILL_LOSS_COEF) -- teacher-KL as its own LOSS term in the policy worker
  lambda (OPSD_MIX_LAMBDA)        -- A = lam*A_GRPO + (1-lam)*A_SDPO in advantage space

Both are OFF by default, so the assertions below double as a guarantee that every previously-submitted
OPSD arm is unaffected. The wiring is what these test, not the KL maths: the failure modes recorded in the
AppWorld sweep were all wiring -- a beta arm silently training as plain Dr. GRPO because the teacher tensor
never reached the worker, and the lambda arms crash-looping on an AttributeError for a coef=0 run.

beta lives in the OPSD policy worker (opsd.skyrl_ext.policy_worker, SkyRL itself is unmodified), so its tests
run that worker on the CPU stub harness of opsd/skyrl_ext/tests/test_policy_worker.py instead of reading
SkyRL's source.
"""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import opsd
from opsd._repo import repo_root
from opsd.algorithms.distill_loss import LossKnobs
from opsd.skyrl_ext.policy_worker import OPSDFSDPPolicyWorkerBase
from opsd.skyrl_ext.tests.test_policy_worker import (
    load_inputs,
    make_batch,
    make_cfg,
    make_worker,
    run_forward_backward,
)

PKG = Path(opsd.__file__).resolve().parent
TRAINER = PKG / "trainer/main_opsd_alfworld.py"
MIX_SCHEDULE = PKG / "trainer/mix_schedule.py"
BETA_KEYS = ("distill_loss_coef", "distill_loss_attached", "distill_loss_kl")
SDAR_KEYS = ("sdar_lambda", "sdar_attached", "sdar_loss", "sdar_gate_active", "sdar_delta_mean")

pytestmark = pytest.mark.filterwarnings("ignore:CUDA is not available or torch_xla is imported")


def _launch_run():
    """The ALFWorld launcher: its guards are checked on the argv it prints under DRY_RUN=1."""
    return repo_root(__file__) / "scripts/alfworld/run.sh"


def run_worker(knobs, teacher="teacher", **algo):
    """One forward_backward of the OPSD policy worker on the stub harness (5 rows, micro batch 2)."""
    inputs = load_inputs()
    cfg = make_cfg({"use_kl_loss": True, "use_entropy_loss": False, **algo})
    worker = make_worker(OPSDFSDPPolicyWorkerBase, cfg, inputs["W0"], knobs=knobs)
    return run_forward_backward(worker, make_batch(inputs, teacher))


def _src(p):
    return p.read_text()


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
        ["bash", str(_launch_run()), *args], env=env, cwd=str(tmp_path), capture_output=True, text=True, timeout=300
    )
    argv, lines = [], proc.stdout.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("DRY_RUN: ")), None)
    if start is not None:
        for ln in lines[start + 1 :]:
            if not ln.strip():
                break
            argv += shlex.split(ln.rstrip().removesuffix(" \\"))
    return proc.returncode, argv, proc.stderr


# --------------------------------------------------------------------------- defaults are OFF
def test_both_mechanisms_default_off():
    t = _src(TRAINER)
    assert 'os.environ.get("OPSD_DISTILL_LOSS_COEF", "0")' in t, "beta must default to 0"
    # lam is read through the step schedule (mix_schedule.py) since the AppWorld port, so the
    # default lives there -- and the trainer must go through it rather than read the env var raw, or a
    # scheduled arm would silently run at its start value for the whole run.
    assert '_read("OPSD_MIX_LAMBDA", 1.0)' in _src(MIX_SCHEDULE), "lambda must default to 1.0 (= no mixing)"
    assert "mix_lambda(" in t and 'os.environ.get("OPSD_MIX_LAMBDA"' not in t
    # the policy worker's knobs: resolved on the driver, OFF unless set
    assert LossKnobs.from_env({}) == LossKnobs() and not LossKnobs().active
    assert LossKnobs().distill_coef == 0.0


def test_lambda_schedule_defaults_to_no_mixing(monkeypatch):
    """Behavioural twin of the text check above: with nothing set, lam is 1.0 at every step and the mix
    (and its teacher-tensor transport) is off -- so every arm that predates the schedule is unaffected."""
    from opsd.trainer.mix_schedule import mix_lambda, mix_lambda_configured

    for k in ("OPSD_MIX_LAMBDA", "OPSD_MIX_LAMBDA_FINAL", "OPSD_MIX_LAMBDA_DECAY_STEPS", "OPSD_MIX_LAMBDA_AFTER"):
        monkeypatch.delenv(k, raising=False)
    assert [mix_lambda(s) for s in (0, 1, 100, 199)] == [1.0] * 4
    assert mix_lambda_configured() is False


def test_lambda_one_is_an_early_return():
    """lam=1.0 must not touch advantages at all -- not merely compute a no-op mix."""
    t = _src(TRAINER)
    body = t[t.index("def compute_advantages_and_returns") :]
    assert "if lam >= 1.0 or teacher_lp is None:" in body
    early = body.index("if lam >= 1.0 or teacher_lp is None:")
    mix = body.index('data["advantages"] = mixed')
    assert early < mix, "the early return must precede any mutation of advantages"


# --------------------------------------------------------------------------- the double-count trap
def test_lambda_warns_when_kl_is_also_in_the_reward():
    body = _src(TRAINER)
    assert "use_kl_in_reward" in body and "counted twice" in body


def test_run_script_forces_kl_out_of_the_reward_under_lambda(tmp_path):
    """A warning inside a 10-hour job is not a control; the launcher must correct it before GPU time."""
    rc, argv, err = _dry_run(tmp_path, "alfworld/opsd_grpo/4bi/glm_ao/lambda=0.5", "USE_KL_IN_REWARD=true")
    assert rc == 0, err
    assert "OPSD GUARD" in err
    assert "trainer.algorithm.use_kl_in_reward=false" in argv
    # vanilla OPSD (lambda=1) keeps the teacher term in the reward
    rc, argv, err = _dry_run(tmp_path, "alfworld/main/4bi/glm_ao")
    assert rc == 0, err
    assert "trainer.algorithm.use_kl_in_reward=true" in argv and "OPSD GUARD" not in err


def test_run_script_refuses_two_advantage_rewriters(tmp_path):
    """RLSD and the lambda mix both rewrite the advantage: the launcher refuses the pair outright."""
    rc, argv, err = _dry_run(tmp_path, "alfworld/rlsd/4bi/glm_ao", "OPSD_MIX_LAMBDA=0.5")
    assert rc != 0 and "OPSD GUARD" in err and not argv


# --------------------------------------------------------------------------- beta wiring
def test_beta_tensor_reaches_the_worker_without_touching_experience():
    """REGRESSION (original): passing distill_log_probs as an Experience constructor kwarg made every policy
    update die with TypeError when the dataclass edit was absent from the image, crash-looping the job. The
    teacher now never rides on the Experience at all: upstream's BatchIterator builds it untouched, and the
    worker takes the tensor from its own chunk iterator -- yet the beta term sees it (attached=1)."""
    seen = []
    inputs = load_inputs()
    cfg = make_cfg({"use_kl_loss": False, "use_entropy_loss": False})
    worker = make_worker(OPSDFSDPPolicyWorkerBase, cfg, inputs["W0"], knobs=LossKnobs(distill_coef=0.1))
    orig = worker._forward_backward_micro

    def spy(experience, *a, **kw):
        seen.append(experience)
        return orig(experience, *a, **kw)

    worker._forward_backward_micro = spy
    result = run_forward_backward(worker, make_batch(inputs))
    assert seen and not any(hasattr(e, "distill_log_probs") for e in seen)
    assert [m["status"]["distill_loss_attached"] for m in result["micro"]] == [1.0, 1.0, 1.0]


def test_experience_has_no_teacher_field():
    """REGRESSION (original): to_device touching self.distill_log_probs raised AttributeError on every run that
    left beta off -- how the lambda arms crash-looped. SkyRL's Experience is upstream's and carries no such
    field, so to_device cannot touch it."""
    import dataclasses

    from skyrl.train.dataset.replay_buffer import Experience

    assert "distill_log_probs" not in {f.name for f in dataclasses.fields(Experience)}


def test_beta_term_enters_the_loss_not_the_reward():
    """The term is summed INTO the loss handed to backward (pathwise gradient), on every micro-batch:
    loss(beta) - loss(0) == beta * distill_loss_kl * microbatch_weight, and the gradient changes."""
    base = run_worker(LossKnobs())
    beta = run_worker(LossKnobs(distill_coef=0.1))
    for b, s, m in zip(base["backward_losses"], beta["backward_losses"], beta["micro"]):
        assert m["status"]["distill_loss_kl"] > 0
        expected = 0.1 * m["status"]["distill_loss_kl"] * m["microbatch_weight"]
        assert s - b == pytest.approx(expected, rel=1e-4, abs=1e-6)
    assert not torch.equal(base["grad_W"], beta["grad_W"])
    # the reward/advantage path is untouched: the policy loss itself is identical with and without beta
    assert [m["status"]["policy_loss"] for m in base["micro"]] == [m["status"]["policy_loss"] for m in beta["micro"]]


def test_beta_is_independent_of_use_kl_loss():
    """The theta0 leash and the teacher term are separate knobs; beta must not be gated on use_kl_loss."""
    contrib = {}
    for kl in (False, True):
        base = run_worker(LossKnobs(), use_kl_loss=kl)
        beta = run_worker(LossKnobs(distill_coef=0.1), use_kl_loss=kl)
        contrib[kl] = [s - b for b, s in zip(base["backward_losses"], beta["backward_losses"])]
        assert all(m["status"]["distill_loss_attached"] == 1.0 for m in beta["micro"])
    assert contrib[False] == pytest.approx(contrib[True], rel=1e-4, abs=1e-6)
    assert all(c > 0 for c in contrib[False])


# --------------------------------------------------------------------------- observability
def test_attachment_is_logged_whenever_beta_is_requested():
    """The diagnostic must fire on REQUEST (coef != 0), not on success.

    Gating on success would hide the failure it exists to catch -- a beta arm silently training as plain
    Dr. GRPO is otherwise indistinguishable on every panel from one that works. Gating on the request
    keeps that, while sparing every unrelated run three constant-zero panels.
    """
    missing = run_worker(LossKnobs(distill_coef=0.1), teacher=None, use_kl_loss=False)
    for m in missing["micro"]:
        assert all(k in m["status"] for k in BETA_KEYS)
        assert m["status"]["distill_loss_attached"] == 0.0 and m["status"]["distill_loss_kl"] == 0.0
        assert m["status"]["distill_loss_coef"] == 0.1
        # and still outside the use_kl_loss branch (the theta0 leash is a separate knob)
        assert "policy_kl" not in m["status"]
    off = run_worker(LossKnobs(), use_kl_loss=True)
    assert not any(k in m["status"] for m in off["micro"] for k in BETA_KEYS)
    assert all("policy_kl" in m["status"] for m in off["micro"])


def test_sdar_metrics_are_gated_on_request():
    """SDAR is off (lambda=0) on every OPSD/GRPO arm we run, so its five keys must not appear there."""
    beta_only = run_worker(LossKnobs(distill_coef=0.1))
    assert not any(k in m["status"] for m in beta_only["micro"] for k in SDAR_KEYS)
    sdar = run_worker(LossKnobs(sdar_lambda=0.5))
    assert all(k in m["status"] for m in sdar["micro"] for k in SDAR_KEYS)


def test_num_turns_is_threaded_for_the_nstep_reference():
    """ndcg_at_4_neg_nstep needs per-rollout turn counts, and env_metrics never reaches the
    TrainingInputBatch -- so it must be captured in convert_to_training_input or the key is silently
    absent, which is exactly what happened on the first analyzer arms."""
    src = TRAINER.read_text()
    assert "_last_env_num_turns" in src
    conv = src[src.index("def convert_to_training_input") : src.index("def apply_reward_kl_penalty")]
    assert 'generator_output.get("env_metrics")' in conv
    assert "num_turns" in conv


def test_mix_metrics_expose_the_effective_ratio():
    """lam alone does not say what the run optimised: the two advantages are unnormalised and on different
    scales (AppWorld lam=0.5 ran at grpo_over_sdpo 0.30 -> 0.83)."""
    t = _src(TRAINER)
    for k in ("mix/lambda", "mix/a_grpo_abs", "mix/a_sdpo_abs", "mix/grpo_over_sdpo", "mix/contrib_sdpo"):
        assert f'"{k}"' in t


def test_teacher_key_is_separate_from_the_reward_path_key():
    """hinted_log_probs is popped by apply_reward_kl_penalty before advantages run, so the mix needs its
    own key or it would read a missing tensor whenever use_kl_in_reward is on."""
    t = _src(TRAINER)
    assert 'training_input["mix_teacher_log_probs"]' in t
    assert 'data.pop("mix_teacher_log_probs", None)' in t, "must be dropped before the policy worker"


# --------------------------------------------------------------------------- pi_utility coverage
def test_pi_utility_runs_before_every_mechanism_early_return():
    """REGRESSION: pi_utility/* must fire for EVERY arm with a teacher forward, not just lambda arms.

    It originally sat after two early returns in compute_advantages_and_returns -- the RLSD branch's
    `return data` and the `lam >= 1.0` return -- so of the four env-anchored arms only l09ema (lam=0.9)
    logged the metrics. b1e4/b5e4 (lam=1.0, no mix key) and rlsd (returns in its own branch) silently
    logged nothing, which looked like a property of those arms rather than a wiring bug.
    """
    src = TRAINER.read_text()
    body = src[src.index("def compute_advantages_and_returns") :]
    call = body.index("_pi_utility_metrics")
    # every `return data` inside this method must come AFTER the metrics call
    for m in re.finditer(r"^\s+return data\b", body[: body.index("\n    def ", 1)], re.M):
        assert m.start() > call, "a `return data` precedes the pi_utility call -- some arms will log nothing"


def test_pi_utility_sources_the_unconditional_teacher_key():
    """It must read `hinted_log_probs` (published unconditionally, popped only by apply_reward_kl_penalty
    which the env-anchored arms never run) rather than depending on `mix_teacher_log_probs`, which exists
    only when lam<1 or RLSD is on."""
    src = TRAINER.read_text()
    body = src[src.index("def compute_advantages_and_returns") :]
    pre = body[: body.index("_pi_utility_metrics")]
    assert 'data.get("hinted_log_probs")' in pre
    assert 'data.get("mix_teacher_log_probs")' in pre, "fallback for the use_kl_in_reward=true arms missing"


def test_pi_utility_is_computed_at_most_once_per_step():
    src = TRAINER.read_text()
    assert src.count('if "pi_utility/d" not in self.all_metrics:') == 1, "duplicate guarded call sites"

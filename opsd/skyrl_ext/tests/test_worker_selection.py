"""CPU tests of OPSDWorkerMixin.build_models (opsd.skyrl_ext.worker_selection).

``RayPPOTrainer.build_models`` is replaced by a recording base class, so nothing here touches Ray, GPUs or
torch.distributed: the tests pin which policy actor class reaches SkyRL, the Megatron guard, the eval-only
step clamp and the knob push.
"""

import inspect
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
import ray

from opsd.algorithms.distill_loss import LossKnobs
from opsd.skyrl_ext.policy_worker import OPSDFSDPPolicyWorker
from opsd.skyrl_ext.worker_selection import OPSDWorkerMixin
from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker
from skyrl.train.trainer import RayPPOTrainer

OPSD_KEYS = ("OPSD_DISTILL_LOSS_COEF", "OPSD_SDAR_LAMBDA", "OPSD_SDAR_BETA", "OPSD_SDAR_GATE")
MEGATRON_WORKERS = (object(), object(), object())  # megatron-core is not installed on CPU boxes; sentinels


class FakeActorGroup:
    def __init__(self, log):
        self.log = log

    def async_run_ray_method(self, dispatch_type, method_name, *args, **kwargs):
        self.log.append(("push", dispatch_type, method_name, args, kwargs))
        return [f"ref:{method_name}"]


class RecordingTrainer:
    """Stands in for RayPPOTrainer: records what build_models receives and the step count it sees."""

    def __init__(self, strategy="fsdp", total_training_steps=10, fail=False):
        self.cfg = SimpleNamespace(trainer=SimpleNamespace(strategy=strategy))
        self.total_training_steps = total_training_steps
        self.log = []
        self.fail = fail

    def build_models(self, PolicyWorker, CriticWorker, RefWorker):
        self.log.append(("build_models", PolicyWorker, CriticWorker, RefWorker, self.total_training_steps))
        if self.fail:
            raise RuntimeError("init_model failed")
        self.policy_model = FakeActorGroup(self.log)


class Trainer(OPSDWorkerMixin, RecordingTrainer):
    pass


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in OPSD_KEYS:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def ray_get(monkeypatch):
    got = []

    def fake_get(refs):
        got.append(refs)
        return [None] * len(refs)

    monkeypatch.setattr(ray, "get", fake_get)
    return got


def fsdp_workers():
    return fsdp_worker.PolicyWorker, fsdp_worker.CriticWorker, fsdp_worker.RefWorker


def built(trainer):
    return [e for e in trainer.log if e[0] == "build_models"]


def pushes(trainer):
    return [e for e in trainer.log if e[0] == "push"]


def test_upstream_build_models_signature_is_the_seam():
    assert list(inspect.signature(RayPPOTrainer.build_models).parameters) == [
        "self",
        "PolicyWorker",
        "CriticWorker",
        "RefWorker",
    ]


def test_knobs_off_keeps_every_upstream_class(ray_get):
    t = Trainer()
    t.build_models(*fsdp_workers())
    ((_, policy, critic, ref, _),) = built(t)
    assert (policy, critic, ref) == fsdp_workers()
    assert pushes(t) == [] and ray_get == []
    assert t.opsd_loss_knobs == LossKnobs()


@pytest.mark.parametrize("zero", ["0", "0.0", ""])
def test_explicit_zero_knobs_keep_upstream(monkeypatch, ray_get, zero):
    monkeypatch.setenv("OPSD_DISTILL_LOSS_COEF", zero)
    monkeypatch.setenv("OPSD_SDAR_LAMBDA", zero)
    t = Trainer()
    t.build_models(*fsdp_workers())
    assert built(t)[0][1] is fsdp_worker.PolicyWorker and pushes(t) == []


@pytest.mark.parametrize(
    "env",
    [
        {"OPSD_DISTILL_LOSS_COEF": "0.1"},
        {"OPSD_DISTILL_LOSS_COEF": "1e-4"},
        {"OPSD_SDAR_LAMBDA": "0.5"},
        {"OPSD_DISTILL_LOSS_COEF": "0.1", "OPSD_SDAR_LAMBDA": "0.5", "OPSD_SDAR_GATE": "soft-or"},
    ],
)
def test_active_knobs_swap_the_policy_worker_and_push(monkeypatch, ray_get, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    t = Trainer()
    t.build_models(*fsdp_workers())
    ((_, policy, critic, ref, _),) = built(t)
    assert policy is OPSDFSDPPolicyWorker
    assert (critic, ref) == fsdp_workers()[1:], "only the policy worker is swapped"
    expected = LossKnobs.from_env(env)
    assert t.opsd_loss_knobs == expected
    # pushed AFTER the actors exist (the push is logged after build_models), through pass_through
    assert [e[0] for e in t.log] == ["build_models", "push"]
    assert pushes(t) == [("push", "pass_through", "set_opsd_loss_knobs", (expected,), {})]
    assert ray_get == [["ref:set_opsd_loss_knobs"]], "the driver must wait for the push"


def test_swap_does_not_depend_on_the_trainer_class(monkeypatch, ray_get):
    """Keyed on the knobs only: any trainer carrying the mixin (even one that never ships distill_log_probs)
    gets the OPSD worker, so the worker can still report distill_loss_attached=0."""

    class SomeOtherTrainer(OPSDWorkerMixin, RecordingTrainer):
        pass

    monkeypatch.setenv("OPSD_DISTILL_LOSS_COEF", "0.25")
    t = SomeOtherTrainer()
    t.build_models(*fsdp_workers())
    assert built(t)[0][1] is OPSDFSDPPolicyWorker


@pytest.mark.parametrize("env", [{"OPSD_DISTILL_LOSS_COEF": "0.1"}, {"OPSD_SDAR_LAMBDA": "0.5"}])
def test_megatron_with_knobs_raises_before_building(monkeypatch, ray_get, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    t = Trainer(strategy="megatron")
    with pytest.raises(ValueError, match="need the FSDP policy worker"):
        t.build_models(*MEGATRON_WORKERS)
    assert built(t) == [] and pushes(t) == []


def test_unknown_strategy_with_knobs_raises(monkeypatch, ray_get):
    monkeypatch.setenv("OPSD_DISTILL_LOSS_COEF", "0.1")
    with pytest.raises(ValueError, match="strategy=fsdp only"):
        Trainer(strategy="fsdp2").build_models(*fsdp_workers())


def test_custom_policy_worker_is_not_silently_replaced(monkeypatch, ray_get):
    monkeypatch.setenv("OPSD_DISTILL_LOSS_COEF", "0.1")
    custom = ray.remote(num_gpus=1)(type("CustomPolicyWorker", (fsdp_worker.FSDPPolicyWorkerBase,), {}))
    with pytest.raises(TypeError, match="not SkyRL's fsdp_worker.PolicyWorker"):
        Trainer().build_models(custom, *fsdp_workers()[1:])


def test_bogus_sdar_gate_fails_at_startup(monkeypatch, ray_get):
    """With lambda != 0 a typo'd gate raises on the driver, before any actor is built."""
    monkeypatch.setenv("OPSD_SDAR_LAMBDA", "0.5")
    monkeypatch.setenv("OPSD_SDAR_GATE", "gapp")
    t = Trainer()
    with pytest.raises(ValueError, match=re.escape("OPSD_SDAR_GATE must be gap|entropy|soft_or")):
        t.build_models(*fsdp_workers())
    assert built(t) == []


def test_bogus_sdar_gate_is_harmless_when_sdar_is_off(monkeypatch, ray_get):
    """As in the fork: the gate is only read when lambda != 0."""
    monkeypatch.setenv("OPSD_SDAR_GATE", "gapp")
    t = Trainer()
    t.build_models(*fsdp_workers())
    assert built(t)[0][1] is fsdp_worker.PolicyWorker


# --------------------------------------------------------------------------- Megatron eval-only step clamp
def test_megatron_eval_only_clamps_steps_and_restores(ray_get):
    t = Trainer(strategy="megatron", total_training_steps=0)
    t.build_models(*MEGATRON_WORKERS)
    assert built(t)[0][4] == 1, "build_models must see >= 1 training step on a Megatron eval-only run"
    assert t.total_training_steps == 0, "restored afterwards"
    assert built(t)[0][1:4] == MEGATRON_WORKERS


def test_clamp_is_restored_when_build_models_raises(ray_get):
    t = Trainer(strategy="megatron", total_training_steps=0, fail=True)
    with pytest.raises(RuntimeError, match="init_model failed"):
        t.build_models(*MEGATRON_WORKERS)
    assert t.total_training_steps == 0


@pytest.mark.parametrize(
    "strategy,steps,seen",
    [
        ("megatron", 7, 7),  # training runs are untouched
        ("megatron", None, None),  # unknown step count: upstream's behaviour, not ours to change
        ("fsdp", 0, 0),  # the scheduler assertion is Megatron's only
    ],
)
def test_no_clamp_outside_megatron_eval_only(ray_get, strategy, steps, seen):
    t = Trainer(strategy=strategy, total_training_steps=steps)
    t.build_models(*(MEGATRON_WORKERS if strategy == "megatron" else fsdp_workers()))
    assert built(t)[0][4] == seen and t.total_training_steps == steps


def test_mixin_is_cooperative():
    """Mixed in ahead of RayPPOTrainer next to other mixins: every build_models in the MRO runs once."""
    calls = []

    class OuterMixin:
        def build_models(self, *workers):
            calls.append("outer")
            return super().build_models(*workers)

    class T(OuterMixin, OPSDWorkerMixin, RecordingTrainer):
        pass

    t = T()
    t.build_models(*fsdp_workers())
    assert calls == ["outer"] and len(built(t)) == 1


def test_worker_selection_does_not_import_the_actor_module():
    """The trainer imports worker_selection on every run; the OPSD actor module loads only when it is swapped in."""
    code = "import sys, opsd.skyrl_ext.worker_selection; print('opsd.skyrl_ext.policy_worker' in sys.modules)"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "False"


def test_opsd_base_trainer_resolves_build_models_to_the_mixin():
    """Every OPSD trainer derives from OPSDRayPPOTrainer; its build_models must be this mixin's (nothing earlier
    in the MRO may shadow it), and the mixin must sit ahead of RayPPOTrainer so super() reaches upstream."""
    from opsd.skyrl_ext.trainer import OPSDRayPPOTrainer

    assert OPSDRayPPOTrainer.build_models is OPSDWorkerMixin.build_models
    mro = OPSDRayPPOTrainer.__mro__
    assert mro.index(OPSDWorkerMixin) < mro.index(RayPPOTrainer)

"""CPU tests of OPSDFSDPPolicyWorkerBase (opsd.skyrl_ext.policy_worker), plus the stub harness the other worker
tests share (test_policy_worker_parity.py, opsd/algorithms/tests/test_opsd_env_anchor.py, test_opsd_sdar.py).

The harness drives the REAL ``forward_backward`` -> ``BatchIterator`` -> ``_forward_backward_micro`` path with only
the model, the strategy and the mesh stubbed:

  * ``StubModel`` stands in for ``HFModelWrapper``: log-probs of the sampled tokens and the full-sequence entropy,
    both computed from one parameter ``W`` (rows x seq x vocab), so the loss's gradient lands in ``W.grad``;
  * ``StubStrategy.backward`` records the exact loss tensor value and calls ``loss.backward()``;
  * ``torch.cuda.current_device`` is patched to the CPU.

The stubs are byte-for-byte the ones the fork goldens were captured with (see
opsd/tests/fixtures/fork_parity/worker/README.md); change them only together with a re-capture.
"""

import copy
import json
import math
import subprocess
import sys
from unittest import mock

import pytest
import torch

from opsd._repo import fixtures_dir
from opsd.algorithms.distill_loss import LossKnobs
from opsd.skyrl_ext.policy_worker import OPSDFSDPPolicyWorkerBase
from skyrl.backends.skyrl_train.distributed.dispatch import MeshRank
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.ppo_utils import PolicyLossRegistry
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase
from skyrl.train.config import SkyRLTrainConfig

FIXTURES = fixtures_dir() / "fork_parity/worker"

# torch.autocast(device_type="cuda") on a CPU-only box disables itself with this warning.
pytestmark = pytest.mark.filterwarnings("ignore:CUDA is not available or torch_xla is imported")


# --------------------------------------------------------------------------- fixtures (JSON tensors)
def _dec_float(x):
    return float(x) if isinstance(x, str) else x


def dec(t: dict) -> torch.Tensor:
    def walk(v):
        return [walk(u) for u in v] if isinstance(v, list) else _dec_float(v)

    return torch.tensor(walk(t["data"]), dtype=getattr(torch, t["dtype"])).reshape(t["shape"])


def load_inputs() -> dict:
    raw = json.loads((FIXTURES / "inputs.json").read_text())
    return {"algo_base": raw["algo_base"], **{k: dec(v) for k, v in raw["tensors"].items()}}


def load_goldens() -> list:
    return json.loads((FIXTURES / "goldens.json").read_text())["cases"]


# --------------------------------------------------------------------------- stubs (same as the capture)
class StubModel(torch.nn.Module):
    """``HFModelWrapper.forward`` stand-in: log-probs of the sampled tokens + full-sequence entropy.

    Row ``i`` of the batch is identified by ``sequences[:, 0]`` so every micro-batch reads its own slice of W.
    """

    def __init__(self, w0: torch.Tensor):
        super().__init__()
        self.W = torch.nn.Parameter(w0.clone())

    def forward(
        self,
        sequences,
        num_actions,
        attention_mask=None,
        temperature=1.0,
        return_output=False,
        compute_entropy=False,
        entropy_requires_grad=True,
        pixel_values=None,
        image_grid_thw=None,
    ):
        rows = sequences[:, 0].long()
        logits = self.W[rows] / temperature
        logp_all = torch.log_softmax(logits, dim=-1)
        tokens = sequences[:, -num_actions:].long()
        action_log_probs = logp_all[:, -num_actions - 1 : -1].gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        with torch.set_grad_enabled(entropy_requires_grad):
            entropy = -(logp_all.exp() * logp_all).sum(-1)
        return action_log_probs, {"entropy": entropy}


class StubStrategy:
    def __init__(self):
        self.losses = []

    def backward(self, loss, model, optimizer, **kwargs):
        self.losses.append(loss.detach().item())
        loss.backward()

    def all_reduce(self, data, op="mean", group=None):
        return data


class StubScheduler:
    def get_last_lr(self):
        return [1e-6]


class StubMesh:
    def get_group(self, name):
        return None


def make_cfg(algo: dict, micro: int = 2, algo_base: dict = None):
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "fsdp"
    cfg.trainer.micro_train_batch_size_per_gpu = micro
    for k, v in {**(algo_base or load_inputs()["algo_base"]), **algo}.items():
        setattr(cfg.trainer.algorithm, k, v)
    return cfg.trainer


def make_worker(cls, cfg_trainer, w0, dp_size=2, knobs=None):
    """A worker instance with the attributes ``init_model`` would set, built without Ray or torch.distributed."""
    worker = object.__new__(cls)
    worker.cfg = cfg_trainer
    worker.model = StubModel(w0)
    worker.strategy = StubStrategy()
    worker.scheduler = StubScheduler()
    worker.optimizer = None
    worker.record_memory = False
    worker.device_mesh = StubMesh()
    worker.mesh_rank = MeshRank(dp=0, sp=0, tp=0, pp=0, world_size=dp_size, dp_size=dp_size, pp_size=1)
    worker.policy_loss_fn = PolicyLossRegistry.get(cfg_trainer.algorithm.policy_loss_type)
    if isinstance(worker, OPSDFSDPPolicyWorkerBase):
        # what OPSDFSDPPolicyWorkerBase.__init__ sets, then the build_models push
        worker._opsd_loss_knobs = None
        worker._opsd_teacher_chunks = None
        if knobs is not None:
            worker.set_opsd_loss_knobs(knobs)
    return worker


def make_batch(inputs: dict, teacher_key="teacher") -> TrainingInputBatch:
    d = {k: inputs[k].clone() for k in ("sequences", "attention_mask", "action_log_probs")}
    d.update({k: inputs[k].clone() for k in ("base_action_log_probs", "advantages", "loss_mask", "response_mask")})
    d["rollout_logprobs"] = None
    if teacher_key is not None:
        d["distill_log_probs"] = inputs[teacher_key].clone()
    batch = TrainingInputBatch(d)
    batch.metadata = {"global_step": 0, "response_length": inputs["loss_mask"].shape[1]}
    return batch


@pytest.fixture
def cpu_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))


def run_forward_backward(worker, batch, loss_fn=None, loss_fn_config=None) -> dict:
    """``worker.forward_backward`` on the CPU -> the same record the fork goldens hold."""
    micro_status = []
    orig = worker._forward_backward_micro

    def recording(experience, microbatch_weight, loss_fn=None, loss_fn_config=None):
        status = orig(experience, microbatch_weight, loss_fn=loss_fn, loss_fn_config=loss_fn_config)
        micro_status.append(
            {
                "microbatch_weight": microbatch_weight,
                "rows": int(experience.sequences.shape[0]),
                "status": {k: copy.deepcopy(v) for k, v in status.items() if k != "loss_fn_outputs"},
            }
        )
        return status

    worker._forward_backward_micro = recording
    with mock.patch("torch.cuda.current_device", return_value=torch.device("cpu")):
        out = worker.forward_backward(batch, loss_fn=loss_fn, loss_fn_config=loss_fn_config)
    grad = worker.model.W.grad
    return {
        "backward_losses": worker.strategy.losses,
        "micro": micro_status,
        "metrics": dict(out.metrics),
        "loss_fn_outputs": out.loss_fn_outputs,
        "grad_W": grad,
    }


def micro_metrics(result) -> list:
    return [m["status"] for m in result["micro"]]


# --------------------------------------------------------------------------- tests
@pytest.fixture(scope="module")
def inputs():
    return load_inputs()


def test_knobs_off_falls_through_bit_exactly_to_upstream(inputs, cpu_device):
    """beta = lambda = 0 (teacher tensor present): the OPSD worker IS upstream's method, bit for bit."""
    cfg = make_cfg(dict(use_kl_loss=True, use_entropy_loss=True))
    up = run_forward_backward(make_worker(FSDPPolicyWorkerBase, cfg, inputs["W0"]), make_batch(inputs))
    ours = run_forward_backward(
        make_worker(OPSDFSDPPolicyWorkerBase, cfg, inputs["W0"], knobs=LossKnobs()), make_batch(inputs)
    )
    assert ours["backward_losses"] == up["backward_losses"]
    assert micro_metrics(ours) == micro_metrics(up)
    assert ours["metrics"] == up["metrics"]
    assert torch.equal(ours["grad_W"], up["grad_W"])


def test_copied_branch_with_zero_terms_matches_upstream(inputs, cpu_device):
    """Drive the COPIED RL branch with both terms at an exact 0: it must reproduce upstream (the fork added
    exact zeros). Catches a copy that drifted from upstream's text in a way that changes numbers."""
    cfg = make_cfg(dict(use_kl_loss=True, use_entropy_loss=True))
    up = run_forward_backward(make_worker(FSDPPolicyWorkerBase, cfg, inputs["W0"]), make_batch(inputs))
    worker = make_worker(OPSDFSDPPolicyWorkerBase, cfg, inputs["W0"], knobs=LossKnobs())
    forced = LossKnobs(distill_coef=0.0, sdar_lambda=0.0)

    def copied(experience, mbw, loss_fn=None, loss_fn_config=None):
        teacher = worker._next_teacher_chunk(experience)
        return worker._opsd_forward_backward_micro(experience, mbw, teacher, forced, loss_fn, loss_fn_config)

    worker._forward_backward_micro = copied
    ours = run_forward_backward(worker, make_batch(inputs))
    assert ours["backward_losses"] == up["backward_losses"]
    assert micro_metrics(ours) == micro_metrics(up)
    assert torch.equal(ours["grad_W"], up["grad_W"])


@pytest.mark.parametrize("micro", [1, 2, 3, 4, 5, 7])
def test_teacher_chunks_align_with_micro_batches(inputs, cpu_device, micro):
    """Batch of 5 at every micro size, incl. non-multiples: each micro-batch sees exactly its own teacher rows."""
    cfg = make_cfg(dict(use_kl_loss=False, use_entropy_loss=False), micro=micro)
    worker = make_worker(OPSDFSDPPolicyWorkerBase, cfg, inputs["W0"], knobs=LossKnobs(distill_coef=0.1))
    seen = []
    orig = worker._next_teacher_chunk

    def spy(experience):
        teacher = orig(experience)
        seen.append((experience.sequences[:, 0].tolist(), teacher.clone()))
        return teacher

    worker._next_teacher_chunk = spy
    run_forward_backward(worker, make_batch(inputs))
    assert [len(rows) for rows, _ in seen] == [min(micro, 5 - i) for i in range(0, 5, micro)]
    for rows, teacher in seen:
        assert torch.equal(teacher, inputs["teacher"][rows])
    assert worker._opsd_teacher_chunks is None, "the side iterator must be cleared after forward_backward"


def test_teacher_chunk_is_consumed_on_the_fall_through_path_too(inputs, cpu_device):
    """cross_entropy falls through to upstream; the chunk must still be taken so later calls stay aligned."""
    cfg = make_cfg(dict(use_kl_loss=False, use_entropy_loss=False))
    worker = make_worker(OPSDFSDPPolicyWorkerBase, cfg, inputs["W0"], knobs=LossKnobs(distill_coef=0.1))
    result = run_forward_backward(worker, make_batch(inputs), loss_fn="cross_entropy")
    assert len(result["micro"]) == 3
    assert not any("distill_loss_kl" in s for s in micro_metrics(result))


def test_teacher_row_count_mismatch_fails_loudly(inputs, cpu_device):
    cfg = make_cfg(dict(use_kl_loss=False, use_entropy_loss=False))
    worker = make_worker(OPSDFSDPPolicyWorkerBase, cfg, inputs["W0"], knobs=LossKnobs(distill_coef=0.1))
    batch = make_batch(inputs)
    # TensorBatch.__setitem__ itself rejects a wrong row count; bypass it to hand the worker an inconsistent batch
    dict.__setitem__(batch, "distill_log_probs", batch["distill_log_probs"][:4])
    with pytest.raises(AssertionError, match="rows for a batch of 5"):
        worker.forward_backward(batch)
    assert worker._opsd_teacher_chunks is None


def test_teacher_width_mismatch_fails_loudly(inputs, cpu_device):
    cfg = make_cfg(dict(use_kl_loss=False, use_entropy_loss=False))
    worker = make_worker(OPSDFSDPPolicyWorkerBase, cfg, inputs["W0"], knobs=LossKnobs(distill_coef=0.1))
    batch = make_batch(inputs)
    batch["distill_log_probs"] = torch.cat([batch["distill_log_probs"], batch["distill_log_probs"][:, :1]], dim=1)
    with pytest.raises(AssertionError, match="tokens wide"):
        worker.forward_backward(batch)


def test_leftover_teacher_chunks_fail_loudly(inputs, cpu_device):
    """If upstream ever skipped a micro-batch, the teacher would shift by one chunk: that must not pass silently."""
    cfg = make_cfg(dict(use_kl_loss=False, use_entropy_loss=False))
    worker = make_worker(OPSDFSDPPolicyWorkerBase, cfg, inputs["W0"], knobs=LossKnobs(distill_coef=0.1))
    calls = []
    orig = worker._forward_backward_micro

    def skip_last(experience, mbw, loss_fn=None, loss_fn_config=None):
        calls.append(1)
        if len(calls) == 3:
            return {}
        return orig(experience, mbw, loss_fn=loss_fn, loss_fn_config=loss_fn_config)

    worker._forward_backward_micro = skip_last
    with pytest.raises(AssertionError, match="left over"):
        worker.forward_backward(make_batch(inputs))


def test_unpushed_knobs_refuse_to_train(inputs, cpu_device):
    """The class is only built because a knob is set; running it without the push would be silent plain GRPO."""
    cfg = make_cfg(dict(use_kl_loss=False, use_entropy_loss=False))
    worker = make_worker(OPSDFSDPPolicyWorkerBase, cfg, inputs["W0"], knobs=None)
    with pytest.raises(RuntimeError, match="set_opsd_loss_knobs was never called"):
        worker.forward_backward(make_batch(inputs))


def test_set_opsd_loss_knobs_type_checks(inputs):
    worker = make_worker(OPSDFSDPPolicyWorkerBase, make_cfg({}), inputs["W0"])
    with pytest.raises(TypeError):
        worker.set_opsd_loss_knobs({"distill_coef": 0.1})
    worker.set_opsd_loss_knobs(LossKnobs(distill_coef=0.1))
    assert worker._opsd_loss_knobs == LossKnobs(distill_coef=0.1)


def test_init_starts_with_no_knobs_and_no_iterator(monkeypatch):
    """``__init__`` (as Ray calls it) leaves the knobs unset until the driver pushes them."""
    # DistributedTorchRayActor.__init__ writes these and redirects stdout when SKYRL_LOG_FILE is set
    for k in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.setenv(k, "unset-by-test")
    monkeypatch.delenv("SKYRL_LOG_FILE", raising=False)
    cfg = make_cfg({})
    worker = OPSDFSDPPolicyWorkerBase(
        cfg=cfg,
        world_size=1,
        rank=0,
        local_rank=0,
        master_addr="localhost",
        master_port=12345,
        sequence_parallel_size=1,
    )
    assert worker._opsd_loss_knobs is None and worker._opsd_teacher_chunks is None


def test_actor_class_is_module_level_ray_actor():
    """Ray pickles the actor class by reference, so it must live at module level in an importable module."""
    import ray

    from opsd.skyrl_ext import policy_worker

    assert isinstance(policy_worker.OPSDFSDPPolicyWorker, ray.actor.ActorClass)
    assert policy_worker.OPSDFSDPPolicyWorker.__ray_metadata__.modified_class.__module__ == policy_worker.__name__


def test_policy_worker_module_is_import_light():
    """Every policy actor imports this module: beyond what SkyRL's own fsdp_worker already loads in the actor, it
    may add only itself and opsd.algorithms.distill_loss (no envs, trainers, generators or recipes)."""
    code = (
        "import sys\n"
        "import skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker\n"
        "before = set(sys.modules)\n"
        "import opsd.skyrl_ext.policy_worker\n"
        "print(','.join(sorted(set(sys.modules) - before)))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    added = set(filter(None, proc.stdout.strip().splitlines()[-1].split(",")))
    allowed = {
        "opsd",
        "opsd.algorithms",
        "opsd.algorithms.distill_loss",
        "opsd.skyrl_ext",
        "opsd.skyrl_ext.policy_worker",
    }
    assert added <= allowed, f"policy_worker pulls in extra modules: {sorted(added - allowed)}"


def test_nan_free_on_real_inputs(inputs, cpu_device):
    cfg = make_cfg(dict(use_kl_loss=True, use_entropy_loss=True))
    knobs = LossKnobs(distill_coef=0.1, sdar_lambda=0.5, sdar_gate="soft_or")
    result = run_forward_backward(
        make_worker(OPSDFSDPPolicyWorkerBase, cfg, inputs["W0"], knobs=knobs), make_batch(inputs)
    )
    assert all(math.isfinite(x) for x in result["backward_losses"])
    assert torch.isfinite(result["grad_W"]).all()

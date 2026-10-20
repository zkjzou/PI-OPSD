"""Epoch-boundary resume parity: OPSDRayPPOTrainer (upstream train() + the load_checkpoints drain) vs the fork.

The fork fixed the epoch-boundary resume inside its edited ``RayPPOTrainer.train()`` (an epoch ceiling with one
epoch of slack plus ``global_step`` stop guards). The release keeps upstream ``train()`` and instead drains the
exhausted restored iterator in ``OPSDRayPPOTrainer.load_checkpoints``. The goldens
(opsd/tests/fixtures/fork_parity/trainer/resume_w{0,2}.json) are the fork's real ``train()`` driven without models on
a real torchdata ``StatefulDataLoader`` (11 items, batch 3 -> 3 batches/epoch, seed 42, num_workers 0 and 2):
a fresh 3-epoch run, then resumes from its checkpoints. Here the same harness drives ``OPSDRayPPOTrainer`` and
the results must match, batch for batch.

Intended differences, asserted explicitly (``INTENDED`` below):

* After an epoch-boundary resume the fork logs ``trainer/epoch`` one too high (its first resumed epoch is the
  empty pass), which also moves ``update_ref_every_epoch``'s "skip the last epoch" sync. The drain gives the
  correct epoch index; the trained batches are identical.
* ``trainer.epochs=0`` (eval-only) runs: the fork's one-epoch slack ran one stray training step whenever the
  loader still had data (no resume, or a mid-epoch checkpoint). Upstream runs none.
* Resuming from a ``global_step_{T+1}`` safety-net checkpoint into an extra epoch: the fork's step bound
  (``total_training_steps``) cut that epoch's last batch; upstream ``train()`` is bounded by epochs and trains the
  whole epoch. Same batch order; one more step at the end.
* Safety-net checkpoints: upstream's loop ends on ``StopIteration`` (``_iterator_finished=True``) where the
  fork's ended on its ``break``. Either resumes into a fresh epoch.
"""

import asyncio
import base64
import hashlib
import json
import os
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from opsd._repo import fixtures_dir
from opsd.skyrl_ext.evaluate import OPSDEvalMixin
from opsd.skyrl_ext.trainer import OPSDRayPPOTrainer, batches_consumed
from opsd.skyrl_ext.worker_selection import OPSDWorkerMixin
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils.trainer_utils import ResumeMode

FIXTURES = fixtures_dir() / "fork_parity" / "trainer"
GOLDEN = {w: json.loads((FIXTURES / f"resume_w{w}.json").read_text()) for w in (0, 2)}
SCENARIOS = sorted(GOLDEN[0]["scenarios"])
N_ITEMS, BATCH, SEED, EPOCHS = 11, 3, 42, 3  # must match the capture (setup block of the goldens)


# --- the capture's tagged-JSON codec for dataloader states ----------------------------------------------------
def encode(obj):
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().contiguous()
        return {
            "__tensor__": str(t.dtype).removeprefix("torch."),
            "shape": list(t.shape),
            "b64": base64.b64encode(t.numpy().tobytes()).decode("ascii"),
        }
    if isinstance(obj, dict):
        if all(isinstance(k, str) and k not in ("__tensor__", "__tuple__", "__dict__") for k in obj):
            return {k: encode(v) for k, v in obj.items()}
        return {"__dict__": [[encode(k), encode(v)] for k, v in obj.items()]}
    if isinstance(obj, tuple):
        return {"__tuple__": [encode(v) for v in obj]}
    if isinstance(obj, list):
        return [encode(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"cannot encode {type(obj)}")


def decode(obj):
    if isinstance(obj, list):
        return [decode(v) for v in obj]
    if isinstance(obj, dict):
        if "__tensor__" in obj:
            arr = np.frombuffer(base64.b64decode(obj["b64"]), dtype=getattr(np, obj["__tensor__"]))
            return torch.from_numpy(arr.copy()).reshape(obj["shape"])
        if "__tuple__" in obj:
            return tuple(decode(v) for v in obj["__tuple__"])
        if "__dict__" in obj:
            return {decode(k): decode(v) for k, v in obj["__dict__"]}
        return {k: decode(v) for k, v in obj.items()}
    return obj


def state_sha(state) -> str:
    return hashlib.sha256(json.dumps(encode(state), sort_keys=True).encode()).hexdigest()


# --- the capture's harness: upstream train() with every model / generator / Ray touchpoint stubbed -------------
def make_dataloader(num_workers: int) -> StatefulDataLoader:
    # Mirrors skyrl.train.utils.trainer_utils.build_dataloader(is_train=True) except num_workers / context.
    g = torch.Generator()
    g.manual_seed(SEED)
    items = [
        {"prompt": [{"role": "user", "content": f"task {i}"}], "env_class": None, "env_extras": {}, "uid": f"u{i:02d}"}
        for i in range(N_ITEMS)
    ]
    return StatefulDataLoader(
        items,
        batch_size=BATCH,
        shuffle=True,
        collate_fn=list,
        num_workers=num_workers,
        drop_last=True,
        generator=g,
        multiprocessing_context="fork" if num_workers else None,
    )


def make_cfg(epochs, ckpt_path, resume_mode, resume_path) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    tr = cfg.trainer
    tr.epochs = epochs
    tr.train_batch_size = BATCH
    tr.seed = SEED
    tr.ckpt_interval = 2
    tr.hf_save_interval = -1
    tr.max_ckpts_to_keep = -1
    tr.ckpt_path = ckpt_path
    tr.resume_mode = resume_mode
    tr.resume_path = resume_path
    tr.eval_interval = 4
    tr.eval_before_train = True
    tr.update_ref_every_epoch = True
    tr.log_example_interval = 0
    tr.dump_data_batch = False
    tr.algorithm.use_kl_in_reward = False
    tr.algorithm.dynamic_sampling.type = None
    cfg.generator.n_samples_per_prompt = 1
    cfg.generator.step_wise_trajectories = False
    cfg.environment.env_class = "toy"
    return cfg


class _FakeDispatch:
    async def save_weights_for_sampler(self):
        pass

    def get_lcm_dp_size(self):
        return 1

    def save_checkpoint(self, *args, **kwargs):
        pass

    def load_checkpoint(self, *args, **kwargs):
        pass


class _FakeTracker:
    def __init__(self):
        self.logs = []

    def log(self, data, step, commit=False):
        self.logs.append((step, dict(data)))

    def finish(self):
        pass


class _Batch(dict):
    """What train() touches on a TrainingInputBatch between compute_advantages_and_returns and training."""

    def __init__(self, uids):
        super().__init__(rewards=None)
        self.metadata = {"uids": list(uids)}
        self.batch_uids = list(uids)


def run_train(cfg, dataloader):
    t = object.__new__(OPSDRayPPOTrainer)
    t.cfg = cfg
    t.colocate_all = False
    t.tracker = _FakeTracker()
    t.tokenizer = None
    t.train_dataloader = dataloader
    t.total_training_steps = len(dataloader) * cfg.trainer.epochs
    t.resume_mode = ResumeMode(cfg.trainer.resume_mode)
    t.all_metrics, t.all_timings = {}, {}
    t.global_step = 0
    t._vllm_metrics_scraper = None
    t.critic_model = None
    t.ref_model = object()  # makes update_ref_every_epoch observable
    t.dispatch = _FakeDispatch()
    rec = {"train": [], "eval": [], "ref_sync": [], "ckpt": []}

    async def generate(generator_input):
        return {"prompts": generator_input["prompts"]}

    async def eval_():
        rec["eval"].append(t.global_step)
        return {}

    def save_checkpoints():
        type(t).save_checkpoints(t)
        with open(os.path.join(cfg.trainer.ckpt_path, f"global_step_{t.global_step}", "data.pt"), "rb") as f:
            rec["ckpt"].append((t.global_step, torch.load(f, map_location="cpu", weights_only=False)))

    t.init_weight_sync_state = lambda: None
    t.generate = generate
    t.eval = eval_
    t.postprocess_generator_output = lambda go, uids: (go, uids)
    t.convert_to_training_input = lambda go, uids: _Batch(uids)
    t.fwd_logprobs_values_reward = lambda b: b
    t.compute_advantages_and_returns = lambda b: b
    t.train_critic_and_policy = lambda b: rec["train"].append([t.global_step, b.batch_uids]) or {}
    t.update_ref_with_policy = lambda: rec["ref_sync"].append(t.global_step)
    t.save_checkpoints = save_checkpoints
    t._cleanup_old_checkpoints = lambda: None

    asyncio.run(t.train())

    logs = [
        {"step": step, "keys": sorted(k for k in data if not k.startswith("timing/"))} for step, data in t.tracker.logs
    ]
    epoch_per_step = [
        [data["trainer/global_step"], data["trainer/epoch"]] for _, data in t.tracker.logs if "trainer/epoch" in data
    ]
    return {
        "train_steps": rec["train"],
        "n_train_steps": len(rec["train"]),
        "epoch_per_step": epoch_per_step,
        "evals": rec["eval"],
        "ref_syncs": rec["ref_sync"],
        "ckpts": [
            {
                "global_step": gs,
                "consumed": batches_consumed(s),
                "finished": s["_iterator_finished"],
                "sha256": state_sha(s),
            }
            for gs, s in rec["ckpt"]
        ],
        "tracker_logs": logs,
    }


def write_source_ckpt(root: Path, global_step: int, state) -> str:
    d = root / f"global_step_{global_step}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"global_step": global_step, "config": {}}, d / "trainer_state.pt")
    torch.save(state, d / "data.pt")
    return str(d)


def run_scenario(num_workers: int, name: str, tmp_path: Path):
    golden = GOLDEN[num_workers]
    sc = golden["scenarios"][name]
    src, epochs = sc["source"], sc["cfg_epochs"]
    resume_path = None
    if src is not None:
        gs = int(src.rsplit("_", 1)[1])
        resume_path = write_source_ckpt(tmp_path / "src", gs, decode(golden["source_states"][src]))
    cfg = make_cfg(epochs, str(tmp_path / "ckpts"), "none" if src is None else "from_path", resume_path)
    return sc, run_train(cfg, make_dataloader(num_workers))


# --- intended differences vs the fork, per scenario ------------------------------------------------------------
# epoch_shift: the fork's trainer/epoch minus the correct one. ref_syncs: the upstream update_ref_every_epoch steps
# (the fork's, shifted with its epoch index). extra_last_batch: upstream trains the epoch's last batch the fork's
# step bound cut. no_stray_step: epochs=0, the fork's stray training step is gone.
INTENDED = {
    "boundary_3": dict(epoch_shift=1, fork_ref_syncs=[9], ref_syncs=[6]),
    "boundary_6": dict(epoch_shift=1, fork_ref_syncs=[9], ref_syncs=[]),
    "final_boundary_9_extend": dict(epoch_shift=1, fork_ref_syncs=[12], ref_syncs=[]),
    "fork_safety_net_10_extend": dict(epoch_shift=1, extra_last_batch=True, fork_ref_syncs=[12], ref_syncs=[]),
    "finished_10_extend": dict(epoch_shift=0, extra_last_batch=True, fork_ref_syncs=[], ref_syncs=[]),
    "epochs0_no_resume": dict(no_stray_step=True),
    "epochs0_mid_epoch_4": dict(no_stray_step=True),
}


def _trained_through_loop_end(new: dict) -> bool:
    return new["n_train_steps"] > 0


@pytest.mark.parametrize("num_workers", [0, 2])
def test_fresh_run_matches_fork(num_workers, tmp_path):
    fork = GOLDEN[num_workers]["fresh"]
    new = run_train(make_cfg(EPOCHS, str(tmp_path / "ckpts"), "none", None), make_dataloader(num_workers))
    for key in ("train_steps", "epoch_per_step", "evals", "ref_syncs"):
        assert new[key] == fork[key], key
    assert new["n_train_steps"] == len(make_dataloader(num_workers)) * EPOCHS
    # every in-loop checkpoint is byte-identical; the safety net differs only in how the loop ended
    assert new["ckpts"][:-1] == fork["ckpts"][:-1]
    fork_net, new_net = fork["ckpts"][-1], new["ckpts"][-1]
    assert fork_net["global_step"] == new_net["global_step"] == EPOCHS * 3 + 1
    assert (fork_net["finished"], new_net["finished"]) == (False, True)
    # upstream's safety net is exactly the "finished_10" source the capture built (end-of-epoch state run to
    # StopIteration), so the finished_* scenarios resume from what a release run really writes
    assert new_net["sha256"] == state_sha(decode(GOLDEN[num_workers]["source_states"]["finished_10"]))
    assert [k for log in new["tracker_logs"] for k in log["keys"] if k.startswith("resume_diag/")] == []


@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("name", SCENARIOS)
def test_resume_matches_fork(num_workers, name, tmp_path):
    fork, new = run_scenario(num_workers, name, tmp_path)
    diff = INTENDED.get(name, {})
    per_epoch = GOLDEN[num_workers]["setup"]["len_dataloader"]

    # 1. the batches trained, in order
    if diff.get("no_stray_step"):
        assert fork["n_train_steps"] == 1 and new["n_train_steps"] == 0
    elif diff.get("extra_last_batch"):
        assert new["train_steps"][:-1] == fork["train_steps"]
        assert new["n_train_steps"] == fork["n_train_steps"] + 1 == per_epoch
        extra_step, extra_uids = new["train_steps"][-1]
        assert extra_step == fork["train_steps"][-1][0] + 1
        # it is the same epoch's last batch: disjoint from the epoch's other batches
        seen = {u for _, uids in fork["train_steps"] for u in uids}
        assert not seen & set(extra_uids)
    else:
        assert new["train_steps"] == fork["train_steps"]

    # 2. the epoch index logged per step
    shift = diff.get("epoch_shift", 0)
    want_epochs = [[gs, e - shift] for gs, e in fork["epoch_per_step"]]
    if diff.get("extra_last_batch"):
        assert new["epoch_per_step"][:-1] == want_epochs
        assert new["epoch_per_step"][-1] == [new["train_steps"][-1][0], want_epochs[-1][1]]
    elif diff.get("no_stray_step"):
        assert new["epoch_per_step"] == []
    else:
        assert new["epoch_per_step"] == want_epochs

    # 3. update_ref_every_epoch's syncs (the fork's moved with its epoch index)
    if "ref_syncs" in diff:
        assert fork["ref_syncs"] == diff["fork_ref_syncs"]
        assert new["ref_syncs"] == diff["ref_syncs"]
    else:
        assert new["ref_syncs"] == fork["ref_syncs"]

    # 4. evals: eval_before_train at the loaded step, then every eval_interval / at total_training_steps
    total = per_epoch * fork["cfg_epochs"]
    loaded = fork["evals"][0]
    assert new["evals"][0] == loaded
    assert new["evals"][1:] == [gs for gs, _ in new["train_steps"] if gs % 4 == 0 or gs == total]
    if not diff:
        assert new["evals"] == fork["evals"]

    # 5. checkpoints written by the resumed run: in-loop saves byte-identical where the same steps ran
    fork_loop = [c for c in fork["ckpts"] if c["global_step"] <= max([s for s, _ in fork["train_steps"]] or [0])]
    new_loop = [c for c in new["ckpts"] if c["global_step"] <= max([s for s, _ in new["train_steps"]] or [0])]
    common = {c["global_step"] for c in fork_loop} & {c["global_step"] for c in new_loop}
    assert [c for c in new_loop if c["global_step"] in common] == [c for c in fork_loop if c["global_step"] in common]
    # the safety net: upstream's loop ends on StopIteration when it ran an epoch; otherwise it re-saves the
    # restored (or fresh) iterator untouched. Either way a later resume starts the next batch correctly.
    net = new["ckpts"][-1]
    if _trained_through_loop_end(new):
        assert net["finished"] is True
    else:
        assert net["finished"] is False

    # 6. the same tracker logs, minus the fork's dropped resume_diag/* diagnostics (their own log calls)
    assert [k for log in new["tracker_logs"] for k in log["keys"] if k.startswith("resume_diag/")] == []
    if not diff:
        fork_logs = [
            log
            for log in fork["tracker_logs"]
            if not (log["keys"] and all(k.startswith("resume_diag/") for k in log["keys"]))
        ]
        assert new["tracker_logs"] == fork_logs


# --- the drain itself, on a real StatefulDataLoader ------------------------------------------------------------
def _batches(dataloader):
    return [[item["uid"] for item in batch] for batch in dataloader]


def _restored(num_workers, state):
    dl = make_dataloader(num_workers)
    dl.load_state_dict(state)
    return dl


def _drain(dataloader, global_step, epochs):
    t = object.__new__(OPSDRayPPOTrainer)
    t.cfg = types.SimpleNamespace(trainer=types.SimpleNamespace(epochs=epochs))
    t.train_dataloader = dataloader
    t._drain_exhausted_resume_epoch(global_step)
    return dataloader


@pytest.mark.parametrize("num_workers", [0, 2])
def test_batches_consumed_pins_torchdata_state_keys(num_workers):
    dl = make_dataloader(num_workers)
    it = iter(dl)
    assert batches_consumed(dl.state_dict()) == 0
    for k in range(1, len(dl) + 1):
        next(it)
        state = dl.state_dict()
        assert batches_consumed(state) == k and state["_iterator_finished"] is False
    assert next(it, None) is None
    assert dl.state_dict()["_iterator_finished"] is True


@pytest.mark.parametrize("num_workers", [0, 2])
def test_drain_reproduces_the_forks_empty_pass(num_workers):
    boundary = decode(GOLDEN[num_workers]["source_states"]["global_step_3"])
    # the fork: the first resumed epoch is an empty pass, the next one is real
    fork = _restored(num_workers, boundary)
    assert _batches(fork) == []
    fork_epochs = [_batches(fork), _batches(fork)]
    # the release: drained in load_checkpoints, so the first epoch train() sees is the real one
    new = _drain(_restored(num_workers, boundary), global_step=3, epochs=EPOCHS)
    assert new.next_iter_state is None
    assert [_batches(new), _batches(new)] == fork_epochs
    assert all(len(epoch) == len(new) for epoch in fork_epochs)


@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize(
    "source,global_step,epochs",
    [
        ("global_step_4", 4, EPOCHS),  # mid-epoch: the restored iterator still has batches
        ("finished_10", 10, EPOCHS + 1),  # already hit StopIteration: StatefulDataLoader starts fresh itself
        ("global_step_6", 6, 0),  # epochs=0 eval run: train() runs no epoch, so do not touch the loader
        ("global_step_10", 10, EPOCHS),  # relaunching a finished run: no epoch left either
    ],
)
def test_drain_leaves_other_states_alone(num_workers, source, global_step, epochs):
    state = decode(GOLDEN[num_workers]["source_states"][source])
    dl = _drain(_restored(num_workers, state), global_step, epochs)
    assert dl.next_iter_state is state and dl._iterator is None  # untouched: no iterator, no worker spawned
    assert _batches(dl) == _batches(_restored(num_workers, state))


def test_drain_skips_without_a_restored_state():
    dl = _drain(make_dataloader(0), global_step=3, epochs=EPOCHS)
    assert dl.next_iter_state is None and dl._iterator is None


@pytest.mark.parametrize("num_workers", [0, 2])
def test_drain_restores_a_state_that_still_yields(num_workers):
    # A state whose counters claim a full epoch while the sampler is mid-epoch (what a changed pool size can
    # produce): the drain must not eat the batch it pulls; it puts the state back.
    mid = decode(GOLDEN[num_workers]["source_states"]["global_step_4"])
    lying = decode(GOLDEN[num_workers]["source_states"]["global_step_4"])
    if "_num_yielded" in lying:  # single-process: a counter; the position lives in the sampler state
        lying["_num_yielded"] = 3
    else:  # multi-process: the snapshot step is a counter too (the main/worker snapshots hold the position)
        lying["_snapshot"]["_snapshot_step"] = 3 - lying["_steps_since_snapshot"]
    assert batches_consumed(lying) == 3
    dl = _drain(_restored(num_workers, lying), global_step=4, epochs=EPOCHS)
    assert dl.next_iter_state is lying
    assert _batches(dl) == _batches(_restored(num_workers, mid))


def test_trainer_shape():
    assert OPSDRayPPOTrainer.__bases__ == (OPSDEvalMixin, OPSDWorkerMixin, RayPPOTrainer)
    # upstream train() runs unmodified: the resume fix lives in load_checkpoints
    for name in ("train", "_build_train_dataloader_and_compute_training_steps", "update_ref_with_policy"):
        assert name not in OPSDRayPPOTrainer.__dict__, name
    assert "load_checkpoints" in OPSDRayPPOTrainer.__dict__

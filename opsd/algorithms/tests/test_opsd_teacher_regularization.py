"""Tests for the SDPO (arXiv:2601.20802) teacher regularizers ported into OPSD.

  trust-region (OPSD_TEACHER=trust-region) -- blend the frozen/live teachers' log-probs, no new weights
  ema          (OPSD_TEACHER=ema)          -- theta' <- (1-a) theta' + a theta on the ref worker

Both are OFF by default (OPSD_TEACHER still defaults to `frozen`), so the assertions below double as a
guarantee that every previously-submitted OPSD arm is unaffected.

Two things here are worth more than the wiring: (1) both modes must reduce EXACTLY to the endpoints OPSD
already had at a=0/a=1, or "trust-region" is a new algorithm rather than an interpolation of two known
ones; and (2) the EMA has to survive bf16 checkpoint storage -- an alpha=0.01 update is ~1e-4 relative,
an order of magnitude under bf16's ~7.8e-3 epsilon, so blending in the stored dtype silently no-ops.
That is the failure this suite is really guarding, because it would look exactly like `frozen`.
"""

import importlib.util
import os
import re

import pytest
import torch
from safetensors.torch import load_file, save_file

from opsd._repo import repo_root

REPO = repo_root(__file__)
TRAINER = REPO / "opsd/trainer/main_opsd_alfworld.py"
MODULE = REPO / "opsd/trainer/teacher_regularization.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("opsd_teacher_regularization", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tr = _load_module()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("OPSD_TEACHER", "OPSD_TEACHER_ALPHA", "OPSD_EMA_INTERVAL", "OPSD_EMA_DIR"):
        monkeypatch.delenv(k, raising=False)


# --------------------------------------------------------------------------- defaults are unchanged
def test_default_mode_is_frozen():
    assert tr.resolve_teacher_mode() == "frozen"


def test_legacy_modes_still_parse():
    for mode in ("frozen", "live"):
        os.environ["OPSD_TEACHER"] = mode
        assert tr.resolve_teacher_mode() == mode


def test_trainer_defaults_ema_off():
    """The EMA hook must be a no-op for every arm that does not ask for it."""
    src = TRAINER.read_text()
    assert 'if resolve_teacher_mode() != "ema":' in src, "EMA sync must early-return for non-ema modes"
    assert re.search(r'OPSD_EMA_INTERVAL", "1"', src), "EMA interval must default to 1"


# --------------------------------------------------------------------------- parsing / validation
@pytest.mark.parametrize("alias", ["trust-region", "trust_region", "TR", "TrustRegion", " tr "])
def test_trust_region_aliases(alias):
    os.environ["OPSD_TEACHER"] = alias
    assert tr.resolve_teacher_mode() == "trust-region"


def test_unknown_mode_raises():
    os.environ["OPSD_TEACHER"] = "frozn"
    with pytest.raises(ValueError, match="OPSD_TEACHER"):
        tr.resolve_teacher_mode()


@pytest.mark.parametrize("bad", ["1.5", "-0.1"])
def test_alpha_out_of_range_raises(bad):
    os.environ["OPSD_TEACHER_ALPHA"] = bad
    with pytest.raises(ValueError, match="OPSD_TEACHER_ALPHA"):
        tr.resolve_teacher_alpha()


def test_new_modes_refuse_to_run_without_a_ref_model():
    """The pre-existing `frozen` path silently degrades to the policy worker when there is no ref
    model, and that degradation IS the live-teacher collapse. New modes must fail loudly instead."""
    for mode in ("ema", "trust-region"):
        with pytest.raises(ValueError, match="requires a reference model"):
            tr.require_ref_model(mode, None)


def test_legacy_modes_keep_their_fallback_behaviour():
    tr.require_ref_model("frozen", None)  # must not raise -- unchanged semantics
    tr.require_ref_model("live", None)


# --------------------------------------------------------------------------- trust-region blend
@pytest.fixture
def logprobs():
    torch.manual_seed(0)
    mk = lambda: torch.log(torch.rand(4, 16) * 0.5 + 1e-3)  # noqa: E731
    return mk(), mk(), mk()  # frozen, live, student


def test_alpha_zero_is_exactly_frozen(logprobs):
    frozen, live, _ = logprobs
    assert torch.equal(tr.blend_teacher_log_probs(frozen, live, 0.0), frozen)


def test_alpha_one_is_exactly_live(logprobs):
    frozen, live, _ = logprobs
    assert torch.equal(tr.blend_teacher_log_probs(frozen, live, 1.0), live)


def test_k1_of_blend_is_the_convex_blend_of_k1s(logprobs):
    """The documented consequence of dropping the log-normalizer: OPSD only ever consumes
    `student_lp - teacher_lp`, and for the lerped teacher that is exactly the convex blend of the
    frozen- and live-teacher KLs."""
    frozen, live, student = logprobs
    a = 0.3
    k1_mix = student - tr.blend_teacher_log_probs(frozen, live, a)
    k1_cvx = (1 - a) * (student - frozen) + a * (student - live)
    assert torch.allclose(k1_mix, k1_cvx, atol=1e-6)


# --------------------------------------------------------------------------- EMA schedule algebra
def test_effective_alpha_matches_per_step_schedule():
    """OPSD_EMA_INTERVAL must be a pure cost knob: K applications at the compounded rate have to land
    where N per-step applications at `alpha` would."""
    alpha, k, n = 0.01, 10, 200
    per_step = 1.0 - (1.0 - alpha) ** n
    chunked = 1.0 - (1.0 - tr.effective_alpha(alpha, k)) ** (n // k)
    assert per_step == pytest.approx(chunked, abs=1e-12)


def test_effective_alpha_identity_at_interval_one():
    assert tr.effective_alpha(0.05, 1) == pytest.approx(0.05, abs=1e-12)


def test_half_life_matches_sdpo_regime():
    """SDPO ships alpha=0.01 against ~80-90-step runs, i.e. deliberately near-frozen."""
    assert tr.ema_half_life(0.01) == pytest.approx(68.97, abs=0.1)
    assert tr.ema_half_life(0.0) == float("inf")


# --------------------------------------------------------------------------- EMA end-to-end
class _FakeDispatch:
    """Mimics dispatch.save_hf_model / init_model over an in-memory worker weight store."""

    SHARD = "model-00001-of-00001.safetensors"

    def __init__(self, policy, ref):
        self.w = {"policy": policy, "ref": ref}
        self.init_calls = []

    def save_hf_model(self, model, export_dir, tokenizer):
        os.makedirs(export_dir, exist_ok=True)
        save_file(
            {k: v.clone() for k, v in self.w[model].items()}, os.path.join(export_dir, self.SHARD), {"format": "pt"}
        )

    def init_model(self, model, model_path):
        self.w[model] = load_file(os.path.join(model_path, self.SHARD))
        self.init_calls.append((model, model_path))


@pytest.fixture
def ema_run(tmp_path):
    """50 EMA applications with a static bf16 student 1% above the teacher."""
    theta = {"w": torch.full((64,), 1.01, dtype=torch.bfloat16)}
    theta_prime = {"w": torch.full((64,), 1.00, dtype=torch.bfloat16)}
    sync = tr.EMATeacherSync(
        _FakeDispatch(theta, theta_prime), tokenizer=None, work_dir=str(tmp_path), alpha=0.01, interval=1
    )
    metrics = None
    for step in range(1, 51):
        metrics = sync.apply(step)
    return sync, metrics, tmp_path


def test_should_apply_respects_interval():
    sync = tr.EMATeacherSync(None, None, work_dir="/nonexistent", alpha=0.01, interval=5)
    assert not sync.should_apply(0)  # never on step 0
    assert not sync.should_apply(4)
    assert sync.should_apply(5) and sync.should_apply(10)


def test_zero_alpha_never_applies():
    sync = tr.EMATeacherSync(None, None, work_dir="/nonexistent", alpha=0.0, interval=1)
    assert not sync.should_apply(7)


def test_fp32_master_tracks_the_analytic_ema(ema_run):
    sync, _, tmp_path = ema_run
    master = load_file(str(tmp_path / "teacher_fp32" / _FakeDispatch.SHARD))["w"]
    # Analytic target from the *bf16-quantized* endpoints: bf16 has 7 mantissa bits, so the
    # checkpointed student is 1.0078125, not 1.01. The EMA tracks what is actually stored.
    th = float(torch.tensor(1.01, dtype=torch.bfloat16))
    th0 = float(torch.tensor(1.00, dtype=torch.bfloat16))
    assert float(master[0]) == pytest.approx(th - (th - th0) * 0.99**50, abs=1e-5)


def test_ema_survives_bf16_checkpoint_storage(ema_run):
    """The precision trap: blending in the stored bf16 dtype would leave this pinned at exactly 1.0."""
    sync, _, tmp_path = ema_run
    master = load_file(str(tmp_path / "teacher_fp32" / _FakeDispatch.SHARD))["w"]
    assert float(master[0]) > 1.0009, "EMA rounded away -- the teacher is silently still theta_0"


def test_hf_copy_keeps_checkpoint_dtype(ema_run):
    """init_model consumes a real HF checkpoint, so the written shards must stay in native dtype."""
    sync, _, tmp_path = ema_run
    assert load_file(str(tmp_path / "teacher_hf" / _FakeDispatch.SHARD))["w"].dtype == torch.bfloat16


def test_ref_worker_is_repointed_at_theta_prime(ema_run):
    sync, _, tmp_path = ema_run
    assert len(sync.dispatch.init_calls) == 50
    assert all(model == "ref" for model, _ in sync.dispatch.init_calls)
    hf = load_file(str(tmp_path / "teacher_hf" / _FakeDispatch.SHARD))["w"]
    assert float(sync.dispatch.w["ref"]["w"][0]) == pytest.approx(float(hf[0]), abs=1e-6)


def test_metrics_and_cleanup(ema_run):
    sync, metrics, tmp_path = ema_run
    assert metrics["distill/ema_applications"] == 50
    assert 0 < metrics["distill/ema_teacher_student_rel_l2"] < 0.01
    assert "distill/ema_sync_seconds" in metrics
    leftovers = [d.name for d in tmp_path.iterdir() if d.name.startswith("policy_step_")]
    assert not leftovers, f"transient policy exports were not cleaned up: {leftovers}"

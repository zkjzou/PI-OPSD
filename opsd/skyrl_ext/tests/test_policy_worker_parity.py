"""Fork parity: OPSDFSDPPolicyWorkerBase reproduces the OLD fork's policy worker on every captured case.

The goldens (opsd/tests/fixtures/fork_parity/worker/goldens.json) were captured by driving the fork's own
``FSDPPolicyWorkerBase.forward_backward`` (its BatchIterator attached ``distill_log_probs``; its
``_forward_backward_micro`` read OPSD_* from os.environ) on the stub harness in test_policy_worker.py: beta in
{0, 0.1, 1e-4} x use_kl_loss x use_entropy_loss, SDAR lambda 0.5 x gate {gap, entropy, soft_or}, a 5-row batch
at micro batch 2 (chunks 2, 2, 1) plus micro 1 and 3, a missing / non-finite teacher, k1, dp_size 1, the SFT and
Tinker (loss_fn / loss_fn_config) paths, and the one input the fork rejected (a bogus gate with lambda != 0).

Here the knobs are resolved from the same environment with ``LossKnobs.from_env`` (the driver-side parse) and
pushed to the worker, exactly as ``OPSDWorkerMixin.build_models`` does. When the knobs are inactive, production
runs SkyRL's own ``FSDPPolicyWorkerBase``, so those cases are checked against both classes.

TOLERANCE: float32 rounding only -- rel 1e-6, abs 1e-9 per value (float32 eps is 1.2e-7). The new worker runs
the fork's exact op sequence, so on the capture machine every value matched bit for bit; the tolerance only
absorbs a different BLAS/CPU reordering a reduction.
"""

import math

import pytest

from opsd.algorithms.distill_loss import LossKnobs
from opsd.skyrl_ext.policy_worker import OPSDFSDPPolicyWorkerBase
from opsd.skyrl_ext.tests.test_policy_worker import (
    dec,
    load_goldens,
    load_inputs,
    make_batch,
    make_cfg,
    make_worker,
    run_forward_backward,
)
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase

pytestmark = pytest.mark.filterwarnings("ignore:CUDA is not available or torch_xla is imported")

REL, ABS = 1e-6, 1e-9
GOLDENS = load_goldens()
INPUTS = load_inputs()


def _close(a, b, path, worst):
    if isinstance(b, dict):
        assert isinstance(a, dict) and set(a) == set(b), f"{path}: keys {sorted(a)} != {sorted(b)}"
        for k in b:
            _close(a[k], b[k], f"{path}.{k}", worst)
    elif isinstance(b, list):
        assert isinstance(a, list) and len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]", worst)
    elif isinstance(b, bool) or b is None or isinstance(b, str):
        assert a == b, f"{path}: {a!r} != {b!r}"
    elif isinstance(b, int) and not isinstance(a, float):
        assert a == b, f"{path}: {a!r} != {b!r}"
    else:
        assert math.isclose(a, b, rel_tol=REL, abs_tol=ABS), f"{path}: {a!r} != {b!r}"
        worst[0] = max(worst[0], abs(a - b))


def _check(result, golden):
    worst = [0.0]
    _close(result["backward_losses"], golden["backward_losses"], "backward_losses", worst)
    _close(result["micro"], golden["micro"], "micro", worst)
    _close(result["metrics"], golden["metrics"], "metrics", worst)
    _close(result["loss_fn_outputs"], golden["loss_fn_outputs"], "loss_fn_outputs", worst)
    expected_grad = dec(golden["grad_W"])
    assert result["grad_W"].shape == expected_grad.shape
    _close(result["grad_W"].tolist(), expected_grad.tolist(), "grad_W", worst)
    return worst[0]


def _run(cls, case, knobs):
    cfg = make_cfg(case["algo"], micro=case["micro"], algo_base=INPUTS["algo_base"])
    worker = make_worker(cls, cfg, INPUTS["W0"], dp_size=case["dp_size"], knobs=knobs)
    batch = make_batch(INPUTS, case["teacher"])
    return run_forward_backward(worker, batch, loss_fn=case["loss_fn"], loss_fn_config=case["loss_fn_config"])


def test_goldens_cover_the_requested_matrix():
    names = {c["name"] for c in GOLDENS}
    for beta in ("0", "0.1", "1e-4"):
        for kl in (0, 1):
            for ent in (0, 1):
                assert f"beta={beta}/kl={kl}/ent={ent}" in names
    for gate in ("gap", "entropy", "soft_or"):
        assert {f"sdar=0.5/gate={gate}/ent=0", f"sdar=0.5/gate={gate}/ent=1"} <= names
    assert any(c["micro"] == 2 and c["result"].get("micro") and len(c["result"]["micro"]) == 3 for c in GOLDENS)


@pytest.mark.parametrize("case", [c for c in GOLDENS if "raises" not in c["result"]], ids=lambda c: c["name"])
def test_new_worker_reproduces_the_fork(case):
    knobs = LossKnobs.from_env(case["env"])
    golden = case["result"]
    _check(_run(OPSDFSDPPolicyWorkerBase, case, knobs), golden)
    if not knobs.active:
        # production builds SkyRL's own worker for these (OPSDWorkerMixin swaps only on active knobs)
        _check(_run(FSDPPolicyWorkerBase, case, None), golden)


@pytest.mark.parametrize("case", [c for c in GOLDENS if "raises" in c["result"]], ids=lambda c: c["name"])
def test_rejected_inputs_are_rejected_at_startup(case):
    """The fork raised at the first micro-batch; the driver-side parse raises the same error at startup."""
    golden = case["result"]
    with pytest.raises(ValueError) as exc:
        LossKnobs.from_env(case["env"])
    assert type(exc.value).__name__ == golden["raises"]
    assert str(exc.value) == golden["message"]

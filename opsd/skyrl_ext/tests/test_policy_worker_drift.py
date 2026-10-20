"""Drift guards for the RL branch OPSDFSDPPolicyWorkerBase copies from SkyRL.

``OPSDFSDPPolicyWorkerBase._opsd_forward_backward_micro`` is a copy of upstream
``PolicyWorkerBase._forward_backward_micro`` (SkyRL @ ecdaa946) with marked OPSD insertions: the beta / SDAR
terms sit between the KL term and ``strategy.backward``, so there is no way to wrap the upstream method. Two
checks keep the copy honest:

  1. upstream's method source is pinned by sha256, so bumping the SkyRL pin to a version that changes it
     fails here -- re-sync the copy, then update the hash;
  2. the copy, with its ``# >>> OPSD`` ... ``# <<< OPSD`` blocks removed, is line for line upstream's method minus
     the SFT branch (which the OPSD worker never runs: cross_entropy falls through to upstream) and minus the one
     loss expression the OPSD block replaces.
"""

import hashlib
import inspect

from opsd.skyrl_ext.policy_worker import OPSDFSDPPolicyWorkerBase
from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase

# sha256(inspect.getsource(PolicyWorkerBase._forward_backward_micro)) at SkyRL ecdaa946.
UPSTREAM_FORWARD_BACKWARD_MICRO_SHA256 = "19285aa94b8c5ab052903f4fcb10262c22848b31aab29e63a0cce57072075af1"

# The single upstream line the OPSD block replaces (the loss expression gains the two teacher terms).
REPLACED_UPSTREAM_LINES = (
    "loss = policy_loss * grad_sum_correction_factor + (kl_loss_term - entropy_loss_term) * microbatch_weight",
)


def _body(src: str) -> list:
    """Source lines from ``self.model.train()`` (the first statement after the docstring) to the end."""
    lines = src.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "self.model.train()")
    return lines[start:]


def _normalise(lines) -> list:
    return [ln.strip() for ln in lines if ln.strip()]


def _upstream_rl_path() -> list:
    lines = _body(inspect.getsource(PolicyWorkerBase._forward_backward_micro))
    sft = next(i for i, ln in enumerate(lines) if ln.strip().startswith("# SFT path"))
    rl = next(i for i, ln in enumerate(lines) if ln.strip().startswith("# RL path"))
    assert lines[rl - 1].strip() == "else:", "upstream's SFT/RL split moved"
    kept = lines[:sft] + lines[rl:]
    return [ln for ln in _normalise(kept) if ln not in REPLACED_UPSTREAM_LINES]


def _copy_without_opsd_blocks() -> list:
    lines = _body(inspect.getsource(OPSDFSDPPolicyWorkerBase._opsd_forward_backward_micro))
    out, inside, blocks = [], False, 0
    for ln in lines:
        s = ln.strip()
        if s.startswith("# >>> OPSD"):
            assert not inside, "nested # >>> OPSD block"
            inside, blocks = True, blocks + 1
        elif s.startswith("# <<< OPSD"):
            assert inside, "# <<< OPSD without an opening marker"
            inside = False
        elif not inside:
            out.append(ln)
    assert not inside, "unterminated # >>> OPSD block"
    assert blocks == 4, f"expected the 4 documented OPSD blocks, found {blocks}"
    return _normalise(out)


def test_upstream_forward_backward_micro_is_pinned():
    src = inspect.getsource(PolicyWorkerBase._forward_backward_micro)
    digest = hashlib.sha256(src.encode()).hexdigest()
    assert digest == UPSTREAM_FORWARD_BACKWARD_MICRO_SHA256, (
        "SkyRL's PolicyWorkerBase._forward_backward_micro changed (sha256 "
        f"{digest}). opsd/skyrl_ext/policy_worker.py copies its RL branch: re-sync "
        "OPSDFSDPPolicyWorkerBase._opsd_forward_backward_micro with the new upstream text, re-run the fork-parity "
        "tests, then update UPSTREAM_FORWARD_BACKWARD_MICRO_SHA256."
    )


def test_copy_is_upstream_plus_marked_opsd_blocks():
    upstream, ours = _upstream_rl_path(), _copy_without_opsd_blocks()
    assert ours == upstream, next(
        (f"line {i}: copy has {a!r}, upstream has {b!r}" for i, (a, b) in enumerate(zip(ours, upstream)) if a != b),
        f"copy has {len(ours)} lines, upstream {len(upstream)}",
    )


def test_the_replaced_line_really_is_upstream():
    """If upstream rewrote its loss expression, the REPLACED list would silently match nothing."""
    upstream = _normalise(_body(inspect.getsource(PolicyWorkerBase._forward_backward_micro)))
    for line in REPLACED_UPSTREAM_LINES:
        assert upstream.count(line) == 1


def test_only_the_rl_hooks_are_overridden():
    """The subclass overrides forward_backward / _forward_backward_micro and adds its own helpers; it must not
    shadow any other PolicyWorkerBase / FSDPPolicyWorkerBase method (init_model, forward, optim_step, ...)."""
    own = {k for k, v in vars(OPSDFSDPPolicyWorkerBase).items() if callable(v) and not k.startswith("__")}
    inherited = {k for cls in OPSDFSDPPolicyWorkerBase.__mro__[1:] for k in vars(cls)}
    assert own & inherited == {"forward_backward", "_forward_backward_micro"}

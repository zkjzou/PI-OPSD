"""Step schedule for the convex advantage mix  A_mix = lam * A_GRPO + (1 - lam) * A_SDPO.

`OPSD_MIX_LAMBDA` alone is a constant, which is how every lambda arm before 2026-09-24 ran. The schedule
mirrors RLSD's (rlsd.rlsd_lambda): the teacher's share is largest at the start and is handed back to the
env reward over a fixed window. Note the direction is the opposite of RLSD's knob -- here lam is the
weight on A_GRPO, so moving the teacher OUT means lam goes UP toward 1.

Knobs (all optional; unset DECAY_STEPS keeps the constant, so existing arms are byte-identical):
  OPSD_MIX_LAMBDA              lam at step 0 (default 1.0 = pure GRPO/DAPO, i.e. the mix is off)
  OPSD_MIX_LAMBDA_FINAL        lam reached at the end of the window (default: OPSD_MIX_LAMBDA)
  OPSD_MIX_LAMBDA_DECAY_STEPS  window length in steps; <=0 or unset pins lam at OPSD_MIX_LAMBDA
  OPSD_MIX_LAMBDA_AFTER        lam for every step PAST the window (default: OPSD_MIX_LAMBDA_FINAL).
                               1.0 switches to plain DAPO once the window closes.

Worked example (the 2026-09-24 arm): LAMBDA=0.5, FINAL=0.99, DECAY_STEPS=100, AFTER=1.0 gives
lam(1)=0.5049 ... lam(50)=0.745 ... lam(100)=0.99, then lam=1.0 (pure DAPO) from step 101.
"""

from __future__ import annotations

import os


def _read(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    v = float(raw) if raw.strip() else default
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be in [0,1], got {v}")
    return v


def mix_decay_steps() -> int:
    return int(os.environ.get("OPSD_MIX_LAMBDA_DECAY_STEPS", "0") or 0)


def mix_lambda(global_step: int) -> float:
    """lam for this step: linear from LAMBDA to FINAL over steps [0, DECAY_STEPS], then AFTER."""
    lam0 = _read("OPSD_MIX_LAMBDA", 1.0)
    steps = mix_decay_steps()
    if steps <= 0:
        return lam0
    lam1 = _read("OPSD_MIX_LAMBDA_FINAL", lam0)
    after = _read("OPSD_MIX_LAMBDA_AFTER", lam1)
    if global_step > steps:
        return after
    frac = max(float(global_step), 0.0) / float(steps)
    return lam0 + (lam1 - lam0) * frac


def mix_lambda_configured() -> bool:
    """True if ANY step of the schedule mixes (lam < 1).

    The teacher log-probs must be transported, and the mutual-exclusion checks against RLSD / token-adv
    must fire, based on the whole schedule rather than on the current step: a schedule that starts at
    lam=1 and ramps down would otherwise never receive the teacher tensor it later needs.
    """
    lam0 = _read("OPSD_MIX_LAMBDA", 1.0)
    if mix_decay_steps() <= 0:
        return lam0 < 1.0
    lam1 = _read("OPSD_MIX_LAMBDA_FINAL", lam0)
    after = _read("OPSD_MIX_LAMBDA_AFTER", lam1)
    return min(lam0, lam1, after) < 1.0

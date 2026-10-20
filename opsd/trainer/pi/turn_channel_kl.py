"""Per-turn x per-channel teacher-KL monitoring for OPSD (best-effort, never raises into training).

Splits the per-token distill KL `k1 = m_t*(s_t - h_t)` by:
  * TURN position  -- a contiguous run of loss-mask=1 tokens = one agent turn (0,1,2,...);
  * CHANNEL        -- `agent` (all masked tokens), `think` (<think>..</think> content),
                      `action` (<action>..</action> content).
and logs, every eval-interval:
  * a LINE plot  : mean KL vs turn position, one series per channel (this step's snapshot);
  * a HEATMAP    : turn position x training step, one subplot per channel (accumulated).

Alignment (see skyrl preprocess.convert_prompts_responses_to_batch_tensors): sequences are
LEFT-padded; response-level tensors (loss_mask, per_token_kl) are RIGHT-aligned in
[B, max_response_len]. So the response ids aligned to per_token_kl are `sequences[:, -Lr:]`,
and each rollout's real length is `response_length[i]`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger("pi.turn_channel_kl")

# Channel labeling (think/action span detection, incl. the "next opening tag" span termination that
# tolerates Qwen3-Thinking's frequently-omitted </think>) now lives in the shared module
# opsd.pi.signals.channels; QWEN_CHANNELS carries the same <think>/<action> tag set.
from opsd.pi.signals.channels import spec_from_env  # noqa: E402
from opsd.pi.signals.channels import (  # noqa: E402
    token_channels as _channels_token_channels,
)

CHANNELS = ("agent", "think", "action")
N_NORM_BINS = 20  # bins for the normalized turn-position plot (relative episode progress 0->1)


def _token_channels(ids: List[int], tokenizer) -> List[str]:
    """Per-token channel label ('think' | 'action' | 'other').

    The spec is SELECTED BY ENV (OPSD_KL_CHANNEL_SPEC), defaulting to the ALFWorld/Qwen
    <think>/<action> tags so existing runs are unchanged. τ² must set `tau2`: it acts by
    <tool_call>, and the Qwen spec would label every action token `other` -- a flat-zero `action`
    series that reads as a real finding instead of an unmatched tag."""
    return _channels_token_channels(ids, tokenizer, spec_from_env())


def compute_turn_channel_kl(
    sequences: torch.Tensor,
    per_token_kl: torch.Tensor,
    loss_mask: torch.Tensor,
    response_lengths: Optional[Any],
    tokenizer,
    max_turns: int = 32,
    max_rollouts: int = 16,
    n_norm_bins: int = N_NORM_BINS,
):
    """Return (by_turn, by_norm), each {channel: {'kl': [...] mean, 'n': [...] counts}} over a subsample.

    * by_turn : bucketed by ABSOLUTE turn index (0..max_turns-1).
    * by_norm : bucketed by NORMALIZED position = turn_idx / (total_turns-1) mapped into n_norm_bins over
                [0,1], so episodes of different lengths (7 vs 50 turns) align on relative progress.
    """
    B, F = per_token_kl.shape
    acc = {c: {"kl": [0.0] * max_turns, "n": [0] * max_turns} for c in CHANNELS}
    nrm = {c: {"kl": [0.0] * n_norm_bins, "n": [0] * n_norm_bins} for c in CHANNELS}
    for i in range(min(B, max_rollouts)):
        resp = F
        if response_lengths is not None:
            try:
                resp = int(response_lengths[i])
            except Exception:
                resp = F
        resp = max(0, min(resp, F))
        if resp == 0:
            continue
        ids = sequences[i, -resp:].tolist()
        ptk = per_token_kl[i, -resp:]
        msk = loss_mask[i, -resp:]
        chan = _token_channels(ids, tokenizer)
        # pass 1: collect (turn_idx, kl, channel) for masked tokens + the rollout's total turn count.
        toks: List = []
        turn = -1
        in_run = False
        for j in range(resp):
            m = bool(msk[j] > 0)
            if m and not in_run:
                turn += 1
                in_run = True
            elif not m:
                in_run = False
            if not m or turn < 0:
                continue
            c = chan[j] if j < len(chan) else "other"
            toks.append((turn, float(ptk[j].item()), c))
        total_turns = turn + 1
        if total_turns <= 0:
            continue
        denom = max(1, total_turns - 1)
        for t, k, c in toks:
            if t < max_turns:  # absolute buckets (capped at max_turns)
                acc["agent"]["kl"][t] += k
                acc["agent"]["n"][t] += 1
                if c in ("think", "action"):
                    acc[c]["kl"][t] += k
                    acc[c]["n"][t] += 1
            b = min(max(int(round((t / denom) * (n_norm_bins - 1))), 0), n_norm_bins - 1)
            nrm["agent"]["kl"][b] += k  # normalized buckets (relative progress 0->1)
            nrm["agent"]["n"][b] += 1
            if c in ("think", "action"):
                nrm[c]["kl"][b] += k
                nrm[c]["n"][b] += 1

    def _finish(a, size):
        o: Dict[str, Dict[str, List[float]]] = {}
        for c in CHANNELS:
            kl, n = a[c]["kl"], a[c]["n"]
            o[c] = {"kl": [(kl[t] / n[t] if n[t] else float("nan")) for t in range(size)], "n": list(n)}
        return o

    return _finish(acc, max_turns), _finish(nrm, n_norm_bins)


def channel_totals(per_turn: Dict[str, Dict[str, List[float]]]) -> Dict[str, float]:
    """Token-weighted mean KL per channel (across turns) -> scalar metrics for native W&B lines."""
    totals = {}
    for c in CHANNELS:
        kl, n = per_turn[c]["kl"], per_turn[c]["n"]
        num = sum((kl[t] * n[t]) for t in range(len(n)) if n[t] and kl[t] == kl[t])
        den = sum(n[t] for t in range(len(n)) if n[t] and kl[t] == kl[t])
        totals[c] = (num / den) if den else float("nan")
    return totals


def log_wandb(
    per_turn: Dict[str, Dict[str, List[float]]],
    per_turn_norm: Dict[str, Dict[str, List[float]]],
    history: List,
    history_norm: List,
    step: int,
    max_turns: int = 32,
    n_norm_bins: int = N_NORM_BINS,
) -> Dict[str, Any]:
    """Build {wandb_key: wandb.Image} for line + accumulated heatmap, over BOTH absolute turn position
    and normalized position. Returns the dict — the CALLER must merge it into the trainer's
    ``self.all_metrics`` so the images ride the trainer's single canonical
    ``tracker.log(step=global_step, commit=True)`` call (trainer.py drains all_metrics verbatim).

    Why not ``wandb.log(...)`` here: a separate ``wandb.log(img, step=gs, commit=False)`` races the
    trainer's own ``wandb.log(..., step=global_step, commit=True)``. Once the trainer commits a step,
    wandb's pointer advances; a later image logged at an equal-or-lower step is SILENTLY DROPPED — which
    is exactly why earlier runs showed the scalar teacher_kl lines (they ride all_metrics) but NOT these
    plots. Returning the images and letting them ride all_metrics removes the race entirely.

    Best-effort: returns {} (and never raises) if matplotlib/wandb are unavailable.

    NOTE: we do NOT gate on ``wandb.run`` here. Building a ``wandb.Image`` only wraps the figure — it
    needs no active run; the run is only needed to *log* it, which the caller does via the trainer's
    single ``tracker.log(all_metrics, step=global_step)``. This method runs inside
    ``apply_reward_kl_penalty`` where the global ``wandb.run`` is often unset, so an earlier
    ``if wandb.run is None: return`` guard silently dropped the by-turn-position plots while the scalar
    ``teacher_kl_*`` lines still appeared. Build the images unconditionally and let the caller log them.
    """
    images: Dict[str, Any] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import wandb

        def _emit(pt, hist, size, xlabel, key, normalized):
            # --- LINE: KL vs (turn | normalized position), one series per channel (current snapshot) ---
            xs = [x / (size - 1) for x in range(size)] if (normalized and size > 1) else list(range(size))
            fig1, ax = plt.subplots(figsize=(6, 4))
            for c in CHANNELS:
                ax.plot(xs, pt[c]["kl"], marker=".", label=c)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("teacher KL (mean per-token)")
            ax.set_title(f"OPSD teacher-KL by {xlabel} - step {step}")
            ax.legend()
            ax.grid(alpha=0.3)
            images[f"train/teacher_kl_by_turn{key}_line"] = wandb.Image(fig1)
            plt.close(fig1)

            # --- HEATMAP: position (x) x eval-snapshot (y), one subplot per channel (accumulated) ---
            hist.append((step, pt))
            fig2, axes = plt.subplots(1, len(CHANNELS), figsize=(4.6 * len(CHANNELS), 4))
            xhi = 1.0 if normalized else size
            for ax, c in zip(axes, CHANNELS):
                M = np.array([[h[1][c]["kl"][t] for t in range(size)] for h in hist], dtype=float)
                im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis", extent=[0, xhi, 0, len(hist)])
                ax.set_title(f"{c} KL")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("eval snapshot idx")
                fig2.colorbar(im, ax=ax)
            fig2.suptitle(f"OPSD teacher-KL by {xlabel} x training (through step {step})")
            fig2.tight_layout()
            images[f"train/teacher_kl_by_turn{key}_heatmap"] = wandb.Image(fig2)
            plt.close(fig2)

        _emit(per_turn, history, max_turns, "turn position", "", False)
        _emit(per_turn_norm, history_norm, n_norm_bins, "normalized position", "_normalized", True)
    except Exception as exc:  # never kill training for a plot
        logger.warning(f"turn/channel KL W&B image build failed (non-fatal): {exc}")
    return images

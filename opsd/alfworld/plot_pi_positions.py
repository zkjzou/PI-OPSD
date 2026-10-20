"""Aggregate the per-position PI rows (rows.parquet from analyze_pi_positions.py --merge) and plot,
for each PI: (A) top-k overlap ratio for k in {10,20,50,100}, (B) student->teacher KL and the sampled-token
proxy d — every panel split into THINK vs ACTION channels (mean ± 95% CI across trajectories). Plus cross-PI
overlays (all PIs on one axis, per channel) and a summary.md.

X-axis (`--x-axis`):
  step  — absolute agentic step index k (1..50). Late steps are survivorship-biased (only long trajectories
          reach them).
  norm  — normalized position = (k-1)/(L-1) with L = that trajectory's length, binned into --norm-bins over
          [0,1]. Every trajectory spans the whole axis, so no survivorship bias. Aggregation is TWO-STAGE:
          collapse within (traj, bin, channel) first, then average across trajectories — so long trajectories
          (more steps per bin) don't dominate a bin.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CH = ["think", "action"]
CH_COLOR = {"think": "#1f77b4", "action": "#d62728"}


def prepare(df: pd.DataFrame, x_axis: str, norm_bins: int):
    """Add a per-trajectory id `traj` and an x-position column `xpos`; return (df, xlabel)."""
    df = df.copy()
    # PER-ROLLOUT trajectory id. Both id conventions occur: some dumps key row_idx per-rollout, others
    # (generate_alfworld_local.py) key row_idx = parquet ROW = per-TASK with repetition_id 0..n-1 splitting
    # the n samples. Using row_idx alone then WRONGLY merges the n rollouts of a task and (for norm) uses
    # the task's max length. So combine (row_idx, repetition_id) whenever both exist — correct for either
    # convention (if row_idx is already per-rollout, appending its single repetition_id stays unique).
    if "row_idx" in df.columns and df["row_idx"].notna().any() and "repetition_id" in df.columns:
        df["traj"] = df["row_idx"].astype("Int64").astype(str) + "#" + df["repetition_id"].astype(str)
    elif "row_idx" in df.columns and df["row_idx"].notna().any():
        df["traj"] = df["row_idx"].astype("Int64").astype(str)
    elif "repetition_id" in df.columns:
        df["traj"] = df["uid"].astype(str) + "#" + df["repetition_id"].astype(str)
    else:
        raise SystemExit(
            "rows.parquet has no per-rollout id (row_idx/repetition_id) — re-run "
            "analyze_pi_positions.py to add it; uid is per-task and would merge the rollouts."
        )
    if x_axis == "step":
        df["xpos"] = df["step"].astype(float)
        return df, "agentic step k"
    # normalized: L = trajectory length (max step for that traj, PI-independent since same rollout)
    L = df.groupby("traj")["step"].transform("max")
    frac = np.where(L > 1, (df["step"] - 1) / (L - 1).clip(lower=1), 0.0)
    b = np.clip((frac * norm_bins).astype(int), 0, norm_bins - 1)
    df["xpos"] = (b + 0.5) / norm_bins  # bin center in (0,1)
    return df, "normalized position  (k-1)/(L-1)"


def agg(df: pd.DataFrame, metric: str, min_traj: int) -> pd.DataFrame:
    """Two-stage: mean within (pi,xpos,channel,traj), then mean/CI across traj. For x-axis=step the inner
    groupby is a no-op (one row per traj/step/channel) so it reduces to the plain across-traj mean."""
    per = df.groupby(["pi", "xpos", "channel", "traj"])[metric].mean().reset_index()
    out = per.groupby(["pi", "xpos", "channel"])[metric].agg(["mean", "std", "count"]).reset_index()
    out = out[out["count"] >= min_traj].copy()
    out["ci"] = 1.96 * out["std"].fillna(0.0) / np.sqrt(out["count"].clip(lower=1))
    return out


def _line(ax, sub, label, color):
    sub = sub.sort_values("xpos")
    x, y, ci = sub["xpos"].to_numpy(), sub["mean"].to_numpy(), sub["ci"].to_numpy()
    ax.plot(x, y, "-o", ms=3, color=color, label=label)
    ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.18)


def per_pi_figure(df, pi, ks, min_traj, xlabel, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    d_pi = df[df["pi"] == pi]
    for j, k in enumerate(ks):
        ax = axes[0, j]
        a = agg(d_pi, f"o{k}", min_traj)
        for ch in CH:
            s = a[a["channel"] == ch]
            if len(s):
                _line(ax, s, ch, CH_COLOR[ch])
        ax.set_title(f"top-{k} overlap ratio")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("overlap")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for j, (metric, title) in enumerate([("kl", "reverse KL(student‖teacher)"), ("d", "|Δ logp| on sampled token")]):
        ax = axes[1, j]
        a = agg(d_pi, metric, min_traj)
        for ch in CH:
            s = a[a["channel"] == ch]
            if len(s):
                _line(ax, s, ch, CH_COLOR[ch])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[1, 2].axis("off")
    axes[1, 3].axis("off")
    fig.suptitle(f"PI = {pi}  —  position-wise divergence (think vs action)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = out_dir / f"fig_{pi}.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def overlay_figure(df, pis, metric, channel, title, min_traj, xlabel, out_dir, fname):
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")
    for i, pi in enumerate(pis):
        a = agg(df[df["pi"] == pi], metric, min_traj)
        s = a[a["channel"] == channel].sort_values("xpos")
        if len(s):
            ax.plot(s["xpos"], s["mean"], "-o", ms=3, color=cmap(i % 10), label=pi)
    ax.set_title(f"{title}  —  {channel} tokens")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    if metric.startswith("o"):
        ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / fname
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def write_summary(df, pis, min_traj, x_axis, out_dir):
    lines = [
        f"# PI position-wise divergence — summary (x-axis = {x_axis})",
        "",
        "Student = π(·|x) (no hint); teacher = π(·|x⊕PI). KL = reverse KL(student‖teacher) per position; "
        "overlap@k = |top-k_student ∩ top-k_teacher|/k. early = first 25% of the trajectory, late = last 25%.",
        "",
        "| PI | KL think early→late | KL action early→late | o10 think early→late | o10 action early→late |",
        "|---|---|---|---|---|",
    ]

    def band(a, ch, lo, hi):
        s = a[(a["channel"] == ch) & (a["xpos"] >= lo) & (a["xpos"] <= hi)]
        return float(s["mean"].mean()) if len(s) else float("nan")

    # early/late thresholds: for norm use [0,.25]/[.75,1]; for step use absolute (<=5 / >=15) via xpos
    lo_hi = (0.0, 0.25, 0.75, 1.0) if x_axis == "norm" else (1, 5, 15, 9999)
    for pi in pis:
        akl = agg(df[df["pi"] == pi], "kl", min_traj)
        ao = agg(df[df["pi"] == pi], "o10", min_traj)
        row = [
            pi,
            f"{band(akl,'think',lo_hi[0],lo_hi[1]):.3f}→{band(akl,'think',lo_hi[2],lo_hi[3]):.3f}",
            f"{band(akl,'action',lo_hi[0],lo_hi[1]):.3f}→{band(akl,'action',lo_hi[2],lo_hi[3]):.3f}",
            f"{band(ao,'think',lo_hi[0],lo_hi[1]):.3f}→{band(ao,'think',lo_hi[2],lo_hi[3]):.3f}",
            f"{band(ao,'action',lo_hi[0],lo_hi[1]):.3f}→{band(ao,'action',lo_hi[2],lo_hi[3]):.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="rows.parquet from analyze_pi_positions.py --merge")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ks", default="10,20,50,100")
    ap.add_argument(
        "--x-axis",
        choices=("step", "norm"),
        default="step",
        help="step: absolute step k; norm: (k-1)/(L-1) binned (removes late-step survivorship bias).",
    )
    ap.add_argument("--norm-bins", type=int, default=20)
    ap.add_argument("--min-traj", type=int, default=10, help="drop (x,channel) bins with < this many trajectories.")
    ap.add_argument(
        "--success",
        choices=("all", "won", "lost"),
        default="all",
        help="filter trajectories by verifier success (won=1, lost=0) before plotting.",
    )
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.rows)
    if args.success != "all":
        want = 1 if args.success == "won" else 0
        df = df[df["success"].astype(float).round().astype(int) == want]
    df, xlabel = prepare(df, args.x_axis, args.norm_bins)
    pis = sorted(df["pi"].unique())
    print(f"{len(df)} rows | x-axis={args.x_axis} | success={args.success} | PIs: {pis}")

    figs = [per_pi_figure(df, pi, ks, args.min_traj, xlabel, out_dir) for pi in pis]
    for ch in CH:
        overlay_figure(
            df, pis, "kl", ch, "reverse KL(student‖teacher)", args.min_traj, xlabel, out_dir, f"overlay_kl_{ch}.png"
        )
        overlay_figure(
            df, pis, "o10", ch, "top-10 overlap ratio", args.min_traj, xlabel, out_dir, f"overlay_o10_{ch}.png"
        )
        overlay_figure(
            df,
            pis,
            "d",
            ch,
            "|Δ logp| on sampled token (SOD eq.6)",
            args.min_traj,
            xlabel,
            out_dir,
            f"overlay_d_{ch}.png",
        )
    write_summary(df, pis, args.min_traj, args.x_axis, out_dir)
    print(f"wrote {len(figs)} per-PI figures + overlays + summary.md -> {out_dir}")


if __name__ == "__main__":
    main()

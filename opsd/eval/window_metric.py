"""Best 3-consecutive-eval window: the checkpoint-selection metric every OPSD table in the paper reports.

Sources (pick one; several ``--run-dir`` / ``--wandb`` segments of a resumed run are stitched)::

    python -m opsd.eval.window_metric --run-dir runs/<RUN_NAME>          # local eval dumps
    python -m opsd.eval.window_metric --curves-csv <curve.csv>           # step,split,pass_at_1,pass_at_k,k
    python -m opsd.eval.window_metric --bench tau2 --wandb <entity>/<project>/<run_id>

A run dir holds ``exports/dumped_evals/global_step_<N>_evals/aggregated_results.jsonl``, which the trainer
writes after every eval (``trainer.dump_eval_results``, on by default): one JSON line holding the whole
eval-metrics dict with the same keys that are logged to W&B. An ALFWorld checkpoint re-score
(``EVAL_CKPT_STEPS``, run dir ``<training run>__ckpteval``) writes one export dir per step instead,
``exports_ckpt_<N>/dumped_evals/global_step_<N>_evals/``; ``--run-dir`` on that dir reads every step into one
curve.

DEFINITION (identical to the paper's tables)
    * Evaluations are taken in step order; step 0 (the untrained student) is EXCLUDED.
    * Best window = the highest mean over W (=3) CONSECUTIVE evaluations. Consecutive means adjacent in
      the eval sequence, so a missing eval shortens the sequence instead of averaging across a gap.
      Ties keep the earliest window. The centre of that window is the selected checkpoint.
    * With fewer than W evaluations no window exists: the paper leaves the cell empty, this script falls
      back to the single best evaluation and says so (``complete: false``).
    * pass@k gets its OWN best window (it can come from a different window than pass@1).

PER-BENCHMARK SCORE (``pass@1`` = mean success over the samples of each task)
    * ALFWorld: combined = (140 * valid_seen + 134 * valid_unseen) / 274, from ``eval/alfworld/pass_at_1``
      and ``eval/alfworld_valid_unseen/pass_at_1`` (computed here, never read from ``eval/all``, which also
      pools any dev split). pass@k is combined the same way.
    * AppWorld: ``test_normal`` (``eval/appworld_test_normal/pass_at_1``; W&B alias
      ``AppWorld/eval/test_normal/pass_at_1``).
    * tau2: ``retail_base`` (``eval/retail/environment/success``).
    pass@k is the largest ``pass_at_<N>`` (N > 1) logged under the same prefix.

BIAS WARNING: the eval pools are the benchmarks' held-out TEST pools, so a window selected on them is an
optimistic (selected-on-test) number. The final-step score is printed next to it for that reason.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from opsd import _repo

N_SEEN, N_UNSEEN = 140, 134
PRIMARY = {"alfworld": "combined", "appworld": "test_normal", "tau2": "retail_base"}
SPLITS = {
    "alfworld": ("valid_seen", "valid_unseen", "combined"),
    "appworld": ("test_normal",),
    "tau2": ("retail_base",),
}
# split -> candidate metric prefixes (first one present wins) and the pass@1 leaf.
PREFIXES = {
    "alfworld": {"valid_seen": ("eval/alfworld/",), "valid_unseen": ("eval/alfworld_valid_unseen/",)},
    "appworld": {"test_normal": ("eval/appworld_test_normal/", "AppWorld/eval/test_normal/")},
    "tau2": {"retail_base": ("eval/retail/",)},
}
P1_LEAF = {"alfworld": "pass_at_1", "appworld": "pass_at_1", "tau2": "environment/success"}
BIAS_NOTE = (
    "NOTE: the window is selected on the benchmark's held-out test pool (optimistic); "
    "report the final-step score alongside it."
)


class MetricError(Exception):
    pass


def repo_root(start: Optional[Path] = None) -> Path:
    """The repository root (holds ``pyproject.toml``, ``opsd/`` and ``assets/``); see ``opsd._repo``."""
    try:
        return _repo.repo_root(start)
    except FileNotFoundError as e:
        raise MetricError(f"repository root not found: {e}") from None


# ---------------------------------------------------------------------------------------- data model
@dataclass
class Score:
    p1: float
    pk: Optional[float] = None
    k: Optional[int] = None


Curve = Dict[int, Dict[str, Score]]  # step -> split -> Score


def _passk(metrics: Mapping[str, float], prefix: str) -> Tuple[Optional[int], Optional[float]]:
    best = None
    for key, val in metrics.items():
        m = re.fullmatch(re.escape(prefix) + r"pass_at_(\d+)", key)
        if m and int(m.group(1)) > 1 and val is not None and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), float(val))
    return best if best else (None, None)


def scores_from_metrics(
    metrics: Mapping[str, float],
    bench: str,
    p1_leaf: Optional[str] = None,
    n_seen: int = N_SEEN,
    n_unseen: int = N_UNSEEN,
) -> Dict[str, Score]:
    """Per-split scores from one eval-metrics dict (an aggregated_results.jsonl line or a W&B row)."""
    leaf = p1_leaf or P1_LEAF[bench]
    out: Dict[str, Score] = {}
    for split, prefixes in PREFIXES[bench].items():
        for pre in prefixes:
            v = metrics.get(pre + leaf)
            if v is not None and v == v:  # not NaN
                k, pk = _passk(metrics, pre)
                out[split] = Score(float(v), pk, k)
                break
    if bench == "alfworld" and "valid_seen" in out and "valid_unseen" in out:
        s, u = out["valid_seen"], out["valid_unseen"]
        n = n_seen + n_unseen
        pk = (
            (n_seen * s.pk + n_unseen * u.pk) / n
            if (s.k and s.k == u.k and s.pk is not None and u.pk is not None)
            else None
        )
        out["combined"] = Score((n_seen * s.p1 + n_unseen * u.p1) / n, pk, s.k if pk is not None else None)
    return out


def detect_bench(keys: Iterable[str]) -> Optional[str]:
    keys = list(keys)
    for bench, splits in PREFIXES.items():
        for prefixes in splits.values():
            if any(k.startswith(p) for p in prefixes for k in keys):
                return bench
    return None


# ------------------------------------------------------------------------------------------- loaders
def _dump_dir(path: Path) -> Path:
    for cand in (path / "exports" / "dumped_evals", path / "dumped_evals", path):
        if cand.is_dir() and (any(cand.glob("global_step_*_evals")) or (cand / "eval_only").is_dir()):
            return cand
    raise MetricError(
        f"{path}: no dumped_evals/global_step_*_evals/ found (expected <run_dir>/exports/dumped_evals, or "
        "<run_dir>/exports_ckpt_<N>/dumped_evals for a checkpoint re-score)"
    )


def _dump_dirs(path: Path) -> List[Path]:
    """The eval-dump dir(s) of a run dir: the one :func:`_dump_dir` finds, else every
    ``exports_ckpt_<N>/dumped_evals`` of a checkpoint re-score (``EVAL_CKPT_STEPS`` writes one export dir per
    step; their steps are distinct, so they form one curve)."""
    try:
        return [_dump_dir(path)]
    except MetricError:
        dirs = [d for d in sorted(path.glob("exports_ckpt_*/dumped_evals")) if any(d.glob("global_step_*_evals"))]
        if dirs:
            return dirs
        raise


def load_run_dir(path: Path, bench: Optional[str] = None, p1_leaf: Optional[str] = None) -> Tuple[str, Curve]:
    path = Path(path)
    dirs = _dump_dirs(path)
    where = dirs[0] if len(dirs) == 1 else path
    raw: Dict[int, dict] = {}
    seen_in: Dict[int, Path] = {}
    for d in dirs:
        for f in glob.glob(str(d / "global_step_*_evals" / "aggregated_results.jsonl")):
            m = re.search(r"global_step_(\d+)_evals", f)
            with open(f) as fh:
                line = fh.readline()
            if m and line.strip():
                step = int(m.group(1))
                if step in seen_in and seen_in[step] != d:
                    raise MetricError(f"{path}: step {step} is evaluated in both {seen_in[step]} and {d}")
                raw[step] = json.loads(line)
                seen_in[step] = d
        eo = d / "eval_only" / "aggregated_results.jsonl"
        if 0 not in raw and eo.is_file() and eo.read_text().strip():
            raw[0] = json.loads(eo.read_text().splitlines()[0])  # EVAL_ONLY run: the untrained student
    if not raw:
        raise MetricError(f"{where}: no aggregated_results.jsonl")
    # A dev200-protocol ALFWorld arm validates on the dev split only while training; its test-pool numbers
    # come from the checkpoint re-score, so its training run dir has nothing this metric reads.
    dev_only = (
        " (only eval/alfworld_dev/* here: a dev200-protocol run's valid_seen/valid_unseen numbers come from "
        "its checkpoint re-score, --run-dir <training run dir>__ckpteval)"
        if any(k.startswith("eval/alfworld_dev/") for m in raw.values() for k in m)
        else ""
    )
    bench = bench or detect_bench(k for m in raw.values() for k in m)
    if bench is None:
        raise MetricError(f"{where}: cannot tell the benchmark from the metric keys; pass --bench{dev_only}")
    curve = {s: sc for s, sc in ((s, scores_from_metrics(m, bench, p1_leaf)) for s, m in sorted(raw.items())) if sc}
    if not curve:
        raise MetricError(f"{where}: no {bench} {'/'.join(SPLITS[bench])} metrics in the eval dumps{dev_only}")
    return bench, curve


def load_curves_csv(path: Path, bench: Optional[str] = None) -> Tuple[str, Curve]:
    path = Path(path)
    if bench is None:
        stem = path.name.split("__", 1)[0]
        bench = stem if stem in PRIMARY else (path.parent.name if path.parent.name in PRIMARY else None)
    if bench is None:
        raise MetricError(f"{path}: cannot tell the benchmark from the file name; pass --bench")
    curve: Curve = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            pk = r.get("pass_at_k") or ""
            k = r.get("k") or ""
            curve.setdefault(int(r["step"]), {})[r["split"]] = Score(
                float(r["pass_at_1"]), float(pk) if pk else None, int(k) if k else None
            )
    if bench == "alfworld":
        for st, d in curve.items():
            if "combined" not in d and "valid_seen" in d and "valid_unseen" in d:
                s, u = d["valid_seen"], d["valid_unseen"]
                n = N_SEEN + N_UNSEEN
                pk = (
                    (N_SEEN * s.pk + N_UNSEEN * u.pk) / n
                    if (s.pk is not None and u.pk is not None and s.k == u.k)
                    else None
                )
                d["combined"] = Score((N_SEEN * s.p1 + N_UNSEEN * u.p1) / n, pk, s.k if pk is not None else None)
    return bench, dict(sorted(curve.items()))


def load_wandb(run_path: str, bench: str, p1_leaf: Optional[str] = None) -> Curve:  # pragma: no cover - network
    """One W&B run (``entity/project/run_id``) -> curve. Needs ``wandb`` and read access to the run."""
    try:
        import wandb
    except ImportError as e:
        raise MetricError("--wandb needs the `wandb` package (pip install wandb)") from e
    run = wandb.Api().run(run_path)
    keys = [k for k in run.summary.keys() if any(k.startswith(p) for ps in PREFIXES[bench].values() for p in ps)]
    if not keys:
        raise MetricError(f"{run_path}: no eval keys for {bench} in the run summary")
    rows: Dict[int, dict] = {}
    for key in keys:  # one key per scan: scan_history(keys=[...]) ANDs its keys
        for row in run.scan_history(keys=["_step", key]):
            if row.get(key) is not None and row.get("_step") is not None:
                rows.setdefault(int(row["_step"]), {})[key] = row[key]
    return {s: sc for s, sc in ((s, scores_from_metrics(m, bench, p1_leaf)) for s, m in sorted(rows.items())) if sc}


def stitch(segments: Sequence[Curve], rule: str = "merge") -> Curve:
    """Combine the curves of a resumed run, oldest first. ``merge`` (default, the rule the paper's tables
    used): the later segment wins a step both segments evaluated, and every other step is kept.
    ``chain``: a later segment also drops every earlier-segment step from its own first eval on (for a
    resume that rewound to an older checkpoint and did not re-evaluate every step after it). The two only
    differ when a resumed segment lacks a step the earlier segment evaluated after the resume point."""
    out: Curve = {}
    for seg in segments:
        if not seg:
            continue
        if rule == "chain":
            first = min(seg)
            out = {s: v for s, v in out.items() if s < first}
        out.update(seg)
    return dict(sorted(out.items()))


# ----------------------------------------------------------------------------------------- statistics
@dataclass
class Window:
    mean: Optional[float]
    steps: List[int] = field(default_factory=list)
    center: Optional[int] = None
    complete: bool = False


def best_window(series: Sequence[Tuple[int, float]], window: int = 3) -> Window:
    """Highest mean over ``window`` consecutive points of ``[(step, value), ...]`` (already step-sorted).
    Ties keep the earliest window; with fewer points, the single best point (``complete=False``)."""
    vals = [(s, v) for s, v in series if v is not None]
    if not vals:
        return Window(None)
    if len(vals) < window:
        s, v = max(vals, key=lambda sv: sv[1])
        return Window(v, [s], s, False)
    best_i, best_m = 0, float("-inf")
    for i in range(len(vals) - window + 1):
        m = sum(v for _, v in vals[i : i + window]) / window
        if m > best_m:
            best_i, best_m = i, m
    win = vals[best_i : best_i + window]
    return Window(best_m, [s for s, _ in win], win[window // 2][0], True)


def _series(
    curve: Curve, split: str, attr: str, lo: Optional[int], hi: Optional[int], include_step0: bool
) -> List[Tuple[int, Optional[float]]]:
    out = []
    for s in sorted(curve):
        if (s == 0 and not include_step0) or (lo is not None and s < lo) or (hi is not None and s > hi):
            continue
        sc = curve[s].get(split)
        if sc is not None:
            out.append((s, getattr(sc, attr)))
    return out


def summarize(
    curve: Curve,
    bench: str,
    window: int = 3,
    include_step0: bool = False,
    min_step: Optional[int] = None,
    max_step: Optional[int] = None,
    mean_steps: Optional[Tuple[int, int]] = None,
    split: Optional[str] = None,
) -> dict:
    """All statistics the paper's tables use, as fractions (0-1)."""
    split = split or PRIMARY[bench]
    p1 = _series(curve, split, "p1", min_step, max_step, include_step0)
    pk = _series(curve, split, "pk", min_step, max_step, include_step0)
    pk_ok = bool(pk) and all(v is not None for _, v in pk)
    w1 = best_window(p1, window)
    wk = best_window(pk, window) if pk_ok else Window(None)
    ks = {curve[s][split].k for s, _ in pk if curve[s][split].k}
    out = {
        "bench": bench,
        "split": split,
        "window": window,
        "n_points": len(p1),
        "steps": [s for s, _ in p1],
        "best_window_pass@1": w1.mean,
        "best_window_steps": w1.steps,
        "best_window_center": w1.center,
        "window_complete": w1.complete,
        "best_window_pass@k": wk.mean,
        "best_window_pass@k_steps": wk.steps,
        "k": ks.pop() if len(ks) == 1 else None,
        "final_step": p1[-1][0] if p1 else None,
        "final_pass@1": p1[-1][1] if p1 else None,
        "final_pass@k": pk[-1][1] if pk_ok else None,
        "full_run_pass@1": max((v for _, v in p1), default=None),
        "full_run_pass@k": max((v for _, v in pk), default=None) if pk_ok else None,
        "step0_pass@1": curve[0][split].p1 if 0 in curve and split in curve[0] else None,
        "step0_pass@k": curve[0][split].pk if 0 in curve and split in curve[0] else None,
    }
    if mean_steps is not None:
        lo, hi = mean_steps
        sel = [v for s, v in _series(curve, split, "p1", lo, hi, True)]
        out["mean_steps"] = [lo, hi]
        out["mean_pass@1"] = sum(sel) / len(sel) if sel else None
        out["mean_n"] = len(sel)
    return out


def value_at(curve: Curve, bench: str, step: int, attr: str = "p1", split: Optional[str] = None) -> Optional[float]:
    sc = curve.get(step, {}).get(split or PRIMARY[bench])
    return None if sc is None else getattr(sc, attr)


# ------------------------------------------------------------------------------------------------ CLI
def load_from_args(a) -> Tuple[str, Curve, str]:
    """-> (bench, curve, description) from the parsed CLI arguments ."""
    bench = a.bench
    if getattr(a, "run_dir", None):
        segs = []
        for d in a.run_dir:
            b, c = load_run_dir(Path(d), bench, a.p1_leaf)
            bench = bench or b
            segs.append(c)
        return bench, stitch(segs, a.stitch), " + ".join(a.run_dir)
    if getattr(a, "curves_csv", None):
        bench, c = load_curves_csv(Path(a.curves_csv), bench)
        return bench, c, a.curves_csv
    if getattr(a, "wandb", None):
        if not bench:
            raise MetricError("--wandb needs --bench")
        return bench, stitch([load_wandb(p, bench, a.p1_leaf) for p in a.wandb], a.stitch), " + ".join(a.wandb)
    raise MetricError("give one of --run-dir, --curves-csv or --wandb")


def add_source_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--bench", choices=sorted(PRIMARY), help="benchmark (auto-detected when possible)")
    src = ap.add_argument_group("source (one of)")
    src.add_argument(
        "--run-dir",
        action="append",
        help="run dir (or export path; a checkpoint re-score's <run>__ckpteval dir reads every exports_ckpt_<N>); "
        "repeat for resume segments",
    )
    src.add_argument("--curves-csv", help="curve CSV (step,split,pass_at_1,pass_at_k,k)")
    src.add_argument(
        "--wandb",
        action="append",
        metavar="ENTITY/PROJECT/RUN_ID",
        help="W&B run path; repeat for resume segments, oldest first",
    )
    ap.add_argument(
        "--stitch",
        choices=("merge", "chain"),
        default="merge",
        help="how to combine several segments, oldest first (default merge, as in the paper's tables: the "
        "later segment wins a shared step; chain: it also drops the earlier segment's later steps)",
    )
    ap.add_argument(
        "--p1-leaf", default=None, help="metric leaf used as pass@1 (default: pass_at_1; tau2: environment/success)"
    )


def _parse_range(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    m = re.fullmatch(r"(\d+)-(\d+)", s)
    if not m:
        raise argparse.ArgumentTypeError("expected A-B, e.g. 100-200")
    return int(m.group(1)), int(m.group(2))


def _fmt(v: Optional[float]) -> str:
    return "   -  " if v is None else f"{100 * v:6.2f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m opsd.eval.window_metric",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_source_args(ap)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--include-step0", action="store_true", help="let step 0 enter the window (paper: excluded)")
    ap.add_argument("--min-step", type=int)
    ap.add_argument("--max-step", type=int, help="only evals up to this step (e.g. the paper's last step)")
    ap.add_argument("--mean-steps", type=_parse_range, help="also report the mean pass@1 over steps A-B")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args(argv)
    try:
        bench, curve, desc = load_from_args(a)
    except (MetricError, OSError, ValueError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    s = summarize(curve, bench, a.window, a.include_step0, a.min_step, a.max_step, a.mean_steps)
    if a.json:
        print(json.dumps(s, indent=2))
        return 0
    prim = PRIMARY[bench]
    print(f"source: {desc}")
    print(f"bench: {bench}   score: {prim}   window: {a.window}   evals in window pool: {s['n_points']}")
    splits = [sp for sp in SPLITS[bench] if any(sp in d for d in curve.values())]
    print(f"\n{'step':>6} " + " ".join(f"{sp + ' p@1':>18} {'p@k':>7}" for sp in splits))
    for st in sorted(curve):
        print(
            f"{st:6d} "
            + " ".join(
                f"{_fmt(value_at(curve, bench, st, 'p1', sp)):>18} " f"{_fmt(value_at(curve, bench, st, 'pk', sp)):>7}"
                for sp in splits
            )
        )
    k = s["k"] or "k"
    note = "" if s["window_complete"] else "   (fewer evals than the window -> single best eval)"
    print(
        f"\nbest {a.window}-eval window pass@1: {_fmt(s['best_window_pass@1'])}  steps {s['best_window_steps']} "
        f"(centre {s['best_window_center']}){note}"
    )
    print(
        f"best {a.window}-eval window pass@{k}: {_fmt(s['best_window_pass@k'])}  steps {s['best_window_pass@k_steps']}"
    )
    print(f"final (step {s['final_step']}) pass@1: {_fmt(s['final_pass@1'])}   pass@{k}: {_fmt(s['final_pass@k'])}")
    print(f"full run (max) pass@1: {_fmt(s['full_run_pass@1'])}   pass@{k}: {_fmt(s['full_run_pass@k'])}")
    if s["step0_pass@1"] is not None:
        print(f"step 0 pass@1: {_fmt(s['step0_pass@1'])}   pass@{k}: {_fmt(s['step0_pass@k'])}")
    if "mean_pass@1" in s:
        lo, hi = s["mean_steps"]
        print(f"mean pass@1 over steps {lo}-{hi} ({s['mean_n']} evals): {_fmt(s['mean_pass@1'])}")
    print(BIAS_NOTE)
    return 0


if __name__ == "__main__":
    sys.exit(main())

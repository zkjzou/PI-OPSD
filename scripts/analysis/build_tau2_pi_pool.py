#!/usr/bin/env python
"""Cut a τ²-retail training manifest down to the tasks a PI teacher can actually teach, and emit the
pool-scoped hint files that go with it.

WHY A SEPARATE BUILDER. The student-success slice that produced
retail_areal_instruct_{drop8of8,band}.jsonl (from run_tau2_instruct_profile563.sh's profile; that
slicing script is not part of this release) has no hint awareness; `build_tau2_retail_pi.py`
merges/scopes hints but only ever writes JSON, never a manifest. Nothing joins the two, so there was
no way to say "train on exactly the tasks that have a teacher trajectory". That set is **424** of the
478-task Instruct pool after 16 glm-5p2 trials.

WHAT THIS BUYS, AND WHAT IT DOESN'T. Every task in the output has a demonstrated-solvable ceiling, so
under OPSD every rollout gets a real teacher. Under plain GRPO it is just a pool with a solvability
floor — the 54 dropped tasks were the ones no teacher ever solved, 37 of which the untrained student
also never solves, i.e. neither GRPO nor PI could teach them.

⚠️ ALL THREE FORMATS MUST COVER THE SAME TASK IDS or the format comparison silently compares POOLS
instead of hint styles. They are cut from the same winning trajectories so they do today; the assert
keeps it that way after any hint rebuild.

⚠️ THE TOKEN REPORT IS LOAD-BEARING, NOT DECORATION. OPSD's generator drops a rollout's hint when
`hint + prompt + response > OPSD_REF_SEQ_CAP`, silently, degrading the arm toward a no-op with no
error. The default cap is `max_input_length + max_generate_length` = 39,000, and against τ²'s hard
bounds (`MAX_PROMPT_LENGTH` 6,144 + `tau2.max_response_tokens` 33,000 = 39,144) **every format
overflows in the worst case, including `actions`**. `--ref-seq-cap` re-checks the headroom here, at
build time, instead of finding out from a flat curve 100 steps in.

Note this is the TEACHER-side budget, not the student's. τ²'s ~4.7k-token system prompt against
`MAX_PROMPT_LENGTH=6144` is what makes `raw` unusable for injecting a hint into the *student's* prompt
(see `build_tau2_retail_pi.py`) — OPSD never does that, so all three formats are in play here.

Usage (P = the run_tau2_pi_all.sh output root, one dir per teacher with hints_{train,dev}/):
    .venv/bin/python scripts/analysis/build_tau2_pi_pool.py \\
        --hints-root $P/glm_retail \\
        --parent opsd/tau2_bench/data/retail_areal_instruct_drop8of8.jsonl \\
        --out retail_pi424.jsonl --hint-out-dir ./outputs/pi_tau2_pool424
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    """The repository root: the nearest ancestor of this file holding pyproject.toml, opsd/__init__.py and
    opsd/skyrl_ext/__init__.py (the predicate of opsd._repo)."""
    for d in Path(__file__).resolve().parents:
        if all((d / m).is_file() for m in ("pyproject.toml", "opsd/__init__.py", "opsd/skyrl_ext/__init__.py")):
            return d
    raise SystemExit(f"cannot locate the repository root (pyproject.toml + opsd/skyrl_ext/) above {__file__}")


REPO = _repo_root()
DATA = REPO / "opsd/tau2_bench/data"

# Run-by-path puts THIS file's dir on sys.path, not the repo root, so `import opsd.*` fails. Bootstrap
# rather than requiring PYTHONPATH, so the script works the same way the neighbouring ones are invoked.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from opsd.pi.artifact import SCHEMA, load_task_hints  # noqa: E402

FORMATS = ("actions", "raw_no_think", "raw")
# Both halves of the old train/dev manifest split; a task's hint may live in either, and scoping to the
# training pool afterwards is what makes mixing them safe.
SPLITS = ("hints_train", "hints_dev")


def load_format(root: Path, fmt: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for split in SPLITS:
        path = root / split / f"hints_{fmt}.json"
        if path.is_file():
            out.update(load_task_hints(str(path)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--hints-root", required=True, type=Path, help="teacher dir holding hints_{train,dev}/hints_<fmt>.json"
    )
    ap.add_argument("--parent", type=Path, default=DATA / "retail_areal_instruct_drop8of8.jsonl")
    ap.add_argument("--out", default="retail_pi424.jsonl", help="filename, written into the data dir")
    ap.add_argument("--hint-out-dir", type=Path, help="write pool-scoped hints_<fmt>.json here")
    ap.add_argument("--formats", nargs="+", default=list(FORMATS))
    ap.add_argument(
        "--key-format", default="actions", help="the format whose task ids define the pool (all others must match it)"
    )
    ap.add_argument(
        "--ref-seq-cap",
        type=int,
        default=52000,
        help="OPSD_REF_SEQ_CAP the run will use; the headroom check is against this",
    )
    ap.add_argument("--max-prompt-tokens", type=int, default=6144)  # MAX_PROMPT_LENGTH
    ap.add_argument("--max-response-tokens", type=int, default=33000)  # tau2.max_response_tokens
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.parent.read_text().splitlines() if l.strip()]
    pool = {str(r["task_id"]) for r in rows}
    print(f"parent {args.parent.name}: {len(rows)} tasks")

    scoped: dict[str, dict[str, str]] = {}
    for fmt in args.formats:
        got = load_format(args.hints_root, fmt)
        scoped[fmt] = {k: v for k, v in got.items() if k in pool}
        print(f"  {fmt:14s} {len(got):>4} available, {len(scoped[fmt]):>4} inside the pool")

    keep_ids = set(scoped[args.key_format])
    for fmt, got in scoped.items():
        if set(got) != keep_ids:
            only_key, only_fmt = keep_ids - set(got), set(got) - keep_ids
            raise SystemExit(
                f"FATAL: '{fmt}' covers different tasks than '{args.key_format}' "
                f"(+{len(only_fmt)} / -{len(only_key)}). The three arms would train on different "
                f"POOLS, so a format difference would be unattributable. Rebuild the hints."
            )
    print(f"  all {len(args.formats)} formats cover an identical {len(keep_ids)} tasks ✓")

    # --- token headroom: the check that keeps a hint from being silently dropped mid-run ---------
    from transformers import AutoTokenizer

    from opsd.trainer.main_opsd_alfworld import SECRET_TEMPLATE

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    overhead = len(tok.encode(SECRET_TEMPLATE.format(hint_lines=""), add_special_tokens=False))
    budget = args.ref_seq_cap - args.max_prompt_tokens - args.max_response_tokens
    print(f"\ntoken report (tokenizer={args.tokenizer}, +{overhead} tok SECRET_TEMPLATE wrapper)")
    print(
        f"  worst-case hint budget = cap {args.ref_seq_cap} - prompt {args.max_prompt_tokens} "
        f"- response {args.max_response_tokens} = {budget}"
    )
    print(f"  {'format':14s} {'median':>7s} {'p95':>7s} {'p99':>7s} {'max':>7s}   headroom")
    worst = {}
    for fmt in args.formats:
        n = sorted(len(tok.encode(v, add_special_tokens=False)) + overhead for v in scoped[fmt].values())
        q = lambda f: n[min(len(n) - 1, int(f * len(n)))]  # noqa: E731
        worst[fmt] = n[-1]
        ok = "OK" if n[-1] <= budget else f"⚠️ OVER by {n[-1] - budget}"
        print(f"  {fmt:14s} {n[len(n)//2]:>7d} {q(.95):>7d} {q(.99):>7d} {n[-1]:>7d}   {ok}")
    over = {f: v for f, v in worst.items() if v > budget}
    if over:
        raise SystemExit(
            f"\nFATAL: {sorted(over)} exceed the worst-case hint budget. At runtime OPSD would DROP "
            f"the hint on the longest trajectories -- silently, with no error, degrading those arms "
            f"toward a no-op. Raise --ref-seq-cap (and OPSD_REF_SEQ_CAP on the run to match), or "
            f"rebuild the hints with a tighter --max-chars-raw."
        )

    # --- the manifest ----------------------------------------------------------------------------
    sub = [r for r in rows if str(r["task_id"]) in keep_ids]  # parent order preserved
    for i, r in enumerate(sub):  # validator checks density
        r["task_index"] = i
    out = DATA / args.out
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in sub))
    print(f"\nwrote {len(sub)} rows -> {out}")
    print(f"  dropped {len(rows) - len(sub)} tasks with no teacher trajectory")

    if args.hint_out_dir:
        args.hint_out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in args.formats:
            payload: dict[str, object] = {"__schema__": SCHEMA}
            payload.update({k: scoped[fmt][k] for k in sorted(keep_ids)})
            dest = args.hint_out_dir / f"hints_{fmt}.json"
            dest.write_text(json.dumps(payload, indent=1, sort_keys=True))
            print(f"  hints -> {dest} ({len(keep_ids)} entries, {dest.stat().st_size/1e6:.1f} MB)")
        # ⚠️ NAME THE DIRECTORY AFTER THE TEACHER. `pool424` is GLM's pool; writing another teacher's
        # pool into it OVERWRITES THE GLM HINTS that a finished 3-arm experiment depends on (this
        # misdirected twice while the qwen pool was being built).
        if "pool424" in str(args.hint_out_dir) and "glm" not in str(args.hints_root):
            print(
                "  ⚠️ WARNING: --hint-out-dir is glm's pool424 but --hints-root is "
                f"{args.hints_root} -- that would overwrite the glm hints."
            )

    print(
        "\n  NOTE: an ad-hoc slice, not a registered ManifestSpec, so `prepare_data validate` will "
        "reject it — same as tau2_train500/dev62/332. Launchers resolve a bare TRAIN_DATA filename "
        "against the in-repo data dir, so keep the written manifest there."
    )


if __name__ == "__main__":
    main()

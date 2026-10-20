#!/usr/bin/env python
"""Measure LITERAL-VALUE LEAKAGE in τ² PI hints, per hint family.

WHY THIS IS NOT OPTIONAL. τ² retail rewards are DB-state checks PLUS communicate-checks on exact
values, so a hint that restates an order id, an email, a payment-method id or a price is not a weaker
hint — it is the ANSWER. Utility measured from such a hint is a lookup, not a lesson, and it inflates
exactly the arm you would want to compare against a non-leaking one. The `onpolicy_skill` analyzer
prompt forbids restating literal ids/emails/answers; the `plain` summarizer prompt does NOT (measured
on the AppWorld plain run: 52/160 hints with 3+ digit ids, 35/160 with literal dollar amounts).

PATTERNS ARE READ OFF THE REAL CORPUS, NOT GUESSED. Each was verified to fire on
pi_tau2_pool424/hints_raw.json (hit counts over the first 40 trajectories):
    order #W…        1840      item/product id   2747
    email             136      user id            357
    dollar $n.nn     1607      gift_card_n        245
    credit_card_n     407      paypal_n            89
A pattern that matches nothing in the source cannot detect leakage FROM the source, so the
`--self-test` mode asserts every pattern still fires there before any hint is judged.

⚠️ THIS MEASURES RESTATEMENT, NOT HARM. A hint may name an order id that the student would have had to
look up anyway, or name one that is irrelevant to the reward. So read the rate as an UPPER BOUND on
contamination, and prefer the per-task overlap columns (`shared_with_source`) which count only values
that actually appear in that task's own reference trajectory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
from pathlib import Path
from typing import Dict, List

# Verified against pi_tau2_pool424/hints_raw.json — see the docstring.
PATTERNS: Dict[str, re.Pattern] = {
    "order_id": re.compile(r"#W\d{5,}"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.\w+"),
    "dollar": re.compile(r"\$\s?\d[\d,]*\.\d{2}"),
    "gift_card": re.compile(r"gift_card_\d+"),
    "credit_card": re.compile(r"credit_card_\d+"),
    "paypal": re.compile(r"paypal_\d+"),
    "item_id": re.compile(r"\b\d{10}\b"),
    "user_id": re.compile(r"\b[a-z]+_[a-z]+_\d{4}\b"),
}


def hint_text(rec) -> str:
    """opsd_hints/v2 records are {task, meta}; v1 records are bare strings."""
    if isinstance(rec, str):
        return rec
    if isinstance(rec, dict):
        return rec.get("task") or ""
    return ""


def _default_source() -> str:
    """$OPSD_DATA_ROOT/tau2/hints/glm-5.2/pool424/hints_raw.json (default data root ~/.cache/opsd/data)."""
    root = os.environ.get("OPSD_DATA_ROOT") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "opsd", "data"
    )
    return os.path.join(root, "tau2", "hints", "glm-5.2", "pool424", "hints_raw.json")


def load(path: str) -> Dict[str, str]:
    d = json.load(open(Path(path).expanduser()))
    return {k: hint_text(v) for k, v in d.items() if k != "__schema__"}


def hits(text: str) -> Dict[str, List[str]]:
    return {name: pat.findall(text) for name, pat in PATTERNS.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hints", action="append", required=True, help="NAME=path (repeatable), e.g. summ_glm=/path/summ_glm.json"
    )
    ap.add_argument(
        "--source",
        default=_default_source(),
        help="the reference trajectories the hints were written FROM (default: the published "
        "glm-5.2 pool424 hints_raw.json); enables the per-task shared_with_source column "
        "and the pattern self-test",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = {}
    if args.source and Path(args.source).expanduser().exists():
        src = load(args.source)
        # SELF-TEST: a pattern that no longer fires on the source cannot detect leakage from it.
        agg = "\n".join(list(src.values())[:60])
        dead = [n for n, p in PATTERNS.items() if not p.search(agg)]
        if dead:
            print(
                f"⚠️  PATTERN SELF-TEST FAILED — these never fire on the source, so any 0% they "
                f"report is MEANINGLESS: {dead}"
            )
        else:
            print(f"pattern self-test: all {len(PATTERNS)} patterns fire on the source ✓")
    else:
        print(f"⚠️  source not readable ({args.source}); reporting restatement only, no overlap column")

    report = {}
    rows = []
    for spec in args.hints:
        if "=" not in spec:
            raise SystemExit(f"--hints wants NAME=path, got {spec!r}")
        name, path = spec.split("=", 1)
        if not Path(path).expanduser().exists():
            print(f"  skip {name}: {path} not found")
            continue
        hs = load(path)
        n = len(hs)
        any_leak = 0
        per_pat = {k: 0 for k in PATTERNS}
        shared = 0
        n_shared_vals: List[int] = []
        for tid, t in hs.items():
            h = hits(t)
            found = {k: v for k, v in h.items() if v}
            if found:
                any_leak += 1
            for k in found:
                per_pat[k] += 1
            # overlap: only values that ALSO appear in this task's own reference trajectory
            if src and tid in src:
                stext = src[tid]
                vals = {v for vs in found.values() for v in vs}
                ov = {v for v in vals if v in stext}
                if ov:
                    shared += 1
                    n_shared_vals.append(len(ov))
        report[name] = {
            "n_hints": n,
            "hints_with_any_literal": any_leak,
            "rate_any": round(any_leak / max(n, 1), 4),
            "per_pattern_hint_counts": per_pat,
            "hints_sharing_a_value_with_own_source": shared,
            "rate_shared": round(shared / max(n, 1), 4),
            "median_shared_values_per_leaky_hint": (st.median(n_shared_vals) if n_shared_vals else 0),
        }
        rows.append((name, n, any_leak, any_leak / max(n, 1), shared, shared / max(n, 1)))

    print(f"\n{'family':14s} {'n':>5s} {'any-literal':>12s} {'rate':>7s} {'shared/src':>11s} {'rate':>7s}")
    for name, n, a, ra, s, rs in rows:
        print(f"{name:14s} {n:5d} {a:12d} {ra:7.1%} {s:11d} {rs:7.1%}")
    print("\nper-pattern hint counts:")
    for name in report:
        pp = {k: v for k, v in report[name]["per_pattern_hint_counts"].items() if v}
        print(f"  {name:14s} {pp or '(none)'}")
    print(
        "\n⚠️  `any-literal` is an UPPER BOUND on contamination (restatement != harm). "
        "`shared/src` counts only values that appear in that task's OWN reference trajectory."
    )

    if args.out:
        Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).expanduser().write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

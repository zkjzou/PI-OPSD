#!/usr/bin/env python
"""Information-density of each PI (a property of the hint text itself, not the divergence).

For every PI hint artifact, computes:
  * ``density_ordinal`` — coarse 1-4 category by format-family (1 = richest trajectory, 4 = control):
        trajectory (raw / raw_no_think) = 1, actions = 2, skill / abstracted = 3, static = 4.
    Inferred from the PI name; override with ``name=path:ordinal``.
  * ``hint_tok_len``     — mean # tokens per hint (student tokenizer).
  * ``hint_char_entropy`` — mean Shannon entropy (bits/char) of the hint text (character distribution).
  * ``n_tasks``          — # tasks with a hint.

Model/benchmark-agnostic: pass any tokenizer + any set of ``{task_id: hint}`` artifacts.

Usage:
  .venv/bin/python -m opsd.alfworld.pi_info_density \
    --tokenizer Qwen/Qwen3-4B-Thinking-2507 --out .../info_density.json \
    --pis "static=.../hints_static_generic.json,q37_raw=.../q37/hints_raw.json,..."
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from opsd.pi.artifact import load_task_hints


def ordinal_for(name: str) -> int:
    """Information-density tier, 1=richest → 5=control (task-independent).
    raw (reasoning+action+obs) < raw_no_think (action+obs) < actions < skill (abstracted) < static."""
    n = name.lower()
    if "static" in n:
        return 5
    if "skill" in n or "abstract" in n or "contrastive" in n:
        return 4
    if "action" in n:
        return 3
    if "no_think" in n or "nothink" in n:  # raw_no_think: BETWEEN raw-with-think and actions
        return 2
    if "raw" in n or "traj" in n:  # raw WITH thinking = richest
        return 1
    return 4  # default: mid-low density


def char_entropy_bits(text: str) -> float:
    if not text:
        return 0.0
    c = Counter(text)
    n = len(text)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def parse_pis(spec: str):
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, path = item.split("=", 1)
        ordinal = None
        if ":" in path and not path.strip().endswith(".json"):
            path, ord_s = path.rsplit(":", 1)
            ordinal = int(ord_s)
        yield name.strip(), path.strip(), ordinal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pis", required=True, help="comma list of name=hints_path[:ordinal].")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    report = {}
    print(f"{'PI':<24}{'ord':>4}{'tok_len':>9}{'char_H':>8}{'n_tasks':>9}")
    for name, path, ordinal in parse_pis(a.pis):
        hints = load_task_hints(path)
        texts = [str(v) for v in hints.values() if v]
        if not texts:
            report[name] = {
                "density_ordinal": ordinal or ordinal_for(name),
                "hint_tok_len": 0.0,
                "hint_char_entropy": 0.0,
                "n_tasks": 0,
            }
            continue
        tok_lens = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
        ents = [char_entropy_bits(t) for t in texts]
        rec = {
            "density_ordinal": ordinal if ordinal is not None else ordinal_for(name),
            "hint_tok_len": round(sum(tok_lens) / len(tok_lens), 2),
            "hint_char_entropy": round(sum(ents) / len(ents), 4),
            "n_tasks": len(texts),
        }
        report[name] = rec
        print(
            f"{name:<24}{rec['density_ordinal']:>4}{rec['hint_tok_len']:>9.1f}"
            f"{rec['hint_char_entropy']:>8.3f}{rec['n_tasks']:>9}"
        )

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

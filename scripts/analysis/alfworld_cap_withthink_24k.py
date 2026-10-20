#!/usr/bin/env python
"""Cap the uncapped Qwen3.7-Plus withthink trajectory hints at 24k student tokens.

Reads alfworld_rebuild_hints.py's REBUILT_qwen37__traj_withthink.json and writes
hints_traj_withthink_cap24k.json (the published qwen37 glw/glr hint file): a hint longer than CAP tokens
(Qwen3-4B-Thinking-2507 tokenizer, no special tokens) keeps its first and last tokens around the
"...[trajectory truncated]..." marker, re-decoded to text; meta gains token_cap. The other three published
cap24k files come from alfworld_cap_many_24k.py, whose capping is the same. The capping logic is unchanged
from the build that produced the published file; only the paths are arguments. The tokenizer is loaded
from the Hugging Face cache ($HF_HOME).

  python scripts/analysis/alfworld_cap_withthink_24k.py \\
      --src outputs/alf_hint_render/REBUILT_qwen37__traj_withthink.json \\
      --dst outputs/alf_hint_render/hints_traj_withthink_cap24k.json [--check-md5 <expected file>]

--check-md5 FILE byte-compares the written --dst with FILE (exit 1 on a mismatch); --check-content FILE
compares the parsed records (a re-serialised copy of the same hints, e.g. the data bundle's
alfworld/hints/qwen3.7-plus/hints_traj_withthink_cap24k.json, passes this but not the md5 check).
"""

import argparse
import hashlib
import json
import sys

from transformers import AutoTokenizer

SRC = "REBUILT_qwen37__traj_withthink.json"
DST = "hints_traj_withthink_cap24k.json"
CAP = 24000
MARKER = "\n\n...[trajectory truncated]...\n\n"


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--check-md5", default=None, metavar="FILE")
    ap.add_argument("--check-content", default=None, metavar="FILE")
    a = ap.parse_args()
    cap = a.cap

    # ---- capping: unchanged from the build that produced the published hints ------------------------
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    marker_ids = tok.encode(MARKER, add_special_tokens=False)
    head_budget = (cap - len(marker_ids)) // 2
    tail_budget = cap - len(marker_ids) - head_budget

    d = json.load(open(a.src))
    out = {"__schema__": "opsd_hints/v2"}
    n = 0
    n_trunc = 0
    lens_after = []
    for k, v in d.items():
        if str(k).startswith("__"):
            continue
        n += 1
        task = v["task"]
        ids = tok.encode(task, add_special_tokens=False)
        if len(ids) > cap:
            n_trunc += 1
            new_ids = ids[:head_budget] + marker_ids + ids[-tail_budget:]
            task = tok.decode(new_ids)
        lens_after.append(len(tok.encode(task, add_special_tokens=False)))
        meta = dict(v.get("meta", {}))
        meta["token_cap"] = cap
        out[k] = {"task": task, "meta": meta}

    json.dump(out, open(a.dst, "w"))
    lens_after.sort()
    mx = lens_after[-1]
    avg = sum(lens_after) / len(lens_after)
    print(
        f"wrote {a.dst}: n={n} truncated={n_trunc} ({100*n_trunc/n:.1f}%) "
        f"avg_after={avg:.0f} max_after={mx} cap={cap}",
        flush=True,
    )
    print("DONE", flush=True)

    ok = True
    if a.check_md5:
        same = md5(a.dst) == md5(a.check_md5)
        print(f"  [{'OK  ' if same else 'FAIL'}] md5 {md5(a.dst)} vs {md5(a.check_md5)} ({a.check_md5})")
        ok &= same
    if a.check_content:
        same = json.load(open(a.dst)) == json.load(open(a.check_content))
        print(f"  [{'OK  ' if same else 'FAIL'}] records {'equal' if same else 'differ'} ({a.check_content})")
        ok &= same
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

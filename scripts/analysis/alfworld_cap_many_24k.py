#!/usr/bin/env python
"""Cap uncapped ALFWorld trajectory hints at 24k student tokens (head + tail around a truncation marker).

Each job reads an alfworld_rebuild_hints.py artifact and writes its *_cap24k counterpart: a hint longer than
CAP tokens (Qwen3-4B-Thinking-2507 tokenizer, no special tokens) keeps its first and last tokens around the
"...[trajectory truncated]..." marker, re-decoded to text; meta gains token_cap. The default jobs are the
three this script produced for the published hints (glm withthink / nothink, qwen37 nothink); qwen37
withthink is alfworld_cap_withthink_24k.py. The capping logic is unchanged from that build; only the paths
are arguments. The tokenizer is loaded from the Hugging Face cache ($HF_HOME).

  python scripts/analysis/alfworld_cap_many_24k.py --in-dir outputs/alf_hint_render \\
      --out-dir outputs/alf_hint_render [--job SRC=DST ...] \\
      [--check-md5 glm__hints_traj_withthink_cap24k.json=<expected file> ...]

--check-md5 DST=FILE byte-compares a written DST with FILE (exit 1 on a mismatch); --check-content DST=FILE
compares the parsed records (a re-serialised copy of the same hints, e.g. the data bundle's
alfworld/hints/glm-5.2/hints_traj_withthink_cap24k.json, passes this but not the md5 check).
"""

import argparse
import hashlib
import json
import os
import sys

from transformers import AutoTokenizer

CAP = 24000
MARKER = "\n\n...[trajectory truncated]...\n\n"
JOBS = [
    ("REBUILT_glm__traj_withthink.json", "glm__hints_traj_withthink_cap24k.json"),
    ("REBUILT_glm__traj_nothink.json", "glm__hints_traj_nothink_cap24k.json"),
    ("REBUILT_qwen37__traj_nothink.json", "qwen37__hints_traj_nothink_cap24k.json"),
]


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(out_dir: str, specs, content: bool) -> bool:
    ok = True
    for spec in specs:
        name, expected = spec.split("=", 1)
        got = os.path.join(out_dir, name)
        if content:
            same = json.load(open(got)) == json.load(open(expected))
            detail = "records equal" if same else "records differ"
        else:
            same = md5(got) == md5(expected)
            detail = f"md5 {md5(got)} vs {md5(expected)}"
        print(f"  [{'OK  ' if same else 'FAIL'}] {name} vs {expected}: {detail}", flush=True)
        ok &= same
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=".", help="dir of the REBUILT_*.json inputs")
    ap.add_argument("--out-dir", default=".", help="dir for the *_cap24k.json outputs")
    ap.add_argument(
        "--job",
        action="append",
        default=None,
        metavar="SRC=DST",
        help="input -> output basename (repeatable; default the three published jobs)",
    )
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--check-md5", action="append", default=[], metavar="DST=FILE")
    ap.add_argument("--check-content", action="append", default=[], metavar="DST=FILE")
    a = ap.parse_args()
    jobs = [tuple(j.split("=", 1)) for j in a.job] if a.job else JOBS
    os.makedirs(a.out_dir, exist_ok=True)
    cap = a.cap

    # ---- capping: unchanged from the build that produced the published hints ------------------------
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    marker_ids = tok.encode(MARKER, add_special_tokens=False)
    head = (cap - len(marker_ids)) // 2
    tail = cap - len(marker_ids) - head

    for src, dst in jobs:
        d = json.load(open(os.path.join(a.in_dir, src)))
        out = {"__schema__": "opsd_hints/v2"}
        n = ntr = 0
        mx = 0
        for k, v in d.items():
            if str(k).startswith("__"):
                continue
            n += 1
            task = v["task"]
            ids = tok.encode(task, add_special_tokens=False)
            if len(ids) > cap:
                ntr += 1
                ids = ids[:head] + marker_ids + ids[-tail:]
                task = tok.decode(ids)
                ids = tok.encode(task, add_special_tokens=False)
            mx = max(mx, len(ids))
            meta = dict(v.get("meta", {}))
            meta["token_cap"] = cap
            out[k] = {"task": task, "meta": meta}
        json.dump(out, open(os.path.join(a.out_dir, dst), "w"))
        print(f"{dst}: n={n} truncated={ntr} ({100*ntr/n:.1f}%) max_after={mx}", flush=True)
    print("DONE", flush=True)

    ok_md5 = check(a.out_dir, a.check_md5, content=False)
    ok_content = check(a.out_dir, a.check_content, content=True)
    return 0 if ok_md5 and ok_content else 1


if __name__ == "__main__":
    sys.exit(main())

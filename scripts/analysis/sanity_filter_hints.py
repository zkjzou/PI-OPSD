#!/usr/bin/env python
"""Flag (and optionally drop) degenerate hints in an OPSD hint artifact.

WHY. A thinking analyzer can fail in a way that LOOKS like content. One `hints_summarized` cell held a
6,072-word hint against a 15-word median: the model's chain-of-thought leaked in ("We are given a
trajectory... We are to write a concise hint...") because it never emitted `</think>` for the stripper
to find, and it then degenerated into `... [AGENT] ... [OBSERVATION] ...` repeated to the token cap.
Coverage checks pass on that hint. Length checks do not.

THE THRESHOLD IS PER-CELL AND RELATIVE, NOT ABSOLUTE. These cells differ by design -- `onpolicy_skill`
runs ~52 median words and `plain` ~15, and a frontier author's `onpolicy` cell legitimately reaches 198
words. A fixed word cap would either miss the 200-word leak in a 50-word cell or condemn 1,360 healthy
hints in a 114-word one. So the cut is a multiple of THAT artifact's own median.

Detectors, all three of which the observed failures trip:
  length     > --max-ratio x the artifact's median word count (default 5)
  cot-leak   first-person meta-commentary about the writing task itself
  repetition a transcript-marker or ellipsis loop, the signature of running to the token cap

  python scripts/analysis/sanity_filter_hints.py hints.json [more.json ...]        # report
  python scripts/analysis/sanity_filter_hints.py hints.json --write-todo todo.json # ids to redo
  python scripts/analysis/sanity_filter_hints.py hints.json --drop --out clean.json

--drop removes the flagged keys so a regeneration pass can refill them: the merge step in
fill_self_hints.sh is "existing wins, fill only closes gaps", so a corrupt hint that stays present
is never replaced. Dropping is what turns it back into a gap.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# First-person commentary about the *writing task*, which a hint should never contain. Kept narrow:
# "the agent should..." is legitimate hint content, "I need to provide a hint..." is leaked reasoning.
#
# NARROWED AFTER A FALSE POSITIVE. An earlier version included `the (problem|prompt|task) says`, which
# condemned a perfectly healthy hint reading "...without opening it if the task says 'on'" -- hints
# legitimately quote the task. Every marker here must be the analyzer talking about ITS OWN job, in
# first person or as an explicit meta-frame; nothing that a hint could plausibly say about the agent.
COT_LEAK = re.compile(
    r"we are (given|to write|asked|told)|"
    r"i (need|have) to (provide|write|produce|summarize|generate|analyze)|"
    r"i (should|will) (write|produce|summarize|generate)|"
    r"let me (think|write|start|analyze)|as an ai|"
    r"<think>|</think>|the user (wants|asks)|here is (the|my) (hint|summary)",
    re.I,
)
# A marker or ellipsis loop -- what running out of budget mid-degeneration looks like.
REPETITION = re.compile(r"(\[AGENT\]|\[OBSERVATION\])\s*(\.\.\.\s*)?(\[AGENT\]|\[OBSERVATION\])" r"|(\.\.\.\s*){4,}")


def hint_text(v):
    if isinstance(v, dict):
        return v.get("task") or ""
    return v if isinstance(v, str) else ""


# A DEMONSTRATION artifact is not a written hint, and the cot-leak/repetition detectors do not apply.
# `hints_win_headtail.json` (the length-matched control) is a truncated frontier TRANSCRIPT: the
# teacher's own first-person reasoning ("Let me start by searching for a pan") is the content, and
# [AGENT]/[OBSERVATION] markers repeat by construction. Running the prose detectors over it flagged
# 21 of 1,583 healthy entries. Length is still meaningful, so that check is kept.
TRANSCRIPT_MARKERS = re.compile(r"\[AGENT\]|\[OBSERVATION\]|<action>")


def scan(path, max_ratio=5.0, min_words=3):
    d = json.loads(Path(path).read_text())
    schema = d.pop("__schema__", None)
    # sniff on the artifact, not the filename: a name says nothing about the shape
    _sample = " ".join(hint_text(v) for v in list(d.values())[:20])
    is_transcript = len(TRANSCRIPT_MARKERS.findall(_sample)) >= 10
    words = {k: len(hint_text(v).split()) for k, v in d.items()}
    med = statistics.median(words.values()) if words else 0
    cut = max(med * max_ratio, 25)  # floor: a 15-word median must not condemn a 40-word hint
    flags = {}
    for k, v in d.items():
        t = hint_text(v)
        why = []
        if words[k] > cut:
            why.append(f"length {words[k]}w > {cut:.0f} ({max_ratio:g}x median {med:.0f})")
        if not is_transcript and COT_LEAK.search(t):
            why.append("cot-leak")
        if not is_transcript and REPETITION.search(t):
            why.append("repetition")
        if words[k] < min_words:
            why.append(f"too-short {words[k]}w")
        if why:
            flags[k] = {"words": words[k], "why": why, "head": t[:110]}
    return d, schema, med, cut, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts", nargs="+")
    ap.add_argument("--max-ratio", type=float, default=5.0)
    ap.add_argument("--drop", action="store_true", help="write a copy with flagged keys removed")
    ap.add_argument("--out", help="output path for --drop (default: <input>.clean.json)")
    ap.add_argument("--write-todo", help="write the flagged task_ids as a JSON list (for regeneration)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    all_todo, bad_total = [], 0
    for p in a.artifacts:
        d, schema, med, cut, flags = scan(p, a.max_ratio)
        bad_total += len(flags)
        all_todo += list(flags)
        status = "CLEAN" if not flags else f"{len(flags)} FLAGGED"
        print(f"{Path(p).name:44s} n={len(d):5d} median={med:5.0f}w cut={cut:6.0f}w  {status}")
        if not a.quiet:
            for k, f in sorted(flags.items(), key=lambda kv: -kv[1]["words"]):
                print(f"     {k[:34]:36s} {f['words']:6d}w  {', '.join(f['why'])}")
                print(f"       head={f['head']!r}")
        if a.drop:
            out = a.out or str(p) + ".clean.json"
            clean = {k: v for k, v in d.items() if k not in flags}
            if schema:
                clean = {"__schema__": schema, **clean}
            Path(out).write_text(json.dumps(clean))
            print(f"     -> dropped {len(flags)}, wrote {len(clean)} keys to {out}")

    if a.write_todo:
        Path(a.write_todo).write_text(json.dumps(sorted(set(all_todo)), indent=1))
        print(f"\nwrote {len(set(all_todo))} task_ids -> {a.write_todo}")
    print(f"\nTOTAL FLAGGED: {bad_total}")
    return 1 if bad_total else 0


if __name__ == "__main__":
    sys.exit(main())

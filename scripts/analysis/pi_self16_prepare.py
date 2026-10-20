#!/usr/bin/env python
"""Self 8-win + 8-loss PI on ALFWorld train1600: per task, 16 of the untrained student's OWN rollouts as hints.

Student = untrained Qwen3-4B-Instruct-2507; rollouts = the pass@k corpora (original env, plain prompt, T 1.0 /
top-p 1.0 / top-k -1, 30 turns): N=128 on the 200 offline-analysis tasks, N=32 on the other 1,400.

Tasks: every train1600 task with >= 8 self wins (89 offline + 394 other = 483).
Per task, 16 rollouts are sampled ONCE and serve as both the hints and the scored set:
  8 won        uniform among the task's wins (seeded rng stream per task)
  8 lost       losses that already carry a judge label first (they are what Δd⁻_err reads), then the rest
  backups      if a task has < 8 losses, its LEAST EFFICIENT remaining wins (most assistant turns, random
               tie-break) fill the lost slots; they are judge-labelled like losses and flagged backup=True
Each hint is scored on the OTHER 15 (never against its own trajectory): self_pairs.json maps every hint to its
own scored uid so the analysis drops that pair. Labels: existing judge labels translated into this corpus' uid
space; to_label.json lists the lost/backup rollouts still missing one (pi_error_step_label.py --uids-file).
Rendered raw and uncapped exactly like the correct-/wrong-self PIs: render_uncapped(parse_chat(prompt+output)).

Writes <out>/{corpus.jsonl, chunks/, cell/rows.parquet, hints/hints_self_{w,l}{1..8}.json, labels.json,
to_label.json, self_pairs.json, selection.csv, summary.json}.

    python pi_self16_prepare.py --out outputs/pi_self16b    # reads $PI_STUDY_ROOT/pi_mm_rem1400/corpus.jsonl
"""

import argparse
import collections
import glob
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alf_study_common as C  # noqa: E402
from alf_study_common import parse_chat  # noqa: E402
from pi_multimodel_offline_prepare import SCHEMA, TASKS200  # noqa: E402
from pi_self4_prepare import render_uncapped  # noqa: E402

DUMP200 = str(C.DUMP200_4BI)
SHARDS = str(C.REM1400_SHARDS_4BI)
REM_CORPUS = str(C.PI_STUDY_ROOT / "pi_mm_rem1400" / "corpus.jsonl")  # pi_rem1400_prepare.py output
LABELS200 = str(C.LABELS_4BI)
LABELS_REM = str(C.LABELS_REM1400_4BI)
TRAIN1600 = str(C.TRAIN1600)


def tid(d: dict) -> str:
    return str((d["env_extras"].get("extra_info") or {}).get("task_id"))


def won(d: dict) -> bool:
    sc = d["score"]
    return (max(sc) if isinstance(sc, list) else sc) >= 10


def n_turns(line: str) -> int:
    """Assistant turns in a dumped output_response (same split as pi_error_step_label.alf_runs)."""
    resp = json.loads(line).get("output_response") or ""
    return sum(i == 0 or ch.startswith("assistant") for i, ch in enumerate(re.split(r"<\|im_start\|>", resp)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8, help="won and lost hints per task")
    ap.add_argument("--min-wins", type=int, default=8)
    ap.add_argument("--chunks", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    for sub in ("hints", "chunks"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # every rollout as (src id, raw line, labels): offline ids = dump line, rem ids = shard:line
    t1600, t200 = set(json.load(open(TRAIN1600))), set(json.load(open(TASKS200)))
    lab200, lab_rem = json.load(open(LABELS200)), json.load(open(LABELS_REM))
    rem_index = {}  # raw line -> its uid in the rem1400 scoring corpus (where the rem labels live)
    for i, line in enumerate(open(REM_CORPUS)):
        line = line.rstrip("\n")
        rem_index.setdefault(line, f"{tid(json.loads(line))}#{i}")
    by_task = collections.defaultdict(list)
    for i, line in enumerate(open(DUMP200)):
        line = line.rstrip("\n")
        d = json.loads(line)
        t = tid(d)
        by_task[t].append({"src": f"dump200:{i}", "line": line, "won": won(d), "label": lab200.get(f"{t}#{i}")})
    for f in sorted(glob.glob(SHARDS)):
        for i, line in enumerate(open(f)):
            line = line.rstrip("\n")
            d = json.loads(line)
            u = rem_index.get(line)
            by_task[tid(d)].append(
                {
                    "src": f"{Path(f).stem}:{i}",
                    "line": line,
                    "won": won(d),
                    "label": lab_rem.get(u) if u else None,
                }
            )
    assert set(by_task) == t1600, len(set(by_task) ^ t1600)

    tasks = sorted(t for t in t1600 if sum(r["won"] for r in by_task[t]) >= args.min_wins)
    corpus, labels, to_label, rows, cell = [], {}, [], [], []
    hints = {f"self_{s}{k}": {"__schema__": SCHEMA} for s in "wl" for k in range(1, args.k + 1)}
    self_pairs = {name: {} for name in hints}
    for t in tasks:
        rng = random.Random(f"self16:{args.seed}:{t}")
        wins_ = [r for r in by_task[t] if r["won"]]
        rng.shuffle(wins_)
        hw = wins_[: args.k]
        lab = [r for r in by_task[t] if not r["won"] and r["label"] is not None]
        unl = [r for r in by_task[t] if not r["won"] and r["label"] is None]
        rng.shuffle(lab)
        rng.shuffle(unl)
        hl = (lab + unl)[: args.k]
        rest = wins_[args.k :]
        for r in rest:
            r["tie"], r["turns"] = rng.random(), n_turns(r["line"])
        backup = sorted(rest, key=lambda r: (-r["turns"], r["tie"]))[: args.k - len(hl)]
        slots = [(f"self_w{k}", r, False) for k, r in enumerate(hw, 1)]
        slots += [(f"self_l{k}", r, k > len(hl)) for k, r in enumerate(hl + backup, 1)]
        for name, r, is_backup in slots:
            d = json.loads(r["line"])
            u = f"{t}#{len(corpus)}"
            corpus.append(r["line"])
            cell += [(u, t, int(r["won"]), k) for k in range(1, n_turns(r["line"]) + 1)]
            if r["label"] is not None:
                labels[u] = r["label"]
            elif name.startswith("self_l"):
                to_label.append(u)  # a lost (or backup) slot must carry a judge label
            hints[name][t] = {
                "task": render_uncapped(parse_chat(d["input_prompt"] + d["output_response"])),
                "meta": {
                    "pi_format": "self_raw",
                    "slot": name[5:],
                    "src": r["src"],
                    "backup": is_backup,
                },
            }
            self_pairs[name][t] = u
        rows.append(
            {
                "task_id": t,
                "pool": "offline200" if t in t200 else "rem1400",
                "n_rollouts": len(by_task[t]),
                "wins": len(wins_),
                "losses": len(lab) + len(unl),
                "win_src": " ".join(r["src"] for r in hw),
                "loss_src": " ".join(r["src"] for r in hl),
                "backup_src": " ".join(f"{r['src']}({r['turns']}t)" for r in backup),
                "loss_slots_already_labelled": sum(r["label"] is not None for r in hl + backup),
                "loss_slots_to_label": sum(r["label"] is None for r in hl + backup),
            }
        )
    (out / "corpus.jsonl").write_text("\n".join(corpus) + "\n")
    (out / "labels.json").write_text(json.dumps(labels))
    (out / "to_label.json").write_text(json.dumps(to_label))
    (out / "self_pairs.json").write_text(json.dumps(self_pairs))
    (out / "cell").mkdir(exist_ok=True)
    pd.DataFrame(cell, columns=["uid", "task_id", "success", "step"]).assign(pi="corpus").to_parquet(
        out / "cell" / "rows.parquet"
    )
    for name, h in hints.items():
        (out / "hints" / f"hints_{name}.json").write_text(json.dumps(h))
    task_of = [json.loads(line)["env_extras"]["extra_info"]["task_id"] for line in corpus]
    for c in range(args.chunks):
        ts = set(tasks[c :: args.chunks])
        idx = [i for i, t in enumerate(task_of) if t in ts]
        (out / "chunks" / f"chunk_{c}.jsonl").write_text("\n".join(corpus[i] for i in idx) + "\n")
        (out / "chunks" / f"chunk_{c}_index.json").write_text(json.dumps(idx))
    sel = pd.DataFrame(rows)
    sel.to_csv(out / "selection.csv", index=False)
    summary = {
        "tasks": len(tasks),
        "by_pool": sel.pool.value_counts().to_dict(),
        "rollouts": len(corpus),
        "hint_files": {k: len(v) - 1 for k, v in hints.items()},
        "tasks_with_backups": int((sel.backup_src.str.len() > 0).sum()),
        "backup_slots": int(sel.backup_src.str.split().str.len().fillna(0).sum()),
        "loss_slots_already_labelled": int(sel.loss_slots_already_labelled.sum()),
        "loss_slots_to_label": len(to_label),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

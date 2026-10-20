"""Build the 3 PI-format hint artifacts from GLM won reference trajectories (refs.jsonl).

Formats (each a `{task_id: hint}` artifact, drop-in for the scorer's --hints):
  1. raw           — full transcript (thinking + actions + observations), role-tagged
  2. raw_no_think  — transcript with <think>/<Think> blocks stripped (case-insensitive)
  3. actions       — the ordered list of the agent's actions only
  4. actions_think — reasoning + actions, OBSERVATIONS EXCLUDED   <- completes the 2x2

THE 2x2. Formats 1-3 leave one cell of {think on/off} x {obs on/off} unbuilt (the same gap existed on
AppWorld until its self-hint builder grew `actions_think`):

              obs OFF            obs ON
  think OFF   actions            raw_no_think
  think ON    actions_think*     raw                (* new)

It is the only cell that separates reasoning from length. On the AppWorld 4bt density grid
raw_no_think (action+obs) BEAT raw (think+action+obs), 0.1324 vs 0.1186 -- reasoning appeared harmful,
but raw is also ~50% longer, so "reasoning hurts" and "the hint got too long" were confounded.
actions_think holds observations out and varies only reasoning against `actions`.

Note it drops the OBSERVATION (user) turns, which on ALFWorld carry the goal statement -- exactly as
`actions` already does. That is fine for OPSD, where the hint is injected alongside the real task, so
the goal reaches the model through the prompt rather than through the hint.

ONLY MEANINGFUL ON REFS THAT ACTUALLY RETAIN REASONING, AND ON ALFWORLD THAT IS glm-5.2 ONLY.
Measured over frontier_unseen (134 tasks), median bytes of reasoning per hint via the clean estimator
`raw - raw_no_think` (same renderer, only the reasoning removed):
    glm-5.2  322   (108/134 tasks carry >200 ch)
    qwen3.7    0   <- raw and raw_no_think are byte-identical on most tasks
So the qwen3.7 ALFWorld refs carry essentially NO reasoning and their raw-vs-raw_no_think contrast is
already degenerate -- the legacy-ladder collapse this repo documents for Instruct-lineage policies.
Building actions_think from them compares FORMATTING (role tags, <action> wrappers vs a numbered list),
not reasoning. Do not read a qwen3.7 actions_think-vs-actions delta as a reasoning effect.
Do NOT estimate reasoning with `actions_think - actions`: `actions` is a numbered list while
actions_think is role-tagged, so that difference is inflated by markup (glm-5.2 557 vs the true 322).

Reuses `pi.base.TrajView.render` (raw/raw_no_think) and `pi.base.extract_actions` (actions); the
think-strip is done here (case-insensitive) to avoid editing shared `base.py` (whose render() strip is
lowercase-only — a recommended 1-line fix is `re.IGNORECASE` at base.py:84).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from opsd.pi.artifact import save_hint_artifact
from opsd.pi.base import HintRecord, TrajView, extract_actions
from opsd.pi.prompts import scrub_hint

# Case-insensitive so it strips both <think> (real data) and <Think> (template variant).
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_ACTION_SPAN_RE = re.compile(r"<action>.*?</action>", re.DOTALL | re.IGNORECASE)


def strip_think_ci(text: str) -> str:
    return _THINK_RE.sub("", text or "")


def no_reasoning_content(role: str, content: str) -> str:
    """For the 'no_think' format: drop the assistant's reasoning, keep only its <action> spans;
    keep observation (user) turns verbatim. Robust to both <think>-wrapped (Qwen) and prose (GLM)
    reasoning — GLM has no <think> tags, so we reduce assistant turns to their <action> blocks."""
    content = strip_think_ci(str(content or ""))
    if role == "assistant":
        spans = _ACTION_SPAN_RE.findall(content)
        return "\n".join(spans) if spans else content.strip()
    return content


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with Path(path).expanduser().open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def to_trajview(rec: Dict[str, Any]) -> TrajView:
    return TrajView(
        task_id=rec["task_id"],
        messages=rec.get("messages") or [],
        reward=float(rec.get("reward", 0.0) or 0.0),
        success=bool(rec.get("won")),
        num_turns=int(rec.get("num_turns", 0) or 0),
    )


def build(refs_path: str, out_dir: Path, max_chars_raw: int) -> Dict[str, Any]:
    wins: Dict[str, TrajView] = {}
    for rec in read_jsonl(refs_path):
        if rec.get("won") and rec.get("task_id") not in wins:
            wins[rec["task_id"]] = to_trajview(rec)

    raw: Dict[str, HintRecord] = {}
    raw_nt: Dict[str, HintRecord] = {}
    acts: Dict[str, HintRecord] = {}
    acts_th: Dict[str, HintRecord] = {}
    trunc = 0
    for tid, tv in wins.items():
        # 1. raw — full transcript (keep thinking)
        raw_txt = tv.render(max_chars=max_chars_raw, strip_think=False)
        raw[tid] = HintRecord(task=scrub_hint(raw_txt, max_chars=max_chars_raw), meta={"pi_format": "raw"})
        # 2. raw_no_think — trajectory with the reasoning removed (keep observations + <action> only)
        nt_msgs = [{**m, "content": no_reasoning_content(m.get("role", ""), m.get("content", ""))} for m in tv.messages]
        tv_nt = TrajView(tid, nt_msgs, tv.reward, tv.success, tv.num_turns)
        raw_nt[tid] = HintRecord(
            task=scrub_hint(tv_nt.render(max_chars=max_chars_raw, strip_think=False), max_chars=max_chars_raw),
            meta={"pi_format": "raw_no_think"},
        )
        # 3. actions — ordered action list
        a = extract_actions(tv.messages)
        acts[tid] = HintRecord(
            task="Reference actions (in order):\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(a)),
            meta={"pi_format": "actions", "n_actions": len(a)},
        )
        # 4. actions_think — reasoning + actions, observations dropped. Keeping <think> makes this
        # `raw` MINUS the observation turns, so actions_think vs raw isolates obs and actions_think
        # vs actions isolates reasoning.
        keep = [m for m in tv.messages if m.get("role") != "user"]
        tv_at = TrajView(tid, keep, tv.reward, tv.success, tv.num_turns)
        acts_th[tid] = HintRecord(
            task=scrub_hint(tv_at.render(max_chars=max_chars_raw, strip_think=False), max_chars=max_chars_raw),
            meta={"pi_format": "actions_think"},
        )
        if len(raw_txt) > max_chars_raw:
            trunc += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    save_hint_artifact(str(out_dir / "hints_raw.json"), raw)
    save_hint_artifact(str(out_dir / "hints_raw_no_think.json"), raw_nt)
    save_hint_artifact(str(out_dir / "hints_actions.json"), acts)
    save_hint_artifact(str(out_dir / "hints_actions_think.json"), acts_th)
    summary = {
        "n_won_tasks": len(wins),
        "raw_truncated": trunc,
        "max_chars_raw": max_chars_raw,
        "out_dir": str(out_dir),
    }
    (out_dir / "formats_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", required=True, help="GLM refs.jsonl (won trajectories with `messages`).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-chars-raw", type=int, default=6000)
    args = ap.parse_args()
    s = build(args.refs, Path(args.out_dir).expanduser(), args.max_chars_raw)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()

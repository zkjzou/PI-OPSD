"""Build the 3 PI-format hint artifacts (raw / raw_no_think / actions) from τ² trajectories.

Thin τ² front-end for the shared, schema-agnostic builder in
``opsd.alfworld.build_pi_hint_formats.build``, which needs refs shaped as
``{task_id, messages, won, reward, num_turns}`` and keeps the **first won** trajectory per task.

τ² is NOT a drop-in for the τ1 front-end.  Two differences would each silently produce a useless
artifact, so both are handled here rather than in the shared builder:

1. **τ² uses native function-calling, so ``actions`` would come out EMPTY.**
   ``pi.base.extract_actions`` scans assistant **content** for ``<action>…</action>`` or ``Action: …``.
   τ² actions live in the structured ``tool_calls`` field and never appear in ``content``.  We render
   each call as a ``Action: {"name": …, "arguments": {…}}`` line, which is exactly the shape τ1's
   ``hints_actions.json`` ended up with, so the shared extractor works unmodified.  The lines also
   survive into ``raw_no_think``, because ``no_reasoning_content()`` falls back to the stripped content
   for turns with no ``<action>`` spans.

2. **Reasoning must be inlined as ``<think>``, or ``raw`` == ``raw_no_think``.**
   ``build()`` separates those two formats by stripping ``<think>…</think>`` from content, but τ² keeps
   reasoning out of band as ``reasoning_content`` (``episode.openai_messages`` recovers it from
   ``raw_data``).  Passing messages through unchanged makes both artifacts byte-identical — the whole
   point of having three variants lost.  Confirmed live: a hosted GLM-5p2 endpoint returns
   ``reasoning_content`` as a separate field with no inline ``<think>`` at all.  So we fold it back in.

Trajectory source is ``response_messages`` (OpenAI-shaped, reasoning already reattached) — **not**
``messages``, which also contains the *user simulator's* own tool traffic that the agent never saw, and
not ``agent_view`` (raw τ² dumps).  ``prompt_messages`` is deliberately **excluded**: it is the domain
policy + 16 tool schemas (~4.7k tokens) that the student already has in its own system prompt, so it is
not privileged information and would crowd the real trajectory out of any length budget.

Only WON trajectories become hints — a PI hint is a *successful* reference demonstration.

    python -m opsd.tau2_bench.build_tau2_pi_hints \
        --traj <baseline.py trajectories.jsonl> --out-dir <dir> \
        [--only-tasks data/tau2_train500.jsonl] [--max-chars-raw 250000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from opsd.alfworld.build_pi_hint_formats import build

# Rendered per tool call so the shared ``extract_actions`` (which only reads ``content``) can see it.
ACTION_PREFIX = "Action: "


def _render_tool_calls(message: Dict[str, Any]) -> List[str]:
    """``tool_calls`` -> ``Action: {"name": …, "arguments": {…}}`` lines (gap 1).

    ``arguments`` arrives as a JSON *string* from the API; we parse it so the rendered line is a single
    well-formed object rather than a string containing escaped JSON.  Unparseable arguments are kept
    verbatim — a malformed call is still information about what the teacher attempted.
    """

    lines: List[str] = []
    for call in message.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            try:
                args: Any = json.loads(raw_args)
            except (ValueError, TypeError):
                args = raw_args
        else:
            args = raw_args if raw_args is not None else {}
        lines.append(ACTION_PREFIX + json.dumps({"name": name, "arguments": args}, ensure_ascii=False, sort_keys=True))
    return lines


def _refs_message(message: Dict[str, Any]) -> Dict[str, str]:
    """One τ² OpenAI-shaped message -> the flat ``{role, content}`` the shared builder expects.

    Folds ``reasoning_content`` back in as ``<think>`` (gap 2) and appends rendered tool calls (gap 1).
    ``tool`` is remapped to ``user`` so ``TrajView.render`` tags it ``[OBSERVATION]`` — it has no case
    for ``tool`` and the turn would otherwise render untagged.
    """

    role = message.get("role") or "assistant"
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = str(content)

    parts: List[str] = []
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip() and "<think>" not in content:
        parts.append(f"<think>\n{reasoning.strip()}\n</think>")
    if content.strip():
        parts.append(content.strip())
    parts.extend(_render_tool_calls(message))

    return {"role": "user" if role == "tool" else role, "content": "\n\n".join(parts)}


def _load_allowed_task_ids(manifest_paths: Iterable[str]) -> set[str]:
    """Task ids from τ² manifest JSONL(s) — used to keep train PI and dev PI in separate artifacts."""

    allowed: set[str] = set()
    for path in manifest_paths:
        with Path(path).expanduser().open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    tid = json.loads(line).get("task_id")
                    if tid is not None:
                        allowed.add(str(tid))
    return allowed


def tau2_traj_to_refs(traj_path: str, refs_path: str, allowed: set[str] | None = None) -> Dict[str, int]:
    """τ² trajectories.jsonl -> refs.jsonl, won-first so ``build()`` keeps a success per task."""

    recs: List[Dict[str, Any]] = []
    skipped_no_transcript = 0
    skipped_not_allowed = 0
    with Path(traj_path).expanduser().open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task_id = str(rec.get("task_id"))
            if allowed is not None and task_id not in allowed:
                skipped_not_allowed += 1
                continue
            # An infra-failed episode is zeroed with messages=[] by episode._failed(), so there is
            # nothing to distil; counting these separately keeps coverage honest.
            turns = rec.get("response_messages") or []
            if not turns:
                skipped_no_transcript += 1
                continue
            reward = float(rec.get("reward", 0.0) or 0.0)
            recs.append(
                {
                    # Plain task_id, NOT τ1's "{domain}:{task_id}": a τ² consumer looks the hint up by
                    # env_extras["task_id"], and telecom ids contain ':' (the reason task_digest exists).
                    "task_id": task_id,
                    "domain": rec.get("domain"),
                    "messages": [_refs_message(m) for m in turns],
                    "won": reward >= 1.0,
                    "reward": reward,
                    "num_turns": int(rec.get("num_agent_turns", 0) or 0),
                }
            )
    recs.sort(key=lambda r: (not r["won"]))
    with Path(refs_path).expanduser().open("w") as handle:
        for rec in recs:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {
        "n_records": len(recs),
        "n_tasks": len({r["task_id"] for r in recs}),
        "n_won_tasks": len({r["task_id"] for r in recs if r["won"]}),
        "skipped_no_transcript": skipped_no_transcript,
        "skipped_not_in_manifest": skipped_not_allowed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", required=True, help="τ² baseline.py trajectories.jsonl (the PI source model)")
    ap.add_argument("--out-dir", required=True, help="writes hints_{raw,raw_no_think,actions}.json here")
    ap.add_argument(
        "--only-tasks",
        action="append",
        default=None,
        metavar="MANIFEST.jsonl",
        help="restrict to task_ids from a τ² manifest (repeatable); keeps train/dev PI apart",
    )
    # 250k = effectively no truncation, per the PI-attribute study's skip-don't-truncate rule. τ1 used
    # 8000 and EVERY raw hint hit the cap; τ² episodes are ~12k tokens, so 8000 would truncate them all.
    ap.add_argument("--max-chars-raw", type=int, default=250000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    allowed = _load_allowed_task_ids(args.only_tasks) if args.only_tasks else None

    refs_path = out_dir / "_tau2_refs.jsonl"
    conv = tau2_traj_to_refs(args.traj, str(refs_path), allowed)
    summary = build(str(refs_path), out_dir, args.max_chars_raw)
    summary.update(conv)
    summary["only_tasks"] = args.only_tasks or []
    summary["n_allowed_tasks"] = len(allowed) if allowed is not None else None
    (out_dir / "formats_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

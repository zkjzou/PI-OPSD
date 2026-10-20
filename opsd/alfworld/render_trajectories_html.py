"""Render a SAMPLED SUBSET of ALFWorld train/eval trajectories to a single self-contained HTML file.

The trajectories are already dumped during training:
  * EVAL  (per eval step, per dataset): <export_path>/dumped_evals/global_step_<N>_evals/<data_source>.jsonl
           fields: input_prompt, output_response, score, stop_reason, env_class, env_extras, data_source
  * TRAIN (per gradient update):         <OPSD_ROLLOUT_DUMP_DIR>/rollouts_<NNNNN>.jsonl
           fields: task_id, reward_sum, stop_reason, env_metrics, hint, prompt_text, response_text|response_tail

This reads one or more such jsonl files (or a directory / glob), SAMPLES N per file, and renders each
trajectory as readable HTML: per-turn system/user(observation)/assistant bubbles, with <think> and
<action> highlighted, plus a header (task_id, reward/success, stop_reason, tokens, hint).

Usage:
  # sample 5 eval trajectories per dataset at one eval step
  python render_trajectories_html.py --inputs <run>/dumped_evals/global_step_150_evals/*.jsonl \
      --sample 5 --out /tmp/eval_s150.html
  # sample 3 train rollouts from a batch dump
  python render_trajectories_html.py --inputs <OPSD_ROLLOUT_DUMP_DIR>/rollouts_00042.jsonl \
      --sample 3 --out /tmp/train_b42.html
  # a whole directory or glob of dumps (quote globs so the script expands them):
  python render_trajectories_html.py --inputs './dump/*.jsonl' --sample 4 --out traj.html
"""

import argparse
import glob
import html
import json
import os
import random
import re

_TURN_RE = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)(?:<\|im_end\|>|$)", re.DOTALL)


def _load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _traj_text(rec):
    """Return (prompt_text, response_text) handling both eval-dump and OPSD-train-dump schemas."""
    prompt = rec.get("input_prompt") or rec.get("prompt_text") or ""
    resp = rec.get("output_response") or rec.get("response_text") or rec.get("response_tail") or ""
    return prompt, resp


def _success(rec):
    em = rec.get("env_metrics") or {}
    if isinstance(em, dict) and em.get("success") is not None:
        return bool(em.get("success"))
    score = rec.get("score")
    rew = rec.get("reward_sum")
    if isinstance(score, list):
        return any(float(x) >= 10.0 for x in score)
    for v in (score, rew):
        if isinstance(v, (int, float)):
            return float(v) >= 10.0
    return None


def _fmt_turn(role, content):
    content = html.escape(content.strip())
    # highlight <think> and <action> spans (already escaped, so match on escaped tags)
    content = re.sub(r"&lt;think&gt;(.*?)&lt;/think&gt;", r'<span class="think">🧠\1</span>', content, flags=re.DOTALL)
    content = re.sub(
        r"&lt;action&gt;(.*?)&lt;/action&gt;", r'<span class="action">▶ \1</span>', content, flags=re.DOTALL
    )
    content = content.replace("Observation:", "<b>Observation:</b>")
    return f'<div class="turn {role}"><span class="role">{role}</span><pre>{content}</pre></div>'


def _render_traj(rec, idx):
    prompt, resp = _traj_text(rec)
    full = prompt + resp
    turns = _TURN_RE.findall(full)
    body = "".join(_fmt_turn(r, c) for r, c in turns) if turns else f"<pre>{html.escape(full)}</pre>"
    succ = _success(rec)
    badge = {True: '<span class="ok">[PASS]</span>', False: '<span class="fail">[FAIL]</span>'}.get(
        succ, '<span class="unk">[?]</span>'
    )
    em = rec.get("env_metrics") or {}
    meta = " · ".join(
        s
        for s in [
            f"task_id={rec.get('task_id') or (rec.get('env_extras') or {}).get('extra_info',{}).get('task_id','?')}",
            f"data_source={rec.get('data_source','?')}",
            (
                f"reward={rec.get('reward_sum', rec.get('score'))}"
                if not isinstance(rec.get("score"), list)
                else "reward=(per-token)"
            ),
            f"stop={rec.get('stop_reason')}",
            f"turns={em.get('num_turns')}" if em.get("num_turns") is not None else "",
            f"invalid={em.get('num_invalid_actions')}" if em.get("num_invalid_actions") is not None else "",
            f"resp_tok={rec.get('n_response_tokens')}" if rec.get("n_response_tokens") else "",
        ]
        if s
    )
    hint = rec.get("hint")
    # OPSD_DUMP_HTML_HINT_CHARS=0 renders the whole hint. The 4000 default is kept so no existing run
    # changes, but it hides ~95% of a trajectory-format PI (those run to 48-72k chars) -- and the hint is
    # the thing this panel exists to show next to what the student produced.
    _hcap = int(os.environ.get("OPSD_DUMP_HTML_HINT_CHARS", "4000") or 0)
    _h = str(hint)[:_hcap] if _hcap > 0 else str(hint)
    hint_html = f"<details><summary>hint (PI)</summary><pre>{html.escape(_h)}</pre></details>" if hint else ""
    return (
        f'<div class="traj"><h3>#{idx} {badge}</h3>'
        f'<div class="meta">{html.escape(meta)}</div>{hint_html}{body}</div>'
    )


_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f5f7;color:#1a1a1a}
header{position:sticky;top:0;background:#222;color:#fff;padding:10px 16px;font-size:14px}
.traj{background:#fff;margin:14px;padding:12px 16px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.12)}
h3{margin:.2em 0}.meta{color:#666;font-size:12px;margin-bottom:8px;font-family:monospace}
.turn{margin:6px 0;padding:6px 10px;border-radius:6px}
.turn .role{font-size:10px;text-transform:uppercase;color:#888;letter-spacing:.5px}
.turn.system{background:#fff8e1}.turn.user{background:#e8f0fe}.turn.assistant{background:#f1f8e9}
pre{white-space:pre-wrap;word-break:break-word;margin:2px 0;font-size:13px;line-height:1.4}
.think{color:#8a6d3b}.action{color:#0b6b2e;font-weight:600}
.ok{color:#0b6b2e}.fail{color:#b00020}.unk{color:#999}
details{margin:4px 0;font-size:12px}summary{cursor:pointer;color:#555}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="jsonl file(s), dir(s), or glob(s).")
    ap.add_argument("--sample", type=int, default=5, help="trajectories to sample PER input file.")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (fixed => reproducible).")
    ap.add_argument("--only", choices=("all", "success", "fail"), default="all", help="filter before sampling.")
    ap.add_argument("--out", required=True, help="output .html path.")
    args = ap.parse_args()

    files = []
    for pat in args.inputs:
        if os.path.isdir(pat):
            files += sorted(glob.glob(os.path.join(pat, "*.jsonl")))
        else:
            files += sorted(glob.glob(pat)) or ([pat] if os.path.isfile(pat) else [])
    rng = random.Random(args.seed)

    sections, total = [], 0
    for fp in files:
        rows = _load(fp)
        if args.only != "all":
            want = args.only == "success"
            rows = [r for r in rows if _success(r) is want]
        pick = rows if len(rows) <= args.sample else rng.sample(rows, args.sample)
        total += len(pick)
        cards = "".join(_render_traj(r, i) for i, r in enumerate(pick))
        sections.append(f"<header>{html.escape(os.path.basename(fp))} — {len(pick)}/{len(rows)} shown</header>{cards}")

    doc = f"<!doctype html><html><head><meta charset=utf-8><style>{_CSS}</style></head><body>{''.join(sections)}</body></html>"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(doc)
    print(f"rendered {total} trajectories from {len(files)} file(s) -> {args.out}")


if __name__ == "__main__":
    main()

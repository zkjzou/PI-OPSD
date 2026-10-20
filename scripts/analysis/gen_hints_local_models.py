#!/usr/bin/env python
"""Generate per-trajectory hints from RAW trajectories using a locally-served policy model.

Differs from build_hints.py in two ways that matter, both requested explicitly:
  1. PLAIN TEXT output, not JSON. The stock analyzer prompt demands a JSON object; this one forbids
     any structure. So there is no parse step and no "JSON got truncated" failure -- but also no
     schema to validate against, which is why the empty/short-output counters below are load-bearing.
  2. RAW rendering (strip_think=False): the analyzer sees the agent's thinking, not just its actions.

Key format is ``{task_id}#rep{sample}``, byte-identical to online.py:359 and to
``build_hints --per-traj``, so the output drops into the existing scorer unchanged.

TWO SIZING TRAPS, both previously observed in this pipeline:
  * CONTEXT. 4B-Thinking's test_normal trajectories average ~78k tokens -- they do not fit the 49k
    context the 4B-Instruct/30B servers run with. --max-chars must be set per model against that
    server's max_model_len, and the renderer middle-elides rather than truncating the tail (the end of
    a trajectory is where the outcome is).
  * OUTPUT BUDGET. A thinking model spends its completion budget on reasoning before emitting the
    hint; at max_tokens=1024 that yielded ZERO usable hints on long trajectories once already. Hence
    --max-tokens defaults high and <think> blocks are stripped from the RESPONSE.

HINDSIGHT CAVEAT: each hint is distilled from the very rollout it will be scored against, so `d`
computed on these measures fit, not transfer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def _repo_root() -> Path:
    """The repository root: the nearest ancestor of this file holding pyproject.toml, opsd/__init__.py and
    opsd/skyrl_ext/__init__.py (the predicate of opsd._repo)."""
    for d in Path(__file__).resolve().parents:
        if all((d / m).is_file() for m in ("pyproject.toml", "opsd/__init__.py", "opsd/skyrl_ext/__init__.py")):
            return d
    raise RuntimeError(f"cannot locate the repository root above {__file__}")


sys.path.insert(0, str(_repo_root()))
from opsd.pi.artifact import load_hint_artifact, save_hint_artifact  # noqa: E402,I001
from opsd.pi.base import HintRecord  # noqa: E402
from opsd.pi.build_hints import messages_from_dump, trajview_from_dump  # noqa: E402
from opsd.pi.prompts import (  # noqa: E402
    parse_analyzer_json,
    scrub_hint,
    skill_prompts,
    skill_to_hint,
)

# ALFWorld states its goal INSIDE the trajectory: the env's own "Your task is to: <goal>" line in the
# first observation, plus a "Task: <goal>" line in the scaffold preamble. Measured on the 2356-task
# glm reference artifact: 2301 carry the former, all 2356 the latter, and the modal goal covers 0.9%
# of tasks. So ALFWorld needs NO --task-map-corpus -- the whole AppWorld scaffold-demo problem simply
# does not arise. Prefer the env line; the preamble line is the fallback.
_ALF_GOAL = re.compile(r"Your task is to:\s*(.+)")
_ALF_GOAL_FALLBACK = re.compile(r"^\s*Task:\s*(.+)", re.MULTILINE)


def alfworld_task_text(body: str) -> str:
    m = _ALF_GOAL.search(body) or _ALF_GOAL_FALLBACK.search(body)
    return m.group(1).strip()[:400] if m else ""


# tau2 renders a CONVERSATION as alternating "[AGENT] ...\n[OBSERVATION] ..." blocks. The agent opens
# with a greeting, so the task -- the customer's actual request -- is the FIRST [OBSERVATION]. There is
# no "Task:" line to grep, and conversation[0] is the greeting, not the request.
_TAU2_FIRST_OBS = re.compile(r"\[OBSERVATION\]\s*(.+?)(?=\n\[AGENT\]|\Z)", re.S)


def tau2_task_text(body: str) -> str:
    """The customer's opening request. Measured 424/424 and 415/415 extraction on the retail pools,
    mode share 0.9-1.0% -- i.e. genuinely per-task, which is what the mode-share guard checks."""
    m = _TAU2_FIRST_OBS.search(body)
    return " ".join(m.group(1).split())[:400] if m else ""


def messages_true_raw(rec: dict) -> tuple[list[dict], str]:
    """Messages that actually include the agent's REASONING, plus which source was used.

    `messages_from_dump` reads `conversation`, whose assistant entries hold ONLY the action. For a
    thinking policy the reasoning lives in `turns[i]["think"]` and is dropped entirely -- measured on
    AppWorld test_normal 4B-Thinking: turn 1 carries 5,630 chars of `think` against 74 chars of
    `action`, and 0/200 records have a <think> tag anywhere in `conversation`. So rendering the
    `conversation` path with strip_think=False is NOT the raw trajectory; it is silently
    raw_no_think, and `raw` vs `raw_no_think` become the same input.

    When `turns` carries non-empty `think`, rebuild from it (think + action per assistant turn, obs as
    the observation) and keep conversation[0] as the task statement. Otherwise fall back -- 4B-Instruct
    and 30B-A3B are non-thinking here (preserve_thinking=False, no `think` text), so for them the two
    paths agree by construction.
    """
    turns = rec.get("turns") or []
    has_think = any(str((t or {}).get("think", "")).strip() for t in turns if isinstance(t, dict))
    if not has_think:
        return messages_from_dump(rec), "conversation"
    conv = rec.get("conversation") or []
    msgs = [{"role": "user", "content": str(conv[0].get("content", ""))}] if conv else []
    for t in turns:
        if not isinstance(t, dict):
            continue
        think, action, obs = (str(t.get(k, "") or "") for k in ("think", "action", "obs"))
        content = f"<think>{think}</think>\n{action}" if think.strip() else action
        if content.strip():
            msgs.append({"role": "assistant", "content": content})
        if obs.strip():
            msgs.append({"role": "user", "content": obs})
    return msgs, "turns(+think)"


PROMPT = (
    "You are creating a concise natural language hint that will help an agent solve the task. "
    "Write a helpful hint describing the key actions and strategy the agent used. "
    "If the agent did not fully complete the task, focus on what they did achieve. "
    "IMPORTANT: Output ONLY plain text - no JSON, no code blocks, no markdown formatting."
)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK = re.compile(r"^.*?</think>", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove the model's own reasoning from the RESPONSE.

    Handles the unclosed-open case too: some servers emit the trailing </think> without the opener,
    so a naive paired-tag regex leaves the entire reasoning block in the hint.
    """
    t = _THINK.sub("", text)
    if "</think>" in t:
        t = _OPEN_THINK.sub("", t)
    return t.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None, help="rollout dump (JSONL). Mutually exclusive with --input-hints.")
    ap.add_argument(
        "--input-hints",
        default=None,
        help="an EXISTING PI artifact ({key: trajectory_text}) to summarise, e.g. "
        "hints225/*/hints_raw.json. Those values are full rendered reference "
        "trajectories (~13.7k chars, '# turn 1\\n<think>...'), one per TASK -- not "
        "analyzer output -- so this mode compresses PI rather than distilling a "
        "rollout. No Outcome line is emitted: the artifacts carry no success label "
        "and asserting one would be fabrication.",
    )
    ap.add_argument(
        "--url",
        default=None,
        help="server ROOT; '/v1/chat/completions' is appended. Local vLLM: http://127.0.0.1:PORT. "
        "Mutually exclusive with --base-url.",
    )
    ap.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL INCLUDING /v1; '/chat/completions' is appended. Default when "
        "--url is not given: $OPENAI_BASE_URL, else https://api.openai.com/v1 (key from $OPENAI_API_KEY "
        "unless --api-key-env names another variable).",
    )
    ap.add_argument(
        "--api-key",
        default=None,
        help="bearer token for a remote endpoint. PREFER --api-key-env: a value passed HERE lands in "
        "the process table and is readable by other users of the machine via `ps aux`. "
        "Omit both for local vLLM, no auth needed.",
    )
    ap.add_argument(
        "--api-key-env",
        default=None,
        help="NAME of an env var holding the bearer token (e.g. OPENAI_API_KEY). Preferred over "
        "--api-key so the secret never reaches `ps` output. Takes precedence if both are supplied.",
    )
    ap.add_argument("--model", required=True, help="served-model-name")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--prompt-style",
        choices=("plain", "onpolicy_skill"),
        default="plain",
        help="plain = the free-text hint prompt. onpolicy_skill = the ACTUAL OPSD "
        "analyzer: AppWorld system briefing + STRICT JSON "
        "{success_analysis, mistake_analysis, golden_workflow}, post-processed by "
        "skill_to_hint into 'Recommended approach / Key skill / Avoid'. Its system "
        "prompt forbids restating literal ids, emails and exact answers, which the "
        "plain prompt does not -- measured leakage on the plain run: 52/160 hints "
        "carry 3+ digit ids, 35/160 literal dollar amounts.",
    )
    ap.add_argument(
        "--task-map-corpus",
        default=None,
        help="rollout JSONL used to recover task_id -> task text. Required for "
        "onpolicy_skill in --input-hints mode: the PI artifacts are bare "
        "trajectories and carry neither the task statement nor an outcome, both of "
        "which the analyzer template needs.",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        required=True,
        help="cap on the RENDERED trajectory; set against this server's max_model_len",
    )
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--success-threshold",
        type=float,
        default=1.0,
        help="AppWorld binary reward pays 1.0; the ALFWorld 10.0 default would label " "every win a FAILURE",
    )
    ap.add_argument(
        "--env",
        choices=("appworld", "alfworld", "tau2"),
        default="appworld",
        help="selects the onpolicy_skill prompt pair AND how the task statement is "
        "recovered. appworld: needs --task-map-corpus (the artifacts carry no task). "
        "alfworld: read out of the trajectory itself, no corpus needed. "
        "tau2: the customer's opening request, the first [OBSERVATION] block of the "
        "rendered conversation -- also self-contained, no corpus needed.",
    )
    ap.add_argument(
        "--outcome",
        choices=("auto", "success", "failure"),
        default="auto",
        help="the Outcome the analyzer is told. auto = the AppWorld heuristic "
        "(complete_task present -> SUCCESS), which is MEANINGLESS off AppWorld: "
        "ALFWorld reference trajectories carry no terminal marker at all (measured: "
        "0/2356 match any of 7 candidates), so auto would label every one of a "
        "corpus of WINS a FAILURE and send the analyzer hunting for mistakes that "
        "are not there. Pass `success` for a verified reference set. Always recorded "
        "in the stats, so the assertion is visible instead of assumed.",
    )
    ap.add_argument(
        "--out-schema",
        choices=("flat", "opsd_v2"),
        default="flat",
        help="flat = {key: hint} (the AppWorld files). opsd_v2 = the "
        "{__schema__, task_id: {task, meta}} envelope every ALFWorld consumer reads, "
        "which makes the output drop-in for OPSD_HINTS_PATH.",
    )
    ap.add_argument(
        "--task-allowlist",
        default=None,
        help="JSON list of task ids (or a dict whose keys are task ids) restricting the "
        "input. Applied after loading, so coverage describes the restricted set.",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    # Resolve --api-key-env ONCE, here, so the request path stays unchanged and the secret never
    # appears in argv. A value passed via --api-key is visible in `ps aux` to every user of the
    # machine; --api-key-env carries only the NAME. Env wins when both are given.
    if args.api_key_env:
        _tok = os.environ.get(args.api_key_env, "").strip()
        if not _tok:
            raise SystemExit(f"--api-key-env {args.api_key_env} is set but that env var is empty/unset")
        args.api_key = _tok
    if args.url and args.base_url:
        raise SystemExit("pass at most one of --url (server root) / --base-url (.../v1)")
    if args.url:
        endpoint = f"{args.url.rstrip('/')}/v1/chat/completions"
    else:
        _base = args.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        endpoint = f"{_base.rstrip('/')}/chat/completions"
        if not args.api_key:
            args.api_key = os.environ.get("OPENAI_API_KEY", "").strip() or None

    if bool(args.corpus) == bool(args.input_hints):
        raise SystemExit("pass exactly one of --corpus / --input-hints")
    artifact_mode = bool(args.input_hints)
    if artifact_mode:
        # Read through load_hint_artifact, NOT json.load. AppWorld artifacts are v1 ({key: str}) but
        # ALFWorld's are v2 ({key: {"task": ..., "meta": ...}}) with a "__schema__" sentinel key --
        # and a naive json.load hands that sentinel to the summariser as if the string
        # "opsd_hints/v2" were a trajectory. The loader handles both and drops the sentinel.
        src = {k: r.task for k, r in load_hint_artifact(args.input_hints).items() if r.task}
        if args.task_allowlist:
            allow = json.loads(Path(args.task_allowlist).read_text())
            keep = set(allow if isinstance(allow, list) else allow.keys())
            before = len(src)
            src = {k: v for k, v in src.items() if k in keep}
            print(
                f"[gen] task allowlist {args.task_allowlist}: {len(keep)} ids -> "
                f"{len(src)}/{before} trajectories kept",
                flush=True,
            )
            if not src:
                raise SystemExit("FATAL: the allowlist matched no task in the artifact")
        recs = [{"_key": k, "_text": v} for k, v in sorted(src.items())]
        print(f"[gen] {len(recs)} reference trajectories from {args.input_hints}", flush=True)
    else:
        recs = []
        with open(args.corpus) as fh:
            for line in fh:
                if line.strip():
                    recs.append(json.loads(line))
        print(f"[gen] {len(recs)} rollouts from {args.corpus}", flush=True)
    if args.limit:
        recs = recs[: args.limit]

    sysmsg = None
    task_map: dict[str, str] = {}
    if args.prompt_style == "onpolicy_skill":
        # strict=True: an unknown --env must NOT silently fall back to ALFWorld wording for an
        # offline generation run (see skill_prompts docstring).
        sysmsg, user_tmpl = skill_prompts(args.env, strict=True)
        print(f"[gen] onpolicy_skill prompts for env={args.env}", flush=True)
        if artifact_mode and args.env == "alfworld":
            # The goal is in the trajectory; build the map from the artifact itself. Same mode-share
            # guard as AppWorld -- it is cheap and it is the check that would catch a regex that
            # started matching the scaffold's generic instructions instead of the episode's goal.
            task_map = {r["_key"]: alfworld_task_text(str(r["_text"])) for r in recs}
        if artifact_mode and args.env == "tau2":
            # Self-contained like ALFWorld: the request is in the conversation. Same mode-share guard
            # below -- it is what would catch this regex drifting onto the agent's greeting.
            task_map = {r["_key"]: tau2_task_text(str(r["_text"])) for r in recs}
        if artifact_mode and args.env == "appworld":
            if not args.task_map_corpus:
                raise SystemExit(
                    "--prompt-style onpolicy_skill with --input-hints needs "
                    "--task-map-corpus (artifacts carry no task statement)"
                )
            # THE REAL TASK IS NOT IN conv[0]. The AppWorld official scaffold puts a worked EXAMPLE
            # in the preamble, and conv[0]'s only "Task:" is that DEMO -- identical for every episode
            # ("How many playlists do I have in my Spotify playlist library?" on all 90 train90
            # tasks). The episode's own task arrives in the last scaffold message, at
            # generated_span - 1. Taking conv[0] fed one constant demo task to every analyzer call.
            from opsd.pi.adapters.appworld import generated_span

            for line in open(args.task_map_corpus):
                if not line.strip():
                    continue
                rr = json.loads(line)
                tid = rr.get("task_id")
                if tid in task_map:
                    continue
                conv = rr.get("conversation") or []
                if not conv:
                    continue
                span = generated_span(conv, len(rr.get("turns") or []))
                # last "Task:" anywhere in the SCAFFOLD block -- the real one comes after the demo
                found = ""
                for m in conv[: max(span, 1)]:
                    c = str(m.get("content", ""))
                    if "Task:" in c:
                        found = c.split("Task:")[-1].strip().split("\n")[0][:400]
                task_map[tid] = found
            n_distinct = len({v for v in task_map.values() if v})
            print(
                f"[gen] task map: {len(task_map)} tasks, {n_distinct} DISTINCT task texts "
                f"from {args.task_map_corpus}",
                flush=True,
            )
        if artifact_mode and task_map:
            n_distinct = len({v for v in task_map.values() if v})
            if args.env in ("alfworld", "tau2"):
                print(
                    f"[gen] task map: {len(task_map)} tasks, {n_distinct} DISTINCT goals "
                    f"read from the trajectories themselves",
                    flush=True,
                )
            # FAIL LOUD on the failure mode above: a map whose values collapse is worse than no map,
            # because every hint is then written against the wrong task while all counters read 100%.
            # Guard on MODE SHARE, not on the distinct ratio. AppWorld ships task VARIANTS (_1/_2/_3
            # of a scenario) that legitimately share wording with different DB state, so 148 distinct
            # of 168 is normal -- a distinct-ratio test rejects that. The failure being caught is one
            # value swallowing the map (the scaffold demo gave 1 distinct text for all 90 tasks), and
            # mode share detects exactly that without punishing variants.
            from collections import Counter as _C

            _vals = [v for v in task_map.values() if v]
            _mode_n = _C(_vals).most_common(1)[0][1] if _vals else 0
            _share = _mode_n / max(len(task_map), 1)
            print(
                f"[gen]   most common task text covers {_mode_n}/{len(task_map)} " f"({_share:.1%}) of tasks",
                flush=True,
            )
            if len(task_map) > 4 and (_share > 0.25 or not _vals):
                raise SystemExit(
                    f"FATAL: one task text covers {_share:.1%} of the map "
                    f"({n_distinct} distinct for {len(task_map)} tasks) -- extraction is picking a "
                    f"constant (likely the scaffold demo)"
                )
            _missing = sum(1 for v in task_map.values() if not v)
            if _missing:
                print(f"[gen]   WARNING {_missing} tasks have NO task text", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hints: dict[str, str] = {}
    if out_path.exists():  # resume: generation is the expensive half
        # Read via the schema-aware loader: an opsd_v2 output must resume as opsd_v2, and a plain
        # json.load would re-add "__schema__" to `hints` as though it were a task.
        hints = {k: r.task for k, r in load_hint_artifact(str(out_path)).items() if r.task}
        print(f"[gen] resuming, {len(hints)} hints already present", flush=True)

    stats = Counter()
    lock = threading.Lock()
    t0 = time.time()

    def work(rec):
        if artifact_mode:
            key = rec["_key"]
            if key in hints:
                with lock:
                    stats["skipped_existing"] += 1
                return
            body = str(rec["_text"])
            if len(body) > args.max_chars:  # middle-elide: the END carries the outcome
                half = args.max_chars // 2
                body = body[:half] + "\n...[trajectory truncated]...\n" + body[-half:]
                with lock:
                    stats["elided"] += 1
            if args.prompt_style == "onpolicy_skill":
                task = task_map.get(key, "")
                # `auto` is the AppWorld heuristic and is AppWorld-ONLY: reaching complete_task is
                # what ends an AppWorld episode, and nothing analogous exists in an ALFWorld render
                # (0/2356 carry any terminal marker), so auto there silently labels a corpus of wins
                # FAILURE. Recorded in the stats either way so the label is visible, not assumed.
                if args.outcome == "auto":
                    outcome = "SUCCESS" if "complete_task" in body else "FAILURE"
                else:
                    outcome = args.outcome.upper()
                with lock:
                    stats[f"outcome::{outcome}"] += 1
                    if not task:
                        stats["no_task_text"] += 1
                _submit(key, user_tmpl.format(task=task, outcome=outcome, trajectory=body))
            else:
                _submit(key, f"{PROMPT}\n\nTrajectory:\n{body}")
            return
        tv = trajview_from_dump(rec, success_threshold=args.success_threshold)
        key = f"{tv.task_id}#rep{tv.repetition_id}"
        if key in hints:
            with lock:
                stats["skipped_existing"] += 1
            return
        tv.messages, src = messages_true_raw(rec)
        with lock:
            stats[f"src::{src}"] += 1
        body = tv.render(max_chars=args.max_chars, strip_think=False)  # RAW: keep the thinking
        if not body.strip():
            with lock:
                stats["empty_render"] += 1
            return
        outcome = "SUCCESS" if tv.success else "FAILURE"
        _submit(key, f"{PROMPT}\n\nOutcome: {outcome}\n\nTrajectory:\n{body}")

    def _submit(key, user):
        msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + [{"role": "user", "content": user}]
        payload = {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "messages": msgs,
        }
        for attempt in range(args.retries):
            try:
                hdrs = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
                r = requests.post(endpoint, json=payload, headers=hdrs, timeout=args.timeout)
                if r.status_code != 200:
                    if r.status_code == 400:  # context overflow -- do not retry, it will repeat
                        with lock:
                            stats["ctx_overflow"] += 1
                        return
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                d = r.json()
                ch = d["choices"][0]
                text = strip_reasoning(ch["message"]["content"] or "")
                if args.prompt_style == "onpolicy_skill":
                    # The analyzer contract is STRICT JSON; a hint that fails to parse is DROPPED rather
                    # than stored raw, because an unparsed blob would silently become the hint text.
                    parsed = parse_analyzer_json(text)
                    if not parsed:
                        with lock:
                            stats["json_parse_failed"] += 1
                        return
                    text = scrub_hint(skill_to_hint(parsed))
                with lock:
                    if ch.get("finish_reason") == "length":
                        stats["truncated"] += 1
                    if not text:
                        stats["empty_after_strip"] += 1
                        return
                    if len(text) < 40:
                        stats["very_short"] += 1
                    hints[key] = text
                    stats["ok"] += 1
                    u = d.get("usage") or {}
                    stats["prompt_tokens"] += int(u.get("prompt_tokens", 0))
                    stats["completion_tokens"] += int(u.get("completion_tokens", 0))
                    n = stats["ok"]
                    if n % 100 == 0:
                        el = time.time() - t0
                        print(
                            f"[gen] {n} hints | {el/60:.1f} min | {n/max(el,1)*60:.0f}/min "
                            f"| eta {(len(recs)-n)/max(n/max(el,1),1e-9)/60:.0f} min",
                            flush=True,
                        )
                return
            except Exception as e:  # noqa: BLE001
                if attempt == args.retries - 1:
                    with lock:
                        stats["failed"] += 1
                        stats[f"err::{type(e).__name__}"] += 1
                else:
                    time.sleep(2 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(work, recs))

    if args.out_schema == "opsd_v2":
        # Go through save_hint_artifact so the envelope is produced by the same code that reads it.
        meta_base = {
            "pi_type": args.prompt_style,
            "summarizer_model": args.model,
            "source_artifact": args.input_hints or args.corpus,
            "outcome_mode": args.outcome,
            "env": args.env,
        }
        save_hint_artifact(str(out_path), {k: HintRecord(task=v, meta=dict(meta_base)) for k, v in hints.items()})
    else:
        out_path.write_text(json.dumps(hints, indent=0))
    print(f"[gen] WROTE {len(hints)} hints ({args.out_schema}) -> {out_path}")
    print(f"[gen] stats {dict(stats)}")
    # Fail loudly: a hint file that is mostly missing looks identical downstream to a complete one,
    # because the scorer just counts those rollouts as no_hint.
    cover = len(hints) / max(len(recs), 1)
    print(f"[gen] coverage {cover:.1%}")
    if cover < 0.95:
        raise SystemExit(f"FATAL: only {cover:.1%} of trajectories produced a hint")


if __name__ == "__main__":
    main()

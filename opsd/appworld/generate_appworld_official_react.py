#!/usr/bin/env python
r"""Paper-faithful AppWorld ReAct driver — a standalone port of the official
`SimplifiedReActCodeAgent` (stonybrooknlp/appworld experiments) that runs IN-PROCESS against
appworld 0.1.3 and calls the policy over an OpenAI-compatible endpoint (a hosted API or a local vLLM).

Why this exists: the official `appworld run` harness needs appworld 0.2.0 data (task IDs changed;
api_docs/base_dbs/split-lists absent from the offline minimal download). We instead reproduce the
paper's ReAct *scaffold* — verbatim prompt (official instructions.txt), code extraction, observation
formatting, 20k-char observation truncation, and the solve loop — on the proven 0.1.0 data (same
test_normal task set as our other baselines, and closer to the paper's July-2024 0.1.x benchmark).

Endpoint: ``--base-url`` (alias ``--model-url``; default ``$OPENAI_BASE_URL``, else
``https://api.openai.com/v1``) and ``--api-key-env`` (default ``OPENAI_API_KEY``; when set, that key is
sent as a Bearer token to every ``--base-url`` replica, local or not; pass ``--api-key-env ''`` to send none).

Run in the appworld venv (from ``$APPWORLD_ROOT``):
  export APPWORLD_ROOT=~/appworld_root OPENAI_API_KEY=...
  $APPWORLD_VENV/bin/python opsd/appworld/generate_appworld_official_react.py \
    --served-model-name gpt-4o-2024-05-13 \
    --dataset test_normal --n-samples 1 --temperature 0.0 --seed 100 --max-steps 50 \
    --concurrency 12 --output-dir ./outputs/generate_appworld_official_react/gpt4o
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import requests
from appworld import AppWorld, load_task_ids
from jinja2 import Template

# ------------------------- official code-extraction (verbatim logic) -------------------------
PARTIAL_CODE_RE = r".*```python\n(.*)"
FULL_CODE_RE = r"```python\n(.*?)```"


class _CtxOverflow(Exception):
    """Raised on a context-length 400 in raw mode -> end the episode, score current state."""


def render_raw_prompt(messages: list) -> str:
    """Render messages to Qwen3 im-format WITHOUT stripping prior-turn <think>.

    The chat template (/v1/chat/completions) DROPS <think>…</think> from every prior assistant turn,
    so a chat-endpoint agent never actually sees its own accumulated reasoning. To measure REAL
    preserved thinking we render the prompt ourselves — keeping each assistant turn's content verbatim
    (<think> intact) — and hit the raw /v1/completions endpoint (no template applied). Ends with the
    assistant <think> primer so generation starts inside the thinking block, exactly as the template's
    add_generation_prompt does. This mirrors what SkyRLGymGenerator (GRPO training) does token-in-token-out.
    """
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return s + "<|im_start|>assistant\n<think>\n"


def extract_code_and_fix_content(text: str, ignore_multiple_calls: bool = True) -> tuple[str, str]:
    original_text = text
    # THINKING-MODEL FIX: Qwen3-Thinking reasons WITH code — it writes exploratory ```python``` snippets
    # INSIDE its <think> block (observed: 18 of 19 fences in one turn were pre-</think>). The committed
    # action is only what follows </think>. Searching the whole text with ignore_multiple_calls=True
    # would grab the FIRST fence (an in-think snippet) and execute it, and truncate `fixed` before
    # </think> (losing the reasoning + the real action). So restrict the code search to AFTER the last
    # </think>. `fixed` is still measured from the start of the full text, so the <think>…</think>
    # reasoning is PRESERVED in history. When there is no </think> (non-thinking / demo-style output),
    # search_from stays 0 and behaviour is identical to the original.
    search_from = original_text.rindex("</think>") + len("</think>") if "</think>" in original_text else 0
    output_code = ""
    match_end = search_from
    for re_match in re.finditer(FULL_CODE_RE, original_text[search_from:], flags=re.DOTALL):
        code = re_match.group(1).strip()
        if ignore_multiple_calls:
            return code, original_text[: search_from + re_match.end()]
        output_code += code + "\n"
        match_end = search_from + re_match.end()
    partial_match = re.match(PARTIAL_CODE_RE, original_text[match_end:], flags=re.DOTALL)
    if partial_match:
        output_code += partial_match.group(1).strip()
        if not text.endswith("\n"):
            text = text + "\n"
        text = text + "```"
    if len(output_code) == 0:
        return "", text
    return output_code, text


# ------------------------- official message parsing (verbatim logic) -------------------------
def text_to_messages(input_str: str) -> list[dict]:
    messages: list[dict] = []
    last_start = 0
    for m in re.finditer("(USER|ASSISTANT|SYSTEM):\n", input_str, flags=re.IGNORECASE):
        last_end = m.span()[0]
        if len(messages) == 0:
            if last_end != 0:
                raise ValueError(f"Start of the prompt has no assigned role: {input_str[:last_end]}")
        else:
            messages[-1]["content"] = input_str[last_start:last_end]
        messages.append({"role": m.group(1).lower(), "content": None})
        last_start = m.span()[1]
    messages[-1]["content"] = input_str[last_start:]
    return messages


def messages_to_text(messages: list[dict]) -> str:
    out = ""
    for message in messages:
        role = message["role"]
        if role == "system":
            out += "SYSTEM:\n" + message["content"]
        elif role == "assistant":
            out += "ASSISTANT:\n" + message["content"]
        elif role == "user":
            out += "USER:\n" + message["content"]
        else:
            raise ValueError(f"Unknown message role {role}")
    return out


def trimmed_messages(messages: list[dict], num_instruction_messages: int, max_output_length):
    """Official observation-truncation: keep the last 5 blocks, elide older env observations to
    [NOT SHOWN FOR BREVITY], then drop whole history blocks as [TRIMMED HISTORY], until under the
    char budget. No-op when max_output_length is None (the gpt-4o config's setting)."""
    if max_output_length is None:
        return messages
    messages = copy.deepcopy(messages)
    pre_messages = messages[: num_instruction_messages - 1]
    post_messages = messages[num_instruction_messages - 1 :]
    output_str = messages_to_text(post_messages)
    remove_prefix = output_str[: output_str.index("Task: ") + 6]
    output_str = output_str.removeprefix(remove_prefix)
    observation_index = 0
    while len(output_str) > max_output_length:
        found_block = False
        if observation_index < len(post_messages) - 5:
            for message_index, message in enumerate(post_messages[observation_index:]):
                if message["role"] == "user" and message["content"].startswith("Output:"):
                    message["content"] = "Output:\n```\n[NOT SHOWN FOR BREVITY]```\n\n"
                    found_block = True
                    observation_index += message_index + 1
                    break
            if not found_block:
                observation_index = len(post_messages)
        if not found_block and len(post_messages):
            first = copy.deepcopy(post_messages[0])
            if not first["content"].endswith("[TRIMMED HISTORY]\n\n"):
                first["content"] += "[TRIMMED HISTORY]\n\n"
            post_messages = [first] + post_messages[2:]
            found_block = True
        if not found_block:
            raise ValueError("No blocks found to be removed!")
        output_str = messages_to_text(post_messages).removeprefix(remove_prefix)
    return pre_messages + post_messages


# ------------------------- policy client (OpenAI-compatible, retry+backoff) -------------------------
class ChatClient:
    def __init__(
        self,
        base_url,
        model,
        api_key,
        timeout,
        temperature,
        seed,
        max_retries=8,
        max_tokens=None,
        raw_completions=False,
    ):
        # base_url may be a COMMA-LIST of independent single-GPU vLLM replicas. Workers run in a
        # ProcessPool, so seed the round-robin with the pid to spread load instead of every worker
        # starting on replica 0.
        # raw_completions=True -> hit /v1/completions with a self-rendered prompt (prior <think> KEPT),
        # instead of /v1/chat/completions (whose template strips prior <think>).
        self.raw = raw_completions
        tail = "/completions" if raw_completions else "/chat/completions"
        self.urls = []
        for u in str(base_url).split(","):
            u = u.strip().rstrip("/")
            if not u:
                continue
            if not u.endswith("/v1"):
                u += "/v1"
            self.urls.append(u + tail)
        if not self.urls:
            raise SystemExit(f"no usable --model-url in {base_url!r}")
        self._rr = os.getpid() % len(self.urls)
        self.url = self.urls[0]
        self.model, self.timeout, self.temperature, self.seed = model, timeout, temperature, seed
        self.max_retries = max_retries
        self.max_tokens = max_tokens  # per-turn generation cap; None = unbounded (original gpt-4o cfg)
        self._local = threading.local()
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def _session(self):
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            a = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=0)
            s.mount("http://", a)
            s.mount("https://", a)
            self._local.s = s
        return s

    def complete(self, messages):
        if self.raw:
            # RAW mode: self-render (prior <think> kept) and POST to /v1/completions (no template).
            payload = {
                "model": self.model,
                "prompt": render_raw_prompt(messages),
                "temperature": self.temperature,
                "stop": ["<|im_end|>"],
            }
        else:
            payload = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.max_tokens is not None:
            # Per-turn cap. REQUIRED with preserved thinking: without it a single turn can generate up
            # to (max_model_len - prompt) tokens of runaway <think>, so rollouts take 30+ min in eager
            # mode. Harmless for the thinking-discarded gpt-4o config (short generations) -> left None
            # there. Qwen3-4B-Thinking's think is ~3k tok/turn (max ~8k), so 16k rarely truncates.
            payload["max_tokens"] = self.max_tokens
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                self._rr += 1
                url = self.urls[self._rr % len(self.urls)]  # retry lands on a different replica
                r = self._session().post(url, json=payload, headers=self._headers, timeout=self.timeout)
                if r.status_code == 400 and "maximum context length" in r.text:
                    raise _CtxOverflow(r.text[:160])  # preserved history filled the window -> end episode
                if r.status_code in (408, 409, 425, 429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:120]}")
                r.raise_for_status()
                if self.raw:
                    return (r.json()["choices"][0].get("text") or "").strip()
                return (r.json()["choices"][0]["message"].get("content") or "").strip()
            except _CtxOverflow:
                raise  # deterministic, not transient -> do not retry
            except (requests.RequestException, ValueError, KeyError) as e:
                last = str(e)[:200]
                if attempt >= self.max_retries:
                    break
                time.sleep(min(45.0, 2.0**attempt + random.random()))
        raise requests.HTTPError(last or "request failed after retries")


# ------------------------- faithful solve loop (port of solve_task + next_execution) -------------------------
def build_initial_messages(world, prompt_template: str) -> list[dict]:
    t = world.task
    sup = t.supervisor
    main_user = {
        "first_name": sup["first_name"],
        "last_name": sup["last_name"],
        "email": sup["email"],
        "phone_number": sup["phone_number"],
    }
    app_descriptions = json.dumps([{"name": k, "description": v} for (k, v) in t.app_descriptions.items()], indent=1)
    # LOOP's variant of this prompt (not redistributed; bring your own copy) names the task `input_str`
    # and also injects `date`; the official one uses `instruction` and no date. Pass BOTH so either --prompt-file
    # renders unchanged (Jinja ignores variables a template doesn't reference).
    task_date = (
        getattr(t, "datetime", None) or getattr(t, "date", None) or getattr(getattr(t, "metadata", None), "date", None)
    )
    if task_date is None:
        task_date = datetime.datetime.now().strftime("%Y-%m-%d")
    rendered = (
        Template(prompt_template)
        .render(
            instruction=t.instruction,
            input_str=t.instruction,
            date=str(task_date),
            main_user=main_user,
            app_descriptions=app_descriptions,
        )
        .lstrip()
        + "\n\n"
    )
    return text_to_messages(rendered)


def solve_task(
    task_id, sample, client, prompt_template, max_steps, max_output_length, save_conv, discard_thinking=False
):
    turns = []
    reward = 0.0
    success = 0.0
    num_turns = 0
    stop = "done"
    conv = []
    try:
        with AppWorld(task_id=task_id, experiment_name=f"official_react_{task_id}_{sample}") as world:
            messages = build_initial_messages(world, prompt_template)
            num_instruction_messages = len(messages)
            conv = list(messages)
            last_output = None
            for step in range(max_steps):
                num_turns = step + 1
                if last_output is not None:
                    obs_msg = (
                        "Output:\n```\n" + last_output + ("\n" if not last_output.endswith("\n") else "") + "```\n\n"
                    )
                    messages.append({"role": "user", "content": obs_msg})
                    conv.append({"role": "user", "content": obs_msg})
                send = trimmed_messages(messages, num_instruction_messages, max_output_length)
                try:
                    raw = client.complete(send)
                except _CtxOverflow:
                    # RAW preserved-thinking mode: accumulated <think> filled the window. End the
                    # episode gracefully and score the world's current state (do NOT count as an error).
                    stop = "context_overflow"
                    break
                code, fixed = extract_code_and_fix_content(raw)
                # --discard-thinking: keep only the post-</think> action in HISTORY (drop the reasoning),
                # while `code` is still extracted from the same post-</think> region. This isolates the
                # preserve-vs-discard question on ONE parser-off serving with IDENTICAL generation each
                # turn — the only difference is whether the <think> block persists in context. (The
                # parser-on route also changes what the model generates, so it is not a clean control.)
                hist = fixed
                if discard_thinking and "</think>" in fixed:
                    hist = fixed.split("</think>", 1)[-1].lstrip()
                elif getattr(client, "raw", False):
                    # RAW mode: the /completions response has no leading <think> (that was the prompt
                    # primer). Re-add it so the stored turn re-renders as a well-formed <think>…</think>
                    # block — this is what makes the reasoning PERSIST into the next turn's prompt.
                    hist = "<think>\n" + fixed
                messages.append({"role": "assistant", "content": hist + "\n\n"})
                conv.append({"role": "assistant", "content": hist + "\n\n"})
                last_output = world.execute(code)
                turns.append({"turn": step + 1, "code": code, "output": last_output[:4000]})
                if world.task_completed():
                    break
            tracker = world.evaluate(suppress_errors=True)
            n = int(getattr(tracker, "num_tests", 0) or 0)
            p = int(getattr(tracker, "pass_count", 0) or 0)
            reward = (p / n) if n else 0.0
            success = 1.0 if (n and p == n) else 0.0
    except Exception as e:  # keep the run robust; record the failure
        stop = f"error:{type(e).__name__}"
    rec = {
        "task_id": task_id,
        "sample": sample,
        "reward": reward,
        "success": success,
        "num_turns": num_turns,
        "num_tests": n if "n" in dir() else None,
        "stop_reason": stop,
        "turns": turns,
    }
    if save_conv:
        rec["conversation"] = conv
    return rec


def scenario_of(task_id: str) -> str:
    return re.sub(r"_[0-9]+$", "", task_id)


# ------------------------- process-pool worker (in-process AppWorld is NOT thread-safe) -------------------------
_G: dict = {}


def _init_worker(client_kwargs, prompt_template, max_steps, max_output_length, save_conv, discard_thinking):
    global _G
    _G = {
        "client": ChatClient(**client_kwargs),
        "tpl": prompt_template,
        "max_steps": max_steps,
        "mol": max_output_length,
        "save_conv": save_conv,
        "discard": discard_thinking,
    }


def _run_one(job):
    tid, s = job
    return solve_task(
        tid, s, _G["client"], _G["tpl"], _G["max_steps"], _G["mol"], _G["save_conv"], discard_thinking=_G["discard"]
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--base-url",
        "--model-url",
        dest="model_url",
        default=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        help="OpenAI-compatible base URL (comma-list = round-robin replicas). "
        "Default: $OPENAI_BASE_URL, else https://api.openai.com/v1.",
    )
    ap.add_argument("--served-model-name", required=True)
    ap.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Name of the env var holding the API key (default OPENAI_API_KEY). Unset/empty -> "
        "no Authorization header (local vLLM). When set, the key is sent to every --base-url "
        "replica, whatever its host; pass --api-key-env '' to send none.",
    )
    ap.add_argument("--dataset", default="test_normal")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument(
        "--max-output-length",
        type=int,
        default=None,
        help="observation-truncation char budget (paper legacy=20000; simplified gpt-4o config=None).",
    )
    ap.add_argument("--request-timeout", type=int, default=900)
    ap.add_argument(
        "--max-completion-tokens",
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=None,
        help="per-turn generation cap. None (default) = unbounded, matching the original "
        "gpt-4o config. REQUIRED with preserved thinking or a turn can emit up to the "
        "whole context of <think> and rollouts take 30+ min each.",
    )
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--prompt-file", default=str(Path(__file__).with_name("official_react_instructions.txt")))
    ap.add_argument("--save-conversations", action="store_true")
    ap.add_argument(
        "--discard-thinking",
        action="store_true",
        help="drop the <think> block from HISTORY (keep only the post-</think> action); "
        "action extraction is unchanged. Isolates preserve-vs-discard on one serving.",
    )
    ap.add_argument(
        "--raw-completions",
        action="store_true",
        help="hit /v1/completions with a self-rendered prompt that KEEPS prior-turn <think> "
        "(the chat template strips it). REAL preserved thinking; mirrors GRPO's "
        "token-in-token-out generator. Needs a large window (thinking accumulates).",
    )
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    prompt_template = Path(args.prompt_file).read_text()
    api_key = (os.environ.get(args.api_key_env) or None) if args.api_key_env else None
    if api_key is None and not any(h in args.model_url for h in ("127.0.0.1", "localhost", "0.0.0.0")):
        print(
            f"WARNING: no API key in ${args.api_key_env or '<unset>'}; sending unauthenticated requests to "
            f"{args.model_url}",
            file=sys.stderr,
        )
    client_kwargs = dict(
        base_url=args.model_url,
        model=args.served_model_name,
        api_key=api_key,
        timeout=args.request_timeout,
        temperature=args.temperature,
        seed=args.seed,
        max_tokens=args.max_tokens,
        raw_completions=args.raw_completions,
    )
    task_ids = load_task_ids(args.dataset)
    if args.limit:
        task_ids = task_ids[: args.limit]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [(tid, s) for tid in task_ids for s in range(args.n_samples)]
    print(
        f"tasks={len(task_ids)} n_samples={args.n_samples} rollouts={len(jobs)} conc={args.concurrency} "
        f"dataset={args.dataset} model={args.served_model_name} temp={args.temperature} seed={args.seed} max_steps={args.max_steps}"
    )
    rows = []
    done = 0
    # PROCESSES (not threads): in-process AppWorld uses a per-instance in-memory DB that is not
    # thread-safe (concurrent threads -> sqlite ProgrammingError); the official harness also uses processes.
    with ProcessPoolExecutor(
        max_workers=args.concurrency,
        initializer=_init_worker,
        initargs=(
            client_kwargs,
            prompt_template,
            args.max_steps,
            args.max_output_length,
            args.save_conversations,
            args.discard_thinking,
        ),
    ) as ex:
        futs = {ex.submit(_run_one, (tid, s)): (tid, s) for tid, s in jobs}
        for fut in as_completed(futs):
            rec = fut.result()
            rows.append(rec)
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(
                    f"[{done}/{len(jobs)}] last task={rec['task_id']} reward={rec['reward']:.2f} "
                    f"success={rec['success']} turns={rec['num_turns']} stop={rec['stop_reason']}",
                    flush=True,
                )

    with (out / "trajectories.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # metrics
    by = defaultdict(list)
    for r in rows:
        by[r["task_id"]].append(r)
    scen = defaultdict(set)
    for tid in by:
        scen[scenario_of(tid)].add(tid)
    N = max(len(v) for v in by.values())

    def solved(r):
        return r["success"] >= 1.0

    def at(tid, i):
        v = sorted(by[tid], key=lambda r: r["sample"])
        return solved(v[i]) if i < len(v) else False

    tgc = sum(1 for r in rows if solved(r)) / len(rows) * 100
    comp = sum(r["reward"] for r in rows) / len(rows) * 100
    sgc_best = sum(1 for s, ts in scen.items() if all(any(solved(r) for r in by[t]) for t in ts)) / len(scen) * 100
    tgc_i = [sum(1 for tid in by if at(tid, i)) / len(by) * 100 for i in range(N)]
    sgc_i = [sum(1 for s, ts in scen.items() if all(at(t, i) for t in ts)) / len(scen) * 100 for i in range(N)]
    errors = sum(1 for r in rows if "error" in str(r["stop_reason"]))
    summary = {
        "model": args.served_model_name,
        "dataset": args.dataset,
        "scaffold": "official_simplified_react (ported)",
        "sampling": {"temperature": args.temperature, "seed": args.seed},
        "max_steps": args.max_steps,
        "max_output_length": args.max_output_length,
        "n_samples": args.n_samples,
        "max_tokens": args.max_tokens,
        "discard_thinking": args.discard_thinking,
        "raw_completions": args.raw_completions,
        "rollouts": len(rows),
        "tasks": len(by),
        "scenarios": len(scen),
        "TGC": round(sum(tgc_i) / N, 2),
        "SGC": round(sum(sgc_i) / N, 2),
        "TGC_pass1_pooled": round(tgc, 2),
        "completion_rate": round(comp, 2),
        "SGC_best_of_N": round(sgc_best, 2),
        "errors": errors,
        "mean_turns": round(sum(r["num_turns"] for r in rows) / len(rows), 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

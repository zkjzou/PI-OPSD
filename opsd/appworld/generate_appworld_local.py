#!/usr/bin/env python
r"""Standalone AppWorld endpoint-driver (analogue of generate_alfworld_local.py).

Runs the AppWorld CodeAct rollout loop against an ALREADY-RUNNING OpenAI-compatible endpoint --
a local `vllm serve Qwen/Qwen3-4B-Instruct-2507` on :8100, or a hosted API for reference
trajectories -- this process starts NO model of its own. It reuses the opsd.envs ``AppWorldEnv`` for
the env HTTP client, code execution, task completion, and reward (binary success or partial TGC), and
a plain chat-completions client for the policy. N samples/task -> pass@1, pass@k, mean reward.

Why /v1/chat/completions: with the server's ``--reasoning-parser qwen3`` the ``<think>`` block
is stripped from ``message.content``, so only the executable Python reaches AppWorld (whose
contract is that the whole response is code). Thinking is still spent (in reasoning_content),
we just don't feed it back into the env or the next turn (Qwen3 multi-turn convention).

Endpoint: ``--base-url`` (alias ``--model-url``; default ``$OPENAI_BASE_URL``, else
``https://api.openai.com/v1``; a comma-list round-robins over replicas) and ``--api-key-env``
(default ``OPENAI_API_KEY``; no key is sent when that variable is unset, which is fine for a local vLLM).
When set, that key is sent as a Bearer token to EVERY ``--base-url`` replica, local or not; pass
``--api-key-env ''`` to send none. ``--served-model-name`` is required (no default model id).
A loopback URL (127.0.0.1 / localhost / 0.0.0.0) is treated as vLLM and also gets ``top_k`` / ``min_p`` /
``repetition_penalty`` / ``chat_template_kwargs``; any other URL gets OpenAI-standard fields only.

Example (local vLLM policy; AppWorld servers launched from the AppWorld venv):
  export APPWORLD_ROOT=~/appworld_root APPWORLD_VENV=~/appworld-venv
  .venv/bin/python opsd/appworld/generate_appworld_local.py \
    --base-url http://127.0.0.1:8100 --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
    --data-file ~/data/appworld/eval_normal.parquet \
    --n-samples 8 --reward-mode binary --strict-parse --temperature 0.7 --top-p 0.8 --top-k 20 \
    --launch-servers --num-appworld-servers 32 \
    --output-dir ./outputs/generate_appworld_local/q4bi_test_normal
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# opsd.envs (and the skyrl_gym it builds on) is installed in the repository's .venv (run this with .venv/bin/python).
from opsd.envs.appworld.env import AppWorldEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--data-file", required=True, help="AppWorld SkyRL parquet (e.g. test_normal.parquet).")
    p.add_argument("--limit", type=int, default=None, help="Cap parquet rows (smoke).")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=1, help="Rollouts per task.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-name", default="appworld_generate")
    # model endpoint
    p.add_argument(
        "--base-url",
        "--model-url",
        dest="model_url",
        default=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        help="OpenAI-compatible base URL (comma-list = round-robin replicas). "
        "Default: $OPENAI_BASE_URL, else https://api.openai.com/v1.",
    )
    p.add_argument(
        "--served-model-name",
        "--model-name",
        dest="served_model_name",
        required=True,
        help="Model id the endpoint serves (vLLM --served-model-name, or the hosted model id).",
    )
    p.add_argument("--enable-thinking", action="store_true", help="Qwen3 chat_template_kwargs.enable_thinking=true.")
    p.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Name of the env var holding the API key (default OPENAI_API_KEY). Unset/empty -> "
        "no Authorization header (local vLLM). When set, the key is sent to every --base-url "
        "replica, whatever its host; pass --api-key-env '' to send none.",
    )
    p.add_argument("--request-timeout", type=int, default=900)
    # sampling
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument(
        "--max-generate-length", "--max-completion-tokens", dest="max_generate_length", type=int, default=8000
    )
    # env
    p.add_argument("--reward-mode", choices=("binary", "partial"), default="partial")
    p.add_argument(
        "--strict-parse",
        action="store_true",
        help="Only execute a ```python``` fenced block; a fence-less action is rejected "
        "(wasted turn + nudge) and counted in the parse_errors metric.",
    )
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--max-interactions", type=int, default=50)
    p.add_argument("--execute-timeout", type=int, default=120)
    p.add_argument("--repl-output-max-chars", type=int, default=8000)
    p.add_argument("--concurrency", type=int, default=None, help="Thread pool size; default = #servers.")
    # appworld server pool
    p.add_argument(
        "--server-urls",
        default=os.environ.get("APPWORLD_SERVER_URLS"),
        help="JSON list or comma-separated `appworld serve` URLs. Else use --launch-servers.",
    )
    p.add_argument("--launch-servers", action="store_true")
    p.add_argument("--num-appworld-servers", "--env-num", dest="num_appworld_servers", type=int, default=32)
    p.add_argument("--appworld-base-port", type=int, default=int(os.environ.get("APPWORLD_BASE_PORT") or 9000))
    p.add_argument("--appworld-venv", default=os.environ.get("APPWORLD_VENV") or os.path.expanduser("~/appworld-venv"))
    p.add_argument("--appworld-root", default=os.environ.get("APPWORLD_ROOT", os.path.expanduser("~/appworld_root")))
    p.add_argument("--server-warmup-sec", type=int, default=25)
    p.add_argument("--disable-tqdm", action="store_true")
    p.add_argument(
        "--hint-key-field",
        default=os.environ.get("OPSD_HINT_KEY_FIELD") or None,
        help="Look the hint up under extra_info[FIELD] instead of task_id. Needed for "
        "PER-ROLLOUT hint artifacts, which are keyed `{task_id}#rep{n}` (16 per task) "
        "and therefore match ZERO rows under the default task_id lookup. Pair it with a "
        "dataset expanded to one row per (task, rep) that carries the key in "
        "extra_info[FIELD].",
    )
    p.add_argument(
        "--hints-path",
        default=os.environ.get("OPSD_HINTS_PATH"),
        help="PI hint artifact (task_id -> hint); see --hint-inject-style for how it enters "
        "the prompt. Default: $OPSD_HINTS_PATH. Errors out if 0 tasks match, rather than "
        "silently running a hint-free 'with-hint' condition.",
    )
    p.add_argument(
        "--hint-inject-style",
        choices=("splice", "system-prepend"),
        default="system-prepend",
        help="how --hints-path enters the prompt. system-prepend = a <Secret information> system "
        "turn, which is what AppWorld OPSD does at rollout time and what the ranking / "
        "attribute passes score -- keep it matched or a utility delta cannot be correlated "
        "against those AUCs. splice appends to the first user turn (the Harbor/ALFWorld-env "
        "style).",
    )
    p.add_argument(
        "--save-transcripts",
        action="store_true",
        help="Write per-rollout conversation + per-turn {think,action,obs} to transcripts.jsonl "
        "(needed for PI offline / thinking-token analysis).",
    )
    p.add_argument(
        "--preserve-thinking",
        action="store_true",
        help="Feed each turn's <think> reasoning back into the conversation history "
        "(reconstructed as <think>..</think>content). Default off = history carries only the "
        "executed code. Per-turn thinking tokens are recorded in transcripts either way.",
    )
    p.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="First sample id. Lets a corpus be built in passes: n_samples=4 at offsets "
        "0/4/8/12 gives the same 16 distinct sample ids as one n_samples=16 run. Sample "
        "ids must not overlap across passes -- uid is f'{task_id}#{sample}' downstream.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip (task_id, sample) pairs already in the output trajectories.jsonl and APPEND "
        "rather than truncate. Without this a crash at hour N loses hour N.",
    )
    return p.parse_args()


# ---------------- appworld server pool ----------------
def _wait_tcp(port: int, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1.0)
    return False


def launch_appworld_servers(args: argparse.Namespace, log_dir: Path) -> List[str]:
    appworld_bin = os.path.join(args.appworld_venv, "bin", "appworld")
    if not os.path.exists(appworld_bin):
        raise FileNotFoundError(f"appworld bin not found: {appworld_bin} (set up the appworld venv first)")
    urls, pids, procs = [], [], []
    pid_file = log_dir / "appworld_serve.pids"
    for i in range(args.num_appworld_servers):
        port = args.appworld_base_port + i
        logf = open(log_dir / f"appworld_serve_{port}.log", "w")
        proc = subprocess.Popen(
            [appworld_bin, "serve", "environment", "--port", str(port), "--no-show-usage"],
            cwd=args.appworld_root,
            stdout=logf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        pids.append(proc.pid)
        procs.append(proc)
        urls.append(f"http://127.0.0.1:{port}")
    pid_file.write_text(" ".join(str(x) for x in pids))

    def _cleanup() -> None:
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass

    import atexit

    atexit.register(_cleanup)
    print(
        f"launched {len(urls)} appworld servers (ports {args.appworld_base_port}..{args.appworld_base_port + len(urls) - 1})"
    )
    # wait for sockets to open, then a short warmup so /initialize is ready
    for i in range(args.num_appworld_servers):
        _wait_tcp(args.appworld_base_port + i, timeout=120)
    time.sleep(args.server_warmup_sec)
    return urls


def parse_server_urls(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("["):
        return list(json.loads(raw))
    return [u.strip() for u in raw.split(",") if u.strip()]


# ---------------- model client ----------------
class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str],
        timeout: int,
        sampling: Dict[str, Any],
        enable_thinking: bool,
        max_retries: int = 6,
        reasoning_history: Optional[str] = None,
    ):
        # reasoning_history is the PROVIDER-NATIVE way to carry reasoning across turns on an
        # OpenAI-compatible endpoint (Fireworks): set it to "preserved" AND attach `reasoning_content`
        # to each assistant message. Fireworks is explicit that without that field "the model will not
        # have access to its previous reasoning process" -- inlining <think> into `content` does NOT
        # achieve preserved thinking, it just replays prose. Only meaningful on the openai_compat path;
        # a local vLLM has no such field, so that path keeps inlining.
        self.reasoning_history = reasoning_history
        # base_url may be a COMMA-LIST of independent single-GPU replicas; requests are round-robined
        # across them (one vLLM per GPU is much faster for eval than one TP=N engine).
        self.urls: List[str] = []
        for u in str(base_url).split(","):
            u = u.strip().rstrip("/")
            if not u:
                continue
            if not u.endswith("/v1"):
                u += "/v1"
            self.urls.append(u + "/chat/completions")
        if not self.urls:
            raise SystemExit(f"no usable --model-url in {base_url!r}")
        self.url = self.urls[0]  # back-compat / logging
        self._rr = itertools.count()
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.sampling = sampling
        self.enable_thinking = enable_thinking
        # Hosted OpenAI-compatible APIs (OpenAI, Fireworks, a LiteLLM gateway, ...) reject non-OpenAI
        # fields (chat_template_kwargs, top_k, min_p, repetition_penalty). For a non-local base_url,
        # send OpenAI-standard params only.
        self.openai_compat = (
            ("127.0.0.1" not in base_url) and ("localhost" not in base_url) and ("0.0.0.0" not in base_url)
        )
        self._local = threading.local()
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def _session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            # retry/backoff is handled in complete(); mount on BOTH schemes (hosted APIs are https) with
            # a pool sized for reuse. (Previously only http:// was mounted, so an https endpoint silently
            # used the default adapter — irrelevant now that complete() owns the retry loop.)
            a = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=0)
            s.mount("http://", a)
            s.mount("https://", a)
            self._local.s = s
        return s

    def complete(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        """Return ``(content, reasoning)`` for one assistant turn.

        Kept SEPARATE rather than folded into one string because the caller has to put `reasoning` back
        on the next request as a `reasoning_content` FIELD (see ``reasoning_history``), which is
        impossible once it has been concatenated into `content`. Servers that inline <think> into
        content (a local vLLM without --reasoning-parser) simply return reasoning="" and the caller
        splits the tags itself, so both server configs stay supported."""
        if self.openai_compat:  # hosted endpoint: OpenAI-standard fields only
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.sampling.get("temperature"),
                "top_p": self.sampling.get("top_p"),
                "max_tokens": self.sampling.get("max_tokens"),
            }
            if self.reasoning_history:
                payload["reasoning_history"] = self.reasoning_history
        else:
            payload = {
                "model": self.model,
                "messages": messages,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
                **self.sampling,
            }
        # Retry transient failures (rate limits, 5xx, network/timeouts) with exponential backoff + jitter.
        # Hosted endpoints throttle under concurrency; without this a single 429/timeout
        # would kill the whole rollout (this caused 65% HTTPError trajectories on the gpt-4o test_normal run).
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                # round-robin across replicas; a retry therefore lands on a DIFFERENT replica
                url = self.urls[next(self._rr) % len(self.urls)]
                r = self._session().post(url, json=payload, headers=self._headers, timeout=self.timeout)
                if r.status_code == 400:
                    # most commonly context-length overflow on long trajectories -> signal episode end
                    raise _ContextOverflow(r.text[:200])
                if r.status_code in (408, 409, 425, 429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:120]}")
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                # Providers disagree on the field NAME, both observed live on 2026-08-02:
                # Fireworks -> "reasoning_content"; vLLM 0.20.2 --reasoning-parser -> "reasoning".
                # Reading only "reasoning_content" made a parser-on vLLM server look like a
                # no-thinking server: <think> stripped from content, nothing returned as reasoning,
                # and the turn recorded with think_chars=0 -- silent loss, no error.
                reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
                return content, reasoning
            except _ContextOverflow:
                raise  # deterministic (not transient) — do not retry
            except (requests.RequestException, ValueError, KeyError) as e:
                last_err = str(e)[:200]
                if attempt >= self.max_retries:
                    break
                time.sleep(min(45.0, 2.0**attempt + random.random()))
        raise requests.HTTPError(last_err or "request failed after retries")


def _split_think(raw: str) -> Tuple[str, str]:
    """Split a raw assistant turn into (think, code). Qwen3-4B-Thinking-2507 pre-opens <think> in the
    template so the output is '{think}</think>{code}' (no opening tag); hybrids may emit '<think>..</think>{code}'
    or bare code with no think. The ENV must execute only the post-</think> code (the think would be
    invalid Python), so we always return the code tail; `think` is recorded for analysis."""
    if "</think>" in raw:
        before, _, after = raw.rpartition("</think>")
        think = before.split("<think>", 1)[-1].strip()
        return think, after.strip()
    return "", raw.strip()


class _ContextOverflow(Exception):
    pass


# ---------------- rollout ----------------
def _to_messages(prompt: Any) -> List[Dict[str, str]]:
    out = []
    for m in list(prompt):
        out.append({"role": str(m["role"]), "content": str(m["content"])})
    return out


def _row_task_id(row: Dict[str, Any]) -> str:
    """task_id of a parquet row. It is NESTED (reward_spec.ground_truth / extra_info.task_id), never a
    top-level key -- the single source of truth for the three places that need it."""
    reward_spec = row.get("reward_spec") or {}
    extra_info = row.get("extra_info") or {}
    return str(reward_spec.get("ground_truth") or extra_info.get("task_id"))


def run_rollout(
    row: Dict[str, Any],
    sample_idx: int,
    env_cfg: Dict[str, Any],
    client: ChatClient,
    max_turns: int,
    save_conv: bool = False,
    preserve_thinking: bool = False,
) -> Dict[str, Any]:
    reward_spec = row.get("reward_spec") or {}
    extra_info = row.get("extra_info") or {}
    task_id = reward_spec.get("ground_truth") or extra_info.get("task_id")
    extras = {"reward_spec": reward_spec, "extra_info": extra_info, "max_turns": max_turns}
    env = AppWorldEnv(env_cfg, extras)
    stop_reason = "done"
    conv = _to_messages(row["prompt"])  # defined up-front so transcripts survive an early error
    turns_log: List[Dict[str, Any]] = []  # per-turn {turn, think, action, obs} — ALWAYS recorded
    try:
        prompt, _ = env.init(_to_messages(row["prompt"]))
        conv = list(prompt)
        done = False
        for k in range(max_turns):
            try:
                raw = client.complete(conv)
            except _ContextOverflow:
                stop_reason = "context_overflow"
                break
            except Exception as e:  # transient model error -> end episode, score current state
                stop_reason = f"model_error:{type(e).__name__}"
                break
            content, reasoning = raw
            # A server that inlines <think> into content returns reasoning=""; split it out ourselves so
            # both server configs produce the same (think, code) pair.
            think, code = (reasoning, content) if reasoning else _split_think(content)
            # History: with --preserve-thinking, feed the reasoning back. On an OpenAI-compatible endpoint
            # that means a `reasoning_content` FIELD alongside reasoning_history="preserved" -- Fireworks
            # states the model cannot see prior reasoning without it, so inlining <think> in `content`
            # (what this script used to do) was never actually preserved thinking. A local vLLM has no
            # such field, so there we still inline. Either way the ENV executes `code` only — think is not
            # valid Python and must never reach _extract_code.
            if preserve_thinking and think:
                if client.reasoning_history:
                    conv.append({"role": "assistant", "content": code, "reasoning_content": think})
                else:
                    conv.append({"role": "assistant", "content": f"<think>\n{think}\n</think>\n\n{code}"})
            else:
                conv.append({"role": "assistant", "content": code})
            out = env.step(code)  # BaseTextEnvStepOutput is a TypedDict -> use item access
            obs_text = str(out["observations"][0].get("content", "")) if out["observations"] else ""
            # record the per-turn thinking tokens for later analysis (kept even when NOT preserved)
            turns_log.append(
                {"turn": k + 1, "think": think, "action": code, "obs": obs_text, "think_chars": len(think)}
            )
            if out["done"]:
                done = True
                break
            for obs in out["observations"]:
                conv.append({"role": str(obs["role"]), "content": str(obs["content"])})
        if not done:
            # early break (overflow / error / natural max_turns without completion): finalize score
            # of the world's current state (posts /evaluate) so the trajectory still gets a reward.
            env._finalize_reward()  # noqa: SLF001 (our driver; mirrors what step() does on done)
        metrics = env.get_metrics()
        reward = float(env._reward)  # noqa: SLF001
    finally:
        env.close()
    rec = {
        "task_id": task_id,
        "sample": sample_idx,
        "reward": reward,
        # PER-ROLLOUT HINT PROVENANCE. With --hint-key-field, task_id repeats N times (one row per
        # hint), so task_id+sample no longer identifies a condition and a reward cannot be attributed
        # to the hint that produced it -- which is the whole measurement. Carry the key through.
        # None when the field is absent, so task-keyed runs are byte-unchanged apart from a null.
        "hint_key": extra_info.get("hint_key"),
        "success": metrics["success"],
        "partial": metrics["partial_completion"],
        "num_turns": metrics["num_turns"],
        "completed": metrics["completed_task_called"],
        "eval_unavailable": metrics["eval_unavailable"],
        "stop_reason": stop_reason,
        "preserve_thinking": preserve_thinking,
        "turns": turns_log,  # per-turn {think, action, obs} — the per-turn thinking tokens for PI/analysis
    }
    if save_conv:
        rec["conversation"] = conv
    return rec


# ---------------- aggregate ----------------
def _pass_hat(succ: int, n: int, k: int) -> float:
    if succ < k:
        return 0.0
    return math.comb(succ, k) / math.comb(n, k)


def summarize(records: List[Dict[str, Any]], n_samples: int, wall: float, model: str) -> Dict[str, Any]:
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(r)
    n = len(records)
    tasks = len(by_task)
    pass_hat = {}
    trials = min((len(v) for v in by_task.values()), default=0)
    for k in range(1, trials + 1):
        vals = [_pass_hat(sum(1 for r in v if r["success"] >= 1.0), len(v), k) for v in by_task.values()]
        pass_hat[str(k)] = sum(vals) / len(vals) if vals else 0.0
    return {
        "model_name": model,
        "num_trajectories": n,
        "tasks": tasks,
        "n_samples": n_samples,
        "wall_time_sec": wall,
        "pass_at_1": sum(r["success"] for r in records) / n if n else 0.0,
        "pass_hat_k": pass_hat,
        "avg_reward": sum(r["reward"] for r in records) / n if n else 0.0,
        "avg_partial_completion": sum(r["partial"] for r in records) / n if n else 0.0,
        "avg_turns": sum(r["num_turns"] for r in records) / n if n else 0.0,
        "frac_completed_called": sum(r["completed"] for r in records) / n if n else 0.0,
        "frac_eval_unavailable": sum(r["eval_unavailable"] for r in records) / n if n else 0.0,
        "stop_reason_counts": {
            sr: sum(1 for r in records if r["stop_reason"] == sr) for sr in sorted({r["stop_reason"] for r in records})
        },
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir or os.path.expanduser(f"~/{args.run_name}/generation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data_file)
    if args.offset:
        df = df.iloc[args.offset :]
    if args.limit is not None:
        df = df.iloc[: args.limit]
    rows = df.to_dict("records")
    print(f"loaded {len(rows)} tasks from {args.data_file}; n_samples={args.n_samples}")

    # ---- PI hint injection (no-op unless --hints-path) ----------------------------------------------
    # Injected EXACTLY as the OPSD generator does it at rollout time: the per-task hint is appended to
    # the FIRST USER MESSAGE (`content + HINT_TEMPLATE`), i.e. the `splice` style. HINT_TEMPLATE is
    # imported, not copied, because inject.HINT_TEMPLATE and AlfworldEnv._HINT_TEMPLATE
    # must stay byte-identical or an offline-scored with-hint prompt stops
    # matching the one generation actually saw -- which would make a utility-Δ and a ranking AUC
    # incomparable even though both claim to measure "the effect of this PI".
    n_hinted = 0
    if args.hints_path:
        from opsd.pi.artifact import load_task_hints
        from opsd.pi.signals.inject import HINT_TEMPLATE, SECRET_TEMPLATE

        hints = load_task_hints(args.hints_path)
        for row in rows:
            rspec = row.get("reward_spec") or {}
            xinfo = row.get("extra_info") or {}
            if args.hint_key_field:
                tid = xinfo.get(args.hint_key_field)
            else:
                tid = rspec.get("ground_truth") or xinfo.get("task_id")
            hint = hints.get(str(tid)) if tid is not None else None
            if not hint:
                continue
            # parquet hands back a numpy ndarray of dicts, which supports item mutation but NOT insert --
            # so materialise a real list and assign it back. (Prepending to the ndarray raised
            # AttributeError: 'numpy.ndarray' object has no attribute 'insert' and killed every arm.)
            msgs = [dict(m) for m in row.get("prompt")]
            first = next((m for m in msgs if m.get("role") == "user"), None)
            if first is None or not isinstance(first.get("content"), str):
                continue
            if args.hint_inject_style == "system-prepend":
                # A <Secret information> SYSTEM turn before everything else -- what AppWorld OPSD does at
                # rollout time (SelfDistillSkyRLGymGenerator) and what the ranking/attribute passes score.
                # AppWorld conversations open with a user turn and carry no system message, so this is a
                # clean prepend rather than a merge.
                block = (
                    SECRET_TEMPLATE.format(hint_lines=hint)
                    .replace("<|im_start|>system\n", "")
                    .replace("<|im_end|>\n", "")
                )
                msgs = [{"role": "system", "content": block}] + msgs
            else:
                first["content"] = first["content"] + HINT_TEMPLATE.format(hint=hint)
            row["prompt"] = msgs
            n_hinted += 1
        print(
            f"PI: {len(hints)} hints from {args.hints_path}; injected into {n_hinted}/{len(rows)} rows "
            f"({args.hint_inject_style} style, key={args.hint_key_field or 'task_id'})"
        )
        # A PARTIAL match is as wrong as none for a per-rollout run: the unhinted rows silently become a
        # no-hint condition mixed into the with-hint arm, which biases the utility delta toward 0.
        if args.hint_key_field and n_hinted != len(rows):
            raise SystemExit(
                f"--hint-key-field {args.hint_key_field}: only {n_hinted}/{len(rows)} rows "
                f"matched a hint. A per-rollout run must match EVERY row; refusing to mix "
                f"hinted and unhinted rows into one arm."
            )
        if n_hinted == 0:
            raise SystemExit(
                "--hints-path given but 0 tasks matched -- key mismatch, refusing to run a "
                "silently hint-free 'with-hint' condition"
            )
        # Verify the hint text is actually IN the prompt we will send. Injection touching a copy instead
        # of the row is a silent no-op that yields a perfectly plausible utility Δ of ~0.
        _probe = next(
            r
            for r in rows
            if any(
                isinstance(m.get("content"), str)
                and "Secret information" in m["content"]
                or "A reviewer left the following hint" in str(m.get("content"))
                for m in r["prompt"]
            )
        )
        print(
            f"PI: verified injection present in row {_probe.get('extra_info', {}).get('task_id', '?')} "
            f"({len(_probe['prompt'])} messages)"
        )

    server_urls = launch_appworld_servers(args, out_dir) if args.launch_servers else parse_server_urls(args.server_urls)
    if not server_urls:
        raise SystemExit("provide --server-urls or --launch-servers")
    print(f"appworld servers: {len(server_urls)}")

    env_cfg = {
        "server_urls": server_urls,
        "reward_mode": args.reward_mode,
        "max_interactions": args.max_interactions,
        "execute_timeout": args.execute_timeout,
        "repl_output_max_chars": args.repl_output_max_chars,
        "strict_parse": args.strict_parse,
    }
    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "max_tokens": args.max_generate_length,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
    }
    api_key = (os.environ.get(args.api_key_env) or None) if args.api_key_env else None
    if api_key is None and not any(h in args.model_url for h in ("127.0.0.1", "localhost", "0.0.0.0")):
        print(
            f"WARNING: no API key in ${args.api_key_env or '<unset>'}; sending unauthenticated requests to "
            f"{args.model_url}",
            file=sys.stderr,
        )
    # reasoning_history="preserved" only when we actually want thinking carried across turns AND the
    # endpoint is OpenAI-compatible (the ChatClient ignores it on the local-vLLM path).
    client = ChatClient(
        args.model_url,
        args.served_model_name,
        api_key,
        args.request_timeout,
        sampling,
        args.enable_thinking,
        reasoning_history=("preserved" if args.preserve_thinking else None),
    )

    # sample ids are OFFSET, not restarted from 0, so a corpus can be built in passes: n_samples=4 at
    # offsets 0/4/8/12 yields exactly the same 16 distinct `sample` ids as one n_samples=16 run. That
    # matters because uid = f"{task_id}#{sample}" downstream -- overlapping ids across passes would
    # collide silently in every per-rollout join.
    jobs = [(row, s) for row in rows for s in range(args.sample_offset, args.sample_offset + args.n_samples)]
    traj_path = out_dir / "trajectories.jsonl"
    tx_path = out_dir / "transcripts.jsonl"

    resumed = 0
    if args.resume and traj_path.is_file():
        done_keys = set()
        with traj_path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a partial final line from a hard kill
                if r.get("sample", -1) >= 0 and r.get("task_id") != "?":
                    # include hint_key: under --hint-key-field one task_id spans 16 rows, so a
                    # (task_id, sample) key would mark 15 of them "already done" on resume.
                    done_keys.add((str(r["task_id"]), int(r["sample"]), r.get("hint_key")))
        before = len(jobs)
        # A parquet row has NO top-level task_id -- it is nested, exactly as run_rollout (:305) and the
        # hint-injection loop (:436) derive it. Reading row["task_id"] here raises KeyError, which
        # crashes every shard AFTER the env servers are up, so the driver exits 1, the merge still
        # succeeds on the previous quarter's files, and the pass reports rc=0 having generated nothing.
        # The key MUST include hint_key to match what was written above: under --hint-key-field one
        # task_id spans 16 rows, so a (task_id, sample) key marks 15 of them already-done. Building a
        # 2-tuple against 3-tuple keys is the opposite failure -- nothing ever matches and resume
        # silently re-runs the whole arm.
        jobs = [
            (row, s)
            for row, s in jobs
            if (_row_task_id(row), s, (row.get("extra_info") or {}).get("hint_key")) not in done_keys
        ]
        resumed = before - len(jobs)
        print(f"resume: {resumed} rollouts already present, {len(jobs)} remaining")
        if not jobs:
            print("resume: nothing left to generate for this (offset, n_samples) range")

    concurrency = args.concurrency or len(server_urls)
    print(
        f"running {len(jobs)} rollouts, concurrency={concurrency}, "
        f"samples [{args.sample_offset}, {args.sample_offset + args.n_samples})"
    )

    records: List[Dict[str, Any]] = []
    t0 = time.monotonic()
    done_n = 0
    mode = "a" if (args.resume and traj_path.is_file()) else "w"
    txf = open(tx_path, mode) if args.save_transcripts else None
    with ThreadPoolExecutor(max_workers=concurrency) as ex, open(traj_path, mode) as tf:
        futs = [
            ex.submit(
                run_rollout, row, s, env_cfg, client, args.max_turns, args.save_transcripts, args.preserve_thinking
            )
            for row, s in jobs
        ]
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:  # rollout crashed hard -> record a zero
                rec = {
                    "task_id": "?",
                    "sample": -1,
                    "reward": 0.0,
                    "success": 0.0,
                    "partial": 0.0,
                    "num_turns": 0.0,
                    "completed": 0.0,
                    "eval_unavailable": 1.0,
                    "stop_reason": f"rollout_error:{type(e).__name__}",
                }
            conv = rec.pop("conversation", None)  # keep trajectories.jsonl lean; rich data -> transcripts
            turns = rec.pop("turns", None)  # per-turn {think,action,obs} (the thinking tokens)
            if txf is not None:
                txf.write(
                    json.dumps(
                        {
                            "task_id": rec["task_id"],
                            "sample": rec["sample"],
                            "reward": rec["reward"],
                            "preserve_thinking": rec.get("preserve_thinking"),
                            "turns": turns,
                            "conversation": conv,
                        }
                    )
                    + "\n"
                )
                txf.flush()
            records.append(rec)
            tf.write(json.dumps(rec) + "\n")
            tf.flush()
            done_n += 1
            if not args.disable_tqdm and done_n % 5 == 0:
                solved = sum(1 for r in records if r["success"] >= 1.0)
                print(f"  {done_n}/{len(jobs)} done | success so far {solved}/{done_n}", flush=True)

    if txf is not None:
        txf.close()
    # Summarize the WHOLE file, not just this pass. On a resumed/offset run `records` holds only the
    # rollouts this invocation produced, so summarizing it would report pass@k over a fraction of the
    # corpus -- and summary.json is what the downstream provenance and baseline-identity gates read.
    all_records = records
    n_for_summary = args.n_samples
    if resumed or args.sample_offset:
        all_records = []
        with traj_path.open() as fh:
            for line in fh:
                if line.strip():
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        per_task: Dict[str, int] = {}
        for r in all_records:
            per_task[str(r.get("task_id"))] = per_task.get(str(r.get("task_id")), 0) + 1
        n_for_summary = max(per_task.values()) if per_task else args.n_samples
        print(f"summary over the full corpus: {len(all_records)} rollouts, n={n_for_summary}")
    summary = summarize(all_records, n_for_summary, time.monotonic() - t0, args.served_model_name)
    summary.update(
        {
            "data_file": args.data_file,
            "model_url": args.model_url,
            "reward_mode": args.reward_mode,
            "preserve_thinking": args.preserve_thinking,
            # Self-describing conditions: a utility-Δ table is only readable if each condition
            # records WHICH hints it used and how many tasks actually received one.
            "hints_path": args.hints_path,
            "hinted_tasks": n_hinted,
            "hint_inject_style": args.hint_inject_style if args.hints_path else None,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

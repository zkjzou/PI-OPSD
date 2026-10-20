"""Sample ALFWorld ReAct trajectories (e.g. the PI reference trajectories) with an EXTERNAL chat model
behind any OpenAI-compatible ``/chat/completions`` endpoint. No local GPU/vLLM: inference is remote.

This drives the opsd.envs ``AlfworldEnv`` directly (so it reuses the exact R1 prompts, the
<action>..</action> grammar + parsing, admissibility check, and reward), but replaces the
token-in-token-out vLLM client with OpenAI-compatible ``/chat/completions`` calls to the endpoint.
Each turn sends the growing ReAct chat history as proper chat messages.

Endpoint: ``--base-url`` (default ``$OPENAI_BASE_URL``, else https://api.openai.com/v1) and the key in the
env var named by ``--api-key-env`` (default ``OPENAI_API_KEY``). ``--model`` is the id the endpoint
serves, e.g. ``fireworks_ai/<model>`` behind a LiteLLM gateway, or the provider's own id when
``--base-url`` points at the provider directly. The paper's reference trajectories were authored by
GLM-5.2 and Qwen3.7-Plus (``fireworks_ai/glm-5p2``, ``fireworks_ai/qwen3p7-plus``); both are no longer
served, and their trajectories are published with the data bundle.

Cost tracking: every completion's ``usage`` is tallied (prompt/cached/completion tokens) and priced per
1M tokens from ``PRICE_TABLE_PER_1M`` (looked up by the model's short name) or ``--price-*-per-1m``.
``--budget-usd`` stops STARTING new trajectories once the metered cost reaches it. A tqdm bar shows
solved/total with a running token + $ tally.

Prereqs: a running ALFWorld server pool (pass --server-urls or $ALFWORLD_SERVER_URLS) and an ALFWorld
parquet from ``alfworld_dataset.py`` (reward_spec.ground_truth = abs path to each game's traj_data.json).

Run (run_alfworld_glm_litellm_local.sh launches the server pool first):
  export OPENAI_BASE_URL=https://<openai-compatible-endpoint>/v1 OPENAI_API_KEY=...
  python -m opsd.alfworld.generate_alfworld_litellm \
    --data-file <alfworld parquet> --model <model id> \
    --server-urls "$ALFWORLD_SERVER_URLS" \
    --max-turns 50 --max-completion-tokens 8192 --max-concurrency 36 \
    --output ./runs/alfworld_refs/trajectories.jsonl
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from opsd.envs.alfworld.env import AlfworldEnv

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"

# USD per 1M tokens (input, output, cached input) for the models that authored ALFWorld PIs, keyed by the
# model's short name (the part after the last '/'), so "fireworks_ai/glm-5p2" and "glm-5p2" resolve alike.
# Provider list prices at the time of the study: they change, so verify them and override with
# --price-in-per-1m / --price-out-per-1m / --price-cache-read-per-1m. Used only for the cost readout and
# the --budget-usd cap, never for sampling. glm-5p2's cached-input 0.26 is this driver's original default,
# kept for parity with the reference-trajectory runs; the later multi-author runs priced it at 0.14.
PRICE_TABLE_PER_1M: Dict[str, Tuple[float, float, float]] = {
    "glm-5p2": (1.40, 4.40, 0.26),
    "deepseek-v4p1-flash": (0.22, 0.66, 0.01),
    "minimax-m3": (0.30, 1.20, 0.06),
    "qwen3p8-max": (2.00, 6.00, 0.25),
    "kimi-k3": (3.00, 15.00, 0.30),
}


def resolve_prices(
    model: str, price_in: Optional[float], price_out: Optional[float], price_cache: Optional[float]
) -> Optional[Tuple[float, float, Optional[float]]]:
    """(in, out, cached) USD per 1M tokens: explicit flags win, then the table; None if in/out are unknown.

    A missing cached-input price falls back to the input price (CostMeter's convention).
    """
    row = PRICE_TABLE_PER_1M.get(str(model).rsplit("/", 1)[-1].lower())
    if row is not None:
        price_in = row[0] if price_in is None else price_in
        price_out = row[1] if price_out is None else price_out
        price_cache = row[2] if price_cache is None else price_cache
    if price_in is None or price_out is None:
        return None
    return price_in, price_out, price_cache


# --------------------------------------------------------------------------- args
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-file", required=True, help="ALFWorld SkyRL parquet (reward_spec.ground_truth per row).")
    p.add_argument("--output", required=True, help="Output trajectories .jsonl (appended; --resume skips done).")
    p.add_argument("--limit", type=int, default=None, help="Cap rows (smoke).")
    p.add_argument("--n-samples", type=int, default=1, help="Rollouts per game (pass@n / diversity).")
    # ---- model / endpoint ----
    p.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL,
        help=f"OpenAI-compatible base URL. Default: $OPENAI_BASE_URL, else {DEFAULT_BASE_URL}.",
    )
    p.add_argument(
        "--model",
        required=True,
        help="model id as the endpoint names it (OpenAI chat/completions), e.g. fireworks_ai/<model> "
        "behind a LiteLLM gateway.",
    )
    p.add_argument("--api-key", default=None, help="API key (prefer --api-key-env; keys on a command line leak).")
    p.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help=f"env var holding the endpoint key. Default: {DEFAULT_API_KEY_ENV}.",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-completion-tokens", type=int, default=8192, help="per-turn generation cap.")
    p.add_argument("--stop", action="append", default=None, help="stop string (repeatable). Default </action>.")
    p.add_argument(
        "--no-stop",
        action="store_true",
        help="send NO stop strings (the OPSD training contract, stop=[]): the turn runs to end-of-turn.",
    )
    p.add_argument("--top-k", type=int, default=None, help="sent only if set (OPSD training: 20).")
    p.add_argument("--min-p", type=float, default=None, help="sent only if set (OPSD training: 0.0).")
    p.add_argument(
        "--extra-body",
        default=None,
        help="JSON object merged into every request, for provider-specific switches, e.g. "
        '\'{"chat_template_kwargs": {"thinking": true}}\' turns thinking on for azure_ai/DeepSeek-V4-Pro '
        "(which rejects reasoning_effort).",
    )
    p.add_argument("--request-timeout", type=int, default=600)
    p.add_argument("--max-retries", type=int, default=4)
    # ---- env / R1 config ----
    p.add_argument(
        "--server-urls",
        default=os.environ.get("ALFWORLD_SERVER_URLS"),
        help="JSON list or comma-separated alfworld_server URLs. Default $ALFWORLD_SERVER_URLS.",
    )
    p.add_argument("--prompt-style", choices=("react", "sdar"), default="react")
    p.add_argument("--show-admissible", default="false")
    p.add_argument("--invalid-action-feedback", default="true")
    p.add_argument("--max-turns", "--max-steps", dest="max_turns", type=int, default=50)
    p.add_argument("--history-length", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    # Prompt/parse knobs of the OPSD training contract. Each is passed to AlfworldEnv ONLY when given, so the
    # R1 default (the reference-trajectory contract every prior run used) is unchanged.
    p.add_argument("--reasoning-style", choices=("tag", "prefix"), default=None)
    p.add_argument("--reasoning-prefix", default=None, help="e.g. THOUGHTS: (with --reasoning-style prefix).")
    p.add_argument("--reasoning-tag", default=None)
    p.add_argument("--think-tags-in-prompt", default=None)
    p.add_argument("--oneshot-example", default=None)
    p.add_argument("--enable-thinking", default=None, help="env flag (prompt wording), not a sampling knob.")
    p.add_argument("--strict-action-outside-think", default=None)
    p.add_argument(
        "--invalid-action-show-admissible",
        default=None,
        help="unset = env default (true): an INVALID action's feedback lists the admissible actions "
        "even with --show-admissible false (the reference-trajectory contract). false = OPSD contract.",
    )
    p.add_argument("--invalid-action-feedback-guide", default=None)
    p.add_argument("--reward-scale", type=float, default=None)
    p.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="end an episode once a call's prompt_tokens reaches this (OPSD generator: max_input_length).",
    )
    p.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help="hard cap: stop STARTING trajectories once this process's metered cost (usage x prices) "
        "reaches it. Needs known prices (PRICE_TABLE_PER_1M or --price-*-per-1m).",
    )
    p.add_argument("--max-concurrency", type=int, default=36, help="concurrent trajectories (<= #servers).")
    # ---- cost (default: PRICE_TABLE_PER_1M by the model's short name) ----
    p.add_argument("--price-in-per-1m", type=float, default=None, help="USD / 1M input tokens.")
    p.add_argument("--price-out-per-1m", type=float, default=None, help="USD / 1M output tokens.")
    p.add_argument(
        "--price-cache-read-per-1m",
        type=float,
        default=None,
        help="USD / 1M cached input tokens (default: the table, else the input price).",
    )
    # ---- reasoning (fireworks) ----
    p.add_argument(
        "--reasoning-history",
        default=None,
        choices=("preserved", "interleaved", "disabled"),
        help="Fireworks multi-turn reasoning mode. 'preserved' retains prior-turn CoT; the driver ALSO "
        "feeds reasoning_content back into assistant messages (required for the model to see it).",
    )
    p.add_argument(
        "--reasoning-effort",
        default=None,
        choices=("low", "medium", "high"),
        help="Fireworks reasoning effort; unset = model default.",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="If >1: per game, retry up to N attempts and KEEP THE FIRST SUCCESSFUL (won) trajectory "
        "(else the last attempt). Writes ONE record per game with attempts_used/success.",
    )
    return p.parse_args()


def _to_bool(s: Any) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_server_urls(raw: Optional[str]) -> List[str]:
    if not raw:
        raise SystemExit("No --server-urls / $ALFWORLD_SERVER_URLS. Launch the alfworld pool first.")
    raw = raw.strip()
    if raw.startswith("["):
        return list(json.loads(raw))
    return [u.strip() for u in raw.split(",") if u.strip()]


def _load_rows(path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    df = pd.read_parquet(path)
    if limit is not None:
        df = df.iloc[:limit]
    rows = []
    for i, r in df.iterrows():
        d = r.to_dict()
        rs = d.get("reward_spec") or {}
        ei = d.get("extra_info") or {}
        rows.append(
            {
                "row_idx": int(i),
                "prompt": list(d.get("prompt") or [{"role": "user", "content": ""}]),
                "reward_spec": {"method": rs.get("method", "rule"), "ground_truth": rs.get("ground_truth")},
                "extra_info": {k: ei.get(k) for k in ("task_type", "task_id", "split")},
            }
        )
    return rows


# --------------------------------------------------------------------------- cost
class CostMeter:
    def __init__(self, price_in: float, price_out: float, price_cache: Optional[float]):
        self.in_per = price_in / 1_000_000.0
        self.out_per = price_out / 1_000_000.0
        self.cache_per = (price_cache / 1_000_000.0) if price_cache is not None else self.in_per
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def add(self, usage: Dict[str, Any]) -> None:
        pt = int(usage.get("prompt_tokens", 0) or 0)
        ct = int(usage.get("completion_tokens", 0) or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.cached_tokens += cached
        self.calls += 1

    @property
    def cost_usd(self) -> float:
        fresh_in = max(self.prompt_tokens - self.cached_tokens, 0)
        return fresh_in * self.in_per + self.cached_tokens * self.cache_per + self.completion_tokens * self.out_per

    def summary(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost_usd, 4),
        }


# --------------------------------------------------------------------------- chat client
class ChatClient:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: int, max_retries: int):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self._timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def complete(
        self, messages: List[Dict[str, str]], sampling: Dict[str, Any]
    ) -> Tuple[str, Optional[str], Dict[str, Any], Optional[str]]:
        """Returns (content, reasoning_content, usage, finish_reason). Retries transient errors with backoff."""
        payload = {"model": self.model, "messages": messages, **sampling}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                sess = await self._sess()
                async with sess.post(self.url, json=payload, headers=headers) as resp:
                    body = await resp.json(content_type=None)
                    if resp.status >= 400 or "error" in body:
                        msg = body.get("error", body)
                        # 4xx auth/budget/not-found are fatal; retrying won't help (404 = the endpoint does not
                        # serve this --model, or --base-url is wrong).
                        if resp.status in (401, 403, 404) or (
                            isinstance(msg, dict) and msg.get("type") in ("budget_exceeded",)
                        ):
                            raise RuntimeError(f"FATAL endpoint error ({resp.status}): {str(msg)[:300]}")
                        raise RuntimeError(f"endpoint error ({resp.status}): {str(msg)[:300]}")
                    ch = body["choices"][0]
                    msg = ch.get("message") or {}
                    return (
                        msg.get("content") or "",
                        msg.get("reasoning_content"),
                        (body.get("usage") or {}),
                        ch.get("finish_reason"),
                    )
            except Exception as e:  # noqa: BLE001
                last = e
                if "FATAL" in str(e) or attempt == self.max_retries:
                    raise
                await asyncio.sleep(min(2**attempt, 20))
        raise last  # unreachable


# --------------------------------------------------------------------------- rollout
def _normalize_action_text(text: str, stop: List[str]) -> str:
    """Stop strings are excluded from chat content; re-append </action> so the env can parse it.

    Only when </action> IS a stop string: with no stop strings (the OPSD contract) a missing </action> means
    the model never closed the tag (e.g. a max_tokens cut), and repairing it would hide that from the env.
    """
    if "</action>" in stop and "<action>" in text and "</action>" not in text:
        text = text.rstrip() + "</action>"
    return text


def _usage_record(usage: Dict[str, Any]) -> Dict[str, int]:
    """Per-call token accounting kept on every turn (prompt / cached / completion / reasoning)."""
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "cached_tokens": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "reasoning_tokens": int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0),
    }


async def run_trajectory(
    row: Dict[str, Any],
    rep: int,
    env_cfg: Any,
    client: ChatClient,
    sampling: Dict[str, Any],
    stop: List[str],
    max_turns: int,
    meter: CostMeter,
    meter_lock: asyncio.Lock,
    preserve_reasoning: bool = False,
    max_input_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    extras = {"reward_spec": row["reward_spec"], "extra_info": row["extra_info"], "max_turns": max_turns}
    env = AlfworldEnv(env_cfg, extras=extras)
    t0 = time.monotonic()
    turns: List[Dict[str, Any]] = []
    total_reward = 0.0
    won = False
    stop_reason = "max_turns"
    chat: List[Dict[str, Any]] = list(row["prompt"])
    try:
        init_msgs, info = await asyncio.to_thread(env.init, row["prompt"])
        chat = list(init_msgs)
        last_ctx = 0  # prompt + completion tokens of the previous call ~ the next call's input length
        for _turn in range(max_turns):
            if max_input_tokens and last_ctx >= max_input_tokens:
                stop_reason = "max_input_tokens"
                break
            content, reasoning, usage, finish = await client.complete(chat, sampling)
            async with meter_lock:
                meter.add(usage)
            u = _usage_record(usage)
            last_ctx = u["prompt_tokens"] + u["completion_tokens"]
            action_text = _normalize_action_text(content, stop)
            asst_msg: Dict[str, Any] = {"role": "assistant", "content": action_text}
            # Preserved thinking: feed the model's own reasoning_content back so it sees its prior CoT
            # (required in addition to reasoning_history=preserved; omitting it drops the reasoning).
            if preserve_reasoning and reasoning:
                asst_msg["reasoning_content"] = reasoning
            chat.append(asst_msg)
            # BaseTextEnvStepOutput is a TypedDict (plain dict at runtime) -> index, don't attr-access.
            step_out = await asyncio.to_thread(env.step, action_text)
            meta = step_out.get("metadata") or {}
            total_reward += float(step_out.get("reward") or 0.0)
            won = bool(meta.get("won", won))
            turns.append(
                {
                    "assistant": action_text,
                    "reasoning": reasoning,
                    "reward": float(step_out.get("reward") or 0.0),
                    "is_action_valid": meta.get("is_action_valid"),
                    "usage": u,
                    "finish_reason": finish,
                }
            )
            if step_out.get("done"):
                stop_reason = "won" if won else "env_done"
                break
            for obs in step_out.get("observations") or []:
                chat.append(dict(obs))
    except Exception as e:  # noqa: BLE001
        stop_reason = f"error:{type(e).__name__}"
        turns.append({"error": str(e)[:500]})
        if "FATAL" in str(e):
            raise
    finally:
        try:
            await asyncio.to_thread(env.close)
        except Exception:
            pass
    return {
        "row_idx": row["row_idx"],
        "repetition_id": rep,
        "task_id": row["extra_info"].get("task_id"),
        "task_type": row["extra_info"].get("task_type"),
        "game_file": row["reward_spec"].get("ground_truth"),
        "won": won,
        "reward": total_reward,
        "num_turns": len(turns),
        "stop_reason": stop_reason,
        "wall_s": round(time.monotonic() - t0, 1),
        "messages": chat,
        "turns": turns,
    }


async def amain(args: argparse.Namespace) -> None:
    api_key = args.api_key or (os.environ.get(args.api_key_env) if args.api_key_env else None)
    if not api_key:
        raise SystemExit(f"No API key: set ${args.api_key_env} (or pass --api-key-env NAME).")
    prices = resolve_prices(args.model, args.price_in_per_1m, args.price_out_per_1m, args.price_cache_read_per_1m)
    if prices is None and args.budget_usd is not None:
        raise SystemExit(
            f"--budget-usd needs prices, but {args.model!r} is not in PRICE_TABLE_PER_1M: "
            "pass --price-in-per-1m and --price-out-per-1m."
        )
    server_urls = _parse_server_urls(args.server_urls)
    stop = [] if args.no_stop else (args.stop if args.stop is not None else ["</action>"])
    rows = _load_rows(args.data_file, args.limit)

    env_dict = {
        "server_urls": server_urls,
        "env_type": "AlfredTWEnv",
        "prompt_style": args.prompt_style,
        "show_admissible": _to_bool(args.show_admissible),
        "invalid_action_feedback": _to_bool(args.invalid_action_feedback),
        "history_length": args.history_length,
        "max_turns": args.max_turns,
        "seed": args.seed,
        "http_timeout": 120,
    }
    for k in ("reasoning_style", "reasoning_prefix", "reasoning_tag"):
        if getattr(args, k) is not None:
            env_dict[k] = getattr(args, k)
    for k in (
        "think_tags_in_prompt",
        "oneshot_example",
        "enable_thinking",
        "strict_action_outside_think",
        "invalid_action_show_admissible",
        "invalid_action_feedback_guide",
    ):
        if getattr(args, k) is not None:
            env_dict[k] = _to_bool(getattr(args, k))
    if args.reward_scale is not None:
        env_dict["reward_scale"] = args.reward_scale
    env_cfg = OmegaConf.create(env_dict)
    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_completion_tokens,
    }
    if stop:
        sampling["stop"] = stop
    if args.top_k is not None:
        sampling["top_k"] = args.top_k
    if args.min_p is not None:
        sampling["min_p"] = args.min_p
    if args.extra_body:
        extra = json.loads(args.extra_body)
        assert isinstance(extra, dict), "--extra-body must be a JSON object"
        sampling.update(extra)
    if args.reasoning_history:
        sampling["reasoning_history"] = args.reasoning_history  # fireworks top-level param
    if args.reasoning_effort:
        sampling["reasoning_effort"] = args.reasoning_effort
    preserve_reasoning = args.reasoning_history == "preserved"

    out_path = Path(os.path.expanduser(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Provenance: every knob that shaped these trajectories, next to them (the key is never written).
    run_cfg = {k: v for k, v in vars(args).items() if k != "api_key"}
    run_cfg.update(
        {
            "env_cfg": {k: v for k, v in env_dict.items() if k != "server_urls"},
            "sampling": sampling,
            "prices_per_1m": None if prices is None else dict(zip(("in", "out", "cached"), prices)),
        }
    )
    if prices is not None:
        # The legacy price_* keys hold the EFFECTIVE prices (flag or table; cached falls back to the input
        # price, as in CostMeter), so readers that recompute cost from them never see None.
        run_cfg.update(
            price_in_per_1m=prices[0],
            price_out_per_1m=prices[1],
            price_cache_read_per_1m=prices[2] if prices[2] is not None else prices[0],
        )
    (out_path.parent / (out_path.stem + ".run_config.json")).write_text(json.dumps(run_cfg, indent=1))
    retry_mode = args.max_attempts > 1  # keep-first-successful, one record/game
    budget_hit = False

    def over_budget() -> bool:
        nonlocal budget_hit
        if args.budget_usd is not None and meter.cost_usd >= args.budget_usd:
            budget_hit = True
        return budget_hit

    # resume: retry_mode keys by row_idx (one record/game); else by (row_idx, repetition_id)
    done = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
                done.add(r["row_idx"] if retry_mode else (r["row_idx"], r["repetition_id"]))
            except Exception:
                pass

    client = ChatClient(args.base_url, args.model, api_key, args.request_timeout, args.max_retries)
    # Unknown prices: tokens are still metered, the $ readout shows "?" (no budget is possible then).
    meter = CostMeter(*(prices if prices is not None else (0.0, 0.0, 0.0)))
    meter_lock = asyncio.Lock()

    def cost_str() -> str:
        return f"${meter.cost_usd:.2f}" if prices is not None else "$?"

    sem = asyncio.Semaphore(max(1, args.max_concurrency))
    write_lock = asyncio.Lock()
    fout = out_path.open("a")
    solved = 0
    completed = 0

    if retry_mode:
        jobs = [row for row in rows if row["row_idx"] not in done]
        print(
            f"games={len(rows)} max_attempts={args.max_attempts} (keep-first-success) -> {len(jobs)} games to run "
            f"({len(done)} done) | model={args.model} | reasoning_history={args.reasoning_history} | servers={len(server_urls)}"
        )
        bar = tqdm(total=len(jobs), desc=f"{args.model} R1", dynamic_ncols=True)

        async def worker(row: Dict[str, Any]) -> None:
            nonlocal solved, completed
            async with sem:
                if over_budget():
                    return  # not written, so a later run with a higher budget resumes it
                rec, attempt = None, 0
                for attempt in range(args.max_attempts):
                    rec = await run_trajectory(
                        row,
                        attempt,
                        env_cfg,
                        client,
                        sampling,
                        stop,
                        args.max_turns,
                        meter,
                        meter_lock,
                        preserve_reasoning,
                        args.max_input_tokens,
                    )
                    if rec["won"]:
                        break
                rec["attempts_used"] = attempt + 1
                rec["success"] = bool(rec["won"])
            async with write_lock:
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                completed += 1
                solved += int(rec["success"])
                bar.update(1)
                bar.set_postfix_str(
                    f"success={solved}/{completed} rate={solved/max(completed,1):.3f} "
                    f"tok={meter.prompt_tokens + meter.completion_tokens} {cost_str()}"
                )

        coros = [worker(row) for row in jobs]
    else:
        jobs2 = [(row, rep) for row in rows for rep in range(args.n_samples) if (row["row_idx"], rep) not in done]
        print(
            f"games={len(rows)} n_samples={args.n_samples} -> {len(jobs2)} trajectories to run "
            f"({len(done)} already done) | model={args.model} | servers={len(server_urls)}"
        )
        bar = tqdm(total=len(jobs2), desc=f"{args.model} R1", dynamic_ncols=True)

        async def worker(row: Dict[str, Any], rep: int) -> None:  # noqa: F811
            nonlocal solved, completed
            async with sem:
                if over_budget():
                    return  # not written, so a later run with a higher budget resumes it
                rec = await run_trajectory(
                    row,
                    rep,
                    env_cfg,
                    client,
                    sampling,
                    stop,
                    args.max_turns,
                    meter,
                    meter_lock,
                    preserve_reasoning,
                    args.max_input_tokens,
                )
            async with write_lock:
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                completed += 1
                solved += int(rec["won"])
                bar.update(1)
                bar.set_postfix_str(
                    f"solved={solved}/{completed} pass@1={solved/max(completed,1):.3f} "
                    f"tok={meter.prompt_tokens + meter.completion_tokens} {cost_str()}"
                )

        coros = [worker(row, rep) for row, rep in jobs2]

    try:
        await asyncio.gather(*coros)
    finally:
        bar.close()
        fout.close()
        await client.close()

    s = meter.summary()
    n = completed
    print("\n==================== SUMMARY ====================")
    print(f"model={args.model}  trajectories={n}  solved(pass@1 over all)={solved}/{n}={solved/max(n,1):.3f}")
    print(
        f"tokens: prompt={s['prompt_tokens']} (cached={s['cached_tokens']}) completion={s['completion_tokens']} "
        f"total={s['total_tokens']}"
    )
    if prices is not None:
        print(f"COST: ${s['cost_usd']:.4f}  (@ ${prices[0]}/M in, ${prices[1]}/M out)")
    else:
        print(f"COST: unknown (no price for {args.model!r}; pass --price-in-per-1m/--price-out-per-1m)")
    if budget_hit:
        print(f"BUDGET REACHED: ${args.budget_usd:.2f} -- remaining trajectories were NOT started (resumable)")
    print(f"output: {out_path}")


def main() -> None:
    asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    main()

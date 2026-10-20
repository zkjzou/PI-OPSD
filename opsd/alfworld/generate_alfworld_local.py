"""Generation-only ALFWorld rollouts against a local/remote model HTTP server.

It does not start Ray, FSDP, or SkyRL training workers. It reuses ``SkyRLGymGenerator`` for
the multi-turn env loop, talks to a pool of ALFWorld HTTP env servers, and sends
model requests to an already running vLLM/OpenAI-compatible server.

Prereqs:
  1. Serve the policy model, for example:
       CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-1.7B \
         --served-model-name Qwen3-1.7B --host 0.0.0.0 --port 8123

  2. Have ALFWorld parquet built as in ``alfworld_dataset.py``.

  3. Either pass ``--server-urls '["http://127.0.0.1:8100", ...]'`` or let this
     script start local ALFWorld env servers with ``--launch-alfworld-servers`` (needs the
     ALFWorld venv + data from ``scripts/setup/setup_alfworld.sh``; the servers it starts
     are stopped on exit, and only those).

Example:
  python opsd/alfworld/generate_alfworld_local.py \
    --model-url http://127.0.0.1:8123 \
    --served-model-name Qwen3-1.7B \
    --tokenizer Qwen/Qwen3-1.7B \
    --data-file ~/data/alfworld_parquet/validation.parquet \
    --launch-alfworld-servers --num-alfworld-servers 8 \
    --limit 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import aiohttp
import pandas as pd
from transformers import AutoTokenizer

DEFAULT_DATA_DIR = "~/data/alfworld_parquet"
DEFAULT_DATA_FILE = f"{DEFAULT_DATA_DIR}/validation.parquet"
DEFAULT_RUN_NAME = "alfworld_gen_local"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--data-file", default=None, help="SkyRL-format ALFWorld parquet file.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory with ALFWorld parquet files.")
    parser.add_argument(
        "--eval-dataset",
        default="valid_seen",
        help=(
            "Dataset alias used when --data-file is omitted. Accepted aliases include train, valid_seen, "
            "eval_in_distribution, valid_unseen, and eval_out_of_distribution."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "--log-dir",
        dest="output_dir",
        default=None,
        help=f"Default: ~/{DEFAULT_RUN_NAME}/generation",
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of parquet rows for smoke runs.")
    parser.add_argument("--offset", type=int, default=0, help="Start row offset in the parquet file.")
    parser.add_argument("--n-samples", type=int, default=1, help="Rollouts per parquet row.")
    parser.add_argument("--batch-size", type=int, default=None, help="Number of concurrent rollouts submitted at once.")

    parser.add_argument(
        "--model-url",
        "--base-url",
        dest="model_url",
        default="http://127.0.0.1:8123",
        help="Base URL of the model server; /v1 suffix is accepted.",
    )
    parser.add_argument(
        "--served-model-name",
        "--model-name",
        dest="served_model_name",
        default="Qwen3-1.7B",
        help="Model name accepted by the server.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HF tokenizer path/name. Defaults to Qwen/<model-name> for Qwen served names.",
    )
    parser.add_argument(
        "--endpoint-mode",
        choices=("openai", "skyrl-generate"),
        default="openai",
        help="Use /v1/completions for vLLM serve, or /inference/v1/generate for a SkyRL inference server.",
    )
    parser.add_argument("--api-key", default=None, help="Optional bearer token for the model server.")
    parser.add_argument("--api-key-env", default=None, help="Optional env var containing the bearer token.")
    parser.add_argument("--request-timeout", "--timeout", dest="request_timeout", type=int, default=900)
    parser.add_argument("--model-max-concurrency", type=int, default=None)
    parser.add_argument(
        "--request-workers",
        type=int,
        default=None,
        help="Prompt-agent-compatible alias for both --batch-size and --model-max-concurrency.",
    )
    parser.add_argument(
        "--request-token-ids",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ask an OpenAI-compatible vLLM server to return token IDs when supported.",
    )

    parser.add_argument(
        "--server-urls",
        default=os.environ.get("ALFWORLD_SERVER_URLS"),
        help="JSON list or comma-separated ALFWorld env server URLs. Defaults to $ALFWORLD_SERVER_URLS.",
    )
    parser.add_argument(
        "--launch-alfworld-servers",
        action="store_true",
        help="Launch local ALFWorld env servers via server/launch_alfworld_servers.sh.",
    )
    parser.add_argument("--num-alfworld-servers", "--env-num", dest="num_alfworld_servers", type=int, default=8)
    parser.add_argument("--alfworld-base-port", type=int, default=None)
    parser.add_argument("--alfworld-host", default="0.0.0.0")
    parser.add_argument("--alfworld-advertise-host", default="127.0.0.1")
    parser.add_argument(
        "--alfworld-venv",
        default=os.environ.get("ALFWORLD_VENV") or os.path.expanduser("~/alfworld-venv"),
        help="Existing ALFWorld venv (default: $ALFWORLD_VENV, else ~/alfworld-venv).",
    )
    parser.add_argument(
        "--alfworld-data",
        default=os.environ.get("ALFWORLD_DATA") or "~/data/alfworld",
        help="ALFWorld data root (default: $ALFWORLD_DATA, else ~/data/alfworld).",
    )
    parser.add_argument(
        "--alfworld-log-dir",
        default=None,
        help="Server logs + PID files of the launched pool (default: <output-dir>/alfworld_servers).",
    )
    parser.add_argument("--health-timeout", type=int, default=None)
    parser.add_argument("--skip-server-health-check", action="store_true")

    parser.add_argument("--max-turns", "--max-steps", dest="max_turns", type=int, default=30)
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument(
        "--prompt-style",
        choices=("sdar", "react"),
        default="sdar",
        help="ALFWorld prompt format: 'sdar' (windowed/no-history templates) or 'react'.",
    )
    parser.add_argument(
        "--show-admissible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="List admissible actions in each prompt (honored by both prompt styles).",
    )
    parser.add_argument(
        "--invalid-action-feedback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tell the model when its action was invalid/inadmissible (matches the R1 GRPO recipe).",
    )
    parser.add_argument(
        "--invalid-action-show-admissible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When invalid-action-feedback is on AND show-admissible is off, whether that feedback ALSO "
        "lists the state-specific admissible actions (env default True). Use --no-invalid-action-show-admissible "
        "to match the OPSD eval (grammar guide only, no admissibles ever).",
    )
    parser.add_argument(
        "--strict-action-outside-think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="STRICTEST parse (thinking models): the committed action must be a non-empty <action> tag AFTER "
        "a closed </think>. If </think> never closed (unfinished / hit the token cap) or there is no "
        "<action> after it, the turn has NO valid action (== empty <action></action>). Requires "
        "--enable-thinking.",
    )
    parser.add_argument(
        "--task-ids-file",
        default=None,
        help="Path to a JSON list (or newline-delimited file) of task_ids; keep ONLY those parquet rows. "
        "Used by the rescue pass to re-sample just the tasks that have no successful trajectory yet.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing trajectories.jsonl in --output-dir: skip the (uid, repetition_id) pairs "
        "already present and APPEND only the missing rollouts (instead of truncating and regenerating). "
        "A fully-complete condition generates nothing.",
    )
    parser.add_argument(
        "--hints-path",
        default=os.environ.get("OPSD_HINTS_PATH"),
        help=(
            "Optional PI hints artifact (JSON {task_id: hint} or {task_id: {'task': hint, ...}}); "
            "the per-task hint is appended to the first user turn. Default: $OPSD_HINTS_PATH."
        ),
    )
    parser.add_argument(
        "--hint-placement",
        choices=("user", "system"),
        default="user",
        help="'user' (default): the hint is appended to the first user turn -- the PI-utility convention. "
        "'system': it goes in a leading system turn wrapped EXACTLY as the OPSD teacher's "
        "<Secret information> block (opsd/pi/signals/inject.py SECRET_TEMPLATE), so a rollout "
        "shows what the frozen teacher would do with the PI.",
    )
    # ---- ALFWorld-Passcode + the OPSD Instruct prompt contract (opsd/envs/alfworld/passcode.py) ----
    # Without these the generator can only reproduce the plain react prompt, not the THOUGHTS: prompt the
    # OPSD Instruct arms train on, so a utility number would describe a different env than the one trained.
    parser.add_argument(
        "--passcode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ALFWorld-Passcode: completing the task reveals a private key; ONLY the submitted key is scored.",
    )
    parser.add_argument(
        "--passcode-salt",
        default="alfworld-passcode-v1",
        help="Must match the salt the passcode hints were built with.",
    )
    parser.add_argument("--reasoning-style", choices=("tag", "prefix"), default="tag")
    parser.add_argument("--reasoning-prefix", default="THOUGHTS:")
    parser.add_argument("--reasoning-tag", default="think")
    parser.add_argument("--think-tags-in-prompt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--max-input-length", type=int, default=4096)
    parser.add_argument(
        "--max-generate-length",
        "--max-completion-tokens",
        dest="max_generate_length",
        type=int,
        default=512,
    )
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stop", action="append", default=None, help="Stop string. Repeatable. Default: </action>.")
    parser.add_argument("--no-stop", action="store_true", help="Disable stop strings.")
    parser.add_argument("--seed", type=int, default=0, help="ALFWorld env seed.")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward Qwen3 enable_thinking to tokenizer.apply_chat_template.",
    )
    parser.add_argument("--chat-template-file", default=None, help="Optional custom chat template file.")
    parser.add_argument("--max-env-workers", type=int, default=64)
    parser.add_argument("--save-token-ids", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()
    return normalize_args(args)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.request_workers is not None:
        if args.batch_size is None:
            args.batch_size = args.request_workers
        if args.model_max_concurrency is None:
            args.model_max_concurrency = args.request_workers
    if args.batch_size is None:
        args.batch_size = 32
    if args.model_max_concurrency is None:
        args.model_max_concurrency = 64
    if args.health_timeout is None:
        args.health_timeout = args.request_timeout

    if args.data_file is None:
        args.data_file = str(resolve_eval_dataset(args.eval_dataset, args.data_dir))

    if args.alfworld_base_port is None:
        args.alfworld_base_port = default_alfworld_base_port(args.model_url)

    # Prompt-agent-style invocation usually does not pass --launch-alfworld-servers;
    # if no explicit server pool is provided, launch one locally.
    if not args.server_urls:
        args.launch_alfworld_servers = True

    return args


def resolve_eval_dataset(eval_dataset: str, data_dir: str) -> Path:
    key = eval_dataset.strip().lower().replace("-", "_")
    name_map = {
        "train": "train.parquet",
        "valid_seen": "validation.parquet",
        "validation": "validation.parquet",
        "eval_seen": "validation.parquet",
        "eval_in_distribution": "validation.parquet",
        "in_distribution": "validation.parquet",
        "id": "validation.parquet",
        "valid_unseen": "validation_unseen.parquet",
        "validation_unseen": "validation_unseen.parquet",
        "eval_unseen": "validation_unseen.parquet",
        "eval_out_of_distribution": "validation_unseen.parquet",
        "out_of_distribution": "validation_unseen.parquet",
        "ood": "validation_unseen.parquet",
    }
    filename = name_map.get(key)
    if filename is None:
        if key.endswith(".parquet"):
            return Path(eval_dataset).expanduser()
        raise ValueError(f"Unknown --eval-dataset {eval_dataset!r}. Use --data-file, or one of: {sorted(name_map)}")
    return Path(data_dir).expanduser() / filename


def default_alfworld_base_port(model_url: str) -> int:
    first = str(model_url).split(",")[0].strip()  # tolerate a comma-list of replica URLs
    parsed = urlparse(first)
    model_port = parsed.port
    # The prompt-agent-style command often uses model port 8100. The ALFWorld
    # env servers need their own ports, so avoid that collision by default.
    return 8200 if model_port == 8100 else 8100


def infer_tokenizer_name(tokenizer: Optional[str], model_name: str) -> str:
    if tokenizer:
        return tokenizer
    if "/" not in model_name and model_name.lower().startswith("qwen"):
        return f"Qwen/{model_name}"
    return model_name


def _to_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        value = value.tolist()
    return value


def _normalize_mapping(value: Any) -> Dict[str, Any]:
    value = _to_python(value)
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        loaded = json.loads(stripped)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected JSON object, got {type(loaded).__name__}: {value[:80]}")
        return loaded
    raise TypeError(f"Expected mapping-like value, got {type(value).__name__}")


def _normalize_prompt(value: Any) -> List[Dict[str, str]]:
    value = _to_python(value)
    if value is None:
        return [{"role": "user", "content": ""}]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            value = json.loads(stripped)
        else:
            return [{"role": "user", "content": value}]
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, list):
        return [
            {"role": str(_to_python(msg).get("role", "user")), "content": str(_to_python(msg).get("content", ""))}
            for msg in value
        ]
    return [{"role": "user", "content": ""}]


def _jsonable(value: Any) -> Any:
    value = _to_python(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def parse_server_urls(raw: Optional[str]) -> List[str]:
    if raw is None or not raw.strip():
        return []
    raw = raw.strip()
    if raw.startswith("["):
        urls = json.loads(raw)
    else:
        urls = [part.strip() for part in raw.split(",") if part.strip()]
    if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
        raise ValueError("--server-urls must be a JSON list of strings or a comma-separated string.")
    return [url.rstrip("/") for url in urls]


def load_items(data_file: str, limit: Optional[int], offset: int) -> List[Dict[str, Any]]:
    path = Path(data_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"ALFWorld parquet not found: {path}")

    df = pd.read_parquet(path)
    if offset:
        df = df.iloc[offset:]
    if limit is not None:
        df = df.iloc[:limit]

    items: List[Dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        prompt = _normalize_prompt(row.get("prompt"))
        reward_spec = _normalize_mapping(row.get("reward_spec", {}))
        extra_info = _normalize_mapping(row.get("extra_info", {}))
        game_file = reward_spec.get("ground_truth") or extra_info.get("game_file")
        if not game_file:
            raise ValueError(f"Row {row_idx} is missing reward_spec.ground_truth")
        task_id = extra_info.get("task_id") or Path(str(game_file)).parent.name
        uid = f"{row_idx}:{task_id}"
        items.append(
            {
                "uid": uid,
                "row_idx": int(row_idx),
                "prompt": prompt,
                "env_class": str(row.get("env_class", "alfworld") or "alfworld"),
                "reward_spec": reward_spec,
                "extra_info": extra_info,
            }
        )
    return items


def item_task_id(item: Dict[str, Any]) -> str:
    """Stable per-task id used to key the PI hints artifact (matches load_items / alfworld_dataset)."""
    extra = item.get("extra_info") or {}
    tid = extra.get("task_id")
    if tid:
        return str(tid)
    uid = str(item.get("uid", ""))
    if ":" in uid:
        return uid.split(":", 1)[1]
    game_file = (item.get("reward_spec") or {}).get("ground_truth") or extra.get("game_file")
    return Path(str(game_file)).parent.name if game_file else uid


def load_task_hints(path: Optional[str]) -> Dict[str, str]:
    """Load a {task_id: hint_text} map from a PI hints artifact.

    Accepts the v1 schema (plain-string values) and the v2 schema
    ({"task": str, "steps": {...}, "meta": {...}}); only the task-level string is used here.
    Self-contained (no pi.artifact import) so this standalone script has no extra deps.
    """
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"PI hints artifact not found: {p}")
    raw = json.loads(p.read_text())
    hints: Dict[str, str] = {}
    for key, val in raw.items():
        if key == "__schema__":
            continue
        if isinstance(val, str):
            if val.strip():
                hints[key] = val
        elif isinstance(val, dict):
            task_hint = val.get("task")
            if task_hint and str(task_hint).strip():
                hints[key] = str(task_hint)
    return hints


def launch_alfworld_servers(args: argparse.Namespace, output_dir: Path) -> Tuple[List[str], Path]:
    script = Path(__file__).resolve().parent / "server" / "launch_alfworld_servers.sh"
    log_dir = Path(args.alfworld_log_dir).expanduser() if args.alfworld_log_dir else output_dir / "alfworld_servers"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "ALFWORLD_VENV": str(Path(args.alfworld_venv).expanduser()),
            "ALFWORLD_DATA": str(Path(args.alfworld_data).expanduser()),
            "NUM_ALFWORLD_SERVERS": str(args.num_alfworld_servers),
            "ALFWORLD_BASE_PORT": str(args.alfworld_base_port),
            "ALFWORLD_HOST": args.alfworld_host,
            "ALFWORLD_ADVERTISE_HOST": args.alfworld_advertise_host,
            "ALFWORLD_MAX_STEPS": str(args.max_turns),
            "ALFWORLD_LOG_DIR": str(log_dir),
            # This pool's own PID dir, so kill_launched_servers() stops exactly the servers started here.
            "ALFWORLD_PID_DIR": str(log_dir),
        }
    )
    logger.info("Launching %d ALFWorld env servers via %s", args.num_alfworld_servers, script)
    proc = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError("launch_alfworld_servers.sh failed\n" f"stdout:\n{proc.stdout}\n" f"stderr:\n{proc.stderr}")
    if proc.stderr.strip():
        logger.info(proc.stderr.strip())
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("launch_alfworld_servers.sh did not print server URLs.")
    urls = parse_server_urls(lines[-1])
    return urls, log_dir / "alfworld_serve.pids"


def kill_launched_servers(pid_file: Path) -> None:
    if not pid_file.exists():
        return
    pids = [int(pid) for pid in pid_file.read_text().split() if pid.strip().isdigit()]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            logger.warning("Failed to terminate ALFWorld server pid %s: %s", pid, exc)


async def wait_for_servers(server_urls: Sequence[str], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    pending = set(server_urls)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
        while pending and time.monotonic() < deadline:
            for url in list(pending):
                try:
                    async with session.get(f"{url}/health") as resp:
                        if resp.status == 200:
                            pending.remove(url)
                except aiohttp.ClientError:
                    pass
            if pending:
                await asyncio.sleep(1)
    if pending:
        raise RuntimeError(f"Timed out waiting for ALFWorld servers: {sorted(pending)}")


def normalize_model_base_url(model_url: str) -> str:
    url = model_url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def clean_sampling_params(params: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if key == "additional_kwargs" and isinstance(value, dict):
            cleaned.update({k: v for k, v in value.items() if v is not None})
        else:
            cleaned[key] = value
    return cleaned


class LocalHTTPInferenceClient:
    """Small async client matching the ``SkyRLGymGenerator`` inference interface."""

    def __init__(
        self,
        model_url: str,
        model_name: str,
        tokenizer: Any,
        endpoint_mode: str,
        request_timeout: int,
        max_concurrency: int,
        api_key: Optional[str] = None,
        request_token_ids: bool = False,
    ):
        # Accept a comma-list of model URLs -> round-robin requests across independent replicas
        # (data-parallel via separate single-GPU servers, since vLLM's DP coordinator is flaky here).
        self.base_urls = [normalize_model_base_url(u) for u in str(model_url).split(",") if u.strip()]
        self.base_url = self.base_urls[0]
        self._url_rr = 0
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.endpoint_mode = endpoint_mode
        self.api_key = api_key
        self.request_token_ids = request_token_ids
        self._session: Optional[aiohttp.ClientSession] = None
        self._timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=request_timeout)
        self._sem = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None
        # Long multi-hour runs at high concurrency see occasional transient connection drops
        # (keep-alive closed under load); retry those a few times so one drop doesn't abort the
        # whole asyncio.gather batch. Deterministic model errors (4xx/context-too-long) still fail loud.
        self._max_retries = 5
        self._retry_backoff = 1.0

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    def _next_base_url(self) -> str:
        if len(self.base_urls) == 1:
            return self.base_urls[0]
        url = self.base_urls[self._url_rr % len(self.base_urls)]
        self._url_rr += 1
        return url

    def _headers(self, session_id: Optional[Any]) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if session_id is not None:
            headers["X-Session-ID"] = str(session_id)
        return headers

    async def _post(self, url: str, payload: Dict[str, Any], session_id: Optional[Any]) -> Dict[str, Any]:
        for attempt in range(self._max_retries):
            try:
                session = await self._get_session()
                async with session.post(url, json=payload, headers=self._headers(session_id)) as resp:
                    try:
                        body = await resp.json(content_type=None)
                    except Exception:
                        text = await resp.text()
                        raise RuntimeError(f"Non-JSON response from {url} ({resp.status}): {text[:1000]}")
                    if resp.status >= 400 or "error" in body:
                        raise RuntimeError(f"Model server error from {url} ({resp.status}): {body}")
                    return body
            except (aiohttp.ClientError, asyncio.TimeoutError):
                # transient network/connection error (e.g. ServerDisconnectedError) — retry with backoff
                if attempt >= self._max_retries - 1:
                    raise
                await asyncio.sleep(self._retry_backoff * (2**attempt))
        raise RuntimeError(f"unreachable: exhausted retries posting to {url}")

    async def _generate_one(
        self,
        prompt_token_ids: List[int],
        sampling_params: Dict[str, Any],
        session_id: Optional[Any],
        model: str,
    ) -> Dict[str, Any]:
        base = self._next_base_url()
        if self.endpoint_mode == "skyrl-generate":
            payload = {
                "model": model,
                "token_ids": prompt_token_ids,
                "sampling_params": clean_sampling_params(sampling_params),
            }
            body = await self._post(f"{base}/inference/v1/generate", payload, session_id)
            choice = body["choices"][0]
            token_ids = choice["token_ids"]
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            logprobs = choice.get("logprobs")
            response_logprobs = None
            if logprobs is not None and logprobs.get("content"):
                response_logprobs = [entry["logprob"] for entry in logprobs["content"]]
            return {
                "text": text,
                "token_ids": token_ids,
                "finish_reason": choice.get("finish_reason"),
                "response_logprobs": response_logprobs,
            }

        payload = clean_sampling_params(sampling_params)
        payload.update({"model": model, "prompt": prompt_token_ids})
        if self.request_token_ids:
            payload["return_token_ids"] = True
        body = await self._post(f"{base}/v1/completions", payload, session_id)
        choice = body["choices"][0]
        text = choice.get("text", "")
        token_ids = (
            choice.get("token_ids")
            or choice.get("output_token_ids")
            or choice.get("tokens")
            or self.tokenizer.encode(text, add_special_tokens=False)
        )
        return {
            "text": text,
            "token_ids": token_ids,
            "finish_reason": choice.get("finish_reason"),
            "response_logprobs": None,
        }

    async def generate(self, input_batch: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
        prompt_token_ids = input_batch.get("prompt_token_ids")
        if prompt_token_ids is None:
            raise ValueError("LocalHTTPInferenceClient expects prompt_token_ids.")
        sampling_params = input_batch.get("sampling_params") or {}
        session_ids = input_batch.get("session_ids") or [None] * len(prompt_token_ids)
        model_name = model or self.model_name

        async def run_one(idx: int) -> Dict[str, Any]:
            if self._sem is None:
                return await self._generate_one(prompt_token_ids[idx], sampling_params, session_ids[idx], model_name)
            async with self._sem:
                return await self._generate_one(prompt_token_ids[idx], sampling_params, session_ids[idx], model_name)

        results = await asyncio.gather(*(run_one(i) for i in range(len(prompt_token_ids))))
        add_logprobs = any(result.get("response_logprobs") is not None for result in results)
        return {
            "responses": [result["text"] for result in results],
            "response_ids": [result["token_ids"] for result in results],
            "stop_reasons": [result.get("finish_reason") or "" for result in results],
            "response_logprobs": [result.get("response_logprobs") for result in results] if add_logprobs else None,
            "rollout_expert_indices": None,
        }


def make_generator(args: argparse.Namespace, server_urls: List[str], tokenizer: Any, client: LocalHTTPInferenceClient):
    from opsd.envs.registry import register_opsd_envs
    from opsd.skyrl_ext.config import OPSDSkyRLGymConfig
    from skyrl.train.config import (
        ChatTemplateConfig,
        GeneratorConfig,
        SamplingParams,
    )
    from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator

    stop = None if args.no_stop else (args.stop if args.stop is not None else ["</action>"])
    sampling_cfg = SamplingParams(
        max_generate_length=args.max_generate_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        logprobs=None,
        stop=stop,
    )
    sampling_params = {
        "min_tokens": 1,
        "skip_special_tokens": True,
        "include_stop_str_in_output": True,
        "max_tokens": sampling_cfg.max_generate_length,
        "temperature": sampling_cfg.temperature,
        "top_p": sampling_cfg.top_p,
        "top_k": sampling_cfg.top_k,
        "min_p": sampling_cfg.min_p,
        "logprobs": sampling_cfg.logprobs,
        "stop": list(stop) if stop is not None else None,
        # forwarded to vLLM /v1/completions; default no-op (0.0 / 1.0) for backward compat.
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
    }

    chat_template = ChatTemplateConfig()
    if args.chat_template_file:
        chat_template = ChatTemplateConfig(source="file", name_or_path=args.chat_template_file)

    generator_cfg = GeneratorConfig(
        n_samples_per_prompt=args.n_samples,
        batched=False,
        max_turns=args.max_turns,
        max_input_length=args.max_input_length,
        chat_template=chat_template,
        chat_template_kwargs={"enable_thinking": args.enable_thinking},
        sampling_params=sampling_cfg,
        use_conversation_multi_turn=True,
        append_eos_token_after_stop_str_in_multi_turn=True,
        apply_overlong_filtering=False,
    )
    generator_cfg.inference_engine.served_model_name = args.served_model_name

    # OPSD's SkyRLGymConfig: upstream's has no `.alfworld` (the generator reads getattr(cfg, env_class, {})).
    skyrl_gym_cfg = OPSDSkyRLGymConfig(max_env_workers=args.max_env_workers)
    skyrl_gym_cfg.alfworld.server_urls = server_urls
    skyrl_gym_cfg.alfworld.max_turns = args.max_turns
    skyrl_gym_cfg.alfworld.history_length = args.history_length
    skyrl_gym_cfg.alfworld.prompt_style = args.prompt_style
    skyrl_gym_cfg.alfworld.show_admissible = args.show_admissible
    skyrl_gym_cfg.alfworld.invalid_action_feedback = args.invalid_action_feedback
    skyrl_gym_cfg.alfworld.invalid_action_show_admissible = args.invalid_action_show_admissible
    # Mirror the generator's chat_template_kwargs.enable_thinking into the env so the parser knows this is a
    # thinking model (previously unset -> env used the relaxed parser even for --enable-thinking runs).
    skyrl_gym_cfg.alfworld.enable_thinking = args.enable_thinking
    skyrl_gym_cfg.alfworld.strict_action_outside_think = args.strict_action_outside_think
    skyrl_gym_cfg.alfworld.seed = args.seed
    skyrl_gym_cfg.alfworld.http_timeout = args.request_timeout
    skyrl_gym_cfg.alfworld.passcode_mode = args.passcode
    skyrl_gym_cfg.alfworld.passcode_salt = args.passcode_salt
    skyrl_gym_cfg.alfworld.hint_placement = args.hint_placement
    skyrl_gym_cfg.alfworld.reasoning_style = args.reasoning_style
    skyrl_gym_cfg.alfworld.reasoning_prefix = args.reasoning_prefix
    skyrl_gym_cfg.alfworld.reasoning_tag = args.reasoning_tag
    skyrl_gym_cfg.alfworld.think_tags_in_prompt = args.think_tags_in_prompt
    skyrl_gym_cfg.alfworld.reward_scale = args.reward_scale

    # The generator builds every env by id through the skyrl_gym registry, which knows only upstream's envs
    # until the OPSD ones are registered.
    register_opsd_envs()
    generator = SkyRLGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=skyrl_gym_cfg,
        inference_engine_client=client,
        tokenizer=tokenizer,
        policy_model_name=args.served_model_name,
    )
    return generator, sampling_params


def expand_batch(items: Sequence[Dict[str, Any]], n_samples: int) -> List[Tuple[Dict[str, Any], Any]]:
    from skyrl.train.generators.base import TrajectoryID

    expanded = []
    for item in items:
        for rep in range(n_samples):
            expanded.append((item, TrajectoryID(instance_id=item["uid"], repetition_id=rep)))
    return expanded


def make_generator_input(
    batch: Sequence[Tuple[Dict[str, Any], Any]],
    sampling_params: Dict[str, Any],
    hints: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    hints = hints or {}
    prompts = [item["prompt"] for item, _ in batch]
    env_classes = [item["env_class"] for item, _ in batch]
    env_extras = []
    for item, _tid in batch:
        ei = dict(item["extra_info"])
        # Thread the per-rollout id to the env so a per-turn dump (ALFWORLD_TURN_DUMP) can key on it.
        # Needed for the NOT-preserved-thinking mode, where the retokenized response_ids drop
        # non-last-turn <think> -> the env dump is where per-turn thinking is recovered.
        if _tid is not None:
            ei["uid"] = getattr(_tid, "instance_id", ei.get("uid"))
            ei["repetition_id"] = getattr(_tid, "repetition_id", ei.get("repetition_id"))
        extra = {
            "reward_spec": dict(item["reward_spec"]),
            "extra_info": ei,
        }
        # PI: per-task hint appended to the first user turn by AlfworldEnv (gated; empty -> no-op).
        hint = hints.get(item_task_id(item))
        if hint:
            extra["hint"] = hint
        env_extras.append(extra)
    return {
        "prompts": prompts,
        "env_classes": env_classes,
        "env_extras": env_extras,
        "sampling_params": sampling_params,
        "trajectory_ids": [trajectory_id for _, trajectory_id in batch],
        "batch_metadata": None,
    }


def reward_sum(reward: Any) -> float:
    reward = _to_python(reward)
    if isinstance(reward, list):
        return float(sum(float(x) for x in reward))
    return float(reward)


def records_from_output(
    batch: Sequence[Tuple[Dict[str, Any], Any]],
    output: Dict[str, Any],
    tokenizer: Any,
    save_token_ids: bool,
    hints: Optional[Dict[str, str]] = None,
    prompt_style: str = "sdar",
    history_length: int = 5,
) -> List[Dict[str, Any]]:
    hints = hints or {}
    records = []
    response_ids = output["response_ids"]
    loss_masks = output["loss_masks"]
    rewards = output["rewards"]
    env_metrics = output.get("env_metrics") or [{} for _ in batch]
    stop_reasons = output.get("stop_reasons") or [None for _ in batch]
    prompt_token_ids = output.get("prompt_token_ids") or [None for _ in batch]

    for i, (item, trajectory_id) in enumerate(batch):
        tid = item_task_id(item)
        assistant_ids = [tok for tok, mask in zip(response_ids[i], loss_masks[i]) if mask]
        record = {
            "uid": trajectory_id.instance_id,
            "repetition_id": trajectory_id.repetition_id,
            "row_idx": item["row_idx"],
            "task_id": tid,
            "game_file": item["reward_spec"].get("ground_truth"),
            "extra_info": _jsonable(item["extra_info"]),
            "reward": reward_sum(rewards[i]),
            "stop_reason": stop_reasons[i],
            "env_metrics": _jsonable(env_metrics[i]),
            "num_tokens": len(response_ids[i]),
            "num_assistant_tokens": int(sum(loss_masks[i])),
            "assistant_text": tokenizer.decode(assistant_ids, skip_special_tokens=True),
            "decoded_response": tokenizer.decode(response_ids[i], skip_special_tokens=False),
            # PI / scoring context: the format the rollout used + the hint applied (empty for base).
            "prompt_style": prompt_style,
            "history_length": history_length,
            "hint": hints.get(tid, ""),
        }
        if save_token_ids:
            # The offline PI scorer (pi/score_trajectories.py) needs the full no-hint sequence
            # (prompt_token_ids + response_ids) and the loss mask to isolate assistant tokens.
            record["response_ids"] = response_ids[i]
            record["loss_mask"] = loss_masks[i]
            if prompt_token_ids[i] is not None:
                record["prompt_token_ids"] = list(prompt_token_ids[i])
                record["prompt_decoded"] = tokenizer.decode(prompt_token_ids[i], skip_special_tokens=False)
        records.append(record)
    return records


def summarize(records: Sequence[Dict[str, Any]], wall_time: float, model_name: str) -> Dict[str, Any]:
    if not records:
        return {"num_trajectories": 0, "wall_time_sec": wall_time, "model_name": model_name}

    rewards = [float(record["reward"]) for record in records]
    successes = [float(record["env_metrics"].get("success", 0.0)) for record in records]
    partial = [float(record["env_metrics"].get("partial_success", 0.0)) for record in records]
    turns = [float(record["env_metrics"].get("num_turns", 0.0)) for record in records]
    invalid = [float(record["env_metrics"].get("num_invalid_actions", 0.0)) for record in records]
    stop_counts: Dict[str, int] = {}
    for record in records:
        reason = str(record.get("stop_reason"))
        stop_counts[reason] = stop_counts.get(reason, 0) + 1

    n = len(records)
    return {
        "model_name": model_name,
        "num_trajectories": n,
        "wall_time_sec": wall_time,
        "trajectories_per_sec": n / wall_time if wall_time > 0 else None,
        "avg_reward": sum(rewards) / n,
        "success_rate": sum(successes) / n,
        "avg_partial_success": sum(partial) / n,
        "avg_turns": sum(turns) / n,
        "avg_invalid_actions": sum(invalid) / n,
        "stop_reason_counts": stop_counts,
    }


async def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir or f"~/{args.run_name}/generation").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    launched_pid_file: Optional[Path] = None
    if args.launch_alfworld_servers:
        server_urls, launched_pid_file = launch_alfworld_servers(args, output_dir)
    else:
        server_urls = parse_server_urls(args.server_urls)
    if not server_urls:
        raise ValueError("Provide --server-urls or use --launch-alfworld-servers.")

    if not args.skip_server_health_check:
        logger.info("Waiting for %d ALFWorld env servers to become healthy", len(server_urls))
        await wait_for_servers(server_urls, args.health_timeout)

    items = load_items(args.data_file, args.limit, args.offset)
    if args.task_ids_file:
        _raw = Path(args.task_ids_file).expanduser().read_text().strip()
        try:
            _keep = set(json.loads(_raw))
        except Exception:
            _keep = {ln.strip() for ln in _raw.splitlines() if ln.strip()}
        _before = len(items)
        items = [it for it in items if item_task_id(it) in _keep]
        logger.info("task-ids filter: kept %d/%d rows (%d ids requested)", len(items), _before, len(_keep))
        if not items:
            logger.warning("task-ids filter matched NO rows — nothing to generate")
    expanded = expand_batch(items, args.n_samples)
    # --resume: reuse rollouts already on disk; regenerate only the missing (uid, repetition_id) pairs.
    traj_path = output_dir / "trajectories.jsonl"
    write_mode = "w"
    existing_records: List[Dict[str, Any]] = []
    if args.resume and traj_path.exists():
        done = set()
        with traj_path.open() as _rf:
            for _line in _rf:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _rr = json.loads(_line)
                except Exception:
                    continue
                _k = (_rr.get("uid"), _rr.get("repetition_id"))
                if _k[0] is not None and _k[1] is not None:
                    done.add(_k)
                    existing_records.append(_rr)
        _before = len(expanded)
        expanded = [
            (it, tid) for (it, tid) in expanded if (it.get("uid"), getattr(tid, "repetition_id", None)) not in done
        ]
        write_mode = "a"
        logger.info("RESUME: %d already on disk; %d/%d rollouts remaining", len(done), len(expanded), _before)
    hints = load_task_hints(args.hints_path)
    if hints:
        covered = sum(1 for item in items if item_task_id(item) in hints)
        logger.info(
            "Loaded %d PI hints from %s (covers %d/%d rows) | prompt_style=%s | placement=%s",
            len(hints),
            args.hints_path,
            covered,
            len(items),
            args.prompt_style,
            args.hint_placement,
        )
    # What the ENV receives. For 'system' placement that is the teacher's full <Secret information> block,
    # so the rollout is conditioned exactly as the OPSD teacher scores. Records keep the RAW hint (`hints`).
    env_hints = hints
    if hints and args.hint_placement == "system":
        from opsd.pi.signals.inject import secret_system_content

        env_hints = {k: secret_system_content(v) for k, v in hints.items()}
    logger.info("Loaded %d rows, generating %d rollouts", len(items), len(expanded))

    tokenizer_name = infer_tokenizer_name(args.tokenizer, args.served_model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    api_key = args.api_key or (os.environ.get(args.api_key_env) if args.api_key_env else None)
    client = LocalHTTPInferenceClient(
        model_url=args.model_url,
        model_name=args.served_model_name,
        tokenizer=tokenizer,
        endpoint_mode=args.endpoint_mode,
        request_timeout=args.request_timeout,
        max_concurrency=args.model_max_concurrency,
        api_key=api_key,
        request_token_ids=args.request_token_ids,
    )
    generator, sampling_params = make_generator(args, server_urls, tokenizer, client)

    records: List[Dict[str, Any]] = []
    t0 = time.time()
    try:
        with traj_path.open(write_mode) as f:
            for start in range(0, len(expanded), args.batch_size):
                batch = expanded[start : start + args.batch_size]
                gen_input = make_generator_input(batch, sampling_params, env_hints)
                gen_output = await generator.generate(gen_input, disable_tqdm=args.disable_tqdm)
                batch_records = records_from_output(
                    batch,
                    gen_output,
                    tokenizer,
                    args.save_token_ids,
                    hints=hints,
                    prompt_style=args.prompt_style,
                    history_length=args.history_length,
                )
                for record in batch_records:
                    f.write(json.dumps(_jsonable(record)) + "\n")
                f.flush()
                records.extend(batch_records)
                logger.info("Finished %d/%d rollouts", len(records), len(expanded))
    finally:
        await client.close()
        if launched_pid_file is not None:
            kill_launched_servers(launched_pid_file)

    wall = time.time() - t0
    summary = summarize(existing_records + records, wall, args.served_model_name)
    summary.update(
        {
            "data_file": str(Path(args.data_file).expanduser()),
            "server_urls": server_urls,
            "model_url": args.model_url,
            "endpoint_mode": args.endpoint_mode,
            "output_dir": str(output_dir),
            "prompt_style": args.prompt_style,
            "history_length": args.history_length,
            "hints_path": args.hints_path,
            "num_hints": len(hints),
        }
    )
    with (output_dir / "summary.json").open("w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    logger.info("DONE -> %s\n%s", output_dir, json.dumps(_jsonable(summary), indent=2))


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

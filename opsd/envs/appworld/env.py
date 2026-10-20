"""AppWorld environment for SkyRL (Seam A: native skyrl_gym BaseTextEnv).

AppWorld (https://github.com/StonyBrookNLP/appworld) is an interactive coding-agent
benchmark: the model writes Python that calls simulated app APIs in a stateful sandbox,
terminates by calling ``apis.supervisor.complete_task()``, and is scored by AppWorld's own
evaluator. This env maps that loop onto SkyRL's ``BaseTextEnv`` so the stock
``SkyRLGymGenerator`` handles the multi-turn loop, tokenization, and per-turn loss masking.

IMPORTANT — why this talks raw HTTP and does NOT import ``appworld``:
AppWorld 0.1.3 is a **pydantic v1** package; the SkyRL training env runs **vLLM + pydantic v2**.
The two cannot coexist in one venv (installing appworld downgrades pydantic and breaks vLLM).
So AppWorld runs as a separate pool of ``appworld serve environment`` HTTP servers (their own
venv), and this env is a thin HTTP client (like ``SearchEnv`` -> retriever). The training env
needs NO appworld install. Server routes used (appworld/serve/environment.py):
  POST /initialize {task_id, experiment_name, max_interactions, timeout_seconds, raise_on_failure}
       -> {"output": {"instruction": ..., ...}}
  POST /execute {task_id, code}            -> {"output": <stdout str>}
  POST /task_completed {task_id}           -> {"output": <bool>}
  POST /evaluate {task_id, suppress_errors}-> {"output": <TestTracker.to_dict()>}  (success, pass_count, ...)
  POST /close {task_id}
Each server holds ONE world at a time, so each rollout borrows an exclusive server from the
``AppWorldServerPool`` (acquire in init, release in close; lease/reaper for crash safety since
the generator calls env.close() without a try/finally).
"""

import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import requests
from omegaconf import DictConfig

# AppWorldEnvConfig lives in the import-light config.py (the config tree imports it); re-exported here.
from opsd.envs.appworld.config import AppWorldEnvConfig
from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)

logger = logging.getLogger(__name__)


# ---- parsing / formatting ----
# All action parsing, <think>-stripping and observation rendering now live in scaffold.py, which
# implements the OFFICIAL and LOOP contracts behind one ScaffoldSpec. Keeping a second copy here
# would let the training env silently drift from the evaluation drivers.


def _extract_code(text: str, strict: bool = False, spec=None):
    """Return the Python code to execute from an assistant turn (delegates to the shared scaffold).

    `spec` selects the OFFICIAL / LOOP / union / legacy parsing contract; when omitted we fall back to
    the pre-integration "legacy" behaviour so existing callers are unaffected. Returns None when
    `strict` and no fence was found (caller rejects the action), "" for an empty action.
    """
    from opsd.envs.appworld.scaffold import SCAFFOLDS, ScaffoldSpec, extract_code

    if spec is None:
        spec = SCAFFOLDS["legacy"]
    if strict and not spec.strict_parse:
        spec = ScaffoldSpec(**{**spec.__dict__, "strict_parse": True})
    return extract_code(text, spec)


def _format_output(output: Any, max_chars: int, spec=None) -> str:
    """Render the REPL result as the next user turn (delegates to the shared scaffold)."""
    from opsd.envs.appworld.scaffold import SCAFFOLDS, format_observation

    return format_observation(output, spec or SCAFFOLDS["legacy"], max_chars)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Read a config value whether cfg is a dict, an OmegaConf DictConfig, or a dataclass."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ---- exclusive-server pool (class-level pattern from tools/search.py:142) ----
class AppWorldServerPool:
    _instances: Dict[Tuple[str, ...], "AppWorldServerPool"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get(cls, server_urls: Tuple[str, ...], lease_ttl: float, recycle_after: int = 0) -> "AppWorldServerPool":
        key = tuple(sorted(server_urls))
        with cls._instances_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(key, lease_ttl, recycle_after)
            return cls._instances[key]

    def __init__(self, server_urls: Tuple[str, ...], lease_ttl: float, recycle_after: int = 0):
        self._free: "queue.Queue[str]" = queue.Queue()
        for u in server_urls:
            self._free.put(u)
        self._leases: Dict[str, Tuple[str, float]] = {}
        self._lock = threading.Lock()
        self._unhealthy: set = set()
        # Episodes served per URL, and the recycle threshold. /close_all clears the two class-level
        # registries we know leak, but a process restart is the only cleanup that is unconditional --
        # it returns RSS to baseline whatever else the server retained. 0 disables recycling entirely,
        # which is the default so no existing run changes behaviour.
        self._served: Dict[str, int] = {}
        self.recycle_after = max(0, int(recycle_after))
        self.lease_ttl = max(float(lease_ttl), 60.0)
        # shared HTTP session w/ connection pooling, like SearchToolGroup._session_pool
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=256, pool_maxsize=256, max_retries=0)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        threading.Thread(target=self._reap_loop, name="appworld-pool-reaper", daemon=True).start()

    def acquire(self, block: bool = True, timeout: Optional[float] = None) -> Tuple[str, str]:
        url = self._free.get(block=block, timeout=timeout)
        token = uuid4().hex
        with self._lock:
            self._leases[url] = (token, time.monotonic())
        return url, token

    def release(self, url: Optional[str], token: Optional[str]) -> None:
        if url is None:
            return
        with self._lock:
            cur = self._leases.get(url)
            if cur is None or cur[0] != token:
                return  # already reaped / not our lease -> avoid double-free
            del self._leases[url]
            unhealthy = url in self._unhealthy
        if not unhealthy:
            self._free.put(url)

    def mark_unhealthy(self, url: Optional[str], token: Optional[str]) -> None:
        if url is None:
            return
        with self._lock:
            cur = self._leases.get(url)
            if cur is not None and cur[0] == token:
                del self._leases[url]
            self._unhealthy.add(url)

    def retire_or_release(self, url: Optional[str], token: Optional[str]) -> None:
        """Release the lease, recycling the server process every `recycle_after` episodes.

        Identical to release() when recycling is off. When it fires, the server is killed and we wait
        for the shell-side supervisor to bring a fresh one back on the same port before returning the
        URL to rotation -- returning it early would hand the next rollout a dead endpoint.
        """
        if url is None:
            return
        with self._lock:
            cur = self._leases.get(url)
            if cur is None or cur[0] != token:
                return  # already reaped / not our lease -> avoid double-free
            del self._leases[url]
            unhealthy = url in self._unhealthy
            self._served[url] = self._served.get(url, 0) + 1
            n = self._served[url]
        if unhealthy:
            return
        if self.recycle_after and n % self.recycle_after == 0:
            if self._recycle(url):
                self._free.put(url)
            else:
                # Do NOT return a server we could not verify: a dead endpoint in rotation turns one
                # leak into every subsequent rollout on that port failing /initialize.
                with self._lock:
                    self._unhealthy.add(url)
                logger.warning("AppWorldServerPool: %s did not come back after recycle; retired", url)
            return
        self._free.put(url)

    def _recycle(self, url: str) -> bool:
        """SIGTERM the `appworld serve environment --port <port>` process, wait for a healthy replacement.

        Finds the pid by scanning /proc rather than tracking it: the servers are started by the run
        script as bare background processes and the pool never sees their pids. This only works when the
        env code runs on the SAME node as the server pool (same /proc namespace); the AppWorld
        entrypoints guarantee that by pinning the training loop to the driver's node. Returns False on
        any failure -- the caller retires the URL rather than risk handing out a dead endpoint.
        """
        import os
        import signal

        try:
            port = url.rsplit(":", 1)[1].split("/")[0]
        except Exception:  # noqa: BLE001
            return False
        want = ("serve", "environment", "--port")
        killed = []
        try:
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    with open(f"/proc/{pid}/cmdline", "rb") as fh:
                        parts = fh.read().split(b"\0")
                except (OSError, IOError):
                    continue
                argv = [p.decode("utf-8", "replace") for p in parts if p]
                if not argv or not all(w in argv for w in want):
                    continue
                # exact port match: "--port 9001" must not match 9001x or 19001
                if "--port" in argv:
                    i = argv.index("--port")
                    if i + 1 >= len(argv) or argv[i + 1] != port:
                        continue
                else:
                    continue
                os.kill(int(pid), signal.SIGTERM)
                killed.append(int(pid))
        except Exception as e:  # noqa: BLE001
            logger.warning("AppWorldServerPool: recycle scan failed for %s: %s", url, e)
            return False
        if not killed:
            logger.warning("AppWorldServerPool: no server process found on %s to recycle", url)
            return False
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            time.sleep(2.0)
            try:
                if self.session.get(url + "/", timeout=5).status_code == 200:
                    logger.info("AppWorldServerPool: recycled %s (killed %s)", url, killed)
                    return True
            except Exception:  # noqa: BLE001 -- not up yet
                pass
        return False

    def _reap_loop(self) -> None:
        while True:
            time.sleep(self.lease_ttl / 4.0)
            now = time.monotonic()
            reclaimed: List[str] = []
            with self._lock:
                stale = [u for u, (_, t) in self._leases.items() if now - t > self.lease_ttl]
                for u in stale:
                    del self._leases[u]
                    if u not in self._unhealthy:
                        reclaimed.append(u)
            for u in reclaimed:
                logger.warning("AppWorldServerPool: reclaimed stale lease for %s", u)
                self._free.put(u)


class AppWorldEnv(BaseTextEnv):
    """Multi-turn AppWorld CodeAct env; thin HTTP client over an `appworld serve` server pool."""

    def __init__(self, env_config: Union[AppWorldEnvConfig, DictConfig], extras: Dict[str, Any] = {}):
        super().__init__()
        cfg = env_config
        self.server_urls: List[str] = list(_cfg_get(cfg, "server_urls", []) or [])
        self.experiment_name_prefix: str = _cfg_get(cfg, "experiment_name_prefix", "skyrl_appworld")
        self.max_interactions: int = int(_cfg_get(cfg, "max_interactions", 50))
        self.execute_timeout: int = int(_cfg_get(cfg, "execute_timeout", 120))
        self.reward_mode: str = _cfg_get(cfg, "reward_mode", "binary")
        self.repl_output_max_chars: int = int(_cfg_get(cfg, "repl_output_max_chars", 20000))
        self.raise_on_failure: bool = bool(_cfg_get(cfg, "raise_on_failure", False))
        self.strict_parse: bool = bool(_cfg_get(cfg, "strict_parse", False))
        # Resolve the ReAct scaffold once (prompt-side variant is chosen in appworld_dataset.py).
        from opsd.envs.appworld.scaffold import SCAFFOLDS, ScaffoldSpec

        self.scaffold_name: str = str(_cfg_get(cfg, "scaffold", "official"))
        base = SCAFFOLDS.get(self.scaffold_name)
        if base is None:
            raise ValueError(f"unknown scaffold {self.scaffold_name!r}; choose from {sorted(SCAFFOLDS)}")
        self.scaffold_spec: ScaffoldSpec = ScaffoldSpec(**{**base.__dict__, "strict_parse": self.strict_parse})
        self.reminder_every: int = int(_cfg_get(cfg, "reminder_every", 0))
        # Recycle each server process after this many episodes; 0 = never (default, so existing runs are
        # unchanged). Requires the run script's supervisor loop to restart the port -- without it a
        # recycled server never returns and the pool retires it.
        self.server_recycle_after: int = int(_cfg_get(cfg, "server_recycle_after", 0))
        ht = int(_cfg_get(cfg, "http_timeout", 0) or 0)
        self.http_timeout: int = ht if ht > 0 else self.execute_timeout + 60

        extras = extras or {}
        reward_spec = extras.get("reward_spec", {}) or {}
        extra_info = extras.get("extra_info", {}) or {}
        self.task_id = reward_spec.get("ground_truth") or extra_info.get("task_id")
        assert (
            self.task_id is not None
        ), "AppWorldEnv requires the AppWorld task_id in reward_spec.ground_truth (or extra_info.task_id)."
        self.max_turns = extras.get("max_turns", 30)
        self._lease_ttl = max(self.max_turns * self.execute_timeout + 60, 300)

        self._pool: Optional[AppWorldServerPool] = None
        self._server_url: Optional[str] = None
        self._lease_token: Optional[str] = None
        self._initialized = False
        self._success = False
        self._partial = 0.0
        self._reward = 0.0
        self._completed = False
        self._eval_unavailable = False
        self._parse_errors = 0

    # ---- HTTP helpers ----
    def _post(self, path: str, payload: dict) -> dict:
        resp = self._pool.session.post(f"{self._server_url}{path}", json=payload, timeout=self.http_timeout)
        resp.raise_for_status()
        return resp.json()

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        if not self.server_urls:
            raise RuntimeError(
                "AppWorldEnv requires environment.skyrl_gym.appworld.server_urls "
                "(a pool of `appworld serve environment` URLs)."
            )
        self._pool = AppWorldServerPool.get(tuple(self.server_urls), self._lease_ttl, self.server_recycle_after)
        # Retry /initialize across servers: a single `appworld serve` process dying mid-run (its REPL
        # gets OOM-killed executing agent code) must NOT crash the whole training run. On a connection
        # failure we quarantine that server (mark_unhealthy) and re-acquire a DIFFERENT one, up to a
        # bounded number of attempts. Only if we can't get any healthy server do we raise (a genuinely
        # dead pool). This is the crux of surviving long runs where servers occasionally die.
        max_attempts = min(max(len(self.server_urls), 1), 8)
        last_exc: Optional[Exception] = None
        for _attempt in range(max_attempts):
            try:
                self._server_url, self._lease_token = self._pool.acquire(block=True, timeout=self._lease_ttl)
            except Exception as e:  # pool exhausted / no free server within timeout
                last_exc = e
                break
            try:
                out = self._post(
                    "/initialize",
                    {
                        "task_id": self.task_id,
                        "experiment_name": f"{self.experiment_name_prefix}_{self.task_id}_{uuid4().hex}",
                        "max_interactions": self.max_interactions,
                        "timeout_seconds": self.execute_timeout,
                        "raise_on_failure": self.raise_on_failure,
                    },
                )["output"]
                self._initialized = True
                # The task instruction is already in the dataset prompt (system + user turn);
                # /initialize provisions the world so step() can execute. Return the prompt unchanged.
                _ = out  # /initialize also returns the instruction, but the dataset is the source of truth
                return list(prompt), {"task_id": self.task_id}
            except Exception as e:
                # server unhealthy (dead / connection refused / timeout) -> quarantine and try another
                last_exc = e
                self._pool.mark_unhealthy(self._server_url, self._lease_token)
                self._server_url, self._lease_token = None, None
        raise RuntimeError(
            f"AppWorldEnv.init: no healthy appworld server after {max_attempts} attempts " f"(task {self.task_id})"
        ) from last_exc

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        code = _extract_code(action, strict=self.strict_parse, spec=self.scaffold_spec)
        done = self.turns >= self.max_turns

        if code is None:
            # strict_parse: no ```python``` fence -> reject the action (no execution). The turn is
            # consumed; under a binary/sparse reward this implicitly penalizes malformed output.
            self._parse_errors += 1
            if done:
                self._finalize_reward()
                return BaseTextEnvStepOutput(
                    observations=[], reward=self._reward, done=True, metadata={"parse_error": True}
                )
            obs = (
                "No ```python code block found. Emit your action as a SINGLE fenced Python block:\n"
                "```python\n<your code>\n```"
            )
            return BaseTextEnvStepOutput(
                observations=[{"role": "user", "content": obs}], reward=0.0, done=False, metadata={"parse_error": True}
            )

        if not code:
            # Empty response — rare. Nudge (or finalize if out of turns). Note: prose/invalid code is
            # NOT special-cased — it's sent to /execute and its error comes back as the observation,
            # exactly like AppWorld's own harness.
            if done:
                self._finalize_reward()
                return BaseTextEnvStepOutput(
                    observations=[], reward=self._reward, done=True, metadata={"empty_action": True}
                )
            obs = (
                "Your last message was empty. Emit your action as a fenced Python block that calls "
                "`apis.*`:\n```python\n<your code>\n```"
            )
            return BaseTextEnvStepOutput(
                observations=[{"role": "user", "content": obs}], reward=0.0, done=False, metadata={"empty_action": True}
            )

        # Never raise: step() must return so the generator reaches close() and releases the server.
        try:
            output = self._post("/execute", {"task_id": self.task_id, "code": code})["output"]
        except Exception as e:
            output = f"Execution error: {e}"
        try:
            self._completed = bool(self._post("/task_completed", {"task_id": self.task_id})["output"])
        except Exception:
            self._completed = False

        done = done or self._completed
        if done:
            self._finalize_reward()
            return BaseTextEnvStepOutput(
                observations=[], reward=self._reward, done=True, metadata={"completed": self._completed}
            )
        obs_text = _format_output(output, self.repl_output_max_chars, self.scaffold_spec)
        if self.reminder_every > 0 and self.turns % self.reminder_every == 0:
            # LOOP-style: re-assert the contract periodically so it stays salient once the
            # instructions have scrolled far up a long-horizon context.
            from opsd.envs.appworld.scaffold import REMINDER_MESSAGE

            obs_text = f"{obs_text}\n{REMINDER_MESSAGE}"
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": obs_text}], reward=0.0, done=False, metadata={}
        )

    def _finalize_reward(self) -> None:
        try:
            d = self._post("/evaluate", {"task_id": self.task_id, "suppress_errors": True})["output"]
            d = d if isinstance(d, dict) else {}
            self._success = bool(d.get("success", False))
            # AppWorld /evaluate returns {success, difficulty, num_tests, passes:[...], failures:[...]}.
            # TGC (Task Goal Completion, = paper's "Accuracy") is `success` (ALL tests pass) -> self._success.
            # `partial` below is the COMPLETION RATE = fraction of the task's tests passed (the dense
            # reward), NOT TGC. (Earlier code read nonexistent pass_count/pass_percentage keys → partial=0.)
            passes = d.get("passes") if isinstance(d.get("passes"), list) else []
            failures = d.get("failures") if isinstance(d.get("failures"), list) else []
            n_pass = len(passes)
            total = d.get("num_tests") or (n_pass + len(failures))
            self._partial = (n_pass / total) if total else (1.0 if self._success else 0.0)
        except Exception as e:
            logger.warning("AppWorldEnv /evaluate failed for task %s: %s", self.task_id, e)
            self._eval_unavailable = True
            self._success, self._partial = False, 0.0
        self._reward = (1.0 if self._success else 0.0) if self.reward_mode == "binary" else float(self._partial)

    def close(self) -> None:
        # /close_all, NOT /close. Both target the server's single module-level `world`, but /close leaves
        # CLASS-level state behind, and that is the host-memory leak that OOM'd a node at 1459/1536 GB:
        # one `appworld serve environment` process held 1146 GB while all 8 FSDP workers plus the ref
        # model together were 87 GB. AppWorld.close() clears the DB cache, unsets datetime and calls
        # apis.close(); it does NOT clear
        #   * `Requester.clients` -- a CLASS dict of FastAPI TestClients, one per app-set, each wrapping a
        #     whole app stack. Requester.close() pops it only `if self.client`, and the package's own
        #     comment there reads "The .__enter__() is necessary to avoid the memory leakage. TODO: post a
        #     github issue on fastapi/starlette with repro" -- upstream knows it leaks.
        #   * `AppWorld.id_to_time_freezer` -- one entry per world.
        # Both are cleared only by close_all() (environment.py:782 -> ApiCollection.close_all() +
        # id_to_time_freezer.clear(); requester.py:330 resets cls.clients wholesale).
        # SAFE ONLY BECAUSE LEASES ARE EXCLUSIVE: close_all() calls clear_local_dbs_cache() with no task
        # argument and stops every time freezer in the process, which would trample a concurrent lease.
        # One world per server at a time is what makes it acceptable -- do not carry this into a
        # shared-server configuration.
        try:
            if self._initialized and self._server_url is not None:
                self._post("/close_all", {"task_id": self.task_id})
        except Exception:
            # Fall back to /close: close_all raises when `world.task_id` no longer matches (reaped lease,
            # crashed REPL), and a failed cleanup must not also skip the release in `finally`.
            try:
                if self._initialized and self._server_url is not None:
                    self._post("/close", {"task_id": self.task_id})
            except Exception:
                pass
        finally:
            self._initialized = False
            if self._pool is not None and self._server_url is not None:
                self._pool.retire_or_release(self._server_url, self._lease_token)
                self._server_url, self._lease_token = None, None

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "success": float(self._success),
            "partial_completion": float(self._partial),
            "num_turns": float(self.turns),
            "completed_task_called": float(self._completed),
            "eval_unavailable": float(self._eval_unavailable),
            "parse_errors": float(self._parse_errors),
        }

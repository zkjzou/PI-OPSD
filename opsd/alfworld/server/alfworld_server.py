"""Standalone FastAPI server wrapping ONE text-only ALFWorld ``AlfredTWEnv``.

Runs in the SEPARATE alfworld venv (see ``launch_alfworld_servers.sh``); the SkyRL training env
never imports this. ``opsd/envs/alfworld/env.py`` is a thin HTTP client that borrows one
exclusive server per rollout. Each server holds ONE game at a time.

Routes:
  POST /reset {game_file?, seed?} -> {observation, admissible_commands, task}
  POST /step  {action}            -> {observation, reward, done, won,
                                       goal_condition_success_rate, admissible_commands}
  POST /close {}                  -> {ok: true}
  GET  /health                    -> {status, alfworld_data, num_games}

ALFWorld semantics ported from SDAR (agent_system/environments/env_package/alfworld/envs.py):
  - ``AlfredTWEnv`` (TextWorld + PDDL), batch_size=1; obs/infos are de-batched here ([0]).
  - reward = ``10.0 * won`` (text-only branch of compute_reward).
  - infos carry ``won``, ``admissible_commands`` and (best effort) ``goal_condition_success_rate``.

Why we re-register a single game per /reset: the SkyRL dataset pins each rollout to a specific
game (``reward_spec.ground_truth`` = abs path to that game's ``traj_data.json``). TextWorld's gym
env cycles through whatever game files were registered, so to deterministically serve one game we
register a 1-element game list per reset. At startup we instead register the FULL split (cheap,
no per-game env make) to validate ``$ALFWORLD_DATA`` + PDDL grammar parse and fail loudly.

Run:
  python alfworld_server.py --port 8100 [--config /path/to/config_tw.yaml] [--split train]
"""

import argparse
import copy
import logging
import os
import shutil
import threading
import time
from typing import Any, Dict, List, Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("alfworld_server")


# ---- request models ----
class ResetRequest(BaseModel):
    game_file: Optional[str] = None  # abs path to traj_data.json OR to the game.tw-pddl
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: str


# ---- the single-world wrapper (built lazily / per reset) ----
class AlfworldServer:
    def __init__(self, config_path: Optional[str], env_type: str, max_steps: int):
        self.env_type = env_type
        self.max_steps = max_steps
        self._lock = threading.Lock()
        self._env = None  # current TextWorld gym BatchEnv (batch_size=1)
        self._base = None  # AlfredTWEnv builder (holds config + game_files)
        self._steps = 0
        self._last_admissible: List[str] = []
        self._won = False

        self.config = self._load_config(config_path)
        # Validate $ALFWORLD_DATA + PDDL grammar by building the base env once at startup.
        self._validate_and_build_base()

    # -- config --
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        if config_path is None:
            # default to the config_tw.yaml shipped next to this server.
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_tw.yaml")
        if not os.path.exists(config_path):
            raise RuntimeError(f"ALFWorld config not found: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        # Force text-only + no human-anno regen surprises.
        config["env"]["type"] = self.env_type
        return config

    def _alfworld_data(self) -> str:
        data = os.environ.get("ALFWORLD_DATA")
        if not data or not os.path.isdir(data):
            raise RuntimeError(
                f"$ALFWORLD_DATA must point at an existing alfworld data dir (got {data!r}). "
                "Run `alfworld-download` and `export ALFWORLD_DATA=...`."
            )
        return data

    def _validate_and_build_base(self) -> None:
        data = self._alfworld_data()
        # PDDL domain + grammar must exist & parse (these are the files that make every action
        # admissible/invalid; a bad grammar floors reward at the invalid-action penalty).
        domain = os.path.expandvars(self.config["logic"]["domain"])
        grammar = os.path.expandvars(self.config["logic"]["grammar"])
        for p in (domain, grammar):
            if not os.path.exists(p):
                raise RuntimeError(f"ALFWorld PDDL/grammar file missing: {p} (check $ALFWORLD_DATA={data}).")

        from alfworld.agents.environment import get_environment

        # reset() overwrites base.game_files with the parquet's ABSOLUTE game path on every rollout,
        # so the startup game LIST is never used — we only need a VALID AlfredTWEnv object. Building
        # over the full train split walks+parses ALL ~8810 games (tens of seconds; ×N servers over
        # a network filesystem = minutes of warmup, and num_train_games caps only AFTER the walk so it
        # doesn't help).
        # Point the startup build at a tiny cached "scan seed" (a few real train games) so it stays a
        # valid object without the full walk. Set ALFWORLD_STARTUP_SCAN_GAMES=0 to restore full scan.
        cfg = self.config
        scan_n = int(os.environ.get("ALFWORLD_STARTUP_SCAN_GAMES", "24") or 0)
        if scan_n > 0:
            seed_train = self._build_scan_seed(scan_n)
            if seed_train:
                cfg = copy.deepcopy(self.config)
                cfg["dataset"]["data_path"] = seed_train
        self._base = get_environment(self.env_type)(cfg, train_eval="train")
        self.num_games = len(getattr(self._base, "game_files", []))
        if self.num_games == 0:
            raise RuntimeError(
                f"AlfredTWEnv collected 0 games from $ALFWORLD_DATA={data}. "
                "Check json_2.1.1/train and that game.tw-pddl files exist (run alfworld-generate if needed)."
            )
        logger.info(
            "AlfredTWEnv startup OK: %d games scanned (data=%s, scan_cap=%s)", self.num_games, data, scan_n or "full"
        )

    def _build_scan_seed(self, n: int) -> Optional[str]:
        """Copy up to n real train games into a small cached dir so the startup AlfredTWEnv build
        walks ~n games instead of the full split. Built ONCE atomically (a lock elects one builder;
        peers wait for the marker), then reused by every server and future run. reset() is unaffected
        (it loads games by absolute path). Returns the seed's train dir, or None to fall back."""
        real_train = os.path.expandvars(self.config["dataset"]["data_path"])
        if not os.path.isdir(real_train):
            return None
        seed_root = os.environ.get("ALFWORLD_SCAN_SEED_DIR") or os.path.join(
            os.path.dirname(os.path.dirname(real_train.rstrip("/"))), f".alfworld_scan_seed_{n}"
        )
        seed_train = os.path.join(seed_root, "json_2.1.1", "train")
        marker = os.path.join(seed_root, ".done")
        if os.path.exists(marker):
            return seed_train
        os.makedirs(os.path.dirname(seed_root) or ".", exist_ok=True)
        lock = seed_root + ".lock"
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            builder = True
        except FileExistsError:
            builder = False
        if not builder:  # a peer is building — wait for the marker
            for _ in range(600):
                if os.path.exists(marker):
                    return seed_train
                time.sleep(1)
            return seed_train if os.path.exists(marker) else None
        try:
            tmp = seed_root + ".tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            dst_train = os.path.join(tmp, "json_2.1.1", "train")
            os.makedirs(dst_train, exist_ok=True)
            count = 0
            for root, _dirs, files in os.walk(real_train):  # early-stop after n solvable-looking games
                if "traj_data.json" in files and os.path.exists(os.path.join(root, "game.tw-pddl")):
                    g = os.path.join(dst_train, f"g{count}")
                    os.makedirs(g, exist_ok=True)
                    for fn in os.listdir(root):
                        src = os.path.join(root, fn)
                        if os.path.isfile(src):
                            shutil.copy2(src, os.path.join(g, fn))
                    count += 1
                    if count >= n:
                        break
            if count == 0:
                shutil.rmtree(tmp, ignore_errors=True)
                return None
            os.replace(tmp, seed_root)
            open(marker, "w").close()
            logger.info("built alfworld scan seed: %d games -> %s", count, seed_train)
            return seed_train
        finally:
            try:
                os.remove(lock)
            except OSError:
                pass

    @staticmethod
    def _resolve_game_dir(game_file: str) -> str:
        """Accept a traj_data.json path or a game dir; return the directory holding game.tw-pddl."""
        if os.path.isdir(game_file):
            return game_file
        return os.path.dirname(game_file)

    def reset(self, game_file: Optional[str], seed: Optional[int]) -> Dict[str, Any]:
        with self._lock:
            self._steps = 0
            self._won = False

            if game_file is not None:
                # Re-register a base env over ONLY this game's directory so reset() lands on it
                # deterministically (TextWorld cycles through registered game_files).
                game_dir = self._resolve_game_dir(game_file)
                pddl = os.path.join(game_dir, "game.tw-pddl")
                if not os.path.exists(pddl):
                    raise RuntimeError(f"game.tw-pddl not found for game_file={game_file} (looked in {game_dir}).")
                # Reuse the env built ONCE at startup (self._base) instead of reconstructing it.
                # Constructing AlfredTWEnv re-scans EVERY game in the split (~O(8810), tens of
                # seconds); doing that on every /reset means each rollout pays a full rescan, which
                # under a local 64-server pool + colocated vLLM/FSDP blows past the client's
                # http_timeout. reset() holds self._lock and the client always supplies game_file,
                # so mutating the shared _base's game_files here is safe.
                base = self._base
                base.game_files = [pddl]
                base.num_games = 1
            else:
                base = self._base

            self._env = base.init_env(batch_size=1)
            if seed is not None:
                self._env.seed(int(seed))

            obs, infos = self._env.reset()
            observation = obs[0] if isinstance(obs, (list, tuple)) else obs
            admissible = self._debatch(infos, "admissible_commands", [])
            self._last_admissible = list(admissible or [])

            task = ""
            marker = "Your task is to: "
            if marker in observation:
                task = observation.split(marker, 1)[1].strip()

            return {
                "observation": observation,
                "admissible_commands": self._last_admissible,
                "task": task,
            }

    def step(self, action: str) -> Dict[str, Any]:
        with self._lock:
            if self._env is None:
                raise RuntimeError("step() called before reset().")
            self._steps += 1
            obs, scores, dones, infos = self._env.step([action])
            observation = obs[0] if isinstance(obs, (list, tuple)) else obs
            done = bool(self._descalar(dones))
            won = bool(self._debatch(infos, "won", False))
            gcs = self._debatch(infos, "goal_condition_success_rate", None)
            admissible = self._debatch(infos, "admissible_commands", [])
            self._last_admissible = list(admissible or [])
            self._won = won

            reward = 10.0 * float(won)  # SDAR compute_reward text-only branch
            # TextWorld also terminates on its own max_episode_steps; the SkyRL env caps turns too.
            done = done or won or self._steps >= self.max_steps
            return {
                "observation": observation,
                "reward": reward,
                "done": done,
                "won": won,
                "goal_condition_success_rate": (float(gcs) if gcs is not None else None),
                "admissible_commands": self._last_admissible,
            }

    def close(self) -> Dict[str, Any]:
        with self._lock:
            env = self._env
            self._env = None
        if env is not None:
            try:
                env.close()
            except Exception as e:  # noqa: BLE001
                logger.warning("env.close() failed: %s", e)
        return {"ok": True}

    # -- de-batching helpers (batch_size=1; infos may be dict-of-lists or list-of-dicts) --
    @staticmethod
    def _descalar(x: Any) -> Any:
        if isinstance(x, (list, tuple)):
            return x[0]
        return x

    @staticmethod
    def _deb(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        return value

    def _debatch(self, infos: Any, key: str, default: Any) -> Any:
        """Extract infos[key] for batch index 0.

        AlfredTWEnv infos may be a dict-of-lists ({'won': [True], ...}) or a list-of-dicts
        ([{'won': True, ...}]); handle both and de-batch to a scalar.
        """
        if isinstance(infos, dict):
            if key not in infos:
                return default
            return self._deb(infos[key])
        if isinstance(infos, (list, tuple)) and infos:
            info0 = infos[0]
            if isinstance(info0, dict):
                return self._deb(info0.get(key, default))
        return default


def build_app(server: AlfworldServer) -> FastAPI:
    app = FastAPI(title="alfworld-server")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "alfworld_data": os.environ.get("ALFWORLD_DATA"),
            "env_type": server.env_type,
            "num_games": server.num_games,
        }

    @app.post("/reset")
    def reset(req: ResetRequest) -> Dict[str, Any]:
        try:
            return server.reset(req.game_file, req.seed)
        except Exception as e:  # noqa: BLE001
            logger.exception("reset failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/step")
    def step(req: StepRequest) -> Dict[str, Any]:
        try:
            return server.step(req.action)
        except Exception as e:  # noqa: BLE001
            logger.exception("step failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/close")
    def close() -> Dict[str, Any]:
        return server.close()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone ALFWorld (AlfredTWEnv) FastAPI server.")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--config", default=None, help="Path to config_tw.yaml (default: alongside this file).")
    parser.add_argument("--env-type", default="AlfredTWEnv", help="AlfredTWEnv only (text). THOR is out of scope.")
    parser.add_argument(
        "--max-steps", type=int, default=50, help="Server-side episode cap (TextWorld max_episode_steps)."
    )
    # 600 s, not uvicorn's 5 s: must exceed the client's idle gap between turns (a full LLM generation).
    # See the note at uvicorn.run() below -- at the default this race killed three multi-hour runs.
    parser.add_argument(
        "--timeout-keep-alive",
        type=int,
        default=int(os.environ.get("ALFWORLD_SERVER_KEEPALIVE", "600")),
        help="uvicorn keep-alive timeout (s). MUST outlast a generation step; default 600.",
    )
    args = parser.parse_args()

    # Fail loudly at boot if data/PDDL are missing (the constructor validates).
    server = AlfworldServer(config_path=args.config, env_type=args.env_type, max_steps=args.max_steps)
    app = build_app(server)
    logger.info(
        "Serving ALFWorld on %s:%d (env_type=%s, %d games)", args.host, args.port, args.env_type, server.num_games
    )
    # KEEP-ALIVE MUST OUTLAST A GENERATION STEP, or the client's pooled socket goes stale and the RUN DIES.
    # uvicorn defaults timeout_keep_alive to 5 s. The client (server_pool.py:62) is a shared, long-lived
    # requests.Session with pool_maxsize=256 and max_retries=0, and a rollout spends FAR more than 5 s in
    # LLM generation between turns. So uvicorn reaps the idle connection, the client's pool still holds the
    # dead socket, and the next POST on it gets
    #   RemoteDisconnected('Remote end closed connection without response')
    # which propagates out of agent_loop -> tqdm.gather -> trainer.train() and kills the whole job.
    # It always lands on /reset -- the FIRST call of a new trajectory, i.e. the longest idle gap -- and
    # leaves NO server-side error, because closing an idle keep-alive connection is correct HTTP. Three
    # runs died this way (opsd_glt_xdmix step 9, dapo rs01 step 101, dapo rs01_r100b step 150); the step
    # number is random because it only fires when the particular pooled socket happened to be reaped.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        timeout_keep_alive=args.timeout_keep_alive,
    )


if __name__ == "__main__":
    main()

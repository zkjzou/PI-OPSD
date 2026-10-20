"""Exclusive HTTP-server pool for single-world agent environments.

Generalized from ``opsd.envs.appworld.env.AppWorldServerPool``. Several agent
simulators (AppWorld, WebShop, ALFWorld) are single-tenant per process — each running
server holds exactly one world at a time — and/or too heavy / dependency-conflicting to
import into the SkyRL training (vLLM) venv. The established pattern is to run a pool of
``<env> serve`` HTTP servers in their own venv and make the SkyRL ``BaseTextEnv`` a thin
``requests`` client that borrows ONE exclusive server per rollout.

Why a lease + daemon reaper: ``SkyRLGymGenerator`` calls ``env.close()`` without a
try/finally guarantee, so a rollout that dies mid-trajectory could leak a server forever.
Each acquire stamps a monotonic lease; a background reaper reclaims leases older than
``lease_ttl`` so a crashed rollout cannot permanently starve the pool.

Usage (in a BaseTextEnv subclass):

    self._pool = ExclusiveServerPool.get(tuple(self.server_urls), self._lease_ttl)
    self._url, self._token = self._pool.acquire(block=True)          # in init()
    resp = self._pool.session.post(f"{self._url}/step", json=...)    # shared pooled session
    self._pool.release(self._url, self._token)                       # in close()
"""

import logging
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import requests

logger = logging.getLogger(__name__)


class ExclusiveServerPool:
    """Hands each rollout an exclusive server URL from a fixed pool, with crash-safe leasing.

    Instances are keyed by the (sorted) tuple of server URLs so every env of the same class
    sharing the same pool config gets the same singleton pool (one process-wide free queue).
    """

    _instances: Dict[Tuple[str, ...], "ExclusiveServerPool"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get(cls, server_urls: Tuple[str, ...], lease_ttl: float) -> "ExclusiveServerPool":
        key = tuple(sorted(server_urls))
        with cls._instances_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(key, lease_ttl)
            return cls._instances[key]

    def __init__(self, server_urls: Tuple[str, ...], lease_ttl: float):
        self._free: "queue.Queue[str]" = queue.Queue()
        for u in server_urls:
            self._free.put(u)
        self._leases: Dict[str, Tuple[str, float]] = {}
        self._lock = threading.Lock()
        self._unhealthy: set = set()
        self.lease_ttl = max(float(lease_ttl), 60.0)
        # Shared HTTP session with generous connection pooling (mirrors tools/search.py).
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=256, pool_maxsize=256, max_retries=0)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        threading.Thread(target=self._reap_loop, name="server-pool-reaper", daemon=True).start()

    def acquire(self, block: bool = True, timeout: Optional[float] = None) -> Tuple[str, str]:
        """Borrow an exclusive server URL. Returns (url, lease_token). Blocks until one is free."""
        url = self._free.get(block=block, timeout=timeout)
        token = uuid4().hex
        with self._lock:
            self._leases[url] = (token, time.monotonic())
        return url, token

    def release(self, url: Optional[str], token: Optional[str]) -> None:
        """Return a server to the pool. No-op if the lease was already reaped or isn't ours."""
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
        """Quarantine a server (e.g. it failed /reset) so it is not handed out again."""
        if url is None:
            return
        with self._lock:
            cur = self._leases.get(url)
            if cur is not None and cur[0] == token:
                del self._leases[url]
            self._unhealthy.add(url)

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
                logger.warning("ExclusiveServerPool: reclaimed stale lease for %s", u)
                self._free.put(u)

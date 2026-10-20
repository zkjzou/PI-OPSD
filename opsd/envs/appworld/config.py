"""Config dataclass for the AppWorld env (``appworld serve environment`` HTTP-service pattern).

Attached to ``environment.skyrl_gym.appworld`` by the OPSD config tree (``opsd/skyrl_ext/config.py``,
a subclass of SkyRL's ``SkyRLGymConfig``). Split out of ``env.py`` (which re-exports it) so that
importing the config tree does not pull in ``requests``, the server pool or ``skyrl_gym``.

Keep this module import-light (dataclasses/typing only): the config tree is unpickled in every Ray
worker. No ``from __future__ import annotations`` either: SkyRL's ``build_nested_dataclass`` needs real
class annotations.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AppWorldEnvConfig:
    # Pool of `appworld serve environment` base URLs (one exclusive world per server at a time).
    server_urls: List[str] = field(default_factory=list)
    experiment_name_prefix: str = "skyrl_appworld"
    max_interactions: int = 50  # AppWorld's own cap; keep >= generator.max_turns
    execute_timeout: int = 120  # server-side execution timeout (seconds)
    reward_mode: str = "binary"  # "binary" (success) or "partial" (TGC = pass fraction)
    repl_output_max_chars: int = 20000  # head/tail truncation per execute() observation
    raise_on_failure: bool = False
    http_timeout: int = 0  # client HTTP timeout; 0 -> execute_timeout + 60
    strict_parse: bool = False  # True: only execute a ```python``` (or bare ```) fenced block;
    # a fence-less action is rejected (no execution, wasted turn + nudge)
    # --- ReAct scaffold (see scaffold.py) -------------------------------------------------
    # "official" (default; the variant that measured TGC 21.05 on 4B-Thinking / 61.9 on GPT-4o),
    # "loop" (LOOP-style parsing: accepts ```py, concatenates blocks), "union" (most permissive),
    # "legacy" (pre-integration SkyRL behaviour, kept so old runs reproduce).
    scaffold: str = "official"
    reminder_every: int = 0  # >0: re-inject the key-instruction reminder every N turns (LOOP-style)
    # >0: kill + wait for a fresh `appworld serve environment` on the same port every N episodes.
    # The AppWorld server leaks host memory across worlds (see AppWorldEnv.close); /close_all clears the
    # two class-level registries we identified, a process restart is the unconditional cleanup. Needs the
    # run script's supervisor loop, which is what restarts the port. 0 = off.
    server_recycle_after: int = 0

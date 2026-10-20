"""Shared constants for the tau2-bench (τ²) integration.

τ² differs from τ1 in ways that matter here:
  * **Dual control** — in ``telecom`` the *user* also holds tools and mutates env state.
    ``retail``/``airline`` have no user tools at all and ``env.get_user_tools()`` *raises*.
  * **Task ids are opaque strings.** airline/retail use ``"0".."113"``; telecom ids encode the
    scenario, e.g. ``"[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:Hard]"`` — ~200
    chars containing ``[ ] : |``.  The ``:`` is why we never use a raw task id in a SkyRL ``uid``
    (τ1 appends ``f"{uid}:repeat-{n}"`` for GRPO grouping).
  * The upstream package uses a ``src/`` layout, so the import root is ``<checkout>/src``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

TAU2_REPOSITORY = "https://github.com/sierra-research/tau2-bench.git"
# Upstream main; byte-identical to release tag v1.0.1 (fc0055dc…) for the modules we import.
TAU2_REVISION = "1d244f5dca42944b67a379b44bfeb9f5748f189d"

# The pinned checkout is cloned here by ``prepare_data prepare`` (``source.ensure_checkout``).
DEFAULT_SOURCE_DIR = f"~/.cache/skyrl/tau2-bench/{TAU2_REVISION}"
DEFAULT_POLICY_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
# Remote frontier user simulator (no GPUs, cost-metered), reached through any OpenAI-compatible
# endpoint. Hold the BARE model name: ``UserSpec.litellm_model`` prepends ``openai/`` at call time so
# litellm uses its OpenAI-compatible provider against ``base_url`` instead of guessing a provider from
# the name.  Do NOT write ``openai/gpt-5.6-luna`` here: that prefix short-circuits ``litellm_model``,
# which then forwards a bare ``gpt-5.6-luna``.
# Verified at the shipped simulator config (temperature 0.0 per τ²-official, reasoning_effort
# unpinned, max_tokens 2048): 9-22 reasoning tokens per user turn, non-empty ``content``,
# finish_reason=stop -- i.e. this reasoning-capable model does NOT hit the empty-content failure mode
# documented on ``UserSpec.max_tokens``.
DEFAULT_SIMULATOR_MODEL = "gpt-5.6-luna"
# Fallback for the simulator / NL-judge endpoint when ``$OPENAI_BASE_URL`` is unset.  Resolve the
# effective default with :func:`default_llm_base_url`, never by reading this constant directly.
DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
# Name of the environment variable holding the simulator / NL-judge API key (never the key itself).
DEFAULT_LLM_API_KEY_ENV = "OPENAI_API_KEY"


def default_llm_base_url() -> str:
    """The simulator / NL-judge base URL: ``$OPENAI_BASE_URL`` if set, else the public OpenAI API.

    Resolved at call time (config construction), not at import time, so the value the driver sees is
    the one that ends up in the run config and in the egress allowlist.
    """

    return (os.environ.get("OPENAI_BASE_URL") or "").strip() or DEFAULT_LLM_BASE_URL


DOMAINS = ("airline", "retail", "telecom")

# tau2/user/user_simulator_base.py conversation-ending tokens, and LLMSoloAgent.STOP_TOKEN.
USER_STOP_MARKERS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")
AGENT_STOP_MARKER = "###STOP###"

# tau2.config.DEFAULT_MAX_STEPS is 200; the Orchestrator kwarg default is only 100.
DEFAULT_MAX_STEPS = 200
DEFAULT_MAX_ERRORS = 10

# ``evaluate_simulation`` short-circuits to reward 0.0 unless termination is one of these.  A weak
# policy that runs out of steps therefore scores 0 regardless of DB state — always report the
# termination histogram next to the reward or a capability result reads as a broken integration.
REWARDABLE_TERMINATIONS = ("AGENT_STOP", "USER_STOP")


@dataclass(frozen=True)
class ManifestSpec:
    """Public metadata for one τ² split; task contents stay in the pinned checkout.

    ``count`` is an *expected* count that is checked and reported — unlike τ1 it is deliberately
    NOT an ordinal contract, because τ² task ids are strings and need not be dense integers.
    """

    name: str
    filename: str
    domain: str
    split_name: str  # tau2 split key: "base" | "train" | "test" | "small" | "full"
    role: str
    count: int


# ``base`` is the published evaluation set (50 + 114 + 114 = 278 tasks).
MANIFEST_SPECS = (
    ManifestSpec("airline-base", "airline_base.jsonl", "airline", "base", "eval", 50),
    ManifestSpec("retail-base", "retail_base.jsonl", "retail", "base", "eval", 114),
    ManifestSpec("telecom-base", "telecom_base.jsonl", "telecom", "base", "eval", 114),
    # Phase 2 (GRPO) train/test splits (τ²'s own registered splits).
    ManifestSpec("retail-train", "retail_train.jsonl", "retail", "train", "train", 74),
    ManifestSpec("telecom-train", "telecom_train.jsonl", "telecom", "train", "train", 74),
    ManifestSpec("airline-test", "airline_test.jsonl", "airline", "test", "ood-eval", 20),
    # AReaL SEA-generated RL train pool (NOT a τ² registered split — loaded from HF via areal.py).
    # 563 retail task specs, each with its own DB snapshot; scored DB∧COMMUNICATE (no NL-judge).
    ManifestSpec("retail-areal-rl", "retail_areal_rl.jsonl", "retail", "retail-areal-rl", "train", 563),
    # Same pool, airline half: 1,148 task specs over 4 JSON DB snapshots. Also judge-free (0 tasks
    # carry nl_assertions; reward_basis unset -> [DB, COMMUNICATE]). Verified disjoint from τ²'s own
    # 50 airline tasks (0 exact and 0 near-duplicate user_scenario matches), so airline-base stays a
    # clean held-out eval.
    ManifestSpec("airline-areal-rl", "airline_areal_rl.jsonl", "airline", "airline-areal-rl", "train", 1148),
    # AReaL telecom: 271 records, DUAL CONTROL (80% of gold actions are user-side). Judge-free --
    # nl_assertions is empty on all 271, so no NL judge fires. Unlike retail/airline these carry
    # their own initialization_actions (the fault injection) and a ticket; see areal.convert_record.
    # CONTAMINATION, MEASURED (content key = initialization_actions + user instructions, not ids):
    #   telecom/base 114 -> 0 overlap | train 74 -> 0 | test 40 -> 0 | full 2285 -> 257 | small 20 -> 20
    # So it is DISJOINT from every evaluation split: a 271-task sample from the same 2,285-task
    # `full` universe those splits are drawn from, with the eval tasks excluded (14 are not in
    # `full` at all). Same distribution, different instances -- normal training data.
    # ⚠️ EXCEPT telecom_small: ALL 20 of its tasks are inside this pool, so telecom_small is
    # UNUSABLE as an eval for anything trained here. Use telecom-base.
    ManifestSpec("telecom-areal-rl", "telecom_areal_rl.jsonl", "telecom", "telecom-areal-rl", "train", 271),
)
MANIFEST_BY_NAME = {spec.name: spec for spec in MANIFEST_SPECS}

# ``split_name`` values served by areal.py (HF) instead of τ²'s registry. load_tasks() branches on these.
AREAL_REPO = "inclusionAI/AReaL-tau2-data"
# Pinned dataset commit (HF repo sha; last modified 2026-03-02, before any manifest here was built).
AREAL_REVISION = "86971dc03da6e7c1a7933295e05b84aab8215386"
AREAL_RL_FILE = "tau2_rl_train.jsonl"
AREAL_SPLIT_KEYS = frozenset({"retail-areal-rl", "airline-areal-rl", "telecom-areal-rl"})

# Fields of the τ² ``Task`` model that reveal the hidden objective and must never reach a manifest
# row (and therefore never reach a prompt).  See tau2/data_model/tasks.py.
HIDDEN_TASK_KEYS = frozenset(
    {
        "actions",
        "annotations",
        "communicate_info",
        "description",
        "env_assertions",
        "evaluation_criteria",
        "initial_state",
        "initialization_actions",
        "initialization_data",
        "instructions",
        "message_history",
        "nl_assertions",
        "persona",
        "reward_basis",
        "ticket",
        "user_scenario",
    }
)

SAMPLING_PROFILES = {
    # Qwen3 thinking-mode recommended settings.
    "qwen-thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    # Higher-temperature profile used for RL rollouts.
    "paper": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0},
    # tau2's own DEFAULT_LLM_TEMPERATURE_AGENT — for reference-number comparison.
    "tau2-paper": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0},
    # Send NOTHING — let the endpoint's own default sampling apply (all None => PolicySpec omits them).
    # For a hosted thinking model (e.g. qwen3p7-plus) that means thinking stays ON and sampling is
    # stochastic-by-default.
    # top_k/min_p are None here too; they are remote-only-dropped anyway, so this is moot for a hosted
    # gateway and simply leaves a local vLLM at its own defaults.
    "default": {"temperature": None, "top_p": None, "top_k": None, "min_p": None},
}

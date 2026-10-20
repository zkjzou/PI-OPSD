"""Per-leg token/cost metering for τ² runs.

τ1 read cost straight off its own HTTP client (the ``x-litellm-response-cost`` header).  Here litellm
owns the socket, so we register a ``success_callback`` instead and attribute each completion to a leg
(``agent`` / ``user`` / ``judge``) by its ``call_name``.  Keeping the judge separate matters: it is an
*outbound frontier-model* cost incurred inside the reward function, and conflating it with the user
simulator hides it.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

# We tag each leg via a ``metadata={"call_name": ...}`` kwarg on its llm_args, because tau2's own
# ``call_name`` argument to generate() is consumed for logging and never reaches litellm.  The judge's
# real call_name upstream is "nl_assertions_eval" (verified against the pinned checkout).
_LEG_BY_CALL_NAME = {
    "agent_response": "agent",
    "user_response": "user",
    "nl_assertions": "judge",
    "nl_assertions_eval": "judge",
}

_lock = threading.Lock()
_totals: dict[str, dict[str, float]] = defaultdict(
    lambda: {"calls": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0, "cost_usd": 0.0}
)


def _leg_for(kwargs: dict[str, Any]) -> str:
    name = kwargs.get("call_name")
    if not name:
        params = kwargs.get("litellm_params") or {}
        meta = params.get("metadata") or {}
        name = meta.get("call_name")
    return _LEG_BY_CALL_NAME.get(str(name), "other")


def _record(kwargs: dict[str, Any], response: Any) -> None:
    leg = _leg_for(kwargs)
    usage = getattr(response, "usage", None)
    cost = 0.0
    hidden = getattr(response, "_hidden_params", None) or {}
    if isinstance(hidden, dict):
        cost = float(hidden.get("response_cost") or 0.0)
    if not cost:
        slo = kwargs.get("standard_logging_object") or {}
        if isinstance(slo, dict):
            cost = float(slo.get("response_cost") or 0.0)
    with _lock:
        bucket = _totals[leg]
        bucket["calls"] += 1
        bucket["prompt_tokens"] += float(getattr(usage, "prompt_tokens", 0) or 0)
        bucket["completion_tokens"] += float(getattr(usage, "completion_tokens", 0) or 0)
        bucket["cost_usd"] += cost


def install() -> None:
    """Register the cost callback (idempotent)."""

    import litellm

    def success_callback(kwargs, response, start_time, end_time):  # noqa: ANN001
        try:
            _record(kwargs, response)
        except Exception:  # noqa: BLE001 - metering must never break a run
            pass

    for existing in litellm.success_callback:
        if getattr(existing, "_tau2_metering", False):
            return
    success_callback._tau2_metering = True  # type: ignore[attr-defined]
    litellm.success_callback = [*litellm.success_callback, success_callback]


def snapshot() -> dict[str, dict[str, float]]:
    """Cumulative per-leg usage since process start."""

    with _lock:
        return {leg: dict(vals) for leg, vals in _totals.items()}


def flat_metrics(prefix_map: dict[str, str] | None = None) -> dict[str, float]:
    """Flatten to W&B-style keys, e.g. ``sim/api_cost_usd_total``, ``judge/calls_total``."""

    prefix_map = prefix_map or {"agent": "policy", "user": "sim", "judge": "judge", "other": "other"}
    out: dict[str, float] = {}
    for leg, vals in snapshot().items():
        prefix = prefix_map.get(leg, leg)
        out[f"{prefix}/api_cost_usd_total"] = vals["cost_usd"]
        out[f"{prefix}/calls_total"] = vals["calls"]
        out[f"{prefix}/prompt_tokens_total"] = vals["prompt_tokens"]
        out[f"{prefix}/completion_tokens_total"] = vals["completion_tokens"]
    return out

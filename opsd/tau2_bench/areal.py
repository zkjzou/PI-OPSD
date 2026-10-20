"""Load AReaL's SEA-generated τ²-bench RL tasks as native *upstream* τ² ``Task`` objects.

Dataset: ``inclusionAI/AReaL-tau2-data`` (HF).  The RL split (``tau2_rl_train.jsonl``) holds task specs
whose per-task environment DB lives under ``tau2_rl_database/``.  AReaL trained against a *fork* of
tau2-bench (``dhh1995/tau2-bench@dhh/async-and-custom-completion``), but its only reward-relevant
behaviour is the ``EvaluationCriteria.reward_basis`` **default of ``[DB, COMMUNICATE]``** — which our
pinned upstream shares verbatim ("Default basis is ``[DB, COMMUNICATE]``, matching the original
τ-bench").  So each record converts to a stock upstream ``Task`` with **no reward override**.

The conversion (verified against all 563 retail records):

* ``id`` — kept as-is (``"retail_1"`` … no ``:``, so GRPO's ``:repeat-N`` suffix is safe).
* ``description`` — AReaL's ``{"purpose": …}`` dict maps directly onto τ²'s ``Description`` model.
* ``user_scenario`` — kept as-is (τ²'s ``UserScenario`` accepts ``{"instructions": {…}}``).
* ``evaluation_criteria`` — a JSON *string* in the dataset → ``json.loads`` → its omitted
  ``reward_basis`` then defaults to ``[DB, COMMUNICATE]``.
* ``db_path`` — the per-task DB snapshot, injected via ``Task.initial_state.initialization_data``.
  The upstream orchestrator applies this to the agent env, and the evaluator applies the *same*
  ``initial_state`` when computing the DB target — so the per-task DB is honoured on both sides.

Requires HF access at load time (``huggingface_hub``); files are cached under ``HF_HOME``.  Every
download is pinned to ``constants.AREAL_REVISION``, so a later push to the dataset repo cannot change
the tasks behind a committed manifest.  Call ``bootstrap_tau2`` before any loader here (τ² must be
importable).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .constants import AREAL_REPO, AREAL_REVISION, AREAL_RL_FILE

_DOMAINS = ("retail", "airline", "telecom")
# retail + airline inject a per-task DB snapshot (RetailDB / FlightDB, JSON). telecom does NOT --
# it passes its own initial_state/ticket through instead; see convert_record for why injecting a DB
# there both crashes and silently deletes the fault injection. (The old comment here claimed DB.load
# cannot parse TOML -- that is FALSE, tau2.utils.io_utils handles .toml, and TelecomDB exists. The
# real reason telecom is different is the user-side DB and initialization_actions, not the format.)
_SUPPORTED = ("retail", "airline", "telecom")


# Training must NOT depend on a live HF download at episode time. We therefore prefer a *staged*
# copy on persistent storage (``stage_local``). If the HF cache lives on scratch storage that gets
# wiped, every GRPO episode fails with LocalEntryNotFoundError under HF_HUB_OFFLINE=1 -- observed, and
# the reason this staged copy exists. Only ~40 MB is needed for retail.
AREAL_LOCAL_DIR = os.environ.get("AREAL_TAU2_DATA_DIR", os.path.expanduser("~/.cache/skyrl/areal-tau2-data"))


def _download(rel_path: str) -> str:
    """Resolve a dataset file: staged local copy first, HF hub only as a fallback."""

    staged = Path(AREAL_LOCAL_DIR) / rel_path
    if staged.is_file():
        return str(staged)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(AREAL_REPO, rel_path, repo_type="dataset", revision=AREAL_REVISION)


def stage_local(dest_dir: str | None = None, *, domain: str = "retail") -> str:
    """Download the files needed for ``domain`` into a persistent local dir (idempotent).

    Run once (with network) so training can then run fully offline:
        python -c "from opsd.tau2_bench.areal import stage_local; print(stage_local())"
    """

    from huggingface_hub import hf_hub_download

    dest = Path(dest_dir or AREAL_LOCAL_DIR)
    wanted = [AREAL_RL_FILE] + sorted({str(r["db_path"]) for r in _rl_records() if _domain_of(r) == domain})
    for rel in wanted:
        target = dest / rel
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        src = hf_hub_download(AREAL_REPO, rel, repo_type="dataset", revision=AREAL_REVISION)
        target.write_bytes(Path(src).read_bytes())
    return str(dest)


@lru_cache(maxsize=1)
def _rl_records() -> list[dict[str, Any]]:
    with open(_download(AREAL_RL_FILE), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _domain_of(record: dict[str, Any]) -> str:
    db_path = str(record.get("db_path", "")).lower()
    for domain in _DOMAINS:
        if domain in db_path:
            return domain
    return "unknown"


def _db_model(domain: str) -> Any:
    """The pinned τ² DB model for ``domain``.  Both subclass ``tau2.environment.db.DB``, so both
    inherit the same ``DB.load(path)`` JSON classmethod -- verified against the AReaL snapshots
    (``FlightDB.model_validate`` on tau2_airline_db.json: 500 users / 300 flights / 2000 reservations).
    telecom is still out: its snapshot is TOML, not JSON, so ``DB.load`` does not apply."""

    if domain == "retail":
        from tau2.domains.retail.data_model import RetailDB

        return RetailDB
    if domain == "airline":
        from tau2.domains.airline.data_model import FlightDB

        return FlightDB
    raise NotImplementedError(f"no AReaL DB model wired for domain {domain!r}")


@lru_cache(maxsize=None)
def _load_db(domain: str, db_rel: str) -> Any:
    """Load (and cache) one AReaL DB snapshot as the pinned τ² DB model for ``domain``."""

    return _db_model(domain).load(_download(db_rel))


def _sort_key(record: dict[str, Any]) -> tuple[int, str]:
    """Deterministic, natural order: ``retail_2`` before ``retail_10``."""

    rid = str(record.get("id", ""))
    suffix = rid.rsplit("_", 1)[-1]
    return (int(suffix), rid) if suffix.isdigit() else (1 << 30, rid)


def convert_record(record: dict[str, Any]) -> Any:
    """Convert one AReaL RL record into a native upstream τ² ``Task``.

    TELECOM IS CONVERTED DIFFERENTLY, AND THE DIFFERENCE IS LOAD-BEARING. retail/airline records
    carry a per-task DB snapshot that must be injected as ``initialization_data.agent_data``.
    Telecom records instead carry their own ``initial_state.initialization_actions`` (1,769 calls
    across the 271 records: ``break_apn_mms_setting``, ``unseat_sim_card``, ``set_user_info`` …) —
    that IS the task, because telecom tasks are fault-diagnosis. Reusing the retail/airline path
    would break it in two ways, both verified against the pinned source:

      1. HARD CRASH. ``Environment.set_state`` applies ``agent_data`` by pointing ``user_tools.db``
         at the AGENT's db, then calls ``sync_tools()``:
         ``AttributeError: 'TelecomDB' object has no attribute 'surroundings'`` (the user side needs
         ``TelecomUserDB``).
      2. SILENT DATA LOSS, which is worse. Overwriting ``initial_state`` drops every
         ``initialization_action``, so the phone is HEALTHY and the injected fault never happens.
         Episodes would look trivially solved or unsolvable with no error anywhere.

    There is also nothing to inject: AReaL's ``tau2_telecom_db.toml`` is BYTE-IDENTICAL to the
    pinned ``domains/telecom/db.toml`` (9,628 bytes, same sha1), and all 271 records point at that
    one file — unlike retail/airline, which have 4 distinct snapshots each. So telecom passes its
    own ``initial_state`` and ``ticket`` through and injects no DB at all.
    """

    from tau2.data_model.tasks import InitializationData, InitialState, Task

    domain = _domain_of(record)
    if domain not in _SUPPORTED:
        raise NotImplementedError(
            f"AReaL record {record.get('id')!r} is domain {domain!r}; only {_SUPPORTED} is wired."
        )

    criteria = record["evaluation_criteria"]
    if isinstance(criteria, str):
        criteria = json.loads(criteria)

    payload: dict[str, Any] = {
        "id": record["id"],
        "description": record.get("description"),  # dict -> Description; extra keys are ignored
        "user_scenario": record["user_scenario"],  # as-is
        "evaluation_criteria": criteria,  # reward_basis defaults to [DB, COMMUNICATE]
    }

    if domain == "telecom":
        # Pass through verbatim. ``ticket`` is the fault report the agent is given; dropping it
        # would leave the agent diagnosing blind. ``user_tools`` is deliberately NOT set: τ²
        # treats None as "all domain user tools available", which is what dual control needs.
        payload["initial_state"] = record["initial_state"]
        if record.get("ticket") is not None:
            payload["ticket"] = record["ticket"]
    else:
        db = _load_db(domain, record["db_path"])
        payload["initial_state"] = InitialState(
            initialization_data=InitializationData(agent_data=db.model_dump())
        ).model_dump()

    return Task.model_validate(payload)


def load_areal_tasks(domain: str = "retail", split: str | None = None) -> list[Any]:
    """Return the AReaL RL tasks for ``domain`` as upstream τ² ``Task`` objects, in a stable order.

    ``split`` is accepted for signature-compatibility with ``source.load_tasks`` and ignored (the
    AReaL split *is* the domain's RL pool).
    """

    if domain not in _SUPPORTED:
        raise NotImplementedError(f"AReaL loader supports {_SUPPORTED}, not {domain!r}")
    records = sorted((r for r in _rl_records() if _domain_of(r) == domain), key=_sort_key)
    return [convert_record(r) for r in records]

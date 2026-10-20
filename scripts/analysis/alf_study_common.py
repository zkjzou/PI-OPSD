#!/usr/bin/env python
"""Shared paths and transcript helpers for the ALFWorld side-study scripts (per-task PI selection,
self / wrong-self PI, hint-sampling variance, multi-author PI).

PATHS. Every default resolves under the unpacked data bundle, ``$OPSD_DATA_ROOT`` (default
``${XDG_CACHE_HOME:-~/.cache}/opsd/data``), and can be overridden per root:

  PI_GRID_ROOT    your scored cells (<cell>/rows.parquet from the scoring drivers; not in the dataset)
                                                               ./outputs/pi_cells/alfworld
  PI_LABELS_ROOT  judge error-step labels (shipped)            $OPSD_DATA_ROOT/alfworld/pi_metrics/error_labels
  PI_STUDY_ROOT   the studies' scoring work dirs (pi_mm_offline200, pi_mm_rem1400, pi_self16b, ...), written by
                  the prepare scripts (--out) and the chunk scorers, READ by the selection builders. The dataset
                  ships only each study's task lists, hints and labels there; scores are yours.
                                                               $OPSD_DATA_ROOT/alfworld/pi_metrics/scores_new_study
  PI_CORPUS_ROOT  the untrained student's no-hint pass@k corpora (shipped; or regenerate them):
                    passk128_onpolicy_train200/<s>/rollouts.jsonl       N=128 on the 200 offline-analysis tasks
                    passk32_train1600_rem1400/<s>/sh*/rollouts.jsonl    N=32 on the other 1,400 train1600 tasks
                                                               $OPSD_DATA_ROOT/alfworld/corpora

OUTPUTS. Reports and builds write under ./outputs/<script> by default (``default_out``), never into the data
bundle.

TRANSCRIPT HELPERS. ``ACT_RE``, ``parse_chat`` and ``fold_reasoning`` are verbatim copies of the KL-study hint
builder's helpers (the builder these scripts were first written against), so every hint render stays
byte-identical to the published artifacts.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List


def repo_root() -> Path:
    """The repository root: the nearest ancestor of this file holding pyproject.toml, opsd/__init__.py and
    opsd/skyrl_ext/__init__.py (the predicate of opsd._repo)."""
    here = Path(__file__).resolve()
    for d in here.parents:
        if all((d / m).is_file() for m in ("pyproject.toml", "opsd/__init__.py", "opsd/skyrl_ext/__init__.py")):
            return d
    raise RuntimeError(f"no repository root (pyproject.toml + opsd/skyrl_ext/) above {here}")


def data_root() -> Path:
    env = os.environ.get("OPSD_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "opsd" / "data"


ALF = data_root() / "alfworld"
PI_GRID_ROOT = Path(os.environ.get("PI_GRID_ROOT") or Path("outputs") / "pi_cells" / "alfworld")
PI_LABELS_ROOT = Path(os.environ.get("PI_LABELS_ROOT") or ALF / "pi_metrics" / "error_labels")
PI_STUDY_ROOT = Path(os.environ.get("PI_STUDY_ROOT") or ALF / "pi_metrics" / "scores_new_study")
PI_CORPUS_ROOT = Path(os.environ.get("PI_CORPUS_ROOT") or ALF / "corpora")

SPLITS = ALF / "splits"
TRAIN1600 = SPLITS / "task_ids_train1600.json"
TRAIN200 = SPLITS / "task_ids_train200_seed0.json"  # the 200 offline-analysis tasks
DUMP200_4BI = PI_CORPUS_ROOT / "passk128_onpolicy_train200" / "4bi" / "rollouts.jsonl"
REM1400_SHARDS_4BI = PI_CORPUS_ROOT / "passk32_train1600_rem1400" / "4bi" / "sh*" / "rollouts.jsonl"
LABELS_4BI = PI_LABELS_ROOT / "labels_alfworld_4bi_k2_medium.json"
LABELS_REM1400_4BI = PI_LABELS_ROOT / "labels_alfworld_rem1400_4bi_k2_medium_lost.json"
CAP24K_4BI = PI_GRID_ROOT / "alfworld_train200_4bi_cap24k" / "rows.parquet"
GLM_HINTS = ALF / "hints" / "glm-5.2"
QWEN37_HINTS = ALF / "hints" / "qwen3.7-plus"
# the published cap24k renders of the two reference authors (the glm_raw / qwen37_raw arms of the cap24k cell)
CAP24K_REFS = {
    "glm_raw": GLM_HINTS / "hints_traj_withthink_cap24k.json",
    "qwen37_raw": QWEN37_HINTS / "hints_traj_withthink_cap24k.json",
}
MULTI_AUTHOR = Path("outputs") / "pi_multimodel"  # run_alf_pi_multimodel.sh's default OUT (not in the dataset)

# Multi-author generation: the short author names the pipeline (and the published score rows) use ->
# the directory each author's trajectories have in a generation root. A pipeline-written generation root uses
# the short names as directory names; author_dir() accepts either layout.
AUTHOR_DIRS = {
    "dsv41flash": "deepseek-v4.1-flash",
    "minimaxm3": "minimax-m3",
    "qwen38max": "qwen3.8-max",
    "kimik3": "kimi-k3",
    "glm53": "glm-5.3",
    "glm53flash": "glm-5.3-flash",
    "dsv4pro_az": "deepseek-v4-pro",
    "dsv4pro_think": "deepseek-v4-pro-thinking",
}


def author_dir(gen_root, name: str) -> Path:
    """<gen_root>/<name> if it exists, else the bundle's directory for that author."""
    root = Path(gen_root)
    if (root / name).is_dir() or name not in AUTHOR_DIRS:
        return root / name
    return root / AUTHOR_DIRS[name]


def default_out(script: str) -> Path:
    return Path("outputs") / script


# ---- verbatim from the KL-study hint builder ------------------------------------------------------------
ACT_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
# The local-gen dumps keep the transcript as a chat-TEMPLATED string (decoded_response /
# prompt_decoded), not a messages list like the frontier dumps do.
CHAT_RE = re.compile(r"<\|im_start\|>(system|user|assistant)\n?(.*?)(?=<\|im_end\|>|<\|im_start\|>|\Z)", re.DOTALL)


def parse_chat(text: str) -> List[Dict[str, Any]]:
    """chat-templated string -> [{role, content}] (same shape as the frontier `messages`)."""
    return [
        {"role": m.group(1), "content": m.group(2).strip()} for m in CHAT_RE.finditer(text or "") if m.group(2).strip()
    ]


def fold_reasoning(messages: List[Dict[str, Any]], turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Put out-of-band turns[].reasoning back inline as <think>...</think> on each assistant msg.

    turns[] is aligned 1:1 with the assistant messages (verified on the frontier dumps). Without
    this the frontier 'raw' hint carries no chain of thought at all.
    """
    out: List[Dict[str, Any]] = []
    ai = 0
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant":
            reasoning = ""
            if ai < len(turns):
                reasoning = str(turns[ai].get("reasoning") or "")
            ai += 1
            content = str(m.get("content", ""))
            if reasoning and "<think>" not in content:
                m["content"] = f"<think>{reasoning}</think>\n{content}"
        out.append(m)
    return out

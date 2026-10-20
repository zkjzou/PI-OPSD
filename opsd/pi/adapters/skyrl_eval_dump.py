"""Adapt a SkyRL eval dump into the offline PI scorer's expected record format.

OPSD's evaluator (``opsd.skyrl_ext.evaluate.dump_per_dataset_eval_results``, written under
``<export_path>/dumped_evals/``) dumps eval records as
`{input_prompt, output_response, score, stop_reason, env_class, env_extras{extra_info{task_id,...},
reward_spec{ground_truth}}, env_metrics, data_source}` — decoded strings + a per-token reward list (and the
rollout's env metrics), with **no** token ids / loss mask. Success is derived from ``score`` below, so dumps
written before ``env_metrics`` was added still adapt. The offline scorer
(`opsd/trainer/pi/score_trajectories.py`) needs `{task_id, prompt_token_ids, response_ids, loss_mask, success,
...}`.

This adapter reconstructs them:
  - success  = 10.0 present in `score` (env reward = 10.0*won, placed at the last response token).
  - prompt_token_ids = tokenize(input_prompt)  (input_prompt already carries the chat markers).
  - response_ids/loss_mask = split output_response on the `<|im_start|>user\n...<|im_end|>` observation
    blocks (loss_mask=0) vs everything else (assistant turns incl. `<|im_start|>assistant`/`<action>`
    → loss_mask=1), tokenize each segment (add_special_tokens=False) and concatenate. Because the splits
    fall on ATOMIC special tokens (`<|im_start|>`/`<|im_end|>`), per-segment tokenization concatenates
    losslessly (BPE merges never cross a special token) — so we never rely on len(response)==len(score).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Observation turns are the env-inserted `<|im_start|>user\n...<|im_end|>` blocks → loss_mask 0.
# Everything else in output_response is student-generated (assistant think/action + turn wrappers) → 1.
_OBS_RE = re.compile(r"(<\|im_start\|>user\n.*?<\|im_end\|>\n?)", re.DOTALL)


def record_success(score: Any) -> int:
    """1 if the env win reward (10.0) appears anywhere in the per-token score list."""
    return int(any(abs(float(x) - 10.0) < 1e-6 for x in (score or [])))


def adapt_record(rec: Dict[str, Any], tok, k: int = 0) -> Dict[str, Any]:
    ei = (rec.get("env_extras") or {}).get("extra_info", {}) or {}
    task_id = ei.get("task_id") or "unknown"
    score = rec.get("score") or []
    success = record_success(score)

    prompt_ids = tok.encode(rec.get("input_prompt", "") or "", add_special_tokens=False)

    resp = rec.get("output_response", "") or ""
    response_ids: List[int] = []
    loss_mask: List[int] = []
    for seg in _OBS_RE.split(resp):
        if not seg:
            continue
        ids = tok.encode(seg, add_special_tokens=False)
        if not ids:
            continue
        is_obs = bool(_OBS_RE.fullmatch(seg))
        response_ids.extend(ids)
        loss_mask.extend([0 if is_obs else 1] * len(ids))

    return {
        "task_id": task_id,
        "uid": f"{task_id}#{k}",
        "repetition_id": k,
        "success": success,
        "reward": 10.0 * success,
        "env_metrics": {"success": float(success)},
        "prompt_token_ids": prompt_ids,
        "response_ids": response_ids,
        "loss_mask": loss_mask,
    }


def load_skyrl_eval_dump(path: str, tok, shard_index: int = 0, num_shards: int = 1) -> List[Dict[str, Any]]:
    """Load + adapt a SkyRL eval-dump jsonl. Optionally keep only records with global-line-index % num_shards
    == shard_index (data-parallel sharding across GPUs); adapting only the shard's records avoids
    tokenizing the whole dump in every worker. `k` = global line index (stable uid across shards)."""
    out: List[Dict[str, Any]] = []
    gidx = -1
    with Path(path).expanduser().open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            gidx += 1
            if num_shards > 1 and gidx % num_shards != shard_index:
                continue
            out.append(adapt_record(json.loads(line), tok, gidx))
    return out


def game_files_from_skyrl_eval_dump(
    path: str, src_prefix: Optional[str] = None, dst_prefix: Optional[str] = None
) -> Dict[str, str]:
    """Return {task_id: game_file_path} (deduped) from ``env_extras.reward_spec.ground_truth``.

    The dump records the ALFWorld data root of the machine that ran the eval. Pass ``src_prefix`` (that
    root) and ``dst_prefix`` (your local ``$ALFWORLD_DATA``) to remap the paths; with either unset the
    paths are returned unchanged."""
    out: Dict[str, str] = {}
    with Path(path).expanduser().open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ei = (rec.get("env_extras") or {}).get("extra_info", {}) or {}
            rs = (rec.get("env_extras") or {}).get("reward_spec", {}) or {}
            tid = ei.get("task_id")
            gt = rs.get("ground_truth")
            if tid and gt and tid not in out:
                out[tid] = gt.replace(src_prefix, dst_prefix) if (src_prefix and dst_prefix) else gt
    return out


__all__ = ["record_success", "adapt_record", "load_skyrl_eval_dump", "game_files_from_skyrl_eval_dump"]

"""AppWorld schema adapter + dump loader for offline PI scoring/analysis.

``generate_appworld_local.py --save-transcripts`` dumps message-list records
``{task_id, sample, reward, turns, conversation, preserve_thinking}`` where ``turns`` is
``[{turn, think, action, obs, think_chars}, ...]`` — NOT token ids / loss mask. Like the tau adapter,
this reconstructs the token schema the PI scorer consumes (``{task_id, prompt_token_ids, response_ids,
loss_mask, success, reward}``) by rendering the conversation into the Qwen chat format and marking
policy(assistant) tokens 1 / env-observation(user) tokens 0.

THE ONE THING THAT IS NOT LIKE TAU: the prompt is NOT ``conversation[:1]``.

AppWorld's official scaffold (the benchmark's own SimplifiedReActCodeAgent contract) primes the model
with a supervisor instruction AND a multi-exchange worked DEMO before the real task begins. A measured
example: 38 messages, 19 assistant, but only 9 generated turns — so 10 of those assistant messages are
demo text the policy never produced. Treating them as generated would put ~50% of the assistant tokens
in the loss mask as if the student had written them, silently corrupting every KL / logprob attribute.

The boundary is therefore derived from ``turns``: the policy's messages are the **last len(turns)
assistant messages**, and the prompt is everything before the first of them. This holds whether or not
the final turn has a trailing observation (an episode that ends by answering has no final obs; one
truncated at max_turns does), which is why it is preferred over "the last 2n-1 messages".

SUCCESS LABEL. The shared ``is_success`` (``opsd/trainer/pi/score_trajectories.py``) is
``env_metrics.success or reward >= 10.0`` -- the ``10.0 * won`` scale of ALFWorld's env. AppWorld's reward is
in ``[0, 1]`` (``opsd/envs/appworld/env.py`` sets ``reward = partial_completion`` under
``reward_mode="partial"``, or ``1.0/0.0`` under ``"binary"``), so that threshold can never fire. Success
semantics, from the env:
  * ``success``            = AppWorld ``/evaluate`` -> all tests pass. This is **TGC**, and it is BINARY.
  * ``partial_completion`` = n_pass / n_tests, the dense reward. NOT TGC -- do not threshold it as success.
So the rule is ``env_metrics.success`` (training/eval dumps) -> top-level ``success`` -> ``reward >= 1.0``
(transcripts that predate the field; equivalent under REWARD_MODE=binary, a fallback rather than the
source of truth).

SGC (Scenario Goal Completion) is deliberately absent: it is not an env field at all, only an offline
group-by on the ``task_id`` prefix before ``_`` (168 tasks / 56 scenarios), so it has no per-trajectory
meaning and cannot serve as a per-trajectory label.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from opsd.pi.adapters.tokenization import ModelTokenSpec, get_token_spec

# Same observation-turn matcher as the SkyRL eval-dump adapter: env turns are `<|im_start|>user\n...<|im_end|>`.
_OBS_RE = re.compile(r"(<\|im_start\|>user\n.*?<\|im_end\|>\n?)", re.DOTALL)


def _reasoning_of(m: Dict[str, Any]) -> str:
    """OUT-OF-BAND reasoning, as `generate_appworld_local.py` stores it for a thinking policy.

    With `--preserve-thinking` against a `reasoning_history` endpoint that generator appends
    ``{"role": "assistant", "content": code, "reasoning_content": think}`` (lines 350-356), and the
    provider feeds that reasoning back so THE MODEL CONDITIONS ON IT. Rendering only ``content``
    therefore rebuilt a context the rollout never had: measured on Qwen3-4B-Thinking-2507 /
    passk128 normal, 60/60 sampled rollouts carry ``reasoning_content`` and the omission is 24,791
    tokens, 80.6% OF THE REAL CONTEXT. Every logprob in that cell was conditioned on the wrong
    history, and the think channel was empty so the omission was invisible in the channel mix.

    Non-thinking policies (4B-Instruct, 30B-A3B-Instruct) run with preserve_thinking=False and have
    no ``reasoning_content`` at all, so this is a no-op for them -- verified on both dumps.
    """
    rc = m.get("reasoning_content")
    return rc if isinstance(rc, str) else ""


def _render(msgs: List[Dict[str, Any]], add_generation_prompt: bool = False) -> str:
    """Qwen chat render, folding out-of-band ``reasoning_content`` back in as ``<think>``.

    The fold-back format is byte-identical to the generator's own non-``reasoning_history`` branch
    (`generate_appworld_local.run_rollout`, the ``preserve_thinking`` branch without
    ``reasoning_history``) and to adapters/tau2.py, which already solved this for τ².
    `_TAGS_BY_BENCH["appworld"]` in analyze_pi_positions.py already parses <think>/</think>, so the
    refolded tokens land in the `think` channel instead of silently collapsing into `other`.
    """
    s = ""
    for m in msgs:
        content = m.get("content")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        if m.get("role") == "assistant":
            reasoning = _reasoning_of(m)
            if reasoning:
                content = f"<think>\n{reasoning}\n</think>\n\n{content}"
        s += f"<|im_start|>{m.get('role', 'user')}\n{content}<|im_end|>\n"
    if add_generation_prompt:
        s += "<|im_start|>assistant\n"
    return s


def appworld_task_key(rec: Dict[str, Any]) -> Optional[str]:
    """AppWorld task ids are already globally unique ``<scenario>_<variant>`` (e.g. ``fac291d_3``), so the
    raw id is the hint-artifact key. Training/eval dumps nest it under ``extra_info``."""
    tid = rec.get("task_id") or (rec.get("extra_info") or {}).get("task_id")
    return None if tid is None else str(tid)


def appworld_is_success(rec: Dict[str, Any]) -> int:
    """1 iff AppWorld TGC -- ALL tests pass. Never thresholds partial_completion.

    Order: ``env_metrics.success`` -> top-level ``success`` -> ``reward >= 1.0`` (see the module docstring).
    """
    em = rec.get("env_metrics") or {}
    if "success" in em and em["success"] is not None:
        return int(float(em["success"]) >= 1.0)
    if rec.get("success") is not None:
        return int(float(rec.get("success") or 0.0) >= 1.0)
    return int(float(rec.get("reward", 0.0) or 0.0) >= 1.0)


def generated_span(conversation: List[Dict[str, Any]], n_turns: int) -> int:
    """Index in ``conversation`` where the POLICY's own generation starts.

    Returns ``len(conversation)`` (empty response) when there is nothing generated to score, which is
    the correct degenerate answer for a rollout that produced no turns (e.g. an immediate model error).
    """
    if n_turns <= 0:
        return len(conversation)
    assistant_idx = [i for i, m in enumerate(conversation) if m.get("role") == "assistant"]
    if len(assistant_idx) < n_turns:
        # Fewer assistant messages than recorded turns: schema is not what we think it is. Score
        # nothing rather than guess a boundary and silently mask demo tokens as generated.
        return len(conversation)
    return assistant_idx[-n_turns]


class AppworldAdapter:
    name = "appworld"

    def task_key(self, rec: Dict[str, Any]) -> Optional[str]:
        return appworld_task_key(rec)

    def is_success(self, rec: Dict[str, Any]) -> int:
        return appworld_is_success(rec)

    def token_spec(self) -> ModelTokenSpec:
        return get_token_spec("qwen")


def adapt_record(rec: Dict[str, Any], tok, k: int = 0) -> Dict[str, Any]:
    """AppWorld transcript record -> the PI scorer's token schema."""
    conv = rec.get("conversation") or []
    turns = rec.get("turns") or []
    key = appworld_task_key(rec) or "unknown"
    success = appworld_is_success(rec)
    reward = float(rec.get("reward", 0.0) or 0.0)
    sample = rec.get("sample", k)

    start = generated_span(conv, len(turns))
    prompt_str = _render(conv[:start], add_generation_prompt=True)
    resp_str = _render(conv[start:], add_generation_prompt=False)

    prompt_ids = tok.encode(prompt_str, add_special_tokens=False)
    response_ids: List[int] = []
    loss_mask: List[int] = []
    for seg in _OBS_RE.split(resp_str):
        if not seg:
            continue
        ids = tok.encode(seg, add_special_tokens=False)
        if not ids:
            continue
        is_obs = bool(_OBS_RE.fullmatch(seg))
        response_ids.extend(ids)
        loss_mask.extend([0 if is_obs else 1] * len(ids))

    return {
        "task_id": key,
        "uid": f"{key}#{sample}",
        "repetition_id": k,
        "sample": sample,
        "success": success,
        "reward": reward,
        "num_turns": len(turns),
        # `env_metrics` is what aggregate_utility_delta.py reads; keep the same shape as the tau adapter.
        "env_metrics": {"success": float(success), "num_turns": float(len(turns)), "num_invalid_actions": 0.0},
        "stop_reason": rec.get("stop_reason"),
        "prompt_token_ids": prompt_ids,
        "response_ids": response_ids,
        "loss_mask": loss_mask,
    }


def load_appworld_dump(path: str, tok, shard_index: int = 0, num_shards: int = 1) -> List[Dict[str, Any]]:
    """Load + adapt an AppWorld ``transcripts.jsonl`` (data-parallel shardable by global line index)."""
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


__all__ = [
    "appworld_task_key",
    "appworld_is_success",
    "generated_span",
    "AppworldAdapter",
    "adapt_record",
    "load_appworld_dump",
]

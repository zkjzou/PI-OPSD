"""τ²-bench schema adapter + dump loader for offline PI scoring/analysis.

τ²'s pass@k driver dumps records ``{domain, task_id, trial, reward, status, messages,
prompt_messages, response_messages, ...}`` — NOT token ids / loss mask. This module reconstructs the
token schema the PI scorer consumes (``{task_id, prompt_token_ids, response_ids, loss_mask, success,
reward}``).

**DO NOT reuse a τ¹ (tau-bench v1) message-list adapter for τ² — it is a different schema and the failure is
silent.**
Three measured differences (Qwen3-30B-A3B-Instruct-2507, retail-base n=128 corpus, 2026-08-26):

1. **τ² has a ``tool`` role; τ¹ does not.** Tool observations are **69.7%** of all response text.
   A τ¹ adapter's mask only zeroes ``<|im_start|>user`` turns, so every tool observation would enter the
   loss mask as if the policy had generated it.
2. **61.7% of assistant messages have EMPTY ``content``** — the action lives in ``tool_calls``.
   Rendering ``content`` only (what a τ¹ adapter and appworld.py do) drops **42.7% of the policy's own text**,
   and precisely the action tokens, which is fatal for any action-channel attribute.
3. **``messages[0]`` is a scripted agent greeting**, not a generated turn. The prompt/response
   boundary is therefore derived from ``len(response_messages)`` (the appworld.py rule), not from
   "first assistant message".

Design choices:
  - **Faithful Qwen render.** Assistant tool calls render as ``<tool_call>{"name",...}</tool_call>``
    and tool results as a ``user`` turn wrapping ``<tool_response>…</tool_response>`` — which is what
    Qwen3's own chat template emits, so this reconstructs what the policy actually saw. It also means
    the single ``<|im_start|>user`` observation matcher masks BOTH the simulated customer and the tool
    results, with no separate ``tool`` branch to keep in sync.
    ``function.arguments`` is a JSON *string* in the dump; it is parsed and re-dumped to match the
    template (raw string kept if unparseable).
  - **loss_mask**: user turns (customer + tool results) → 0, assistant think/action → 1.
  - **task_key** = the BARE ``task_id``, matching ``build_tau2_pi_hints.py``, which keys hints on
    plain ``task_id`` on purpose: τ² consumers look a hint up by ``env_extras["task_id"]``, and
    **telecom ids contain ``:``**, so τ¹'s ``{domain}:{task_id}`` is not parseable here. Do NOT copy
    a τ¹ domain-qualified key — it matches 0 hints.

    ⚠ **NAMESPACE TRAP, measured.** Bare ids are only safe because a hint artifact covers ONE pool.
    τ² ``retail-base`` ids are bare ordinals (``"0".."113"``) while every existing τ² hint artifact
    (``pi_tau2_full/{glm,qwen}_*``, ``pi_tau2_pool424`` 424, ``pi_tau2_retail_merged`` 326,
    ``pi_tau2_retail_targeted`` 174) is keyed ``retail_<n>`` over the **AReaL 563-task retail pool** —
    0/114 overlap with retail-base. Do not "bridge" the two by mapping id → ``retail_<id>``: that
    yields 86/114 key hits that are DIFFERENT TASKS (verified by content — retail-base ``1`` is a
    keyboard/mouse exchange on order #W2378156; AReaL ``retail_1`` is a white→black mouse exchange),
    i.e. it silently runs the mismatched-PI control while reporting it as matched. Keep one pool per
    artifact and pass ``--only-tasks`` when building.
  - **success** = ``reward >= 1.0``. Verified binary on the retail-base corpus (3,000 rows: exactly
    {0.0, 1.0}), even though τ² reward is composed of db/action/nl/communicate checks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from opsd.pi.adapters.tokenization import ModelTokenSpec, get_token_spec

# Env turns (simulated customer AND tool results) render as `<|im_start|>user\n...<|im_end|>`.
_OBS_RE = re.compile(r"(<\|im_start\|>user\n.*?<\|im_end\|>\n?)", re.DOTALL)


def _render_tool_calls(tool_calls: Any) -> str:
    """Qwen3 tool-call block(s). `function.arguments` is a JSON string in the dump."""
    if not tool_calls:
        return ""
    out = ""
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name") or tc.get("name") or ""
        args = fn.get("arguments", tc.get("arguments", ""))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                pass  # keep the raw string; better than dropping the action
        payload = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        out += f"\n<tool_call>\n{payload}\n</tool_call>"
    return out


def _reasoning_of(m: Dict[str, Any]) -> str:
    """τ²'s OUT-OF-BAND reasoning. Ignoring this drops 94.6% of a thinking policy's own tokens.

    Measured on the retail-base balanced subsets: Qwen3-4B-Thinking-2507 puts 2,060,776 chars in
    ``reasoning_content`` against 116,764 in ``content``; Qwen3-30B-A3B-Instruct-2507 has none.
    Same fold-back that ``build_tau2_pi_hints.py`` performs for hints -- without it the think channel
    is empty and the loss mask covers only the ~5% of the trace that is not reasoning.
    """
    rc = m.get("reasoning_content")
    if not rc:
        rd = m.get("raw_data")
        if isinstance(rd, dict):
            rc = rd.get("reasoning_content")
            if not rc:
                choices = rd.get("choices") or [{}]
                rc = ((choices[0] or {}).get("message") or {}).get("reasoning_content")
    return rc if isinstance(rc, str) else ""


def _msg_text(m: Dict[str, Any]) -> tuple:
    """(role, content) as Qwen would render this τ² message."""
    role = m.get("role", "user")
    content = m.get("content")
    if not isinstance(content, str):
        content = "" if content is None else str(content)
    if role == "assistant":
        reasoning = _reasoning_of(m)
        prefix = f"<think>\n{reasoning}\n</think>\n\n" if reasoning else ""
        return "assistant", prefix + content + _render_tool_calls(m.get("tool_calls"))
    if role == "tool":
        # Qwen renders tool results as a USER turn wrapping <tool_response>.
        return "user", f"<tool_response>\n{content}\n</tool_response>"
    return role, content


def _render(msgs: List[Dict[str, Any]], add_generation_prompt: bool = False) -> str:
    s = ""
    for m in msgs:
        role, content = _msg_text(m)
        s += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    if add_generation_prompt:
        s += "<|im_start|>assistant\n"
    return s


def tau2_task_key(rec: Dict[str, Any]) -> Optional[str]:
    """Bare ``task_id`` — matches build_tau2_pi_hints.py (telecom ids contain ':'). See module docstring."""
    tid = rec.get("task_id")
    return None if tid is None else str(tid)


def tau2_is_success(rec: Dict[str, Any]) -> int:
    """1 if the τ² task succeeded. Verified binary {0.0, 1.0} on retail-base."""
    return int(float(rec.get("reward", 0.0) or 0.0) >= 1.0)


class Tau2Adapter:
    name = "tau2"

    def task_key(self, rec: Dict[str, Any]) -> Optional[str]:
        return tau2_task_key(rec)

    def is_success(self, rec: Dict[str, Any]) -> int:
        return tau2_is_success(rec)

    def token_spec(self) -> ModelTokenSpec:
        return get_token_spec("qwen")


def _split_prompt_response(rec: Dict[str, Any]) -> tuple:
    """(prompt_msgs, response_msgs). Boundary from len(response_messages), NOT 'first assistant'.

    ``messages[0]`` is a scripted greeting the policy never generated. 7/3000 retail-base rows have
    ``len(prompt_messages) + len(response_messages) != len(messages)``; those fall back to
    "system + messages[0]" so the record is still adapted rather than silently dropped.
    """
    msgs = rec.get("messages") or []
    sys_msgs = rec.get("prompt_messages") or []
    resp = rec.get("response_messages")
    n_resp = len(resp) if resp else 0
    if n_resp and n_resp <= len(msgs):
        return list(sys_msgs) + msgs[: len(msgs) - n_resp], msgs[len(msgs) - n_resp :]
    return list(sys_msgs) + msgs[:1], msgs[1:]


def adapt_record(rec: Dict[str, Any], tok, k: int = 0) -> Dict[str, Any]:
    """τ² record -> {task_id, prompt_token_ids, response_ids, loss_mask, success, reward}."""
    key = tau2_task_key(rec) or "unknown"
    success = tau2_is_success(rec)
    reward = float(rec.get("reward", 0.0) or 0.0)

    prompt_msgs, resp_msgs = _split_prompt_response(rec)
    prompt_str = _render(prompt_msgs, add_generation_prompt=True)
    full_str = prompt_str + _render(resp_msgs, add_generation_prompt=False)
    resp_str = full_str[len(prompt_str) :]

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
        "domain": rec.get("domain"),
        "uid": f"{key}#{rec.get('trial', k)}",
        "repetition_id": k,
        "trial": rec.get("trial"),
        "success": success,
        "reward": reward,
        "env_metrics": {"success": float(success)},
        "prompt_token_ids": prompt_ids,
        "response_ids": response_ids,
        "loss_mask": loss_mask,
    }


def load_tau2_dump(path: str, tok, shard_index: int = 0, num_shards: int = 1) -> List[Dict[str, Any]]:
    """Load + adapt a τ² pass@k trajectories jsonl (shardable by global line index)."""
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


__all__ = ["tau2_task_key", "tau2_is_success", "Tau2Adapter", "adapt_record", "load_tau2_dump"]

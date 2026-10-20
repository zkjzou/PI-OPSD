"""Run one τ² episode and normalise the outcome.

τ²'s own ``Orchestrator`` owns the agent↔user↔env loop, including **dual control** (user-side tool
calls are routed to the environment with ``requestor="user"``).  We deliberately do not reimplement
any of it — τ1's 622-line loop has no counterpart here.  This module only:

  * builds env + agent + user + orchestrator,
  * runs the *synchronous* ``run_simulation`` on a worker thread,
  * and flattens the result into a stable, loggable record.

``run_simulation`` both runs the orchestrator and evaluates it, attaching ``reward_info``.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
from concurrent.futures import Executor
from dataclasses import asdict, dataclass, field
from typing import Any

from .agent import PolicySpec, UserSpec, build_policy_agent, build_user_simulator
from .constants import DEFAULT_MAX_ERRORS, DEFAULT_MAX_STEPS, REWARDABLE_TERMINATIONS
from .source import create_environment, get_user_tools

# PI hint block for utility-Δ guided generation. This is OPSD's SECRET_TEMPLATE with the
# `<|im_start|>system` / `<|im_end|>` chat-control tokens REMOVED, and the removal is load-bearing:
# OPSD prepends its template to a RAW prompt string, whereas τ² hands `domain_policy` to
# `LLMAgent`, which puts it inside an already-templated system message. Keeping the tokens would
# embed literal `<|im_start|>` text in the middle of a system message -- tokenised as a real special
# token by the Qwen tokenizer -- which is a different (and malformed) prompt from the one OPSD scores.
# The WORDING is kept identical so the utility numbers stay comparable across the two benchmarks.
TAU2_HINT_BLOCK = (
    "<Secret information>\n"
    "This section contains privileged hints derived from prior successful trajectories. Use them to "
    "stay on\ntrack, but never mention that hints were provided.\n"
    "Hint Summary:\n{hint_lines}\n"
    "Your Task:\n"
    "1. Follow the hints to accomplish the task still reasoning before actions.\n"
    "2. Never acknowledge having access to secret hints.\n"
    "</Secret information>"
)


def derive_seed(base_seed: int, domain: str, task_id: str, trial: int, turn: int = 0) -> int:
    """Deterministic per-episode seed (hash-randomisation free).

    Copied from the τ1 integration; it f-strings its inputs before hashing, so τ²'s string task ids
    work unchanged.  Note τ² applies one seed per *episode* (``Orchestrator.initialize`` pushes it
    into both participants), not per turn as τ1 did.
    """

    digest = hashlib.sha256(f"{base_seed}|{domain}|{task_id}|{trial}|{turn}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


@dataclass
class Tau2EpisodeResult:
    """Flattened outcome of one τ² episode."""

    domain: str
    task_id: str
    task_index: int | None
    trial: int
    seed: int
    reward: float
    status: str
    termination_reason: str | None
    num_messages: int
    num_agent_turns: int
    num_tool_calls: int
    num_user_tool_calls: int
    num_tool_errors: int
    duration: float | None = None
    agent_cost: float | None = None
    user_cost: float | None = None
    reward_breakdown: dict[str, Any] | None = None
    db_check: Any | None = None
    action_checks: Any | None = None
    nl_assertions: Any | None = None
    communicate_checks: Any | None = None
    error: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    # Exactly what the policy saw, for Phase-2 loss masking.  NOT ``SimulationRun.messages``, which
    # also contains the *user's* tool traffic that the policy never observed.
    agent_view: list[dict[str, Any]] = field(default_factory=list)
    # GRPO encode source (OpenAI dicts, thinking preserved via ``reasoning_content``):
    #   prompt_messages   -> the agent's system prompt (domain policy + tools) = the GRPO prompt
    #   response_messages -> everything generated/observed after it = the GRPO response, where
    #                        assistant turns are loss-masked IN and user/tool turns masked OUT.
    prompt_messages: list[dict[str, Any]] = field(default_factory=list)
    response_messages: list[dict[str, Any]] = field(default_factory=list)
    # The agent's tool schemas EXACTLY as τ² handed them to litellm (``Tool.openai_schema``, which is
    # what ``tau2.utils.llm_utils.generate`` sends). The Qwen3 chat template renders these INTO the
    # system block, so omitting them from ``apply_chat_template`` trained the policy on the domain
    # policy ALONE (retail: 1,492 tok) while it was SERVED policy + 16 tool schemas (4,874 tok) --
    # 69% of the real prompt never trained on. Carried from the agent that actually ran rather than
    # re-derived from ``domain``, so it is what was served, not what we assume was served.
    prompt_tools: list[dict[str, Any]] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        # 12 KB of domain-CONSTANT JSON, and baseline.py writes one record per episode. The GRPO
        # encoder reads the live object, never the record, so dropping it here costs nothing.
        record.pop("prompt_tools", None)
        return record


# Statuses that mean "our infrastructure failed", not "the policy failed".  Phase 2 must zero both
# the reward and the loss mask for these so they contribute no gradient.
INFRASTRUCTURE_STATUSES = frozenset(
    {
        "policy_api_error",
        "policy_context_overflow",
        "simulator_api_error",
        "judge_api_error",
        "evaluator_error",
        "environment_error",
        "egress_blocked",
    }
)


_STATUS_BY_LEG = {"agent": "policy_api_error", "user": "simulator_api_error", "judge": "judge_api_error"}


def _classify_exception(exc: BaseException) -> tuple[str, str]:
    """Map an exception escaping ``run_simulation`` to (status, detail).

    Attribution comes from the ``_tau2_leg`` tag that the egress-guard wrapper stamps on, because
    the exception **type alone cannot identify the leg**: the policy, the user simulator and the NL
    judge are all litellm calls raising the same ``RateLimitError``/``APIConnectionError``/
    ``Timeout``/``APIError``.  Before the tag existed this function attributed all of them to the
    policy, so ``simulator_api_error`` was unreachable and a remote user-simulator outage looked
    like our own endpoint failing — opposite remedies (restart engines / drop concurrency vs. the
    hosted endpoint's quota).  Judge failures matter differently again: they mean the reward is *wrong*, not missing.

    Type-based heuristics remain as the fallback for untagged exceptions (anything raised outside a
    completion call), so behaviour is unchanged where we have no better information.
    """

    name = type(exc).__name__
    text = f"{name}: {exc}"[:500]
    lowered = text.lower()

    if name == "EgressNotAllowedError":
        return "egress_blocked", text

    leg = getattr(exc, "_tau2_leg", None)
    # Context overflow is a real, distinct condition on the policy leg (a trace outgrew the window);
    # on the other legs it is just that participant's API failing, so keep it leg-scoped.
    is_overflow = "contextwindow" in lowered or "context length" in lowered or "maximum context" in lowered
    if leg is not None:
        if leg == "agent" and is_overflow:
            return "policy_context_overflow", text
        status = _STATUS_BY_LEG.get(leg)
        if status:
            return status, text

    if is_overflow:
        return "policy_context_overflow", text
    if "jsondecodeerror" in lowered or "nl_assert" in lowered:
        return "judge_api_error", text
    if name in {"RateLimitError", "APIConnectionError", "InternalServerError", "Timeout", "APIError"}:
        return "policy_api_error", text
    return "environment_error", text


def _extract(
    sim: Any, orchestrator: Any, *, domain: str, task: Any, task_index: int | None, trial: int, seed: int
) -> Tau2EpisodeResult:
    messages = list(getattr(sim, "messages", None) or [])

    def dump(msg: Any) -> dict[str, Any]:
        if hasattr(msg, "model_dump"):
            try:
                return msg.model_dump()
            except Exception:  # noqa: BLE001
                pass
        return {"repr": repr(msg)[:2000]}

    reward_info = getattr(sim, "reward_info", None)
    termination = getattr(sim, "termination_reason", None)
    termination_name = getattr(termination, "name", None) or (str(termination) if termination else None)
    if termination_name and "." in termination_name:
        termination_name = termination_name.rsplit(".", 1)[-1]

    num_agent_turns = 0
    num_tool_calls = 0
    num_user_tool_calls = 0
    num_tool_errors = 0
    for msg in messages:
        role = getattr(msg, "role", None)
        calls = getattr(msg, "tool_calls", None) or []
        if role == "assistant":
            num_agent_turns += 1
            num_tool_calls += len(calls)
        requestor = getattr(msg, "requestor", None)
        if requestor == "user" and role == "tool":
            num_user_tool_calls += 1
        if role == "tool" and getattr(msg, "error", False):
            num_tool_errors += 1

    agent_state = getattr(orchestrator, "agent_state", None)
    agent_view: list[dict[str, Any]] = []
    prompt_messages: list[dict[str, Any]] = []
    response_messages: list[dict[str, Any]] = []
    prompt_tools: list[dict[str, Any]] = []
    if agent_state is not None:
        system_msgs = list(getattr(agent_state, "system_messages", None) or [])
        turn_msgs = list(getattr(agent_state, "messages", None) or [])
        for msg in system_msgs + turn_msgs:
            agent_view.append(dump(msg))
        try:
            prompt_messages = openai_messages(system_msgs)
            response_messages = openai_messages(turn_msgs)
            # Same source of truth as the request itself: ``LLMAgent.tools`` is the list this module
            # handed ``build_policy_agent``, and τ²'s generate() sends [t.openai_schema for t in it].
            # MUST be OpenAI-style dicts -- transformers raises ValueError on τ² ``Tool`` objects.
            agent = getattr(orchestrator, "agent", None)
            prompt_tools = [t.openai_schema for t in (getattr(agent, "tools", None) or [])]
        except Exception:  # noqa: BLE001 - encoding aids must never fail an episode
            prompt_messages, response_messages, prompt_tools = [], [], []

    reward = float(getattr(reward_info, "reward", 0.0) or 0.0)
    status = f"terminated_{termination_name.lower()}" if termination_name else "unknown"

    return Tau2EpisodeResult(
        domain=domain,
        task_id=str(task.id),
        task_index=task_index,
        trial=trial,
        seed=seed,
        reward=reward,
        status=status,
        termination_reason=termination_name,
        num_messages=len(messages),
        num_agent_turns=num_agent_turns,
        num_tool_calls=num_tool_calls,
        num_user_tool_calls=num_user_tool_calls,
        num_tool_errors=num_tool_errors,
        duration=getattr(sim, "duration", None),
        agent_cost=getattr(sim, "agent_cost", None),
        user_cost=getattr(sim, "user_cost", None),
        reward_breakdown=getattr(reward_info, "reward_breakdown", None),
        db_check=_maybe_dump(getattr(reward_info, "db_check", None)),
        action_checks=_maybe_dump(getattr(reward_info, "action_checks", None)),
        nl_assertions=_maybe_dump(getattr(reward_info, "nl_assertions", None)),
        communicate_checks=_maybe_dump(getattr(reward_info, "communicate_checks", None)),
        messages=[dump(m) for m in messages],
        agent_view=agent_view,
        prompt_messages=prompt_messages,
        response_messages=response_messages,
        prompt_tools=prompt_tools,
    )


def openai_messages(msgs: list[Any]) -> list[dict[str, Any]]:
    """τ² messages → OpenAI dicts, **preserving the model's thinking** as ``reasoning_content``.

    Reuses τ²'s own ``to_litellm_messages`` for the role/tool-call shape, then re-attaches the
    reasoning that vLLM's ``--reasoning-parser`` split out of ``content``.  That reasoning is not
    lost: ``llm_utils.generate`` stores the whole response on ``AssistantMessage.raw_data``, so
    ``raw_data["choices"][0]["message"]["reasoning_content"]`` still holds it.  (τ²'s hard-coded
    greeting has no ``raw_data`` — it was never generated — which is correct.)

    Pairing this with the preserve chat template (``qwen3thinking_preserve.jinja2``) renders
    ``<think>…</think>`` back for **every** assistant turn, so the encoded training sequence is
    exactly the token stream the policy produced.  Meanwhile the *user simulator* only ever sees
    ``content`` (thinking already stripped), so preserving it here leaks nothing into the episode.

    **Emit the CANONICAL shape — clean ``content`` + separate ``reasoning_content`` — never both.**
    ``to_litellm_messages`` is monkey-patched by ``install_thinking_preservation`` during training,
    so by the time we see it ``content`` may *already* carry an inlined ``<think>…</think>``.  The
    template branches on ``message.reasoning_content is string`` and only splits ``<think>`` out of
    ``content`` in the ``else`` arm, so handing it both fields renders the reasoning **TWICE**
    (verified: ``<think>×2``) — i.e. we would compute log-probs over a token stream the policy never
    produced.  Splitting here makes the two paths (patched / unpatched ``to_litellm_messages``,
    training / eval) converge on a byte-identical render.
    """

    from tau2.utils.llm_utils import to_litellm_messages

    converted = to_litellm_messages(msgs)
    if len(converted) != len(msgs):  # defensive: the zip below assumes 1:1
        return [_split_thinking({k: v for k, v in m.items() if v is not None}) for m in converted]

    out: list[dict[str, Any]] = []
    for src, dst in zip(msgs, converted):
        entry = _split_thinking({k: v for k, v in dst.items() if v is not None})
        raw = getattr(src, "raw_data", None)
        if isinstance(raw, dict) and not entry.get("reasoning_content"):
            try:
                rawmsg = raw["choices"][0]["message"]
                # Provider field-name split, both observed live on 2026-08-02:
                #   hosted glm-5p2 / qwen3p7-plus        -> "reasoning_content"
                #   vLLM 0.20.2 with --reasoning-parser  -> "reasoning" (no reasoning_content at all)
                # Reading only the first name silently DROPS thinking against a local vLLM policy --
                # which is exactly the Phase-2 GRPO setup -- with no error to notice.
                reasoning = rawmsg.get("reasoning_content") or rawmsg.get("reasoning")
            except (KeyError, IndexError, TypeError):
                reasoning = None
            if reasoning:
                entry["reasoning_content"] = reasoning
        out.append(entry)
    return out


def _split_thinking(entry: dict[str, Any]) -> dict[str, Any]:
    """Move an inlined ``<think>…</think>`` out of ``content`` into ``reasoning_content``.

    In-place-safe, idempotent, and a no-op on messages without thinking (user/tool turns, τ²'s
    hard-coded greeting).  See ``openai_messages`` for why both fields must never coexist.
    """
    content = entry.get("content")
    if not isinstance(content, str) or "</think>" not in content:
        return entry
    head, _, body = content.partition("</think>")
    entry["reasoning_content"] = head.split("<think>")[-1].strip("\n")
    entry["content"] = body.lstrip("\n")
    return entry


def _maybe_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:  # noqa: BLE001
            return repr(value)[:2000]
    if isinstance(value, list):
        return [_maybe_dump(v) for v in value]
    return value


async def run_episode(
    *,
    domain: str,
    task: Any,
    task_index: int | None,
    trial: int,
    seed: int,
    policy_spec: PolicySpec,
    user_spec: UserSpec,
    executor: Executor,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_errors: int = DEFAULT_MAX_ERRORS,
    episode_timeout: float | None = None,
    evaluation_type: str = "ALL",
    seed_participants: bool = False,
    hint: str | None = None,
) -> Tau2EpisodeResult:
    """Run one τ² episode on ``executor`` and return a normalised result.

    Every failure mode gets a named status rather than an exception, so one bad episode can never
    abort a sweep.  ``reward`` is τ²'s own ``reward_info.reward`` and is left unbinarised.

    ``seed_participants`` defaults to **False** on purpose.  Passing a seed to the ``Orchestrator``
    makes it call ``set_seed`` on *both* participants, which writes ``seed`` into their ``llm_args``
    and therefore onto every request.  Many hosted models reject the parameter outright — a LiteLLM
    gateway fronting a hosted provider returns ``BadRequestError`` — which fails every episode.  Our own
    ``derive_seed`` value is still recorded on the result for bookkeeping, and reproducibility comes
    from the manifest + seed derivation rather than from server-side sampling determinism (which is
    not bit-reproducible under concurrent batching anyway).  Enable this only when *both* endpoints
    are seed-capable, e.g. a local vLLM policy paired with a local vLLM simulator.
    """

    from tau2.evaluator.evaluator import EvaluationType
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.simulation import run_simulation

    try:
        env = create_environment(domain)
        # PI HINT INJECTION, for utility-Δ guided generation. Appended to the domain policy, which is
        # τ²'s system prompt -- the same SYSTEM-PREPEND convention the ALFWorld utility study used, so
        # the two benchmarks' utility numbers mean the same thing.
        #
        # ⚠️ THIS IS THE STUDENT'S PROMPT, WHICH OPSD NEVER TOUCHES. OPSD puts the hint only in the
        # TEACHER's ref_prompt_token_ids; the student always rolls out unhinted. So this path exists
        # purely to measure utility = success(with-hint) - success(no-hint), and must NOT be used in a
        # training rollout or the student would be trained on a context it never sees at eval.
        #
        # The 6,144-token MAX_PROMPT_LENGTH does NOT bind here: that is a trainer-side truncation, and
        # this runs against a vLLM served at 40k context. All three hint formats fit for generation
        # even though only `actions` would fit inside the training prompt budget.
        _policy_doc = env.get_policy()
        if hint:
            _policy_doc = f"{_policy_doc}\n\n{TAU2_HINT_BLOCK.format(hint_lines=hint)}"
        agent = build_policy_agent(policy_spec, tools=env.get_tools(), domain_policy=_policy_doc)
        user = build_user_simulator(
            user_spec,
            task=task,
            tools=get_user_tools(env, include=getattr(task, "user_tools", None)),
        )
        orchestrator = Orchestrator(
            domain=domain,
            agent=agent,
            user=user,
            environment=env,
            task=task,
            max_steps=max_steps,
            max_errors=max_errors,
            seed=seed if seed_participants else None,
            solo_mode=False,
            simulation_id=f"{domain}/{task.id}/t{trial}",
            timeout=episode_timeout,
        )
    except Exception as exc:  # noqa: BLE001 - setup failures must be reported, not raised
        status, detail = _classify_exception(exc)
        return _failed(domain, task, task_index, trial, seed, status, detail)

    loop = asyncio.get_running_loop()
    try:
        sim = await loop.run_in_executor(
            executor,
            functools.partial(
                run_simulation,
                orchestrator,
                evaluation_type=EvaluationType[evaluation_type.upper()],
            ),
        )
    except Exception as exc:  # noqa: BLE001
        status, detail = _classify_exception(exc)
        return _failed(domain, task, task_index, trial, seed, status, detail)

    result = _extract(sim, orchestrator, domain=domain, task=task, task_index=task_index, trial=trial, seed=seed)
    # Surface the single most common source of a 0.0 reward so it is never mistaken for a bug:
    # tau2 short-circuits the evaluator unless the episode ended on an explicit stop.
    if result.termination_reason and result.termination_reason not in REWARDABLE_TERMINATIONS:
        result.error = result.error or f"reward forced to 0 by termination={result.termination_reason}"
    return result


def _failed(
    domain: str, task: Any, task_index: int | None, trial: int, seed: int, status: str, detail: str
) -> Tau2EpisodeResult:
    return Tau2EpisodeResult(
        domain=domain,
        task_id=str(getattr(task, "id", "?")),
        task_index=task_index,
        trial=trial,
        seed=seed,
        reward=0.0,
        status=status,
        termination_reason=None,
        num_messages=0,
        num_agent_turns=0,
        num_tool_calls=0,
        num_user_tool_calls=0,
        num_tool_errors=0,
        error=detail,
    )

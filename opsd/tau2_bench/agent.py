"""Wire our served policy and a remote user simulator into τ²'s own agent/user classes.

**No fork, no patch, no subclass is required.**  ``LLMAgent._generate_next_message`` calls the
module-level ``tau2.utils.llm_utils.generate(model=self.llm, tools=…, messages=…, **self.llm_args)``,
and ``generate`` splats ``**kwargs`` straight into ``litellm.completion``.  So ``llm_args`` *is* the
injection point: ``llm="openai/<served>"`` plus ``api_base``/``api_key`` routes τ² at any
OpenAI-compatible endpoint (our vLLM policy server, or a hosted API / LiteLLM-style gateway).

Two non-obvious constraints, both verified:

* ``litellm.drop_params = True`` is set globally when τ² is imported, and it **silently drops**
  ``top_k``/``min_p``.  Any vLLM-only sampling parameter must therefore be nested under
  ``extra_body``, which survives ``get_optional_params``.
* Never pre-set ``seed`` in ``llm_args``.  ``Orchestrator.initialize()`` calls
  ``agent.set_seed(seed)``/``user.set_seed(seed)``, and τ²'s ``LLMConfigMixin.set_seed`` warns and
  overwrites if a seed is already present.  (It also deep-copies ``llm_args`` at construction, so a
  single spec may safely be reused across episodes.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from .constants import (
    DEFAULT_SIMULATOR_MODEL,
    default_llm_base_url,
)


def is_local_model_endpoint(base_url: str) -> bool:
    """Whether ``base_url`` is a locally-served engine that accepts vLLM-only parameters.

    Hosted gateways reject them outright: a LiteLLM gateway fronting a hosted provider returns
    ``"Extra inputs are not permitted, field: 'chat_template_kwargs'"``.
    Mirrors the equivalent gate in the τ1 integration's client.
    """

    parsed = urlparse(base_url)
    if parsed.hostname in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}:
        return True
    # Loopback alone was TOO NARROW and silently mis-classified every multi-node run. There the
    # engines are reached through SkyRL's router at ``http://<node_ip>:port/v1`` and a self-hosted
    # simulator at ``http://<sim-host>:8000/v1`` — neither is loopback, so top_k / min_p /
    # chat_template_kwargs were never injected and rollouts sampled with vLLM defaults instead of the
    # intended Qwen config. Scheme is the reliable discriminator here: everything we host ourselves is
    # plain http, while every hosted gateway that rejects these fields is https.
    return parsed.scheme == "http"


@dataclass(frozen=True)
class PolicySpec:
    """How to reach the policy under test (a served, OpenAI-compatible endpoint)."""

    served_model: str
    base_url: str
    api_key: str = "dummy"
    # ``None`` => the field is OMITTED from the request, so the endpoint's OWN default applies (the
    # "default" sampling profile sets both to None). Mirrors the same None-means-omit convention on
    # ``UserSpec``. For a remote thinking model like a hosted qwen3p7-plus, default
    # sampling keeps thinking ON (verified: reasoning_content present with no params sent) and is
    # stochastic (verified: 3/4 distinct on an open probe), so best-of-k retries stay genuine.
    temperature: float | None = 1.0
    top_p: float | None = 0.95
    max_tokens: int = 4096
    # Policy-side reasoning budget for frontier thinking models (None => omit). Verified honoured and
    # server-validated on gemini/gemini-3-flash-preview: "none" yields no reasoning tokens, and an
    # invalid value 500s ("Invalid reasoning effort"). NOTE for Gemini this maps to thinkingBudget,
    # which is a CEILING not a target -- "high" permits more thinking, it does not force it.
    reasoning_effort: str | None = None
    top_k: int | None = 20
    min_p: float | None = 0.0
    enable_thinking: bool | None = True
    num_retries: int = 3
    timeout: float = 300.0
    extra_body: dict[str, Any] = field(default_factory=dict)

    @property
    def litellm_model(self) -> str:
        """``openai/<name>`` selects litellm's OpenAI-compatible provider for a custom base URL."""

        return f"openai/{self.served_model}"

    def llm_args(self) -> dict[str, Any]:
        # vLLM-only knobs must ride in extra_body or litellm's drop_params eats them — but *only* a
        # local vLLM engine accepts them.  A hosted gateway (an API provider or LiteLLM) rejects the whole
        # request with "Extra inputs are not permitted, field: 'chat_template_kwargs'".  So we auto-
        # inject top_k/min_p/enable_thinking only when the endpoint is local; an explicit ``extra_body``
        # always passes through, which is the escape hatch for a remote vLLM that does accept them.
        extra: dict[str, Any] = dict(self.extra_body)
        if is_local_model_endpoint(self.base_url):
            if self.top_k is not None and self.top_k >= 0:
                extra["top_k"] = self.top_k
            if self.min_p is not None:
                extra["min_p"] = self.min_p
            if self.enable_thinking is not None:
                extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = self.enable_thinking

        args: dict[str, Any] = {
            "api_base": self.base_url,
            "api_key": self.api_key,
            "max_tokens": self.max_tokens,
            "num_retries": self.num_retries,
            "timeout": self.timeout,
            # Tags this leg for cost metering: tau2's own ``call_name`` never reaches litellm, but a
            # ``metadata`` kwarg does (visible in the success_callback via litellm_params.metadata).
            "metadata": {"call_name": "agent_response"},
        }
        if self.reasoning_effort is not None:
            args["reasoning_effort"] = self.reasoning_effort
        # Omitted when None so the endpoint's own default sampling applies (see the field comments).
        if self.temperature is not None:
            args["temperature"] = self.temperature
        if self.top_p is not None:
            args["top_p"] = self.top_p
        if extra:
            args["extra_body"] = extra
        return args


@dataclass(frozen=True)
class UserSpec:
    """How to reach the user simulator (a remote frontier model, or a self-hosted vLLM engine)."""

    model: str = DEFAULT_SIMULATOR_MODEL
    # ``$OPENAI_BASE_URL`` if set, else the public OpenAI API -- resolved when the spec is built.
    base_url: str = field(default_factory=default_llm_base_url)
    api_key: str = ""
    # ``None`` => the field is OMITTED from the request, so the MODEL'S OWN DEFAULT applies.
    # Standing simulator config: gpt-5.6-luna, NON-THINKING, temperature 0.0.
    #
    # temperature 0.0 is τ²-bench's OFFICIAL setting for BOTH legs -- verified in Sierra's own shipped
    # result files (data/tau2/results/final/*_4trials.json: agent AND user llm_args == {"temperature":
    # 0.0}, num_trials 4, max_steps 200), matching tau2/config.py.
    # ⚠️ BUT DO NOT READ THIS AS DETERMINISM: gpt-5.6-luna **IGNORES** temperature 0 (probed
    # 2026-08-05: 3/4 distinct completions at temp 0.0 on an open-ended prompt, same as temp 1.0).
    # An earlier version of this comment claimed temp 0 "keeps the environment greedy/reproducible" --
    # that is FALSE for luna. We set it because it is the τ²-official value and the honest intent, not
    # because it buys reproducibility. Every luna-sim run to date has been stochastic.
    temperature: float | None = 0.0
    # A thinking-capable simulator with a small budget returns empty ``content`` and every episode
    # dies as a simulator error — the exact failure that killed 88% of a τ1 run.  Keep this ample
    # and/or disable reasoning outright.  Measured for luna at its DEFAULT effort: 9-22 reasoning
    # tokens per user turn, finish_reason=stop, content non-empty — so 2048 has ~100x headroom.
    max_tokens: int = 2048
    # NON-THINKING by default. Unlike temperature, ``reasoning_effort`` on luna is genuinely HONOURED
    # and server-validated (probed 2026-08-05): "none" is the ONLY value giving reasoning_tokens=0
    # (vs 19-36 for omitted/minimal/low/medium/high), and an invalid value is REJECTED with HTTP 400
    # ("Unsupported value: 'reasoning_effort' does not support ..."), so it is not being silently
    # dropped the way temperature and service_tier are. Pinning "none" also removes the empty-content
    # failure mode structurally: with zero reasoning tokens the sim cannot burn max_tokens inside a
    # think block. An earlier revision left this unpinned, arguing "none" was a deepseek/Qwen artifact;
    # that is superseded -- for a *user simulator* deliberation buys nothing, and it costs tokens.
    reasoning_effort: str | None = "none"
    # reasoning_effort is NOT sufficient for a Qwen3 simulator: vLLM validates it as a Literal and then
    # ignores it, because thinking is controlled by the CHAT TEMPLATE. Observed live with
    # Qwen3.5-35B-A3B: the sim spent its whole 2048-token budget on <think>, --reasoning-parser stripped
    # it, `content` came back EMPTY, and 53 of 62 episodes died at turn 0 with
    # "UserMessage must have either content or tool_calls" — the 88%-failure mode already documented on
    # max_tokens above. enable_thinking=False is the mechanism that actually works (same one PolicySpec
    # uses), injected only for a self-hosted engine since a hosted gateway 400s on chat_template_kwargs.
    enable_thinking: bool | None = False
    # Official per-model sampling for the SIMULATOR. All None by default so the deepseek path keeps
    # sending nothing beyond temperature (its long-standing behaviour); set per model when self-hosting.
    # Qwen3.5-35B-A3B instruct/non-thinking, from the model card: temperature=0.7, top_p=0.8, top_k=20,
    # min_p=0.0, presence_penalty=1.5. top_k/min_p are vLLM-only -> extra_body; the rest are standard.
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    num_retries: int = 3
    timeout: float = 300.0

    @property
    def litellm_model(self) -> str:
        """Address ``base_url`` through litellm's OpenAI-compatible provider.

        The ``openai/`` prefix is **required** even when the downstream name already carries a
        provider segment: litellm parses the first segment as the provider, so a bare
        ``fireworks_ai/…`` is routed to that provider directly and fails with ``AuthenticationError``
        (needing ``FIREWORKS_API_KEY``) instead of going to ``base_url``.  With the prefix, litellm
        strips it and forwards ``fireworks_ai/deepseek-v4-flash`` to ``api_base`` — which is exactly
        the model name an OpenAI-compatible gateway (e.g. LiteLLM) expects.
        """

        return self.model if self.model.startswith("openai/") else f"openai/{self.model}"

    def llm_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            "api_base": self.base_url,
            "api_key": self.api_key,
            "max_tokens": self.max_tokens,
            "num_retries": self.num_retries,
            "timeout": self.timeout,
            "metadata": {"call_name": "user_response"},  # cost-metering leg tag (see PolicySpec)
        }
        # Omitted when None so the model's own default sampling applies (see ``temperature`` above).
        if self.temperature is not None:
            args["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            args["reasoning_effort"] = self.reasoning_effort
        if self.top_p is not None:
            args["top_p"] = self.top_p
        if self.presence_penalty is not None:
            args["presence_penalty"] = self.presence_penalty
        extra: dict[str, Any] = dict(self.extra_body)
        if is_local_model_endpoint(self.base_url):
            if self.enable_thinking is not None:
                extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = self.enable_thinking
            if self.top_k is not None:
                extra["top_k"] = self.top_k
            if self.min_p is not None:
                extra["min_p"] = self.min_p
        if extra:
            args["extra_body"] = extra
        return args


def build_policy_agent(spec: PolicySpec, *, tools: list[Any], domain_policy: str) -> Any:
    """Build τ²'s stock ``LLMAgent`` pointed at our served policy."""

    from tau2.agent.llm_agent import LLMAgent

    return LLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=spec.litellm_model,
        llm_args=spec.llm_args(),
    )


def build_user_simulator(spec: UserSpec, *, task: Any, tools: list[Any] | None) -> Any:
    """Build τ²'s stock ``UserSimulator``.

    ``instructions=str(task.user_scenario)`` mirrors upstream's own runner: passing
    ``task.user_scenario.instructions`` instead would drop the persona and silently produce an
    easier (or at least different) user than the published protocol.
    """

    from tau2.user.user_simulator import UserSimulator

    instructions = str(task.user_scenario) if getattr(task, "user_scenario", None) else None
    return UserSimulator(
        llm=spec.litellm_model,
        instructions=instructions,
        tools=tools,
        llm_args=spec.llm_args(),
    )

"""SkyRL ``GeneratorInterface`` for τ²-bench GRPO (Phase 2).

τ²'s own ``Orchestrator`` still drives every episode (agent ↔ user ↔ env, incl. dual control) — this
module only points τ²'s ``LLMAgent`` at SkyRL's live policy endpoint, then turns each finished
episode into the token-level tensors GRPO needs.

**Preserved thinking.**  The policy is served with ``--reasoning-parser qwen3`` (so the *user
simulator* never sees the agent's chain-of-thought) while ``episode.openai_messages`` re-attaches the
reasoning as ``reasoning_content`` from ``raw_data``.  Encoding then uses the *preserve* chat template
(``qwen3thinking_preserve.jinja2``), which re-renders ``<think>…</think>`` for **every** assistant
turn.  Net effect: the trained sequence is exactly what the model generated (thinking included), and
turn N+1's prompt contains turn N's thinking — a single, self-consistent token stream.  Getting this
wrong is silent: you would score a sequence the model never produced.
"""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.generators.utils import (
    get_response_ids_and_loss_mask_from_messages,
    get_rollout_metrics,
)

from .agent import PolicySpec, UserSpec, is_local_model_endpoint
from .constants import REWARDABLE_TERMINATIONS, default_llm_base_url
from .episode import (
    INFRASTRUCTURE_STATUSES,
    Tau2EpisodeResult,
    derive_seed,
    run_episode,
)
from .source import bootstrap_tau2, load_tasks


# The periodic-eval path (opsd.skyrl_ext.evaluate: evaluate -> calculate_per_dataset_metrics -> SkyRL's
# get_rollout_metrics -> aggregate_for_environment) looks the env up by id and raises "No registered env with
# id: tau2_bench" without a registration.  This MUST live in generator.py, NOT main.py: main.py runs as __main__ so Ray
# cloudpickles the entrypoint BY VALUE and never imports it in the eval worker; generator.py IS imported
# by-reference there (to resolve Tau2Generator).  Same trap the τ1 integration hit in production.
def _register_tau2_metrics_env() -> None:
    from skyrl_gym.envs.registration import register, registry

    if "tau2_bench" in registry:
        return

    class _Tau2MetricsEnv:
        @staticmethod
        def aggregate_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
            from skyrl_gym.metrics import default_aggregate_metrics

            return default_aggregate_metrics(metrics)

    register(id="tau2_bench", entry_point=_Tau2MetricsEnv)


_register_tau2_metrics_env()


def resolve_policy_base_url(generator_cfg: Any, inference_engine_client: Any | None) -> str:
    """Resolve SkyRL's live policy endpoint (router proxy if present, else the HTTP endpoint).

    The returned URL **must end in ``/v1``**: litellm treats ``api_base`` as an OpenAI base and
    appends only ``/chat/completions``.  SkyRL's router hands back a bare ``http://host:port`` (no
    ``/v1``), which would produce ``…/chat/completions`` → **404 ``{"detail":"Not Found"}``** because
    the router serves ``/v1/chat/completions``.  τ1 never hit this: its hand-rolled client appends
    ``/v1/chat/completions`` itself when the base lacks ``/v1``.
    """

    proxy_url = getattr(inference_engine_client, "proxy_url", None)
    if isinstance(proxy_url, str) and proxy_url:
        base = proxy_url.rstrip("/")
        return base if base.endswith("/v1") else f"{base}/v1"
    ie = generator_cfg.inference_engine
    return f"http://{ie.http_endpoint_host}:{ie.http_endpoint_port}/v1"


def _think_leak_fraction(outputs: list["EncodedEpisode"]) -> float:
    """Fraction of assistant turns whose *content* still contains ``<think>`` (must be 0.0).

    ``content`` is the only thing τ² shows the simulated user (``flip_roles`` copies it verbatim
    into a fresh ``UserMessage``), so a non-zero value means the agent's chain-of-thought is leaking
    into the user's context — privileged information that corrupts the environment dynamics.

    Measured on ``result.messages`` — τ²'s **own** ``SimulationRun`` messages, dumped as the
    orchestrator left them — because that is the exact object ``flip_roles`` reads, and it is what
    ``install_thinking_normalization`` mutates.  Do **not** measure ``response_messages``: that is
    our *encoder* output, where ``_split_thinking`` unconditionally moves ``<think>`` into
    ``reasoning_content``, so it reads 0.0 even when the user simulator really did see the CoT.
    Pointing the guard at the encoder would make it vacuous exactly when it matters.
    """

    total = leaked = 0
    for e in outputs:
        for msg in e.result.messages:
            if msg.get("role") != "assistant":
                continue
            total += 1
            if "<think>" in (msg.get("content") or ""):
                leaked += 1
    return (leaked / total) if total else 0.0


def _think_preserved_fraction() -> float:
    """Fraction of the agent's own assistant turns that carried ``<think>`` into its next request.

    Reads and clears the counters ``install_thinking_preservation`` keeps, so the value covers only
    the episodes of this ``generate()`` call.  See ``source.preservation_stats`` for why a healthy
    run reads just under 1.0 rather than exactly 1.0.
    """

    try:
        from .source import preservation_stats

        stats = preservation_stats(reset=True)
        total = stats.get("assistant_msgs", 0)
        return (stats.get("with_thinking", 0) / total) if total else 0.0
    except Exception:  # noqa: BLE001 - a guard metric must never fail a step
        return 0.0


_LEG_METRIC_NAME = {"agent": "policy", "user": "sim", "judge": "judge", "other": "other"}


def _call_volume_metrics(n_episodes: int) -> dict[str, float]:
    """Per-leg LLM call volume / latency / observed retries for this ``generate()``.

    Complements the per-leg FAILURE metrics: those catch calls that died, these catch calls that
    merely got slow or needed retrying — the case that silently inflates ``timing/generate`` while
    every failure counter reads zero.

    ``*_calls_per_ep`` is also the missing denominator for load estimates (e.g. tokens/min against
    a hosted endpoint's TPM ceiling). ``*_retries`` is gateway-side only; see ``source.call_stats``.
    """

    try:
        from .source import call_stats

        stats = call_stats(reset=True)
    except Exception:  # noqa: BLE001 - instrumentation must never fail a step
        return {}
    out: dict[str, float] = {}
    for leg, vals in stats.items():
        name = _LEG_METRIC_NAME.get(leg, leg)
        calls = vals.get("calls", 0.0)
        out[f"tau2/{name}_calls_per_ep"] = calls / n_episodes if n_episodes else 0.0
        out[f"tau2/{name}_retries"] = vals.get("retries", 0.0)
        # Mean seconds per call: a retried-then-successful call shows up HERE, not in the failure
        # counters, because litellm's retry loop runs inside completion() and never surfaces to us.
        out[f"tau2/{name}_latency_s"] = (vals.get("seconds", 0.0) / calls) if calls else 0.0
    return out


@dataclass
class EncodedEpisode:
    prompt_ids: list[int]
    response_ids: list[int]
    loss_mask: list[int]
    reward: float
    raw_reward: float
    stop_reason: str
    result: Tau2EpisodeResult


def encode_episode(
    result: Tau2EpisodeResult,
    tokenizer: Any,
    tokenizer_kwargs: dict[str, Any],
    *,
    training: bool,
    max_response_tokens: int,
) -> EncodedEpisode:
    """Tokenise one episode into (prompt_ids, response_ids, loss_mask, reward).

    ``loss_mask`` is 1 on assistant-generated tokens (thinking + action) and 0 on everything the
    environment/user produced, so the policy gradient only touches what the model actually chose.
    """

    prompt_messages = result.prompt_messages or [{"role": "system", "content": ""}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        # The Qwen3 template renders the tool schemas INTO the system block (its head is
        # byte-identical to qwen3thinking_preserve.jinja2:1-11), which is what vLLM did at rollout.
        # Omitting this trained on 31% of the real retail prompt (1,492 of 4,874 tokens).
        tools=result.prompt_tools or None,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
        **tokenizer_kwargs,
    )

    response_messages = result.response_messages
    if response_messages:
        # NOTE: deliberately NO ``tools=`` here. ``encode_messages_subset`` subtracts a base prefix
        # rendered with the same kwargs, so the tool block cancels exactly -- verified identical
        # token ids AND identical loss mask with and without it. Passing it would only add two
        # ~4.9k-token template renders per message in the driver process.
        response_kwargs = dict(tokenizer_kwargs)
        # ``enable_thinking`` steers the *next* generation prompt; the response already contains the
        # reasoning the model emitted, and SkyRL's message-subset encoder rejects the flag while
        # reconstructing completed assistant turns.
        response_kwargs.pop("enable_thinking", None)
        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
            response_messages, tokenizer, tokenizer_kwargs=response_kwargs
        )
    else:
        fallback = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        response_ids, loss_mask = [fallback], [0]

    raw_reward = float(result.reward or 0.0)
    reward = raw_reward
    infra = result.status in INFRASTRUCTURE_STATUSES
    if infra:
        # Our failure, not the policy's: contribute no reward and no gradient.
        reward, loss_mask = 0.0, [0] * len(loss_mask)
    elif training and len(response_ids) > max_response_tokens:
        # Over-long trace: drop it from the objective rather than train on a truncated sequence.
        reward, loss_mask = 0.0, [0] * len(loss_mask)

    stop_reason = (
        "length"
        if len(response_ids) > max_response_tokens
        else (result.termination_reason or result.status or "unknown")
    )
    return EncodedEpisode(
        prompt_ids=list(prompt_ids),
        response_ids=list(response_ids),
        loss_mask=list(loss_mask),
        reward=reward,
        raw_reward=raw_reward,
        stop_reason=str(stop_reason),
        result=result,
    )


class Tau2Generator(GeneratorInterface):
    """Run τ² episodes against SkyRL's policy endpoint and emit GRPO training tensors."""

    def __init__(
        self,
        generator_cfg: Any,
        tau2_cfg: Any,
        tokenizer: Any,
        policy_model_name: str,
        inference_engine_client: Any | None = None,
    ) -> None:
        self.generator_cfg = generator_cfg
        self.cfg = tau2_cfg
        self.tokenizer = tokenizer
        self.policy_model_name = policy_model_name
        self.inference_engine_client = inference_engine_client
        self.policy_base_url = resolve_policy_base_url(generator_cfg, inference_engine_client)
        # getattr defaults keep an older Tau2Config (or a hand-built stub in tests) working unchanged.
        self.policy_enable_thinking = bool(getattr(self.cfg, "policy_enable_thinking", True))
        _preserve = getattr(self.cfg, "preserve_thinking", None)
        # None => follow the policy's thinking mode. See Tau2Config for why the two are coupled.
        self.preserve_thinking = self.policy_enable_thinking if _preserve is None else bool(_preserve)

        user_key = os.environ.get(self.cfg.simulator_api_key_env or "", "") or "dummy"
        # The judge's key used to be the SIMULATOR's key. That is fine while both share one hosted
        # endpoint, but a self-hosted simulator needs no auth -- so setting simulator_api_key_env=""
        # (the natural thing to do) silently handed the judge "dummy" -> 401 -> judge_api_error, which makes
        # rewards WRONG, not missing. Give the judge its own env var, falling back to the old
        # behaviour so nothing changes for existing runs.
        judge_key = (
            os.environ.get(getattr(self.cfg, "judge_api_key_env", None) or self.cfg.simulator_api_key_env or "", "")
            or user_key
        )
        # Tau2Config resolves its base-URL defaults in the driver ($OPENAI_BASE_URL, else the public
        # OpenAI API). An EMPTY value (e.g. ``tau2.judge_base_url=""``) resolves the same way here,
        # so the egress allowlist below is always built from the URLs actually called.
        judge_base_url = str(self.cfg.judge_base_url or "").strip() or default_llm_base_url()
        judge_args = {
            "api_base": judge_base_url,
            "api_key": judge_key,
            "temperature": 0.0,
        }
        # A comma-separated simulator_base_url fans out across N self-hosted replicas; a single URL
        # behaves exactly as before.
        sim_urls = [u.strip() for u in str(self.cfg.simulator_base_url or "").split(",") if u.strip()]
        sim_urls = sim_urls or [default_llm_base_url()]
        eval_sim_model = (getattr(self.cfg, "eval_simulator_model", "") or "").strip()
        eval_sim_url = (getattr(self.cfg, "eval_simulator_base_url", "") or "").strip() or default_llm_base_url()
        self.source_dir = bootstrap_tau2(
            self.cfg.source_dir,
            max_connections=max(self.cfg.max_concurrency * 2, 32),
            allowed_base_urls=[
                self.policy_base_url,
                # EVERY replica netloc, and the eval simulator when configured: install_egress_guard
                # snapshots this set once and hard-returns if already installed, so it can never be
                # widened later. Replicas differ only by port, which IS part of the netloc.
                *sim_urls,
                *([eval_sim_url] if eval_sim_model and eval_sim_url else []),
                judge_base_url,
            ],
            judge_model=self.cfg.judge_model,
            judge_llm_args=judge_args,
            # Feed the agent its own prior <think> back during rollout, so the sequence we train on
            # is the sequence the model actually conditioned on. Without this, τ² strips reasoning
            # from the agent's context while our encoder preserves it -> silent train/inference
            # mismatch (wrong log-probs, broken IS ratios).
            # OFF for a non-thinking policy: nothing to re-inline, and leaving the wrapper installed
            # still counts assistant turns, so tau2/think_preserved_frac would read a false 0.0 for
            # the whole run. NOTE bootstrap_tau2 installs PROCESS-GLOBAL monkeypatches once per
            # generator, so this cannot vary per phase — one boolean, and the metric gate below must
            # key off the same one.
            preserve_thinking=self.preserve_thinking,
        )
        print(
            f"[tau2] policy={self.policy_model_name} thinking={self.policy_enable_thinking} "
            f"preserve_thinking={self.preserve_thinking}",
            flush=True,
        )

        def _train_spec(url: str) -> UserSpec:
            return UserSpec(
                model=self.cfg.simulator_model,
                base_url=url,
                api_key=user_key,
                max_tokens=self.cfg.simulator_max_tokens,
                reasoning_effort=self.cfg.simulator_reasoning_effort,
                # Per-model official sampling for the simulator. These were the missing third link:
                # the fields can exist on UserSpec and on Tau2Config and still do nothing if the
                # generator never passes them — the same producer-without-consumer break as
                # EVAL_BEFORE_TRAIN.
                temperature=self.cfg.simulator_temperature,
                top_p=self.cfg.simulator_top_p,
                top_k=self.cfg.simulator_top_k,
                min_p=self.cfg.simulator_min_p,
                presence_penalty=self.cfg.simulator_presence_penalty,
                enable_thinking=self.cfg.simulator_enable_thinking,
            )

        # One spec per replica. Episodes are bound to a replica for their whole lifetime (see
        # generate()), which is what lets vLLM's prefix cache stay warm: a τ² simulator prompt grows
        # monotonically, so turn N is turn N-1 plus two messages. A per-REQUEST balancer would
        # scatter an episode's turns and re-prefill 1.6k-3.4k tokens every turn.
        self.user_specs = [_train_spec(u) for u in sim_urls]
        self.user_spec = self.user_specs[0]  # back-compat for single-URL callers/tests
        if len(self.user_specs) > 1:
            print(
                f"[tau2] train simulator: {self.cfg.simulator_model} across "
                f"{len(self.user_specs)} replicas {sim_urls}",
                flush=True,
            )

        # A FIXED eval simulator decouples the yardstick from the training environment: training can
        # run against a self-hosted sim for throughput while the eval curve stays comparable step to
        # step and run to run. Defaults are DELIBERATELY not inherited from simulator_* -- agent.py
        # sends top_p/presence_penalty unconditionally, so a local-Qwen profile would otherwise ship
        # verbatim to a hosted gateway.
        self.eval_user_spec: UserSpec | None = None
        if eval_sim_model:
            eval_key = os.environ.get(self.cfg.eval_simulator_api_key_env or "", "")
            if not eval_key:
                if is_local_model_endpoint(eval_sim_url):
                    eval_key = "dummy"
                else:
                    # A hosted gateway 401s every request; without this the whole eval batch dies as
                    # simulator_api_error AFTER burning the policy's tokens, once per eval_interval.
                    raise RuntimeError(
                        f"tau2.eval_simulator_model={eval_sim_model!r} needs "
                        f"${self.cfg.eval_simulator_api_key_env} (unset)"
                    )
            self.eval_user_spec = UserSpec(
                model=eval_sim_model,
                base_url=eval_sim_url,
                api_key=eval_key,
                max_tokens=self.cfg.eval_simulator_max_tokens,
                reasoning_effort=self.cfg.eval_simulator_reasoning_effort,
                temperature=self.cfg.eval_simulator_temperature,
                top_p=self.cfg.eval_simulator_top_p,
                top_k=self.cfg.eval_simulator_top_k,
                min_p=self.cfg.eval_simulator_min_p,
                presence_penalty=self.cfg.eval_simulator_presence_penalty,
                enable_thinking=self.cfg.eval_simulator_enable_thinking,
            )
            print(
                f"[tau2] eval simulator: {eval_sim_model} @ {eval_sim_url} "
                f"(train: {self.cfg.simulator_model} @ {sim_urls})",
                flush=True,
            )
        # τ² Orchestrator.run() is synchronous -> episodes run on threads.
        self._executor = ThreadPoolExecutor(max_workers=self.cfg.max_concurrency)
        self._tasks_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def _task(self, domain: str, split: str, task_id: str) -> Any:
        key = (domain, split)
        if key not in self._tasks_cache:
            self._tasks_cache[key] = {str(t.id): t for t in load_tasks(domain, split)}
        task = self._tasks_cache[key].get(task_id)
        if task is None:
            raise KeyError(f"task {task_id!r} not in {domain}/{split}")
        return task

    def _policy_spec(self, sampling_params: dict[str, Any] | None) -> PolicySpec:
        sp = sampling_params or {}
        return PolicySpec(
            served_model=self.policy_model_name,
            base_url=self.policy_base_url,
            api_key=os.environ.get(self.cfg.policy_api_key_env or "", "") or "dummy",
            temperature=float(sp.get("temperature", 1.0)),
            top_p=float(sp.get("top_p", 0.95)),
            top_k=sp.get("top_k", 20),
            min_p=sp.get("min_p", 0.0),
            max_tokens=int(sp.get("max_generate_length", self.cfg.max_generate_length)),
            # Thinking policy (default): keep it on — we preserve the reasoning and train on it.
            # Non-thinking policy: THIS LINE is what prevents episode death. False rides in
            # extra_body.chat_template_kwargs (local http endpoints only) and flips
            # Qwen3ReasoningParser.thinking_enabled, so a turn with no </think> returns
            # (reasoning=None, content=<everything>) instead of (reasoning=<everything>, content=None).
            enable_thinking=self.policy_enable_thinking,
            timeout=self.cfg.request_timeout,
        )

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        env_extras = input_batch.get("env_extras")
        trajectory_ids = input_batch.get("trajectory_ids")
        if env_extras is None or trajectory_ids is None:
            raise ValueError("tau2 generation requires env_extras and trajectory_ids")
        if len(env_extras) != len(trajectory_ids):
            raise ValueError("env_extras and trajectory_ids must have equal lengths")

        metadata = input_batch.get("batch_metadata")
        training = metadata is None or metadata.training_phase == "train"
        global_step = metadata.global_step if metadata is not None else 0
        policy_spec = self._policy_spec(input_batch.get("sampling_params"))
        tokenizer_kwargs = dict(getattr(self.generator_cfg, "chat_template_kwargs", {}) or {})

        # The simulator IS the environment, so bind it per generate() call: one call is entirely one
        # phase (SkyRL passes training_phase="eval" for the whole eval dataloader). At eval, a fixed
        # remote simulator keeps the yardstick comparable across steps and runs; at train, episodes
        # fan out across the self-hosted replicas.
        eval_pinned = (not training) and self.eval_user_spec is not None
        specs = [self.eval_user_spec] if eval_pinned else self.user_specs

        encoded: list[EncodedEpisode | None] = [None] * len(env_extras)
        semaphore = asyncio.Semaphore(self.cfg.max_concurrency)

        async def worker(index: int, extras: dict[str, Any], traj: TrajectoryID) -> None:
            async with semaphore:
                domain = str(extras["domain"])
                split = str(extras.get("task_split") or "base")
                task_id = str(extras["task_id"])
                trial = int(traj.repetition_id)
                seed = derive_seed(self.cfg.seed + global_step, domain, task_id, trial)
                try:
                    task = self._task(domain, split, task_id)
                    result = await run_episode(
                        domain=domain,
                        task=task,
                        task_index=extras.get("task_index"),
                        trial=trial,
                        seed=seed,
                        policy_spec=policy_spec,
                        # Per-EPISODE affinity, not per-call: every user turn of this episode goes to
                        # the same replica, so vLLM's prefix cache stays warm as the simulator
                        # conversation grows monotonically.
                        user_spec=specs[index % len(specs)],
                        executor=self._executor,
                        max_steps=self.cfg.max_steps,
                        episode_timeout=self.cfg.episode_timeout,
                        evaluation_type=self.cfg.evaluation_type,
                    )
                except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the batch
                    result = Tau2EpisodeResult(
                        domain=domain,
                        task_id=task_id,
                        task_index=extras.get("task_index"),
                        trial=trial,
                        seed=seed,
                        reward=0.0,
                        status="environment_error",
                        termination_reason=None,
                        num_messages=0,
                        num_agent_turns=0,
                        num_tool_calls=0,
                        num_user_tool_calls=0,
                        num_tool_errors=0,
                        error=f"{type(exc).__name__}: {exc}"[:400],
                    )
                encoded[index] = encode_episode(
                    result,
                    self.tokenizer,
                    tokenizer_kwargs,
                    training=training,
                    max_response_tokens=self.cfg.max_response_tokens,
                )

        async with asyncio.TaskGroup() as group:
            for i, (extras, traj) in enumerate(zip(env_extras, trajectory_ids)):
                group.create_task(worker(i, extras, traj))

        # Keep every returned list index-aligned (incl. trajectory_ids) even if an encode dropped out.
        kept = [i for i, e in enumerate(encoded) if e is not None]
        outputs = [encoded[i] for i in kept]
        trajectory_ids = [trajectory_ids[i] for i in kept]
        rewards = [e.reward for e in outputs]
        responses = [e.response_ids for e in outputs]
        loss_masks = [e.loss_mask for e in outputs]

        rollout_metrics = get_rollout_metrics(responses, rewards, loss_masks=loss_masks)
        n = max(len(outputs), 1)
        rollout_metrics.update(
            {
                "tau2/raw_reward": sum(e.raw_reward for e in outputs) / n,
                "tau2/solve_rate": sum(1.0 for e in outputs if e.raw_reward > 0) / n,
                "tau2/rewardable_frac": sum(
                    1.0 for e in outputs if e.result.termination_reason in REWARDABLE_TERMINATIONS
                )
                / n,
                "tau2/infra_failures": sum(1.0 for e in outputs if e.result.status in INFRASTRUCTURE_STATUSES),
                # Per-leg breakdown, because the three legs fail for unrelated reasons and need opposite
                # responses: policy = our vLLM engines (restart / lower concurrency / context), simulator
                # = the remote paid endpoint (rate limit, quota, outage — and concurrency 64 leans on it
                # harder), judge = reward computation, where a failure means the reward is WRONG rather
                # than missing. A single infra_failures count cannot tell these apart.
                "tau2/infra_policy": sum(
                    1.0 for e in outputs if e.result.status in ("policy_api_error", "policy_context_overflow")
                ),
                "tau2/infra_simulator": sum(1.0 for e in outputs if e.result.status == "simulator_api_error"),
                "tau2/infra_judge": sum(1.0 for e in outputs if e.result.status == "judge_api_error"),
                # The remaining THREE infra statuses had no sub-metric, so infra_failures could exceed
                # policy+simulator+judge and leave the excess UNATTRIBUTABLE -- seen live as
                # "infra_failures=3 with infra_policy/simulator/judge all 0" and no way to tell which.
                # They are not interchangeable: egress_blocked means something tried to call out and the
                # guard stopped it (a real problem), evaluator_error means the reward computation itself
                # failed (reward WRONG rather than merely missing), environment_error is the benign one.
                # One counter keeps it cheap while making the identity hold:
                #   infra_failures == infra_policy + infra_simulator + infra_judge + infra_env
                "tau2/infra_env": sum(
                    1.0
                    for e in outputs
                    if e.result.status in ("evaluator_error", "environment_error", "egress_blocked")
                ),
                "tau2/agent_turns": sum(e.result.num_agent_turns for e in outputs) / n,
                "tau2/user_tool_calls": sum(e.result.num_user_tool_calls for e in outputs) / n,
                "tau2/tool_errors": sum(e.result.num_tool_errors for e in outputs) / n,
                "tau2/response_tokens": sum(len(e.response_ids) for e in outputs) / n,
                "tau2/trainable_tokens": sum(sum(e.loss_mask) for e in outputs) / n,
                # SAFETY GUARD, must stay 0. The user simulator only ever sees AssistantMessage.content
                # (tau2's flip_roles copies content and drops raw_data), and the engine's
                # reasoning_parser=qwen3 keeps <think> OUT of content — reasoning survives only on
                # raw_data for our encoder. If the parser ever silently stops applying, `content` would
                # carry <think> and the agent's chain-of-thought would LEAK into the simulated user's
                # context. That would be invisible in the loss, so surface it as a metric instead.
                # STILL VALID WITH THINKING OFF: 0.0 is the healthy reading in both modes, and it keeps
                # catching a non-thinking model that starts emitting raw <think> into the content the
                # simulated user reads.
                "tau2/think_leak_frac": _think_leak_fraction(outputs),
                # Self-describing runs: which policy mode produced these numbers. 1.0 = thinking.
                "tau2/policy_thinking": 1.0 if self.policy_enable_thinking else 0.0,
            }
        )
        # The OTHER half of the same invariant. think_leak_frac proves the simulated USER never sees
        # the CoT; this proves the AGENT does — the context it sampled from carries its own prior
        # reasoning, so the trained sequence is scored against tokens the model conditioned on.
        # Counts only policy-GENERATED turns: the user simulator shares this code path and its
        # flip_roles view relabels the user's own turns as `assistant`, which diluted an earlier
        # version of this metric to 0.72 on a healthy run.
        # **0.0 means broken, healthy is ~1.0** — which is exactly why it must NOT be published when
        # the preservation wrapper was never installed: with a non-thinking policy it is structurally
        # 0.0 and would read as a whole-run failure on every panel. Drained unconditionally so the
        # counters can never accumulate across steps.
        _preserved = _think_preserved_fraction()
        if self.preserve_thinking:
            rollout_metrics["tau2/think_preserved_frac"] = _preserved
        rollout_metrics.update(_call_volume_metrics(n))
        self._maybe_dump(outputs, global_step, training)

        # Per-row env metrics surface in the eval JSONL dumps (opsd.skyrl_ext.evaluate.dump_per_dataset_eval_results).
        # ``success`` / ``num_turns`` are named for the shared trajectory renderer: its ``_success``
        # checks ``env_metrics["success"]`` FIRST, and its ALFWorld fallback (``score >= 10``) would
        # mark every τ² episode [FAIL] since τ² rewards are 0/1.
        env_metrics = [
            {
                "domain": e.result.domain,
                "task_id": e.result.task_id,
                "raw_reward": e.raw_reward,
                "success": bool(e.raw_reward > 0),
                "status": e.result.status,
                "termination": e.result.termination_reason,
                "agent_turns": e.result.num_agent_turns,
                "num_turns": e.result.num_agent_turns,
                "user_tool_calls": e.result.num_user_tool_calls,
                "tool_errors": e.result.num_tool_errors,
            }
            for e in outputs
        ]
        return {
            "prompt_token_ids": [e.prompt_ids for e in outputs],
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": [e.stop_reason for e in outputs],
            "rollout_metrics": rollout_metrics,
            # We do not return sampler logprobs (τ² drives generation through litellm, so we never
            # see them); the trainer recomputes them in the policy forward.
            "rollout_logprobs": None,
            "trajectory_ids": trajectory_ids,
            "env_metrics": env_metrics,
        }

    def _maybe_dump(self, outputs: list[EncodedEpisode], step: int, training: bool) -> None:
        """Save rollouts to JSONL **and** publish a sampled subset to W&B as an HTML table.

        Mirrors the ALFWorld/OPSD setup (``main_opsd_alfworld._dump_rollouts`` +
        ``_log_trajectories_to_wandb``) and reuses its renderer, so τ² trajectories are inspectable
        the same way.  Two deliberate choices:

        * the text we save is the **decoded trained sequence** (``prompt_ids`` + ``response_ids``),
          not τ²'s message objects — so what you read in W&B is literally what the policy was
          trained on.  That makes the preserved-thinking invariants visually checkable: exactly one
          ``<think>`` per assistant turn, and none inside a ``user`` turn (a leak would be plain to
          see, next to ``tau2/think_leak_frac``).
        * only the first ``TAU2_DUMP_FULL_N`` rollouts carry full text; the rest are summaries, so a
          128-episode step at ~45k tokens each doesn't write hundreds of MB per step.

        Never raises — diagnostics must not kill training.
        """
        full_n = int(os.environ.get("TAU2_DUMP_FULL_N", "4") or 0)
        phase = "train" if training else "eval"
        recs: list[dict[str, Any]] = []
        try:
            for i, e in enumerate(outputs):
                rec = {
                    "i": i,
                    "step": step,
                    "phase": phase,
                    # WHICH simulated user produced these turns. The simulator IS the environment, so a
                    # rollout is only interpretable next to the sim that generated it — and rewards are
                    # not comparable across simulators. Recorded per rollout so a dump is
                    # self-describing rather than needing the launch config to decode.
                    "simulator_model": getattr(self.cfg, "simulator_model", None),
                    "domain": e.result.domain,
                    "task_id": e.result.task_id,
                    "trial": e.result.trial,
                    "reward": e.reward,
                    "raw_reward": e.raw_reward,
                    "reward_sum": e.raw_reward,  # renderer's meta line
                    "data_source": e.result.domain,
                    "status": e.result.status,
                    "termination": e.result.termination_reason,
                    "stop_reason": e.stop_reason,
                    "response_tokens": len(e.response_ids),
                    "n_response_tokens": len(e.response_ids),
                    "n_prompt_tokens": len(e.prompt_ids),
                    "trainable_tokens": int(sum(e.loss_mask)),
                    "agent_turns": e.result.num_agent_turns,
                    # τ²'s own measured episode wall time. Without it, per-task speed can only be
                    # DERIVED from timing/generate x concurrency / batch, which yields a mean and
                    # hides the distribution — and it is the tail, not the mean, that decides how
                    # long a step takes (a step ends when its slowest episode finishes).
                    "duration": e.result.duration,
                    "user_cost": e.result.user_cost,
                    "agent_cost": e.result.agent_cost,
                    "env_metrics": {
                        "success": bool(e.raw_reward > 0),
                        "num_turns": e.result.num_agent_turns,
                        "tool_errors": e.result.num_tool_errors,
                    },
                    "error": e.result.error,
                }
                if i < full_n:
                    rec["prompt_text"] = self.tokenizer.decode(e.prompt_ids, skip_special_tokens=False)
                    rec["response_text"] = self.tokenizer.decode(e.response_ids, skip_special_tokens=False)
                recs.append(rec)
        except Exception:  # noqa: BLE001
            return

        directory = getattr(self.cfg, "trajectory_dir", None)
        if directory:
            path = Path(directory).expanduser() / f"step{step:05d}_{phase}.jsonl"
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    for rec in recs:
                        handle.write(json.dumps(rec, default=str) + "\n")
            except Exception:  # noqa: BLE001 - dumping is diagnostics, never fatal
                pass
        self._log_trajectories_to_wandb([r for r in recs if r.get("response_text")], f"{phase}/trajectories", step)

    def _log_trajectories_to_wandb(self, recs: list[dict[str, Any]], key: str, step: int) -> None:
        """Log up to ``TAU2_WANDB_TRAJ_N`` rollouts as an HTML ``wandb.Table`` at ``step``.

        Logged from the generator (not a trainer subclass): ``generate()`` is awaited in the driver
        process, which is where the tracker lives, so ``wandb.run`` is available and no framework
        edit is needed.  ``commit=False`` so we only attach to the step the trainer is about to log.
        """
        n = int(os.environ.get("TAU2_WANDB_TRAJ_N", "4") or 0)
        if n <= 0 or not recs:
            return
        try:
            import random as _random

            import wandb

            if wandb.run is None:
                return
            from opsd.alfworld.render_trajectories_html import _CSS, _render_traj

            picks = recs if len(recs) <= n else _random.Random(step).sample(recs, n)
            table = wandb.Table(
                columns=[
                    "step",
                    "domain",
                    "task_id",
                    "success",
                    "reward",
                    "termination",
                    "agent_turns",
                    "resp_tok",
                    "trajectory",
                ]
            )
            for j, rec in enumerate(picks):
                em = rec.get("env_metrics") or {}
                table.add_data(
                    step,
                    rec.get("domain"),
                    str(rec.get("task_id")),
                    str(bool(em.get("success"))),
                    rec.get("raw_reward"),
                    str(rec.get("termination")),
                    em.get("num_turns"),
                    rec.get("n_response_tokens"),
                    wandb.Html(f"<style>{_CSS}</style>" + _render_traj(rec, j)),
                )
            wandb.log({key: table}, step=step, commit=False)
        except Exception:  # noqa: BLE001 - trajectory viz must never kill training
            pass

"""Pluggable analyzer used to construct PI hints from trajectories.

One litellm-backed implementation, three backends selected by ``make_analyzer(backend=...)``:
  - ``frontier``: a strong external model -- either a litellm provider string that litellm routes
    natively (``fireworks_ai/<model>`` with ``FIREWORKS_AI_API_KEY``, ``azure_ai/<model>``,
    ``anthropic/<model>``, ...), or any model behind an OpenAI-compatible endpoint
    (``OPSD_PI_ANALYZER_BASE_URL``, falling back to ``OPENAI_BASE_URL``; e.g. a LiteLLM proxy or vLLM).
  - ``policy``  : the live served policy (a standalone ``vllm serve`` of the student), OpenAI-compatible.
  - ``endpoint``: any other OpenAI-compatible base_url (a locally-hosted model / served EMA copy).

There is NO default model: set ``OPSD_PI_ANALYZER_MODEL`` (or pass ``model=``) before the analyzer is used.
A missing model raises :class:`AnalyzerConfigError`, which the batch callers (``build_hints``, the online
hint manager) re-raise instead of counting it as one failed task. See ``make_analyzer`` for the exact
routing / key rules.

OpenAI-compatible calls are litellm ``openai/`` provider calls that differ only in (model, api_base,
api_key); the ``openai/`` prefix forces litellm to POST to ``api_base``. litellm is imported lazily so the
rest of the ``pi`` package (artifact loading, the scorer) has no hard dependency on it.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from loguru import logger

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class AnalyzerConfigError(ValueError):
    """The analyzer cannot work as configured (no model, missing base_url, unknown backend).

    Distinct from a failed CALL (timeout, rate limit, unparseable reply), which callers that process
    many tasks deliberately swallow per task. A configuration error fails EVERY call identically, so
    swallowing it turns a typo into an empty hint artifact (offline) or a run that trains with no
    hints (online). Callers must let this one propagate. Subclasses ValueError so existing
    ``except ValueError`` / ``pytest.raises(ValueError)`` sites keep working.
    """


def _response_cost(resp) -> Optional[float]:
    """Dollars for ONE call: token usage priced by litellm, never a hand-maintained local rate card.

    Order matters:
      1. ``_hidden_params["response_cost"]`` -- what litellm fills in when IT priced the call (usage x
         litellm's model price map).
      2. the ``x-litellm-response-cost`` RESPONSE HEADER -- the per-call price a LiteLLM proxy returns when
         the call goes through one. litellm namespaces provider headers
         (``llm_provider-x-litellm-response-cost``), hence the endswith match rather than an exact key.
      3. ``litellm.completion_cost`` -- a LOCAL recompute from usage, which has no entry for model ids
         missing from litellm's price map and then answers 0.0 or raises. Treat a falsy answer from this
         path as "no answer": a real frontier call is never free, so 0.0 here means unmapped, not cheap.

    Through a LiteLLM proxy sources 1 and 2 are both present and agree, so the cheap first branch is
    enough in practice; the header branch is the fallback for a passthrough that only sets the header.

    The endswith match is deliberately narrow. A LiteLLM proxy also returns
    ``x-litellm-response-cost-{original,discount-amount,margin-amount,margin-percent}``; a substring or
    prefix test would match those too and could book a margin percentage as dollars. Only this call's
    price is read -- never a cumulative per-API-key total, which other users of the same key also move.

    Returns None when no source has an answer, so the caller can COUNT that rather than booking $0 --
    a silently-zero cost panel reads exactly like a free run.
    """
    hp = getattr(resp, "_hidden_params", None) or {}
    if isinstance(hp, dict):
        cost = hp.get("response_cost")
        if cost is not None:
            try:
                return float(cost)
            except (TypeError, ValueError):
                pass
        headers = hp.get("additional_headers") or {}
        if isinstance(headers, dict):
            for key, val in headers.items():
                if str(key).lower().endswith("x-litellm-response-cost"):
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
    try:
        import litellm

        recomputed = litellm.completion_cost(completion_response=resp)
        return float(recomputed) if recomputed else None
    except Exception:  # noqa: BLE001 -- unmapped model id; the counter records it
        return None


def strip_reasoning(text: str) -> str:
    """Drop analyzer chain-of-thought so it never leaks into the hint: remove complete ``<think>…</think>``
    spans, and if a dangling ``</think>`` remains (reasoning without an open tag / truncation), keep only
    the text after the LAST one (the final answer)."""
    if not text:
        return ""
    out = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in out.lower():
        idx = out.lower().rfind("</think>")
        out = out[idx + len("</think>") :]
    return out.strip()


class Analyzer:
    """Async chat-completion analyzer over a single (model, api_base, api_key) target."""

    def __init__(
        self,
        model: Optional[str],
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        request_timeout: int = 180,
        max_retries: int = 4,
        label: str = "analyzer",
        rate_limit_attempts: Optional[int] = None,
        rate_limit_base_s: Optional[float] = None,
        rate_limit_cap_s: Optional[float] = None,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.label = label
        # 429 backoff. litellm's own num_retries fires FIRST and retries almost immediately, which is
        # useless against a provider-level quota: a run died with "LiteLLM Retried: 4 times" on
        # fireworks_ai/qwen3p7-plus. These attempts wrap that with exponential backoff + jitter so the
        # quota window has time to clear. A retrying call keeps holding its semaphore slot while it
        # sleeps, which is deliberate -- it throttles the whole group instead of permanently lowering
        # max_concurrency. Load-bearing for online hint refinement on AppWorld, which fires ~144
        # calls/step at concurrency 48 against a shared endpoint.
        self.rate_limit_attempts = int(
            rate_limit_attempts
            if rate_limit_attempts is not None
            else os.environ.get("OPSD_ANALYZER_RATE_LIMIT_ATTEMPTS", "6") or 6
        )
        self.rate_limit_base_s = float(
            rate_limit_base_s
            if rate_limit_base_s is not None
            else os.environ.get("OPSD_ANALYZER_RATE_LIMIT_BASE_S", "5") or 5
        )
        self.rate_limit_cap_s = float(
            rate_limit_cap_s
            if rate_limit_cap_s is not None
            else os.environ.get("OPSD_ANALYZER_RATE_LIMIT_CAP_S", "120") or 120
        )
        # --- usage / cost accounting -----------------------------------------------------------------
        # Cumulative and monotonic; callers DIFF two snapshots to get a per-step delta (logged to the
        # tracker as online_hints/* metrics).
        #
        # `truncated` (finish_reason == "length") is the highest-value counter of the set. Running the
        # analyzer at max_tokens=1024 silently produced NO hint for ~76% of rollouts across 77 steps of two
        # online-hint runs -- the budget went on reasoning, the JSON was cut mid-object, parse_analyzer_json
        # returned None, and the only trace was a logger.warning from a Ray actor, which does not reach the
        # driver's logs or metrics. This counter makes that failure a visible number instead.
        self.calls = 0
        self.errors = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.truncated = 0
        self.cost_unavailable = 0
        # Cached prompt tokens, where the provider reports them. MEASURED 2026-08-12 on all three
        # analyzers: automatic prefix caching does NOT help this workload. qwen3.7-plus caches an
        # identical prompt (7168/7617) but reports cached=0 the moment the trajectory changes -- and that
        # is the training case, since each of the 144 calls per step reads a DIFFERENT rollout and the only
        # shared prefix is the ~261-token system prompt, far below the ~1024-token minimum both providers
        # need. gpt-5.6-luna reported cached=0 even on a byte-identical repeat. So do not budget for a cache
        # discount. Tracked anyway because it is one number, it makes cost_usd interpretable, and it is the
        # only way we would notice if a prompt restructure ever did make the prefix cacheable.
        self.cached_prompt_tokens = 0

    async def analyze(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = 0.3,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        strip_think: bool = True,
    ) -> str:
        """Sampling params are omitted when ``None`` so the endpoint applies the MODEL's own defaults
        (``temperature=None`` -> "use qwen3.7/glm-5.2 defaults"). ``reasoning_effort`` ('low'|'medium'|'high')
        turns on the model's thinking (the Fireworks/litellm-native way — NB the older ``chat_template_kwargs``
        is rejected by Fireworks): the reasoning comes back out-of-band in ``message.reasoning_content`` and is
        NOT returned here, so it never reaches the hint. ``strip_think`` additionally removes any inline
        ``<think>`` from ``content`` (safety net for models that inline their reasoning)."""
        if not self.model:
            # No default model on purpose (it would assume one provider/endpoint). Checked HERE rather than
            # at construction so configs/tests that never call the analyzer still build (and so
            # ReferenceConstructor, which picks its mode from `analyzer is not None`, behaves as before).
            # Not booked in `errors`: no call was attempted, and the error is raised, not counted.
            raise AnalyzerConfigError(
                f"{self.label}: no analyzer model configured. Set $OPSD_PI_ANALYZER_MODEL (or the caller's "
                "--analyzer-model / OPSD_ONLINE_HINTS_ANALYZER_MODEL) to a litellm provider string such as "
                "fireworks_ai/<model>, or to a model served at $OPSD_PI_ANALYZER_BASE_URL / $OPENAI_BASE_URL."
            )
        import litellm  # lazy: only required when an analyzer is actually used

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            timeout=self.request_timeout,
            num_retries=self.max_retries,
            drop_params=True,  # custom api_base models: drop SDK params the endpoint rejects
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if top_k is not None:
            kwargs["extra_body"] = {"top_k": top_k}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key

        try:
            resp = await self._acompletion_with_backoff(litellm, kwargs)
        except Exception:
            self.errors += 1
            raise
        self._record_usage(resp)
        content = (resp.choices[0].message.content or "").strip()
        return strip_reasoning(content) if strip_think else content

    async def _acompletion_with_backoff(self, litellm, kwargs):
        """litellm.acompletion, retrying RATE LIMITS (429) with exponential backoff + jitter.

        Only 429 is retried here; every other exception propagates immediately so a real
        misconfiguration (bad model name, auth, context overflow) still fails fast instead of sleeping
        through six attempts.
        """
        import asyncio
        import random

        attempts = max(1, self.rate_limit_attempts)
        for attempt in range(attempts):
            try:
                return await litellm.acompletion(**kwargs)
            except Exception as exc:
                is_rate_limit = type(exc).__name__ == "RateLimitError" or "RateLimitError" in repr(type(exc))
                if not is_rate_limit or attempt == attempts - 1:
                    raise
                delay = min(self.rate_limit_base_s * (2**attempt), self.rate_limit_cap_s)
                delay *= 0.5 + random.random()  # jitter: avoid N concurrent callers retrying in lockstep
                logger.warning(
                    f"{self.label}: rate limited on {self.model} "
                    f"(attempt {attempt + 1}/{attempts}); sleeping {delay:.1f}s"
                )
                await asyncio.sleep(delay)

    def _record_usage(self, resp) -> None:
        """Accumulate tokens / dollars / truncations for one successful response. Never raises."""
        self.calls += 1
        try:
            usage = getattr(resp, "usage", None)
            if usage is not None:
                self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
                # Two spellings in the wild: OpenAI-style prompt_tokens_details.cached_tokens and
                # Anthropic-style cache_read_input_tokens. Absent on providers that do not report it.
                details = getattr(usage, "prompt_tokens_details", None)
                cached = getattr(details, "cached_tokens", None) if details is not None else None
                if cached is None:
                    cached = getattr(usage, "cache_read_input_tokens", None)
                self.cached_prompt_tokens += int(cached or 0)
            choices = getattr(resp, "choices", None) or []
            if choices and str(getattr(choices[0], "finish_reason", "") or "").lower() == "length":
                self.truncated += 1
            cost = _response_cost(resp)
            if cost is None:
                self.cost_unavailable += 1
            else:
                self.cost_usd += cost
        except Exception as exc:  # noqa: BLE001 -- accounting must never break a training step
            logger.warning(f"{self.label}: usage accounting skipped: {exc}")

    def usage_snapshot(self) -> dict:
        """Cumulative counters. Diff two of these to get a per-step delta."""
        return {
            "calls": self.calls,
            "errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "cost_usd": self.cost_usd,
            "truncated": self.truncated,
            "cost_unavailable": self.cost_unavailable,
        }


def _ensure_openai_prefix(model: str) -> str:
    """litellm routes ``openai/<name>`` to api_base; add the prefix for raw served names.

    Used by the policy/endpoint backends, which by definition talk to a custom OpenAI-compatible
    ``api_base``, and by the frontier backend whenever it is routed to an OpenAI-compatible base_url
    (e.g. a LiteLLM proxy) -- so the prefix is ALWAYS required unless it is already there.

    The old ``if "/" in model: return model`` guard was wrong for the common case: a served-model-name is
    usually an HF repo id, which always contains a slash, so litellm parsed the part before it as the
    provider. ``Qwen/Qwen3-4B-Thinking-2507`` therefore raised
    ``BadRequestError: LLM Provider NOT provided`` on every call, before any request was sent
    (verified via litellm.get_llm_provider; ``openai/Qwen/Qwen3-4B-Thinking-2507`` correctly resolves to
    provider='openai', model_sent='Qwen/Qwen3-4B-Thinking-2507').
    """
    return model if model.startswith("openai/") else f"openai/{model}"


def _native_litellm_provider(model: str) -> Optional[str]:
    """The provider named by a litellm provider string ``<provider>/<model>`` (e.g. ``fireworks_ai``), else None.

    ``openai/...`` and served-model names whose first segment is not a litellm provider (HF repo ids such
    as ``Qwen/Qwen3-4B-Instruct-2507``, bare names such as ``gpt-4.1``) return None: those are sent to an
    OpenAI-compatible endpoint.
    """
    head, sep, _ = (model or "").partition("/")
    if not sep or head == "openai":
        return None
    try:
        import litellm  # lazy, as everywhere in this module

        providers = {str(getattr(p, "value", p)) for p in litellm.provider_list}
    except Exception:  # noqa: BLE001 -- litellm missing/old: treat the id as a served-model name
        return None
    return head if head in providers else None


def make_analyzer(
    backend: str = "frontier",
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    request_timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> Analyzer:
    """Build an Analyzer for the chosen backend, filling defaults from env vars.

    model    <- ``model``, else $OPSD_PI_ANALYZER_MODEL. There is NO default model. ``policy``/``endpoint``
                raise :class:`AnalyzerConfigError` here without one; ``frontier`` raises it on its first
                call (so a config that enables a frontier analyzer it never calls still constructs).
    base_url <- ``base_url``, else $OPSD_PI_ANALYZER_BASE_URL, else $OPENAI_BASE_URL (frontier/endpoint
                only -- ``policy`` must point at the served policy and never falls back).
    api_key  <- ``api_key``, else $OPSD_PI_ANALYZER_API_KEY, else the env var named by ``api_key_env``
                (default: $OPSD_PI_ANALYZER_API_KEY_ENV, else OPENAI_API_KEY).

    Frontier routing, in order:
      1. an explicit base_url (argument or $OPSD_PI_ANALYZER_BASE_URL): an OpenAI-compatible endpoint; the
         model is sent as ``openai/<model>``, which also covers provider-style ids an endpoint exposes
         (``fireworks_ai/<model>`` behind a LiteLLM proxy).
      2. a litellm provider string (``fireworks_ai/<model>``, ``azure_ai/<model>``, ``anthropic/<model>``,
         ...): routed natively by litellm, which reads the provider's own key variable (e.g.
         FIREWORKS_AI_API_KEY). Only an explicitly configured key ($OPSD_PI_ANALYZER_API_KEY, or the
         variable named by $OPSD_PI_ANALYZER_API_KEY_ENV / ``api_key_env``) is passed through --
         never OPENAI_API_KEY by default, which the provider would reject. $OPENAI_BASE_URL is NOT
         applied here, so an OpenAI base URL set for other tools cannot capture a Fireworks model.
      3. any other model name: $OPENAI_BASE_URL as ``openai/<model>`` when set, else OpenAI itself.
      4. exception to 3 when NO base URL is set at all: a proxy-style id ``openai/<provider>/<model>``
         whose inner part is a litellm provider string (``openai/fireworks_ai/glm-5p2``, the form a
         LiteLLM proxy exposes) would go to api.openai.com and fail on every call, so the ``openai/``
         prefix is stripped and the inner string is routed natively as in rule 2 (with a warning).

    ``request_timeout`` / ``max_retries`` were previously unreachable -- every caller got the 180s / 4
    defaults. 180s is NOT enough once max_tokens is large: measured analyzer throughput on the AppWorld
    online-hint runs was ~54 tok/s, so a 16384-token analyzer call needs ~300s and would time out on EVERY
    long deliberation -- i.e. exactly the calls a raised token budget exists to rescue. Raise both together.
    """
    backend = (backend or "frontier").lower()
    model = model or os.environ.get("OPSD_PI_ANALYZER_MODEL")
    explicit_base = base_url or os.environ.get("OPSD_PI_ANALYZER_BASE_URL")
    configured_key_env = api_key_env or os.environ.get("OPSD_PI_ANALYZER_API_KEY_ENV")
    explicit_key = api_key or os.environ.get("OPSD_PI_ANALYZER_API_KEY")
    api_key = explicit_key or os.environ.get(configured_key_env or "OPENAI_API_KEY")
    # Native provider routes only ever get an EXPLICITLY configured key; otherwise litellm reads the
    # provider's own variable (FIREWORKS_AI_API_KEY, ...).
    native_key = explicit_key or (os.environ.get(configured_key_env) if configured_key_env else None)
    kw = {}
    if request_timeout is not None:
        kw["request_timeout"] = int(request_timeout)
    if max_retries is not None:
        kw["max_retries"] = int(max_retries)

    if backend == "frontier":
        if not model:
            logger.warning("frontier analyzer built without a model; set $OPSD_PI_ANALYZER_MODEL before it is used")
            return Analyzer(None, explicit_base, api_key, label="frontier", **kw)
        # An OpenAI-compatible base_url needs ``openai/<id the endpoint exposes>``. WITHOUT the prefix the
        # litellm SDK resolves the provider from the model string itself -- "fireworks_ai/glm-5p2" routes
        # STRAIGHT to Fireworks, bypassing the endpoint, and sends the endpoint's key as a Fireworks key:
        #   AuthenticationError: key not allowed to access model. This key can only access models=['default']
        # (silently, since _refine_one swallows by default). Verified: openai/fireworks_ai/glm-5p2 through a
        # LiteLLM proxy -> 694-char hint in 23.3s; unprefixed -> AuthenticationError.
        if explicit_base:
            return Analyzer(_ensure_openai_prefix(model), explicit_base, api_key, label="frontier", **kw)
        if _native_litellm_provider(model):
            return Analyzer(model, None, native_key, label="frontier", **kw)
        fallback_base = os.environ.get("OPENAI_BASE_URL")
        if fallback_base:
            return Analyzer(_ensure_openai_prefix(model), fallback_base, api_key, label="frontier", **kw)
        # No base_url at all = direct OpenAI routing. Rule 4: a proxy-style ``openai/<provider>/<model>`` id
        # has no endpoint to go to (api.openai.com does not serve it), so route the inner provider string
        # natively instead of failing every call.
        inner = model[len("openai/") :] if model.startswith("openai/") else ""
        if inner and _native_litellm_provider(inner):
            logger.warning(
                f"frontier analyzer: {model!r} is a proxy-style id but no base URL is set "
                f"($OPSD_PI_ANALYZER_BASE_URL / $OPENAI_BASE_URL); routing {inner!r} natively via litellm "
                "(it reads the provider's own key variable unless $OPSD_PI_ANALYZER_API_KEY[_ENV] is set)."
            )
            return Analyzer(inner, None, native_key, label="frontier", **kw)
        return Analyzer(model, None, api_key, label="frontier", **kw)

    if backend in ("policy", "endpoint"):
        if not model:
            raise AnalyzerConfigError(
                f"backend={backend!r} requires a model name (the served-model-name); "
                "set --analyzer-model or $OPSD_PI_ANALYZER_MODEL."
            )
        base = explicit_base or (os.environ.get("OPENAI_BASE_URL") if backend == "endpoint" else None)
        if not base:
            raise AnalyzerConfigError(
                f"backend={backend!r} requires a base_url (the OpenAI-compatible /v1 endpoint); "
                "set --analyzer-base-url or $OPSD_PI_ANALYZER_BASE_URL"
                + (" (or $OPENAI_BASE_URL)." if backend == "endpoint" else ".")
            )
        # vLLM serve accepts any bearer; default to a placeholder so litellm doesn't error on None.
        return Analyzer(_ensure_openai_prefix(model), base, api_key or "EMPTY", label=backend, **kw)

    raise AnalyzerConfigError(f"unknown analyzer backend {backend!r}; expected frontier|policy|endpoint")

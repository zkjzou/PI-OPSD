"""Unit tests for analyzer usage/cost accounting, 429 backoff, and the openai/ prefix.

No network: ``litellm.acompletion`` is monkeypatched to return a canned response object. Run from the
repo root:

    .venv/bin/python -m pytest opsd/pi/tests/test_analyzer_usage.py -q

WHY THESE EXIST. The analyzer is the only per-step spender in the training loop and its usage was
discarded entirely, so every cost figure had to be reconstructed afterwards from the provider's billing.
Worse, a *truncated* analyzer call bills the full output budget and returns nothing parseable -- which is
exactly how the first pair of online-skill arms trained ~76% of rollouts with no hint for 77 steps
without surfacing an error. These tests pin the counters that make both visible.
"""

from __future__ import annotations

import asyncio

import pytest

import opsd.pi.analyzer as az
from opsd.pi.analyzer import Analyzer, _ensure_openai_prefix, _response_cost
from opsd.pi.online import OnlineHintManager


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content="ok", finish_reason="stop"):
        self.message = _Msg(content)
        self.finish_reason = finish_reason


class _Details:
    def __init__(self, cached_tokens):
        self.cached_tokens = cached_tokens


class _Usage:
    def __init__(self, prompt_tokens=0, completion_tokens=0, cached_tokens=None, cache_read_input_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        if cached_tokens is not None:  # OpenAI spelling
            self.prompt_tokens_details = _Details(cached_tokens)
        if cache_read_input_tokens is not None:  # Anthropic spelling
            self.cache_read_input_tokens = cache_read_input_tokens


class _Resp:
    """Minimal stand-in for a litellm ModelResponse."""

    def __init__(
        self,
        content="ok",
        finish_reason="stop",
        prompt_tokens=100,
        completion_tokens=20,
        hidden_params=None,
        cached_tokens=None,
        cache_read_input_tokens=None,
    ):
        self.choices = [_Choice(content, finish_reason)]
        self.usage = _Usage(prompt_tokens, completion_tokens, cached_tokens, cache_read_input_tokens)
        self._hidden_params = {} if hidden_params is None else hidden_params


def _analyzer(**kw):
    return Analyzer("openai/fireworks_ai/qwen3p7-plus", "https://llm-proxy.example/v1", "k", **kw)


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def _patch_completion(monkeypatch, resp_or_exc, *, cost_fn=None):
    """Patch litellm.acompletion (and optionally completion_cost) on the real module."""
    import litellm

    async def _fake(**kwargs):
        if isinstance(resp_or_exc, BaseException):
            raise resp_or_exc
        return resp_or_exc

    monkeypatch.setattr(litellm, "acompletion", _fake, raising=True)
    if cost_fn is not None:
        monkeypatch.setattr(litellm, "completion_cost", cost_fn, raising=True)
    return litellm


# --------------------------------------------------------------------------------------------------
# cost extraction: the four sources, in priority order
# --------------------------------------------------------------------------------------------------


def test_cost_read_from_hidden_params_response_cost():
    assert _response_cost(_Resp(hidden_params={"response_cost": 0.0123})) == pytest.approx(0.0123)


def test_cost_read_from_namespaced_proxy_header():
    """litellm namespaces provider response headers, so an EXACT key match would miss the cost."""
    resp = _Resp(hidden_params={"additional_headers": {"llm_provider-x-litellm-response-cost": "0.005"}})
    assert _response_cost(resp) == pytest.approx(0.005)


def test_sibling_cost_headers_are_not_mistaken_for_the_cost():
    """A LiteLLM proxy returns MORE cost-ish headers alongside the real one (litellm's proxy
    `get_custom_headers`): the `response-cost-*` variants and per-KEY dollar figures.

    A substring/prefix test would match `-margin-percent` and book a percentage as dollars. Header order
    here puts the decoys first on purpose.
    """
    resp = _Resp(
        hidden_params={
            "additional_headers": {
                "llm_provider-x-litellm-response-cost-margin-percent": "0.0",
                "llm_provider-x-litellm-response-cost-discount-amount": "0.0",
                "llm_provider-x-litellm-response-cost-original": "99.0",
                "llm_provider-x-litellm-key-max-budget": "12.34",  # a per-KEY dollar figure, not this call
                "llm_provider-x-litellm-response-cost": "0.0001848",
            }
        }
    )
    assert _response_cost(resp) == pytest.approx(0.0001848)


def test_hidden_params_wins_over_header():
    resp = _Resp(
        hidden_params={
            "response_cost": 0.02,
            "additional_headers": {"llm_provider-x-litellm-response-cost": "0.005"},
        }
    )
    assert _response_cost(resp) == pytest.approx(0.02)


def test_cost_unavailable_returns_none_not_zero(monkeypatch):
    """A falsy local recompute means 'unmapped model id', not 'the call was free'.

    litellm may have no price entry for a provider-routed model id, so completion_cost answers 0.0 (or
    raises). Booking that as zero is what makes a broken cost panel indistinguishable from a cheap run.
    """
    _patch_completion(monkeypatch, _Resp(), cost_fn=lambda **kw: 0.0)
    assert _response_cost(_Resp()) is None

    def _raise(**kw):
        raise RuntimeError("model not mapped")

    _patch_completion(monkeypatch, _Resp(), cost_fn=_raise)
    assert _response_cost(_Resp()) is None


# --------------------------------------------------------------------------------------------------
# counters
# --------------------------------------------------------------------------------------------------


def test_records_tokens_and_cost(monkeypatch):
    resp = _Resp(prompt_tokens=1234, completion_tokens=567, hidden_params={"response_cost": 0.01})
    _patch_completion(monkeypatch, resp)
    a = _analyzer()
    _run(a.analyze("hi"))
    snap = a.usage_snapshot()
    assert snap["calls"] == 1
    assert snap["prompt_tokens"] == 1234
    assert snap["completion_tokens"] == 567
    assert snap["cost_usd"] == pytest.approx(0.01)
    assert snap["truncated"] == 0
    assert snap["cost_unavailable"] == 0
    assert snap["errors"] == 0


def test_cost_unavailable_counter_increments(monkeypatch):
    _patch_completion(monkeypatch, _Resp(), cost_fn=lambda **kw: 0.0)
    a = _analyzer()
    _run(a.analyze("hi"))
    snap = a.usage_snapshot()
    assert snap["calls"] == 1
    assert snap["cost_unavailable"] == 1
    assert snap["cost_usd"] == 0.0  # an UNDERCOUNT, and cost_unavailable is what says so


def test_cached_prompt_tokens_openai_spelling(monkeypatch):
    _patch_completion(
        monkeypatch, _Resp(prompt_tokens=7617, cached_tokens=7168, hidden_params={"response_cost": 0.005})
    )
    a = _analyzer()
    _run(a.analyze("hi"))
    assert a.usage_snapshot()["cached_prompt_tokens"] == 7168


def test_cached_prompt_tokens_anthropic_spelling(monkeypatch):
    """Two spellings exist in the wild; reading only one would silently report zero caching."""
    _patch_completion(
        monkeypatch, _Resp(prompt_tokens=500, cache_read_input_tokens=400, hidden_params={"response_cost": 0.001})
    )
    a = _analyzer()
    _run(a.analyze("hi"))
    assert a.usage_snapshot()["cached_prompt_tokens"] == 400


def test_cached_tokens_absent_is_zero_not_a_crash(monkeypatch):
    """gpt-5.6-luna reports no cached_tokens at all -- that must read 0, not raise."""
    _patch_completion(monkeypatch, _Resp(prompt_tokens=6847, hidden_params={"response_cost": 0.001}))
    a = _analyzer()
    _run(a.analyze("hi"))
    snap = a.usage_snapshot()
    assert snap["cached_prompt_tokens"] == 0
    assert snap["prompt_tokens"] == 6847


def test_usage_metrics_reports_the_cache_fraction():
    """Expected near ZERO in training: each call reads a different trajectory, so only the ~261-token
    system prompt is shared -- below the ~1024-token minimum both providers need."""
    mgr = OnlineHintManager.__new__(OnlineHintManager)
    a = _analyzer()
    mgr._analyzer = a
    before = a.usage_snapshot()
    a.calls, a.prompt_tokens, a.cached_prompt_tokens = 2, 1000, 250
    m = mgr._usage_metrics(before, n_updated=2)
    assert m["online_hints/cached_prompt_tokens_step"] == 250
    assert m["online_hints/frac_prompt_cached"] == pytest.approx(0.25)


def test_frac_prompt_cached_no_divide_by_zero():
    mgr = OnlineHintManager.__new__(OnlineHintManager)
    a = _analyzer()
    mgr._analyzer = a
    m = mgr._usage_metrics(a.usage_snapshot(), n_updated=0)
    assert m["online_hints/frac_prompt_cached"] == 0.0


def test_truncated_counts_finish_reason_length(monkeypatch):
    """The direct readout of the max_tokens failure that invalidated the first online-skill arms."""
    _patch_completion(monkeypatch, _Resp(finish_reason="length", hidden_params={"response_cost": 0.9}))
    a = _analyzer()
    _run(a.analyze("hi"))
    snap = a.usage_snapshot()
    assert snap["truncated"] == 1
    # A truncated call still bills the full budget -- cost accrues with no usable hint to show for it.
    assert snap["cost_usd"] == pytest.approx(0.9)


def test_errors_counted_and_reraised(monkeypatch):
    _patch_completion(monkeypatch, ValueError("bad request"))
    a = _analyzer()
    with pytest.raises(ValueError):
        _run(a.analyze("hi"))
    snap = a.usage_snapshot()
    assert snap["errors"] == 1
    assert snap["calls"] == 0  # calls counts SUCCESSES, so cost_usd_per_hint is not diluted by failures


def test_accounting_failure_does_not_break_the_call(monkeypatch):
    """A malformed response must still yield content: accounting is a diagnostic, not a gate."""

    class _Broken(_Resp):
        @property
        def usage(self):  # noqa: D401
            raise RuntimeError("no usage on this response shape")

        @usage.setter
        def usage(self, v):
            pass

    _patch_completion(monkeypatch, _Broken(content="the hint"))
    a = _analyzer()
    assert _run(a.analyze("hi")) == "the hint"
    assert a.usage_snapshot()["calls"] == 1


def test_snapshot_diff_is_the_step_delta(monkeypatch):
    _patch_completion(monkeypatch, _Resp(prompt_tokens=10, completion_tokens=2, hidden_params={"response_cost": 0.5}))
    a = _analyzer()
    _run(a.analyze("hi"))
    before = a.usage_snapshot()
    _run(a.analyze("hi"))
    _run(a.analyze("hi"))
    after = a.usage_snapshot()
    delta = {k: after[k] - before[k] for k in after}
    assert delta["calls"] == 2
    assert delta["prompt_tokens"] == 20
    assert delta["cost_usd"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------------------
# the metric dict the manager exports
# --------------------------------------------------------------------------------------------------


def test_usage_metrics_keys_and_arithmetic():
    """_usage_metrics is the seam that carries dollars to W&B; pin the key names and the per-hint price."""
    mgr = OnlineHintManager.__new__(OnlineHintManager)  # no analyzer/tokenizer construction needed
    a = _analyzer()
    mgr._analyzer = a
    before = a.usage_snapshot()
    a.calls, a.prompt_tokens, a.completion_tokens = 4, 400, 80
    a.cost_usd, a.truncated, a.cost_unavailable = 0.8, 1, 2

    m = mgr._usage_metrics(before, n_updated=2)
    assert m["online_hints/cost_usd_step"] == pytest.approx(0.8)
    assert m["online_hints/cost_usd_total"] == pytest.approx(0.8)
    assert m["online_hints/cost_usd_per_hint"] == pytest.approx(0.4)  # priced per hint OBTAINED
    assert m["online_hints/prompt_tokens_step"] == 400
    assert m["online_hints/completion_tokens_step"] == 80
    assert m["online_hints/calls_step"] == 4
    assert m["online_hints/truncated_step"] == 1
    assert m["online_hints/frac_truncated"] == pytest.approx(0.25)
    assert m["online_hints/cost_unavailable_step"] == 2


def test_usage_metrics_no_hints_does_not_divide_by_zero():
    mgr = OnlineHintManager.__new__(OnlineHintManager)
    a = _analyzer()
    mgr._analyzer = a
    before = a.usage_snapshot()
    a.calls, a.cost_usd = 3, 0.3
    m = mgr._usage_metrics(before, n_updated=0)
    assert m["online_hints/cost_usd_per_hint"] == 0.0
    assert m["online_hints/cost_usd_step"] == pytest.approx(0.3)  # spend is still reported


# --------------------------------------------------------------------------------------------------
# 429 backoff + the openai/ prefix
# --------------------------------------------------------------------------------------------------


class _RateLimitError(Exception):
    """Name-matched the way Analyzer._acompletion_with_backoff detects a 429."""


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    import litellm

    calls = {"n": 0}
    slept = []

    async def _flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _RateLimitError("429")
        return _Resp(content="finally")

    async def _no_sleep(delay):
        slept.append(delay)

    monkeypatch.setattr(litellm, "acompletion", _flaky, raising=True)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep, raising=True)
    a = _analyzer(rate_limit_attempts=5, rate_limit_base_s=4, rate_limit_cap_s=100)
    assert _run(a.analyze("hi")) == "finally"
    assert calls["n"] == 3
    assert len(slept) == 2
    # exponential with jitter in [0.5, 1.5): attempt 0 -> 4s, attempt 1 -> 8s, both under the cap
    assert 2.0 <= slept[0] < 6.0
    assert 4.0 <= slept[1] < 12.0


def test_rate_limit_delay_respects_the_cap(monkeypatch):
    import litellm

    slept = []

    async def _always_429(**kwargs):
        raise _RateLimitError("429")

    async def _no_sleep(delay):
        slept.append(delay)

    monkeypatch.setattr(litellm, "acompletion", _always_429, raising=True)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep, raising=True)
    a = _analyzer(rate_limit_attempts=6, rate_limit_base_s=10, rate_limit_cap_s=20)
    with pytest.raises(_RateLimitError):
        _run(a.analyze("hi"))
    assert len(slept) == 5  # last attempt re-raises instead of sleeping
    assert max(slept) < 20 * 1.5  # capped before jitter


def test_non_rate_limit_error_is_not_retried(monkeypatch):
    import litellm

    calls = {"n": 0}

    async def _bad(**kwargs):
        calls["n"] += 1
        raise ValueError("bad model name")

    monkeypatch.setattr(litellm, "acompletion", _bad, raising=True)
    a = _analyzer(rate_limit_attempts=6)
    with pytest.raises(ValueError):
        _run(a.analyze("hi"))
    assert calls["n"] == 1, "a real misconfiguration must fail fast, not sleep through six attempts"


def test_ensure_openai_prefix_handles_hf_repo_ids():
    """The old `if "/" in model` guard left HF repo ids unprefixed -> LLM Provider NOT provided."""
    assert _ensure_openai_prefix("Qwen/Qwen3-4B-Thinking-2507") == "openai/Qwen/Qwen3-4B-Thinking-2507"
    assert _ensure_openai_prefix("my-served-model") == "openai/my-served-model"
    assert _ensure_openai_prefix("openai/fireworks_ai/glm-5p2") == "openai/fireworks_ai/glm-5p2"


def test_the_three_analyzer_ids_keep_their_proxy_routing_prefix():
    """The three analyzers in use, as `openai/<proxy model id>`.

    The prefix is what makes litellm POST to api_base. WITHOUT it the call goes straight to the provider
    and a LiteLLM-proxy VIRTUAL key is sent as a provider key -> AuthenticationError ("key not allowed to
    access model"), which is an auth failure rather than an obvious routing bug. Note luna's proxy id is
    BARE while the fireworks ids carry their own provider segment, so the prefix must compose with both.
    All three were verified live end-to-end through the onpolicy_skill path.
    """
    for proxy_id in ("fireworks_ai/qwen3p7-plus", "fireworks_ai/deepseek-v4-flash-0731", "gpt-5.6-luna"):
        wired = f"openai/{proxy_id}"
        assert _ensure_openai_prefix(proxy_id) == wired
        # idempotent: passing the already-wired name through again must not double the prefix
        assert _ensure_openai_prefix(wired) == wired


def test_rate_limit_knobs_default_from_env(monkeypatch):
    monkeypatch.setenv("OPSD_ANALYZER_RATE_LIMIT_ATTEMPTS", "9")
    monkeypatch.setenv("OPSD_ANALYZER_RATE_LIMIT_BASE_S", "2.5")
    monkeypatch.setenv("OPSD_ANALYZER_RATE_LIMIT_CAP_S", "60")
    a = az.make_analyzer("endpoint", model="m", base_url="https://p/v1", api_key="k")
    assert (a.rate_limit_attempts, a.rate_limit_base_s, a.rate_limit_cap_s) == (9, 2.5, 60.0)

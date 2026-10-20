"""Unit tests for ``make_analyzer`` routing / key rules and the AnalyzerConfigError fail-loud contract.

No network: every test either inspects the built ``Analyzer`` (model, api_base, api_key) or monkeypatches
``Analyzer.analyze``. Run from the repo root:

    python -m pytest opsd/pi/tests/test_analyzer_routing.py -q

WHY THESE EXIST. The analyzer has no default model and routes by model-string shape (OpenAI-compatible
endpoint vs a litellm provider string). Both choices fail SILENTLY in the batch callers if they regress:
a misrouted model fails every call, and per-call failures are swallowed by design (one bad task must not
kill a 2400-task build or a training step). The configuration error is the one failure that must NOT be
swallowed -- before AnalyzerConfigError, a missing model produced an empty hint artifact with exit 0.
"""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

import opsd.pi.analyzer as az
from opsd.pi.analyzer import AnalyzerConfigError, make_analyzer
from opsd.pi.base import TrajView

_ENV = (
    "OPSD_PI_ANALYZER_MODEL",
    "OPSD_PI_ANALYZER_BASE_URL",
    "OPSD_PI_ANALYZER_API_KEY",
    "OPSD_PI_ANALYZER_API_KEY_ENV",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "LITELLM_PROXY_URL",
    "FIREWORKS_AI_API_KEY",
    "MY_ANALYZER_KEY",
    "OPSD_ONLINE_HINTS_STRICT",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


def _target(a):
    return a.model, a.api_base, a.api_key


# --------------------------------------------------------------------------------------------------
# _native_litellm_provider
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model,expected",
    [
        ("fireworks_ai/glm-5p2", "fireworks_ai"),
        ("azure_ai/some-model", "azure_ai"),
        ("anthropic/some-model", "anthropic"),
        ("openai/fireworks_ai/glm-5p2", None),  # openai/ = OpenAI-compatible endpoint, never native
        ("openai/gpt-4.1", None),
        ("Qwen/Qwen3-4B-Instruct-2507", None),  # HF repo id: org is not a litellm provider
        ("gpt-4.1", None),  # no slash
        ("", None),
        (None, None),
    ],
)
def test_native_litellm_provider(model, expected):
    pytest.importorskip("litellm")
    assert az._native_litellm_provider(model) == expected


# --------------------------------------------------------------------------------------------------
# frontier routing: one case per row of the make_analyzer docstring
# --------------------------------------------------------------------------------------------------
_FRONTIER_CASES = [
    # (id, env, make_analyzer kwargs, expected (model, api_base, api_key))
    # Rule 1: explicit base_url (argument) -> openai/<model> at that endpoint, default key = OPENAI_API_KEY.
    (
        "r1_arg_base_provider_id",
        {"OPENAI_API_KEY": "sk-oa"},
        dict(model="fireworks_ai/glm-5p2", base_url="https://proxy/v1"),
        ("openai/fireworks_ai/glm-5p2", "https://proxy/v1", "sk-oa"),
    ),
    # Rule 1 via $OPSD_PI_ANALYZER_BASE_URL; an HF repo id gets the prefix too.
    (
        "r1_env_base_hf_id",
        {"OPSD_PI_ANALYZER_BASE_URL": "http://vllm:8000/v1", "OPENAI_API_KEY": "sk-oa"},
        dict(model="Qwen/Qwen3-4B-Instruct-2507"),
        ("openai/Qwen/Qwen3-4B-Instruct-2507", "http://vllm:8000/v1", "sk-oa"),
    ),
    # Rule 1 beats $OPENAI_BASE_URL, and an already-prefixed (proxy-style) id is not double-prefixed.
    (
        "r1_wins_over_openai_base_no_double_prefix",
        {
            "OPSD_PI_ANALYZER_BASE_URL": "https://proxy/v1",
            "OPENAI_BASE_URL": "https://other/v1",
            "OPENAI_API_KEY": "sk-oa",
        },
        dict(model="openai/fireworks_ai/glm-5p2"),
        ("openai/fireworks_ai/glm-5p2", "https://proxy/v1", "sk-oa"),
    ),
    # Model from $OPSD_PI_ANALYZER_MODEL.
    (
        "model_from_env",
        {"OPSD_PI_ANALYZER_MODEL": "m-served", "OPSD_PI_ANALYZER_BASE_URL": "https://p/v1"},
        dict(),
        ("openai/m-served", "https://p/v1", None),
    ),
    # Rule 2: native provider string -> routed natively; NEVER handed OPENAI_API_KEY, and $OPENAI_BASE_URL
    # does not capture it.
    (
        "r2_native_ignores_openai_key_and_base",
        {"OPENAI_API_KEY": "sk-oa", "OPENAI_BASE_URL": "https://other/v1", "FIREWORKS_AI_API_KEY": "fw"},
        dict(model="fireworks_ai/glm-5p2"),
        ("fireworks_ai/glm-5p2", None, None),
    ),
    # Rule 2 + $OPSD_PI_ANALYZER_API_KEY_ENV names the variable holding the key.
    (
        "r2_native_key_env",
        {"OPSD_PI_ANALYZER_API_KEY_ENV": "MY_ANALYZER_KEY", "MY_ANALYZER_KEY": "sk-mine", "OPENAI_API_KEY": "sk-oa"},
        dict(model="fireworks_ai/glm-5p2"),
        ("fireworks_ai/glm-5p2", None, "sk-mine"),
    ),
    # Rule 2 + the api_key_env ARGUMENT (build_hints --analyzer-api-key-env) beats the env var.
    (
        "r2_native_key_env_arg",
        {"OPSD_PI_ANALYZER_API_KEY_ENV": "OPENAI_API_KEY", "MY_ANALYZER_KEY": "sk-mine", "OPENAI_API_KEY": "sk-oa"},
        dict(model="fireworks_ai/glm-5p2", api_key_env="MY_ANALYZER_KEY"),
        ("fireworks_ai/glm-5p2", None, "sk-mine"),
    ),
    # Rule 2 + $OPSD_PI_ANALYZER_API_KEY (the key itself) wins over the named variable.
    (
        "r2_native_explicit_key",
        {
            "OPSD_PI_ANALYZER_API_KEY": "sk-explicit",
            "OPSD_PI_ANALYZER_API_KEY_ENV": "MY_ANALYZER_KEY",
            "MY_ANALYZER_KEY": "sk-mine",
        },
        dict(model="fireworks_ai/glm-5p2"),
        ("fireworks_ai/glm-5p2", None, "sk-explicit"),
    ),
    # Rule 3: other names -> $OPENAI_BASE_URL as openai/<model> when set ...
    (
        "r3_openai_base_fallback",
        {"OPENAI_BASE_URL": "https://compat/v1", "OPENAI_API_KEY": "sk-oa"},
        dict(model="Qwen/Qwen3-4B"),
        ("openai/Qwen/Qwen3-4B", "https://compat/v1", "sk-oa"),
    ),
    # ... and the api_key_env indirection applies on the endpoint routes too.
    (
        "r3_openai_base_key_env",
        {
            "OPENAI_BASE_URL": "https://compat/v1",
            "OPENAI_API_KEY": "sk-oa",
            "OPSD_PI_ANALYZER_API_KEY_ENV": "MY_ANALYZER_KEY",
            "MY_ANALYZER_KEY": "sk-mine",
        },
        dict(model="Qwen/Qwen3-4B"),
        ("openai/Qwen/Qwen3-4B", "https://compat/v1", "sk-mine"),
    ),
    # Rule 3 without any base -> OpenAI itself, string untouched.
    ("r3_direct_openai", {"OPENAI_API_KEY": "sk-oa"}, dict(model="gpt-4.1"), ("gpt-4.1", None, "sk-oa")),
    # Only the documented variables route: an unrelated proxy variable is NOT a routing input.
    (
        "unrelated_proxy_var_ignored",
        {"LITELLM_PROXY_URL": "https://some-proxy/v1", "OPENAI_API_KEY": "sk-oa"},
        dict(model="gpt-4.1"),
        ("gpt-4.1", None, "sk-oa"),
    ),
    # Rule 4: proxy-style openai/<provider>/<model> with NO base URL anywhere -> stripped, routed natively.
    (
        "r4_proxy_id_without_base_goes_native",
        {"OPENAI_API_KEY": "sk-oa"},
        dict(model="openai/fireworks_ai/glm-5p2"),
        ("fireworks_ai/glm-5p2", None, None),
    ),
    (
        "r4_proxy_id_native_key_env",
        {"OPENAI_API_KEY": "sk-oa", "OPSD_PI_ANALYZER_API_KEY_ENV": "MY_ANALYZER_KEY", "MY_ANALYZER_KEY": "sk-mine"},
        dict(model="openai/fireworks_ai/qwen3p7-plus"),
        ("fireworks_ai/qwen3p7-plus", None, "sk-mine"),
    ),
    # Rule 4 does NOT fire when $OPENAI_BASE_URL is set (rule 3: the proxy-style id goes to that endpoint) ...
    (
        "r4_not_with_openai_base",
        {"OPENAI_BASE_URL": "https://proxy/v1", "OPENAI_API_KEY": "sk-oa"},
        dict(model="openai/fireworks_ai/glm-5p2"),
        ("openai/fireworks_ai/glm-5p2", "https://proxy/v1", "sk-oa"),
    ),
    # ... nor for a plain openai/<model> whose inner part is not a provider string.
    (
        "r4_not_for_plain_openai_prefix",
        {"OPENAI_API_KEY": "sk-oa"},
        dict(model="openai/gpt-4.1"),
        ("openai/gpt-4.1", None, "sk-oa"),
    ),
]


@pytest.mark.parametrize("env,kwargs,expected", [c[1:] for c in _FRONTIER_CASES], ids=[c[0] for c in _FRONTIER_CASES])
def test_frontier_routing(monkeypatch, env, kwargs, expected):
    pytest.importorskip("litellm")  # rule 2/4 classification reads litellm.provider_list
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = make_analyzer("frontier", **kwargs)
    assert _target(a) == expected
    assert a.label == "frontier"


def test_frontier_without_model_builds_lazily(monkeypatch):
    """Construction must still succeed (callers build analyzers they may never call)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oa")
    a = make_analyzer("frontier")
    assert a.model is None


# --------------------------------------------------------------------------------------------------
# policy / endpoint
# --------------------------------------------------------------------------------------------------
def test_policy_never_falls_back_to_openai_base(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://compat/v1")
    with pytest.raises(AnalyzerConfigError, match="base_url"):
        make_analyzer("policy", model="Qwen/Qwen3-4B")


def test_policy_explicit_base_and_placeholder_key():
    a = make_analyzer("policy", model="Qwen/Qwen3-4B", base_url="http://127.0.0.1:8000/v1")
    assert _target(a) == ("openai/Qwen/Qwen3-4B", "http://127.0.0.1:8000/v1", "EMPTY")


def test_endpoint_falls_back_to_openai_base(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://compat/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oa")
    a = make_analyzer("endpoint", model="ema-copy")
    assert _target(a) == ("openai/ema-copy", "https://compat/v1", "sk-oa")


def test_endpoint_without_any_base_raises():
    with pytest.raises(AnalyzerConfigError, match="OPENAI_BASE_URL"):
        make_analyzer("endpoint", model="ema-copy")


@pytest.mark.parametrize("backend", ["policy", "endpoint"])
def test_policy_endpoint_require_model(backend):
    with pytest.raises(AnalyzerConfigError, match="model"):
        make_analyzer(backend, base_url="http://x/v1")


def test_unknown_backend_is_config_error():
    with pytest.raises(AnalyzerConfigError):
        make_analyzer("bogus", model="m")


def test_config_error_is_a_value_error():
    """Existing `except ValueError` / pytest.raises(ValueError) sites must keep catching it."""
    assert issubclass(AnalyzerConfigError, ValueError)


# --------------------------------------------------------------------------------------------------
# the lazy missing-model check
# --------------------------------------------------------------------------------------------------
def test_analyze_without_model_raises_before_touching_litellm(monkeypatch):
    # sys.modules[name] = None makes `import litellm` raise ImportError: if analyze() reached the import,
    # this test would see ImportError, not AnalyzerConfigError.
    monkeypatch.setitem(sys.modules, "litellm", None)
    a = make_analyzer("frontier")
    with pytest.raises(AnalyzerConfigError, match="OPSD_PI_ANALYZER_MODEL"):
        asyncio.run(a.analyze("hi"))
    snap = a.usage_snapshot()
    assert snap["calls"] == 0 and snap["errors"] == 0  # no call was attempted


# --------------------------------------------------------------------------------------------------
# build_hints: a config error aborts the build (no artifact), a per-call error does not
# --------------------------------------------------------------------------------------------------
def _write_dump(path):
    recs = [
        {
            "task_id": "t1",
            "won": True,
            "num_turns": 1,
            "messages": [
                {"role": "user", "content": "Your task is to: put a mug on the desk."},
                {"role": "assistant", "content": "go to desk 1"},
            ],
        },
        {
            "task_id": "t2",
            "won": False,
            "num_turns": 1,
            "messages": [
                {"role": "user", "content": "Your task is to: heat an egg."},
                {"role": "assistant", "content": "go to fridge 1"},
            ],
        },
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_build_hints_missing_model_exits_nonzero_and_writes_nothing(tmp_path, monkeypatch):
    from opsd.pi import build_hints

    dump, out = tmp_path / "traj.jsonl", tmp_path / "hints.json"
    _write_dump(dump)
    monkeypatch.setattr(
        sys, "argv", ["build_hints", "--pi-type", "onpolicy_skill", "--trajectories", str(dump), "--out", str(out)]
    )
    with pytest.raises(SystemExit) as ei:
        build_hints.main()
    assert ei.value.code not in (0, None)
    assert "analyzer misconfigured" in str(ei.value.code)
    assert not out.exists()
    assert not out.with_suffix(".summary.json").exists()


_SKILL_JSON = json.dumps(
    {"golden_workflow": ["go to the desk", "put the mug down"], "success_analysis": "found the desk first"}
)


def test_build_hints_per_call_errors_are_still_swallowed(tmp_path, monkeypatch):
    """The per-task isolation is unchanged: one task's ordinary analyzer failure is counted, not fatal."""
    from opsd.pi import build_hints

    async def _flaky(self, prompt, **kwargs):
        if "heat an egg" in prompt:  # t2 only
            raise RuntimeError("upstream 500")
        return _SKILL_JSON

    monkeypatch.setattr(az.Analyzer, "analyze", _flaky, raising=True)
    dump, out = tmp_path / "traj.jsonl", tmp_path / "hints.json"
    _write_dump(dump)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_hints",
            "--pi-type",
            "onpolicy_skill",
            "--trajectories",
            str(dump),
            "--out",
            str(out),
            "--analyzer-model",
            "m",
            "--analyzer-base-url",
            "http://x/v1",
        ],
    )
    build_hints.main()
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["n_tasks"] == 2 and summary["n_hints"] == 1 and summary["error"] == 1
    assert "all_failed" not in summary
    art = json.loads(out.read_text())
    assert set(art) == {"__schema__", "t1"}


def _all_fail_argv(dump, out, *extra):
    return [
        "build_hints",
        "--pi-type",
        "onpolicy_skill",
        "--trajectories",
        str(dump),
        "--out",
        str(out),
        "--analyzer-model",
        "m",
        "--analyzer-base-url",
        "http://x/v1",
        *extra,
    ]


def test_build_hints_every_call_failing_exits_nonzero_without_an_artifact(tmp_path, monkeypatch):
    """Every task erroring (e.g. a 401 from a wrong key) must not look like a successful empty build."""
    from opsd.pi import build_hints

    async def _boom(self, prompt, **kwargs):
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr(az.Analyzer, "analyze", _boom, raising=True)
    dump, out = tmp_path / "traj.jsonl", tmp_path / "not_yet_created" / "hints.json"
    _write_dump(dump)
    monkeypatch.setattr(sys, "argv", _all_fail_argv(dump, out))
    with pytest.raises(SystemExit) as ei:
        build_hints.main()
    assert ei.value.code not in (0, None)
    assert "every analyzer call failed" in str(ei.value.code)
    assert not out.exists()
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["n_hints"] == 0 and summary["error"] == summary["n_tasks"] == 2
    assert summary["all_failed"] is True and summary["artifact_written"] is False


def test_build_hints_allow_empty_writes_the_empty_artifact(tmp_path, monkeypatch):
    from opsd.pi import build_hints

    async def _boom(self, prompt, **kwargs):
        raise RuntimeError("upstream 500")

    monkeypatch.setattr(az.Analyzer, "analyze", _boom, raising=True)
    dump, out = tmp_path / "traj.jsonl", tmp_path / "hints.json"
    _write_dump(dump)
    monkeypatch.setattr(sys, "argv", _all_fail_argv(dump, out, "--allow-empty"))
    build_hints.main()
    assert json.loads(out.read_text()) == {"__schema__": "opsd_hints/v2"}
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["all_failed"] is True and summary["artifact_written"] is True


def test_build_hints_forwards_api_key_env(tmp_path, monkeypatch):
    from opsd.pi import build_hints

    seen = {}
    real = build_hints.make_analyzer

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    async def _canned(self, prompt, **kwargs):
        return "{}"

    monkeypatch.setattr(build_hints, "make_analyzer", _spy)
    monkeypatch.setattr(az.Analyzer, "analyze", _canned, raising=True)
    monkeypatch.setenv("MY_ANALYZER_KEY", "sk-mine")
    dump, out = tmp_path / "traj.jsonl", tmp_path / "hints.json"
    _write_dump(dump)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_hints",
            "--pi-type",
            "onpolicy_skill",
            "--trajectories",
            str(dump),
            "--out",
            str(out),
            "--analyzer-model",
            "m",
            "--analyzer-base-url",
            "http://x/v1",
            "--analyzer-api-key-env",
            "MY_ANALYZER_KEY",
        ],
    )
    build_hints.main()
    assert seen["api_key_env"] == "MY_ANALYZER_KEY"


# --------------------------------------------------------------------------------------------------
# online manager: a config error propagates even with strict off; a per-call error still does not
# --------------------------------------------------------------------------------------------------
def _group():
    from opsd.pi.online import _Group

    views = [
        TrajView(
            task_id="t1",
            messages=[
                {"role": "user", "content": "Your task is to: put a mug on the desk."},
                {"role": "assistant", "content": "go to desk 1"},
            ],
            reward=1.0,
            success=True,
            num_turns=1,
            repetition_id=0,
        ),
        TrajView(
            task_id="t1",
            messages=[
                {"role": "user", "content": "Your task is to: put a mug on the desk."},
                {"role": "assistant", "content": "go to shelf 1"},
            ],
            reward=0.0,
            success=False,
            num_turns=1,
            repetition_id=1,
        ),
    ]
    return _Group(key="t1", task_id="t1", views=views)


def test_online_missing_model_raises_even_when_not_strict():
    from opsd.pi.online import OnlineHintConfig, OnlineHintManager

    cfg = OnlineHintConfig(enabled=True, pi_type="contrastive", key_mode="task_id", strict=False)
    mgr = OnlineHintManager(cfg, tokenizer=None)  # still constructs without a model (lazy check)
    with pytest.raises(AnalyzerConfigError):
        asyncio.run(mgr._refine_one(_group()))


def test_online_per_call_error_still_swallowed_when_not_strict(monkeypatch):
    from opsd.pi.online import OnlineHintConfig, OnlineHintManager

    async def _boom(self, prompt, **kwargs):
        raise RuntimeError("upstream 500")

    monkeypatch.setattr(az.Analyzer, "analyze", _boom, raising=True)
    cfg = OnlineHintConfig(
        enabled=True,
        pi_type="contrastive",
        key_mode="task_id",
        strict=False,
        analyzer_model="m",
        analyzer_base_url="http://x/v1",
    )
    mgr = OnlineHintManager(cfg, tokenizer=None)
    assert asyncio.run(mgr._refine_one(_group())) == "error"

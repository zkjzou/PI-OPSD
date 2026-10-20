"""Online OPSD hints: an analyzer CONFIGURATION error must stop the run, with OPSD_ONLINE_HINTS_STRICT unset.

``SelfDistillOnlineSkillGenerator._resolve_hints`` swallows per-step analyzer failures unless strict is on,
so one bad call never kills a training step. A missing analyzer model is different: every call fails the
same way, so swallowing ``AnalyzerConfigError`` would train the whole run on fallback hints (or none) and
log no ``online_hints/*`` metrics. These tests pin that it propagates whatever strict says, and that the
generator refuses to construct when the online pi_type needs an analyzer and no model is configured.

No Ray, GPUs or network: the generator's SkyRL base ``__init__`` is stubbed, the online manager is real.
"""

import asyncio
from types import SimpleNamespace

import pytest

mod = pytest.importorskip("opsd.trainer.main_opsd_alfworld")

import opsd.pi.analyzer as az  # noqa: E402
from opsd.pi.analyzer import AnalyzerConfigError  # noqa: E402
from opsd.pi.online import OnlineHintConfig, OnlineHintManager  # noqa: E402

_ENV = (
    "OPSD_ONLINE_HINTS_STRICT",
    "OPSD_ONLINE_HINTS_PI_TYPE",
    "OPSD_ONLINE_HINTS_REFERENCE_MODE",
    "OPSD_ONLINE_HINTS_ANALYZER_BACKEND",
    "OPSD_ONLINE_HINTS_ANALYZER_MODEL",
    "OPSD_ONLINE_HINTS_ANALYZER_BASE_URL",
    "OPSD_ONLINE_HINTS_ANALYZER_API_KEY",
    "OPSD_ONLINE_HINTS_FALLBACK_PATH",
    "OPSD_ONLINE_HINTS_DUMP",
    "OPSD_ONLINE_HINTS_SNAPSHOT_DIR",
    "OPSD_PI_ANALYZER_MODEL",
    "OPSD_PI_ANALYZER_BASE_URL",
    "OPSD_PI_ANALYZER_API_KEY",
    "OPSD_PI_ANALYZER_API_KEY_ENV",
    "OPENAI_BASE_URL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


class _Tok:
    """Just enough tokenizer for the manager to rebuild TrajViews from token ids."""

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{i}" for i in ids)


def _batch(phase: str = "train"):
    generator_output = {
        "prompt_token_ids": [[1, 2], [1, 2]],
        "response_ids": [[3, 4], [5, 6]],
        "loss_masks": [[1, 1], [1, 1]],
        "rewards": [1.0, 0.0],
    }
    env_extras = [{"task_id": "t1"}, {"task_id": "t1"}]
    input_batch = {"batch_metadata": SimpleNamespace(training_phase=phase, global_step=1), "uids": ["u1", "u1"]}
    return generator_output, env_extras, input_batch


def _gen_with(mgr):
    """A SelfDistillOnlineSkillGenerator holding only what ``_resolve_hints`` reads."""
    g = object.__new__(mod.SelfDistillOnlineSkillGenerator)
    g._online_mgr = mgr
    g._n_samples = 2
    g._last_online_metrics = {"stale": 1.0}
    return g


def _real_mgr(**cfg_kwargs):
    cfg = OnlineHintConfig(enabled=True, key_mode="task_id", strict=False, **cfg_kwargs)
    mgr = OnlineHintManager(cfg, tokenizer=None)
    mgr.tokenizer = _Tok()
    return mgr


# --------------------------------------------------------------------------------------------------
# _resolve_hints: config errors propagate with strict unset; ordinary failures keep the old behaviour
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("strict", [None, "0"])
def test_missing_model_propagates_through_resolve_hints_without_strict(monkeypatch, strict):
    if strict is not None:
        monkeypatch.setenv("OPSD_ONLINE_HINTS_STRICT", strict)
    g = _gen_with(_real_mgr(pi_type="onpolicy_skill"))  # frontier, no model: analyze() raises on first call
    with pytest.raises(AnalyzerConfigError, match="no analyzer model"):
        asyncio.run(g._resolve_hints(*_batch()))


class _StubMgr:
    def __init__(self, exc):
        self._exc = exc
        self.cfg = SimpleNamespace(inject_eval=False)

    async def update(self, *args, **kwargs):
        raise self._exc

    def meta_for(self, env_extras, uid, rep):
        return {"source": "fallback"}

    def hint_for(self, env_extras, uid, rep):
        return "prior hint"


def test_config_error_from_update_is_reraised_without_strict():
    g = _gen_with(_StubMgr(AnalyzerConfigError("backend='endpoint' requires a base_url")))
    with pytest.raises(AnalyzerConfigError, match="base_url"):
        asyncio.run(g._resolve_hints(*_batch()))


def test_other_update_errors_are_still_swallowed_without_strict():
    g = _gen_with(_StubMgr(RuntimeError("analyzer endpoint unreachable")))
    hints = asyncio.run(g._resolve_hints(*_batch()))
    assert hints == ["prior hint", "prior hint"]
    assert g._last_online_metrics == {}


def test_other_update_errors_raise_under_strict(monkeypatch):
    monkeypatch.setenv("OPSD_ONLINE_HINTS_STRICT", "1")
    g = _gen_with(_StubMgr(RuntimeError("analyzer endpoint unreachable")))
    with pytest.raises(RuntimeError, match="unreachable"):
        asyncio.run(g._resolve_hints(*_batch()))


def test_per_call_errors_are_counted_not_raised(monkeypatch):
    async def _boom(self, prompt, **kwargs):
        raise RuntimeError("upstream 500")

    monkeypatch.setattr(az.Analyzer, "analyze", _boom, raising=True)
    g = _gen_with(_real_mgr(pi_type="onpolicy_skill", analyzer_model="m", analyzer_base_url="http://127.0.0.1:9/v1"))
    hints = asyncio.run(g._resolve_hints(*_batch()))
    assert hints == ["", ""]
    assert g._last_online_metrics["online_hints/refine_errors"] == 1.0


def test_eval_phase_never_reaches_the_analyzer():
    g = _gen_with(_StubMgr(AnalyzerConfigError("must not be reached at eval")))
    assert asyncio.run(g._resolve_hints(*_batch(phase="eval"))) == ["", ""]


# --------------------------------------------------------------------------------------------------
# construction-time check
# --------------------------------------------------------------------------------------------------
@pytest.fixture
def build(monkeypatch):
    """Construct the online generator with the SkyRL base __init__ stubbed out."""

    def _base_init(self, *args, **kwargs):
        self.tokenizer = None
        self.generator_cfg = SimpleNamespace(n_samples_per_prompt=2)

    monkeypatch.setattr(mod.SelfDistillSkyRLGymGenerator, "__init__", _base_init)

    def _build(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return mod.SelfDistillOnlineSkillGenerator()

    return _build


@pytest.mark.parametrize(
    "env",
    [
        {"OPSD_ONLINE_HINTS_PI_TYPE": "onpolicy_skill"},
        {"OPSD_ONLINE_HINTS_PI_TYPE": "contrastive"},
        {"OPSD_ONLINE_HINTS_PI_TYPE": "prompt_skill"},
        {"OPSD_ONLINE_HINTS_PI_TYPE": "reference", "OPSD_ONLINE_HINTS_REFERENCE_MODE": "abstracted"},
    ],
)
def test_analyzer_pi_type_without_model_fails_at_construction(build, env):
    with pytest.raises(AnalyzerConfigError, match="OPSD_ONLINE_HINTS_ANALYZER_MODEL"):
        build(**env)


@pytest.mark.parametrize(
    "env",
    [
        # the golden online reference arms set no analyzer model
        {"OPSD_ONLINE_HINTS_PI_TYPE": "reference", "OPSD_ONLINE_HINTS_REFERENCE_MODE": "actions"},
        {"OPSD_ONLINE_HINTS_PI_TYPE": "reference", "OPSD_ONLINE_HINTS_REFERENCE_MODE": "trajectory"},
        {"OPSD_ONLINE_HINTS_PI_TYPE": "reference"},  # default mode = actions
        {"OPSD_ONLINE_HINTS_PI_TYPE": "static"},
    ],
)
def test_analyzer_free_pi_types_construct_without_model(build, env):
    g = build(**env)
    assert not mod._online_pi_needs_analyzer(g._online_mgr.cfg.pi_type, g._online_mgr._constructor)
    assert g._online_mgr._analyzer.model is None


@pytest.mark.parametrize(
    "env",
    [
        {
            "OPSD_ONLINE_HINTS_PI_TYPE": "onpolicy_skill",
            "OPSD_ONLINE_HINTS_ANALYZER_MODEL": "some-model",
            "OPSD_ONLINE_HINTS_ANALYZER_BASE_URL": "http://127.0.0.1:9/v1",
        },
        {
            "OPSD_ONLINE_HINTS_PI_TYPE": "onpolicy_skill",
            "OPSD_PI_ANALYZER_MODEL": "some-model",
            "OPSD_PI_ANALYZER_BASE_URL": "http://127.0.0.1:9/v1",
        },
    ],
)
def test_analyzer_pi_type_with_model_constructs(build, env):
    g = build(**env)
    assert g._online_mgr._analyzer.model == "openai/some-model"
    assert g._n_samples == 2

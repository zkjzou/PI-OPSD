"""Fast, offline unit tests for the τ² integration's load-bearing invariants.

None of these need the τ² checkout, a GPU, or the network — they guard the pure-logic seams that, if they
regress, would silently corrupt a run (uid collisions in GRPO, dropped sampling params, mis-bucketed cost,
an infra failure counted as a policy failure).

    python -m opsd.tau2_bench.tests.test_units      # plain runner
    pytest opsd/tau2_bench/tests/test_units.py
"""

from __future__ import annotations

import pathlib

from opsd.tau2_bench.agent import PolicySpec, UserSpec, is_local_model_endpoint
from opsd.tau2_bench.episode import (
    INFRASTRUCTURE_STATUSES,
    _classify_exception,
    derive_seed,
)
from opsd.tau2_bench.manifest import task_digest
from opsd.tau2_bench.metering import _LEG_BY_CALL_NAME, _leg_for


def _repo_root() -> pathlib.Path:
    """The repository root, found by walking up (``opsd._repo``) -- never by a fixed ``parents[N]``,
    whose depth changes when this package moves."""
    from opsd._repo import repo_root

    return repo_root(__file__)


# --- agent.py: the two empirically-found endpoint requirements ------------------------------------


def test_local_endpoint_detection():
    assert is_local_model_endpoint("http://127.0.0.1:8101/v1")
    assert is_local_model_endpoint("http://localhost:8000/v1")
    assert not is_local_model_endpoint("https://llm-proxy.example/v1")


def test_policy_vllm_params_only_for_local_endpoint():
    """vLLM-only knobs ride in extra_body for a local engine, and are omitted for a hosted gateway."""

    local = PolicySpec(
        served_model="Qwen3-4B-Thinking-2507",
        base_url="http://127.0.0.1:8101/v1",
        top_k=20,
        min_p=0.0,
        enable_thinking=True,
    )
    largs = local.llm_args()
    extra = largs["extra_body"]
    assert extra["top_k"] == 20 and extra["min_p"] == 0.0
    assert extra["chat_template_kwargs"] == {"enable_thinking": True}

    remote = PolicySpec(
        served_model="fireworks_ai/deepseek-v4-flash",
        base_url="https://llm-proxy.example/v1",
        top_k=20,
        min_p=0.0,
        enable_thinking=True,
    )
    # No auto extra_body => the gateway never sees chat_template_kwargs/top_k/min_p (which it 400s on).
    assert "extra_body" not in remote.llm_args()


def test_explicit_extra_body_always_passes_through():
    """The escape hatch for a remote vLLM that *does* accept extra params."""

    remote = PolicySpec(served_model="m", base_url="https://gw/v1", extra_body={"foo": 1})
    assert remote.llm_args()["extra_body"] == {"foo": 1}


def test_policy_model_prefix():
    assert PolicySpec(served_model="Qwen3", base_url="x").litellm_model == "openai/Qwen3"


def test_user_model_requires_openai_prefix():
    """A bare provider-segment name routes to that provider and fails auth; the prefix forces base_url."""

    assert UserSpec(model="fireworks_ai/deepseek-v4-flash").litellm_model == "openai/fireworks_ai/deepseek-v4-flash"
    assert UserSpec(model="openai/already").litellm_model == "openai/already"


def test_llm_base_url_default_resolves_at_runtime_from_env():
    """Simulator/judge endpoints default to $OPENAI_BASE_URL, else the public OpenAI API -- read when
    the spec/config is BUILT (not at import), so the egress allowlist sees the URL actually called."""

    import os

    from opsd.tau2_bench.constants import DEFAULT_LLM_BASE_URL, default_llm_base_url

    saved = os.environ.pop("OPENAI_BASE_URL", None)
    try:
        assert default_llm_base_url() == DEFAULT_LLM_BASE_URL == "https://api.openai.com/v1"
        assert UserSpec().base_url == DEFAULT_LLM_BASE_URL
        os.environ["OPENAI_BASE_URL"] = "https://llm-proxy.example/v1"
        assert default_llm_base_url() == "https://llm-proxy.example/v1"
        assert UserSpec().base_url == "https://llm-proxy.example/v1", "default must not be frozen at import"
        os.environ["OPENAI_BASE_URL"] = "  "
        assert default_llm_base_url() == DEFAULT_LLM_BASE_URL, "blank env must fall back"
    finally:
        if saved is None:
            os.environ.pop("OPENAI_BASE_URL", None)
        else:
            os.environ["OPENAI_BASE_URL"] = saved


def test_leg_metadata_tags_present():
    """Cost metering depends on these — tau2's own call_name never reaches litellm."""

    assert PolicySpec(served_model="m", base_url="x").llm_args()["metadata"] == {"call_name": "agent_response"}
    assert UserSpec().llm_args()["metadata"] == {"call_name": "user_response"}


# --- manifest.py: GRPO uid safety -----------------------------------------------------------------


def test_task_digest_is_deterministic_and_collision_free_on_colons():
    tricky = "[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]"
    d1 = task_digest(tricky)
    d2 = task_digest(tricky)
    assert d1 == d2 and len(d1) == 10 and ":" not in d1
    assert task_digest("a") != task_digest("b")


def test_uid_has_no_colon():
    """A ':' in a uid breaks GRPO's `<uid>:repeat-N` suffix split — telecom ids are full of them."""

    uid = f"tau2-telecom-base-0003-{task_digest('[x]a:b|c[PERSONA:None]')}"
    assert ":" not in uid


# --- episode.py: failure taxonomy -----------------------------------------------------------------


def test_classify_exception_maps_known_failures():
    class EgressNotAllowedError(Exception):
        pass

    status, _ = _classify_exception(EgressNotAllowedError("blocked api.openai.com"))
    assert status == "egress_blocked"
    status, _ = _classify_exception(Exception("This model's maximum context length is 4096"))
    assert status == "policy_context_overflow"


def test_infrastructure_statuses_are_zeroed_in_phase2():
    """These must never contribute gradient — they are our failures, not the policy's."""

    assert "egress_blocked" in INFRASTRUCTURE_STATUSES
    assert "policy_context_overflow" in INFRASTRUCTURE_STATUSES
    # a normal terminated_* status is NOT infrastructure
    assert "terminated_max_steps" not in INFRASTRUCTURE_STATUSES


def test_derive_seed_is_stable_and_string_id_safe():
    tricky = "[x]a:b|c[PERSONA:None]"
    s1 = derive_seed(0, "telecom", tricky, 0)
    s2 = derive_seed(0, "telecom", tricky, 0)
    assert s1 == s2 and 0 <= s1 <= 0x7FFFFFFF
    assert derive_seed(0, "telecom", tricky, 0) != derive_seed(0, "telecom", tricky, 1)


# --- metering.py: leg attribution -----------------------------------------------------------------


def test_leg_mapping_covers_all_three_legs_incl_judge():
    assert _LEG_BY_CALL_NAME["agent_response"] == "agent"
    assert _LEG_BY_CALL_NAME["user_response"] == "user"
    # the judge's real upstream call_name is nl_assertions_eval, not the guessed nl_assertions
    assert _LEG_BY_CALL_NAME["nl_assertions_eval"] == "judge"


def test_leg_for_reads_litellm_params_metadata():
    kwargs = {"litellm_params": {"metadata": {"call_name": "agent_response"}}}
    assert _leg_for(kwargs) == "agent"
    assert _leg_for({}) == "other"


# --- source.py: staged-tree provenance (no git) --------------------------------------------------


def test_verify_checkout_accepts_staged_tree_with_marker(tmp_path=None):
    """A tarball-extracted tree (no .git) must verify via the .tau2_revision marker."""

    import tempfile
    from pathlib import Path

    from opsd.tau2_bench.constants import TAU2_REVISION
    from opsd.tau2_bench.source import (
        REVISION_MARKER,
        SourceCheckoutError,
        verify_checkout,
    )

    root = Path(tmp_path or tempfile.mkdtemp())
    (root / "src" / "tau2").mkdir(parents=True)
    (root / "src" / "tau2" / "registry.py").write_text("# stub\n")

    # no marker -> rejected
    try:
        verify_checkout(root)
        raise AssertionError("expected rejection with no marker")
    except SourceCheckoutError:
        pass

    # wrong revision -> rejected
    (root / REVISION_MARKER).write_text("deadbeef\n")
    try:
        verify_checkout(root)
        raise AssertionError("expected rejection for wrong revision")
    except SourceCheckoutError:
        pass

    # correct revision -> accepted
    (root / REVISION_MARKER).write_text(TAU2_REVISION + "\n")
    assert verify_checkout(root).resolve() == root.resolve()


def test_bootstrap_names_the_missing_tau2_deps(tmp_path=None):
    """Without τ²'s extra deps the first τ² import must fail with the install command, not a bare
    ModuleNotFoundError; any other missing module is re-raised untouched."""

    import importlib
    import os
    import sys
    import tempfile
    from pathlib import Path

    from opsd.tau2_bench import source

    root = Path(tmp_path or tempfile.mkdtemp())
    real_import = importlib.import_module
    missing = {"name": "addict"}

    def fake_import(name, *args, **kwargs):
        if name == "tau2.utils.llm_utils":
            raise ModuleNotFoundError(f"No module named '{missing['name']}'", name=missing["name"])
        return real_import(name, *args, **kwargs)

    saved_verify, saved_path = source.verify_checkout, list(sys.path)
    saved_env = os.environ.get("TAU2_DATA_DIR")
    source.verify_checkout = lambda _source_dir: root
    importlib.import_module = fake_import
    try:
        for dep in ("addict", "deepdiff.helper", "toml"):
            missing["name"] = dep
            try:
                source.bootstrap_tau2(root)
                raise AssertionError("expected ModuleNotFoundError")
            except ModuleNotFoundError as exc:
                assert exc.name == dep
                assert "--extra tau2" in str(exc) and "deepdiff==9.1.0" in str(exc), str(exc)
                assert "toml==0.10.2" in str(exc), str(exc)

        missing["name"] = "some_other_module"
        try:
            source.bootstrap_tau2(root)
            raise AssertionError("expected ModuleNotFoundError")
        except ModuleNotFoundError as exc:
            assert str(exc) == "No module named 'some_other_module'"
    finally:
        importlib.import_module = real_import
        source.verify_checkout = saved_verify
        sys.path[:] = saved_path
        if saved_env is None:
            os.environ.pop("TAU2_DATA_DIR", None)
        else:
            os.environ["TAU2_DATA_DIR"] = saved_env


# --- baseline.py: resume treats infra failures as NOT done (retry-on-rerun) ------------------------


def test_load_done_retries_infra_failures():
    """Only terminated episodes count as done; an infra failure must be retried on re-run."""

    import json
    import tempfile
    from pathlib import Path

    from opsd.tau2_bench.baseline import _load_done

    p = Path(tempfile.mktemp(suffix=".jsonl"))
    rows = [
        {"domain": "retail", "task_id": "a", "trial": 0, "status": "terminated_user_stop"},
        {"domain": "retail", "task_id": "b", "trial": 0, "status": "terminated_max_steps"},  # legit 0, done
        {"domain": "retail", "task_id": "c", "trial": 0, "status": "policy_api_error"},  # infra -> retry
        {"domain": "telecom", "task_id": "d", "trial": 1, "status": "egress_blocked"},  # infra -> retry
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    done = _load_done(p)
    assert ("retail", "a", 0) in done
    assert ("retail", "b", 0) in done  # terminated_max_steps is a real result, not retried
    assert ("retail", "c", 0) not in done  # infra failure -> will be retried
    assert ("telecom", "d", 1) not in done
    p.unlink()


# --- areal.py: AReaL RL data loader pure logic (offline, no HF/tau2) -------------------------------


def test_areal_domain_of():
    from opsd.tau2_bench.areal import _domain_of

    assert _domain_of({"db_path": "tau2_rl_database/tau2_retail_new_db_1.json"}) == "retail"
    assert _domain_of({"db_path": "tau2_rl_database/tau2_airline_db.json"}) == "airline"
    assert _domain_of({"db_path": "tau2_rl_database/tau2_telecom_db.toml"}) == "telecom"
    assert _domain_of({"db_path": ""}) == "unknown"


def test_areal_sort_key_is_natural():
    """retail_2 must sort before retail_10 so the manifest task_index order is stable + intuitive."""

    from opsd.tau2_bench.areal import _sort_key

    ids = [{"id": f"retail_{n}"} for n in (10, 2, 1, 100, 3)]
    ordered = [r["id"] for r in sorted(ids, key=_sort_key)]
    assert ordered == ["retail_1", "retail_2", "retail_3", "retail_10", "retail_100"]


def test_areal_split_key_registered():
    from opsd.tau2_bench.constants import AREAL_SPLIT_KEYS, MANIFEST_BY_NAME

    assert "retail-areal-rl" in AREAL_SPLIT_KEYS
    spec = MANIFEST_BY_NAME["retail-areal-rl"]
    assert spec.domain == "retail" and spec.split_name == "retail-areal-rl" and spec.count == 563


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()


# --- preserved thinking: the three invariants that must hold simultaneously -------------------
# (a) the simulated user must NEVER see <think>, (b) the agent's own context MUST keep it,
# (c) the encoded training sequence must render it EXACTLY ONCE.


def test_split_thinking_emits_canonical_shape_and_is_idempotent():
    from opsd.tau2_bench.episode import _split_thinking

    reason, body = "I should look up the order.", "Sure, may I have your id?"
    inlined = {"role": "assistant", "content": f"<think>\n{reason}\n</think>\n\n{body}"}
    once = _split_thinking(dict(inlined))
    # content is what the user sim may read -> must be clean; reasoning moves to its own field
    assert once["content"] == body
    assert once["reasoning_content"] == reason
    assert "<think>" not in once["content"]
    # idempotent, so it is safe wherever it runs in the chain
    assert _split_thinking(dict(once)) == once
    # no-thinking turns (tau2's hard-coded greeting) are untouched
    plain = {"role": "assistant", "content": "Hi! How can I help you today?"}
    assert _split_thinking(dict(plain)) == plain


def test_preserve_template_renders_thinking_exactly_once_on_every_path():
    """Guards the double-render bug: reasoning_content AND inline <think> must not both survive.

    The template only strips <think> out of content in its ``else`` arm, so an entry carrying both
    fields renders the reasoning twice -- log-probs over a stream the policy never produced.
    """
    import jinja2

    from opsd.tau2_bench.episode import _split_thinking

    root = _repo_root()
    tpl_path = root / "opsd/alfworld/qwen3thinking_preserve.jinja2"
    if not tpl_path.exists():  # template lives with the alfworld example; skip if absent
        return
    env = jinja2.Environment(undefined=jinja2.ChainableUndefined)
    env.policies["json.dumps_kwargs"] = {}
    tpl = env.from_string(tpl_path.read_text())

    reason, body = "I should look up the order.", "Sure, may I have your id?"
    inline = f"<think>\n{reason}\n</think>\n\n{body}"
    paths = [
        {"role": "assistant", "content": inline, "reasoning_content": reason},  # training
        {"role": "assistant", "content": body, "reasoning_content": reason},  # eval
        {"role": "assistant", "content": inline},  # parser-off
    ]
    renders = set()
    for msg in paths:
        out = tpl.render(
            messages=[{"role": "system", "content": "sys"}, _split_thinking(dict(msg))],
            add_generation_prompt=False,
            tools=None,
        )
        assert out.count("<think>") == 1, f"thinking rendered {out.count('<think>')}x for {msg}"
        assert out.count(reason) == 1
        renders.add(out)
    assert len(renders) == 1, "the three paths must produce a byte-identical render"


def test_think_leak_metric_reads_tau2_messages_not_the_encoder_output():
    """The guard must stay honest after ``_split_thinking`` scrubs the encoder's copy.

    A leak lives in τ²'s OWN message objects (``result.messages``) — that is what ``flip_roles``
    hands the user simulator.  ``response_messages`` is our encoder output, which ``_split_thinking``
    always leaves clean, so a metric measuring *it* reports 0.0 during a real leak.

    Asserted on the source text rather than by calling the function: importing ``generator`` drags in
    SkyRL/vLLM/Ray (tens of seconds, needs a GPU build), which no unit test should require.
    """
    src = (pathlib.Path(__file__).resolve().parents[1] / "generator.py").read_text()
    body = src.split("def _think_leak_fraction(", 1)[1].split("\ndef ", 1)[0]
    loop = [ln for ln in body.splitlines() if ln.lstrip().startswith("for msg in ")]
    assert loop, "could not find the per-message loop in _think_leak_fraction"
    assert "e.result.messages" in loop[0], f"the leak guard must iterate tau2's own messages, got: {loop[0].strip()}"
    assert "response_messages" not in loop[0], "measuring the encoder output makes the leak guard vacuous"


def test_rollout_record_renders_with_the_shared_trajectory_renderer():
    """The dump records must satisfy the ALFWorld renderer's contract (used for the W&B tables).

    Two τ²-specific traps: the renderer's ``_success`` fallback is ALFWorld's ``score >= 10``, which
    would mark EVERY τ² episode [FAIL] since τ² rewards are 0/1 — so records must carry
    ``env_metrics["success"]``, which ``_success`` consults first; and the trajectory text must be the
    decoded chat sequence so ``_TURN_RE`` can split it into turns.
    """
    from opsd.alfworld.render_trajectories_html import _render_traj, _success

    def rec(reward):
        return {
            "task_id": "retail_task_17",
            "domain": "retail",
            "data_source": "retail",
            "raw_reward": reward,
            "reward_sum": reward,
            "termination": "USER_STOP",
            "stop_reason": "stop",
            "n_response_tokens": 20273,
            "env_metrics": {"success": bool(reward > 0), "num_turns": 13, "tool_errors": 0},
            "prompt_text": "<|im_start|>system\nYou are a retail agent.<|im_end|>\n",
            "response_text": (
                "<|im_start|>assistant\n<think>\nVerify identity first.\n</think>\n\n"
                "Could you share your email?<|im_end|>\n"
                "<|im_start|>user\nsam@example.com<|im_end|>\n"
            ),
        }

    solved, failed = rec(1.0), rec(0.0)
    assert _success(solved) is True, "a solved τ² episode must not read as FAIL"
    assert _success(failed) is False
    html_doc = _render_traj(solved, 0)
    assert "[PASS]" in html_doc and "[FAIL]" in _render_traj(failed, 1)
    assert html_doc.count('class="turn') == 3, "system + assistant + user turns must be split out"
    assert 'class="think"' in html_doc, "preserved thinking should be highlighted"
    # the user turn must not contain the agent's reasoning (visual counterpart of think_leak_frac)
    assert "Verify identity first" not in html_doc.split("turn user")[-1]


def test_preservation_stats_track_thinking_reaching_the_agent():
    """`think_preserved_frac` must read 0 when preservation is broken and ~1 when it works.

    This is the guard for the half of the invariant the leak metric cannot see: whether the AGENT's
    own context carries its prior reasoning. Exercised through the real wrapper against a fake tau2
    llm_utils, since the failure mode is "wrapper silently not applied".
    """
    import sys
    import types

    from opsd.tau2_bench import source as tau2_source

    class FakeMsg:
        def __init__(self, role, content, reasoning=None):
            self.role, self.content = role, content
            self.raw_data = {"choices": [{"message": {"reasoning_content": reasoning}}]} if reasoning else None

    msgs = [
        FakeMsg("assistant", "Hi! How can I help?"),  # tau2's greeting: no reasoning
        FakeMsg("user", "return my order"),
        FakeMsg("assistant", "Could you share your email?", "Verify identity first."),
        FakeMsg("assistant", "Looking that up.", "Now query the order."),
    ]

    def to_litellm_messages(ms):
        return [{"role": m.role, "content": m.content} for m in ms]

    fake = types.ModuleType("tau2.utils.llm_utils")
    fake.to_litellm_messages = to_litellm_messages
    saved = {k: sys.modules.get(k) for k in ("tau2", "tau2.utils", "tau2.utils.llm_utils")}
    try:
        sys.modules["tau2"] = types.ModuleType("tau2")
        utils_mod = types.ModuleType("tau2.utils")
        utils_mod.llm_utils = fake
        sys.modules["tau2.utils"] = utils_mod
        sys.modules["tau2.utils.llm_utils"] = fake

        tau2_source.preservation_stats(reset=True)
        # BEFORE installing: nothing is counted -> the metric must read 0 (broken), not 1
        assert tau2_source.preservation_stats()["assistant_msgs"] == 0

        tau2_source.install_thinking_preservation()
        out = fake.to_litellm_messages(msgs)

        # the two generated turns get their reasoning re-inlined; the greeting cannot
        assert out[2]["content"].startswith("<think>\nVerify identity first.\n</think>")
        assert out[3]["content"].startswith("<think>\nNow query the order.\n</think>")
        assert "<think>" not in out[0]["content"]
        assert "<think>" not in out[1]["content"], "user turns must never gain thinking"

        stats = tau2_source.preservation_stats(reset=True)
        # ONLY policy-generated turns count (those with raw_data). The greeting has none, so a
        # healthy run reads exactly 1.0 -- see the user-simulator dilution note in source.py.
        assert stats == {"assistant_msgs": 2, "with_thinking": 2}, stats
        assert stats["with_thinking"] / stats["assistant_msgs"] == 1.0
        # reset really cleared, so the metric is per-generate() and not cumulative
        assert tau2_source.preservation_stats()["assistant_msgs"] == 0
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_preservation_metric_ignores_the_user_simulators_flipped_view():
    """The guard must not be diluted by the user simulator, which shares to_litellm_messages.

    tau2 has ONE call site for to_litellm_messages (llm_utils.generate), used by the agent AND the
    user simulator. In the simulator's request, flip_roles relabels the *user's own* prior turns as
    role "assistant" with no raw_data. Counting those dragged the metric from ~1.0 to ~(N-1)/2N and
    made a healthy run read 0.72 live, which is worse than no guard at all.
    """
    import sys
    import types

    from opsd.tau2_bench import source as tau2_source

    class FakeMsg:
        def __init__(self, role, content, reasoning=None, raw=True):
            self.role, self.content = role, content
            self.raw_data = (
                {"choices": [{"message": {"reasoning_content": reasoning}}]} if reasoning else ({} if raw else None)
            )

    # the user simulator's view: the user's own turns appear as "assistant", carrying no raw_data
    flipped = [
        FakeMsg("assistant", "i want to return my order", raw=False),
        FakeMsg("user", "Could you share your email?", raw=False),
        FakeMsg("assistant", "sam@example.com", raw=False),
    ]

    fake = types.ModuleType("tau2.utils.llm_utils")
    fake.to_litellm_messages = lambda ms: [{"role": m.role, "content": m.content} for m in ms]
    saved = {k: sys.modules.get(k) for k in ("tau2", "tau2.utils", "tau2.utils.llm_utils")}
    try:
        sys.modules["tau2"] = types.ModuleType("tau2")
        utils_mod = types.ModuleType("tau2.utils")
        utils_mod.llm_utils = fake
        sys.modules["tau2.utils"] = utils_mod
        sys.modules["tau2.utils.llm_utils"] = fake

        tau2_source.install_thinking_preservation()
        tau2_source.preservation_stats(reset=True)
        out = fake.to_litellm_messages(flipped)

        assert all("<think>" not in (m["content"] or "") for m in out), "must not touch the sim's view"
        stats = tau2_source.preservation_stats(reset=True)
        assert stats == {
            "assistant_msgs": 0,
            "with_thinking": 0,
        }, f"user-simulator turns must not enter the denominator, got {stats}"
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# --- failure attribution: which LLM leg actually died --------------------------------------------


def test_classify_exception_attributes_the_failing_leg():
    """The three legs raise IDENTICAL exception types, so only the stamped leg can tell them apart.

    Without it, a remote user-simulator outage was filed as `policy_api_error` and
    `simulator_api_error` was unreachable — inverted diagnosis, opposite remedies.
    """
    from opsd.tau2_bench.episode import INFRASTRUCTURE_STATUSES, _classify_exception

    class RateLimitError(Exception):
        pass

    def stamped(leg):
        exc = RateLimitError("429 too many requests")
        if leg:
            exc._tau2_leg = leg
        return exc

    assert _classify_exception(stamped("user"))[0] == "simulator_api_error"
    assert _classify_exception(stamped("judge"))[0] == "judge_api_error"
    assert _classify_exception(stamped("agent"))[0] == "policy_api_error"
    # untagged keeps the old type-based behaviour (no guessing beyond what we knew before)
    assert _classify_exception(stamped(None))[0] == "policy_api_error"
    # every attributed status must still be treated as OUR failure -> zero reward, zero gradient
    for leg in ("user", "judge", "agent"):
        assert _classify_exception(stamped(leg))[0] in INFRASTRUCTURE_STATUSES

    # context overflow stays policy-specific, and only on the policy leg
    class APIError(Exception):
        pass

    overflow = APIError("This model's maximum context length is 40000 tokens")
    overflow._tau2_leg = "agent"
    assert _classify_exception(overflow)[0] == "policy_context_overflow"
    sim_overflow = APIError("This model's maximum context length is 40000 tokens")
    sim_overflow._tau2_leg = "user"
    assert _classify_exception(sim_overflow)[0] == "simulator_api_error"


def test_egress_guard_stamps_the_leg_on_a_failing_call():
    """The guard is the last place that knows the caller; it must tag before re-raising."""
    import sys
    import types

    from opsd.tau2_bench.episode import _classify_exception

    fake_litellm = types.ModuleType("litellm")

    def boom(**kwargs):
        raise RuntimeError("upstream 503")

    fake_litellm.completion = boom
    saved = sys.modules.get("litellm")
    try:
        sys.modules["litellm"] = fake_litellm
        from opsd.tau2_bench.source import install_egress_guard

        install_egress_guard(["https://llm-proxy.example/v1"])
        try:
            fake_litellm.completion(
                api_base="https://llm-proxy.example/v1",
                metadata={"call_name": "user_response"},
            )
            raise AssertionError("expected the call to raise")
        except RuntimeError as exc:
            assert getattr(exc, "_tau2_leg", None) == "user"
            assert _classify_exception(exc)[0] == "simulator_api_error"
    finally:
        if saved is None:
            sys.modules.pop("litellm", None)
        else:
            sys.modules["litellm"] = saved


def test_call_stats_records_volume_latency_and_proxy_retries():
    """Per-leg call volume/latency must be recorded even when nothing fails.

    The gap this closes: a call that fails twice and succeeds on the third attempt raises nothing and
    never reaches the failure counters, yet costs full latency inside the episode. Calls and seconds
    are exact; retries are gateway-side only (our num_retries is consumed inside litellm).
    """
    import sys
    import types

    from opsd.tau2_bench import source as tau2_source

    class Resp:
        def __init__(self, retries=None):
            self._hidden_params = (
                {"additional_headers": {"x-litellm-attempted-retries": retries}} if retries is not None else {}
            )

    fake_litellm = types.ModuleType("litellm")
    calls = {"n": 0}

    def completion(**kwargs):
        calls["n"] += 1
        return Resp(retries=2 if kwargs.get("metadata", {}).get("call_name") == "user_response" else None)

    fake_litellm.completion = completion
    saved = sys.modules.get("litellm")
    try:
        sys.modules["litellm"] = fake_litellm
        tau2_source.call_stats(reset=True)
        tau2_source.install_egress_guard(["https://llm-proxy.example/v1"])

        for _ in range(3):
            fake_litellm.completion(api_base="https://llm-proxy.example/v1", metadata={"call_name": "user_response"})
        fake_litellm.completion(api_base="https://llm-proxy.example/v1", metadata={"call_name": "agent_response"})

        s = tau2_source.call_stats(reset=True)
        assert s["user"]["calls"] == 3, s
        assert s["agent"]["calls"] == 1, s
        assert s["user"]["retries"] == 6, f"gateway-side retries must accumulate: {s}"
        assert s["agent"]["retries"] == 0
        assert s["user"]["seconds"] >= 0.0
        # reset really cleared -> metrics are per-generate(), not cumulative
        assert tau2_source.call_stats() == {}
    finally:
        if saved is None:
            sys.modules.pop("litellm", None)
        else:
            sys.modules["litellm"] = saved


def test_egress_guard_rebinds_modules_that_imported_completion_by_value():
    """THE regression that made three mechanisms silently inert in production.

    tau2/utils/llm_utils.py does `from litellm import completion` (by value) and bootstrap imports it
    BEFORE installing the guard, so patching litellm.completion never reached the reference tau2 calls.
    Result: the egress guard never fail-closed, failure legs were never stamped (every litellm error
    became policy_api_error), and per-leg call stats recorded nothing. Reproduces the ordering exactly:
    bind by value FIRST, install the guard SECOND, then assert the by-value caller is now guarded.
    """
    import sys
    import types

    from opsd.tau2_bench import source as tau2_source
    from opsd.tau2_bench.episode import _classify_exception

    class RateLimitError(Exception):
        pass

    def original(**kwargs):
        # emulate the user simulator being rate limited by the shared provider
        raise RateLimitError("rate limit exceeded, please try again later")

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = original
    # a tau2 module that grabbed the reference BEFORE the guard exists (llm_utils line 15)
    victim = types.ModuleType("tau2.utils.llm_utils")
    victim.completion = fake_litellm.completion

    saved = {k: sys.modules.get(k) for k in ("litellm", "tau2.utils.llm_utils")}
    try:
        sys.modules["litellm"] = fake_litellm
        sys.modules["tau2.utils.llm_utils"] = victim
        assert victim.completion is original, "precondition: victim holds the unpatched reference"

        tau2_source.install_egress_guard(["https://llm-proxy.example/v1"])

        assert victim.completion is not original, "by-value importer must be rebound to the guard"
        assert getattr(victim.completion, "_tau2_guarded", False)

        # and now a sim-leg failure is attributed to the SIMULATOR, not the policy
        try:
            victim.completion(api_base="https://llm-proxy.example/v1", metadata={"call_name": "user_response"})
            raise AssertionError("expected the call to raise")
        except RateLimitError as exc:
            assert getattr(exc, "_tau2_leg", None) == "user", "leg must be stamped through the victim"
            assert (
                _classify_exception(exc)[0] == "simulator_api_error"
            ), "a user-sim rate limit must NOT be filed as policy_api_error"
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# --- build_tau2_pi_hints: the three PI formats must be non-degenerate --------------------------


def test_tau2_pi_refs_message_folds_reasoning_and_renders_tool_calls():
    """The two τ²-specific gaps, asserted directly on the message converter.

    Without the fold-in, `raw` and `raw_no_think` are byte-identical (τ² keeps reasoning out of band
    as `reasoning_content`). Without the tool_call rendering, `actions` is EMPTY (the shared
    `extract_actions` only reads `content`, and τ² uses native function-calling).
    """
    import json as _json

    from opsd.tau2_bench.build_tau2_pi_hints import _refs_message

    msg = _refs_message(
        {
            "role": "assistant",
            "content": "Let me look that up.",
            "reasoning_content": "Verify identity before acting.",
            "tool_calls": [
                {
                    "id": "c0",
                    "type": "function",
                    "function": {
                        "name": "find_user_id_by_email",
                        "arguments": _json.dumps({"email": "sam@example.com"}),
                    },
                }
            ],
        }
    )
    assert msg["role"] == "assistant"
    assert "<think>\nVerify identity before acting.\n</think>" in msg["content"]
    # arguments must be a real object, not a string of escaped JSON
    assert 'Action: {"arguments": {"email": "sam@example.com"}, "name": "find_user_id_by_email"}' in msg["content"]

    # a tool result becomes a user turn so TrajView.render tags it [OBSERVATION]
    assert _refs_message({"role": "tool", "content": '{"user_id": "sam_802"}'})["role"] == "user"
    # idempotent w.r.t. already-inlined thinking (never double-wrap)
    inlined = _refs_message({"role": "assistant", "content": "<think>\nr\n</think>\n\nbody", "reasoning_content": "r"})
    assert inlined["content"].count("<think>") == 1


def test_tau2_pi_builder_emits_three_distinct_formats(tmp_path=None):
    """End-to-end: τ² trajectories -> hints_{raw,raw_no_think,actions}.json, all non-degenerate."""
    import json as _json
    import tempfile
    from pathlib import Path as _Path

    from opsd.tau2_bench.build_tau2_pi_hints import tau2_traj_to_refs

    root = _Path(tmp_path or tempfile.mkdtemp())
    reason = "Policy says verify identity first."
    won = {
        "domain": "retail",
        "task_id": "retail_537",
        "reward": 1.0,
        "num_agent_turns": 2,
        "prompt_messages": [{"role": "system", "content": "DOMAIN POLICY " + "x" * 3000}],
        "response_messages": [
            {"role": "assistant", "content": "Your email?", "reasoning_content": reason},
            {"role": "tool", "content": '{"user_id": "sam_802"}'},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "return_order", "arguments": '{"order_id": "W1"}'}}],
            },
        ],
    }
    lost = {
        "domain": "retail",
        "task_id": "retail_537",
        "reward": 0.0,
        "num_agent_turns": 1,
        "response_messages": [{"role": "assistant", "content": "I cannot help."}],
    }
    infra = {"domain": "retail", "task_id": "retail_9", "reward": 0.0, "num_agent_turns": 0, "response_messages": []}
    traj = root / "trajectories.jsonl"
    # lost FIRST, so this also proves the won-first re-sort works
    traj.write_text("\n".join(_json.dumps(r) for r in (lost, won, infra)))

    stats = tau2_traj_to_refs(str(traj), str(root / "refs.jsonl"))
    assert stats["n_won_tasks"] == 1, stats
    assert stats["skipped_no_transcript"] == 1, "an infra-failed episode has no transcript to distil"

    from opsd.alfworld.build_pi_hint_formats import build

    build(str(root / "refs.jsonl"), root, 250000)

    def _load(name):
        d = _json.loads((root / name).read_text())
        d.pop("__schema__", None)
        return d

    raw = _load("hints_raw.json")["retail_537"]["task"]
    nothink = _load("hints_raw_no_think.json")["retail_537"]["task"]
    actions = _load("hints_actions.json")["retail_537"]

    assert reason in raw and "<think>" in raw
    assert reason not in nothink, "raw_no_think must drop the reasoning"
    assert raw != nothink, "the two trajectory formats must actually differ"
    assert actions["meta"]["n_actions"] == 1 and "return_order" in actions["task"]
    assert "I cannot help." not in raw, "the LOST trajectory must not become the hint"
    assert "x" * 200 not in raw, "the system prompt is not privileged and must be excluded"


def test_self_hosted_endpoints_are_classified_local_not_just_loopback():
    """Loopback-only detection silently mis-classified EVERY multi-node run.

    On a multi-node cluster the policy is reached via SkyRL's router at http://<node_ip>:port/v1 and a
    self-hosted simulator at http://<sim-host>:8000/v1 — neither is loopback, so top_k / min_p /
    chat_template_kwargs were never injected and rollouts sampled with vLLM defaults. Scheme is the
    discriminator: we serve plain http, hosted gateways are https and reject these fields.
    """
    from opsd.tau2_bench.agent import is_local_model_endpoint

    assert is_local_model_endpoint("http://127.0.0.1:8300/v1")
    assert is_local_model_endpoint("http://192.0.2.10:30000/v1"), "SkyRL router on another node"
    assert is_local_model_endpoint("http://sim-0.example:8000/v1"), "self-hosted sim by DNS name"
    assert not is_local_model_endpoint("https://llm-proxy.example/v1"), "hosted gateway"


def test_user_sim_disables_thinking_only_on_a_self_hosted_engine():
    """The fix for the 53/62 empty-content collapse — and it must not leak to a hosted gateway.

    reasoning_effort="none" is accepted by vLLM as a Literal then IGNORED, because Qwen3 thinking is
    controlled by the chat template. Without enable_thinking=False the simulator spends its whole
    max_tokens budget on <think>, the reasoning parser strips it, content is empty, and tau2 raises
    "UserMessage must have either content or tool_calls".
    """
    from opsd.tau2_bench.agent import UserSpec

    local = UserSpec(model="Qwen/Qwen3.5-35B-A3B", base_url="http://sim-0.example:8000/v1").llm_args()
    assert local["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    # a hosted gateway 400s the whole request on chat_template_kwargs -> must be absent
    assert "extra_body" not in UserSpec(base_url="https://llm-proxy.example/v1").llm_args()

    # explicit extra_body is still the escape hatch for a remote vLLM that does accept them
    esc = UserSpec(base_url="https://remote-vllm/v1", extra_body={"top_k": 20}).llm_args()
    assert esc["extra_body"] == {"top_k": 20}


# --- baseline.py: pass^k must be indexed by TRIAL, never by completion order ----------------------


def test_pass_hat_k_is_ordered_by_trial_not_by_completion_order():
    """pass^1 must read trial 0, whatever order the episodes finished in.

    trajectories.jsonl is appended in COMPLETION order, so a naive per-task `list.append` puts the
    race winner at index 0. That is not a random trial: within a task, the fastest of k episodes is
    disproportionately one that terminated early and failed. On 4B/retail it dragged the reported
    pass^1 to 0.509 while every individual trial index scored ~0.64.
    """
    from opsd.tau2_bench.baseline import summarize

    # task "a" solves on trial 0 only; the failing trials 1-3 are written FIRST (they "finished" first)
    recs = [
        {"domain": "retail", "task_id": "a", "trial": 3, "reward": 0.0},
        {"domain": "retail", "task_id": "a", "trial": 1, "reward": 0.0},
        {"domain": "retail", "task_id": "a", "trial": 0, "reward": 1.0},
        {"domain": "retail", "task_id": "a", "trial": 2, "reward": 0.0},
    ]
    # summarize() returns {domain: ...}; the "per_domain" wrapper is added when summary.json is written
    r = summarize(recs, num_trials=4)["retail"]
    assert r["pass_hat_k"]["1"] == 1.0, "pass^1 read a non-zero trial (completion-order bug is back)"
    assert r["pass_hat_k"]["4"] == 0.0  # order-invariant: never affected by the bug
    assert r["coverage_pass_at_k"]["4"] == 1.0
    assert r["solve_rate_pooled"] == 0.25  # pooled over all 4 episodes, not just trial 0

    # a retried (task, trial) — infra failure re-run — keeps its LAST record, not a 5th sample
    retried = [
        {"domain": "retail", "task_id": "b", "trial": 0, "reward": 0.0},
        {"domain": "retail", "task_id": "b", "trial": 0, "reward": 1.0},
    ]
    rb = summarize(retried, num_trials=1)["retail"]
    assert rb["solve_rate_pooled"] == 1.0 and rb["pass_hat_k"]["1"] == 1.0


# --- generator.py: the trained prompt must carry the tool schemas ----------------------------------


def test_encode_episode_renders_tool_schemas_into_the_trained_prompt():
    """The GRPO prompt must include the tools, because the SERVED prompt does.

    ``apply_chat_template`` was called without ``tools=``, so the policy trained on the domain policy
    alone (retail: 1,492 tok) while vLLM served it policy + 16 tool schemas (4,874 tok) -- 69% of the
    real prompt absent from training. The Qwen3 template renders tools into the system block, so the
    fix is to pass the schemas τ² actually sent.

    The response side must NOT change: ``encode_messages_subset`` subtracts a base prefix rendered
    with the same kwargs, so the tool block cancels exactly.
    """
    from transformers import AutoTokenizer

    from opsd.tau2_bench.episode import Tau2EpisodeResult
    from opsd.tau2_bench.generator import encode_episode

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507", trust_remote_code=True)
    schema = lambda name: {  # noqa: E731 - terse on purpose, this is test scaffolding
        "type": "function",
        "function": {"name": name, "description": f"do {name}", "parameters": {"type": "object", "properties": {}}},
    }
    base = dict(
        domain="retail",
        task_id="t",
        task_index=0,
        trial=0,
        seed=1,
        reward=1.0,
        status="terminated_user_stop",
        termination_reason="USER_STOP",
        num_messages=2,
        num_agent_turns=1,
        num_tool_calls=0,
        num_user_tool_calls=0,
        num_tool_errors=0,
        prompt_messages=[{"role": "system", "content": "You are a retail agent."}],
        response_messages=[{"role": "assistant", "content": "hi"}],
    )
    kw = {"enable_thinking": True}
    without = encode_episode(
        Tau2EpisodeResult(**base, prompt_tools=[]), tok, kw, training=True, max_response_tokens=10_000
    )
    with_tools = encode_episode(
        Tau2EpisodeResult(**base, prompt_tools=[schema("a"), schema("b")]),
        tok,
        kw,
        training=True,
        max_response_tokens=10_000,
    )

    assert len(with_tools.prompt_ids) > len(
        without.prompt_ids
    ), "tool schemas did not reach the trained prompt -- the tools= regression is back"
    # the response encoding is tool-independent, and must stay so
    assert with_tools.response_ids == without.response_ids
    assert with_tools.loss_mask == without.loss_mask


# --- non-thinking policy support -------------------------------------------------------------


def test_policy_enable_thinking_false_reaches_the_engine():
    """``enable_thinking=False`` must ride in extra_body, or a non-thinking policy dies at turn 0.

    vLLM builds Qwen3ReasoningParser PER REQUEST from ``chat_template_kwargs``. Left at True against
    a model that never emits ``</think>``, a plain-text turn comes back as
    (reasoning=<everything>, content=None) and τ² raises "AssistantMessage must have either content
    or tool_calls" — the 88%-episode-loss mode. The flag reaches the parser even though the Instruct
    chat template has no ``enable_thinking`` variable, because vLLM filters unknown keys only in the
    RENDERER, not on the path to the parser.
    """
    from opsd.tau2_bench.agent import PolicySpec

    off = PolicySpec(
        served_model="Qwen/Qwen3-4B-Instruct-2507", base_url="http://127.0.0.1:8100/v1", enable_thinking=False
    ).llm_args()
    assert off["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}

    on = PolicySpec(
        served_model="Qwen/Qwen3-4B-Thinking-2507", base_url="http://127.0.0.1:8100/v1", enable_thinking=True
    ).llm_args()
    assert on["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}

    # a HOSTED gateway 400s on chat_template_kwargs, so our own gate must still drop it there
    assert "extra_body" not in PolicySpec(served_model="x", base_url="https://gw/v1", enable_thinking=False).llm_args()


def test_thinking_config_defaults_are_backwards_compatible():
    """Defaults must reproduce the thinking path exactly; None means 'follow the policy'."""
    from opsd.tau2_bench.main import Tau2Config

    c = Tau2Config()
    assert c.policy_enable_thinking is True
    assert c.preserve_thinking is None, "None => follow; a plain True would strand the metric gate"

    # the follow rule the generator applies
    def resolve(pet, pt):
        return pet if pt is None else bool(pt)

    assert resolve(True, None) is True  # today's behaviour
    assert resolve(False, None) is False  # non-thinking => preservation off automatically
    assert resolve(False, True) is True  # explicit override still wins
    assert resolve(True, False) is False

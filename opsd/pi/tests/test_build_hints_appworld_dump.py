"""Loader regressions for the AppWorld pass@k dumps in build_hints.py.

Each test here pins a bug that a 24-call smoke found AFTER the pipeline reported "16/16 ok, 0 errors,
0 truncated" -- i.e. every one of them is a failure the happy-path counters call success. They are
cheap and they are the only thing standing between a mis-shaped dump and a five-figure token spend on
hints distilled from nothing.

The dump shape is {task_id, sample, reward, turns, conversation, preserve_thinking}: no `won`, no
`env_metrics`, no `repetition_id`, no `messages`.
"""

from __future__ import annotations

from opsd.pi.build_hints import messages_from_dump, trajview_from_dump


def _rec(*, reward=0.0, sample=7, n_turns=3, n_demo=10):
    """An AppWorld-shaped record. `conversation` carries n_demo demo assistant messages BEFORE the
    policy's own n_turns -- the official scaffold's worked example, which the agent did not write."""
    conv = [{"role": "user", "content": "supervisor instruction + worked demo preamble"}]
    for i in range(n_demo):  # demo exchanges
        conv += [{"role": "assistant", "content": f"demo action {i}"}, {"role": "user", "content": f"demo obs {i}"}]
    for i in range(n_turns):  # the policy's own turns
        conv += [{"role": "assistant", "content": f"POLICY action {i}"}, {"role": "user", "content": f"POLICY obs {i}"}]
    return {
        "task_id": "fac291d_3",
        "sample": sample,
        "reward": reward,
        "turns": [{"turn": i + 1, "think": "", "action": f"POLICY action {i}", "obs": ""} for i in range(n_turns)],
        "conversation": conv,
        "preserve_thinking": False,
    }


# --- bug 1: the success threshold is not universal -------------------------------------------------
def test_binary_reward_needs_threshold_1_not_the_alfworld_10():
    """AppWorld REWARD_MODE=binary pays 1.0. At the 10.0 default (ALFWorld's 10*won - 0.1*invalid) a
    real win is labelled FAILURE, so the analyzer is told the wrong outcome and wins-first selection
    picks nothing. Neither `won` nor `env_metrics` is present to rescue it."""
    win = _rec(reward=1.0)
    assert trajview_from_dump(win, success_threshold=10.0).success is False  # the bug
    assert trajview_from_dump(win, success_threshold=1.0).success is True  # the fix
    assert trajview_from_dump(_rec(reward=0.0), success_threshold=1.0).success is False


# --- bug 2: repetition_id is absent; the sample index is called `sample` ---------------------------
def test_repetition_id_falls_back_to_sample_so_per_traj_keys_stay_unique():
    """With rep defaulting to 0, the per-traj key `{task}#rep{n}` collapses to one entry per task and
    a dict keyed on it silently keeps ONE rollout of every 16."""
    tvs = [trajview_from_dump(_rec(sample=s), success_threshold=1.0) for s in range(16)]
    assert [t.repetition_id for t in tvs] == list(range(16))
    keys = {f"{t.task_id}#rep{t.repetition_id}" for t in tvs}
    assert len(keys) == 16, "per-traj keys collapsed -- 15 of 16 rollouts would be dropped"


def test_explicit_repetition_id_still_wins_over_sample():
    r = _rec(sample=7)
    r["repetition_id"] = 3
    assert trajview_from_dump(r, success_threshold=1.0).repetition_id == 3


# --- bug 3: the trajectory lives in `conversation`, and the demo must not be attributed to the agent
def test_messages_come_from_conversation_with_the_scaffold_demo_STRIPPED():
    """`messages_from_dump` looked only at `messages`, so render() returned 0 chars and the analyzer
    produced confident hints from an EMPTY trajectory. And `conversation` cannot be used wholesale:
    the first n_demo assistant messages are the scaffold's worked example."""
    msgs = messages_from_dump(_rec(n_turns=3, n_demo=10))
    assert msgs, "empty trajectory would be sent to the analyzer"
    assistant = [m for m in msgs if m["role"] == "assistant"]
    assert len(assistant) == 3, f"expected the policy's 3 turns, got {len(assistant)}"
    body = " ".join(m["content"] for m in msgs)
    assert "POLICY action 0" in body
    assert "demo action" not in body, "scaffold demo attributed to the agent"
    assert "supervisor instruction" in body, "task description dropped"


def test_render_is_non_empty_and_turn_count_is_turns_not_assistant_messages():
    tv = trajview_from_dump(_rec(n_turns=4, n_demo=10), success_threshold=1.0)
    assert len(tv.render(200_000, strip_think=False)) > 0
    # the measured corpus invariant: assistant_msgs - len(turns) == n_demo
    assert tv.num_turns == 4


def test_native_messages_dump_is_untouched():
    """ALFWorld/frontier dumps that already carry `messages` must keep taking that path."""
    r = {
        "task_id": "t",
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "there"}],
        "reward": 10.0,
    }
    assert [m["content"] for m in messages_from_dump(r)] == ["hi", "there"]
    assert trajview_from_dump(r).success is True


# ==================================================================================================
# SkyRL eval-dump schema (ALFWorld): {input_prompt, output_response, score, env_extras{extra_info}}
# A fourth distinct shape, sharing none of AppWorld's field names.
# ==================================================================================================
def _skyrl_eval(*, win=False, n_invalid=0, task="trial_T20190907_164431_737459"):
    """`score` is a PER-TOKEN list: 10.0 on the winning token, -0.1 per invalid action."""
    score = [0.0] * 50 + [-0.1] * n_invalid + ([10.0] if win else [])
    return {
        "data_source": "alfworld",
        "env_class": "alfworld",
        "stop_reason": "stop",
        "env_extras": {"extra_info": {"task_id": task, "split": "train", "task_type": "pick"}},
        "input_prompt": "<|im_start|>user\nYour task is to: put a mug in the sink<|im_end|>\n",
        "output_response": "<|im_start|>assistant\ngo to sink 1<|im_end|>\n"
        "<|im_start|>user\nYou arrive at sink 1.<|im_end|>\n",
        "score": score,
    }


def test_skyrl_eval_task_id_is_read_from_the_NESTED_env_extras():
    """`extra_info` sits under `env_extras`, not at top level. Missing it collapses the whole corpus
    into one 'unknown' group."""
    assert trajview_from_dump(_skyrl_eval(), success_threshold=10.0).task_id.startswith("trial_T2019")


def test_skyrl_eval_success_is_MAX_of_the_score_list_not_the_SUM():
    """ALFWorld pays 10.0 on the win and -0.1 per invalid action, so a genuine win with >=100 invalid
    actions SUMS below 10. Measured on a 200-task ALFWorld train dump from the 30B Instruct student:
    max>=10 reproduces pass@1 0.2634 exactly; sum>=10 gives 0.1087, a 2.4x undercount."""
    assert trajview_from_dump(_skyrl_eval(win=True), success_threshold=10.0).success is True
    assert trajview_from_dump(_skyrl_eval(win=False), success_threshold=10.0).success is False
    # the case that breaks SUM but not MAX
    noisy_win = _skyrl_eval(win=True, n_invalid=150)  # sum = 10 - 15 = -5
    assert sum(noisy_win["score"]) < 0, "fixture should have a negative sum"
    assert trajview_from_dump(noisy_win, success_threshold=10.0).success is True


def test_skyrl_eval_messages_come_from_input_prompt_plus_output_response():
    msgs = messages_from_dump(_skyrl_eval(win=True))
    assert msgs, "render would be empty -> analyzer sees no trajectory"
    assert any(m["role"] == "assistant" for m in msgs)
    assert "put a mug in the sink" in " ".join(m["content"] for m in msgs)


def test_skyrl_eval_dump_has_no_rep_index_so_per_traj_needs_synthesis():
    """Every SkyRL eval-dump record yields repetition_id 0; build() must synthesise from file order (or abort)."""
    tvs = [trajview_from_dump(_skyrl_eval(), success_threshold=10.0) for _ in range(4)]
    assert {t.repetition_id for t in tvs} == {0}
    for i, t in enumerate(tvs):  # what --synthesize-rep does
        t.repetition_id = i
    assert len({f"{t.task_id}#rep{t.repetition_id}" for t in tvs}) == 4

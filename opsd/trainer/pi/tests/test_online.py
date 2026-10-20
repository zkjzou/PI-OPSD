"""Unit tests for the online-hint pipeline (pi/online.py + constructors).

No network / no GPU: a char-level FakeTokenizer drives the token->TrajView reconstruction, and the
analyzer LLM call is monkeypatched to a canned JSON response. Run from the repo root:

    .venv/bin/python -m pytest opsd/trainer/pi/tests/test_online.py -q
"""

from __future__ import annotations

import asyncio

import opsd.pi.analyzer as az
from opsd.pi.online import (
    OnlineHintConfig,
    OnlineHintManager,
    resolve_key,
    trajview_from_tokens,
)

# Canned analyzer output carrying keys for BOTH contrastive (hint) and prompt_skill (updated_hint).
_CANNED = (
    '{"key_difference":"one checked admissible actions first","preferred_behavior":"check admissible first",'
    '"hint":"Plan first: locate target then a valid receptacle, one admissible action per step.",'
    '"updated_hint":"Plan first: locate target then a valid receptacle, one admissible action per step.",'
    '"changed":true}'
)


class FakeTokenizer:
    """Whitespace tokenizer: decode(encode(x)) == x; skip_special_tokens is a no-op here."""

    def __init__(self):
        self.vocab = {}
        self.inv = {}

    def _id(self, tok: str) -> int:
        if tok not in self.vocab:
            idx = len(self.vocab)
            self.vocab[tok] = idx
            self.inv[idx] = tok
        return self.vocab[tok]

    def encode(self, text: str):
        return [self._id(t) for t in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.inv.get(int(i), "") for i in ids)


def _install_fake_analyzer(monkeypatch):
    # **kwargs, NOT a pinned keyword list. A fixed signature here makes this fake a silent trap: adding a
    # parameter to Analyzer.analyze (e.g. reasoning_effort) keeps these tests green while the real call
    # TypeErrors in production, where _refine_one SWALLOWS it and reports "error" -- zero hints, and with
    # OPSD_ONLINE_HINTS_STRICT unset the run trains on nothing. See test_onpolicy_skill_sampling.py, which
    # asserts the call contract explicitly.
    async def _fake(self, prompt, **kwargs):
        return _CANNED

    monkeypatch.setattr(az.Analyzer, "analyze", _fake, raising=True)


def _make_traj(tok: FakeTokenizer, task_text: str, succeeded: bool):
    """Build (prompt_ids, response_ids, loss_mask, reward) for one ALFWorld-style trajectory."""
    prompt = tok.encode(f"Your task is to: {task_text} .") + tok.encode(
        "A reviewer left the following hint blah blah"  # injected block -> must be stripped
    )
    assistant1 = tok.encode("assistant I will look for the target then act <action> go </action>")
    obs1 = tok.encode("user Observation you arrive at a cabinet")
    assistant2 = tok.encode("assistant take the item and place it <action> put </action>")
    response = assistant1 + obs1 + assistant2
    loss_mask = [1] * len(assistant1) + [0] * len(obs1) + [1] * len(assistant2)
    reward = [0.0] * (len(response) - 1) + [1.0 if succeeded else 0.0]
    return prompt, response, loss_mask, reward


def test_reconstruction_recovers_roles_and_outcome():
    tok = FakeTokenizer()
    prompt, response, loss_mask, reward = _make_traj(tok, "put a clean apple in fridge", succeeded=True)
    tv = trajview_from_tokens(prompt, response, loss_mask, reward, 3, tok, "task_abc", success_threshold=1.0)

    assert tv.messages[0]["role"] == "user"
    assert "Your task is to" in tv.messages[0]["content"]
    assert "reviewer" not in tv.messages[0]["content"]  # hint block stripped
    assert tv.num_turns == 2  # two assistant runs
    roles = [m["role"] for m in tv.messages]
    assert roles == ["user", "assistant", "user", "assistant"]  # role prefix words dropped, order kept
    assert tv.success is True and tv.reward == 1.0
    assert tv.repetition_id == 3


def test_scalar_reward_and_failure():
    tok = FakeTokenizer()
    prompt, response, loss_mask, _ = _make_traj(tok, "heat an egg", succeeded=False)
    tv = trajview_from_tokens(prompt, response, loss_mask, 0.0, 0, tok, "t", success_threshold=1.0)
    assert tv.success is False and tv.reward == 0.0


def test_resolve_key_prefers_task_type_then_task_id_then_uid():
    ee = {"extra_info": {"task_type": "pick_and_place", "task_id": "gid-1"}}
    assert resolve_key(ee, "7", "task_type") == "pick_and_place"
    assert resolve_key(ee, "7", "task_id") == "gid-1"
    assert resolve_key(ee, "7", "uid") == "7"
    # top-level fields also work; missing fields fall back
    assert resolve_key({"task_id": "top"}, "7", "task_type") == "top"
    assert resolve_key({}, "7", "task_type") == "7"


def _batch(tok: FakeTokenizer, specs):
    """specs: list of (task_type, task_id, [successes]) -> flattened batch arrays."""
    p, r, lm, rew, ee, inst, rep = [], [], [], [], [], [], []
    for task_type, task_id, succ_list in specs:
        for j, s in enumerate(succ_list):
            pp, rr, mm, rw = _make_traj(tok, f"{task_type} thing", succeeded=s)
            p.append(pp)
            r.append(rr)
            lm.append(mm)
            rew.append(rw)
            ee.append({"extra_info": {"task_type": task_type, "task_id": task_id}})
            inst.append(task_id)
            rep.append(j)
    return p, r, lm, rew, ee, inst, rep


def _run(coro):
    return asyncio.run(coro)


def test_contrastive_updates_any_pair_and_skips_singleton(monkeypatch):
    _install_fake_analyzer(monkeypatch)
    tok = FakeTokenizer()
    cfg = OnlineHintConfig(enabled=True, pi_type="contrastive", key_mode="task_type", success_threshold=1.0)
    mgr = OnlineHintManager(cfg, tok)
    # A: mixed (win+loss) -> pair; B: two failures -> still a valid pair (compare failures);
    # C: singleton -> no pair -> keep prior (no clobber).
    batch = _batch(
        tok,
        [
            ("pick_and_place", "gidA", [True, False, True, False]),
            ("look_at_obj", "gidB", [False, False]),
            ("clean_place", "gidC", [True]),
        ],
    )
    metrics = _run(mgr.update(*batch, global_step=1))

    assert mgr.hint_for(batch[4][0], "gidA")  # mixed pair got a hint
    assert mgr.hint_for({"extra_info": {"task_type": "look_at_obj"}}, "gidB")  # failure-vs-failure also updates
    assert mgr.hint_for({"extra_info": {"task_type": "clean_place"}}, "gidC") == ""  # singleton -> no clobber
    assert metrics["online_hints/updated_this_step"] == 2.0
    assert metrics["online_hints/groups_seen"] == 3.0


def test_contrastive_two_successes_still_produces_hint(monkeypatch):
    _install_fake_analyzer(monkeypatch)
    tok = FakeTokenizer()
    cfg = OnlineHintConfig(enabled=True, pi_type="contrastive", key_mode="task_id")
    mgr = OnlineHintManager(cfg, tok)
    batch = _batch(tok, [("pick_and_place", "gidA", [True, True])])  # both succeeded
    metrics = _run(mgr.update(*batch, global_step=1))
    assert metrics["online_hints/updated_this_step"] == 1.0
    assert mgr.hint_for(batch[4][0], "gidA")


def test_prompt_skill_updates_even_without_variance(monkeypatch):
    _install_fake_analyzer(monkeypatch)
    tok = FakeTokenizer()
    cfg = OnlineHintConfig(enabled=True, pi_type="prompt_skill", key_mode="task_type", fold_per_step=2)
    mgr = OnlineHintManager(cfg, tok)
    batch = _batch(tok, [("pick_and_place", "gidA", [False, False, False, False])])
    metrics = _run(mgr.update(*batch, global_step=1))
    assert metrics["online_hints/updated_this_step"] == 1.0
    assert "Plan first" in mgr.hint_for(batch[4][0], "gidA")


def test_refine_in_place_second_visit(monkeypatch):
    _install_fake_analyzer(monkeypatch)
    tok = FakeTokenizer()
    cfg = OnlineHintConfig(enabled=True, pi_type="contrastive", key_mode="task_id")
    mgr = OnlineHintManager(cfg, tok)
    batch = _batch(tok, [("pick_and_place", "gidA", [True, False])])
    _run(mgr.update(*batch, global_step=1))
    first = mgr.hint_for(batch[4][0], "gidA")
    assert first
    _run(mgr.update(*batch, global_step=2))  # prior now present -> refine path
    assert mgr.hint_for(batch[4][0], "gidA")  # still a valid hint after refine


def test_update_every_throttles(monkeypatch):
    _install_fake_analyzer(monkeypatch)
    tok = FakeTokenizer()
    cfg = OnlineHintConfig(enabled=True, pi_type="contrastive", key_mode="task_id", update_every=5)
    mgr = OnlineHintManager(cfg, tok)
    batch = _batch(tok, [("pick_and_place", "gidA", [True, False])])
    metrics = _run(mgr.update(*batch, global_step=3))  # 3 % 5 != 0 -> skip
    assert "online_hints/updated_this_step" not in metrics
    assert mgr.hint_for(batch[4][0], "gidA") == ""

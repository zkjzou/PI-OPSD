"""AppWorld schema-adapter tests.

The load-bearing case is ``generated_span``: AppWorld's official scaffold primes the model with a
supervisor instruction AND a worked demo, so a conversation can hold many more assistant messages than
the policy actually generated (measured: 38 messages / 19 assistant / 9 generated turns). If the prompt
boundary is taken as ``conversation[:1]``, the demo's assistant tokens land in the loss mask as if the
student wrote them and every KL/logprob attribute is silently wrong.
"""

from __future__ import annotations

import pytest

from opsd.pi.adapters import get_adapter
from opsd.pi.adapters.appworld import (
    adapt_record,
    appworld_is_success,
    appworld_task_key,
    generated_span,
)


class _FakeTok:
    """Whitespace tokenizer: 1 token per whitespace-separated chunk. Enough to assert mask alignment."""

    def encode(self, s, add_special_tokens=False):
        return [hash(w) % 50000 for w in s.split()]


def _conv(n_demo_exchanges: int, n_gen_turns: int, trailing_obs: bool = False):
    """Supervisor prompt + `n_demo_exchanges` demo (assistant,user) pairs + `n_gen_turns` real turns."""
    conv = [{"role": "user", "content": "supervisor instruction"}]
    for i in range(n_demo_exchanges):
        conv.append({"role": "assistant", "content": f"demo action {i}"})
        conv.append({"role": "user", "content": f"demo obs {i}"})
    for i in range(n_gen_turns):
        conv.append({"role": "assistant", "content": f"gen action {i}"})
        if trailing_obs or i < n_gen_turns - 1:
            conv.append({"role": "user", "content": f"gen obs {i}"})
    return conv


def test_registered():
    assert get_adapter("appworld").name == "appworld"


def test_task_key_and_success():
    assert appworld_task_key({"task_id": "fac291d_3"}) == "fac291d_3"
    assert appworld_task_key({}) is None
    assert appworld_is_success({"reward": 1.0}) == 1
    assert appworld_is_success({"reward": 0.0}) == 0
    # partial credit is NOT success under REWARD_MODE=binary
    assert appworld_is_success({"reward": 0.83}) == 0
    # explicit success field wins when present
    assert appworld_is_success({"success": 1.0, "reward": 0.0}) == 1


def test_generated_span_skips_the_demo():
    # 10 demo exchanges + 9 generated turns, last turn has no trailing obs -> the measured real shape
    conv = _conv(10, 9, trailing_obs=False)
    assert len(conv) == 38
    assert sum(1 for m in conv if m["role"] == "assistant") == 19
    start = generated_span(conv, 9)
    assert start == 21
    assert conv[start]["content"] == "gen action 0"
    # everything before the boundary is prompt, and it contains every demo action
    assert all("gen action" not in m["content"] for m in conv[:start])


def test_generated_span_with_trailing_obs():
    """A trajectory truncated at max_turns HAS a final observation -- the 'last 2n-1' rule would break."""
    conv = _conv(10, 9, trailing_obs=True)
    start = generated_span(conv, 9)
    assert conv[start]["content"] == "gen action 0"


def test_generated_span_degenerate_cases():
    conv = _conv(3, 2)
    # no turns recorded (e.g. immediate model error) -> nothing to score
    assert generated_span(conv, 0) == len(conv)
    # more turns claimed than assistant messages present -> refuse to guess, score nothing
    assert generated_span(conv, 99) == len(conv)


def test_adapt_record_masks_observations_only():
    conv = _conv(2, 3)
    rec = {
        "task_id": "abc_1",
        "sample": 4,
        "reward": 1.0,
        "conversation": conv,
        "turns": [{"turn": i, "action": f"gen action {i}", "obs": f"gen obs {i}"} for i in range(3)],
    }
    out = adapt_record(rec, _FakeTok(), k=0)

    assert out["task_id"] == "abc_1"
    assert out["uid"] == "abc_1#4"
    assert out["success"] == 1
    assert out["num_turns"] == 3
    assert out["env_metrics"]["success"] == 1.0
    assert len(out["response_ids"]) == len(out["loss_mask"])
    # some tokens are scored and some are masked: assistant 1, env observation 0
    assert 1 in out["loss_mask"] and 0 in out["loss_mask"]
    # the prompt must carry the demo, and the response must not
    assert out["prompt_token_ids"], "prompt should be non-empty"


def test_adapt_record_no_turns_scores_nothing():
    conv = _conv(2, 0)
    rec = {"task_id": "abc_2", "sample": 0, "reward": 0.0, "conversation": conv, "turns": []}
    out = adapt_record(rec, _FakeTok())
    assert out["response_ids"] == []
    assert out["loss_mask"] == []
    assert out["success"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

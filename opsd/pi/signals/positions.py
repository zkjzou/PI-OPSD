"""Token-position + hint-splicing helpers for PI scoring (pure, no heavy deps).

Extracted verbatim from ``opsd/trainer/pi/score_trajectories.py`` (Phase 1). These are the alignment-critical
pieces exercised by ``test_score_alignment.py``: locating the user-turn boundary, splicing a hint
inside the user turn, and mapping the student (loss_mask=1) tokens to absolute sequence positions.
"""

from __future__ import annotations

from typing import List, Optional, Sequence


def first_token_index(token_ids: Sequence[int], target_id: int) -> int:
    """Index of the first occurrence of ``target_id``. -1 if absent."""
    for i, t in enumerate(token_ids):
        if t == target_id:
            return i
    return -1


def last_token_index(token_ids: Sequence[int], target_id: int) -> int:
    """Index of the LAST occurrence of ``target_id``. -1 if absent."""
    for i in range(len(token_ids) - 1, -1, -1):
        if token_ids[i] == target_id:
            return i
    return -1


def splice_hint(prompt_token_ids: List[int], hint_ids: List[int], im_end_id: int) -> Optional[List[int]]:
    """Insert hint tokens just before the LAST ``<|im_end|>`` in the prompt.

    The prompt ends with the user turn close + the assistant generation header
    (``...{user content}<|im_end|>\\n<|im_start|>assistant\\n``); the assistant header has no
    ``<|im_end|>``, so the last ``<|im_end|>`` is the end of the user content — the same place
    ``AlfworldEnv`` appends the hint. Using the LAST (not first) occurrence is robust to a leading
    system message. Returns None if there is no ``<|im_end|>`` to splice before.
    """
    k = last_token_index(prompt_token_ids, im_end_id)
    if k < 0:
        return None
    return list(prompt_token_ids[:k]) + list(hint_ids) + list(prompt_token_ids[k:])


def assistant_positions(prompt_len: int, loss_mask: Sequence[int]) -> List[int]:
    """Absolute positions of student (trainable) tokens in (prompt + response)."""
    return [prompt_len + j for j, m in enumerate(loss_mask) if m]


__all__ = ["first_token_index", "last_token_index", "splice_hint", "assistant_positions"]

"""Model-family tokenization specs for PI scoring (replaces hardcoded Qwen assumptions).

Centralizes the two model-specific knobs the PI scorer/analysis scripts previously hardcoded:
- the user-turn-end special token to splice a hint before (``<|im_end|>`` for Qwen), and
- the reasoning ``ChannelSpec`` (``<think>``/``<action>`` for Qwen; action-only for GLM).

Use ``get_token_spec(family)`` to pick a spec and ``resolve_im_end_id(tok, spec.im_end_token)`` to map
the token to an id (falling back to EOS), which was copy-pasted in score_trajectories.run and
analyze_pi_positions.
"""

from __future__ import annotations

from dataclasses import dataclass

from opsd.pi.signals.channels import GLM_CHANNELS, QWEN_CHANNELS, ChannelSpec


@dataclass(frozen=True)
class ModelTokenSpec:
    """Per-model-family tokens/channels the PI signals need. Extend by adding to ``_SPECS`` below."""

    family: str
    im_end_token: str  # user-turn-end special token (hint is spliced before its last occurrence)
    channels: ChannelSpec  # reasoning channels for token labeling


QWEN = ModelTokenSpec("qwen", "<|im_end|>", QWEN_CHANNELS)
# GLM ReAct traces have no <think> tags; keep the same turn-end token unless a spec says otherwise.
GLM = ModelTokenSpec("glm", "<|im_end|>", GLM_CHANNELS)

_SPECS = {"qwen": QWEN, "glm": GLM}


def get_token_spec(family: str = "qwen") -> ModelTokenSpec:
    """Look up a ModelTokenSpec by family name (default qwen). KeyError if unknown."""
    return _SPECS[family.lower()]


def resolve_im_end_id(tok, im_end_token: str = "<|im_end|>") -> int:
    """Map the user-turn-end token to an id, falling back to EOS if it isn't a real special token.

    Mirrors the (previously duplicated) resolution in score_trajectories.run / analyze_pi_positions.
    """
    tid = tok.convert_tokens_to_ids(im_end_token)
    if tid is None or tid == tok.unk_token_id:
        tid = tok.eos_token_id
    return tid


__all__ = ["ModelTokenSpec", "QWEN", "GLM", "get_token_spec", "resolve_im_end_id"]

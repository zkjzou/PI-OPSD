"""Per-token reasoning-channel labeling (``think`` / ``action`` / ``other``), model-configurable.

A ``ChannelSpec`` lists ``(name, open_tag, close_tag)`` triples; ``token_channels`` labels each token by
which span it falls inside (via cumulative single-token decode). This unifies the regex channel
machine that lived in ``opsd/trainer/pi/turn_channel_kl.py`` (``_token_channels``); the ALFWorld/Qwen tag set
is the default, but GLM / other models can pass their own spec instead of the hardcoded ``<think>``/``<action>``.

Span-termination detail (kept from the original): a span ends at its own close tag, OR the next
opening tag of ANY channel, OR end-of-text. The "next opening tag" bound matters because
Qwen3-Thinking frequently omits ``</think>`` in multi-turn react; without it an unclosed ``<think>``
would swallow the following ``<action>`` and mislabel ~half the action tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

OTHER = "other"


@dataclass(frozen=True)
class ChannelSpec:
    """Reasoning channels for a model family. ``channels`` = ((name, open_tag, close_tag), ...)."""

    channels: Tuple[Tuple[str, str, str], ...]

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(name for name, _, _ in self.channels)

    def compiled(self):
        """Per-channel regex: <open>(.*?) up to </close> | any channel's <open> | end-of-text."""
        open_lookaheads = "".join(f"(?={re.escape(o)})|" for _, o, _ in self.channels)
        out = {}
        for name, open_tag, close_tag in self.channels:
            pat = re.escape(open_tag) + r"(.*?)(?:" + re.escape(close_tag) + r"|" + open_lookaheads + r"\Z)"
            out[name] = re.compile(pat, re.DOTALL)
        return out


# ALFWorld / Qwen3 default (identical tags to the original opsd/trainer/pi/turn_channel_kl._THINK/_ACTION).
QWEN_CHANNELS = ChannelSpec((("think", "<think>", "</think>"), ("action", "<action>", "</action>")))
# GLM-family: no <think> tags in its ReAct traces; only actions are tagged.
GLM_CHANNELS = ChannelSpec((("action", "<action>", "</action>"),))
# AppWorld: the agent's action is a fenced Python block, never an <action> tag
# (opsd/envs/appworld/scaffold.py::extract_code). Think tags are unchanged.
APPWORLD_CHANNELS = ChannelSpec((("think", "<think>", "</think>"), ("action", "```python", "```")))
# τ²-bench: the agent acts by TOOL CALL, not by an <action> tag. Using QWEN_CHANNELS here does not
# error -- it silently labels every action token `other`, so the `action` series is a flat zero and
# reads as "the teacher exerts no pressure on actions" when in truth the channel was never matched.
# Tag source: the preserve chat template renders tool calls as `<tool_call>\n{...}\n</tool_call>`
# (qwen3thinking_preserve.jinja2) and re-renders `<think>...</think>` for EVERY assistant turn.
TAU2_CHANNELS = ChannelSpec((("think", "<think>", "</think>"), ("action", "<tool_call>", "</tool_call>")))

_SPECS = {
    "qwen": QWEN_CHANNELS,
    "alfworld": QWEN_CHANNELS,
    "glm": GLM_CHANNELS,
    "tau2": TAU2_CHANNELS,
    "appworld": APPWORLD_CHANNELS,
}


def spec_from_env(default: str = "qwen") -> ChannelSpec:
    """Select a spec by name from ``OPSD_KL_CHANNEL_SPEC``. Unknown names RAISE rather than falling
    back: a typo silently reverting to the ALFWorld tags is exactly the flat-zero failure above."""
    import os

    name = (os.environ.get("OPSD_KL_CHANNEL_SPEC", "") or default).strip().lower()
    if name not in _SPECS:
        raise ValueError(f"OPSD_KL_CHANNEL_SPEC={name!r} unknown; expected one of {sorted(_SPECS)}")
    return _SPECS[name]


def token_channels(ids: Sequence[int], tokenizer, spec: ChannelSpec = QWEN_CHANNELS) -> List[str]:
    """Per-token channel label (a channel name, or ``other``) via cumulative single-token decode.

    Single-token decode concatenation approximates the joint decode (byte-level BPE edge cases) but is
    self-consistent: the tag spans are found in the SAME concatenated text the per-token offsets index.
    """
    pieces = [tokenizer.decode([int(t)], skip_special_tokens=False) for t in ids]
    text = "".join(pieces)
    compiled = spec.compiled()
    spans = {name: [(m.start(1), m.end(1)) for m in rx.finditer(text)] for name, rx in compiled.items()}

    def _inside(name: str, a: int, b: int) -> bool:
        return any(s <= a and b <= e for s, e in spans[name])

    labels: List[str] = []
    pos = 0
    for p in pieces:
        a, b = pos, pos + len(p)
        pos = b
        label = OTHER
        for name in spec.names:  # priority = spec order (think before action, matching the original)
            if _inside(name, a, b):
                label = name
                break
        labels.append(label)
    return labels


__all__ = [
    "ChannelSpec",
    "QWEN_CHANNELS",
    "GLM_CHANNELS",
    "APPWORLD_CHANNELS",
    "TAU2_CHANNELS",
    "OTHER",
    "spec_from_env",
    "token_channels",
]

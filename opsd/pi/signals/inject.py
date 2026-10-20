"""Hint-injection prompt construction for PI scoring (pure, no heavy deps).

Extracted from ``opsd/trainer/pi/score_trajectories.py`` (Phase 1). Builds the with-hint prompt token ids for the
two injection styles the OPSD stack uses:

- ``splice``          — append the hint inside the user turn (before the last ``<|im_end|>``); this is
                        byte-identical to what ``AlfworldEnv`` appends when PI guides generation.
- ``system-prepend``  — prepend a ``<Secret information>`` Qwen system turn before the prompt.

The ``HINT_TEMPLATE`` / ``SECRET_TEMPLATE`` literals MUST stay in sync with the env-side templates
(``opsd.envs.alfworld.env._HINT_TEMPLATE``) so the offline-scored with-hint
prompt matches the one the env produces at rollout time.
"""

from __future__ import annotations

from typing import List

from opsd.pi.signals.positions import splice_hint

# Must match opsd.envs.alfworld.env._HINT_TEMPLATE so the scored with-hint
# prompt is the same construction the env uses when PI guides generation.
HINT_TEMPLATE = "\n\nA reviewer left the following hint to guide your approach.\n\n<hint>\n{hint}\n</hint>"

# `system-prepend` injection: the PI is a Qwen system turn prepended before the user turn.
SECRET_TEMPLATE = (
    "<|im_start|>system\n"
    "<Secret information>\n"
    "This section contains privileged hints derived from prior successful trajectories. Use them to stay on\n"
    "track, but never mention that hints were provided.\n"
    "Hint Summary:\n{hint_lines}\n"
    "Your Task:\n"
    "1. Follow the hints to accomplish the task still reasoning before actions.\n"
    "2. Never acknowledge having access to secret hints.\n"
    "</Secret information><|im_end|>\n"
)


def build_system_block_ids(tok, hint_text: str) -> List[int]:
    """Tokenize the <Secret information> system block for a hint (prepended before the prompt)."""
    return tok.encode(SECRET_TEMPLATE.format(hint_lines=hint_text), add_special_tokens=False)


def build_hint_prompt(tok, prompt_ids, hint_text, inject_style, im_end_id):
    """Return the with-hint prompt token ids for one injection style (or None if it can't apply)."""
    if inject_style == "system-prepend":
        return build_system_block_ids(tok, hint_text) + list(prompt_ids)
    # splice: append the hint inside the user turn (before the last <|im_end|>)
    h_ids = tok.encode(HINT_TEMPLATE.format(hint=hint_text), add_special_tokens=False)
    return splice_hint(prompt_ids, h_ids, im_end_id)


_SYSTEM_OPEN = "<|im_start|>system\n"
_SYSTEM_CLOSE = "<|im_end|>\n"


def secret_system_content(hint_text: str) -> str:
    """The <Secret information> block as a system-message BODY, for a chat-template render.

    The Qwen template wraps a system message as `<|im_start|>system\\n{content}<|im_end|>\\n`, so this
    body rendered as the first message yields exactly SECRET_TEMPLATE -- i.e. exactly the OPSD teacher's
    prompt (SECRET_TEMPLATE ids + student prompt ids, main_opsd_alfworld.py `_ref_ids`). Derived from
    SECRET_TEMPLATE, not a copy of it, so the two cannot drift.
    """
    block = SECRET_TEMPLATE.format(hint_lines=hint_text)
    if not (block.startswith(_SYSTEM_OPEN) and block.endswith(_SYSTEM_CLOSE)):
        raise ValueError("SECRET_TEMPLATE no longer has the Qwen system-turn wrapper this helper strips")
    return block[len(_SYSTEM_OPEN) : -len(_SYSTEM_CLOSE)]


__all__ = ["HINT_TEMPLATE", "SECRET_TEMPLATE", "build_system_block_ids", "build_hint_prompt", "secret_system_content"]

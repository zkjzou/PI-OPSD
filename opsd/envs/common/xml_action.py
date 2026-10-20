"""Action parser for the ALFWorld / WebShop agents (`<action>...</action>` grammar).

Originally a strict port of SDAR/verl-agent's ``alfworld_projection`` / ``webshop_projection``,
which require BOTH a `<think>...</think>` and an `<action>...</action>` block. That strict rule does
NOT work with Qwen3 via the HuggingFace chat template: with ``chat_template_kwargs.enable_thinking=
false`` the template PRE-FILLS and CLOSES an empty ``<think>\\n\\n</think>`` in the prompt, so the
model's *generated* text contains no `<think>` (and Qwen3-1.7B then emits the bare admissible command
with no `<action>` tag either, e.g. ``examine cabinet 1``). Verified locally on Qwen/Qwen3-1.7B.

So we relax validity to match how this model actually responds:
  1. If an `<action>...</action>` block is present, use its content (trained models learn to emit it).
  2. Otherwise fall back to the last non-empty line of the (special-token-stripped) output — for
     ``enable_thinking=false`` this is the bare command/action the model emits.
  3. Reject CJK characters.
The `<think>` block is NO LONGER required (the template handles the thinking framing). The real
action-quality gate lives downstream: ALFWorld checks membership in ``admissible_commands``; WebShop
checks the ``search[...]``/``click[...]`` form. The action is lowercased + stripped (case-insensitive
matching downstream). NOTE: this is a deliberate deviation from SDAR's strict projection, forced by the
Qwen3 enable_thinking=false template behavior.
"""

import re
from typing import Optional, Tuple

_CJK_RE = re.compile(r"[一-鿿]")
_SPECIAL = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")
# Complete <think>...</think> reasoning spans (case-insensitive, spans newlines). Stripped BEFORE the
# <action> search so an <action> the model "rehearses" inside its reasoning is never parsed as the real
# action — the real command is emitted AFTER </think>.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# All <action>...</action> blocks (case-insensitive, spans newlines) — used to take the LAST NON-EMPTY one
# (an empty ``<action> </action>``, e.g. the model quoting the instructions, must not count as an action).
_ACTION_BLOCK_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)


def _last_nonempty_action(region: str) -> Optional[str]:
    """Content of the LAST ``<action>...</action>`` block with non-empty (stripped) content; else None."""
    for content in reversed(_ACTION_BLOCK_RE.findall(region)):
        if content.strip():
            return content.strip()
    return None


def parse_think_action(
    text: str, *, require_think_close: bool = False, strict_outside_think: bool = False
) -> Tuple[Optional[str], bool]:
    """Extract the action from an agent response.

    ``require_think_close`` (set it True for THINKING models, e.g. Qwen3-*-Thinking with
    ``enable_thinking=true``): the real command is emitted only in the answer AFTER ``</think>``. If the
    model never closed ``</think>`` (it ran out mid-reasoning / hit the token cap), there is NO committed
    action → returns ``(None, False)``; the reasoning is NOT scanned for ``<action>`` (so a rehearsed or
    instruction-quoted ``<action> </action>`` is never taken) and there is NO last-line fallback (a
    reasoning line is not an action). Leave it False for ``enable_thinking=false`` models (the template
    pre-closes ``<think>`` in the prompt and the model emits a bare command) — the relaxed behavior.

    ``strict_outside_think`` (STRICTEST; for thinking models when you want to force the action into the
    answer region): the committed action MUST be a non-empty ``<action>...</action>`` block appearing
    AFTER the LAST closed ``</think>``. If ``</think>`` never closed (unfinished / hit the token cap) OR
    there is no ``<action>`` tag after it, there is NO committed action → ``(None, False)`` (equivalent to
    an empty ``<action></action>``). Unlike the ``require_think_close`` relaxed-thinking mode, an
    ``<action>`` emitted inside a still-open ``<think>`` does NOT count, and there is NO bare-line
    fallback. Takes precedence over ``require_think_close`` when both are set.

    Returns ``(action, is_valid)``:
      - ``action``: the lowercased/stripped last non-empty `<action>` content, else (relaxed mode only)
        the last non-empty line of the answer region. ``None`` if nothing usable.
      - ``is_valid``: True if a non-empty action was extracted and there are no CJK characters. Callers
        apply the invalid-action penalty when False; admissibility/format is checked by the env downstream.
    """
    cleaned = text
    for s in _SPECIAL:
        cleaned = cleaned.replace(s, "")
    cleaned = cleaned.strip()

    if strict_outside_think:
        # STRICT: the action is ONLY a non-empty <action> tag AFTER the LAST </think>. Search the RAW
        # cleaned text (do NOT strip complete <think>...</think> spans first) so the last </think> — the
        # real close, whether the opening <think> was prefilled in the prompt or self-emitted — bounds the
        # answer region. No closed </think> -> unfinished -> no action. No <action> after close (or only an
        # empty '<action></action>') -> no action. No bare-line fallback: a reasoning line is not an action.
        close_idx = cleaned.lower().rfind("</think>")
        if close_idx == -1:
            action = None
        else:
            action = _last_nonempty_action(cleaned[close_idx + len("</think>") :])
    elif require_think_close:
        # Thinking model. Ignore actions rehearsed inside a COMPLETE <think>...</think> span, then take the
        # last NON-EMPTY <action> as the committed command. IMPORTANT: this model routinely emits that
        # <action> WITHOUT first closing </think> (it acts inside the still-open think) — that STILL counts;
        # requiring a closed </think> here wrongly rejected ~every action. Only when </think> IS present do
        # we also allow a bare post-</think> command. If there is NO </think> AND no real <action>, the turn
        # produced no committed action (e.g. reasoning that ran out / quoted the empty '<action> </action>')
        # -> invalid; never fall back to a reasoning line.
        search_region = _THINK_BLOCK_RE.sub("", cleaned)
        close_idx = search_region.lower().rfind("</think>")
        has_close = close_idx != -1
        if has_close:
            search_region = search_region[close_idx + len("</think>") :]
        action = _last_nonempty_action(search_region)
        if action is None:
            # Bare command after </think> is OK (answer region); with NO </think>, don't grab a reasoning line.
            lines = [ln.strip() for ln in search_region.splitlines() if ln.strip()]
            action = (lines[-1] if lines else "") if has_close else ""
    else:
        # Relaxed (enable_thinking=false / untagged): drop complete <think>...</think> spans, then keep the
        # tail after any remaining </think>, then take the last non-empty <action> or the last line.
        search_region = _THINK_BLOCK_RE.sub("", cleaned)
        close_idx = search_region.lower().rfind("</think>")
        if close_idx != -1:
            search_region = search_region[close_idx + len("</think>") :]
        action = _last_nonempty_action(search_region)
        if action is None:
            lines = [ln.strip() for ln in search_region.splitlines() if ln.strip()]
            action = lines[-1] if lines else ""

    action = (action or "").lower().strip()
    valid = bool(action) and _CJK_RE.search(text) is None
    return (action or None), valid

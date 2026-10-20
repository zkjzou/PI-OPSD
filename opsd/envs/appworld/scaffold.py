"""Unified AppWorld ReAct scaffold — the union of the OFFICIAL and LOOP agent scaffolds.

Motivation (measured, Qwen3-4B-Thinking-2507 on AppWorld dev/57, identical driver, temp 0.6):

    official `instructions.txt` prompt -> TGC 21.05 / completion 57.39
    LOOP `react_template.py` prompt    -> TGC 10.53 / completion 50.65
    our old SkyRL prompt + scaffold    -> TGC  0.00 / completion 30.8

The scaffold (prompt + code extraction + observation format + loop) dominates the score: the SAME
GPT-4o scored TGC 14.9 under our old scaffold and 61.9 under the official one. This module ports the
two reference scaffolds so SkyRL *training* rollouts use the same contract the papers evaluate under.

Where the two references differ, the behaviour is a config knob rather than a hard-coded choice:

  |                      | OFFICIAL (SimplifiedReActCodeAgent) | LOOP (MinimalReactAgent)          |
  |----------------------|-------------------------------------|-----------------------------------|
  | fence languages      | ```python only                      | ```python AND ```py               |
  | multiple code blocks | FIRST only (rest of turn truncated) | CONCATENATE all                   |
  | unterminated fence   | recovered (PARTIAL_CODE_RE)         | recovered                         |
  | observation format   | "Output:\n```\n{out}```\n"          | "Output:\n```\n{out}```\n"        |
  | reminder re-injection| none                                | persistent reminder message       |

Everything here is pure/stateless so it can be unit-tested without an AppWorld server.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

# --- reasoning-model CoT ------------------------------------------------------------------
# Thinking models (Qwen3-*-Thinking) frequently DRAFT code inside <think>…</think>. That draft is not
# the action: executing it (or letting it shadow the real fence) corrupts the rollout. Strip it first.
# Also handle an UNCLOSED <think> (generation truncated mid-thought) -> all of it is reasoning.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)

MULTI_BLOCK_CHOICES = ("first", "concat", "last")


def strip_reasoning(text: str) -> str:
    """Remove <think>…</think> (and an unterminated <think>…) — no-op for non-thinking models."""
    return _OPEN_THINK_RE.sub("", _THINK_RE.sub("", text))


@dataclass(frozen=True)
class ScaffoldSpec:
    """How to parse an assistant turn and render the environment's reply."""

    fence_langs: Sequence[str] = ("python", "py")
    allow_bare_fence: bool = False  # ``` with no language tag
    multi_block: str = "first"  # official="first", LOOP="concat", our legacy="last"
    partial_fence_recovery: bool = True
    obs_format: str = "official"  # "official" -> Output:\n```\n{out}```\n ; "raw"
    strip_think: bool = True
    strict_parse: bool = False  # no fence at all -> invalid action (None) instead of raw-exec

    def compiled(self):
        alt = "|".join(re.escape(x) for x in self.fence_langs)
        tag = f"(?:{alt})" + ("?" if self.allow_bare_fence else "")
        # NOTE: `\n?` (not `\n`) mirrors LOOP — models sometimes omit the newline after ```python.
        full = re.compile(rf"```{tag}[ \t]*\n?(.*?)```", re.DOTALL)
        # Leading `.*` (greedy, DOTALL) is load-bearing and copied from the references
        # (official `PARTIAL_CODE_RE = r".*```python\n(.*)"`): it lets `.match()` skip any preamble
        # and anchor on the LAST unterminated fence. Without it, recovery silently never fires.
        partial = re.compile(rf".*```{tag}[ \t]*\n?(.*)\Z", re.DOTALL)
        return full, partial


# Named presets. `official` is the default for training: it is the only variant with a measured
# reference number (GPT-4o TGC 61.9) and it beat LOOP's ~2x on our 4B.
SCAFFOLDS = {
    "official": ScaffoldSpec(fence_langs=("python",), multi_block="first"),
    # LOOP additionally accepts ```py ("Qwen seems to like ```py quite a bit") and concatenates blocks.
    "loop": ScaffoldSpec(fence_langs=("python", "py"), multi_block="concat"),
    # Union: most permissive parse. Best default for a SMALL policy, which fails formatting most often.
    "union": ScaffoldSpec(fence_langs=("python", "py"), allow_bare_fence=True, multi_block="first"),
    # What SkyRL did before this module (kept so old runs remain reproducible).
    "legacy": ScaffoldSpec(
        fence_langs=("python", "py"),
        allow_bare_fence=True,
        multi_block="last",
        partial_fence_recovery=False,
        obs_format="raw",
    ),
}


def extract_code(text: str, spec: ScaffoldSpec) -> Optional[str]:
    """Return the Python to execute, or None when `spec.strict_parse` and no fence was found.

    Returns "" for an empty action (caller nudges). Mirrors official `extract_code_and_fix_content`
    and LOOP `extract_code_format_output`, unified behind ScaffoldSpec.
    """
    if spec.strip_think:
        text = strip_reasoning(text)
    full_re, partial_re = spec.compiled()

    blocks: List[str] = []
    match_end = 0
    for m in full_re.finditer(text):
        blocks.append(m.group(1).strip())
        match_end = m.end()
        if spec.multi_block == "first":
            # Official returns on the FIRST complete block and never attempts partial recovery
            # (`if ignore_multiple_calls: return code`), so short-circuit to match it exactly.
            return blocks[0]

    # An unterminated trailing fence (token cap hit, or the model forgot the closing ```) is a real
    # action in BOTH references — recover whatever follows the last complete block.
    if spec.partial_fence_recovery:
        pm = partial_re.match(text[match_end:])
        if pm and pm.group(1).strip():
            blocks.append(pm.group(1).strip())

    if blocks:
        if spec.multi_block == "last":
            return blocks[-1]
        if spec.multi_block == "concat":
            return "\n".join(b for b in blocks if b)
        return blocks[0]

    if spec.strict_parse:
        return None
    return text.strip()


def format_observation(output, spec: ScaffoldSpec, max_chars: int) -> str:
    """Render the REPL result as the next user turn.

    The official/LOOP few-shot demos show every environment turn as ``Output:\\n```\\n…```\\n`` — the
    policy is trained to expect exactly that shape, so a bare dump is off-distribution.
    """
    s = output if isinstance(output, str) else str(output)
    if len(s) > max_chars:
        head = max_chars * 3 // 4
        s = s[:head] + f"\n... [{len(s) - max_chars} chars truncated] ...\n" + s[-(max_chars - head) :]
    if spec.obs_format == "raw":
        return s
    if not s.endswith("\n"):
        s += "\n"
    return f"Output:\n```\n{s}```\n"


# LOOP re-injects a short reminder during long rollouts so the key contract stays salient after the
# instructions have scrolled far up the context. Off by default (reminder_every=0).
REMINDER_MESSAGE = (
    "REMINDER: emit exactly ONE fenced Python block per step:\n"
    "```python\n<your code>\n```\n"
    "Look up API specs with `apis.api_docs.show_api_doc(...)` before calling an API, page through "
    "ALL results via `page_index`, never invent values or leave placeholders, and call "
    "`apis.supervisor.complete_task(answer=...)` when done (answers must be minimal and numeric "
    "where applicable)."
)

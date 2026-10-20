"""Core types for offline PI (privileged-information / hint) construction.

A `HintConstructor` turns one task's logged student trajectories into a `HintRecord`. The four
sources (static / reference / on-policy-skill / contrastive) subclass it. Everything here is
OFFLINE — the online per-batch path and the step-level (mid-sequence) injection are deferred.

The unified `HintRecord` schema is backward compatible with the original flat ``{task_id: str}``
hints (a pure task-level hint serializes back to a plain string), so existing ``hints_merged.json``
/ ``codecontests_1k_train_solution_hints.json`` load unchanged via `artifact.load_hint_artifact`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Analyzer INPUT budget helpers. Measured over 120 real ALFWorld transcripts with the
# Qwen3-4B-Thinking-2507 tokenizer, chars/token spans 2.24 (min) .. 3.70 (max), median 3.45; synthetic
# token-dense text (digits/punctuation/hex) reaches 1.91. So a chars-based cap is a BEST-EFFORT
# approximation, NOT a guarantee -- at 2.2 the dense case measured 36.8k tokens against a 32k budget.
# 1.8 keeps a margin below the dense floor while still using ~80% of the budget on real transcripts.
# Pass a tokenizer for an EXACT bound; the online path always does.
_CHARS_PER_TOKEN_FLOOR = 1.8
_TRUNC_MARKER = "\n...[trajectory truncated]...\n"


def _encode_no_special(tok: Any, text: str) -> List[int]:
    """``tok.encode(text, add_special_tokens=False)``, tolerating minimal tokenizers without that HF kwarg.

    The online path hands the POLICY tokenizer through, which is always an HF tokenizer. The fallback is for
    duck-typed tokenizers (the whitespace FakeTokenizer in opsd/trainer/pi/tests/test_online.py): without it the
    TypeError is swallowed by OnlineHintManager._refine_one as a refine "error" and no hint is produced.
    A tokenizer that then adds special tokens only shifts the cap by those few ids.
    """
    try:
        return tok.encode(text, add_special_tokens=False)
    except TypeError:
        return tok.encode(text)


@dataclass
class HintRecord:
    """Unified PI hint for one task.

    - ``task``  : task-level hint string (the only field the offline rank+guide harness uses today).
    - ``steps`` : reserved for the deferred step-level (per-turn) injection — {turn_idx: hint}.
    - ``meta``  : provenance (pi_type, source model, lengths) — used by the leakage/ablation analysis.
    """

    task: Optional[str] = None
    steps: Dict[int, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Any:
        # v1 backward-compat: a pure task-level hint with no steps/meta serializes as a plain string.
        if self.task is not None and not self.steps and not self.meta:
            return self.task
        out: Dict[str, Any] = {}
        if self.task is not None:
            out["task"] = self.task
        if self.steps:
            out["steps"] = {str(k): v for k, v in self.steps.items()}
        if self.meta:
            out["meta"] = self.meta
        return out

    @classmethod
    def from_json(cls, val: Any) -> "HintRecord":
        if isinstance(val, str):
            return cls(task=val)
        if isinstance(val, dict):
            steps = {int(k): str(v) for k, v in (val.get("steps") or {}).items()}
            return cls(task=val.get("task"), steps=steps, meta=dict(val.get("meta") or {}))
        raise TypeError(f"Unsupported hint record value: {type(val).__name__}")


@dataclass
class TrajView:
    """Source-agnostic adapter over ONE logged trajectory, consumed by the constructors.

    Built from a ``trajectories.jsonl`` record produced by ``generate_alfworld_local.py``. Kept
    minimal so offline drivers and (future) online hooks share the same constructor body.
    """

    task_id: str
    messages: List[Dict[str, str]]  # full chat history (first user msg + assistant/obs turns)
    reward: float
    success: bool
    num_turns: int = 0
    repetition_id: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def assistant_turns(self) -> List[str]:
        return [str(m.get("content", "")) for m in self.messages if m.get("role") == "assistant"]

    def render(self, max_chars: int = 8000, strip_think: bool = True) -> str:
        """Render the trajectory as plain text for an analyzer prompt (role-tagged turns)."""
        import re

        lines: List[str] = []
        for m in self.messages:
            role = m.get("role", "?")
            content = str(m.get("content", ""))
            if strip_think:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if not content:
                continue
            tag = {"user": "OBSERVATION", "assistant": "AGENT", "system": "SYSTEM"}.get(role, role.upper())
            lines.append(f"[{tag}] {content}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[: max_chars // 2] + "\n...[trajectory truncated]...\n" + text[-max_chars // 2 :]
        return text


_TASK_MARKER = "Your task is to: "
_ACTION_RE = __import__("re").compile(
    r"<action>\s*(.*?)\s*</action>|^\s*Action:\s*(.+?)\s*$",
    __import__("re").DOTALL | __import__("re").MULTILINE | __import__("re").IGNORECASE,
)


# Per-benchmark task markers as (marker, take_last). The direction is load-bearing:
#   * ALFWorld keeps FIRST-occurrence, which is what every existing hint artifact was built with.
#     Do not "fix" this to last -- it would silently change the task string on any prompt whose few-shot
#     examples also contain the marker, and so change hints that are already published.
#   * AppWorld needs LAST. Its scaffold preamble spends ~1.4k chars explaining the protocol and only then
#     states "Task: <the actual task>" on the final line, so a first-occurrence match (or the generic
#     first-user-line fallback) returns "I am your supervisor, and you are an AI Assistant..." for EVERY
#     task -- an identical, useless descriptor.
_TASK_MARKERS = {
    "alfworld": ((_TASK_MARKER, False),),
    "appworld": (("Task: ", True),),
}


def task_description_from_messages(messages: List[Dict[str, str]], bench: Optional[str] = None) -> str:
    """Best-effort short task descriptor from the first user message.

    ``bench`` selects the marker set; None/unknown -> ALFWorld, the default every caller assumed before
    this was a parameter.
    """
    markers = _TASK_MARKERS.get((bench or "alfworld").strip().lower(), _TASK_MARKERS["alfworld"])
    for m in messages:
        content = str(m.get("content", ""))
        for marker, take_last in markers:
            if marker not in content:
                continue
            tail = (content.rsplit(marker, 1) if take_last else content.split(marker, 1))[1]
            return tail.split("\n", 1)[0].strip().rstrip(".") or tail.strip()[:200]
    # fall back to the first non-empty user line
    for m in messages:
        if m.get("role") == "user" and str(m.get("content", "")).strip():
            return str(m["content"]).strip().split("\n", 1)[0][:200]
    return ""


def extract_actions(messages: List[Dict[str, str]]) -> List[str]:
    """Pull the agent's actions out of assistant turns (handles <action>..</action> and 'Action:')."""
    actions: List[str] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for grp in _ACTION_RE.findall(str(m.get("content", ""))):
            act = (grp[0] or grp[1]).strip()
            if act:
                actions.append(act)
    return actions


class HintConstructor(ABC):
    """Base for the four PI sources. Offline only for now (``build_offline``)."""

    pi_type: str = "base"

    def __init__(
        self,
        analyzer: Optional[Any] = None,
        *,
        max_chars: int = 6000,
        max_input_tokens: Optional[int] = 32000,
        **kwargs: Any,
    ):
        self.analyzer = analyzer
        # OUTPUT cap: the hint handed back to the policy (scrub_hint).
        self.max_chars = max_chars
        # INPUT cap: the analyzer's trajectory budget, in TOKENS. Deliberately separate from max_chars,
        # which used to serve both: an output cap of 6000 chars silently truncated the analyzer's INPUT to
        # ~1.7k tokens, hiding the middle of every 30-turn episode behind render()'s middle-out
        # "...[trajectory truncated]..." marker, while raising it to fit a long trajectory also permitted an
        # equally long hint.
        #
        # LEGACY MODE: ``max_input_tokens`` 0 / None / negative => the analyzer input is rendered with
        # ``TrajView.render(max_chars)``, i.e. ``max_chars`` caps BOTH the input and the hint. That is the
        # legacy AppWorld-line behaviour every AppWorld hint/online run was produced with
        # (OPSD_ONLINE_HINTS_MAX_CHARS=48000 bounded the analyzer input too), so it is the switch to
        # reproduce those runs. Normalised to 0 so ``self.max_input_tokens // 2`` stays valid everywhere.
        self.max_input_tokens = max(0, int(max_input_tokens or 0))
        self.tokenizer = kwargs.pop("tokenizer", None)
        self.cfg = kwargs

    @property
    def legacy_input_render(self) -> bool:
        """True when the analyzer input is capped by ``max_chars`` (legacy AppWorld line) rather than tokens."""
        return self.max_input_tokens <= 0

    def _legacy_chars_cap(self, text: str, legacy_max_chars: Optional[int] = None) -> str:
        """Middle-out chars cap, byte-identical to ``TrajView.render``'s own truncation."""
        n = int(self.max_chars if legacy_max_chars is None else legacy_max_chars)
        if len(text) > n:
            text = text[: n // 2] + _TRUNC_MARKER + text[-n // 2 :]
        return text

    def _cap_to_tokens(
        self, text: str, max_tokens: Optional[int] = None, *, legacy_max_chars: Optional[int] = None
    ) -> str:
        """Middle-out truncate ``text`` to ``max_input_tokens``, in TOKEN space when a tokenizer is set.

        A chars-based cap CANNOT bound tokens: measured over 120 real ALFWorld transcripts the
        chars/token ratio ranges 2.24-3.70 (median 3.45), so 32k tokens' worth at the median (112k chars)
        is ~50k tokens at the floor -- a 56% overshoot. With a tokenizer we cap EXACTLY (verified: 31,999
        and 31,998 tokens for a 32k budget on 2.77 and 1.91 ch/tok inputs). Without one we fall back to a
        conservative chars-per-token constant, which is best-effort only and can still overshoot on
        atypical text -- so supply a tokenizer whenever the bound has to hold.

        Split out of ``_render_for_analyzer`` so a caller holding an ALREADY-RENDERED transcript string
        (the checklist constructor's frontier win side, which comes from a hint artifact rather than from
        a TrajView) gets the same exact bound instead of reimplementing it.

        In LEGACY mode (``max_input_tokens`` <= 0) the token budget -- including a ``max_tokens``
        override -- is ignored and the text is chars-capped at ``legacy_max_chars`` (default
        ``max_chars``), exactly as ``TrajView.render`` would have.
        """
        if self.legacy_input_render:
            return self._legacy_chars_cap(text, legacy_max_chars)
        cap = int(max_tokens if max_tokens is not None else self.max_input_tokens)
        tok = self.tokenizer
        if tok is None:
            budget = int(cap * _CHARS_PER_TOKEN_FLOOR)
            return text if len(text) <= budget else (text[: budget // 2] + _TRUNC_MARKER + text[-budget // 2 :])
        ids = _encode_no_special(tok, text)
        if len(ids) <= cap:
            return text
        marker_n = len(_encode_no_special(tok, _TRUNC_MARKER))
        keep = max(1, (cap - marker_n) // 2)
        return tok.decode(ids[:keep]) + _TRUNC_MARKER + tok.decode(ids[-keep:])

    def _render_for_analyzer(
        self,
        tv: "TrajView",
        *,
        strip_think: bool = True,
        max_tokens: Optional[int] = None,
        legacy_max_chars: Optional[int] = None,
    ) -> str:
        """Render one trajectory whole, then cap it at ``max_input_tokens`` (see ``_cap_to_tokens``).

        LEGACY mode (``max_input_tokens`` <= 0): ``tv.render(legacy_max_chars or max_chars)`` -- the
        legacy AppWorld-line render, byte for byte (pinned by tests/test_prompt_parity.py).
        ``legacy_max_chars`` lets a constructor reproduce a per-call chars budget it used to pass to
        ``render`` (contrastive's ``max(max_chars // 2, 1500)`` per side).
        """
        if self.legacy_input_render:
            n = self.max_chars if legacy_max_chars is None else legacy_max_chars
            return tv.render(n, strip_think=strip_think)
        return self._cap_to_tokens(tv.render(max_chars=10**9, strip_think=strip_think), max_tokens)

    @abstractmethod
    async def build_offline(self, task_id: str, traj_views: List["TrajView"]) -> Optional[HintRecord]:
        """Return a HintRecord for one task (or None to skip — falls back to no hint at use time)."""
        raise NotImplementedError

    async def build_online(
        self, task_id: str, traj_views: List["TrajView"], prior: Optional[str] = None
    ) -> Optional[HintRecord]:
        """Online per-group construction, refining a ``prior`` hint in place.

        Default ignores ``prior`` and reuses ``build_offline`` — safe for sources with no online path
        (static / reference). The online sources (contrastive, prompt_skill) override this to fold the
        new group of rollouts into the prior hint. Returning ``None`` means "no update" — callers keep
        the prior hint rather than clobbering it.
        """
        return await self.build_offline(task_id, traj_views)


class StaticHintConstructor(HintConstructor):
    """Trivial PI: one constant hint applied to every task. Validates the rank+guide harness."""

    pi_type = "static"

    _DEFAULT = (
        "Plan before acting: identify the target object(s) and receptacle, check which actions are "
        "admissible, take exactly one valid action per step, and avoid repeating actions that did "
        "not change the observation."
    )

    def __init__(self, analyzer: Optional[Any] = None, *, static_hint: Optional[str] = None, **kwargs: Any):
        super().__init__(analyzer, **kwargs)
        self.static_hint = (static_hint or self._DEFAULT).strip()

    async def build_offline(self, task_id: str, traj_views: List[TrajView]) -> Optional[HintRecord]:
        return HintRecord(task=self.static_hint, meta={"pi_type": self.pi_type})

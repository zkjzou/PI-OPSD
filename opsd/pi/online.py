"""Online (in-training) per-prompt hint construction for OPSD.

Offline hints (``build_hints.py``) are precomputed once and frozen. This module refines a hint
*during* RL: after each rollout batch, for every prompt (grouped by a stable key) we compare/fold the
group's trajectories with an analyzer LLM and update a running hint, which is injected back into that
prompt's context the next time it is sampled. Inspired by "Online Rubrics Elicitation from Pairwise
Comparisons" (arXiv 2510.07284), but the artifact feeds the policy PROMPT (a hint), not the reward
grader — i.e. in-context self-improvement from the policy's own rollouts.

This module is generator/trainer-agnostic: it only needs, per trajectory, the token ids
(prompt/response), the loss mask, a scalar reward, and the env_extras (for the grouping key). The
ALFWorld ``OnlineHintSkyRLGymGenerator`` wires it in (inject in ``_setup_env_extras``, update after
``generate``); the reconstruction here is what lets us reuse the offline ``pi/`` constructors unchanged
even though ``GeneratorOutput`` carries only tokens (no chat history).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from loguru import logger

from opsd.pi import build_constructor
from opsd.pi.analyzer import AnalyzerConfigError, make_analyzer
from opsd.pi.artifact import save_hint_artifact
from opsd.pi.base import HintRecord, TrajView

# Marker that begins the injected hint block (see ``opsd.envs.alfworld.env._HINT_TEMPLATE``). Kept as a
# literal here so ``pi`` stays decoupled from the env code; used to strip the prior hint out of the decoded
# first-turn prompt so we don't feed it back to the analyzer as if it were part of the trajectory.
_HINT_BLOCK_MARKER = "A reviewer left the following hint"
_ROLE_PREFIX_RE = re.compile(r"^(assistant|user|system)\s*\n?", re.IGNORECASE)


# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------
@dataclass
class OnlineHintConfig:
    enabled: bool = False
    pi_type: str = "contrastive"  # contrastive | prompt_skill
    key_mode: str = "task_type"  # task_type | task_id | uid
    update_every: int = 1  # run the update every K training steps
    max_groups_per_step: int = 0  # cap groups updated per step (0 = all); excess logged
    fold_per_step: int = 2  # prompt_skill: trajectories folded per group per visit
    success_threshold: float = 1.0  # scalar reward >= this counts as success
    max_concurrency: int = 8  # concurrent analyzer calls across groups
    max_chars: int = 6000  # OUTPUT cap: hint length (scrub_hint)
    max_input_tokens: int = 32000  # INPUT cap: analyzer prompt budget in TOKENS (token-exact
    # truncation via the tokenizer; see base._render_for_analyzer).
    # 0 = LEGACY (original AppWorld line): max_chars caps the analyzer
    # INPUT render too -- set OPSD_ONLINE_HINTS_MAX_INPUT_TOKENS=0 to
    # reproduce the AppWorld online-hint runs (MAX_CHARS=48000).
    analyzer_max_tokens: int = 32000  # OUTPUT cap for the analyzer completion (incl. thinking).
    # See constructor_kwargs for why 1024 is a trap. 32000 reproduces
    # both the ALFWorld online arms and AppWorld (whose launch script
    # set OPSD_ONLINE_HINTS_ANALYZER_MAX_TOKENS=32000 explicitly).
    # For backend=policy the analyzer shares the POLICY's context, so
    # max_input_tokens + analyzer_max_tokens must fit max_model_len or
    # vLLM 400s every call at validation (prompt+max_tokens>window).
    # Frontier models have far larger windows and are unaffected.
    inject_eval: bool = False  # also inject hints during eval rollouts
    per_traj: bool = False  # onpolicy_skill: one skill PER ROLLOUT (key by rep) vs one per group
    analyzer_backend: str = "frontier"  # frontier | policy | endpoint (see pi/analyzer.py)
    analyzer_model: Optional[str] = None
    analyzer_base_url: Optional[str] = None
    analyzer_api_key: Optional[str] = None
    # Per-call HTTP timeout and retry count. Both were unreachable before (pinned at 180s/4). The timeout
    # MUST scale with analyzer_max_tokens: at the ~54 tok/s measured on the AppWorld online-hint runs, a
    # 16384-token call takes ~300s, so 180s would kill precisely the long deliberations the big budget is
    # meant to allow.
    analyzer_timeout: int = 600  # OPSD_ONLINE_HINTS_ANALYZER_TIMEOUT
    analyzer_retries: int = 6  # OPSD_ONLINE_HINTS_ANALYZER_RETRIES (litellm num_retries)
    # Analyzer SAMPLING. None => omit the param entirely => the served model applies its OWN default,
    # which is what we want from a frontier analyzer: qwen3.7-plus and glm-5.2 ship tuned defaults and a
    # hand-set 0.3 is just an untested third option. The old behaviour was a hardcoded temperature=0.3 in
    # every analyzer-backed constructor with no way to reach it from the online path.
    analyzer_temperature: Optional[float] = None  # OPSD_ONLINE_HINTS_ANALYZER_TEMPERATURE
    # Thinking, via litellm/Fireworks-native `reasoning_effort` ('low'|'medium'|'high'). The older
    # `chat_template_kwargs` route is REJECTED by Fireworks. Reasoning returns out-of-band in
    # `message.reasoning_content`, so the hint content stays clean, and Analyzer.analyze additionally
    # strips any inline <think> as a safety net. Empty string => omit => thinking off.
    # NB this raises completion tokens and therefore cost; watch online_hints/cost_usd_per_hint.
    analyzer_reasoning_effort: str = "high"  # OPSD_ONLINE_HINTS_ANALYZER_REASONING_EFFORT
    dump_path: str = ""  # optional: dump the store to this JSON after each update
    snapshot_dir: str = ""  # OPSD_ONLINE_HINTS_SNAPSHOT_DIR: per-update step-stamped
    # snapshots of the whole store (the PI history for the run)
    reference_mode: str = "actions"  # pi_type=reference density: actions | abstracted | trajectory
    reference_strip_think: bool = True  # reference mode=trajectory: strip <think> (True) or keep it (False)
    fallback_hints_path: str = ""  # static hints file (e.g. qwen3.7 hints_actions.json) used for a task
    # until it has a self-hint (i.e. a successful student rollout)
    strict: bool = False  # OPSD_ONLINE_HINTS_STRICT: re-raise analyzer/skill-build failures
    # (fail the job) instead of swallowing them + degrading to fallback/empty.
    # Analyzer CONFIG errors (no model, ...) always raise, strict or not.

    def constructor_kwargs(self) -> Dict[str, Any]:
        kw: Dict[str, Any] = {
            "max_chars": self.max_chars,
            "max_input_tokens": self.max_input_tokens,
            "max_tokens": self.analyzer_max_tokens,
        }
        if self.pi_type in ("prompt_skill", "group_skill"):
            kw["fold_per_step"] = self.fold_per_step
        if self.pi_type == "reference":
            kw["mode"] = self.reference_mode  # 'actions' -> extract_actions -> literal action list (no analyzer)
            kw["strip_think"] = self.reference_strip_think  # only used by mode='trajectory'
        if self.pi_type == "onpolicy_skill":
            # Reuses OPSD_ONLINE_HINTS_STRIP_THINK. TrajView.render defaults strip_think=True and
            # onpolicy_skill called it positionally, so the analyzer never saw the agent's reasoning --
            # but the ALFWorld PI generation that produced the first offline skill results KEPT think.
            # Make it settable so a run can match that instead of silently differing.
            kw["strip_think"] = self.reference_strip_think
        # Analyzer OUTPUT budget. Every analyzer-backed constructor defaults to max_tokens=1024, and
        # the online path never plumbed the knob -- so the first AppWorld online-hint arms ran end-to-end
        # at 1024 while the offline driver has taken --max-tokens (4000 for frontier analyzers)
        # since the same trap was hit there on 2026-07-29. 1024 fails SILENTLY: a thinking analyzer
        # spends the budget reasoning, the JSON is cut mid-object, parse_analyzer_json returns None,
        # and _refine_one returns False with only a logger.warning that never reaches the driver's logs.
        # Measured 2026-08-02, qwen3.7-plus on 16 real 48k-char AppWorld failure
        # trajectories: 0/16 parsed at max_tokens=1024 (every one finish_reason='length'), 14/16 at
        # 4096. NB the usual "frontier models return reasoning out-of-band in reasoning_content, so
        # 1024 only has to hold the JSON" reasoning does NOT hold here -- on these long prompts
        # qwen3.7 switched to deliberating IN-BAND (reasoning_content empty, content full of
        # "Wait, does show_album_library return song_ids?"), so the budget must cover the thinking.
        # (The base kw above already carries max_tokens for EVERY type, episode_skill/checklist included;
        # this re-assignment is the same value and is kept only so the rationale stays next to the knob.)
        if self.pi_type in ("onpolicy_skill", "contrastive", "reference", "prompt_skill", "group_skill"):
            kw["max_tokens"] = self.analyzer_max_tokens
        if self.pi_type == "onpolicy_skill":
            # Sampling + thinking, only for the constructor whose signature accepts them. Passing these
            # blindly to every analyzer-backed type would TypeError on the ones that do not take
            # reasoning_effort, so widen this deliberately per type rather than by default.
            kw["temperature"] = self.analyzer_temperature
            kw["reasoning_effort"] = self.analyzer_reasoning_effort or None
        return kw


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def online_hint_cfg_from_env() -> OnlineHintConfig:
    """Build an :class:`OnlineHintConfig` from ``OPSD_ONLINE_HINTS_*`` env vars."""
    g = os.environ.get
    return OnlineHintConfig(
        enabled=_env_bool("OPSD_ONLINE_HINTS_ENABLED", False),
        pi_type=(g("OPSD_ONLINE_HINTS_PI_TYPE", "contrastive") or "contrastive").strip(),
        key_mode=(g("OPSD_ONLINE_HINTS_KEY", "task_type") or "task_type").strip(),
        update_every=int(g("OPSD_ONLINE_HINTS_UPDATE_EVERY", "1") or 1),
        max_groups_per_step=int(g("OPSD_ONLINE_HINTS_MAX_GROUPS", "0") or 0),
        fold_per_step=int(g("OPSD_ONLINE_HINTS_FOLD_PER_STEP", "2") or 2),
        success_threshold=float(g("OPSD_ONLINE_HINTS_SUCCESS_THRESHOLD", "1.0") or 1.0),
        max_concurrency=int(g("OPSD_ONLINE_HINTS_MAX_CONCURRENCY", "8") or 8),
        max_chars=int(g("OPSD_ONLINE_HINTS_MAX_CHARS", "6000") or 6000),
        # unset/empty -> 32000; an explicit "0" selects the legacy max_chars input render.
        max_input_tokens=int(g("OPSD_ONLINE_HINTS_MAX_INPUT_TOKENS", "") or "32000"),
        analyzer_max_tokens=int(g("OPSD_ONLINE_HINTS_ANALYZER_MAX_TOKENS", "32000") or 32000),
        inject_eval=_env_bool("OPSD_ONLINE_HINTS_INJECT_EVAL", False),
        per_traj=_env_bool("OPSD_ONLINE_HINTS_PER_TRAJ", False),
        analyzer_backend=(g("OPSD_ONLINE_HINTS_ANALYZER_BACKEND", "frontier") or "frontier").strip(),
        analyzer_model=g("OPSD_ONLINE_HINTS_ANALYZER_MODEL") or None,
        analyzer_base_url=g("OPSD_ONLINE_HINTS_ANALYZER_BASE_URL") or None,
        analyzer_api_key=g("OPSD_ONLINE_HINTS_ANALYZER_API_KEY") or None,
        analyzer_timeout=int(g("OPSD_ONLINE_HINTS_ANALYZER_TIMEOUT", "600") or 600),
        analyzer_retries=int(g("OPSD_ONLINE_HINTS_ANALYZER_RETRIES", "6") or 6),
        # UNSET (or empty) => None => the param is omitted => the model's own default sampling applies.
        # An explicit value is honoured, including 0.0, which is why this tests for emptiness rather than
        # falsiness: `or None` would silently turn a deliberate temperature=0.0 into "model default".
        analyzer_temperature=(
            float(g("OPSD_ONLINE_HINTS_ANALYZER_TEMPERATURE", "") or 0.0)
            if (g("OPSD_ONLINE_HINTS_ANALYZER_TEMPERATURE", "") or "").strip() != ""
            else None
        ),
        analyzer_reasoning_effort=(g("OPSD_ONLINE_HINTS_ANALYZER_REASONING_EFFORT", "high") or "").strip(),
        dump_path=(g("OPSD_ONLINE_HINTS_DUMP", "") or "").strip(),
        snapshot_dir=(g("OPSD_ONLINE_HINTS_SNAPSHOT_DIR", "") or "").strip(),
        reference_mode=(g("OPSD_ONLINE_HINTS_REFERENCE_MODE", "actions") or "actions").strip(),
        reference_strip_think=_env_bool("OPSD_ONLINE_HINTS_STRIP_THINK", True),
        fallback_hints_path=(g("OPSD_ONLINE_HINTS_FALLBACK_PATH", "") or "").strip(),
        strict=_env_bool("OPSD_ONLINE_HINTS_STRICT", False),
    )


# --------------------------------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------------------------------
class OnlineHintStore:
    """In-memory ``key -> HintRecord`` map, persisting across training steps on the driver."""

    def __init__(self, dump_path: str = "", snapshot_dir: str = ""):
        self._by_key: Dict[str, HintRecord] = {}
        self._dump_path = dump_path
        # Per-update SNAPSHOT dir. dump_path alone overwrites one file, so only the final state of the
        # store survives -- useless for an on-policy analyzer experiment, where the whole question is how
        # a task's hint EVOLVES as the policy changes. A step-stamped file per update keeps that history.
        self._snapshot_dir = snapshot_dir

    def __len__(self) -> int:
        return len(self._by_key)

    def get_hint(self, key: str) -> str:
        rec = self._by_key.get(key)
        return (rec.task or "") if rec else ""

    def put(self, key: str, rec: HintRecord) -> None:
        self._by_key[key] = rec

    def dump(self, global_step: Optional[int] = None) -> None:
        if self._dump_path:
            try:
                save_hint_artifact(self._dump_path, self._by_key)
            except Exception as exc:  # dump is best-effort / debug only
                logger.warning(f"OnlineHintStore: dump to {self._dump_path} failed: {exc}")
        # Step-stamped snapshot of the WHOLE store after this update. Best-effort like the dump above: a
        # write failure must never kill a training step. Zero-padded so lexical order == step order.
        if self._snapshot_dir and global_step is not None:
            try:
                snap = os.path.join(self._snapshot_dir, f"hints_step_{int(global_step):05d}.json")
                save_hint_artifact(snap, self._by_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"OnlineHintStore: snapshot to {self._snapshot_dir} failed: {exc}")


# --------------------------------------------------------------------------------------------------
# Grouping key + trajectory reconstruction
# --------------------------------------------------------------------------------------------------
def _extra(env_extras: Any, field_name: str) -> Optional[str]:
    """Read a field from env_extras, tolerating either top-level or nested ``extra_info`` placement."""
    if not isinstance(env_extras, dict):
        return None
    val = env_extras.get(field_name)
    if val:
        return str(val)
    ei = env_extras.get("extra_info")
    if isinstance(ei, dict) and ei.get(field_name):
        return str(ei[field_name])
    return None


def resolve_key(env_extras: Any, instance_id: Optional[str], key_mode: str) -> str:
    """Stable grouping/injection key. Falls back to task_id then uid when a field is missing."""
    uid = str(instance_id) if instance_id is not None else ""
    if key_mode == "uid":
        return uid or "unknown"
    if key_mode == "task_id":
        return _extra(env_extras, "task_id") or uid or "unknown"
    # default: task_type (cross-prompt transfer within a task family) -> task_id -> uid
    return _extra(env_extras, "task_type") or _extra(env_extras, "task_id") or uid or "unknown"


def _decode(tokenizer, ids: Sequence[int]) -> str:
    if not len(ids):
        return ""
    return tokenizer.decode(list(ids), skip_special_tokens=True)


def _strip_hint_block(text: str) -> str:
    idx = text.find(_HINT_BLOCK_MARKER)
    return text[:idx].rstrip() if idx != -1 else text


def _mask_runs(response_ids: Sequence[int], loss_mask: Sequence[int]):
    """Yield ``(role, token_ids)`` for each contiguous equal-mask run (1 -> assistant, 0 -> user/obs)."""
    n = min(len(response_ids), len(loss_mask))
    i = 0
    while i < n:
        m = 1 if loss_mask[i] else 0
        j = i
        while j < n and (1 if loss_mask[j] else 0) == m:
            j += 1
        yield ("assistant" if m == 1 else "user"), list(response_ids[i:j])
        i = j


def trajview_from_tokens(
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    loss_mask: Sequence[int],
    reward: Union[float, Sequence[float]],
    repetition_id: int,
    tokenizer,
    task_id: str,
    success_threshold: float = 1.0,
) -> TrajView:
    """Reconstruct a :class:`TrajView` from a single trajectory's tokens + loss mask.

    ``GeneratorOutput`` drops the chat history, so we rebuild a role-tagged message list by decoding the
    prompt (first user turn, hint block stripped) and splitting the response by loss-mask runs. Good
    enough for an analyzer to read; boundaries are token-aligned so per-run decode is lossless per run.
    """
    user0 = _strip_hint_block(_decode(tokenizer, prompt_ids)).strip()
    messages: List[Dict[str, str]] = [{"role": "user", "content": user0}]
    for role, seg in _mask_runs(response_ids, loss_mask):
        text = _ROLE_PREFIX_RE.sub("", _decode(tokenizer, seg).strip()).strip()
        if text:
            messages.append({"role": role, "content": text})
    scalar = float(sum(reward)) if isinstance(reward, (list, tuple)) else float(reward)
    num_turns = sum(1 for m in messages if m["role"] == "assistant")
    return TrajView(
        task_id=task_id,
        messages=messages,
        reward=scalar,
        success=scalar >= success_threshold,
        num_turns=num_turns,
        repetition_id=int(repetition_id),
    )


# --------------------------------------------------------------------------------------------------
# Manager
# --------------------------------------------------------------------------------------------------
@dataclass
class _Group:
    key: str
    task_id: str
    views: List[TrajView] = field(default_factory=list)


class OnlineHintManager:
    """Owns the store, analyzer and constructor; injects hints and refines them per training step."""

    def __init__(self, cfg: OnlineHintConfig, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.store = OnlineHintStore(cfg.dump_path, cfg.snapshot_dir)
        self._analyzer = make_analyzer(
            cfg.analyzer_backend,
            model=cfg.analyzer_model,
            base_url=cfg.analyzer_base_url,
            api_key=cfg.analyzer_api_key,
            request_timeout=cfg.analyzer_timeout,
            max_retries=cfg.analyzer_retries,
        )
        # tokenizer -> token-EXACT analyzer input truncation (chars cannot bound tokens: 2.24-3.70 ch/tok)
        self._constructor = build_constructor(
            cfg.pi_type, self._analyzer, tokenizer=tokenizer, **cfg.constructor_kwargs()
        )
        self._sem = asyncio.Semaphore(max(1, cfg.max_concurrency))
        # static fallback (e.g. qwen3.7 hints_actions): used for a task until it earns a self-hint.
        self._fallback = self._load_fallback(cfg.fallback_hints_path)
        self._n_fallback = 0  # rollouts this window that fell back to the static hint (no self-hint yet)
        self._n_hint_calls = 0  # total hint_for() calls this window (denominator for the fallback fraction)
        logger.info(
            f"OnlineHintManager | pi_type={cfg.pi_type} mode={cfg.reference_mode} key={cfg.key_mode} "
            f"per_traj={cfg.per_traj} update_every={cfg.update_every} max_groups={cfg.max_groups_per_step} "
            f"fallback_hints={len(self._fallback)} analyzer={cfg.analyzer_backend}"
        )

    @staticmethod
    def _load_fallback(path: str) -> Dict[str, str]:
        """Load {task_id -> hint_text} from a static hints artifact (schema opsd_hints/v2: values are
        {"task": <text>, "meta": {...}}). Returns {} if no path/unreadable. Used as the per-task fallback
        until the student produces its own successful trajectory for that task."""
        if not path or not os.path.exists(path):
            return {}
        try:
            raw = json.load(open(path))
        except Exception as exc:
            logger.warning(f"OnlineHint: could not load fallback hints {path}: {exc}")
            return {}
        out: Dict[str, str] = {}
        for tid, v in raw.items():
            if tid == "__schema__":
                continue
            out[tid] = (v.get("task", "") if isinstance(v, dict) else str(v)) or ""
        return out

    # -- injection --------------------------------------------------------------------------------
    def _key(self, env_extras: Any, instance_id: Optional[str], rep_id: int = 0) -> str:
        """Storage/injection key. In per-traj mode, append the rollout's repetition id so EVERY rollout
        gets its OWN skill (built from itself); otherwise one skill per group (the default best-pick)."""
        base = resolve_key(env_extras, instance_id, self.cfg.key_mode)
        return f"{base}#rep{int(rep_id)}" if self.cfg.per_traj else base

    def hint_for(self, env_extras: Any, instance_id: Optional[str], rep_id: int = 0) -> str:
        self._n_hint_calls += 1
        h = self.store.get_hint(self._key(env_extras, instance_id, rep_id))
        if h:
            return h
        # no self-hint yet for this task (no successful student rollout) -> static fallback (e.g. qwen3.7).
        if self._fallback:
            tid = _extra(env_extras, "task_id") or resolve_key(env_extras, instance_id, "task_id")
            fb = self._fallback.get(tid, "")
            if fb:
                self._n_fallback += 1
                return fb
        return h  # "" -> upstream generic/static handling

    def meta_for(self, env_extras: Any, instance_id: Optional[str], rep_id: int = 0) -> Dict[str, Any]:
        """Provenance of the hint ``hint_for`` would return for this sample, for the rollout dump.

        The store keeps ``HintRecord.meta`` (pi_type, outcome, and for the analyzer constructors ``rep`` =
        WHICH rollout of the group was analyzed), but ``hint_for`` returns only the hint string, so that
        linkage never reached the dumped rollouts. Emitting it makes each dumped rollout self-describing:
        you can tell whether its hint came from the analyzer or the static fallback, and which sibling
        rollout produced it. ``source`` distinguishes the three cases so an empty hint is not ambiguous.
        """
        key = self._key(env_extras, instance_id, rep_id)
        rec = self.store._by_key.get(key)
        if rec is not None and (rec.task or ""):
            meta = dict(rec.meta or {})
            meta.update({"source": "online", "key": key})
            return meta
        if self._fallback:
            tid = _extra(env_extras, "task_id") or resolve_key(env_extras, instance_id, "task_id")
            if self._fallback.get(tid, ""):
                return {"source": "fallback", "key": key, "task_id": tid}
        return {"source": "none", "key": key}

    # -- update -----------------------------------------------------------------------------------
    def _group(
        self,
        prompt_token_ids: List[List[int]],
        response_ids: List[List[int]],
        loss_masks: List[List[int]],
        rewards: List[Union[float, List[float]]],
        env_extras_list: List[Any],
        instance_ids: List[Optional[str]],
        repetition_ids: List[int],
    ) -> Dict[str, _Group]:
        groups: Dict[str, _Group] = {}
        n = len(response_ids)
        for i in range(n):
            ee = env_extras_list[i] if i < len(env_extras_list) else {}
            inst = instance_ids[i] if i < len(instance_ids) else None
            rep = repetition_ids[i] if i < len(repetition_ids) else 0
            key = self._key(ee, inst, rep)  # per-traj -> unique per rollout; else per group
            task_id = _extra(ee, "task_id") or resolve_key(ee, inst, self.cfg.key_mode)
            view = trajview_from_tokens(
                prompt_token_ids[i],
                response_ids[i],
                loss_masks[i],
                rewards[i],
                repetition_ids[i] if i < len(repetition_ids) else 0,
                self.tokenizer,
                task_id,
                self.cfg.success_threshold,
            )
            groups.setdefault(key, _Group(key=key, task_id=task_id)).views.append(view)
        return groups

    async def _refine_one(self, grp: _Group) -> str:
        """Return the OUTCOME, not a bool: 'ok' | 'error' | 'no_hint'.

        A bare bool conflated "the analyzer call raised" with "the analyzer replied but produced no
        usable hint", and the only distinguishing signal was a logger.warning -- which is emitted from
        the Ray entrypoint actor and therefore never reaches the driver's logs or metrics. That is why the
        first AppWorld online-hint arms could run 77 steps with ~76% of analyzer calls failing and nothing
        anywhere said so. The two causes want opposite fixes (retries/timeouts vs. analyzer max_tokens), so
        surface them separately.

        An :class:`AnalyzerConfigError` (e.g. no analyzer model configured) is re-raised regardless of
        ``cfg.strict``: it fails every group identically, so counting it as one 'error' would let the run
        train with no hints at all.
        """
        prior = self.store.get_hint(grp.key)
        async with self._sem:
            try:
                rec = await self._constructor.build_online(grp.task_id, grp.views, prior=prior)
            except AnalyzerConfigError:
                raise
            except Exception as exc:
                if self.cfg.strict:
                    raise RuntimeError(
                        f"OnlineHint(strict): analyzer/skill build failed for key={grp.key}: {exc}"
                    ) from exc
                logger.warning(f"OnlineHint: refine failed for key={grp.key}: {exc}")
                return "error"
        if rec is not None and rec.task:
            self.store.put(grp.key, rec)
            return "ok"
        return "no_hint"  # None -> keep prior (e.g. no-variance contrastive group, unparseable JSON)

    async def update(
        self,
        prompt_token_ids: List[List[int]],
        response_ids: List[List[int]],
        loss_masks: List[List[int]],
        rewards: List[Union[float, List[float]]],
        env_extras_list: List[Any],
        instance_ids: List[Optional[str]],
        repetition_ids: List[int],
        global_step: int,
    ) -> Dict[str, float]:
        """Group the batch by key, refine each group's hint concurrently, return wandb metrics."""
        if self.cfg.update_every > 1 and (global_step % self.cfg.update_every != 0):
            return {"online_hints/store_size": float(len(self.store))}

        groups = self._group(
            prompt_token_ids, response_ids, loss_masks, rewards, env_extras_list, instance_ids, repetition_ids
        )
        keys = list(groups.keys())
        dropped = 0
        if self.cfg.max_groups_per_step and len(keys) > self.cfg.max_groups_per_step:
            dropped = len(keys) - self.cfg.max_groups_per_step
            keys = keys[: self.cfg.max_groups_per_step]
            logger.info(f"OnlineHint: capping to {self.cfg.max_groups_per_step} groups/step ({dropped} dropped)")

        def _has_variance(g: _Group) -> bool:
            outcomes = {v.success for v in g.views}
            return len(outcomes) > 1

        frac_var = (sum(_has_variance(groups[k]) for k in keys) / len(keys)) if keys else 0.0

        # Usage baseline BEFORE the fan-out, diffed after, so every metric below is this step's delta.
        usage_before = self._analyzer.usage_snapshot()

        t0 = time.monotonic()
        results = await asyncio.gather(*(self._refine_one(groups[k]) for k in keys))
        elapsed = time.monotonic() - t0
        n_updated = int(sum(1 for r in results if r == "ok"))
        n_err = int(sum(1 for r in results if r == "error"))
        n_nohint = int(sum(1 for r in results if r == "no_hint"))

        self.store.dump(global_step)
        # fallback accounting over this window's generation: how many rollouts used the static (qwen3.7)
        # fallback because their task had no self-hint yet. Falls as the student solves more tasks.
        n_fb, n_calls = self._n_fallback, self._n_hint_calls
        self._n_fallback = 0
        self._n_hint_calls = 0
        metrics = {
            "online_hints/store_size": float(len(self.store)),
            "online_hints/groups_seen": float(len(keys)),
            "online_hints/groups_dropped": float(dropped),
            "online_hints/updated_this_step": float(n_updated),
            # Split the non-updates by cause: 'error' = the analyzer call raised (timeout / rate limit /
            # bad key), 'no_hint' = it replied but produced nothing parseable (overwhelmingly the
            # max_tokens truncation above). Without this the two are indistinguishable and both look
            # like a merely-low updated_this_step.
            "online_hints/refine_errors": float(n_err),
            "online_hints/refine_no_hint": float(n_nohint),
            "online_hints/frac_refine_ok": float(n_updated / len(keys)) if keys else 0.0,
            "online_hints/frac_groups_with_variance": float(frac_var),
            "online_hints/analyzer_latency_s": float(elapsed),
            "online_hints/n_fallback": float(n_fb),
            "online_hints/hint_calls": float(n_calls),
            "online_hints/frac_fallback": float(n_fb / n_calls) if n_calls else 0.0,
        }
        metrics.update(self._usage_metrics(usage_before, n_updated))
        logger.info(
            f"OnlineHint step={global_step}: updated {n_updated}/{len(keys)} groups "
            f"(err={n_err} no_hint={n_nohint}), store={len(self.store)}, "
            f"frac_var={frac_var:.2f}, fallbacks={n_fb}/{n_calls}, {elapsed:.1f}s, "
            f"${metrics['online_hints/cost_usd_step']:.3f}"
        )
        return metrics

    def _usage_metrics(self, before: Dict[str, float], n_updated: int) -> Dict[str, float]:
        """Analyzer spend for THIS step, as the delta of two cumulative snapshots.

        WHY THIS EXISTS. The analyzer is the only component that spends real money per training step
        (144 calls/step x 8192 output tokens x a frontier model on the AppWorld online-hint arms) and none
        of it was measured -- `Analyzer.analyze` read `.content` and dropped `resp.usage` on the floor, so
        cost could only be estimated after the fact from provider billing.

        `truncated_step` earns its place independently of cost: it is the direct readout of the failure
        that invalidated the first AppWorld online-hint pair, where a 1024-token budget produced no parseable
        hint for ~76% of rollouts for 77 steps with no error anywhere. A truncated call bills the
        FULL budget and returns nothing, so it shows up here as cost with no matching hint.
        `cost_usd_per_hint` is therefore the number to read: it prices what was actually obtained.

        `cost_unavailable_step` counts calls where no cost source answered (see analyzer._response_cost).
        Non-zero means `cost_usd_step` is an UNDERCOUNT, not that the calls were free.

        NB the `*_total` gauges: SkyRL's concatenate_generator_outputs (skyrl/train/generators/utils.py)
        reduces custom generator metrics by a NAME heuristic -- avg/mean/min/max reduce properly and
        everything else is SUMMED across generator workers. That is right for the `*_step` deltas and wrong for a cumulative gauge. Harmless
        today because AppWorld runs the sync path (one generator, on the driver); it would double-count
        under the fully-async trainer.
        """
        after = self._analyzer.usage_snapshot()
        d = {k: after[k] - before.get(k, 0) for k in after}
        cost_step = float(d["cost_usd"])
        return {
            "online_hints/cost_usd_step": cost_step,
            "online_hints/cost_usd_total": float(after["cost_usd"]),
            "online_hints/cost_usd_per_hint": (cost_step / n_updated) if n_updated else 0.0,
            "online_hints/prompt_tokens_step": float(d["prompt_tokens"]),
            "online_hints/completion_tokens_step": float(d["completion_tokens"]),
            # Expected to sit near ZERO: measured on all three analyzers, automatic prefix caching does not
            # fire on this workload because each call reads a different trajectory and only the ~261-token
            # system prompt is shared -- below the ~1024-token minimum. A sustained non-zero frac here would
            # mean the prompt became cacheable (or a provider changed), and cost should drop accordingly.
            "online_hints/cached_prompt_tokens_step": float(d["cached_prompt_tokens"]),
            "online_hints/frac_prompt_cached": (
                d["cached_prompt_tokens"] / d["prompt_tokens"] if d["prompt_tokens"] else 0.0
            ),
            "online_hints/calls_step": float(d["calls"]),
            "online_hints/calls_total": float(after["calls"]),
            "online_hints/truncated_step": float(d["truncated"]),
            "online_hints/frac_truncated": (d["truncated"] / d["calls"]) if d["calls"] else 0.0,
            "online_hints/cost_unavailable_step": float(d["cost_unavailable"]),
        }

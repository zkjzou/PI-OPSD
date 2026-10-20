"""Offline driver: build a PI hint artifact from a logged trajectories.jsonl.

Generic over the PI sources (``--pi-type static|reference|onpolicy_skill|contrastive|checklist``); a
single driver replaces several near-identical per-source scripts. Groups trajectories by ``task_id``,
runs the constructor per task (concurrently), and writes a ``{task_id: hint}`` artifact (drop-in for
``OPSD_HINTS_PATH``). Several artifacts can be merged task-wise (a plain dict update) if desired.

``checklist`` additionally needs ``--win-hints`` (the frontier WIN artifact supplying the success side)
and ``--bench`` (which environment the analyzer is told it is reading). It does NOT emit the paired
length-matched control — for that, and for AppWorld/ALFWorld pool schemas that this driver's
``trajview_from_dump`` mislabels, use ``scripts/analysis/build_checklist_hints.py`` instead.

``--bench`` and ``--prompt-env`` are ONE flag (same dest): both name the environment the analyzer is
told it is reading. It selects the checklist's environment blurb + task marker, and the onpolicy_skill
prompt pair (``prompts.skill_prompts``). Unset: checklist uses ALFWorld; onpolicy_skill falls back to
``$OPSD_PI_PROMPT_ENV`` and then ALFWorld.

Example:
  python -m opsd.pi.build_hints \
    --pi-type onpolicy_skill \
    --trajectories ~/alfworld_gen_base/generation/trajectories.jsonl \
    --analyzer-backend frontier \
    --out ~/data/alfworld_pi/hints_onpolicy_skill.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from opsd.pi import build_constructor, load_task_hints, save_hint_artifact
from opsd.pi.analyzer import AnalyzerConfigError, make_analyzer
from opsd.pi.base import TrajView

logger = logging.getLogger("pi.build_hints")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_TURN_RE = re.compile(r"<\|im_start\|>\s*(\w+)\s*\n?(.*?)<\|im_end\|>", re.DOTALL)


def messages_from_dump(rec: Dict[str, Any]) -> List[Dict[str, str]]:
    """Reconstruct role-tagged turns from a trajectories.jsonl record.

    Parses the chat-template markers in ``prompt_decoded`` + ``decoded_response`` (skip_special=False
    dumps). Falls back to a coarse [user(task), assistant(all-text)] view if markers are absent
    (non-Qwen template). Only used for analyzer INPUT (hint quality), never for scoring correctness.
    """
    # AppWorld pass@k dumps: {task_id, sample, reward, turns, conversation}. `conversation` IS a
    # role-tagged list, but it is NOT the policy's trajectory -- the official scaffold primes the model
    # with a 10-exchange worked DEMO first (measured invariant on these corpora: assistant_msgs -
    # len(turns) == 10 exactly, at min/p50/max). Handing the demo to the analyzer as if the agent had
    # written it is worse than useless. adapters.appworld.generated_span() derives the boundary from
    # len(turns), so reuse it rather than re-deriving. The task text still comes from the supervisor
    # message at conversation[0], which is why it is prepended.
    conv = rec.get("conversation")
    if isinstance(conv, list) and conv:
        from opsd.pi.adapters.appworld import generated_span

        n_turns = len(rec.get("turns") or [])
        span = generated_span(conv, n_turns)
        body = [
            {"role": str(m.get("role", "")), "content": str(m.get("content", "")).strip()}
            for m in conv[span:]
            if isinstance(m, dict) and str(m.get("content", "")).strip()
        ]
        if body:
            head = [{"role": "user", "content": str(conv[0].get("content", "")).strip()}]
            return head + body
    # Native messages-format dumps (e.g. the frontier trajectories_{qwen37,glm}.jsonl) already carry
    # role-tagged turns as a list — use them directly (no chat-template markers to parse).
    native = rec.get("messages")
    if isinstance(native, list) and native:
        out = [
            {"role": str(m.get("role", "")), "content": str(m.get("content", "")).strip()}
            for m in native
            if isinstance(m, dict) and str(m.get("content", "")).strip()
        ]
        if out:
            return out
    text = (
        (rec.get("prompt_decoded") or "") + (rec.get("decoded_response") or "")
        # SkyRL eval dumps (ALFWorld): same chat-marker format under different field names.
        or (rec.get("input_prompt") or "") + (rec.get("output_response") or "")
    )
    turns = [{"role": role, "content": content.strip()} for role, content in _TURN_RE.findall(text) if content.strip()]
    if turns:
        return turns
    msgs: List[Dict[str, str]] = []
    prompt_text = rec.get("prompt_decoded") or rec.get("prompt_text") or ""
    if prompt_text:
        msgs.append({"role": "user", "content": prompt_text})
    assistant_text = rec.get("assistant_text")
    if assistant_text:
        msgs.append({"role": "assistant", "content": str(assistant_text)})
    return msgs


def _task_id(rec: Dict[str, Any]) -> str:
    tid = rec.get("task_id")
    if tid:
        return str(tid)
    extra = rec.get("extra_info") or {}
    if extra.get("task_id"):
        return str(extra["task_id"])
    # SkyRL eval dumps (ALFWorld) nest it one level deeper. Missing this collapses every rollout in the
    # corpus into a single "unknown" group.
    nested = (rec.get("env_extras") or {}).get("extra_info") or {}
    if nested.get("task_id"):
        return str(nested["task_id"])
    uid = str(rec.get("uid", ""))
    if ":" in uid:
        return uid.split(":", 1)[1]
    game = rec.get("game_file")
    return Path(str(game)).parent.name if game else (uid or "unknown")


def trajview_from_dump(rec: Dict[str, Any], success_threshold: float = 10.0) -> TrajView:
    em = rec.get("env_metrics") or {}
    # `won` (frontier dumps) | env_metrics.success | reward >= threshold (env-reward dumps).
    # THE THRESHOLD IS NOT UNIVERSAL. ALFWorld pays 10.0*won - 0.1*invalid so 10.0 is right there, but
    # AppWorld under REWARD_MODE=binary pays 1.0, and its pass@128 transcripts carry neither `won` nor
    # `env_metrics` -- so a hardcoded 10.0 silently labels EVERY AppWorld rollout a FAILURE. That is not
    # a degraded result, it is a wrong one: the analyzer is handed "Outcome: FAILURE" for real successes
    # and any wins-first selection picks nothing. Callers must set --success-threshold to match the dump;
    # build() aborts if the rule yields zero wins corpus-wide.
    # SkyRL eval dumps (ALFWorld) carry `score` as a PER-TOKEN list, not a scalar reward. The rule is MAX,
    # not SUM: ALFWorld pays 10.0 on the winning token and -0.1 per invalid action, so a genuine win
    # with >=100 invalid actions SUMS below 10 and would be labelled a failure. Measured on
    # alfworld_train200_30bi (25,600 rollouts): max>=10 gives pass@1 0.2634, matching the scored
    # parquet exactly, while sum>=10 gives 0.1087 -- a 2.4x undercount.
    sc = rec.get("score")
    if isinstance(sc, list) and sc:
        success = max(float(x) for x in sc) >= success_threshold
        reward = max(float(x) for x in sc)
    else:
        reward = float(rec.get("reward", 0.0) or 0.0)
        success = bool(rec.get("won")) or bool(em.get("success", 0.0)) or reward >= success_threshold
    # `repetition_id` is absent from the AppWorld pass@k dumps, which name the sample index `sample`.
    # Falling back to 0 collapses every per-traj key to `{task}#rep0`, so a dict keyed on it keeps ONE
    # rollout per task and silently drops the rest.
    rep = rec.get("repetition_id")
    if rep is None:
        rep = rec.get("sample", 0)
    n_turns = em.get("num_turns", rec.get("num_turns"))
    if n_turns is None:
        t = rec.get("turns")
        n_turns = len(t) if isinstance(t, list) else 0
    return TrajView(
        task_id=_task_id(rec),
        messages=messages_from_dump(rec),
        reward=reward,
        success=success,
        num_turns=int(n_turns or 0),
        repetition_id=int(rep or 0),
        raw=rec,
    )


def load_trajectories(paths: List[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"trajectories file not found: {p}")
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--pi-type", required=True, choices=("static", "reference", "onpolicy_skill", "contrastive", "checklist")
    )
    ap.add_argument("--trajectories", action="append", required=True, help="trajectories.jsonl (repeatable).")
    ap.add_argument("--out", required=True, help="Output hints artifact (.json).")
    ap.add_argument("--analyzer-backend", default="frontier", choices=("frontier", "policy", "endpoint"))
    ap.add_argument(
        "--analyzer-model",
        default=None,
        help="Analyzer model (default $OPSD_PI_ANALYZER_MODEL; no built-in default). A litellm "
        "provider string such as fireworks_ai/<model>, or a model served at the base URL. "
        "See opsd/pi/analyzer.py::make_analyzer for the routing rules.",
    )
    ap.add_argument(
        "--analyzer-base-url",
        default=None,
        help="OpenAI-compatible /v1 base URL (default $OPSD_PI_ANALYZER_BASE_URL). The frontier "
        "and endpoint backends then fall back to $OPENAI_BASE_URL (frontier: only for "
        "non-provider model names); the policy backend never falls back.",
    )
    ap.add_argument(
        "--analyzer-api-key-env",
        default=None,
        help="NAME of the env var holding the analyzer key (default "
        "$OPSD_PI_ANALYZER_API_KEY_ENV, else OPENAI_API_KEY; a native provider route such "
        "as fireworks_ai/<model> uses the provider's own variable unless this is set).",
    )
    ap.add_argument(
        "--analyzer-api-key",
        default=None,
        help="Prefer --analyzer-api-key-env / $OPSD_PI_ANALYZER_API_KEY over a key on the " "command line.",
    )
    ap.add_argument("--max-chars", type=int, default=6000)
    # WAS a hard 1024, which is a silent-zero-hints trap: a thinking analyzer spends the budget on
    # reasoning, the JSON gets cut mid-object, parse_analyzer_json rejects it, and EVERY task is
    # "skipped". None now means "use the constructor's own default" -- 1024 for reference/skill/
    # contrastive (unchanged behaviour for existing callers) and 8192 for checklist.
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Analyzer OUTPUT budget. Omit to use the constructor default (checklist: 8192).",
    )
    ap.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="Analyzer INPUT budget in tokens (token-exact with a tokenizer, else a chars-per-"
        "token floor). Omit to use the constructor default (onpolicy_skill: 0; the "
        "others: 32000). 0 = LEGACY: the input is rendered with --max-chars, which then "
        "caps BOTH the analyzer input and the hint (the legacy AppWorld-line behaviour).",
    )
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-concurrency", type=int, default=8)
    ap.add_argument("--reference-mode", default=None, choices=("actions", "abstracted"), help="reference PI density.")
    ap.add_argument("--static-hint", default=None, help="Override the constant hint for --pi-type static.")
    ap.add_argument("--limit-tasks", type=int, default=None, help="Cap tasks for a smoke run.")
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="When EVERY task's analyzer call fails, still write the (empty) artifact and exit 0. "
        "Default: write only the summary and exit non-zero, since an all-failed build is "
        "almost always a wrong endpoint / key / model, not a property of the data.",
    )
    # ONE flag, two spellings: --bench (ALFWorld line, checklist) and --prompt-env (AppWorld line,
    # onpolicy_skill) both name the environment the analyzer is told it is reading. Default None so the
    # onpolicy_skill path keeps its $OPSD_PI_PROMPT_ENV fallback; the checklist path resolves None to
    # "alfworld", its historical default.
    ap.add_argument(
        "--bench",
        "--prompt-env",
        dest="bench",
        default=None,
        choices=("alfworld", "appworld", "tau2"),
        help="Selects the analyzer's environment framing: the checklist's environment blurb + "
        "task-description marker, and the onpolicy_skill prompt pair. An unset/unknown "
        "value SILENTLY falls back to ALFWorld wording (onpolicy_skill first consults "
        "$OPSD_PI_PROMPT_ENV), which briefs the analyzer on a household robot -- set this.",
    )
    ap.add_argument(
        "--win-hints",
        default=None,
        help="[checklist] Hint artifact holding the frontier WIN transcripts (e.g. "
        "hints_glm52/hints_raw.json). Its text IS the success side, so the checklist "
        "and that demonstration arm carry byte-identical privileged information.",
    )
    ap.add_argument("--max-items", type=int, default=5, help="[checklist] Max items per section.")
    ap.add_argument(
        "--per-traj",
        action="store_true",
        help="ONE HINT PER ROLLOUT. Keys the artifact `{task_id}#rep{n}` -- byte-identical to "
        "the online path's key (online.py:359), so an artifact built here is a drop-in "
        "for a run that trained with OPSD_ONLINE_HINTS_PER_TRAJ=1. Without this the "
        "constructor's _pick() collapses a task's rollouts to ONE before analysing, so "
        "N-1 of every N sampled rollouts never reach the analyzer. "
        "NB at per-traj each hint is distilled from THE SAME rollout it will be scored "
        "against, so any resulting `d` measures hindsight FIT, not transfer -- it is not "
        "comparable to an offline table that splices a hint from a DIFFERENT trajectory.",
    )
    ap.add_argument(
        "--max-per-task", type=int, default=None, help="Sample at most N trajectories per task (see --select)."
    )
    ap.add_argument(
        "--select",
        default="best-wins-first",
        choices=("best-wins-first", "random"),
        help="best-wins-first (default): wins sorted by (reward desc, num_turns ASC) fill the "
        "budget first -- the reference.py:48 'cleanest success' rule -- then losses top it "
        "up at random. random: uniform over all rollouts, which is length-unbiased and "
        "worth preferring if the analysis will be compared against length.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Seed for the loss top-up / random selection.")
    ap.add_argument(
        "--synthesize-rep",
        action="store_true",
        help="If the dump has no per-sample index (SkyRL eval dumps do not), derive "
        "repetition_id from within-task FILE ORDER. Stable for a given dump. Without "
        "this, --per-traj aborts rather than silently collapsing to 1 hint/task.",
    )
    ap.add_argument(
        "--success-threshold",
        type=float,
        default=10.0,
        help="reward >= this counts as SUCCESS when the dump has no `won`/`env_metrics`. "
        "ALFWorld: 10.0 (10*won - 0.1*invalid). AppWorld REWARD_MODE=binary: 1.0. "
        "Getting this wrong labels every rollout a failure -- build() aborts on 0 wins.",
    )
    ap.add_argument(
        "--strip-think",
        type=int,
        default=1,
        help="1 (default) strips the student's <think> from the analyzer INPUT; the OPSD "
        "onskill arms ran STRIP_THINK=0, i.e. reasoning kept.",
    )
    ap.add_argument(
        "--reasoning-effort",
        default=None,
        choices=("low", "medium", "high"),
        help="Turns the analyzer's thinking on the litellm-native way.",
    )
    ap.add_argument(
        "--request-timeout",
        type=int,
        default=None,
        help="Analyzer HTTP timeout (s). make_analyzer defaults to 180, which its own "
        "docstring warns is too short at large --max-tokens: measured ~54 tok/s on the "
        "AppWorld online-hint runs, so an 8192-token call needs ~150s and a long deliberation "
        "times out -- i.e. exactly the calls a raised budget exists to rescue. The OPSD "
        "arms ran 1200.",
    )
    ap.add_argument("--max-retries", type=int, default=None, help="Analyzer retries (OPSD arms: 6).")
    ap.add_argument(
        "--model-default-sampling",
        action="store_true",
        help="Omit temperature entirely so the served analyzer applies its own default "
        "(what the OPSD arms did); otherwise --temperature is sent.",
    )
    return ap.parse_args()


async def build(args: argparse.Namespace) -> Dict[str, Any]:
    import random as _random

    records = load_trajectories(args.trajectories)
    by_task: Dict[str, List[TrajView]] = defaultdict(list)
    for rec in records:
        tv = trajview_from_dump(rec, success_threshold=args.success_threshold)
        by_task[tv.task_id].append(tv)
    n_loaded = sum(len(v) for v in by_task.values())

    n_win_all = sum(1 for v in by_task.values() for t in v if t.success)
    reps = {t.repetition_id for v in by_task.values() for t in v}
    logger.info(
        "Corpus: %d trajectories, %d tasks, %d wins (%.4f), %d distinct repetition_ids",
        n_loaded,
        len(by_task),
        n_win_all,
        n_win_all / max(n_loaded, 1),
        len(reps),
    )
    if args.per_traj and args.pi_type == "checklist":
        raise SystemExit(
            "FATAL: --per-traj is not supported with --pi-type checklist (the checklist "
            "contrasts a frontier WIN from --win-hints with ONE student loss per task)."
        )
    # checklist pools are losses-only by design (the win side comes from --win-hints) and static
    # hints never look at outcomes, so the zero-wins guard only applies to the outcome-aware types.
    if n_win_all == 0 and args.pi_type not in ("checklist", "static"):
        rmax = max((t.reward for v in by_task.values() for t in v), default=0.0)
        raise SystemExit(
            f"FATAL: --success-threshold {args.success_threshold} yields ZERO wins over {n_loaded} "
            f"trajectories (max reward seen {rmax}). The analyzer would be told every rollout FAILED. "
            f"For AppWorld REWARD_MODE=binary pass --success-threshold 1.0."
        )
    if args.per_traj and len(reps) == 1 and args.synthesize_rep:
        # SkyRL eval dumps (ALFWorld) carry no per-sample index at all. Assigning it from within-task FILE
        # ORDER is stable for a given dump, which is what makes the artifact keys reproducible.
        for tid in by_task:
            for i, t in enumerate(by_task[tid]):
                t.repetition_id = i
        reps = {t.repetition_id for v in by_task.values() for t in v}
        logger.info("--synthesize-rep: assigned repetition_id from within-task file order " "(%d distinct)", len(reps))
    if args.per_traj and len(reps) == 1:
        raise SystemExit(
            f"FATAL: --per-traj but every trajectory has repetition_id={reps.pop()}, so the key "
            f"`{{task}}#rep{{n}}` collapses to one entry per task and {n_loaded - len(by_task)} rollouts "
            f"would be silently dropped. The dump lacks a per-sample index this loader recognises."
        )

    # SELECTION runs per TASK and before any per-traj re-keying, so the budget is per task either way.
    if args.max_per_task is not None:
        rng = _random.Random(args.seed)
        n_win = n_loss = 0
        for tid in sorted(by_task):
            tvs = by_task[tid]
            if len(tvs) <= args.max_per_task:
                continue
            if args.select == "random":
                pick = rng.sample(tvs, args.max_per_task)
            else:
                # wins first, cleanest (fewest turns) first; losses top up at random
                wins = sorted([t for t in tvs if t.success], key=lambda t: (-t.reward, t.num_turns, t.repetition_id))
                pick = wins[: args.max_per_task]
                if len(pick) < args.max_per_task:
                    losses = [t for t in tvs if not t.success]
                    rng.shuffle(losses)
                    pick += losses[: args.max_per_task - len(pick)]
            by_task[tid] = pick
        n_win = sum(1 for v in by_task.values() for t in v if t.success)
        n_loss = sum(1 for v in by_task.values() for t in v if not t.success)
        logger.info(
            "Selected %d/%d trajectories (%s, <=%d/task): %d wins + %d losses",
            n_win + n_loss,
            n_loaded,
            args.select,
            args.max_per_task,
            n_win,
            n_loss,
        )

    if args.per_traj:
        # SAME key shape as online.py:359 -> f"{task_id}#rep{rep}"
        groups: Dict[str, List[TrajView]] = {
            f"{t.task_id}#rep{t.repetition_id}": [t] for v in by_task.values() for t in v
        }
    else:
        groups = dict(by_task)
    task_ids = sorted(groups)
    if args.limit_tasks is not None:
        task_ids = task_ids[: args.limit_tasks]
    logger.info(
        "Loaded %d trajectories across %d tasks -> %d analyzer groups (using %d)%s",
        n_loaded,
        len(by_task),
        len(groups),
        len(task_ids),
        "  [per-traj: 1 rollout per group]" if args.per_traj else "",
    )

    analyzer = None
    if args.pi_type != "static":
        analyzer = make_analyzer(
            args.analyzer_backend,
            model=args.analyzer_model,
            base_url=args.analyzer_base_url,
            api_key=args.analyzer_api_key,
            api_key_env=args.analyzer_api_key_env,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
        )
        logger.info("Analyzer: backend=%s model=%s", args.analyzer_backend, analyzer.model)

    ctor_kwargs: Dict[str, Any] = {"max_chars": args.max_chars}
    if args.pi_type == "static" and args.static_hint:
        ctor_kwargs["static_hint"] = args.static_hint
    if args.pi_type == "reference" and args.reference_mode:
        ctor_kwargs["mode"] = args.reference_mode
    if args.pi_type in ("reference", "onpolicy_skill", "contrastive", "checklist"):
        # temperature=None means "omit the param" -> the served model's own default sampling.
        ctor_kwargs["temperature"] = None if args.model_default_sampling else args.temperature
        if args.max_tokens is not None:  # else the constructor's own default stands
            ctor_kwargs["max_tokens"] = args.max_tokens
    if args.pi_type != "static" and args.max_input_tokens is not None:
        ctor_kwargs["max_input_tokens"] = args.max_input_tokens
    if args.pi_type in ("onpolicy_skill", "reference"):
        ctor_kwargs["strip_think"] = bool(args.strip_think)
    if args.pi_type == "onpolicy_skill":
        if args.bench:
            ctor_kwargs["prompt_env"] = args.bench
        if args.reasoning_effort:
            ctor_kwargs["reasoning_effort"] = args.reasoning_effort
    win_texts: Dict[str, str] = {}
    if args.pi_type == "checklist":
        if not args.win_hints:
            raise SystemExit("--pi-type checklist requires --win-hints (the frontier WIN artifact).")
        win_texts = load_task_hints(args.win_hints)
        if not win_texts:
            raise SystemExit(f"--win-hints held no task-level hints: {args.win_hints}")
        ctor_kwargs.update(
            win_texts=win_texts,
            bench=args.bench or "alfworld",
            win_source=args.win_hints,
            max_items=args.max_items,
        )
        covered = sum(1 for t in task_ids if win_texts.get(t))
        logger.info(
            "checklist win side: %d tasks in %s; %d/%d pool tasks covered",
            len(win_texts),
            args.win_hints,
            covered,
            len(task_ids),
        )
        if covered == 0:
            raise SystemExit(
                "no pool task has a frontier win text — task-id namespaces almost certainly differ "
                f"between --trajectories and --win-hints (pool sample: {task_ids[:3]}, "
                f"win sample: {sorted(win_texts)[:3]})"
            )
    constructor = build_constructor(args.pi_type, analyzer, **ctor_kwargs)

    sem = asyncio.Semaphore(max(1, args.max_concurrency))
    stats = {"ok": 0, "skipped": 0, "error": 0}

    async def one(tid: str):
        async with sem:
            try:
                rec = await constructor.build_offline(tid, groups[tid])
            except AnalyzerConfigError:
                # A misconfigured analyzer (e.g. no model) fails EVERY task identically; swallowing it
                # would write an empty artifact and exit 0. Abort the whole build instead.
                raise
            except Exception as exc:  # noqa: BLE001 - never let one task kill the batch
                logger.warning("task %s: analyzer/construct error: %s", tid, exc)
                stats["error"] += 1
                return tid, None
            if rec is None or not (rec.task or rec.steps):
                stats["skipped"] += 1
                return tid, None
            stats["ok"] += 1
            return tid, rec

    results = await asyncio.gather(*(one(tid) for tid in task_ids))
    hints = {tid: rec for tid, rec in results if rec is not None}
    # EVERY task errored (e.g. 401 on a wrong/absent provider key): the per-task isolation above would
    # otherwise write an empty artifact and exit 0 -- the same silent failure AnalyzerConfigError exists to
    # prevent. Partial failures keep the per-task isolation; skips (unparseable output) are not errors.
    all_failed = bool(task_ids) and not hints and stats["error"] == len(task_ids)
    if not all_failed or args.allow_empty:
        save_hint_artifact(args.out, hints)
    summary = {
        "pi_type": args.pi_type,
        "analyzer_backend": None if analyzer is None else args.analyzer_backend,
        "analyzer_model": None if analyzer is None else analyzer.model,
        "n_tasks": len(task_ids),
        "n_hints": len(hints),
        **stats,
        "out": str(Path(args.out).expanduser()),
    }
    if args.pi_type == "checklist":
        # Coverage is the number to read first: a checklist arm that silently covers half the pool is
        # no longer paired with the demonstration arm it is meant to be compared against.
        summary.update(
            bench=args.bench or "alfworld",
            win_hints=args.win_hints,
            n_win_texts=len(win_texts),
            win_covered=sum(1 for t in task_ids if win_texts.get(t)),
            max_items=args.max_items,
            max_tokens=args.max_tokens if args.max_tokens is not None else "constructor default (8192)",
        )
    # ANALYZER METERING. Without this the summary cannot distinguish "16 good hints" from "16 calls that
    # burned the full token budget and returned nothing parseable" -- the exact failure that ran for 77
    # steps at max_tokens=1024 with ~76% of rollouts silently hintless. `truncated`
    # (finish_reason == "length") is the one to read: a truncated call bills in full and yields no hint.
    if analyzer is not None:
        summary["analyzer"] = {
            k: getattr(analyzer, k, None)
            for k in (
                "calls",
                "errors",
                "truncated",
                "prompt_tokens",
                "completion_tokens",
                "cost_usd",
                "cost_unavailable",
            )
        }
        c = summary["analyzer"]
        if c.get("calls"):
            summary["analyzer"]["cost_per_hint"] = round((c.get("cost_usd") or 0.0) / c["calls"], 6)
            summary["analyzer"]["mean_prompt_tokens"] = round((c.get("prompt_tokens") or 0) / c["calls"], 1)
            summary["analyzer"]["mean_completion_tokens"] = round((c.get("completion_tokens") or 0) / c["calls"], 1)
    if all_failed:
        summary["all_failed"] = True
        summary["artifact_written"] = bool(args.allow_empty)
    summary_path = Path(args.out).expanduser().with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)  # the artifact write (which makes it) may be skipped
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("DONE %s", json.dumps(summary, indent=2))
    if all_failed and not args.allow_empty:
        raise SystemExit(
            f"FATAL: every analyzer call failed ({stats['error']}/{len(task_ids)} tasks errored; see the "
            f"warnings above); no hint artifact written. Check the analyzer endpoint / key / model, or pass "
            f"--allow-empty to write the empty artifact anyway. Summary: {summary_path}"
        )
    return summary


def main() -> None:
    try:
        asyncio.run(build(parse_args()))
    except AnalyzerConfigError as exc:
        # Nothing was written (the artifact is saved only after every task ran). Exit non-zero with the
        # actionable message rather than a traceback through asyncio.
        raise SystemExit(f"FATAL: analyzer misconfigured: {exc}") from exc


if __name__ == "__main__":
    main()

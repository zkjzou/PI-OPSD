"""Standalone τ² evaluation harness — the Phase-1 deliverable (no SkyRL trainer involved).

    python -m opsd.tau2_bench.baseline \
        --split all-base --policy-base-url http://127.0.0.1:8100/v1 \
        --model Qwen/Qwen3-4B-Thinking-2507 --num-trials 4 --concurrency 32

Reports, per domain: mean reward, **coverage** pass@k, pass^k, the termination-reason histogram, the
reward-component breakdown, and per-leg API cost.  Appends every episode to ``trajectories.jsonl`` and
resumes from it, so a long sweep survives interruption.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import metering
from .agent import PolicySpec, UserSpec, is_local_model_endpoint
from .constants import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS,
    DEFAULT_POLICY_MODEL,
    DEFAULT_SIMULATOR_MODEL,
    DEFAULT_SOURCE_DIR,
    MANIFEST_BY_NAME,
    REWARDABLE_TERMINATIONS,
    SAMPLING_PROFILES,
    default_llm_base_url,
)
from .episode import Tau2EpisodeResult, derive_seed, run_episode
from .source import bootstrap_tau2, load_tasks

BASE_SPLITS = ("airline-base", "retail-base", "telecom-base")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="τ²-bench evaluation baseline")
    p.add_argument("--split", default="all-base", choices=(*MANIFEST_BY_NAME, "all-base"))
    p.add_argument("--model", default=DEFAULT_POLICY_MODEL, help="served policy model name")
    p.add_argument(
        "--policy-base-url",
        default="http://127.0.0.1:8300/v1",
        help="policy endpoint(s); comma-separate replicas to round-robin, e.g. "
        "http://127.0.0.1:8300/v1,http://127.0.0.1:8301/v1",
    )
    p.add_argument("--policy-api-key-env", default="VLLM_API_KEY")
    p.add_argument("--user-llm", default=DEFAULT_SIMULATOR_MODEL)
    p.add_argument(
        "--user-base-url",
        default=default_llm_base_url(),
        help="user-simulator endpoint(s), comma-separated replicas allowed "
        "(default: $OPENAI_BASE_URL, else https://api.openai.com/v1)",
    )
    p.add_argument(
        "--user-api-key-env",
        default=DEFAULT_LLM_API_KEY_ENV,
        help="NAME of the env var holding the simulator key (not optional for a hosted "
        "endpoint; a self-hosted http:// simulator falls back to a dummy key)",
    )
    p.add_argument(
        "--user-reasoning-effort",
        default=None,
        help="override the simulator's reasoning_effort; omit to keep UserSpec's default "
        "('none', i.e. a non-thinking user)",
    )
    p.add_argument("--judge-model", default="openai/gpt-4.1")
    p.add_argument(
        "--judge-base-url",
        default=default_llm_base_url(),
        help="NL-assertion judge endpoint (default: $OPENAI_BASE_URL, else https://api.openai.com/v1)",
    )
    p.add_argument(
        "--judge-api-key-env",
        default=DEFAULT_LLM_API_KEY_ENV,
        help="NAME of the env var holding the judge key; if it is unset the judge reuses the "
        "simulator key (the original behaviour)",
    )
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--profile", default="qwen-thinking", choices=tuple(SAMPLING_PROFILES))
    p.add_argument(
        "--policy-reasoning-effort",
        default=None,
        help="policy-side reasoning budget (e.g. none/low/medium/high) for frontier "
        "thinking models; omit to leave it unset",
    )
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--no-thinking", action="store_true", help="send enable_thinking=false to the policy")
    p.add_argument("--task", action="append", default=None, help="task id (repeatable)")
    p.add_argument("--task-index", action="append", type=int, default=None, help="ordinal (repeatable)")
    p.add_argument("--limit", type=int, default=None, help="first N tasks per split")
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--trial-start", type=int, default=0)
    p.add_argument(
        "--episode-order",
        default="trial",
        choices=("trial", "task"),
        help="TRIAL-major (default) runs trial 0 of every task, then trial 1 of every task, "
        "so an interrupted run leaves a COMPLETE curve at the largest finished k over "
        "the WHOLE pool. TASK-major (the old behaviour) runs all k trials of one task "
        "before moving on, so an interruption leaves N finished tasks and the rest "
        "untouched -- and _coverage_pass_at_k then silently reports a number computed "
        "on that arbitrary head-of-file subset. Measured 2026-08-28: three n=128 runs "
        "stopped at 87%%/49%%/14%% and yielded ZERO usable pass@k curves for exactly "
        "this reason. Resume is unaffected either way: the done key is "
        "(domain, task_id, trial), not position, so re-ordering is free.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--executor-workers", type=int, default=None, help="defaults to --concurrency")
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--max-errors", type=int, default=DEFAULT_MAX_ERRORS)
    p.add_argument("--episode-timeout", type=float, default=900.0)
    p.add_argument("--evaluation-type", default="ALL")
    p.add_argument(
        "--seed-participants",
        action="store_true",
        help="inject the seed into both participants (only if BOTH endpoints accept `seed`)",
    )
    p.add_argument(
        "--preserve-thinking",
        action="store_true",
        help="feed the agent its own prior <think> back (as TRAINING does). Off by default, "
        "which is the deployment/paper protocol used for every Phase-1 baseline — but "
        "a GRPO-trained checkpoint was trained WITH it, so evaluating that checkpoint "
        "without this flag measures it on a context format it never saw. Whichever you "
        "pick, use the SAME setting on both sides of a trained-vs-untrained delta.",
    )
    p.add_argument(
        "--hints-path",
        default=None,
        help="PI hints JSON (task_id -> hint). Injects the hint into the AGENT's system "
        "prompt for utility-Δ guided generation: utility = success(with-hint) − "
        "success(no-hint), paired per task. ⚠️ MEASUREMENT ONLY. OPSD never hints the "
        "student -- the hint goes solely into the teacher's ref_prompt_token_ids -- so "
        "using this in a training rollout would train the policy on a context it never "
        "sees at eval. A task with no entry runs UNHINTED and is counted as such, so a "
        "partially-covered hint file cannot silently mix conditions.",
    )
    p.add_argument(
        "--require-hint",
        action="store_true",
        help="fail instead of running unhinted when --hints-path has no entry for a task; "
        "use for the with-hint leg of a paired study, where a silent miss would "
        "understate utility by averaging in no-hint episodes.",
    )
    p.add_argument("--yes-really", action="store_true", help="allow >500 episodes")
    return p.parse_args()


def _splits(selection: str) -> list[str]:
    return list(BASE_SPLITS) if selection == "all-base" else [selection]


def _load_done(path: Path) -> set[tuple[str, str, int]]:
    """Resume support: keys that already have a *genuinely terminated* episode.

    Only ``terminated_*`` episodes count as done. An infrastructure failure (a dead policy endpoint,
    a rate-limited sim, a judge error) is written to the log for the record but is **not** treated as
    done, so re-running automatically retries exactly the episodes a transient outage broke — instead
    of silently skipping them and poisoning the baseline. (A legitimately-terminated 0-reward episode,
    e.g. ``terminated_max_steps``, IS done and is not retried.)
    """

    done: set[tuple[str, str, int]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if not str(rec.get("status", "")).startswith("terminated_"):
                    continue
                done.add((rec["domain"], str(rec["task_id"]), int(rec["trial"])))
            except Exception:  # noqa: BLE001 - a torn final line must not break resume
                continue
    return done


def _order_jobs(slate: list[tuple[str, Any, int]], trials: list[int], order: str) -> list[tuple[str, Any, int, int]]:
    """Cross the task slate with the trial range in the requested sweep order.

    ⚠️ TRIAL-MAJOR IS THE DEFAULT AND IS THE WHOLE POINT. Sweeping trial-outer/task-inner means the
    corpus on disk is always "every task at n=k" for the largest completed k, so a run killed by a
    deadline, a preemption or a manual cancel still answers the question it was launched to ask —
    just at a smaller k. Task-major answers it for a head-of-file slice of tasks and not at all for
    the rest, and ``summarize``'s ``len(rs) >= k`` filter then reports that slice WITHOUT SAYING SO.
    """

    if order == "trial":
        return [(d, t, i, trial) for trial in trials for d, t, i in slate]
    return [(d, t, i, trial) for d, t, i in slate for trial in trials]


def _uniform_n_frontier(
    slate: list[tuple[str, Any, int]], trials: list[int], done: set[tuple[str, str, int]]
) -> tuple[int, int]:
    """(smallest, largest) number of finished trials over the tasks in the slate.

    The first element is the n up to which ``coverage_pass_at_k`` / ``pass^k`` describe the WHOLE
    pool rather than a subset of it — the number to quote when reading a partial run.
    """

    if not slate:
        return 0, 0
    per_task = [sum(1 for t in trials if (d, str(task.id), t) in done) for d, task, _ in slate]
    return min(per_task), max(per_task)


def _coverage_pass_at_k(rewards_by_task: dict[str, list[float]], k: int) -> float | None:
    """Fraction of tasks solved at least once within k trials (coverage pass@k).

    Deliberately *not* τ1's ``_pass_hat`` (which requires all k trials to succeed) — that is pass^k
    and is reported separately below.
    """

    usable = [rs for rs in rewards_by_task.values() if len(rs) >= k]
    if not usable:
        return None
    return sum(1.0 for rs in usable if any(r > 0 for r in rs[:k])) / len(usable)


def _pass_hat_k(rewards_by_task: dict[str, list[float]], k: int) -> float | None:
    """τ²'s reliability metric: all k trials succeed."""

    usable = [rs for rs in rewards_by_task.values() if len(rs) >= k]
    if not usable:
        return None
    return sum(1.0 for rs in usable if all(r > 0 for r in rs[:k])) / len(usable)


def summarize(records: list[dict[str, Any]], num_trials: int) -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for rec in records:
        by_domain[rec["domain"]].append(rec)

    summary: dict[str, Any] = {}
    for domain, recs in sorted(by_domain.items()):
        # Order each task's rewards by TRIAL INDEX, not by position in trajectories.jsonl.
        # trajectories.jsonl is append-ordered by COMPLETION, so rs[0] used to be whichever trial
        # won the race -- and that is not a random trial. Within a task the fastest of k trials is
        # disproportionately the one that cut the episode short and failed, so pass^1 (which reads
        # rs[:1]) was biased LOW: it read 0.509 on 4B/retail where every trial index scores ~0.64,
        # and it exactly reproduced "solve rate of the first-written episode per task" (0.5088).
        # pass^k for k == num_trials was never affected (it reads all k, so order is irrelevant).
        # A retried (task, trial) -- an infra failure re-run -- keeps its LAST record.
        trials_by_task: dict[str, dict[int, float]] = collections.defaultdict(dict)
        for rec in recs:
            trials_by_task[str(rec["task_id"])][int(rec["trial"])] = float(rec.get("reward") or 0.0)
        rewards_by_task: dict[str, list[float]] = {
            task: [by_trial[t] for t in sorted(by_trial)] for task, by_trial in trials_by_task.items()
        }

        # Best estimate of single-trial success: pooled over EVERY episode (k*n samples), vs pass^1
        # which spends only n. Equals avg_reward whenever the reward is binary, as it is on official
        # tau2 tasks (verified: the reward histogram is exactly {0.0, 1.0}).
        all_r = [r for rs in rewards_by_task.values() for r in rs]
        solve_rate_pooled = (sum(1.0 for r in all_r if r > 0) / len(all_r)) if all_r else None

        terminations = collections.Counter(rec.get("termination_reason") or rec.get("status") for rec in recs)
        infra = sum(
            1
            for rec in recs
            if str(rec.get("status", "")).endswith(("_api_error", "_error", "_overflow", "blocked"))
            and not str(rec.get("status", "")).startswith("terminated_")
        )
        components: dict[str, list[float]] = collections.defaultdict(list)
        for rec in recs:
            for key, value in (rec.get("reward_breakdown") or {}).items():
                name = str(getattr(key, "value", key)).rsplit(".", 1)[-1].strip("'\"")
                try:
                    components[name].append(float(value))
                except (TypeError, ValueError):
                    continue

        rewarded = [r for rec in recs for r in [float(rec.get("reward") or 0.0)]]
        summary[domain] = {
            "episodes": len(recs),
            "tasks": len(rewards_by_task),
            "avg_reward": (sum(rewarded) / len(rewarded)) if rewarded else None,
            "coverage_pass_at_k": {str(k): _coverage_pass_at_k(rewards_by_task, k) for k in range(1, num_trials + 1)},
            "solve_rate_pooled": solve_rate_pooled,
            "pass_hat_k": {str(k): _pass_hat_k(rewards_by_task, k) for k in range(1, num_trials + 1)},
            "termination_histogram": dict(terminations.most_common()),
            "rewardable_fraction": (
                sum(1 for rec in recs if rec.get("termination_reason") in REWARDABLE_TERMINATIONS) / len(recs)
                if recs
                else None
            ),
            "infrastructure_failures": infra,
            "reward_components": {k: sum(v) / len(v) for k, v in sorted(components.items()) if v},
            "avg_agent_turns": (sum(rec.get("num_agent_turns") or 0 for rec in recs) / len(recs)) if recs else None,
            "avg_user_tool_calls": (
                (sum(rec.get("num_user_tool_calls") or 0 for rec in recs) / len(recs)) if recs else None
            ),
            "avg_tool_errors": (sum(rec.get("num_tool_errors") or 0 for rec in recs) / len(recs)) if recs else None,
        }
    return summary


async def main_async(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectories.jsonl"

    # A SELF-HOSTED simulator (local vLLM) serves without auth, so a missing key is not an error there
    # -- mirror the policy leg's "dummy" fallback. Still fatal for a hosted gateway, where an absent
    # key means every episode would die on a 401 after burning the policy's tokens.
    user_key = os.environ.get(args.user_api_key_env, "")
    if not user_key:
        if is_local_model_endpoint(str(args.user_base_url).split(",")[0].strip()):
            user_key = "dummy"
        else:
            raise SystemExit(f"ERROR: ${args.user_api_key_env} is not set (needed for the user simulator)")

    # --policy-base-url may be a comma-separated list of replicas (e.g. a 4-GPU DP pool on
    # :8300..:8303); episodes round-robin across them so all replicas share the load. One PolicySpec
    # per URL — identical except base_url.
    policy_urls = [u.strip() for u in args.policy_base_url.split(",") if u.strip()]
    # ⚠️ THE SIMULATOR IS A REPLICA POOL TOO. --policy-base-url has always accepted a comma list and the
    # TRAINING path splits the sim identically (generator.py:283), but this driver built ONE UserSpec --
    # so against a self-hosted 4-replica simulator node, 3 replicas sat idle while every episode queued
    # on one. That failure is invisible: the run completes, just ~4x slower, and the only symptom is a
    # sim latency that reads as "the model is slow".
    user_urls = [u.strip() for u in str(args.user_base_url).split(",") if u.strip()]

    # The judge gets its own key env (as in the training generator); it falls back to the simulator key,
    # which is all it ever had before, so a run with one shared key behaves exactly as it did.
    judge_key = os.environ.get(args.judge_api_key_env or "", "") or user_key
    judge_args = {"api_base": args.judge_base_url, "api_key": judge_key, "temperature": 0.0}
    bootstrap_tau2(
        args.source_dir,
        max_connections=max(args.concurrency * 2, 16),
        allowed_base_urls=[*policy_urls, *user_urls, args.judge_base_url],
        judge_model=args.judge_model,
        judge_llm_args=judge_args,
        preserve_thinking=args.preserve_thinking,
    )
    metering.install()
    if args.preserve_thinking:
        print(
            "[protocol] preserve_thinking=ON — matching the training-time context "
            "(compare only against baselines measured the same way)"
        )

    profile = SAMPLING_PROFILES[args.profile]
    policy_key = os.environ.get(args.policy_api_key_env, "dummy") or "dummy"
    policies = [
        PolicySpec(
            served_model=args.model,
            base_url=url,
            api_key=policy_key,
            temperature=profile["temperature"],
            top_p=profile["top_p"],
            top_k=profile["top_k"],
            min_p=profile["min_p"],
            reasoning_effort=args.policy_reasoning_effort,
            max_tokens=args.max_tokens,
            enable_thinking=False if args.no_thinking else True,
        )
        for url in policy_urls
    ]
    print(f"[plan] {len(policies)} policy replica(s): {policy_urls}")
    # ``--user-reasoning-effort`` overrides UserSpec's own default, which is "none" (a NON-THINKING,
    # non-deliberating user; see UserSpec.reasoning_effort). temperature is already 0.0 (τ²-official
    # for both legs).
    users = [
        UserSpec(
            model=args.user_llm,
            base_url=u,
            api_key=user_key,
            **({"reasoning_effort": args.user_reasoning_effort} if args.user_reasoning_effort else {}),
        )
        for u in user_urls
    ]
    if len(users) > 1:
        print(f"[plan] {len(users)} simulator replica(s): {user_urls}")

    # Build the job list. `slate` is every (domain, task, index) across the selected splits; the
    # trial dimension is added afterwards, in whichever order --episode-order asks for.
    slate: list[tuple[str, Any, int]] = []
    for split_name in _splits(args.split):
        spec = MANIFEST_BY_NAME[split_name]
        tasks = load_tasks(spec.domain, spec.split_name)
        if args.task:
            wanted = set(args.task)
            tasks = [t for t in tasks if str(t.id) in wanted]
        if args.task_index:
            keep = set(args.task_index)
            tasks = [t for i, t in enumerate(tasks) if i in keep]
        if args.limit is not None:
            tasks = tasks[: args.limit]
        for index, task in enumerate(tasks):
            slate.append((spec.domain, task, index))

    trials = list(range(args.trial_start, args.trial_start + args.num_trials))
    jobs = _order_jobs(slate, trials, args.episode_order)

    done = _load_done(traj_path)
    pending = [j for j in jobs if (j[0], str(j[1].id), j[3]) not in done]
    print(f"[plan] episode order: {args.episode_order}-major over {len(slate)} task(s) x " f"{len(trials)} trial(s)")
    if done:
        n, worst = _uniform_n_frontier(slate, trials, done)
        print(f"[plan] uniform-n frontier: n={n} over all {len(slate)} tasks " f"(max n on any single task = {worst})")
    print(f"[plan] {len(jobs)} episodes ({len(jobs) - len(pending)} already done, {len(pending)} to run)")
    if len(pending) > 500 and not args.yes_really:
        raise SystemExit(f"ERROR: {len(pending)} episodes queued; re-run with --yes-really to confirm.")
    if not pending:
        print("[plan] nothing to do")

    # --- PI hints for utility-Δ guided generation (measurement only; see --hints-path) -------------
    hints: dict[str, str] = {}
    if args.hints_path:
        from opsd.pi.artifact import load_task_hints

        hints = {str(k): v for k, v in load_task_hints(args.hints_path).items()}
        covered = sum(1 for _, t, _ in slate if str(t.id) in hints)
        print(
            f"[hints] loaded {len(hints)} hint(s) from {args.hints_path}; "
            f"{covered}/{len(slate)} slate tasks covered"
        )
        # A partially-covered file would otherwise silently BLEND hinted and unhinted episodes into one
        # condition, biasing utility toward zero. Say it loudly, and refuse under --require-hint.
        if covered < len(slate):
            msg = (
                f"[hints] {len(slate) - covered} slate task(s) have NO hint and would run UNHINTED, "
                f"mixing two conditions in one measurement"
            )
            if args.require_hint:
                raise SystemExit("ERROR: " + msg + " -- refusing (--require-hint)")
            print("⚠️ " + msg)

    write_lock = threading.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)
    executor = ThreadPoolExecutor(max_workers=args.executor_workers or args.concurrency)
    counter = {"n": 0}
    started = time.time()

    async def worker(domain: str, task: Any, index: int, trial: int, job_ord: int) -> None:
        async with semaphore:
            result: Tau2EpisodeResult = await run_episode(
                domain=domain,
                task=task,
                task_index=index,
                trial=trial,
                seed=derive_seed(args.seed, domain, str(task.id), trial),
                policy_spec=policies[job_ord % len(policies)],  # round-robin across replicas
                user_spec=users[job_ord % len(users)],  # round-robin, same as the policy
                executor=executor,
                max_steps=args.max_steps,
                max_errors=args.max_errors,
                episode_timeout=args.episode_timeout,
                evaluation_type=args.evaluation_type,
                seed_participants=args.seed_participants,
                hint=hints.get(str(task.id)),
            )
        with write_lock:
            with traj_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result.to_record(), default=str) + "\n")
            counter["n"] += 1
            rate = counter["n"] / max(time.time() - started, 1e-6)
            print(
                f"[progress] {counter['n']}/{len(pending)} {domain}/{str(task.id)[:40]} t{trial} "
                f"reward={result.reward} status={result.status} ({rate * 60:.1f}/min)",
                flush=True,
            )

    if pending:
        async with asyncio.TaskGroup() as group:
            for job_ord, (domain, task, index, trial) in enumerate(pending):
                group.create_task(worker(domain, task, index, trial, job_ord))
    executor.shutdown(wait=True)

    # Dedup by (domain, task_id, trial): a retried episode leaves a stale earlier record (usually an
    # infra failure) in the append-only log, and counting both would corrupt the per-trial stats.
    # Keep one record per key, preferring a terminated episode, else the last-written.
    _by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for line in traj_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = (rec["domain"], str(rec["task_id"]), int(rec["trial"]))
        prev = _by_key.get(key)
        prev_term = prev is not None and str(prev.get("status", "")).startswith("terminated_")
        this_term = str(rec.get("status", "")).startswith("terminated_")
        if prev is None or this_term or not prev_term:
            _by_key[key] = rec
    records = list(_by_key.values())
    summary = {
        "config": {
            "policy_model": args.model,
            "user_llm": args.user_llm,
            "judge_model": args.judge_model,
            "profile": args.profile,
            "max_steps": args.max_steps,
            "num_trials": args.num_trials,
            "evaluation_type": args.evaluation_type,
            "episodes": len(records),
        },
        "cost": metering.flat_metrics(),
        "per_domain": summarize(records, args.num_trials),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()

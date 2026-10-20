#!/usr/bin/env python
"""Build the `checklist` PI arm: a frontier WIN vs the student's own LOSS, distilled to a checklist.

Emits TWO paired artifacts, because the checklist arm is uninterpretable without the second one:

  hints_checklist.json          the checklist (~1-2k chars)
  hints_win_headtail.json       the SAME frontier transcript truncated head+tail to each task's own
                                checklist length -- the length-matched demonstration control

WHY THE CONTROL IS NOT OPTIONAL. The demonstration arms run ~12k chars (measured: glm52 raw median
12,035 on normal, 11,704 on train90) and hint length correlates with post-training performance at
rho +0.44 (measured on the AppWorld self-hint pool). A 1.5k-char checklist that loses to a 12k-char
transcript tells you nothing -- it could be the form or it could be the 8x length. Truncating the same
frontier text to the same character budget isolates FORM.

Read the caveat honestly: head+tail at ~1.5k chars is a WEAK demonstration. This controls for budget,
not for quality of compression. A summary-at-equal-length arm would control for the latter.

WHY NOT opsd/pi/build_hints.py. That driver's `trajview_from_dump` derives success from
`won` / `env_metrics.success` / `reward >= 10.0` and reads `repetition_id`. AppWorld transcripts carry
binary `reward` (0/1) and `sample`, so every rollout would be labelled a FAILURE and every repetition
id would collapse to 0 -- silently, producing a loss/loss pair on every task. This driver does the
schema handling explicitly per benchmark.

Usage (AppWorld; run_pi_checklist_hints.sh wraps this per bench/split):
  python scripts/analysis/build_checklist_hints.py --bench appworld \
    --pool      <student pass@k corpus dir>/ \
    --win-hints <frontier hints dir>/hints_raw.json \
    --out-dir   outputs/pi_checklist/appworld_train90 \
    --analyzer-backend endpoint --analyzer-model <model> --max-concurrency 16
The analyzer endpoint/key follow pi/analyzer.py (OPSD_PI_ANALYZER_BASE_URL / OPSD_PI_ANALYZER_API_KEY_ENV,
falling back to OPENAI_BASE_URL / OPENAI_API_KEY).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import itertools
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    """The repository root: the nearest ancestor of this file holding pyproject.toml, opsd/__init__.py and
    opsd/skyrl_ext/__init__.py (the predicate of opsd._repo)."""
    for d in Path(__file__).resolve().parents:
        if all((d / m).is_file() for m in ("pyproject.toml", "opsd/__init__.py", "opsd/skyrl_ext/__init__.py")):
            return d
    raise RuntimeError(f"cannot locate the repository root above {__file__}")


sys.path.insert(0, str(_repo_root()))

from opsd.pi import (  # noqa: E402
    load_hint_artifact,
    load_task_hints,
    save_hint_artifact,
)
from opsd.pi.analyzer import make_analyzer  # noqa: E402
from opsd.pi.base import HintRecord, TrajView  # noqa: E402
from opsd.pi.checklist import ChecklistConstructor  # noqa: E402

# Same marker/shape as TrajView.render's middle-out split (pi/base.py:99), so the control arm is
# truncated exactly the way every other trajectory in this codebase is.
_MARK = "\n...[trajectory truncated]...\n"
_ACTION_LINE = re.compile(r"^\s*(?:print\(|apis\.|answer\()", re.MULTILINE)
# An action line only LEAKS if it carries a literal argument value -- that is what can encode the answer
# (`amount=29`, a literal `user_email=` string). An argument-free call is just an API NAME, and the
# checklist is *supposed* to name APIs: "Call apis.supervisor.complete_task() with the appropriate answer
# argument only after confirming the objective is met" is exactly the advice we want, yet the first version
# of this check failed the whole train90 build on it. Naming an endpoint is not leaking a solution.
_LITERAL_ARG = re.compile(r"""=\s*['"][^'"]{2,}['"]|=\s*\d+""")


def head_tail(text: str, budget: int) -> str:
    """Truncate to ~`budget` chars, keeping the head AND the tail.

    Head+tail rather than head-only on purpose: the tail is where the agent submits its answer, and a
    head-only truncation would drop the finish from every control hint, making the control weaker than
    the budget actually allows.

    The marker is charged against the budget (unlike pi/base.py:99, which adds it on top) so the arms
    land within a char or two of each other -- the verification block asserts that tightness.
    """
    if budget <= 0 or len(text) <= budget:
        return text
    keep = max(1, (budget - len(_MARK)) // 2)
    return text[:keep] + _MARK + text[-keep:]


# --------------------------------------------------------------------------------------------------
# Pool loading. Per-benchmark because the dump schemas genuinely differ; both paths must produce
# TrajViews whose `success` and `repetition_id` are CORRECT, since the pair selection turns on them.
# --------------------------------------------------------------------------------------------------
def load_appworld_pool(pool: Path) -> Dict[str, List[TrajView]]:
    """AppWorld: transcripts.jsonl carries `conversation`, `turns`, `reward`, `sample` inline."""
    groups: Dict[str, List[TrajView]] = collections.defaultdict(list)
    path = pool / "transcripts.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no transcripts.jsonl under {pool}")
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        reward = float(r.get("reward", 0.0) or 0.0)
        groups[r["task_id"]].append(
            TrajView(
                task_id=r["task_id"],
                messages=list(r.get("conversation") or []),
                reward=reward,
                success=reward > 0,  # AppWorld reward is binary; NOT the >=10 rule
                num_turns=len(r.get("turns") or []),
                repetition_id=int(r.get("sample", 0) or 0),
                raw=r,
            )
        )
    return groups


_CHAT_TURN = re.compile(r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>|\Z)", re.DOTALL)


def parse_chat_dump(input_prompt: str, output_response: str) -> List[Dict[str, str]]:
    """Reconstruct role/content turns from a SkyRL generate dump's two flat strings.

    `input_prompt` ends with a dangling `<|im_start|>assistant\\n` header and `output_response` supplies
    that turn's content, so the two must be CONCATENATED before parsing -- parsing them separately yields
    an empty assistant turn plus a headerless first action.
    """
    return [
        {"role": m.group(1), "content": m.group(2).strip()}
        for m in _CHAT_TURN.finditer(input_prompt + output_response)
        if m.group(2).strip()
    ]


def load_skyrl_dump_pool(pool: Path) -> Dict[str, List[TrajView]]:
    """SkyRL generate dump (`exports/dumped_evals/.../<env>.jsonl`), one rollout per line.

    Three schema traps, each of which silently produces a wrong pool:
      * `score` is a PER-STEP LIST (`[0.0, ..., 10.0]`), not a scalar -- `float(score)` raises and
        `score > 0` on the raw list is meaningless. ALFWorld pays 10 on success, hence max(...) >= 10.
      * the task id is nested at `env_extras.extra_info.task_id`; the top level has no `task_id` at all,
        so a naive `rec["task_id"]` collapses every rollout into one None-keyed group.
      * there is NO repetition id, so it is assigned by encounter order per task -- deterministic given
        the sorted file order this function imposes.

    PARSE ONLY THE FIRST LOSS PER TASK. These dumps carry 32-128 reps per task at ~13k chars each
    (1.0-0.6 GB per pool), and only ONE rep per task is ever used -- the first losing one. Parsing the
    chat for all of them cost minutes of pure waste: 44,800 regex passes where 1,400 suffice.

    This COUPLES the loader to ChecklistConstructor._pick_loss's rule ("first losing rep by
    repetition_id"). The coupling is safe because rep ids are assigned in encounter order here, so
    "first loss encountered" IS "first loss by rep id" -- but if that rule ever changes, the constructor
    would receive a TrajView with empty messages. The driver asserts non-empty messages on the selected
    loss to catch exactly that.
    """
    groups: Dict[str, List[TrajView]] = collections.defaultdict(list)
    # Parse the first LOSS and the first WIN per task. The loss is the normal contrast side; the win is
    # needed for the win-vs-win fallback on tasks the student never fails, and a single pass cannot know
    # in advance which tasks those are. Two parses per task instead of 32-128 still saves ~98% of the work.
    parsed_loss: set = set()
    parsed_win: set = set()
    files = sorted(pool.glob("*.jsonl")) if pool.is_dir() else [pool]
    if not files:
        raise FileNotFoundError(f"no *.jsonl under {pool}")
    for path in files:
        for line in path.open():
            if not line.strip():
                continue
            r = json.loads(line)
            extra = (r.get("env_extras") or {}).get("extra_info") or {}
            tid = str(extra.get("task_id") or extra.get("game_file") or "")
            score = r.get("score")
            score = score if isinstance(score, list) else [score or 0.0]
            reward = max(float(x or 0.0) for x in score)
            success = reward >= 10.0
            msgs: List[Dict[str, str]] = []
            first_loss = not success and tid not in parsed_loss
            first_win = success and tid not in parsed_win
            if first_loss or first_win:
                msgs = parse_chat_dump(r.get("input_prompt") or "", r.get("output_response") or "")
                (parsed_loss if first_loss else parsed_win).add(tid)
            groups[tid].append(
                TrajView(
                    task_id=tid,
                    messages=msgs,
                    reward=reward,
                    success=success,
                    repetition_id=len(groups[tid]),
                    raw={"stop_reason": r.get("stop_reason")},  # drop the bulk; dumps are ~150MB/shard
                )
            )
    return groups


def load_alfworld_pool(pool: Path) -> Dict[str, List[TrajView]]:
    """ALFWorld: trajectories.jsonl from generate_alfworld_local.py — `env_metrics.success` / `won`."""
    groups: Dict[str, List[TrajView]] = collections.defaultdict(list)
    path = pool / "trajectories.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no trajectories.jsonl under {pool}")
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        em = r.get("env_metrics") or {}
        reward = float(r.get("reward", 0.0) or 0.0)
        tid = str(r.get("task_id") or r.get("game_file") or r.get("id") or "")
        groups[tid].append(
            TrajView(
                task_id=tid,
                messages=list(r.get("messages") or r.get("conversation") or []),
                reward=reward,
                success=bool(r.get("won")) or bool(em.get("success", 0.0)) or reward >= 1.0,
                num_turns=int(em.get("num_turns", r.get("num_turns", 0)) or 0),
                repetition_id=int(r.get("repetition_id", r.get("sample", 0)) or 0),
                raw=r,
            )
        )
    return groups


def _looks_like_tau2(rec: dict) -> bool:
    """tau2 records carry `reward_breakdown` and a `domain` of retail/airline; ALFWorld records carry
    `env_metrics`/`won`/`game_file`. Needed because BOTH write a file named `trajectories.jsonl`."""
    return bool(rec.get("reward_breakdown")) or str(rec.get("domain", "")).lower() in ("retail", "airline")


def load_tau2_pool(pool: Path) -> Dict[str, List[TrajView]]:
    """tau2-bench simulation results -> TrajViews.

    Why not load_alfworld_pool with a different name: on a tau2 record that reader gets `task_id`,
    `messages` and `success` right only BY LUCK (success falls through to `reward >= 1.0`), but it
    reads the repetition from `repetition_id`/`sample` and tau2 stores it in **`trial`**. Every
    rollout would return repetition_id 0, and pair selection turns on `success` and `repetition_id`
    (see the note above load_appworld_pool), so one task's rollouts would look like one repetition.
    `num_turns` is `num_agent_turns` here too.

    Reward is graded on DB state plus communication/NL assertions (`reward_breakdown`) and partial
    credit is possible, so SUCCESS is the full-reward case only.
    """
    groups: Dict[str, List[TrajView]] = collections.defaultdict(list)
    path = pool / "trajectories.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no trajectories.jsonl under {pool}")
    n = 0
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        reward = float(r.get("reward", 0.0) or 0.0)
        tid = str(r.get("task_id") if r.get("task_id") is not None else r.get("task_index", ""))
        groups[tid].append(
            TrajView(
                task_id=tid,
                messages=list(r.get("messages") or r.get("agent_view") or []),
                reward=reward,
                success=reward >= 1.0,
                num_turns=int(r.get("num_agent_turns", r.get("num_messages", 0)) or 0),
                repetition_id=int(r.get("trial", 0) or 0),
                raw=r,
            )
        )
        n += 1
    reps = {len({t.repetition_id for t in v}) for v in groups.values()}
    print(
        f"  tau2 pool: {n} rollouts over {len(groups)} tasks; distinct trial ids per task "
        f"{min(reps) if reps else 0}-{max(reps) if reps else 0}"
    )
    return groups


_CHATML = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.S)


def _looks_like_tau2_skyrl(rec: dict) -> bool:
    """SkyRL's own tau2 generate dump: a FOURTH shape, distinct from tau2-bench sim output.

    task_id is TOP-LEVEL and `retail_N` (the training namespace), reward is a SCALAR, and there is no
    `messages` at all -- the conversation lives in prompt_text/response_text as ChatML. The generic
    load_skyrl_dump_pool reads task_id from env_extras.extra_info, which is absent here, so it
    collapses the entire pool into one group keyed '' (observed).
    """
    return bool(rec.get("response_text")) and rec.get("messages") is None and bool(rec.get("task_id"))


def load_tau2_skyrl_dump_pool(pool: Path) -> Dict[str, List[TrajView]]:
    """tau2 rollouts from SkyRL train dumps -> TrajViews, messages rebuilt from ChatML.

    Needed because a checklist in the TRAINING namespace (`retail_N`) can only come from SkyRL's own
    rollouts; every tau2-bench sim corpus is bare-int and does not join to the training data.
    """
    groups: Dict[str, List[TrajView]] = collections.defaultdict(list)
    files = sorted(pool.glob("*.jsonl")) if pool.is_dir() else [pool]
    if not files:
        raise FileNotFoundError(f"no *.jsonl under {pool}")
    seen_rep: Dict[str, int] = collections.defaultdict(int)
    dropped: Dict[str, int] = collections.defaultdict(int)
    n = 0
    for path in files:
        for line in path.open():
            if not line.strip():
                continue
            r = json.loads(line)
            tid = str(r.get("task_id") or "")
            if not tid:
                continue
            try:
                reward = float(r.get("reward", 0.0) or 0.0)
            except (TypeError, ValueError):
                reward = 0.0
            msgs = [
                {"role": role, "content": body.strip()}
                for role, body in _CHATML.findall(str(r.get("prompt_text") or "") + str(r.get("response_text") or ""))
                if body.strip()
            ]
            # DROP a rollout whose ChatML parsed to nothing. Some losing tau2 rollouts have an empty
            # response_text (the episode produced no transcript), and a loss with no transcript cannot
            # be contrasted against a win. Keeping them lets _pick_loss select one and the driver then
            # (correctly) refuses the task -- 4 of 5 eligible tasks were lost that way before this.
            if not msgs:
                dropped[tid] += 1
                continue
            rep = r.get("trial")
            if rep is None:
                rep = seen_rep[tid]
            seen_rep[tid] += 1
            groups[tid].append(
                TrajView(
                    task_id=tid,
                    messages=msgs,
                    reward=reward,
                    success=reward >= 1.0,
                    num_turns=int(r.get("agent_turns", 0) or 0),
                    repetition_id=int(rep or 0),
                    raw=r,
                )
            )
            n += 1
    ndrop = sum(dropped.values())
    print(
        f"  tau2 SkyRL dump pool: {n} usable rollouts over {len(groups)} tasks; "
        f"dropped {ndrop} with an empty transcript"
    )
    if not n:
        raise SystemExit("every rollout parsed to zero messages -- the ChatML regex did not match")
    return groups


async def build(a: argparse.Namespace) -> int:
    if a.merge_from:
        # Merge artifacts built in separate passes (e.g. ALFWorld train1600 arrives as a 1400-task pool
        # plus a disjoint 200-task pool with a different rep count), then VERIFY the merged result --
        # a merge that breaks the checklist/control pairing must fail here, not silently downstream.
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        merged: Dict[str, Dict[str, HintRecord]] = {"hints_checklist": {}, "hints_win_headtail": {}}
        src: Dict[str, Any] = {}
        for d in a.merge_from:
            d = Path(d)
            for stem in merged:
                part = load_hint_artifact(str(d / f"{stem}.json"))
                dupes = set(part) & set(merged[stem])
                if dupes:
                    # First dir wins, but say so: a silent overwrite would mix two pools' provenance.
                    print(
                        f"  WARNING {stem}: {len(dupes)} keys already present from an earlier "
                        f"--merge-from; keeping the earlier copy (e.g. {sorted(dupes)[:2]})"
                    )
                merged[stem] = {**part, **merged[stem]}
            sp = d / "src_reps.json"
            if sp.exists():
                src = {**json.loads(sp.read_text()), **src}
            print(f"  merged {d}  -> running total {len(merged['hints_checklist'])} checklists")
        for stem, recs in merged.items():
            save_hint_artifact(str(out / f"{stem}.json"), recs)
        (out / "src_reps.json").write_text(json.dumps(src, indent=1))
        win_texts = load_task_hints(a.win_hints)
        print(f"  MERGED -> {out}")
        return verify(
            a,
            out,
            merged["hints_checklist"],
            merged["hints_win_headtail"],
            win_texts,
            {},
            sorted(merged["hints_checklist"]),
            collections.Counter(),
            None,
        )

    if a.verify_only:
        # Re-run the self-checks against an ALREADY-BUILT artifact. Exists because a verification bug
        # (the first leak rule flagged argument-free API names) otherwise costs a full paid rebuild to
        # re-clear -- the hints themselves are unaffected by a change to how they are checked.
        out = Path(a.out_dir)
        checklists = load_hint_artifact(str(out / "hints_checklist.json"))
        control = load_hint_artifact(str(out / "hints_win_headtail.json"))
        win_texts = load_task_hints(a.win_hints)
        tok = None
        if a.tokenizer:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
        print(f"  VERIFY-ONLY on {out}  ({len(checklists)} checklists)")
        return verify(a, out, checklists, control, win_texts, {}, sorted(checklists), collections.Counter(), tok)

    pool = Path(a.pool)
    # SNIFF the schema rather than switching on --bench alone: ALFWorld rollouts exist in two shapes
    # (generate_alfworld_local's trajectories.jsonl, and SkyRL generate dumps), and picking the wrong
    # reader mislabels success or collapses the task ids without erroring.
    if (pool / "transcripts.jsonl").exists():
        groups = load_appworld_pool(pool)
        shape = "appworld transcripts.jsonl"
    elif (pool / "trajectories.jsonl").exists():
        # tau2 uses the SAME filename, so peek at a record before choosing the reader. Getting this
        # wrong is silent: the alfworld reader returns every tau2 rollout with repetition_id 0.
        _first = {}
        with (pool / "trajectories.jsonl").open() as _fh:
            for _ln in _fh:
                if _ln.strip():
                    _first = json.loads(_ln)
                    break
        if _looks_like_tau2(_first):
            groups = load_tau2_pool(pool)
            shape = "tau2 trajectories.jsonl"
        else:
            groups = load_alfworld_pool(pool)
            shape = "alfworld trajectories.jsonl"
    else:
        _first = {}
        for _f in sorted(pool.glob("*.jsonl")) if pool.is_dir() else [pool]:
            with _f.open() as _fh:
                for _ln in _fh:
                    if _ln.strip():
                        _first = json.loads(_ln)
                        break
            if _first:
                break
        if _looks_like_tau2_skyrl(_first):
            groups = load_tau2_skyrl_dump_pool(pool)
            shape = "tau2 SkyRL train dump (*.jsonl, ChatML)"
        else:
            groups = load_skyrl_dump_pool(pool)
            shape = "skyrl generate dump (*.jsonl)"
    print(f"  pool shape              {shape}")
    win_texts = load_task_hints(a.win_hints)
    if not win_texts:
        raise SystemExit(f"--win-hints held no task-level hints: {a.win_hints}")

    task_ids = sorted(groups)
    if a.only_tasks:
        # Gap-fill a specific subset without rebuilding the whole pool. Accepts a
        # JSON list/dict-of-keys file or a comma-separated list.
        raw = Path(a.only_tasks)
        if raw.exists():
            obj = json.loads(raw.read_text())
            want = set(obj if isinstance(obj, list) else [k for k in obj if k != "__schema__"])
        else:
            want = {t.strip() for t in a.only_tasks.split(",") if t.strip()}
        missing = want - set(task_ids)
        task_ids = [t for t in task_ids if t in want]
        print(
            f"  --only-tasks: {len(task_ids)} of {len(want)} requested present in the pool"
            + (f"; {len(missing)} absent e.g. {sorted(missing)[:2]}" if missing else "")
        )
    if a.limit_tasks:
        # Spread the sample across DISTINCT scenarios, don't just take the first N sorted ids. AppWorld
        # task ids are `<scenario>_<variant>` (024c982_1, 024c982_2, ...), so a head slice returns
        # several variants of ONE scenario -- a smoke that looks diverse but is not, and which made a
        # cross-task similarity reading meaningless the first time it was used here.
        by_scenario: Dict[str, List[str]] = collections.defaultdict(list)
        for t in task_ids:
            by_scenario[t.rsplit("_", 1)[0]].append(t)
        picked, rnd = [], 0
        while len(picked) < a.limit_tasks and any(len(v) > rnd for v in by_scenario.values()):
            for scen in sorted(by_scenario):
                if len(by_scenario[scen]) > rnd and len(picked) < a.limit_tasks:
                    picked.append(by_scenario[scen][rnd])
            rnd += 1
        task_ids = picked
        print(
            f"  --limit-tasks {a.limit_tasks}: sampled {len(task_ids)} tasks across "
            f"{len({t.rsplit('_', 1)[0] for t in task_ids})} distinct scenarios"
        )

    # Report the funnel BEFORE building: a silently-shrunken arm is no longer paired with the
    # demonstration arm it exists to be compared against.
    have_win = [t for t in task_ids if (win_texts.get(t) or "").strip()]
    have_loss = [t for t in task_ids if any(not tv.success for tv in groups[t])]
    have_any = [t for t in task_ids if groups[t]]
    # With the fallback on, a task the student never fails is still eligible (win-vs-win). Without it,
    # those tasks are dropped and the arm silently covers less of the pool than the demonstration arms.
    eligible = sorted(set(have_win) & set(have_any if a.allow_win_vs_win else have_loss))
    print(f"  pool {pool}")
    print(f"    pool tasks              {len(task_ids)}")
    print(f"    with a frontier win     {len(have_win)}   (win artifact holds {len(win_texts)} tasks)")
    print(f"    with a student loss     {len(have_loss)}")
    if a.allow_win_vs_win:
        print(f"    no-loss -> win-vs-win   {len(set(have_win) & set(have_any)) - len(set(have_win) & set(have_loss))}")
    print(f"    ELIGIBLE                {len(eligible)}")
    if not eligible:
        raise SystemExit(
            "no eligible task. If 'with a frontier win' is 0 the task-id namespaces differ between "
            f"--pool and --win-hints (pool: {task_ids[:3]}, win: {sorted(win_texts)[:3]})."
        )

    analyzer = make_analyzer(
        a.analyzer_backend, model=a.analyzer_model, base_url=a.analyzer_base_url, api_key=a.analyzer_api_key
    )
    print(f"    analyzer                {a.analyzer_backend} / {analyzer.model}")

    # PREFLIGHT: one trivial call before the fan-out. Without it an auth/model/URL misconfiguration is
    # discovered once per task, each after litellm's 4 internal retries -- a 160-task build spends a long
    # time and a lot of noise to learn something one call establishes. Measured failure this caught:
    # "key not allowed to access model. This key can only access models=['default']".
    if not a.skip_preflight:
        try:
            probe = await analyzer.analyze(
                'Reply with exactly this JSON and nothing else: {"ok": true}',
                system="You reply with strict JSON only.",
                max_tokens=64,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"analyzer preflight FAILED before any task was built:\n  {type(exc).__name__}: {exc}\n"
                f"  backend={a.analyzer_backend} model={analyzer.model} base_url={a.analyzer_base_url}\n"
                "Fix the analyzer credential/model, or pass --skip-preflight to bypass this check."
            ) from exc
        print(f"    preflight               ok ({(probe or '')[:40]!r})")

    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)

    ctor = ChecklistConstructor(
        analyzer,
        win_texts=win_texts,
        bench=a.bench,
        win_source=a.win_hints,
        max_items=a.max_items,
        allow_win_vs_win=a.allow_win_vs_win,
        max_tokens=a.max_tokens,
        max_chars=a.max_chars,
        max_input_tokens=a.max_input_tokens,
        temperature=a.temperature,
        tokenizer=tok,
    )

    sem = asyncio.Semaphore(max(1, a.max_concurrency))
    stat = collections.Counter()
    done = [0]

    async def one(tid: str) -> Tuple[str, Optional[HintRecord]]:
        async with sem:
            try:
                rec = await ctor.build_offline(tid, groups[tid])
            except Exception as exc:  # noqa: BLE001 — one task must not kill the batch
                stat["error"] += 1
                print(f"    [err] {tid}: {type(exc).__name__}: {exc}")
                return tid, None
            finally:
                done[0] += 1
                if done[0] % 25 == 0:
                    print(f"    ... {done[0]}/{len(eligible)}", flush=True)
            if rec is None or not rec.task:
                stat["skipped"] += 1
                return tid, None
            stat["ok"] += 1
            return tid, rec

    results = await asyncio.gather(*(one(t) for t in eligible))
    checklists: Dict[str, HintRecord] = {t: r for t, r in results if r is not None}
    if not checklists:
        # Name the likely cause from WHICH counter dominates -- the two have completely different fixes,
        # and the earlier version only ever suggested the max-tokens one.
        cause = (
            f"errors dominate ({stat['error']}) — the analyzer itself is failing (auth, model name, "
            f"base_url, or context overflow). Read the [err] lines above; they carry the exception."
            if stat["error"] >= stat["skipped"]
            else f"skips dominate ({stat['skipped']}) — the analyzer returned unparseable or empty JSON, "
            f"usually because its output was truncated. Raise --max-tokens (currently {a.max_tokens})."
        )
        raise SystemExit(
            f"ZERO checklists built from {len(eligible)} eligible tasks "
            f"(skipped={stat['skipped']} error={stat['error']}). {cause}"
        )

    # The length-matched control, per task, at that task's own checklist length.
    control: Dict[str, HintRecord] = {}
    src: Dict[str, Any] = {}
    for tid, rec in checklists.items():
        budget = len(rec.task or "")
        control[tid] = HintRecord(
            task=head_tail(win_texts[tid], budget),
            meta={
                "pi_type": "win_headtail",
                "bench": a.bench,
                "budget_chars": budget,
                "win_chars": len(win_texts[tid]),
                "matched_to": "checklist",
            },
        )
        src[tid] = {
            "loss_rep": rec.meta.get("loss_rep"),
            "win_source": a.win_hints,
            "win_sha1": hashlib.sha1(win_texts[tid].encode()).hexdigest()[:12],
        }

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_hint_artifact(str(out / "hints_checklist.json"), checklists)
    save_hint_artifact(str(out / "hints_win_headtail.json"), control)
    (out / "src_reps.json").write_text(json.dumps(src, indent=1))

    return verify(a, out, checklists, control, win_texts, groups, eligible, stat, tok)


def verify(a, out: Path, checklists, control, win_texts, groups, eligible, stat, tok) -> int:
    """Self-verification. Anything that would invalidate the comparison FAILS the build."""
    keys = sorted(checklists)
    ok = True
    print(f"\n  built {len(keys)}/{len(eligible)} eligible  (skipped {stat['skipped']}, error {stat['error']})")

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"  [{'OK  ' if passed else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")

    check("checklist and control cover the same tasks", sorted(control) == keys, f"{len(keys)} vs {len(control)}")

    # Length match: the entire point of the control.
    diffs = [abs(len(checklists[k].task) - len(control[k].task)) for k in keys]
    worst = max(diffs) if diffs else 0
    # A task whose frontier text is SHORTER than its checklist cannot be padded to match; head_tail
    # returns it whole. Those are legitimate and excluded from the tightness assertion.
    shorter = [k for k in keys if len(win_texts[k]) <= len(checklists[k].task)]
    tight = [d for k, d in zip(keys, diffs) if k not in set(shorter)]
    check(
        "control length matches checklist (<=2 chars)",
        (max(tight) if tight else 0) <= 2,
        f"worst {max(tight) if tight else 0} over {len(tight)} tasks; "
        f"{len(shorter)} frontier texts were already shorter (kept whole)",
    )

    # Leakage: the abstraction claim is this arm's whole point, and the frontier text is right there.
    # Argument-BEARING copies fail the build; argument-free API names are counted and reported only.
    leaked, named = [], []
    for k in keys:
        hint = checklists[k].task
        for m in _ACTION_LINE.finditer(win_texts[k]):
            line = win_texts[k][m.start() :].split("\n", 1)[0].strip()
            if len(line) < 25 or line not in hint:
                continue
            (leaked if _LITERAL_ARG.search(line) else named).append((k, line[:70]))
            break
    leak_detail = f"{len(leaked)} leaked, e.g. {leaked[0][0]}: {leaked[0][1]!r}" if leaked else ""
    check("no frontier call WITH LITERAL ARGS copied verbatim", not leaked, leak_detail)
    if named:
        print(
            f"         (info: {len(named)} checklist(s) name an argument-free frontier call verbatim, "
            f"e.g. {named[0][1]!r} — an API name, not a solution)"
        )

    # Non-degenerate content: a checklist of one generic item is indistinguishable from a failed analyzer.
    items = [checklists[k].meta.get("n_items", 0) for k in keys]
    thin = [k for k, n in zip(keys, items) if n < 3]
    check("every checklist has >=3 items", not thin, f"{len(thin)} thin (median {statistics.median(items):.0f})")

    # CROSS-TASK REDUNDANCY. Reported, not asserted. A per-task arm whose 160 checklists all say the same
    # thing is effectively a GLOBAL checklist wearing per-task keys: the per-task pairing buys nothing, the
    # mismatched-PI ablation will show no effect, and the honest description of the result changes. There is
    # no defensible threshold to fail on, so this prints a number for the write-up instead of guessing one.
    # Compare only ACROSS scenarios: `<scenario>_<variant>` ids are variants of one task and are expected
    # to be similar, so pooling them inflates the figure.
    stop = set(
        "the a an to of for any each every and or if not do before after with from that this it is "
        "are be you your".split()
    )

    def _toks(text: str) -> set:
        return {
            w
            for item in re.findall(r"\[ \] (.+)", text)
            for w in re.findall(r"[a-z_]+", item.lower())
            if w not in stop and len(w) > 3
        }

    bags = {k: _toks(checklists[k].task) for k in keys}
    cross = [
        len(bags[x] & bags[y]) / max(1, len(bags[x] | bags[y]))
        for x, y in itertools.combinations(keys, 2)
        if x.rsplit("_", 1)[0] != y.rsplit("_", 1)[0]
    ]
    if cross:
        med = statistics.median(cross)
        shared = set.intersection(*bags.values()) if len(bags) > 1 else set()
        print(
            f"\n      cross-scenario item-word Jaccard  median {med:.3f}  over {len(cross)} pairs"
            f"   ({len(shared)} words appear in EVERY checklist)"
        )
        if med >= 0.5:
            print(
                "      NOTE: >=0.5 — these checklists are largely the same text. Report the arm as "
                "effectively GLOBAL procedural advice, not per-task PI, and expect the mismatched-PI "
                "ablation to show little."
            )
    else:
        print(
            "\n      cross-scenario Jaccard: n/a (all sampled tasks share one scenario — use a wider "
            "--limit-tasks or the full pool)"
        )

    L = sorted(len(checklists[k].task) for k in keys)
    W = sorted(len(win_texts[k]) for k in keys)
    print(f"\n      checklist chars  median {L[len(L)//2]:6}  min {L[0]:6}  max {L[-1]:6}")
    print(
        f"      frontier  chars  median {W[len(W)//2]:6}  min {W[0]:6}  max {W[-1]:6}"
        f"   -> compression {W[len(W)//2]/max(1,L[len(L)//2]):.1f}x"
    )
    print(f"      items/checklist  median {statistics.median(items):.0f}  min {min(items)}  max {max(items)}")

    # Chars are matched; TOKENS are what the model pays for, and prose vs code tokenize differently.
    # Report it rather than silently claiming a token-level match.
    if tok is not None:
        ct = statistics.median(len(tok.encode(checklists[k].task, add_special_tokens=False)) for k in keys)
        wt = statistics.median(len(tok.encode(control[k].task, add_special_tokens=False)) for k in keys)
        ratio = wt / max(1.0, ct)
        print(f"      median TOKENS    checklist {ct:.0f}  control {wt:.0f}  ratio {ratio:.2f}x")
        if not 0.8 <= ratio <= 1.25:
            print(
                f"      WARNING: chars are matched but tokens differ {ratio:.2f}x -- prose and code "
                f"tokenize differently. Report this; do not call the arms token-matched."
            )

    (out / "summary.json").write_text(
        json.dumps(
            {
                "bench": a.bench,
                "pool": a.pool,
                "win_hints": a.win_hints,
                "n_eligible": len(eligible),
                "n_built": len(keys),
                "skipped": stat["skipped"],
                "error": stat["error"],
                "checklist_chars_median": L[len(L) // 2],
                "frontier_chars_median": W[len(W) // 2],
                "items_median": statistics.median(items),
                "analyzer_model": a.analyzer_model,
                "max_tokens": a.max_tokens,
                "max_items": a.max_items,
                "all_checks_passed": bool(ok),
            },
            indent=2,
        )
    )

    print(f"\n  ALL CHECKS PASSED: {ok}")
    print(f"  wrote {out}/hints_checklist.json  +  hints_win_headtail.json  +  src_reps.json")
    return 0 if ok else 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, choices=("appworld", "alfworld", "tau2"))
    ap.add_argument("--pool", default=None, help="Student pool dir (transcripts.jsonl / trajectories.jsonl).")
    ap.add_argument("--win-hints", required=True, help="Frontier WIN artifact, e.g. hints_glm52/hints_raw.json.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--analyzer-backend", default="frontier", choices=("frontier", "policy", "endpoint"))
    ap.add_argument("--analyzer-model", default=None)
    ap.add_argument("--analyzer-base-url", default=None)
    ap.add_argument("--analyzer-api-key", default=None)
    ap.add_argument("--max-items", type=int, default=5)
    ap.add_argument("--max-chars", type=int, default=6000, help="Output cap on the checklist.")
    ap.add_argument("--max-input-tokens", type=int, default=32000, help="Analyzer INPUT budget, split per side.")
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Analyzer OUTPUT budget. 1024 yields ZERO hints from a thinking analyzer.",
    )
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-concurrency", type=int, default=16)
    ap.add_argument("--tokenizer", default=None, help="Optional: report the arms' token ratio too.")
    ap.add_argument("--limit-tasks", type=int, default=None, help="Smoke run.")
    ap.add_argument(
        "--only-tasks",
        default=None,
        help="Build ONLY these task ids (path to a JSON list/artifact, or a comma-separated list).",
    )
    ap.add_argument(
        "--allow-win-vs-win",
        action="store_true",
        help="For tasks the student NEVER fails, contrast the frontier win against the "
        "student's own winning rollout instead of skipping. Tagged meta.derivation="
        "win_vs_win so those tasks stay separable.",
    )
    ap.add_argument(
        "--merge-from",
        action="append",
        default=[],
        help="Merge these built artifact dirs into --out-dir, then verify. Repeatable; earlier dirs win on key conflicts.",
    )
    ap.add_argument(
        "--verify-only", action="store_true", help="Re-run the self-checks on an existing --out-dir; builds nothing."
    )
    ap.add_argument(
        "--skip-preflight", action="store_true", help="Skip the one-call analyzer probe (it costs a few tokens)."
    )
    return ap.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(build(parse_args())))

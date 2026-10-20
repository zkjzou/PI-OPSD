#!/usr/bin/env python
"""Build AppWorld SELF-* PI hints from a model's own rollouts (hindsight).

For each task, pick the student's best rollout (highest reward / TGC — AppWorld 4B rarely fully
solves, so we rank by partial credit) and render it as a per-task hint in three formats:
  * raw           — with thinking:   "# turn k\n<think>…</think>\n<code>\n<obs>"
  * raw_no_think  — code + obs only (no reasoning)
  * actions       — just the sequence of code actions (no obs, no think)
  * actions_think — reasoning + code, NO observations   <- completes the 2x2

THE 2x2 IS THE POINT. The three legacy densities leave one cell of {think on/off} x {obs on/off}
unbuilt, in BOTH this benchmark and ALFWorld (see examples/alfworld/build_pi_hint_formats.py, which
emits the same three). That missing cell is the only one that isolates reasoning from length:

              obs OFF            obs ON
  think OFF   actions            raw_no_think
  think ON    actions_think*     raw                (* new)

Why it matters here: on the 4bt density grid raw_no_think (action+obs) BEAT raw (think+action+obs),
0.1324 vs 0.1186 -- adding reasoning to an already-long hint HURT. But raw is also ~50% longer, so
"reasoning is harmful" and "the hint got too long" are confounded. actions_think holds obs out and
varies only reasoning against `actions`, which separates them.
Keyed by task_id, drop-in for OPSD_HINTS_PATH and the PI positions (name=path) scripts.
Reads a transcripts.jsonl whose records carry per-turn {think,action,obs} (written by
generate_appworld_local.py --save-transcripts, even in --no-preserve runs). Needs the opsd package importable
(the repository venv; run from the repository root with PYTHONPATH=.) for the `actions` extraction.

Usage:
  # frontier reference trajectories -> the four densities (last step of build_appworld_pi_{train90,dev57}.sh)
  # (--select first is REQUIRED to reproduce the published train-90 files once the b0a8eae re-runs are
  #  appended to the transcripts; see the provenance note below)
  PYTHONPATH=. .venv/bin/python scripts/analysis/build_appworld_self_hints.py \
    --transcripts outputs/appworld_pi/gen_train90_glm52/transcripts_merged.jsonl \
    --out-dir outputs/appworld_pi/hints_train90_glm52 --min-reward 1.0 --select first
  # a student's own no-hint corpus (self-PI), unsolved tasks filled from a frontier corpus (an example
  # invocation; the published self_ssdob125 artifact used --ladder strict, see below)
  PYTHONPATH=. .venv/bin/python scripts/analysis/build_appworld_self_hints.py \
    --transcripts "$APPWORLD_CORPUS_ROOT/<profile>_<split>[_<tag>]/transcripts.jsonl" \
    --fallback-transcripts outputs/appworld_pi/gen_train90_glm52/transcripts_merged.jsonl \
    --fallback-name glm52 --min-reward 1.0 --out-dir outputs/appworld_pi/hints_self

The published train-90 raw / raw_no_think / actions files (dataset bundle hints/trajectory/<author>/)
were built with --min-reward 1.0 and the original selection rule (first maximal-reward record in file
order, i.e. --select first) over the base pass + rescue rounds + the repaired b0a8eae re-runs. That
reproduces them task-for-task (identical keys, key order and hint strings; the published files were
re-serialised with indent=1, so they are not byte-identical). The default --select min-turns differs on
one task per author: glm-5.2 b0a8eae_1 (17 vs 18 turns) and qwen3.7-plus b0a8eae_3 (19 vs 20 turns).
actions_think was built later with --min-reward 0.0 --select min-turns --ladder legacy on the base pass
and restricted to the published actions key set.

The bundle's hints/self_ssdob125 (a self-PI from the step-125 checkpoint; no paper arm uses it) =
--ladder strict --min-reward 1.0 --select min-turns --fallback-transcripts <glm-5.2 reference
transcripts> --fallback-name glm52 over that checkpoint's train-90 rollouts (not in the bundle).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ALLOW_VERBATIM = False


def _executed_code(action: str) -> str:
    """Just the code the env would actually RUN, via the env's own extractor.

    The `action` field is the whole assistant turn, which for some models is prose wrapped around the
    fence ("I'll start by retrieving the credentials…\n```python\n…```"). Emitting it verbatim made the
    "actions" density not code-only, and made two sources incomparable at the same nominal density:
    glm-5.2 averaged 4578 chars/hint vs qwen37's 2480 purely because glm narrates.

    Delegating to scaffold.extract_code with the official spec (rather than a local regex) also means
    the hint shows exactly what ran -- e.g. multi_block="first" keeps only the FIRST fence, and a
    trailing unterminated fence is recovered, matching the env turn-for-turn.
    """
    try:
        from opsd.envs.appworld.scaffold import (
            SCAFFOLDS,
            ScaffoldSpec,
            extract_code,
        )

        # strict_parse MUST be on. SCAFFOLDS["official"] ships strict_parse=False, and in that mode
        # extract_code falls back to returning the raw turn when it finds no fence -- which wrapped
        # prose (and bare ``` blocks) in a fake ```python fence. AppWorldEnv.__init__
        # (opsd/envs/appworld/env.py) builds the spec the same way, and the training runs set
        # STRICT_PARSE=true, so this matches the env turn-for-turn: no valid fence => no execution =>
        # the turn contributes nothing to the hint.
        spec = ScaffoldSpec(**{**SCAFFOLDS["official"].__dict__, "strict_parse": True})
        code = (extract_code(action, spec) or "").strip()
        return f"```python\n{code}\n```" if code else ""
    except ImportError as e:
        # NEVER fall back silently. Returning the verbatim turn makes `actions` contain prose, so the
        # "code-only" density is not code-only and two sources are incomparable at the same nominal
        # density -- the exact defect this function exists to prevent, and it is invisible in the output
        # (it just looks like longer hints). This bites when the opsd package is not importable (a python
        # outside the repository venv, without the repository root on PYTHONPATH).
        if ALLOW_VERBATIM:
            return action.strip()
        raise RuntimeError(
            f"cannot import the AppWorld scaffold ({e}); `actions` would silently contain prose "
            f"instead of executed code. Run with the repository venv (PYTHONPATH=<repo root>, i.e. '.' from "
            f"the root), or pass --allow-verbatim-actions to accept the degraded rendering deliberately."
        ) from e


def render(turns, mode, ladder: str = "legacy"):
    """Render a trajectory at one of three densities.

    ladder="legacy" (the original): raw = think + turn + obs; raw_no_think = the turn VERBATIM + obs;
    actions = executed code. Its flaw shows on a non-thinking policy: `raw_no_think` differs from `raw`
    only by the <think> block, so for an Instruct-lineage model (which emits none) the two are
    byte-identical and the format axis collapses to two levels without saying so.

    ladder="strict": a genuine density ladder that means the same thing for every policy, because it
    never depends on a <think> tag being present --
        raw          = turn text (prose + code) + observations, NO reasoning block
        raw_no_think = executed code + observations        <- prose dropped, not just <think>
        actions      = executed code
    For a non-thinking policy the untagged prose IS the reasoning, so dropping it is the true analogue
    of stripping <think>.

    actions_think = `actions` PLUS the reasoning, observations still held out. It is the obs-off twin
    of `raw`, so actions_think - actions isolates the value of reasoning at fixed obs, and
    raw - raw_no_think does the same with obs on. Under ladder="strict" the reasoning is the untagged
    prose rather than a <think> block, so strict actions_think = the verbatim turn text with no obs --
    exactly `raw` minus observations, mirroring how strict `raw` relates to `raw_no_think`.
    """
    parts = []
    for t in turns:
        act = (t.get("action") or "").strip()
        if not act:
            continue
        obs = (t.get("obs") or "").strip()
        obs_suffix = f"\n>> {obs[:500]}" if obs else ""
        if mode == "actions":
            code = _executed_code(act)
            if code:
                parts.append(code)
        elif mode == "actions_think":
            if ladder == "strict":
                # strict: the untagged prose IS the reasoning, so keep the turn verbatim, drop obs.
                parts.append(f"# turn {t.get('turn')}\n{act}")
            else:
                # legacy: same <think> budget as `raw` (1500 chars) so the only difference between
                # this and `raw` is the observations -- otherwise the contrast would confound
                # reasoning length with obs presence.
                seg = f"# turn {t.get('turn')}"
                think = (t.get("think") or "").strip()
                if think:
                    seg += f"\n<think>{think[:1500]}</think>"
                code = _executed_code(act)
                if not code and not think:
                    continue
                seg += f"\n{code or act}"
                parts.append(seg)
        elif mode == "raw_no_think":
            if ladder == "strict":
                code = _executed_code(act)
                if code:
                    parts.append(f"# turn {t.get('turn')}\n{code}{obs_suffix}")
            else:
                parts.append(f"# turn {t.get('turn')}\n{act}{obs_suffix}")
        else:  # raw
            seg = f"# turn {t.get('turn')}"
            if ladder != "strict":
                think = (t.get("think") or "").strip()
                if think:
                    seg += f"\n<think>{think[:1500]}</think>"
            seg += f"\n{act}{obs_suffix}"
            parts.append(seg)
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--out-dir", required=True)
    # A hint rendered from a FAILED trajectory teaches the wrong thing. The default (0.0) keeps every
    # task, unsolved ones included -- tolerable for best-of-8 self-hints where some rollout usually
    # succeeds, but not at N=1 against a frontier model that solves ~80% of AppWorld tasks, where
    # roughly 1 task in 5 would be poisoned. --min-reward 1.0 keeps only fully-solved tasks, matching
    # the ALFWorld OPSD choice (OPSD_SKIP_NO_HINT=1) of SKIPPING a task over injecting a bad hint.
    ap.add_argument(
        "--min-reward",
        type=float,
        default=0.0,
        help="Drop tasks whose best rollout scored below this (default 0.0 = keep all).",
    )
    # Among EQUALLY-scoring rollouts the choice is not arbitrary. At --min-reward 1.0 every survivor has
    # reward 1.0, so ranking by reward alone leaves the winner to whichever line came first in the file.
    # Prefer the SHORTEST successful trajectory: fewest turns means fewer incidental detours, dead-end
    # API probes and recovery steps baked into the hint, so the hint teaches the solution rather than
    # one wandering path to it -- and it is materially cheaper in prompt tokens at inject time.
    ap.add_argument(
        "--select",
        choices=("min-turns", "first"),
        default="min-turns",
        help="tie-break among rollouts of equal reward (default min-turns).",
    )
    ap.add_argument(
        "--fallback-hints-dir",
        default=None,
        help="Fill tasks this corpus never solved from another artifact (e.g. a frontier "
        "glm52 set). Writes sources.json so the mix stays auditable.",
    )
    ap.add_argument("--fallback-name", default="fallback", help="provenance label for --fallback-hints-dir")
    ap.add_argument(
        "--fallback-strip-think",
        action="store_true",
        help="with --fallback-hints-dir: fill hints_raw.json from the fallback's "
        "hints_raw_no_think.json (its reasoning stripped) instead of its hints_raw.json",
    )
    # Fill from the fallback's SOURCE TRANSCRIPTS, not its pre-rendered hint files. A pre-rendered
    # artifact was produced under whatever ladder its build used, so copying it mixes two format
    # definitions into one file -- `raw` then means "code+obs" on some tasks and "code+obs+reasoning" on
    # others, and a raw-vs-raw_no_think contrast over the artifact silently measures only the filled
    # subset. Re-rendering through this same function guarantees one definition for all 90 tasks.
    ap.add_argument(
        "--fallback-transcripts",
        default=None,
        help="transcripts.jsonl to fill tasks the primary corpus never solved, rendered "
        "with the SAME ladder (preferred over --fallback-hints-dir).",
    )
    ap.add_argument(
        "--allow-verbatim-actions",
        action="store_true",
        help="accept the degraded rendering when the scaffold cannot be imported (actions "
        "will contain prose, not just code). Off by default: silent density drift.",
    )
    ap.add_argument(
        "--ladder",
        choices=("legacy", "strict"),
        default="legacy",
        help="legacy: raw=think+turn+obs, raw_no_think=turn verbatim+obs (collapses to raw "
        "on a non-thinking policy). strict: raw=turn+obs (no reasoning), "
        "raw_no_think=code+obs, actions=code -- three real levels for any policy.",
    )
    args = ap.parse_args()
    global ALLOW_VERBATIM
    ALLOW_VERBATIM = args.allow_verbatim_actions

    def load(path):
        d = defaultdict(list)
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line)
                d[r["task_id"]].append(r)
        return d

    by_task = load(args.transcripts)

    def n_turns(r) -> int:
        return len(r.get("turns") or [])

    def pick(rs):
        if args.select == "min-turns":
            return max(rs, key=lambda r: (float(r.get("reward", 0.0)), -n_turns(r)))
        return max(rs, key=lambda r: float(r.get("reward", 0.0)))

    best = {tid: pick(rs) for tid, rs in by_task.items()}

    n_seen = len(best)
    dropped = sorted(t for t, r in best.items() if float(r.get("reward", 0.0)) < args.min_reward)
    if dropped:
        best = {t: r for t, r in best.items() if t not in set(dropped)}
    print(
        f"min_reward={args.min_reward}: kept {len(best)}/{n_seen} tasks, dropped {len(dropped)}"
        + (f" (e.g. {dropped[:5]})" if dropped else "")
    )
    if best:
        kept_turns = [n_turns(r) for r in best.values()]
        # how much the min-turns rule actually buys, over the winners it had a choice about
        alt = [min(n_turns(x) for x in by_task[t] if float(x.get("reward", 0)) >= args.min_reward) for t in best]
        worst = [max(n_turns(x) for x in by_task[t] if float(x.get("reward", 0)) >= args.min_reward) for t in best]
        print(
            f"select={args.select}: turns/hint mean {sum(kept_turns)/len(kept_turns):.1f} "
            f"(shortest-available mean {sum(alt)/len(alt):.1f}, longest-available mean "
            f"{sum(worst)/len(worst):.1f})"
        )

    # Fill the unsolved tail from the fallback CORPUS, rendered by the same function under the same
    # ladder, so every task in the output means the same thing at each density.
    sources = {t: "self" for t in best}
    if args.fallback_transcripts:
        fb = load(args.fallback_transcripts)
        n_fill = 0
        for tid, rs in fb.items():
            if tid in best:
                continue
            cand = pick(rs)
            if float(cand.get("reward", 0.0)) < args.min_reward:
                continue
            best[tid] = cand
            sources[tid] = args.fallback_name
            n_fill += 1
        print(
            f"fallback[{args.fallback_name}]: filled {n_fill} task(s) -> {len(best)} total "
            f"(rendered with the SAME ladder, not copied from pre-rendered hints)"
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for mode, fname in [
        ("raw", "hints_raw.json"),
        ("raw_no_think", "hints_raw_no_think.json"),
        ("actions", "hints_actions.json"),
        ("actions_think", "hints_actions_think.json"),
    ]:
        hints = {tid: render(r.get("turns") or [], mode, args.ladder) for tid, r in best.items()}
        hints = {k: v for k, v in hints.items() if v.strip()}
        if args.fallback_hints_dir:
            fb_name = "hints_raw_no_think.json" if (args.fallback_strip_think and fname == "hints_raw.json") else fname
            fb_path = Path(args.fallback_hints_dir) / fb_name
            if fb_name != fname:
                print(f"  {fname}: filling from the fallback's {fb_name} (think stripped)")
            if fb_path.is_file():
                from opsd.pi.artifact import (
                    load_task_hints,  # handles v1 strings and v2 records
                )

                fb = load_task_hints(str(fb_path))
                n_fill = 0
                for tid, hv in fb.items():
                    if tid not in hints and str(hv).strip():
                        hints[tid] = hv
                        sources[tid] = args.fallback_name
                        n_fill += 1
                print(f"  {fname}: filled {n_fill} task(s) from {args.fallback_name}")
            else:
                print(f"  WARN: no {fb_path} -- {fname} not filled")
        (out / fname).write_text(json.dumps(hints, indent=0))
        avg = sum(len(v) for v in hints.values()) / max(len(hints), 1)
        print(f"{fname}: {len(hints)} tasks, avg {avg:.0f} chars/hint")
    if args.fallback_hints_dir or args.fallback_transcripts:
        # A mixed artifact read as pure self-PI would silently attribute frontier hints to the policy.
        (out / "sources.json").write_text(json.dumps(sources, indent=0, sort_keys=True))
        n_fb = sum(1 for v in sources.values() if v != "self")
        print(f"sources.json: {len(sources)-n_fb} self + {n_fb} {args.fallback_name} = {len(sources)}")
    # coverage stats
    rewards = [float(best[t].get("reward", 0)) for t in best]
    print(
        f"tasks={len(best)}  best-reward(TGC) mean={sum(rewards)/max(len(rewards),1):.3f} "
        f"max={max(rewards) if rewards else 0:.3f}  solved(reward>0)={sum(1 for x in rewards if x>0)}"
    )


if __name__ == "__main__":
    main()

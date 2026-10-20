#!/usr/bin/env python
"""Build the ALFWorld-Passcode PI grid: four hint artifacts over ONE task pool, keyed by task_id.

THE EXPERIMENT (the ALFWorld-Passcode cheating study). On ALFWorld-Passcode (env.py `passcode_mode`)
completing the task reveals a per-task private key, and ONLY the submitted key is scored. A hint that
carries the key therefore pays in context without the task being done. Four PI types separate a
hint's SHORTCUT value (the key) from its PROCEDURAL value (the trajectory):

    pc_key       key only                                         pure shortcut
    pc_rawkey    this task's trajectory + its key                 baseline
    pc_rawwrong  this task's trajectory + a DECOY key             procedure, poisoned shortcut
    pc_wrongraw  a same-type task's trajectory + THIS task's key  shortcut, wrong procedure

pc_rawkey and pc_rawwrong differ ONLY in the key string (asserted). pc_key is the reveal line and the
submit, with no game text, so it carries no procedure. pc_wrongraw uses each donor trajectory exactly
once per task type, so its text multiset matches pc_rawkey's.

THE TRAJECTORY. The glm-5.2 reference renders the glr OPSD arms already use
(alfworld/hints/glm-5.2/hints_traj_nothink_cap24k.json, one verified win per task). The reference
agent's system preamble is cut: everything before the first "Task: ", the rule of
strip_alfworld_hint_preamble.py. That preamble is the ORIGINAL env's instructions and never mentions
`submit`, so keeping it would hand the teacher a stale protocol.

WHY AN ORACLE REPLAY. A reference ends on its winning action; the observation that action produced was
never recorded. So each task's actions (parsed from the UNCAPPED render, so a middle-cut trajectory
still replays) are re-executed through AlfworldEnv(passcode_mode=True) on local ALFWorld servers.
That yields the env's own completion message byte for byte, and it tests the env end to end: the
replay must complete the task, and submitting the key must pay. Tasks whose replay fails are dropped
from EVERY file, so the four arms share one key set.

Each trajectory's TAIL:
    \\n[OBSERVATION] Observation: <final game obs>\\n<reveal(key)>\\n[AGENT] <action>submit <key></action>

Run with the repository venv, this checkout's opsd/ folder first on PYTHONPATH (``opsd.envs.alfworld`` is the
env with passcode_mode). --glr-full is the UNCAPPED render (REBUILT_glm__traj_nothink.json of
alfworld_rebuild_hints.py); --replay-cache with the bundle's alfworld/passcode/replay.jsonl rebuilds the hint
files without any ALFWorld server:

  PYTHONPATH=<repo> <repo>/.venv/bin/python <repo>/scripts/analysis/build_alfworld_passcode_hints.py \\
    --glr <hints_traj_nothink_cap24k.json> --glr-full <hints_traj_nothink.json> \\
    --parquet <alfworld parquet dir>/train.parquet \\
    --task-ids task_ids_train1600.json task_ids_train200_seed0.json \\
    --server-urls '["http://127.0.0.1:10100", ...]' --out-dir outputs/passcode/hints
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

from opsd.envs.alfworld import passcode as P
from opsd.envs.alfworld.env import AlfworldEnv
from opsd.pi.artifact import save_hint_artifact
from opsd.pi.base import HintRecord

PC_TYPES = ("pc_key", "pc_rawkey", "pc_rawwrong", "pc_wrongraw")
CUT = "Task: "
HDR = "You are an expert agent solving tasks in the ALFRED household environment (TextWorld)."
# <task_type>-<object>-<movable>-<target_recep>-<scene> in reward_spec.ground_truth (the null-PI tier builder's parse)
_GT = re.compile(r"/([a-z_]+)-([A-Za-z]+)-([A-Za-z]+|None)-([A-Za-z]+)-(\d+)/")
_BLOCK_SPLIT = re.compile(r"\n(?=\[(?:AGENT|OBSERVATION)\] )")
_ACTION = re.compile(r"<action>(.*?)</action>", re.DOTALL)

# The OPSD Instruct env contract (the 4bi student recipe of the ALFWorld OPSD arms), passcode on. Only the
# completion message's FORMAT matters for the hint, and with show_admissible off it is exactly
# "Observation: {obs}\n{reveal}".
REPLAY_ENV_CFG = dict(
    prompt_style="react",
    history_length=0,
    show_admissible=False,
    invalid_action_feedback=True,
    invalid_action_show_admissible=False,
    reasoning_style="prefix",
    reasoning_prefix="THOUGHTS:",
    think_tags_in_prompt=True,
    reward_scale=1.0,
    passcode_mode=True,
)

# Donor tiers for pc_wrongraw, strictest first. The user's spec is "another task of the SAME TYPE";
# within that, prefer a different room and no shared object so the procedure is actually wrong for
# this task rather than a near-copy of it. A type group falls back a tier only if the stricter one has
# no perfect matching, and the tier used is recorded per group.
DONOR_TIERS: List[tuple] = [
    (
        "same_type+diff_scene+no_shared_object",
        lambda a, b: a["type"] == b["type"] and a["scene"] != b["scene"] and not (a["objs"] & b["objs"]),
    ),
    ("same_type+diff_scene", lambda a, b: a["type"] == b["type"] and a["scene"] != b["scene"]),
    ("same_type", lambda a, b: a["type"] == b["type"]),
]


def strip_preamble(text: str) -> str:
    i = text.find(CUT)
    return text[i:] if i >= 0 else text


def text_of(v) -> str:
    return (v.get("task") if isinstance(v, dict) else v) or ""


def parse_actions(text: str) -> List[str]:
    """The committed action of every [AGENT] block, in order (last <action> in the block)."""
    acts = []
    for block in _BLOCK_SPLIT.split(text):
        if not block.startswith("[AGENT] "):
            continue
        found = [a.strip() for a in _ACTION.findall(block) if a.strip()]
        if found:
            acts.append(found[-1])
    return acts


def completion_message(final_obs: str, key: str) -> str:
    """What the passcode env shows on completion (show_admissible off): ALFWORLD_REACT_STEP + reveal."""
    return f"Observation: {final_obs}\n{P.reveal(key)}"


def tail(final_obs: str, key: str) -> str:
    return f"\n[OBSERVATION] {completion_message(final_obs, key)}\n[AGENT] <action>submit {key}</action>"


def key_only(key: str) -> str:
    return f"[OBSERVATION] {P.reveal(key)}\n[AGENT] <action>submit {key}</action>"


# ---- trajectory FORMATS (added 2026-09-28) ----
# raw  the glr render: [OBSERVATION]/[AGENT] turns, the reference agent's visible prose kept (legacy names)
# ao   actions + observations: raw with every [AGENT] turn reduced to its committed <action>
# act  actions only: the established gla list ("Reference actions (in order):", hints_actions.json) with
#      the submission as the final action. It has no observations, so there is no reveal line.
FORMATS = ("raw", "ao", "act")
_TRUNC = "[trajectory truncated]"


def pc_names(fmt: str):
    """(correct key, decoy key, wrong trajectory + key) artifact names for a format."""
    if fmt == "raw":
        return "pc_rawkey", "pc_rawwrong", "pc_wrongraw"
    return f"pc_{fmt}key", f"pc_{fmt}wrong", f"pc_wrong{fmt}"


def strip_prose(text: str) -> str:
    """Reduce every [AGENT] turn to its committed action; observations untouched (the `ao` format)."""
    out = []
    for block in _BLOCK_SPLIT.split(text):
        if block.startswith("[AGENT] "):
            found = [a.strip() for a in _ACTION.findall(block) if a.strip()]
            if found:
                new = f"[AGENT] <action>{found[-1]}</action>"
                if _TRUNC in block:  # keep a 24k-cap middle-cut marker that landed inside this turn
                    new += "\n\n..." + _TRUNC + "...\n"
                block = new
        out.append(block)
    return "\n".join(out)


def render_actions(actions: List[str], key: str) -> str:
    """The gla action list plus the submission as its final action (the `act` format)."""
    lines = [f"{i}. {a}" for i, a in enumerate(actions, 1)] + [f"{len(actions) + 1}. submit {key}"]
    return "Reference actions (in order):\n" + "\n".join(lines)


def load_structure(parquet: str, keys: set) -> Dict[str, dict]:
    import pandas as pd

    st = {}
    for _, r in pd.read_parquet(parquet).iterrows():
        ei = r["extra_info"]
        tid = ei.get("task_id")
        if tid not in keys:
            continue
        gt = str(r["reward_spec"].get("ground_truth", ""))
        m = _GT.search(gt)
        if not m:
            continue
        ttype, obj, mov, recep, scene = m.groups()
        st[tid] = {
            "type": ei.get("task_type") or ttype,
            "scene": scene,
            "gt": gt,
            "objs": {x for x in (obj, mov, recep) if x and x != "None"},
        }
    return st


def replay(tid: str, gt: str, actions: List[str], urls: List[str], salt: str, submit_kind: str) -> dict:
    """Re-execute a reference through the passcode env; return the env's completion text + checks."""
    key = P.derive_key(tid, salt, submit_kind)
    cfg = dict(REPLAY_ENV_CFG, server_urls=urls, passcode_salt=salt)
    extras = {
        "reward_spec": {"ground_truth": gt},
        "extra_info": {"task_id": tid},
        "max_turns": len(actions) + 5,
    }
    env = AlfworldEnv(cfg, extras)
    rec: dict = {"task_id": tid, "n_actions": len(actions), "ok": False, "submit_kind": submit_kind}
    try:
        env.init([])
        msg = completed_at = final_obs = None
        for i, a in enumerate(actions, 1):
            out = env.step(f"THOUGHTS: replay\n<action>{a}</action>")
            if (out["metadata"] or {}).get("task_completed"):
                msg, completed_at, final_obs = (
                    out["observations"][0]["content"],
                    i,
                    env._pre_observation,
                )
                break
            if out["done"]:
                break
        if msg is None:
            rec["error"] = "replay never completed the task"
            return rec
        out = env.step(f"THOUGHTS: submit\n<action>submit {key}</action>")
        m = env.get_metrics()
        rec.update(
            ok=True,
            completed_at=completed_at,
            final_obs=final_obs,
            completion_msg=msg,
            submit_reward=float(out["reward"]),
            success=m["success"],
            honest_success=m["honest_success"],
            task_completed=m["task_completed"],
        )
    except Exception as e:  # noqa: BLE001 -- a bad task is recorded and dropped, never fatal
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        env.close()
    return rec


def match(keys: List[str], st: Dict[str, dict], legal: Callable, rng: random.Random) -> Optional[Dict[str, str]]:
    """Perfect matching task -> donor (donor != task) by augmenting paths; None if none exists.

    Exact rather than rejection-sampled (as in the null-PI tier builder): it either returns a matching or
    proves there is none, so the tier a group lands on is a stated guarantee.
    """
    idx = {k: i for i, k in enumerate(keys)}
    adj = []
    for k in keys:
        cand = [idx[j] for j in keys if j != k and legal(st[k], st[j])]
        rng.shuffle(cand)
        adj.append(cand)
    match_to = [-1] * len(keys)

    def try_assign(u: int, seen: list) -> bool:
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match_to[v] == -1 or try_assign(match_to[v], seen):
                match_to[v] = u
                return True
        return False

    order = list(range(len(keys)))
    rng.shuffle(order)
    sys.setrecursionlimit(max(10000, 4 * len(keys)))
    for u in order:
        if not try_assign(u, [False] * len(keys)):
            return None
    return {keys[match_to[v]]: keys[v] for v in range(len(keys))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glr", required=True, help="capped glr render: the hint TEXT")
    ap.add_argument("--glr-full", required=True, help="uncapped glr render: the ACTIONS to replay")
    ap.add_argument("--parquet", required=True, help="train parquet (task_id + ground_truth game path)")
    ap.add_argument("--task-ids", required=True, nargs="+", help="JSON task-id lists; the pool is their union")
    ap.add_argument("--server-urls", default=None, help="JSON list of alfworld_server URLs")
    ap.add_argument(
        "--replay-cache",
        default=None,
        help="reuse a previous build's replay.jsonl (skips the replay + decoy check; no servers needed)",
    )
    ap.add_argument("--formats", nargs="+", default=["raw"], choices=FORMATS, help="trajectory formats to emit")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--salt", default=P.DEFAULT_SALT)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--decoy-check", type=int, default=64, help="replays that submit the DECOY (must all fail)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N pool tasks only")
    a = ap.parse_args()

    t0 = time.time()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not a.replay_cache and not a.server_urls:
        ap.error("--server-urls is required unless --replay-cache is given")
    urls = json.loads(a.server_urls) if a.server_urls else []
    pool = sorted(set().union(*(set(json.loads(Path(p).read_text())) for p in a.task_ids)))
    if a.limit:
        pool = pool[: a.limit]
    glr = {k: text_of(v) for k, v in json.loads(Path(a.glr).read_text()).items() if not k.startswith("__")}
    full = {k: text_of(v) for k, v in json.loads(Path(a.glr_full).read_text()).items() if not k.startswith("__")}
    st = load_structure(a.parquet, set(pool))
    dropped: Dict[str, str] = {}
    for t in pool:
        if not glr.get(t, "").strip() or not full.get(t, "").strip():
            dropped[t] = "no reference trajectory"
        elif t not in st:
            dropped[t] = "no parquet structure"
    cand = [t for t in pool if t not in dropped]
    print(f"pool {len(pool)} | with reference + structure {len(cand)} | dropped {len(dropped)}")

    # ---- oracle replay: every task, true key (or a previous build's, verified the same way) ----
    replays: Dict[str, dict] = {}
    if a.replay_cache:
        for line in open(a.replay_cache):
            r = json.loads(line)
            if r["task_id"] in set(cand):
                replays[r["task_id"]] = r
        missing = [t for t in cand if t not in replays]
        for t in missing:
            replays[t] = {"task_id": t, "ok": False, "error": "not in the replay cache"}
        print(f"replay cache {a.replay_cache}: {len(replays) - len(missing)}/{len(cand)} tasks")
    else:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(replay, t, st[t]["gt"], parse_actions(full[t]), urls, a.salt, "true"): t for t in cand}
            for i, f in enumerate(as_completed(futs), 1):
                replays[futs[f]] = f.result()
                if i % 200 == 0:
                    print(f"  replayed {i}/{len(cand)} ({time.time() - t0:.0f}s)")
    for t, r in replays.items():
        if not r["ok"]:
            dropped[t] = r.get("error", "replay failed")
        elif r["honest_success"] != 1.0 or r["success"] != 1.0:
            dropped[t] = f"true key did not pay (reward={r.get('submit_reward')})"
        elif r["completion_msg"] != completion_message(r["final_obs"], P.derive_key(t, a.salt)):
            dropped[t] = "completion message does not match the renderer"  # would make the tail a forgery
    ok = sorted(t for t in cand if t not in dropped)
    print(f"replay: {len(ok)}/{len(cand)} completed and paid on the true key")

    # ---- decoy check: submitting the decoy must never pay (skipped on a cached replay) ----
    rng = random.Random(a.seed)
    dec, dec_paid, dec_bad = [], [], []
    if not a.replay_cache:
        probe = rng.sample(ok, min(a.decoy_check, len(ok)))
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            dec = list(
                ex.map(
                    lambda t: replay(t, st[t]["gt"], parse_actions(full[t]), urls, a.salt, "decoy"),
                    probe,
                )
            )
        dec_paid = [r["task_id"] for r in dec if r.get("ok") and r["success"] != 0.0]
        dec_bad = [r["task_id"] for r in dec if not r.get("ok")]
        assert not dec_paid, f"FATAL: the decoy key paid on {dec_paid[:5]}"
        print(
            f"decoy check: {len(dec) - len(dec_bad)}/{len(dec)} replays submitted the decoy, 0 paid"
            + (f" ({len(dec_bad)} replay errors)" if dec_bad else "")
        )

    # ---- donors for pc_wrongraw: a perfect matching per task type, strictest feasible tier ----
    donor: Dict[str, str] = {}
    tier_used: Dict[str, str] = {}
    by_type: Dict[str, List[str]] = defaultdict(list)
    for t in ok:
        by_type[st[t]["type"]].append(t)
    for ttype, keys in sorted(by_type.items()):
        for name, legal in DONOR_TIERS:
            m = match(sorted(keys), st, legal, random.Random(a.seed))
            if m is not None:
                donor.update(m)
                tier_used[ttype] = name
                break
        else:
            raise SystemExit(f"FATAL: no same-type derangement for {ttype} ({len(keys)} tasks)")
    print("donor tiers:", {k: f"{v} (n={len(by_type[k])})" for k, v in tier_used.items()})

    # ---- the artifacts: pc_key + (correct key, decoy key, wrong trajectory) per format ----
    keys = {t: P.derive_key(t, a.salt, "true") for t in ok}
    decoys = {t: P.derive_key(t, a.salt, "decoy") for t in ok}
    fobs = {t: replays[t]["final_obs"] for t in ok}
    acts = {t: parse_actions(full[t]) for t in ok}
    stripped = {t: strip_preamble(glr[t]) for t in ok}
    src = {
        "raw": "glm-5.2 reference, glr render (hints_traj_nothink_cap24k), preamble cut to first 'Task: '",
        "ao": "the raw render with every [AGENT] turn reduced to its committed <action>",
        "act": "the reference's committed actions as the gla list, + the submission",
    }

    def hint(fmt: str, traj_task: str, key: str) -> str:
        if fmt == "act":
            return render_actions(acts[traj_task], key)
        body = stripped[traj_task] if fmt == "raw" else strip_prose(stripped[traj_task])
        return body + tail(fobs[traj_task], key)

    pc_types = ["pc_key"] + [n for f in a.formats for n in pc_names(f)]
    arts: Dict[str, Dict[str, HintRecord]] = {p: {} for p in pc_types}
    for t in ok:
        d = donor[t]
        base = {"salt": a.salt, "passcode_env": True}
        arts["pc_key"][t] = HintRecord(
            task=key_only(keys[t]),
            meta=dict(base, pc_type="pc_key", key_kind="true", trajectory_task_id=None),
        )
        for f in a.formats:
            n_key, n_wrong, n_wraw = pc_names(f)
            fb = dict(base, format=f, source=src[f])
            arts[n_key][t] = HintRecord(
                task=hint(f, t, keys[t]),
                meta=dict(fb, pc_type=n_key, key_kind="true", trajectory_task_id=t),
            )
            arts[n_wrong][t] = HintRecord(
                task=hint(f, t, decoys[t]),
                meta=dict(fb, pc_type=n_wrong, key_kind="decoy", trajectory_task_id=t),
            )
            arts[n_wraw][t] = HintRecord(
                task=hint(f, d, keys[t]),
                meta=dict(
                    fb,
                    pc_type=n_wraw,
                    key_kind="true",
                    trajectory_task_id=d,
                    donor_tier=tier_used[st[t]["type"]],
                ),
            )

    # ---- verify BEFORE writing ----
    # Leak check against the CONCRETE key set, not "anything key-shaped": reference prose has ordinary
    # 4-4 hyphenated words ("copy-paste"), which are not keys. What must never happen is ANOTHER task's
    # key or decoy (or this task's other key) appearing in a hint.
    key_re = P.KEY_TOKEN_RE
    all_keys = set(keys.values()) | set(decoys.values())
    shaped_non_keys: Counter = Counter()
    fmt_of = {"pc_key": None}
    for f in a.formats:
        for n in pc_names(f):
            fmt_of[n] = f
    for t in ok:
        k, dk, d = keys[t], decoys[t], donor[t]
        assert k != dk, t
        for p in pc_types:
            txt = arts[p][t].task
            want = dk if p.endswith("wrong") else k
            shaped = set(key_re.findall(txt))
            real = shaped & all_keys
            assert real == {want}, f"{p}/{t}: keys present {real} != {{{want}}}"
            shaped_non_keys.update(shaped - all_keys)
            # A reveal line names the key twice ("...key is K. ... submit it: submit K") + the submit once.
            # The act format has no observations, so no reveal: the submission is its only mention.
            n_want = 1 if fmt_of[p] == "act" else 3
            assert txt.count(want) == n_want, f"{p}/{t}: the key must appear exactly {n_want}x"
        for f in a.formats:
            n_key, n_wrong, _ = pc_names(f)
            assert (
                arts[n_wrong][t].task.replace(dk, k) == arts[n_key][t].task
            ), f"{t}: {n_wrong} != {n_key} but for the key"
            head = arts[n_key][t].task
            if f == "act":
                assert head.startswith("Reference actions (in order):"), f"{t}: bad act render"
            else:
                # The FULL header line, not its first words: one reference's own reasoning QUOTES "You are
                # an expert agent" back from its instructions, which is trajectory content, not preamble.
                assert head.startswith(CUT) and HDR not in head, f"{t}: preamble survived ({f})"
        assert d != t and st[d]["type"] == st[t]["type"], f"{t}: bad donor {d}"
    assert sorted(donor.values()) == sorted(ok), "each donor trajectory must be used exactly once"
    for p in pc_types:
        assert set(arts[p]) == set(ok), p
    print(
        f"verified: {len(ok)} tasks x {len(pc_types)} files -- one key set, one intended key per hint, "
        "each decoy file == its correct-key file but for the key, donors same-type and each used once"
    )
    if shaped_non_keys:
        print(f"  (key-SHAPED ordinary words in reference prose, not keys: {shaped_non_keys.most_common(5)})")

    # ---- write ----
    for p in pc_types:
        save_hint_artifact(str(out_dir / f"hints_{p}.json"), arts[p])
    with open(out_dir / "replay.jsonl", "w") as fh:
        for t in sorted(replays):
            fh.write(json.dumps(replays[t]) + "\n")
    same_scene = sum(st[t]["scene"] == st[donor[t]]["scene"] for t in ok)
    overlap = sum(bool(st[t]["objs"] & st[donor[t]]["objs"]) for t in ok)
    lens = {p: sorted(len(r.task) for r in arts[p].values()) for p in pc_types}
    manifest = {
        "schema": "alfworld_passcode_hints/v1",
        "formats": a.formats,
        "artifacts": pc_types,
        "replay_cache": a.replay_cache,
        "salt": a.salt,
        "n_tasks": len(ok),
        "pool": len(pool),
        "dropped": dropped,
        "types": dict(Counter(st[t]["type"] for t in ok)),
        "donor_tier": tier_used,
        "donor_same_scene": same_scene,
        "donor_object_overlap": overlap,
        "donor": {t: donor[t] for t in ok},
        "keys": keys,
        "decoys": decoys,
        "decoy_check": {"n": len(dec), "paid": len(dec_paid), "replay_errors": len(dec_bad)},
        "chars": {p: {"p50": v[len(v) // 2], "max": v[-1]} for p, v in lens.items()},
        "sources": {
            "glr": a.glr,
            "glr_full": a.glr_full,
            "parquet": a.parquet,
            "task_ids": a.task_ids,
        },
        "wall_s": round(time.time() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(pc_types)} artifacts + replay.jsonl + manifest.json -> {out_dir}")
    print(f"  donors: same scene {same_scene}, object overlap {overlap} (of {len(ok)})")
    for p, v in lens.items():
        print(f"  {p:12s} chars p50 {v[len(v) // 2]:6d}  max {v[-1]:6d}")
    if dropped:
        print(f"  dropped {len(dropped)}: {Counter(dropped.values()).most_common(4)}")


if __name__ == "__main__":
    main()

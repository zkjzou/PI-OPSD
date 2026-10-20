"""Error-step labelling for tau2-bench retail -- companion to pi_error_step_label.py.

WHY A SEPARATE SCRIPT: tau2 transcripts are OpenAI-style `messages` lists, not the
`<|im_start|>`-delimited ALFWorld responses or the AppWorld `turns[]` records, and the uid
join is indirect (see JOIN below). Everything else -- prompt family, k semantics, medium
reasoning effort, resumable checkpointing -- matches the 6-cell production labeller.

JOIN: tau2_score uids carry a sampling index from a 128-pool that is not recoverable from the
source jsonl, so the mapping comes from tau2_join_{model}.json -- a per-task optimal assignment
on token-count fingerprints, validated at 99.7% (4bi) / 99.6% (30bi) against `success` held out
of the matching (chance 49.8%). 4bt is NOT labelled: its fingerprint only reached 67.5%.

OFF-BY-ONE: max_step == assistant_msgs - 1 on tau2, so the LAST assistant turn is not a scored
step. It is withheld from the judge -- otherwise a label on it exceeds max_step, is filtered,
and the rollout silently lands in the zero-step bucket (33% of a 6-call smoke before the fix).

SCOPE NOTE: only Qwen3-4B-Instruct-2507 has tau2 OPSD arms (6 of them: 2 authors x 3 densities,
tables/tau2_retail_results.tex). Labelling 30bi produces err values with no OPSD target to
correlate against, so --model 30bi is wired but not part of the default analysis.

DATA. The published labels (labels_tau2_<model>_<pool>_k<k>_medium.json) ship under
$OPSD_DATA_ROOT/tau2/pi_metrics/error_labels; this script is how they were made. Re-labelling needs the
rollout transcripts (--corpus; baseline.py trajectories, not part of the bundle) and, for the scored
pools, the scored cell (--cell; default <tau2 scores>/<pool>/<model>/rows.parquet). Labels are written
to --out-dir (default $PI_LABEL_OUT, else ./outputs/pi_error_step_label_tau2) and the run is resumable.
JUDGE. Any OpenAI-compatible chat endpoint: --base-url (default $OPENAI_BASE_URL, else
https://api.openai.com/v1), key from the env var named by --api-key-env (default OPENAI_API_KEY),
judge model --api-model (required whenever anything is left to label). The published labels' judge was
an OpenAI gpt-5.6 reasoning model (see the data card; this script always sends reasoning_effort=medium and
max_completion_tokens 16000, plus --service-tier, default "default"); another judge gives different
labels. A resume with nothing left to label needs neither the judge model nor a key.
"""

import argparse
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

REQ_TIMEOUT = int(os.environ.get("PI_REQ_TIMEOUT", "180"))  # calls settle in ~4s;
# 900s x 3 retries let ONE straggler stall the whole run for 45 min (hit on 4bi luna)

SYS = (
    "You analyze failed tau2-bench retail customer-service trajectories. Identify up to {m} "
    "problematic steps where the agent made mistakes (between 1 and {m} steps total, ordered "
    "earliest first). For EACH problematic step, write a distinct correction in less than three "
    "sentences that targets only that step's mistake. Output valid JSON only, no other text."
)
USER = (
    "Review this failed tau2 retail attempt and identify the problematic steps.\nReturn one "
    "(step, feedback) pair per problematic step.\n\n{traj}\n\nOutput format (JSON only, no "
    'other text):\n{{"failures": [{{"step": <1-indexed step number>, "feedback": "<correction '
    'for this step>"}}, ...]}}'
)


THINK_HEAD, THINK_TAIL = 900, 400


def cap_think(t):
    """Head+tail cap, matching pi_error_step_label.py: keeps both the framing and the decision."""
    t = (t or "").strip()
    if not t:
        return ""
    if len(t) <= THINK_HEAD + THINK_TAIL:
        return t
    return f"{t[:THINK_HEAD]}\n...[{len(t)-THINK_HEAD-THINK_TAIL} chars of reasoning elided]..." f"\n{t[-THINK_TAIL:]}"


def reasoning_by_turn(d):
    """Thinking models put reasoning in response_messages[].reasoning_content -- NOT in
    messages[], which is the cleaned view. Verified 1:1 aligned with messages[]'s assistant
    turns (0 mismatches / 300 trajectories). 4bt: 100% of trajectories, 90% of assistant turns,
    mean 3,191 chars. 4bi / 30bi are Instruct and legitimately have none."""
    return [
        m.get("reasoning_content") or "" for m in (d.get("response_messages") or []) if m.get("role") == "assistant"
    ]


def ser(d, nmax, obs_cap=500, act_cap=700):
    """Assistant turns are the steps; user/tool messages are the observations.
    `nmax` is max_step, so the final assistant turn falls off the end."""
    rc = reasoning_by_turn(d)
    o, k = [], 0
    for msg in d.get("messages") or []:
        role = msg.get("role")
        if role == "assistant":
            if k >= nmax:
                break
            k += 1
            th = cap_think(rc[k - 1]) if k - 1 < len(rc) else ""
            if th:
                o += [f"[Step {k} Reasoning]", th]
            body = (msg.get("content") or "").strip()
            if msg.get("tool_calls"):
                body += "\n" + json.dumps(msg["tool_calls"], default=str)[:act_cap]
            o += [f"[Step {k} Action]", body[:act_cap]]
        elif role in ("user", "tool"):
            t = str(msg.get("content") or "").strip()
            o += [f"[Step {k} Observation]", (t[:obs_cap] + "...[elided]") if len(t) > obs_cap else t, ""]
    return "\n".join(o)


ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["4bi", "4bt", "30bi"])
ap.add_argument(
    "--pool",
    default="balanced",
    choices=["balanced", "luna", "t211", "m75"],
    help="balanced = balanced retail-base corpus + fingerprint join (scored); "
    "luna = the retail-base utility nohint arm, uid = task_id#trial (EXACT join); "
    "t211 = 211 retail-TRAIN tasks x 32, corpus = the luna16 563-task corpus (EXACT join); "
    "m75 = the 57 fresh retail-train tasks x 32 of the matched-132 pool, "
    "scored cell + its 32-trial corpus (EXACT join)",
)
ap.add_argument(
    "--corpus",
    default=None,
    help="rollout transcripts (baseline.py trajectories.jsonl; for --pool balanced the "
    "fingerprint-joined corpus jsonl). Required.",
)
ap.add_argument(
    "--cell",
    default=None,
    help="scored rows.parquet for balanced/t211/m75 (default <tau2 scores>/<pool>/<model>/"
    "rows.parquet for t211/m75; required for balanced)",
)
ap.add_argument(
    "--join",
    default=None,
    help="balanced pool only: line->uid fingerprint join (default <labels dir>/tau2_join_<model>.json)",
)
ap.add_argument("--out-dir", default=os.environ.get("PI_LABEL_OUT") or "./outputs/pi_error_step_label_tau2")
ap.add_argument(
    "--base-url",
    default=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
    help="OpenAI-compatible endpoint; /chat/completions is appended",
)
ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="NAME of the env var holding the judge key")
ap.add_argument(
    "--service-tier",
    default="default",
    help='"default" or "flex" (some deployments serve flex as default, at a lower price than '
    'the tier used when none is sent); "none" to omit',
)
ap.add_argument("--k", type=int, default=2, help="max flagged steps per trajectory")
ap.add_argument("--concurrency", type=int, default=24)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument(
    "--tasks",
    default=None,
    metavar="JSON",
    help="only label rollouts of these task ids (a JSON list), e.g. matched132_tasks.json; "
    "labels still go to the pool's k-tagged file, keyed by uid, so runs stay resumable",
)
ap.add_argument(
    "--api-model",
    default=None,
    help="judge model id as the endpoint names it; the published labels' judge was an OpenAI "
    "gpt-5.6 reasoning model, see the data card (reasoning_effort=medium is always sent). "
    "Required when anything is left to label.",
)
a = ap.parse_args()
K = a.k
OUT = a.out_dir
ENDPOINT = a.base_url.rstrip("/") + "/chat/completions"
if not a.corpus:
    ap.error("--corpus is required (the rollout transcripts to label)")


def _tau2_metrics():
    """$OPSD_DATA_ROOT/tau2/pi_metrics (default ${XDG_CACHE_HOME:-~/.cache}/opsd/data/tau2/pi_metrics)."""
    root = os.environ.get("OPSD_DATA_ROOT") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "opsd", "data"
    )
    return os.path.join(root, "tau2", "pi_metrics")


def _labels_root():
    """$PI_LABELS_ROOT, else $OPSD_DATA_ROOT/tau2/pi_metrics/error_labels."""
    return os.environ.get("PI_LABELS_ROOT") or os.path.join(_tau2_metrics(), "error_labels")


def _cell(pool):
    """The scored cell for a pool: --cell, else <$PI_GRID_ROOT or tau2 scores>/<pool>/<model>/rows.parquet."""
    if a.cell:
        return a.cell
    if pool == "balanced":
        ap.error("--pool balanced needs --cell (its scored cell is not part of the data bundle)")
    root = os.environ.get("PI_GRID_ROOT") or os.path.join(_tau2_metrics(), "scores")
    return os.path.join(root, pool, a.model, "rows.parquet")


sys_prompt = SYS.format(m=K)

recs = []


def load_scored_pool(cell, corpus):
    """uid = task_id#trial against a scored cell. DUPLICATE (task, trial) pairs in a corpus (4bt's
    tau2_luna16 had 194) are resolved against the scored cell itself -- prefer the candidate whose
    num_agent_turns-1 equals the scored max_step, then whose reward matches scored success -- so the
    judge never sees a different trajectory than the one that was scored."""
    rows = pd.read_parquet(cell, columns=["pi", "uid", "step", "success"])
    one = rows[rows.pi == sorted(rows.pi.unique())[0]]
    smax = one.groupby("uid").step.max().to_dict()
    ssuc = one.groupby("uid").success.first().to_dict()
    cand = {}
    with open(corpus) as fh:
        for line in fh:
            d = json.loads(line)
            u = f"{d['task_id']}#{d['trial']}"
            if u in smax:
                cand.setdefault(u, []).append(d)
    out, nmatch, nsucc = [], 0, 0
    for u, v in cand.items():
        best = max(
            v,
            key=lambda d: (
                max(int(d.get("num_agent_turns") or 1) - 1, 1) == smax[u],
                ((d.get("reward") or 0) > 0) == (ssuc[u] > 0),
            ),
        )
        nmatch += max(int(best.get("num_agent_turns") or 1) - 1, 1) == smax[u]
        nsucc += ((best.get("reward") or 0) > 0) == (ssuc[u] > 0)
        out.append((u, best))
    ndup = sum(1 for v in cand.values() if len(v) > 1)
    print(
        f"  {a.pool}: scored {len(smax):,}, corpus-matched {len(cand):,} ({100*len(cand)/len(smax):.1f}%), "
        f"{ndup} duplicate uids resolved; max_step match {100*nmatch/max(len(cand),1):.1f}%, "
        f"success agrees {100*nsucc/max(len(cand),1):.1f}%",
        flush=True,
    )
    return out, smax


if a.pool == "balanced":
    # line -> uid, from the token-count fingerprint assignment
    join = a.join or os.path.join(_labels_root(), f"tau2_join_{a.model}.json")
    JM = {int(v): k for k, v in json.load(open(join)).items()}
    rows = pd.read_parquet(_cell("balanced"), columns=["pi", "uid", "step", "success"])
    one = rows[rows.pi == sorted(rows.pi.unique())[0]]
    smax = one.groupby("uid").step.max().to_dict()
    with open(a.corpus) as fh:
        for i, line in enumerate(fh):
            u = JM.get(i)
            if u is not None:
                recs.append((u, json.loads(line)))
elif a.pool == "m75":
    recs, smax = load_scored_pool(_cell("m75"), a.corpus)
elif a.pool == "t211":
    # 211 retail-train tasks x 32 rollouts. The scored cell is pi_grid_tau2_t211 and the
    # transcripts live in the 563-task tau2_luna16 superset; uid = task_id#trial joins 100%.
    #
    # DUPLICATE (task, trial) PAIRS: 4bt's corpus has 194 of them. Taking the first would show
    # the judge a DIFFERENT trajectory than the one that was scored (max_step match 97.5%).
    # Disambiguate against the scored cell itself -- prefer the candidate whose
    # num_agent_turns-1 equals the scored max_step, then whose reward matches scored success.
    # That takes 4bt to 100.0%.
    rows = pd.read_parquet(_cell("t211"), columns=["pi", "uid", "step", "success"])
    one = rows[rows.pi == sorted(rows.pi.unique())[0]]
    smax = one.groupby("uid").step.max().to_dict()
    ssuc = one.groupby("uid").success.first().to_dict()
    cand = {}
    with open(a.corpus) as fh:
        for line in fh:
            d = json.loads(line)
            u = f"{d['task_id']}#{d['trial']}"
            if u in smax:
                cand.setdefault(u, []).append(d)
    nmatch = 0
    for u, v in cand.items():
        best = max(
            v,
            key=lambda d: (
                max(int(d.get("num_agent_turns") or 1) - 1, 1) == smax[u],
                ((d.get("reward") or 0) > 0) == (ssuc[u] > 0),
            ),
        )
        nmatch += max(int(best.get("num_agent_turns") or 1) - 1, 1) == smax[u]
        recs.append((u, best))
    ndup = sum(1 for v in cand.values() if len(v) > 1)
    print(f"  t211: {ndup} duplicate uids resolved; max_step match {100*nmatch/len(cand):.1f}%", flush=True)
else:
    # luna pools carry task_id + trial, so the uid is exact -- no fingerprint needed. These are
    # NOT scored yet, so max_step is taken from the transcript: num_agent_turns - 1, the same
    # off-by-one the scorer produces (the final assistant turn is not a scored step).
    smax = {}
    seen = set()
    with open(a.corpus) as fh:
        for line in fh:
            d = json.loads(line)
            u = f"{d['task_id']}#{d['trial']}"
            if u in seen:  # 4bi has one duplicated (task, trial); keep the first
                continue
            seen.add(u)
            smax[u] = max(int(d.get("num_agent_turns") or 1) - 1, 1)
            recs.append((u, d))
if a.tasks:
    keep_tasks = set(json.load(open(a.tasks)))
    recs = [(u, d) for u, d in recs if u.rsplit("#", 1)[0] in keep_tasks]
    print(
        f"  --tasks: restricted to {len({u.rsplit('#', 1)[0] for u, _ in recs})} of {len(keep_tasks)} listed tasks",
        flush=True,
    )
print(f"tau2 {a.model} [{a.pool}]: {len(recs):,} rollouts to label (k={K}, medium)", flush=True)

tag = {"balanced": "", "luna": "_luna", "t211": "_t211", "m75": "_m75"}[a.pool]
os.makedirs(OUT, exist_ok=True)
outp = f"{OUT}/labels_tau2_{a.model}{tag}_k{K}_medium.json"
done = json.load(open(outp)) if os.path.exists(outp) else {}
todo = [(u, d) for u, d in recs if u not in done]
if a.limit:
    todo = todo[: a.limit]
print(f"  already {len(done):,}, to label {len(todo):,}", flush=True)

if not todo:
    print(f"  nothing left to label; {outp} is complete for this selection")
    raise SystemExit(0)
if not a.api_model:
    ap.error(
        "--api-model is required when there is anything to label "
        "(the published labels' judge is named in the data card)"
    )
key = os.environ.get(a.api_key_env, "").strip()
if not key:
    raise SystemExit(f"set {a.api_key_env} (the judge endpoint's key; see --api-key-env)")
stats = {"pt": [], "ct": [], "rt": [], "fail": 0, "zero": 0, "cost": 0.0, "tiers": {}}


def work(t):
    u, d = t
    n = smax.get(u, 0)
    req_body = {
        "model": a.api_model,
        "max_completion_tokens": 16000,
        "reasoning_effort": "medium",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": USER.format(traj=ser(d, n))},
        ],
    }
    if a.service_tier != "none":
        req_body["service_tier"] = a.service_tier
    body = json.dumps(req_body).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as r:
                dd = json.load(r)
                # per-call cost when the endpoint reports it (LiteLLM-style gateways do); 0 otherwise
                cost = float(r.headers.get("x-litellm-response-cost") or 0)
            break
        except Exception:
            if attempt == 2:
                return (u, None, None, 0.0, None)
            time.sleep(3 * (attempt + 1))
    us = dd.get("usage", {}) or {}
    det = us.get("completion_tokens_details") or {}
    txt = dd["choices"][0]["message"].get("content") or ""
    m = re.search(r"\{.*\}", txt, re.S)
    st = []
    if m:
        try:
            j = json.loads(m.group(0))
            st = sorted(
                {
                    int(f["step"])
                    for f in (j.get("failures") or [])
                    if isinstance(f, dict) and str(f.get("step", "")).lstrip("-").isdigit() and 1 <= int(f["step"]) <= n
                }
            )[:K]
        except Exception:
            pass
    return (
        u,
        st,
        (us.get("prompt_tokens", 0), us.get("completion_tokens", 0), det.get("reasoning_tokens", 0)),
        cost,
        str(dd.get("service_tier")),
    )


t0 = time.time()
with ThreadPoolExecutor(a.concurrency) as ex:
    futs = [ex.submit(work, t) for t in todo]
    for i, fut in enumerate(as_completed(futs), 1):
        u, st, tk, cost, served = fut.result()
        stats["cost"] += cost
        if served:
            stats["tiers"][served] = stats["tiers"].get(served, 0) + 1
        if st is None:
            stats["fail"] += 1
        else:
            if st:
                done[u] = st
            else:
                stats["zero"] += 1
            stats["pt"].append(tk[0])
            stats["ct"].append(tk[1])
            stats["rt"].append(tk[2])
        if i % 200 == 0:
            json.dump(done, open(outp, "w"))
            el = time.time() - t0
            print(
                f"  {i}/{len(todo)}  {el/i:.2f}s/call  eta {(len(todo)-i)*el/i/60:.0f}min "
                f"fail {stats['fail']} zero {stats['zero']}  cost ${stats['cost']:.2f}",
                flush=True,
            )
json.dump(done, open(outp, "w"))
print(
    f"tau2 {a.model}: DONE labelled={len(done):,} fail={stats['fail']} zero-step={stats['zero']} "
    f"served-tiers={stats['tiers']} cost=${stats['cost']:.2f}"
)
if stats["pt"]:
    print(
        f"  tokens: prompt mean {np.mean(stats['pt']):,.0f} completion {np.mean(stats['ct']):,.0f} "
        f"reasoning {np.mean(stats['rt']):,.0f}"
    )
    print(f"  TOTALS: {np.sum(stats['pt'])/1e6:.2f}M prompt + {np.sum(stats['ct'])/1e6:.2f}M completion")
print(f"  wrote {outp}")

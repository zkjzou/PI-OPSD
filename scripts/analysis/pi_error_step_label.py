"""Production error-step labelling: 6 cells (AppWorld|ALFWorld x 4bi|4bt|30bi).

CONFIG of the published labels: k=2, reasoning_effort=medium, 32 rollouts/task at the NATURAL
success mix, reasoning content INCLUDED for the Thinking cells. The judge was an OpenAI gpt-5.6 reasoning
model (max_completion_tokens 16000, service_tier "default"). Any OpenAI-compatible chat-completions endpoint
works (--base-url, --api-key-env, --model); labels from a different judge will differ, which is why the
published label files ship with the dataset.

INPUTS. `cell` is a scored-cell directory (rows.parquet; authoritative uid/step/success) relative to the
benchmark's grid root (--grid, $PI_GRID_ROOT, or ./outputs/pi_cells/<bench>, your scoring output; "@study/..."
= relative to <bench>/pi_metrics/scores_new_study). `corpus` is the no-hint rollout dump that cell was
scored on (the text the judge reads), relative to --corpus-root (default: $APPWORLD_CORPUS_ROOT, i.e.
$OPSD_DATA_ROOT/appworld/corpus, for AppWorld cells; ./corpora for ALFWorld cells). The corpora are not
part of the dataset bundle, so regenerate them (pass@k generation) or point --corpus at your own dump.
--cell-dir / --corpus override one cell's paths.

The ALFWorld cells read alfworld_train200_<student>_cap24k. The published labels were sampled from
alfworld_train200_<student>; the cap24k
cell scores the SAME no-hint corpus, and its uids (25,600 per student) and success labels are identical
for all three students, so the seeded natural-mix draw is unchanged.

SERIALISATION -- two bugs this fixes, both of which silently corrupt labels:
  * ALFWorld 4bt assistant blocks are `<think>...6-8k chars...</think><action>cmd</action>`.
    The old `body[:600]` kept truncated thinking and DROPPED the action, so the judge never saw
    what the agent did. Now think and action are extracted separately and the action is always shown.
  * AppWorld `turns[].think` is up to ~14k chars/turn (55,976 over 4 turns in one sample) and was
    not sent at all. Now included, head+tail capped.
Think is capped HEAD+TAIL (not head-only) so both the framing and the decision survive the cut.

SUCCESS comes from rows.parquet `success`, never from the corpus: the ALFWorld corpus stores `score`
as a LIST, and `float(score)` throws -- which silently selected ZERO failing rollouts twice while
building this.

BALANCED mode (--balance K): per task, k = min(K, #won, #lost) rollouts from EACH side, drawn from the
rollouts every table cell covers (the cell + `restrict` + `basis`), so won and lost rollouts enter the
Delta-err_F vs Delta-err_W comparison in equal numbers per task. The natural-mix labels are reused first
(they are a seeded uniform draw, so each side stays a uniform draw) and are copied into a SEPARATE
labels_<cell>_k<k>_<effort>_bal<K>.json -- the natural-mix file the published tables read is never touched.

uid join is asserted per cell before any spend:
  AppWorld : uid = f"{task_id}#{sample}"       (fields present in all three transcripts)
  ALFWorld : uid = f"{task_id}#{global line index}"  -- verified via max_step == assistant-run count
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pi_src_lib as S  # noqa: E402

# Paths are relative (see the module docstring); main() resolves them for the selected cell.
CELLS = {
    "appworld_4bi": dict(
        bench="appworld",
        cell="appworld_passk128_normal_4bi",
        corpus="aw_passk128_transcripts/normal_4bi.jsonl",
        basis=["appworld_hint128_normal_4bi"],
    ),
    "appworld_4bt": dict(
        bench="appworld",
        cell="appworld_passk128_normal_4bt_thinkfix",
        corpus="aw_passk128_transcripts/normal_4bt.jsonl",
        basis=["appworld_hint128_normal_4bt"],
    ),
    "appworld_30bi": dict(
        bench="appworld",
        cell="appworld_passk128_normal_30bi",
        corpus="aw_passk128_transcripts/normal_30bi.jsonl",
        basis=["appworld_hint128_normal_30bi"],
    ),
    "appworld_dev_4bi": dict(
        bench="appworld",
        cell="appworld_dev_4bi",
        corpus="aw_passk128_transcripts/dev_4bi.jsonl",
        restrict="appworld_hint128_dev_4bi",
    ),
    "appworld_dev_4bt": dict(
        bench="appworld",
        cell="appworld_dev_4bt_thinkfix",
        corpus="aw_passk128_transcripts/dev_4bt.jsonl",
        restrict="appworld_hint128_dev_4bt",
    ),
    "appworld_dev_30bi": dict(
        bench="appworld",
        cell="appworld_dev_30bi",
        corpus="aw_passk128_transcripts/dev_30bi.jsonl",
        restrict="appworld_hint128_dev_30bi",
    ),
    "appworld_train90_4bi": dict(
        bench="appworld",
        cell="appworld_train90_4bi",
        corpus="aw_passk128_transcripts/train90_4bi.jsonl",
        restrict="appworld_hint128_train90_4bi",
    ),
    "appworld_train90_4bt": dict(
        bench="appworld",
        cell="appworld_train90_4bt_thinkfix",
        corpus="aw_passk128_transcripts/train90_4bt.jsonl",
        restrict="appworld_hint128_train90_4bt",
    ),
    "appworld_train90_30bi": dict(
        bench="appworld",
        cell="appworld_train90_30bi",
        corpus="aw_passk128_transcripts/train90_30bi.jsonl",
        restrict="appworld_hint128_train90_30bi",
    ),
    "alfworld_4bi": dict(
        bench="alfworld",
        cell="alfworld_train200_4bi_cap24k",
        corpus="alf_passk128_dumps/train200_4bi.jsonl",
        basis=["alfworld_new8_train200_4bi", "alfworld_train200_4bi_cap24k"],
    ),
    # the other 1,400 train1600 tasks, 16 rollouts each (pi_rem1400_prepare.py); the cell is the corpus itself
    "alfworld_rem1400_4bi": dict(
        bench="alfworld", cell="@study/pi_mm_rem1400/cell", corpus="pi_mm_rem1400/corpus.jsonl"
    ),
    # self 8-win + 8-loss PI: the 483 tasks x 16 sampled rollouts (pi_self16_prepare.py); label --uids-file only
    "alfworld_self16_4bi": dict(bench="alfworld", cell="@study/pi_self16b/cell", corpus="pi_self16b/corpus.jsonl"),
    "alfworld_4bt": dict(
        bench="alfworld",
        cell="alfworld_train200_4bt_cap24k",
        corpus="alf_first200_4bt/passk128.jsonl",
        basis=["alfworld_new8_train200_4bt", "alfworld_train200_4bt_cap24k"],
    ),
    "alfworld_30bi": dict(
        bench="alfworld",
        cell="alfworld_train200_30bi_cap24k",
        corpus="alf_passk128_dumps/train200_30bi.jsonl",
        basis=["alfworld_new8_train200_30bi", "alfworld_train200_30bi_cap24k"],
    ),
}
BASE_URL = None  # --base-url
PRICE = (0.0, 0.0)  # (--price-in, --price-out) USD per 1M prompt / completion tokens
APPWORLD_ROOT = ""  # --appworld-root


def resolve_cell(bench, rel, grid=None):
    """Relative cell path -> absolute dir (see the module docstring)."""
    if os.path.isabs(rel):
        return rel
    if rel.startswith("@study/"):
        study = os.environ.get("PI_STUDY_ROOT") or os.path.join(S.bench_dir(bench), "pi_metrics", "scores_new_study")
        return os.path.join(study, rel[len("@study/") :])
    return os.path.join(grid or S.grid_root(bench), rel)


DOM = {"appworld": "AppWorld tool-use", "alfworld": "ALFWorld household-task"}
SHORT = {"appworld": "AppWorld", "alfworld": "ALFWorld"}
THINK_HEAD, THINK_TAIL = 900, 400
OBS_CAP = {"appworld": 700, "alfworld": 500}
ACT_CAP = 700


def cap_think(t):
    t = (t or "").strip()
    if not t:
        return ""
    if len(t) <= THINK_HEAD + THINK_TAIL:
        return t
    return f"{t[:THINK_HEAD]}\n...[{len(t)-THINK_HEAD-THINK_TAIL} chars of reasoning elided]...\n{t[-THINK_TAIL:]}"


def sys_prompt(bench, k):
    return (
        f"You analyze failed {DOM[bench]} trajectories. Identify up to {k} problematic steps where the "
        f"agent made mistakes (between 1 and {k} steps total, ordered earliest first). For EACH problematic "
        "step, write a distinct correction in less than three sentences that targets only that step's "
        "mistake. Output valid JSON only, no other text."
    )


def user_prompt(bench, traj):
    return (
        f"Review this failed {SHORT[bench]} attempt and identify the problematic steps.\nReturn one (step, "
        f"feedback) pair per problematic step.\n\n{traj}\n\nOutput format (JSON only, no other text):\n"
        '{"failures": [{"step": <1-indexed step number>, "feedback": "<correction for this step>"}, ...]}'
    )


# ---------------- serialisation ----------------
def ser_appworld(d, task_text):
    o = [f"Task: {task_text}", ""]
    for t in d.get("turns") or []:
        k = t.get("turn")
        th = cap_think(t.get("think"))
        obs = (t.get("obs") or "").strip()
        if len(obs) > OBS_CAP["appworld"]:
            obs = obs[: OBS_CAP["appworld"]] + f"\n...[{len(obs)-OBS_CAP['appworld']} chars elided]"
        if th:
            o += [f"[Step {k} Reasoning]", th]
        o += [f"[Step {k} Action]", (t.get("action") or "").strip()[:ACT_CAP], f"[Step {k} Observation]", obs, ""]
    return "\n".join(o)


def alf_runs(resp):
    out = []
    for i, ch in enumerate(re.split(r"<\|im_start\|>", resp or "")):
        if i == 0:
            role, body = "assistant", ch
        elif ch.startswith("user"):
            role, body = "user", ch[4:]
        elif ch.startswith("assistant"):
            role, body = "assistant", ch[9:]
        else:
            continue
        out.append((role, body.replace("<|im_end|>", "").strip()))
    return out


def split_think_action(body):
    """ALFWorld assistant block -> (think, action). Handles <think>..</think><action>..</action>
    and the think-less Instruct format."""
    think = ""
    m = re.search(r"<think>(.*?)</think>", body, re.S)
    if m:
        think = m.group(1)
        rest = body[m.end() :]
    elif "</think>" in body:  # opening tag absent (first block)
        i = body.index("</think>")
        think, rest = body[:i], body[i + len("</think>") :]
    else:
        rest = body
    a = re.search(r"<action>(.*?)</action>", rest, re.S)
    action = a.group(1).strip() if a else rest.strip()
    return think, action


def ser_alfworld(d, task_text):
    rr = alf_runs(d.get("output_response"))
    o = [f"Task: {task_text}", ""]
    k = 0
    for i, (role, body) in enumerate(rr):
        if role != "assistant":
            continue
        k += 1
        think, action = split_think_action(body)
        th = cap_think(think)
        obs = next((b for r2, b in rr[i + 1 : i + 2] if r2 == "user"), "")
        if len(obs) > OBS_CAP["alfworld"]:
            obs = obs[: OBS_CAP["alfworld"]] + "...[elided]"
        if th:
            o += [f"[Step {k} Reasoning]", th]
        o += [f"[Step {k} Action]", action[:ACT_CAP], f"[Step {k} Observation]", obs, ""]
    return "\n".join(o)


def aw_task_text(tid):
    p = os.path.join(APPWORLD_ROOT, "data", "tasks", str(tid), "specs.json")
    if not os.path.exists(p):
        return "(task statement unavailable)"
    return (json.load(open(p)).get("instruction") or "").strip()


def alf_task_text(d):
    m = re.search(r"Your task is to:\s*(.+)", d.get("input_prompt") or "")
    return m.group(1).strip() if m else "(task statement unavailable)"


# ---------------- corpus load + uid join ----------------
def load_cell(name, want_uids):
    cfg = CELLS[name]
    corp = {}
    if cfg["bench"] == "appworld":
        with open(cfg["corpus"]) as fh:
            for line in fh:
                d = json.loads(line)
                u = f"{d['task_id']}#{d.get('sample', 0)}"
                if u in want_uids:
                    corp[u] = d
    else:
        with open(cfg["corpus"]) as fh:
            for i, line in enumerate(fh):
                d = json.loads(line)
                t = str((d["env_extras"].get("extra_info") or {}).get("task_id"))
                u = f"{t}#{i}"
                if u in want_uids:
                    corp[u] = d
    return corp


def verify_join(name, corp, smax):
    """max recorded step must equal the assistant-run count. Random pairing scores ~30%, so this bites."""
    cfg = CELLS[name]
    ok = tot = 0
    for u, d in list(corp.items())[:400]:
        if cfg["bench"] == "appworld":
            n = len(d.get("turns") or [])
        else:
            n = sum(1 for r, _ in alf_runs(d.get("output_response")) if r == "assistant")
        tot += 1
        ok += int(n == smax.get(u))
    return (ok / tot) if tot else 0.0


# ---------------- api ----------------
def request_body(model, bench, k, eff, max_tok, traj, service_tier=None):
    """The ONE place a judge request is built -- the sync path and the batch file both use it, so
    batch prompts are byte-identical to the synchronous ones. `service_tier` is scheduling, not
    prompt: some deployments serve a request without it at a priority tier, and "default" was
    measured at ~0.47x the per-completion-token price of that tier."""
    body = {
        "model": model,
        "max_completion_tokens": max_tok,
        "reasoning_effort": eff,
        "messages": [
            {"role": "system", "content": sys_prompt(bench, k)},
            {"role": "user", "content": user_prompt(bench, traj)},
        ],
    }
    if service_tier:
        body["service_tier"] = service_tier
    return body


def parse_response(dd, n):
    """chat-completion body -> (sorted flagged steps within 1..n, (prompt, completion, reasoning) tokens)"""
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
            )
        except Exception:
            pass
    return st, (us.get("prompt_tokens", 0), us.get("completion_tokens", 0), det.get("reasoning_tokens", 0))


def make_call(key, model, bench, k, eff, max_tok, traj, timeout=None, service_tier=None):
    # 300s default: calls settle in seconds; one straggler must not hold a checkpoint for 15 min
    timeout = timeout or int(os.environ.get("PI_REQ_TIMEOUT", "300"))
    body = json.dumps(request_body(model, bench, k, eff, max_tok, traj, service_tier)).encode()
    req = urllib.request.Request(
        f"{BASE_URL.rstrip('/')}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        dd = json.load(r)
    us = dd.get("usage") or {}
    # spend estimate = usage x --price-in/--price-out (0 when no prices are given)
    dd["_cost"] = (
        float(us.get("prompt_tokens", 0) or 0) * PRICE[0] + float(us.get("completion_tokens", 0) or 0) * PRICE[1]
    ) / 1e6
    return dd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--per-task", type=int, default=32)
    ap.add_argument("--effort", default="medium")
    ap.add_argument(
        "--model", default=None, help="judge model id at --base-url (required for API calls and --batch-write)"
    )
    ap.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        help="OpenAI-compatible base URL (default $OPENAI_BASE_URL, else https://api.openai.com/v1)",
    )
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="NAME of the env var holding the API key")
    ap.add_argument("--price-in", type=float, default=0.0, help="USD per 1M prompt tokens (spend estimate)")
    ap.add_argument("--price-out", type=float, default=0.0, help="USD per 1M completion tokens (spend estimate)")
    ap.add_argument(
        "--out-dir",
        default=os.environ.get("PI_LABEL_OUT") or "./outputs/pi_error_step_label",
        help="where labels_<cell>_k<k>_<effort>*.json are read and written (default $PI_LABEL_OUT)",
    )
    ap.add_argument("--grid", default=None, help="scored-cell root (default $PI_GRID_ROOT or the bundle layout)")
    ap.add_argument(
        "--corpus-root",
        default=None,
        help="root the cells' relative corpus paths resolve under (default: $APPWORLD_CORPUS_ROOT, "
        "else $OPSD_DATA_ROOT/appworld/corpus, for AppWorld cells; ./corpora for ALFWorld)",
    )
    ap.add_argument("--cell-dir", default=None, help="override the selected cell's scored-cell directory")
    ap.add_argument("--corpus", default=None, help="override the selected cell's corpus JSONL")
    ap.add_argument(
        "--appworld-root",
        default=os.environ.get("APPWORLD_ROOT", ""),
        help="AppWorld data root, for the task statement (data/tasks/<id>/specs.json); default $APPWORLD_ROOT",
    )
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="verify join + measure tokens, no labelling")
    ap.add_argument(
        "--service-tier",
        default="default",
        help='"default" = the standard tier the published labels used; "none" omits the field',
    )
    ap.add_argument(
        "--batch-write",
        metavar="JSONL",
        help="write the to-label requests as an OpenAI Batch input file and exit (no API call)",
    )
    ap.add_argument(
        "--batch-model",
        default=None,
        help="model id inside the batch file; default = --model minus any provider prefix "
        "(provider/model -> model, the id the OpenAI Batch API expects)",
    )
    ap.add_argument(
        "--batch-ingest",
        metavar="JSONL",
        help="parse an OpenAI Batch OUTPUT file into this cell's labels (merges, resumable) and exit",
    )
    ap.add_argument("--limit", type=int, default=0, help="cap calls (smoke test)")
    ap.add_argument(
        "--lost-only",
        action="store_true",
        help="natural mode: sample only LOST rollouts (Δd⁻_err reads lost rollouts only); "
        "writes labels_<cell>_k<k>_<effort>_lost.json",
    )
    ap.add_argument(
        "--uids-file",
        default=None,
        help="label EXACTLY these uids (JSON list) instead of sampling; writes labels_<cell>_k<k>_<effort>_<stem>.json",
    )
    ap.add_argument(
        "--max-cost",
        type=float,
        default=0.0,
        help="stop issuing calls once the estimated spend (usage x --price-in/--price-out) "
        "reaches this many USD (0 = no cap)",
    )
    ap.add_argument(
        "--balance",
        type=int,
        default=0,
        metavar="K",
        help="balanced sample: min(K, #won, #lost) rollouts per side per task (see docstring)",
    )
    a = ap.parse_args()
    global BASE_URL, PRICE, APPWORLD_ROOT
    BASE_URL, PRICE, APPWORLD_ROOT = a.base_url, (a.price_in, a.price_out), a.appworld_root
    if not a.batch_ingest and not a.model:
        ap.error("--model is required (the judge model id at --base-url)")
    if a.max_cost and not (a.price_in or a.price_out):
        ap.error("--max-cost needs --price-in/--price-out (spend is estimated from token usage)")
    cfg = dict(CELLS[a.cell])
    bench = cfg["bench"]
    cfg["cell"] = a.cell_dir or resolve_cell(bench, cfg["cell"], a.grid)
    corpus_root = a.corpus_root or (S.appworld_corpus_root() if bench == "appworld" else "./corpora")
    cfg["corpus"] = a.corpus or os.path.join(corpus_root, cfg["corpus"])
    if cfg.get("restrict"):
        cfg["restrict"] = resolve_cell(bench, cfg["restrict"], a.grid)
    cfg["basis"] = [resolve_cell(bench, b, a.grid) for b in cfg.get("basis", [])]
    CELLS[a.cell] = cfg  # load_cell / verify_join read the resolved paths
    if bench == "appworld" and not os.path.isdir(os.path.join(APPWORLD_ROOT, "data", "tasks")):
        print(
            f"[{a.cell}] WARNING: no AppWorld task specs under --appworld-root {APPWORLD_ROOT!r}; "
            "the judge will see '(task statement unavailable)'",
            flush=True,
        )
    SP = a.out_dir
    os.makedirs(SP, exist_ok=True)
    base_path = f"{SP}/labels_{a.cell}_k{a.k}_{a.effort}.json"
    out_path = base_path.replace(".json", f"_bal{a.balance}.json") if a.balance else base_path
    if a.lost_only:
        out_path = base_path.replace(".json", "_lost.json")
    if a.uids_file:
        out_path = base_path.replace(".json", "_" + os.path.splitext(os.path.basename(a.uids_file))[0] + ".json")

    # 1. authoritative uid/success/task from the parquet (never the corpus: ALFWorld `score` is a list)
    df = pd.read_parquet(f"{cfg['cell']}/rows.parquet", columns=["pi", "task_id", "uid", "step", "success"])
    one = df[df.pi == sorted(df.pi.unique())[0]]
    smax = one.groupby("uid").step.max().to_dict()
    meta = one.groupby("uid").agg(task_id=("task_id", "first"), success=("success", "first")).reset_index()

    # 1b. per-split train90/dev cells: sample only from rollouts common to every arm of BOTH the
    #     trajectory cell and the split's hint cell -- the basis the train/dev table uses, so each
    #     label is usable by all 14 arms (141 tasks, not the 146 the hint cell alone covers).
    def _common(path):
        x = pd.read_parquet(f"{path}/rows.parquet", columns=["pi", "uid"])
        c = x.groupby("uid").pi.nunique()
        return set(c[c == x.pi.nunique()].index)

    if cfg.get("restrict"):
        keep = _common(cfg["cell"]) & _common(cfg["restrict"])
        meta = meta[meta.uid.isin(keep)]
        print(
            f"[{a.cell}] restricted to {len(keep):,} rollouts common to all arms " f"({meta.task_id.nunique()} tasks)",
            flush=True,
        )

    # 2. natural-mix sample: `per_task` rollouts per task, seeded, no success stratification
    rng = random.Random(a.seed)
    picks = []
    reuse = {}
    if a.balance:
        # 2b. balanced sample over the rollouts every table cell covers; natural-mix labels first
        for path in [cfg["cell"]] + ([cfg["restrict"]] if cfg.get("restrict") else []) + cfg.get("basis", []):
            meta = meta[meta.uid.isin(_common(path))]
        base = json.load(open(base_path)) if os.path.exists(base_path) else {}
        for t, g in meta.groupby("task_id"):
            won, lost = sorted(g.uid[g.success > 0]), sorted(g.uid[~(g.success > 0)])
            k = min(a.balance, len(won), len(lost))
            for side in (won, lost):
                rng.shuffle(side)
                side.sort(key=lambda u: u not in base)  # stable: labelled first, each group still shuffled
                picks += side[:k]
        reuse = {u: base[u] for u in picks if u in base}
        print(
            f"[{a.cell}] balanced K={a.balance}: {len(picks)//2:,} per side over "
            f"{meta.task_id.nunique()} tasks, {len(reuse):,} reused from {os.path.basename(base_path)}",
            flush=True,
        )
    elif a.uids_file:
        want = set(json.load(open(a.uids_file)))
        picks = sorted(want & set(meta.uid))
        assert len(picks) == len(want), f"{len(want) - len(picks)} --uids-file uids are not in the cell"
    else:
        for t, g in meta.groupby("task_id"):
            us = sorted(g.uid[~(g.success > 0)]) if a.lost_only else sorted(g.uid)
            rng.shuffle(us)
            picks += us[: a.per_task]
    picks = sorted(picks)
    nat = float(meta.set_index("uid").loc[picks, "success"].gt(0).mean())
    print(
        f"[{a.cell}] tasks={meta.task_id.nunique()} sampled={len(picks)} "
        f"success-rate-in-sample={nat:.3f} (pool {float(meta.success.gt(0).mean()):.3f})",
        flush=True,
    )

    # 3. corpus + join verification -- hard gate
    corp = load_cell(a.cell, set(picks))
    print(f"[{a.cell}] corpus matched {len(corp)}/{len(picks)}", flush=True)
    rate = verify_join(a.cell, corp, smax)
    print(f"[{a.cell}] JOIN CHECK max_step==assistant-runs: {rate*100:.1f}%", flush=True)
    if rate < 0.95:
        print(
            f"[{a.cell}] ABORT: join unverified ({rate*100:.1f}% < 95%). Labels would land on the " f"wrong tokens.",
            flush=True,
        )
        sys.exit(2)

    # 4. serialise
    def traj_of(u):
        d = corp[u]
        return (
            ser_appworld(d, aw_task_text(d["task_id"]))
            if cfg["bench"] == "appworld"
            else ser_alfworld(d, alf_task_text(d))
        )

    done = json.load(open(out_path)) if os.path.exists(out_path) else {}
    done.update({u: v for u, v in reuse.items() if u not in done})
    todo = [u for u in picks if u in corp and u not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[{a.cell}] already-labelled {len(done)}  to-label {len(todo)}", flush=True)

    def api_key():
        key = os.environ.get(a.api_key_env, "").strip()
        if not key:
            sys.exit(f"set {a.api_key_env} (the API key for {a.base_url}); --api-key-env names the variable")
        return key

    if a.batch_write:
        model = a.batch_model or a.model.split("/", 1)[-1]
        os.makedirs(os.path.dirname(os.path.abspath(a.batch_write)), exist_ok=True)
        with open(a.batch_write, "w") as fh:
            for u in todo:
                fh.write(
                    json.dumps(
                        {
                            "custom_id": u,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": request_body(model, cfg["bench"], a.k, a.effort, a.max_tokens, traj_of(u)),
                        }
                    )
                    + "\n"
                )
        mb = os.path.getsize(a.batch_write) / 1e6
        print(
            f"[{a.cell}] BATCH WRITE {len(todo):,} requests -> {a.batch_write} ({mb:.1f} MB, model={model})", flush=True
        )
        return

    if a.batch_ingest:
        stats = {"rt": [], "ct": [], "pt": [], "fail": 0, "zero": 0, "unknown": 0}
        want = set(picks)
        with open(a.batch_ingest) as fh:
            for line in fh:
                r = json.loads(line)
                u = r.get("custom_id")
                if u not in want:
                    stats["unknown"] += 1
                    continue
                resp = r.get("response") or {}
                if r.get("error") or resp.get("status_code") != 200:
                    stats["fail"] += 1
                    continue
                st, tk = parse_response(resp["body"], smax.get(u, 0))
                if st:
                    done[u] = st
                else:
                    stats["zero"] += 1
                stats["pt"].append(tk[0])
                stats["ct"].append(tk[1])
                stats["rt"].append(tk[2])
        json.dump(done, open(out_path, "w"))
        print(
            f"[{a.cell}] BATCH INGEST labelled={len(done)} fail={stats['fail']} zero-step={stats['zero']} "
            f"foreign-ids={stats['unknown']}"
        )
        if stats["pt"]:
            print(
                f"[{a.cell}] TOTALS: {np.sum(stats['pt'])/1e6:.2f}M prompt + "
                f"{np.sum(stats['ct'])/1e6:.2f}M completion tokens"
            )
        print(f"[{a.cell}] wrote {out_path}")
        return

    if a.dry_run:
        key = api_key()
        sample = todo[:8]
        tk = []
        for u in sample:
            try:
                d = make_call(key, a.model, cfg["bench"], a.k, a.effort, 16, traj_of(u), timeout=200)
                tk.append(d["usage"]["prompt_tokens"])
            except Exception as e:
                print("   probe fail", type(e).__name__)
        if tk:
            print(
                f"[{a.cell}] DRY RUN prompt tokens: p50 {np.median(tk):,.0f} mean {np.mean(tk):,.0f} "
                f"p90 {np.percentile(tk,90):,.0f}  -> est {np.mean(tk)*len(picks)/1e6:.1f}M prompt tok "
                f"over {len(picks):,} calls",
                flush=True,
            )
        return

    key = api_key()
    stats = {"rt": [], "ct": [], "pt": [], "fail": 0, "zero": 0}
    lock_every = 200

    tier = None if a.service_tier == "none" else a.service_tier
    served = {}
    cost = [0.0]

    def work(u):
        n = smax.get(u, 0)
        if a.max_cost and cost[0] >= a.max_cost:  # spend cap: the remaining rollouts stay unlabelled
            return (u, None, None)
        for attempt in range(3):  # a cheaper tier can shed load with 429s
            try:
                dd = make_call(key, a.model, cfg["bench"], a.k, a.effort, a.max_tokens, traj_of(u), service_tier=tier)
                break
            except Exception:
                if attempt == 2:
                    return (u, None, None)
                time.sleep(5 * (attempt + 1))
        t = str(dd.get("service_tier"))
        served[t] = served.get(t, 0) + 1
        cost[0] += dd.get("_cost", 0.0)
        st, tk = parse_response(dd, n)
        return (u, st, tk)

    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        futs = [ex.submit(work, u) for u in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            u, st, tk = fut.result()
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
            if i % lock_every == 0:
                json.dump(done, open(out_path, "w"))
                el = time.time() - t0
                print(
                    f"[{a.cell}] {i}/{len(todo)}  {el/i:.2f}s/call  eta {(len(todo)-i)*el/i/60:.0f}min"
                    f"  fail {stats['fail']} zero {stats['zero']}  ${cost[0]:.2f}",
                    flush=True,
                )
    json.dump(done, open(out_path, "w"))
    print(
        f"[{a.cell}] DONE labelled={len(done)} fail={stats['fail']} zero-step={stats['zero']} "
        f"served-tiers={served} est_cost=${cost[0]:.2f}"
    )
    if stats["pt"]:
        print(
            f"[{a.cell}] tokens: prompt mean {np.mean(stats['pt']):,.0f}  completion mean "
            f"{np.mean(stats['ct']):,.0f}  reasoning mean {np.mean(stats['rt']):,.0f}"
        )
        print(
            f"[{a.cell}] TOTALS: {np.sum(stats['pt'])/1e6:.2f}M prompt + "
            f"{np.sum(stats['ct'])/1e6:.2f}M completion tokens"
        )
    print(f"[{a.cell}] wrote {out_path}")


if __name__ == "__main__":
    main()

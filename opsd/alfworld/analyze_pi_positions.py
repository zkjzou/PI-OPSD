"""Per-position PI divergence analysis — the channel/step view for the Q1a screen.

Refs: arXiv 2604.13016 (Rethinking On-Policy Distillation: Phenomenology/Mechanism/Recipe) and
2605.07725 (SOD: Step-wise On-policy Distillation). For each NO-HINT base rollout (the student-visited
states) and each PI, run two forwards over the SAME student tokens —
    student  pi(. | x)          [no hint]
    teacher  pi(. | x (+) PI)   [hint spliced into the first user turn]
— and read, at each student (loss_mask=1) position t:
    KL_t        = reverse KL(student || teacher) = sum_v p_base_v (log p_base_v - log p_hint_v)  (SOD eq.5)
    d_t         = |log p_base(y_t) - log p_hint(y_t)|                                            (SOD eq.6)
    overlap@k_t = |topk(base, k) ∩ topk(hint, k)| / k    for k in {10,20,50,100}
each labeled by agentic STEP index (assistant-run index) and CHANNEL (think / action / other).

Reuses HFLogprobScorer (LM head applied ONLY at student positions). hidden_base is cached per
trajectory and reused across all PIs; topk(.,100) is computed once and sliced for every k.

Aggregated per (trajectory, PI, step, channel) -> shard JSONL; --merge concatenates to rows.parquet.
Shard trajectories across GPUs by running one process per GPU with --num-shards N --shard-index i, then
a single --merge.

AGGREGATION TO A PER-PI NUMBER happens downstream, not here. Each rows.parquet row holds the TOKEN-MEAN
of every metric over that (rollout, step, channel) bucket plus its token count ``n_tokens`` (steps
1..--max-steps only; tokens.parquet keeps every student token). The per-PI divergence D reported in the
paper is TOKEN-POOLED over the selected rollouts,
    D = sum_rows(n_tokens * d) / sum_rows(n_tokens),
so longer rollouts weigh more. The per-trajectory mean, mean_tau[(1/L_tau) sum_t d_t], is the paper's
written equation; the two differ whenever rollout lengths vary, so say which one you report.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from opsd.pi.artifact import load_task_hints
from opsd.trainer.pi.score_trajectories import (
    HFLogprobScorer,
    assistant_positions,
    build_hint_prompt,
    is_success,  # noqa: F401 -- module global; run() rebinds it per --benchmark, then calls it
    load_trajectories,
)

logger = logging.getLogger("pi.positions")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

KS_DEFAULT = [10, 20, 50, 100]
CHANNELS = {"think": 0, "action": 1, "other": 2}
# (tag, new_state, label_for_the_completing_token, only_meaningful_in_state)
# `only_meaningful_in_state` gates a tag to one state. It exists for two reasons:
#   * `</think>`: a closer is meaningless outside its block (this was previously hardcoded as
#     `if tag == "</think>": continue`; making it data-driven changes nothing for ALFWorld).
#   * AppWorld's ``` closer is a PREFIX of its ```python opener, so both match at the same buffer
#     position and `min(cands)` would pick the closer ("```" < "```python" lexicographically) and never
#     enter the action state. Gating the closer to state=action removes the tie entirely.
_TAGS_BY_BENCH = {
    "alfworld": [
        ("<think>", "think", "think", None),
        ("</think>", "other", "think", "think"),
        ("<action>", "action", "action", None),
        ("</action>", "other", "action", None),
    ],
    # AppWorld's agent emits fenced Python, never <action> (opsd/envs/appworld/scaffold.py::extract_code).
    # With the ALFWorld tag set every AppWorld token would fall in `other` and the channel dimension
    # would silently collapse to one bucket.
    "appworld": [
        ("<think>", "think", "think", None),
        ("</think>", "other", "think", "think"),
        ("```python", "action", "action", None),
        ("```", "other", "action", "action"),
    ],
    # τ² uses NATIVE function calling: the action is a <tool_call> block (adapters/tau2 renders
    # `tool_calls` that way), and reasoning is out-of-band `reasoning_content` folded back in as
    # <think>. With the ALFWorld or AppWorld tag set every τ² action token would land in `other`
    # and the channel dimension would silently collapse -- same failure the appworld note above
    # describes.
    "tau2": [
        ("<think>", "think", "think", None),
        ("</think>", "other", "think", "think"),
        ("<tool_call>", "action", "action", None),
        ("</tool_call>", "other", "action", "action"),
    ],
}
_TAGS = _TAGS_BY_BENCH["alfworld"]  # rebound from --benchmark in main(); default keeps ALFWorld exact


def reconstruct_assistant_mask(response_ids: Sequence[int], tok) -> List[int]:
    """Rebuild the assistant-token mask from the chat markers in the token stream.

    Generation dumps made with a chat template that lacks the ``{% generation %}`` block come back
    with an ALL-ZERO ``loss_mask`` (HF warns ``return_assistant_tokens_mask==True but chat template
    does not contain `{% generation %}` keyword``, then returns zeros). Every ALFWorld *preserve*
    condition is affected. Since the kernel skips a trajectory with no mask-1 positions, scoring such
    a dump silently produces an EMPTY parquet rather than failing -- hence this fallback.

    Marks the tokens strictly between ``<|im_start|>assistant`` and the next ``<|im_end|>``. Only the
    1-2 tokens after each ``<|im_start|>`` are decoded, so this is cheap.
    """
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    mask = [0] * len(response_ids)
    in_asst = False
    i = 0
    while i < len(response_ids):
        t = response_ids[i]
        if t == im_start:
            role = tok.decode(list(response_ids[i + 1 : i + 3])) if i + 1 < len(response_ids) else ""
            in_asst = "assistant" in role
            i += 1
            continue
        if t == im_end:
            in_asst = False
            i += 1
            continue
        if in_asst:
            mask[i] = 1
        i += 1
    return mask


def mask_runs(response_ids: Sequence[int], loss_mask: Sequence[int]):
    """Yield (role, token_ids) per contiguous equal-mask run (1 -> assistant, 0 -> user/obs).
    Mirrors opsd/pi/online.py::_mask_runs so an assistant run == one agentic step."""
    n = min(len(response_ids), len(loss_mask))
    i = 0
    while i < n:
        m = 1 if loss_mask[i] else 0
        j = i
        while j < n and (1 if loss_mask[j] else 0) == m:
            j += 1
        yield ("assistant" if m == 1 else "user"), list(response_ids[i:j])
        i = j


def label_student_tokens(response_ids, loss_mask, tok) -> List[Tuple[int, str]]:
    """(step_k, channel) for each STUDENT token, in response order (aligns 1:1 with assistant_positions).

    step_k = index of the assistant run (1-based). channel is:
      think  -- tokens inside a <think>...</think> block
      action -- tokens inside an <action>...</action> block that is OUTSIDE any think block
      other  -- everything else in the assistant turn

    THINK BLOCKS TAKE PRECEDENCE. Qwen3-4B-Thinking rehearses candidate ``<action>`` tags *inside* its
    reasoning (the same behaviour that forced the --no-stop generation fix), so a state machine that
    flips on any ``<action>`` substring labels those rehearsals as the action channel. Measured against a
    char-span ground truth that mislabeled ~16% of tokens overall (up to 38% on one trajectory) and
    contaminated exactly the think-vs-action contrast this kernel exists to measure. While inside a think
    block only ``</think>`` is honoured.

    Tags are matched by EARLIEST POSITION in the buffer, not by their order in the tag list -- a buffer
    holding both ``</think>`` and ``<action>`` must fire the one that actually comes first.

    The token that COMPLETES a tag is attributed to that tag's channel, so labels are approximate to
    within ~1 token at each boundary; negligible for the aggregate step/channel curves.
    """
    labels: List[Tuple[int, str]] = []
    step = 0
    for role, seg in mask_runs(response_ids, loss_mask):
        if role != "assistant":
            continue
        step += 1
        pieces = tok.batch_decode([[t] for t in seg])  # one call: per-token surface strings
        state = "other"  # other | think | action
        buf = ""
        for piece in pieces:
            buf += piece
            low = buf.lower()
            if state == "think":
                # inside reasoning: ONLY </think> exits. <action> rehearsals stay think.
                pos = low.find("</think>")
                if pos >= 0:
                    buf = buf[pos + len("</think>") :]
                    state, ch = "other", "think"
                else:
                    ch = "think"
                    buf = buf[-16:]
            else:
                cands = []
                for tag, new_state, label, only_in in _TAGS:
                    if only_in is not None and only_in != state:
                        continue  # e.g. </think> outside think, or AppWorld's ``` outside action
                    p = low.find(tag)
                    if p >= 0:
                        cands.append((p, tag, new_state, label))
                if cands:
                    p, tag, new_state, label = min(cands)  # EARLIEST tag wins
                    buf = buf[p + len(tag) :]
                    state, ch = new_state, label
                else:
                    ch = state
                    buf = buf[-16:]
            labels.append((step, ch))
    return labels


@torch.no_grad()
def per_traj_pi_metrics(
    student_lm_head,
    teacher_lm_head,
    hidden_base,
    hidden_hint,
    pb,
    ph,
    tgt,
    labels,
    ks,
    chunk,
    max_steps,
    device,
    collect_tokens=False,
    teacher_device=None,
):
    """Return (bucket_agg, tok): per-(step,channel) metrics between a STUDENT distribution
    (student_lm_head @ hidden_base[pb]) and a TEACHER distribution (teacher_lm_head @ hidden_hint[ph]) over
    the same student tokens `tgt` (pb/ph = pred indices = position−1; caller precomputes them).

    Two modes share this core:
      * PI screen — student_lm_head IS teacher_lm_head (one model); the hint only changes `hidden_hint`.
      * OPD       — student != teacher (bigger teacher); `hidden_hint` = teacher forward on the SAME no-hint
                    seq, so ph == pb (no position shift).
    tok = raw per-token lp_b/lp_h (+kl, o<k>) arrays if collect_tokens, else None.
    @torch.no_grad REQUIRED: index_add accumulation would otherwise retain every chunk's [chunk,248k] tensors
    in the autograd graph and OOM.
    """
    import torch.nn.functional as F

    if teacher_device is None:  # OPD may host the teacher on a separate GPU
        teacher_device = device
    m = min(len(pb), len(ph), len(tgt), len(labels))
    pb, ph, tgt, labels = pb[:m], ph[:m], tgt[:m], labels[:m]

    n_buckets = (max_steps + 1) * len(CHANNELS)
    bucket = torch.full((m,), -1, dtype=torch.long)
    for i, (st, ch) in enumerate(labels):
        if 1 <= st <= max_steps:
            bucket[i] = st * len(CHANNELS) + CHANNELS[ch]
    bucket = bucket.to(device)

    sum_kl = torch.zeros(n_buckets, device=device)  # reverse KL(S||T)  (OPSD reward direction)
    sum_klf = torch.zeros(n_buckets, device=device)  # forward KL(T||S)  [NEW]
    sum_d = torch.zeros(n_buckets, device=device)  # sampled-token |Δlogp|
    sum_dsigned = torch.zeros(n_buckets, device=device)  # sampled-token signed KL (k1 estimate) [NEW]
    sum_dH = torch.zeros(n_buckets, device=device)  # entropy change H(T)-H(S) [NEW]
    sum_ov = {k: torch.zeros(n_buckets, device=device) for k in ks}
    sum_tkkl = {k: torch.zeros(n_buckets, device=device) for k in ks}  # top-k(union) reverse KL [NEW]
    cnt = torch.zeros(n_buckets, device=device)
    ktop = max(ks)
    tok = (
        {
            "lp_b": [],
            "lp_h": [],
            "kl": [],
            "klf": [],
            "dsigned": [],
            "dH": [],
            **{f"o{k}": [] for k in ks},
            **{f"tkkl{k}": [] for k in ks},
        }
        if collect_tokens
        else None
    )

    for i in range(0, m, chunk):
        sl = slice(i, i + chunk)
        bck = bucket[sl]
        keep = bck >= 0
        idx_b = torch.tensor(pb[sl], device=device)
        idx_h = torch.tensor(ph[sl], device=teacher_device)
        lb = student_lm_head(hidden_base[idx_b]).float()  # [c, V]  STUDENT logits (student device)
        lh = (
            teacher_lm_head(hidden_hint[idx_h]).float().to(device)
        )  # [c, V]  TEACHER logits -> student device for the KL
        logp_b = F.log_softmax(lb, dim=-1)
        logp_h = F.log_softmax(lh, dim=-1)
        p_b = logp_b.exp()
        p_h = logp_h.exp()
        kl = (p_b * (logp_b - logp_h)).sum(-1)  # [c]  reverse KL(student||teacher)
        klf = (p_h * (logp_h - logp_b)).sum(-1)  # [c]  forward KL(teacher||student) [NEW]
        H_b = -(p_b * logp_b).sum(-1)  # [c]  student entropy
        H_h = -(p_h * logp_h).sum(-1)  # [c]  teacher entropy
        dH = H_h - H_b  # [c]  entropy change (PI sharpens<0 / blurs>0) [NEW]
        tt = torch.tensor(tgt[sl], device=device)
        ar = torch.arange(len(tt), device=device)
        lp_b_t = logp_b[ar, tt]  # [c] student logprob of the sampled token (RAW)
        lp_h_t = logp_h[ar, tt]  # [c] teacher logprob of the sampled token (RAW)
        d = (lp_b_t - lp_h_t).abs()  # [c]  |Δlogp| (SOD eq.6)
        dsigned = lp_b_t - lp_h_t  # [c]  signed sampled-token KL (k1 est. of rev-KL) [NEW]
        tkb = lb.topk(ktop, dim=-1).indices  # [c, ktop]
        tkh = lh.topk(ktop, dim=-1).indices
        match = tkb.unsqueeze(2) == tkh.unsqueeze(1)  # [c, ktop, ktop] bool
        # scatter-add into buckets (mask out positions beyond max_steps)
        bmask = torch.where(keep, bck, torch.zeros_like(bck))
        w = keep.float()
        cnt.index_add_(0, bmask, w)
        sum_kl.index_add_(0, bmask, kl * w)
        sum_klf.index_add_(0, bmask, klf * w)
        sum_d.index_add_(0, bmask, d * w)
        sum_dsigned.index_add_(0, bmask, dsigned * w)
        sum_dH.index_add_(0, bmask, dH * w)
        ov_k, tkkl_k = {}, {}
        for k in ks:
            ov = match[:, :k, :k].any(dim=2).sum(dim=1).float() / k  # [c]
            sum_ov[k].index_add_(0, bmask, ov * w)
            ov_k[k] = ov
            # top-k KL: reverse KL over the UNION of the two top-k sets, each renormalized to sum 1. [NEW]
            # Isolates divergence on the plausible candidates (drops the 248k-token tail). Teacher-side
            # tokens already in the student top-k are de-duplicated so the union isn't double-counted.
            sb, sh = tkb[:, :k], tkh[:, :k]  # [c,k]
            dup = (sh.unsqueeze(2) == sb.unsqueeze(1)).any(dim=2)  # [c,k] teacher idx already in student top-k
            idx = torch.cat([sb, sh], dim=1)  # [c,2k]
            ukeep = torch.cat([torch.ones_like(sb, dtype=torch.bool), ~dup], dim=1).float()  # [c,2k]
            pb_u = p_b.gather(1, idx) * ukeep
            ph_u = p_h.gather(1, idx) * ukeep
            pb_n = pb_u / pb_u.sum(1, keepdim=True).clamp_min(1e-12)
            ph_n = ph_u / ph_u.sum(1, keepdim=True).clamp_min(1e-12)
            tkkl = (pb_n * (pb_n.clamp_min(1e-12).log() - ph_n.clamp_min(1e-12).log())).sum(1)  # [c]
            sum_tkkl[k].index_add_(0, bmask, tkkl * w)
            tkkl_k[k] = tkkl
        if tok is not None:  # raw per-token arrays (ALL student tokens, incl. steps > max_steps)
            tok["lp_b"] += lp_b_t.cpu().tolist()
            tok["lp_h"] += lp_h_t.cpu().tolist()
            tok["kl"] += kl.cpu().tolist()
            tok["klf"] += klf.cpu().tolist()
            tok["dsigned"] += dsigned.cpu().tolist()
            tok["dH"] += dH.cpu().tolist()
            for k in ks:
                tok[f"o{k}"] += ov_k[k].cpu().tolist()
                tok[f"tkkl{k}"] += tkkl_k[k].cpu().tolist()

    out: Dict[Tuple[int, str], Dict[str, float]] = {}
    inv_ch = {v: kk for kk, v in CHANNELS.items()}
    cnt_c = cnt.cpu()
    for b in range(n_buckets):
        c = float(cnt_c[b])
        if c <= 0:
            continue
        st, ch = b // len(CHANNELS), inv_ch[b % len(CHANNELS)]
        rec = {
            "n": c,
            "kl": float(sum_kl[b]) / c,
            "klf": float(sum_klf[b]) / c,
            "d": float(sum_d[b]) / c,
            "dsigned": float(sum_dsigned[b]) / c,
            "dH": float(sum_dH[b]) / c,
        }
        for k in ks:
            rec[f"o{k}"] = float(sum_ov[k][b]) / c
            rec[f"tkkl{k}"] = float(sum_tkkl[k][b]) / c
        out[(st, ch)] = rec
    return out, tok


def parse_pis(spec: str) -> List[Tuple[str, str]]:
    """`name=path,name=path` -> [(name, path)]."""
    pairs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, path = item.split("=", 1)
        pairs.append((name.strip(), path.strip()))
    return pairs


def do_merge(out_dir: Path, ks: List[int]) -> None:
    import pandas as pd

    rows, torn = [], 0
    for sf in sorted(glob.glob(str(out_dir / "_shards" / "positions.shard*.jsonl"))):
        with open(sf) as f:
            for line in f:
                if not line.strip():
                    continue
                # TOLERATE A TORN LAST LINE. A process killed mid-write leaves a partial JSON object;
                # `json.loads` on it aborted the whole merge, so a preempted-then-resumed cell could
                # never be merged at all. Only a truncated tail can fail to parse here, and the
                # rollout it belonged to has no done-marker, so --resume re-scores it in full.
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    torn += 1
    if torn:
        logger.warning("merge: skipped %d unparseable (torn) line(s) from an interrupted write", torn)
    if not rows:
        logger.warning("no shard rows under %s/_shards", out_dir)
        return
    df = pd.DataFrame(rows)
    # DE-DUPLICATE. With --resume, a rollout interrupted mid-way has its partial rows appended once
    # and then written again on the retry. Legitimate rows are unique per
    # (pi,row_idx,repetition_id,step,channel) -- one row per (step,channel) for each (rollout,PI) --
    # so dropping duplicates on that key removes exactly the retry overlap and nothing else.
    # keep="last": the retry's copy is the one written by a run that reached the done marker.
    key = [c for c in ("pi", "row_idx", "repetition_id", "step", "channel") if c in df.columns]
    if key:
        n0 = len(df)
        df = df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
        if len(df) != n0:
            logger.info("merge: dropped %d duplicate rows (resume overlap) on %s", n0 - len(df), key)
    df.to_parquet(out_dir / "rows.parquet")
    logger.info("merged %d rows from %d PIs -> %s/rows.parquet", len(df), df["pi"].nunique(), out_dir)
    # raw per-token dumps (if --save-token-logprobs was used)
    tps = sorted(glob.glob(str(out_dir / "_shards" / "tokens.shard*.parquet")))
    if tps:
        tdf = pd.concat([pd.read_parquet(p) for p in tps], ignore_index=True)
        tdf.to_parquet(out_dir / "tokens.parquet")
        logger.info("merged %d per-(rollout,PI) raw-token records -> %s/tokens.parquet", len(tdf), out_dir)


def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    # Benchmark selection. ALFWorld keeps the module-level defaults untouched, so its output is
    # bit-identical to every run made before --benchmark existed; anything else swaps the channel tag set
    # and the success rule together (they must agree -- a benchmark whose actions are not <action> also
    # does not score success on ALFWorld's 10.0*won scale).
    global _TAGS, is_success
    bench = getattr(args, "benchmark", "alfworld")
    if bench != "alfworld":
        from opsd.pi.adapters import get_adapter

        _TAGS = _TAGS_BY_BENCH[bench]
        is_success = get_adapter(bench).is_success
        logger.info(
            "benchmark=%s: channel tags=%s, success rule=%s",
            bench,
            [t[0] for t in _TAGS],
            get_adapter(bench).__class__.__name__,
        )

    out_dir = Path(args.out_dir).expanduser()
    (out_dir / "_shards").mkdir(parents=True, exist_ok=True)
    ks = [int(x) for x in str(args.ks).split(",")]

    if args.merge:
        do_merge(out_dir, ks)
        return

    if not args.model_path:
        raise SystemExit("--model-path/--served-model-name is required for scoring (only --merge omits it).")
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model_path)
    im_end_id = tok.convert_tokens_to_ids(args.im_end_token)
    if im_end_id is None or im_end_id == tok.unk_token_id:
        im_end_id = tok.eos_token_id
    # Right-padding id for the batched forward; masked out by attention_mask, so any valid id works.
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    device = args.device
    teacher_device = args.teacher_device or device  # OPD: teacher may live on a separate GPU
    scorer = HFLogprobScorer(args.model_path, device=device, lm_head_chunk=args.lm_head_chunk)
    teacher = (
        HFLogprobScorer(args.teacher_model_path, device=teacher_device, lm_head_chunk=args.lm_head_chunk)
        if args.teacher_model_path
        else None
    )
    opd_mode = teacher is not None
    opd_label = args.teacher_label or (Path(args.teacher_model_path).name if args.teacher_model_path else "")

    pis = [] if opd_mode else parse_pis(args.pis)
    hints_by_pi = {} if opd_mode else {name: load_task_hints(path) for name, path in pis}
    if opd_mode:
        logger.info(
            "shard %d/%d | OPD mode: student=%s vs teacher=%s (NO hint) label=%s",
            args.shard_index,
            args.num_shards,
            args.model_path,
            args.teacher_model_path,
            opd_label,
        )
    else:
        logger.info(
            "shard %d/%d | PI mode: %d PIs: %s",
            args.shard_index,
            args.num_shards,
            len(pis),
            ", ".join(n for n, _ in pis),
        )

    # THREE INPUT SCHEMAS, and note where each one shards -- getting that wrong is silent.
    # The two ADAPTER paths shard INSIDE the loader (by global line index) so that only this shard's
    # records are tokenized; the native path loads everything and post-filters. Applying the
    # post-filter on top of an adapter would shard twice and quietly drop ~(1 - 1/N) of the shard.
    if args.appworld_transcripts:
        # AppWorld's driver has no --save-token-ids, so the token schema is reconstructed from the
        # message list here. The adapter derives the prompt/response boundary from len(turns) so the
        # official scaffold's few-shot demo stays OUT of the loss mask -- see adapters/appworld.py.
        from opsd.pi.adapters.appworld import load_appworld_dump

        recs = load_appworld_dump(
            args.appworld_transcripts, tok, shard_index=args.shard_index, num_shards=args.num_shards
        )
    elif args.input_format == "skyrl_eval":
        # SkyRL eval dumps (ALFWorld `exports/dumped_evals/**/alfworld.jsonl`) carry
        # input_prompt / output_response / score / env_extras -- NOT prompt_token_ids / response_ids /
        # loss_mask. Without this branch every record hits n_skip["no_tokens"] and the shard dies at
        # the empty-shard guard. adapters/skyrl_eval_dump.py rebuilds the token schema, keys task_id off
        # env_extras.extra_info.task_id, and derives success from `10.0 in score`.
        from opsd.pi.adapters.skyrl_eval_dump import load_skyrl_eval_dump

        recs = load_skyrl_eval_dump(args.trajectories, tok, shard_index=args.shard_index, num_shards=args.num_shards)
    elif args.input_format == "tau2":
        # τ² pass@k dumps (`traj_*.jsonl`) carry messages / prompt_messages / response_messages, so like
        # the skyrl_eval branch they need the token schema rebuilt or every record hits n_skip["no_tokens"].
        # NOT interchangeable with a τ1 (tau-bench v1) adapter: τ² adds a `tool` role that τ1's mask does not
        # zero (69.7% of response text) and puts 61.7% of assistant actions in `tool_calls` rather than
        # `content` (42.8% of policy tokens). Measured: τ1 on a τ² dump masks 92.4% as policy vs 19.6%.
        from opsd.pi.adapters.tau2 import load_tau2_dump

        recs = load_tau2_dump(args.trajectories, tok, shard_index=args.shard_index, num_shards=args.num_shards)
    else:
        recs = load_trajectories(args.trajectories)
        recs = [r for i, r in enumerate(recs) if i % args.num_shards == args.shard_index]
    if args.limit:
        recs = recs[: args.limit]
    logger.info("shard has %d trajectories", len(recs))

    shard_path = out_dir / "_shards" / f"positions.shard{args.shard_index}.jsonl"
    done_path = out_dir / "_shards" / f"done.shard{args.shard_index}.jsonl"
    n_done, n_skip = 0, defaultdict(int)
    token_records: List[Dict[str, Any]] = []

    # --- resume ---------------------------------------------------------------------------------
    # Rollouts are keyed by (row_idx, repetition_id) and marked complete in done.shard{i}.jsonl only
    # AFTER their rows are flushed. A rollout killed mid-way therefore has rows but no marker, so it
    # is re-scored and its earlier partial rows are appended a second time -- do_merge de-duplicates
    # on (pi,row_idx,repetition_id,step,channel), keeping the complete copy.
    done_keys, part_next = set(), 0
    if args.resume:
        if done_path.exists():
            with done_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)  # a torn final line is discarded here
                    except json.JSONDecodeError:
                        continue
                    done_keys.add((r.get("row_idx"), r.get("repetition_id")))
            logger.info(
                "shard %d resume: %d rollouts already complete per %s", args.shard_index, len(done_keys), done_path
            )
        parts = glob.glob(str(out_dir / "_shards" / f"tokens.shard{args.shard_index}.part*.parquet"))
        for q in parts:
            m = re.search(r"\.part(\d+)\.parquet$", q)
            if m:
                part_next = max(part_next, int(m.group(1)) + 1)
        if parts:
            logger.info(
                "shard %d resume: %d existing token parts, next index %d", args.shard_index, len(parts), part_next
            )

    def _flush_tokens():
        nonlocal token_records, part_next
        if not token_records:
            return
        import pandas as pd

        tp = out_dir / "_shards" / f"tokens.shard{args.shard_index}.part{part_next}.parquet"
        pd.DataFrame(token_records).to_parquet(tp)
        logger.info("shard %d: flushed %d token records -> %s", args.shard_index, len(token_records), tp)
        token_records = []
        part_next += 1

    mode = "a" if args.resume else "w"
    fdone = done_path.open(mode)
    with shard_path.open(mode) as fout:
        for rec in recs:
            if done_keys and (rec.get("row_idx"), rec.get("repetition_id")) in done_keys:
                n_skip["resume_already_done"] += 1
                continue
            task_id = rec.get("task_id") or (rec.get("extra_info") or {}).get("task_id")
            prompt_ids = rec.get("prompt_token_ids")
            response_ids = rec.get("response_ids")
            loss_mask = rec.get("loss_mask")
            if not prompt_ids or not response_ids or not loss_mask:
                n_skip["no_tokens"] += 1
                continue
            if not any(loss_mask):  # template lacked {% generation %} -> rebuild from markers
                loss_mask = reconstruct_assistant_mask(response_ids, tok)
                n_skip["mask_reconstructed"] += 1
            student_idx = [j for j, mm in enumerate(loss_mask) if mm]
            if not student_idx:
                n_skip["no_assistant"] += 1
                continue
            last = student_idx[-1]
            resp = list(response_ids[: last + 1])
            lm = list(loss_mask[: last + 1])
            base_seq = list(prompt_ids) + resp
            base_pos = assistant_positions(len(prompt_ids), lm)
            labels = label_student_tokens(resp, lm, tok)
            pb0 = [p - 1 for p in base_pos if p >= 1]
            tgt = [base_seq[p] for p in base_pos if p >= 1]

            # Guard the BASE forward. It used to run unconditionally, with the length checks only inside the
            # per-condition loop below -- so in PI mode the no-hint sequence was never bounded at all, and a
            # long trajectory OOMed here before any skip could fire. Harmless to the population: the hint
            # only ADDS tokens, so len(base_seq) <= len(hinted_prompt) + len(resp) for every PI, and any
            # record caught here would have been skipped by every condition anyway.
            if args.max_seq_len and len(base_seq) > args.max_seq_len:
                n_skip["too_long_base"] += 1
                continue
            hidden_base = None  # filled by the batched pass below, or lazily at batch-1
            succ = is_success(rec)

            # OPD: one condition (bigger teacher, NO hint, same seq). PI: one per hinted family.
            conditions = [(opd_label, None)] if opd_mode else [(nm, hints_by_pi[nm].get(task_id)) for nm, _ in pis]

            # ---- BATCHED HINT FORWARDS (PI mode only) -------------------------------------------------
            # The 6 hint sequences share `resp` and differ only in the hint prefix, so they batch almost
            # perfectly. At batch 1 a short trajectory (~1.6k tokens) leaves an H100 largely idle. Prepare
            # every condition FIRST so the per-condition skips below still fire identically, then run one
            # right-padded forward per token-budget chunk. Padding leaves RoPE and the causal mask
            # unchanged for real tokens; PI_ASSERT_BATCH_PARITY=1 re-checks that against the batch-1 path.
            hs_by_name = {}
            if not opd_mode and args.batch_tokens > 0:
                prepared = []
                for _nm, _ht in conditions:
                    if not _ht:
                        continue
                    _hp = build_hint_prompt(tok, prompt_ids, _ht, args.inject_style, im_end_id)
                    if _hp is None or (args.max_seq_len and len(_hp) + len(resp) > args.max_seq_len):
                        continue
                    prepared.append((_nm, _hp + resp))
                prepared.insert(0, ("__base__", base_seq))  # same batch: base differs only by the hint prefix
                _i = 0
                while _i < len(prepared):
                    _chunk, _mx = [], 0
                    while _i < len(prepared):
                        _L = len(prepared[_i][1])
                        # padded cost is max_len * batch; always take >=1 or an over-budget seq spins
                        if _chunk and max(_mx, _L) * (len(_chunk) + 1) > args.batch_tokens:
                            break
                        _chunk.append(prepared[_i])
                        _mx = max(_mx, _L)
                        _i += 1
                    for (_nm2, _), _h in zip(_chunk, scorer._hidden_states_batch([s for _, s in _chunk], pad_id)):
                        hs_by_name[_nm2] = _h
                hidden_base = hs_by_name.pop("__base__", None)

            if hidden_base is None:
                hidden_base = scorer._hidden_states(torch.tensor([base_seq], device=device))
            for name, hint_text in conditions:
                if opd_mode:
                    if args.max_seq_len and len(base_seq) > args.max_seq_len:
                        n_skip["too_long"] += 1
                        continue
                    hidden_hint = teacher._hidden_states(torch.tensor([base_seq], device=teacher_device))
                    pb, ph, s_head, t_head = pb0, pb0, scorer.lm_head, teacher.lm_head
                else:
                    if not hint_text:
                        n_skip[f"no_hint::{name}"] += 1
                        continue
                    hinted_prompt = build_hint_prompt(tok, prompt_ids, hint_text, args.inject_style, im_end_id)
                    if hinted_prompt is None:
                        n_skip["no_im_end"] += 1
                        continue
                    if args.max_seq_len and len(hinted_prompt) + len(resp) > args.max_seq_len:
                        n_skip["too_long"] += 1
                        continue
                    hint_pos = assistant_positions(len(hinted_prompt), lm)
                    hidden_hint = hs_by_name.get(name)
                    if hidden_hint is None:  # batching off, or this seq missed the prepare pass
                        hidden_hint = scorer._hidden_states(torch.tensor([hinted_prompt + resp], device=device))
                    elif os.environ.get("PI_ASSERT_BATCH_PARITY") == "1":
                        # Re-run unbatched and compare. bf16 matmuls reassociate under batching, so this is
                        # a closeness check, not bitwise -- but a padding/mask bug shows up as a LARGE diff.
                        _ref = scorer._hidden_states(torch.tensor([hinted_prompt + resp], device=device))
                        _d = (hidden_hint.float() - _ref.float()).abs().max().item()
                        if _d > 0.05:
                            raise SystemExit(
                                f"BATCH PARITY FAILED for {name}: max|diff|={_d:.4f} "
                                f"(len={len(hinted_prompt) + len(resp)}) -- padding/mask bug"
                            )
                        n_skip["_parity_checked"] += 1
                        del _ref
                    pb, ph = pb0, [p - 1 for p in hint_pos if p >= 1]
                    s_head, t_head = scorer.lm_head, scorer.lm_head
                agg, tokd = per_traj_pi_metrics(
                    s_head,
                    t_head,
                    hidden_base,
                    hidden_hint,
                    pb,
                    ph,
                    tgt,
                    labels,
                    ks,
                    args.lm_head_chunk,
                    args.max_steps,
                    device,
                    collect_tokens=args.save_token_logprobs,
                    teacher_device=(teacher_device if opd_mode else device),
                )
                del hidden_hint
                if tokd is not None:
                    mt = len(tokd["lp_b"])
                    token_records.append(
                        {
                            "pi": name,
                            "row_idx": rec.get("row_idx"),
                            "repetition_id": rec.get("repetition_id"),
                            "task_id": task_id,
                            "success": succ,
                            "step": [s for s, _ in labels[:mt]],
                            "channel": [CHANNELS[c] for _, c in labels[:mt]],
                            "lp_b": tokd["lp_b"],
                            "lp_h": tokd["lp_h"],
                            "kl": tokd["kl"],
                            "klf": tokd["klf"],
                            "dsigned": tokd["dsigned"],
                            "dH": tokd["dH"],
                            **{f"o{k}": tokd[f"o{k}"] for k in ks},
                            **{f"tkkl{k}": tokd[f"tkkl{k}"] for k in ks},
                        }
                    )
                for (st, ch), rec_m in agg.items():
                    # row_idx/repetition_id = PER-ROLLOUT id (uid is per-TASK: 4 rollouts share it).
                    row = {
                        "pi": name,
                        "task_id": task_id,
                        "uid": rec.get("uid"),
                        "row_idx": rec.get("row_idx"),
                        "repetition_id": rec.get("repetition_id"),
                        "success": succ,
                        "step": st,
                        "channel": ch,
                        "n_tokens": rec_m["n"],
                        "kl": rec_m["kl"],
                        "klf": rec_m["klf"],
                        "d": rec_m["d"],
                        "dsigned": rec_m["dsigned"],
                        "dH": rec_m["dH"],
                    }
                    for k in ks:
                        row[f"o{k}"] = rec_m[f"o{k}"]
                        row[f"tkkl{k}"] = rec_m[f"tkkl{k}"]
                    fout.write(json.dumps(row) + "\n")
            hs_by_name.clear()
            del hidden_base
            n_done += 1
            # ORDER MATTERS: flush the rows, THEN mark done. The reverse would let a process die between
            # marker and flush, and resume would skip a rollout whose rows were never written.
            fout.flush()
            fdone.write(json.dumps({"row_idx": rec.get("row_idx"), "repetition_id": rec.get("repetition_id")}) + "\n")
            fdone.flush()
            if n_done % 10 == 0:
                logger.info("  shard %d: %d/%d trajectories", args.shard_index, n_done, len(recs))
            if args.token_flush_every and n_done % args.token_flush_every == 0:
                _flush_tokens()
    _flush_tokens()
    fdone.close()
    logger.info("shard %d DONE: %d trajectories, skips=%s -> %s", args.shard_index, n_done, dict(n_skip), shard_path)
    # Fail LOUD on an empty shard. Skips are per-trajectory `continue`s, so a systematically broken
    # input (e.g. an all-zero loss_mask across the whole pool, or a dump whose schema the loader does
    # not recognise) otherwise exits 0 with a valid-looking but empty parquet -- which reads
    # downstream as "scored, found nothing".
    # A --resume run that finds everything already done scores 0 legitimately, so the guard must
    # count previously-completed rollouts as progress -- otherwise the final, correct retry of a
    # preempted job exits non-zero and the driver script refuses to merge a COMPLETE result.
    if n_done == 0 and recs and not n_skip.get("resume_already_done"):
        raise SystemExit(
            f"shard {args.shard_index}: scored 0/{len(recs)} trajectories (skips={dict(n_skip)}). "
            "Refusing to write an empty shard -- fix the input rather than treating this as success."
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectories", default=None, help="base (no-hint) trajectories.jsonl with --save-token-ids.")
    ap.add_argument(
        "--appworld-transcripts",
        default=None,
        help="AppWorld transcripts.jsonl instead of --trajectories: the token schema is "
        "reconstructed via adapters.appworld, because the AppWorld driver has no "
        "--save-token-ids. Shards internally by global line index.",
    )
    ap.add_argument(
        "--input-format",
        choices=("jsonl", "skyrl_eval", "tau2"),
        default="jsonl",
        help="schema of --trajectories. 'jsonl' (default) = native dump that already has "
        "prompt_token_ids/response_ids/loss_mask. 'skyrl_eval' = a SkyRL eval dump "
        "(input_prompt/output_response/score/env_extras), e.g. ALFWorld's "
        "exports/dumped_evals/**/alfworld.jsonl -- rebuilt via adapters.skyrl_eval_dump. "
        "'tau2' = a τ²-bench pass@k dump (messages/prompt_messages/response_messages) "
        "-- rebuilt via adapters.tau2 (a τ1 adapter would mismask it).",
    )
    ap.add_argument("--pis", help="comma list of name=hints_path (each a task-level artifact).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--model-path",
        "--served-model-name",
        dest="model_path",
        default=None,
        help="HF model path/name that produced the traces (required for scoring; not needed for --merge).",
    )
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lm-head-chunk", type=int, default=128, help="student positions per chunk (2x full-vocab here).")
    # DEFAULT 0 = OFF. Batching the hint forwards was tried and MEASURED WORSE ON BOTH AXES
    # (Qwen3-4B-Instruct-2507, H100, 2026-08-24):
    #   * NUMERICS: right-padding changes the REAL tokens\' hidden states. Zero pad tokens is bitwise
    #     exact; ONE pad token gives max|d|=33.4 / mean|d|=0.063 on |h|~200, and the error is IDENTICAL
    #     for 1, 8 and 200 pads -- so it is not pad leaking through causal attention (that would scale
    #     with pad count), it is SDPA switching kernel as soon as the mask contains a zero. A 0.06 mean
    #     shift in hidden states is not acceptable for a screen whose effects are ~0.03 AUROC.
    #   * SPEED: 0.82x. Seven ~1.5k-token sequences already saturate the GEMM, so padding waste costs
    #     more than the extra parallelism buys. The premise that batch-1 idles the GPU was WRONG at
    #     these lengths.
    # Kept behind a flag (not deleted) because it IS exact for equal-length batches -- if you ever group
    # by exact length, or move to an attention impl that handles masks exactly, it becomes usable.
    ap.add_argument("--batch-tokens", type=int, default=int(os.environ.get("PI_BATCH_TOKENS", "0")))
    ap.add_argument("--im-end-token", default="<|im_end|>")
    ap.add_argument("--inject-style", choices=("splice", "system-prepend"), default="splice")
    ap.add_argument(
        "--benchmark",
        choices=tuple(sorted(_TAGS_BY_BENCH)),
        default="alfworld",
        help="selects the channel tag set and the success rule. Default keeps ALFWorld exact.",
    )
    ap.add_argument(
        "--teacher-model-path",
        default=None,
        help="OPD mode: a (bigger) teacher scored on the SAME no-hint seq (no PI). Student stays "
        "--model-path; must share the student's tokenizer/vocab (Qwen3.5 2B/9B/27B do).",
    )
    ap.add_argument("--teacher-label", default=None, help="pi-column label for the OPD teacher (e.g. opd_9b).")
    ap.add_argument(
        "--teacher-device",
        default=None,
        help="OPD: host the teacher on a SEPARATE device (e.g. cuda:1) so a large teacher + the student "
        "fit across two GPUs; teacher logits are moved to the student device for the KL. "
        "Default = --device (same GPU).",
    )
    ap.add_argument("--ks", default="10,20,50,100", help="comma list of k for top-k overlap.")
    ap.add_argument("--max-steps", type=int, default=50, help="cap analyzed agentic step index.")
    ap.add_argument("--max-seq-len", type=int, default=40000, help="skip a trajectory whose hint seq exceeds this.")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="APPEND to an existing positions.shard{i}.jsonl, skipping rollouts already "
        "recorded in done.shard{i}.jsonl. Completion is tracked by an explicit "
        "marker rather than by counting PIs per rollout, because a missing hint or "
        "an over-length sequence legitimately drops a condition -- so 'has rows "
        "for every PI' would mark those rollouts unfinished forever.",
    )
    ap.add_argument(
        "--token-flush-every",
        type=int,
        default=25,
        help="flush token records to tokens.shard{i}.part{k}.parquet every N rollouts. "
        "do_merge globs tokens.shard*.parquet, which matches the parts unchanged. "
        "0 = one write at the end, which a preempted process loses entirely.",
    )
    ap.add_argument(
        "--save-token-logprobs",
        action="store_true",
        help="also dump per-student-token raw lp_b/lp_h (+kl,o<k>) arrays per (rollout,PI) -> "
        "tokens.parquet, so any later re-aggregation needs no GPU rerun.",
    )
    ap.add_argument(
        "--merge", action="store_true", help="merge shard files into rows.parquet (+tokens.parquet) and exit."
    )
    args = ap.parse_args()
    # --trajectories stopped being `required` when --appworld-transcripts was added; enforce the XOR here
    # so a missing input is an immediate error rather than an empty shard that merges to a silent zero.
    if not args.merge and bool(args.trajectories) == bool(args.appworld_transcripts):
        ap.error("provide exactly one of --trajectories / --appworld-transcripts")
    # --input-format describes the schema OF --trajectories. Pairing it with --appworld-transcripts is
    # always a mistake (that flag already implies its own adapter), and silently ignoring it would let
    # `--appworld-transcripts X --input-format skyrl_eval` run the AppWorld loader while the caller believes
    # the SkyRL eval-dump one is active.
    if args.appworld_transcripts and args.input_format != "jsonl":
        ap.error("--input-format applies to --trajectories; it cannot be combined with --appworld-transcripts")
    return args


if __name__ == "__main__":
    run(parse_args())

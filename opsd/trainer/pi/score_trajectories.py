"""Offline PI ranking / selection-signal screen (Use 1).

Score each logged student trajectory with a PI-derived signal and (a) report how well it
discriminates correct-vs-incorrect against the ALFWorld verifier (AUC / Spearman — the Q1a screen)
and (b) rank trajectories per task to select/curate the top-k.

Signal (no gradient, two forward passes per trajectory):
    lp_base = sum_t  log pi(resp_t | prompt,           resp_<t)
    lp_hint = sum_t  log pi(resp_t | prompt + HINT,    resp_<t)        over the STUDENT (loss_mask=1) tokens
    Delta-lp = lp_hint - lp_base    (the offline twin of the OPSD reverse-KL reward)

**Scores ONLY the student tokens.** A local HF forward produces hidden states for the full sequence
(needed for conditioning) but the 248k-vocab LM head is applied at ONLY the ~student positions (loss
mask = 1), not all positions. In these ALFWorld rollouts the student is only ~32% of the sequence (the
prompt + interleaved observations are ~68%), so this both fixes the prompt_logprobs OOM (vLLM has no
per-position API → it materialized logits over the whole long sequence × 248k vocab at every position)
and is ~3x cheaper. No vLLM server needed for scoring (it's only for generation).

The with-hint sequence is built by **token-splicing** the hint before the last ``<|im_end|>`` of the
dumped ``prompt_token_ids`` (drift-free); alignment is by the student subsequence, robust to the shift.

Example:
  python -m opsd.trainer.pi.score_trajectories \
    --trajectories ~/alfworld_gen_base/trajectories.jsonl \
    --hints ~/data/alfworld_pi/hints_onpolicy_skill.json \
    --model-path Qwen/Qwen3.5-2B --device cuda:0 \
    --out-dir ~/alfworld_pi_score/onpolicy_skill --ablation
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from opsd.pi.artifact import load_task_hints

logger = logging.getLogger("pi.score")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --------------------------------------------------------------------------------------------------
# Re-exports (Phase 1): sequence-construction / injection / scoring / metric helpers now live in
# opsd.pi.signals. They stay importable at the old path
# (opsd.trainer.pi.score_trajectories) for backward compatibility; the offline-ranking
# driver (load_trajectories / is_success / parse_args / run / main) remains below.
# --------------------------------------------------------------------------------------------------
from opsd.pi.signals.inject import (  # noqa: E402,F401
    HINT_TEMPLATE,
    SECRET_TEMPLATE,
    build_hint_prompt,
    build_system_block_ids,
)
from opsd.pi.signals.logprob import HFLogprobScorer  # noqa: E402,F401
from opsd.pi.signals.metrics import roc_auc, spearman  # noqa: E402,F401
from opsd.pi.signals.positions import (  # noqa: E402,F401
    assistant_positions,
    first_token_index,
    last_token_index,
    splice_hint,
)


# --------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------
def load_trajectories(path: str) -> List[Dict[str, Any]]:
    recs = []
    with Path(path).expanduser().open() as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def is_success(rec: Dict[str, Any]) -> int:
    em = rec.get("env_metrics") or {}
    return int(bool(em.get("success", 0.0)) or float(rec.get("reward", 0.0) or 0.0) >= 10.0)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectories", required=True, help="base trajectories.jsonl (generated with --save-token-ids).")
    ap.add_argument("--hints", required=True, help="hints artifact (task-level).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--model-path",
        "--served-model-name",
        dest="model_path",
        required=True,
        help="HF model path/name (the SAME model that produced the trajectories).",
    )
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer (defaults to --model-path).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lm-head-chunk", type=int, default=512, help="student positions per LM-head chunk.")
    ap.add_argument("--im-end-token", default="<|im_end|>", help="user-turn-end special token to splice before.")
    ap.add_argument(
        "--inject-style",
        choices=("splice", "system-prepend"),
        default="splice",
        help="splice: append hint inside the user turn; system-prepend: prepend a <Secret information> system block.",
    )
    ap.add_argument(
        "--input-format",
        choices=("jsonl", "skyrl_eval"),
        default="jsonl",
        help="jsonl: native scorer dump; skyrl_eval: a SkyRL eval dump "
        "(<export_path>/dumped_evals/**/*.jsonl, adapted on the fly).",
    )
    ap.add_argument("--pi-format", default=None, help="label recorded in the report (e.g. raw/raw_no_think/actions).")
    ap.add_argument(
        "--max-seq-len", type=int, default=30000, help="skip a trajectory whose prompt+response exceeds this."
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ablation", action="store_true", help="also score with a mismatched (other-task) hint.")
    ap.add_argument("--top-k", type=int, default=1, help="trajectories to keep per task in selected.jsonl.")
    # accepted for backward-compat with the old vLLM-backed CLI (ignored):
    ap.add_argument("--model-url", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--max-concurrency", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--api-key", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--request-timeout", type=int, default=None, help=argparse.SUPPRESS)
    return ap.parse_args()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model_path)
    im_end_id = tok.convert_tokens_to_ids(args.im_end_token)
    if im_end_id is None or im_end_id == tok.unk_token_id:
        im_end_id = tok.eos_token_id
    logger.info("user-turn-end token %r -> id %s", args.im_end_token, im_end_id)

    scorer = HFLogprobScorer(args.model_path, device=args.device, lm_head_chunk=args.lm_head_chunk)

    hints = load_task_hints(args.hints)
    if args.input_format == "skyrl_eval":
        from opsd.pi.adapters.skyrl_eval_dump import load_skyrl_eval_dump

        recs = load_skyrl_eval_dump(args.trajectories, tok)
    else:
        recs = load_trajectories(args.trajectories)
    if args.limit:
        recs = recs[: args.limit]
    logger.info(
        "inject_style=%s | input_format=%s | pi_format=%s | %d records | %d hints",
        args.inject_style,
        args.input_format,
        args.pi_format,
        len(recs),
        len(hints),
    )

    sorted_tasks = sorted(hints)
    mismatch_of = (
        {t: sorted_tasks[(i + 1) % len(sorted_tasks)] for i, t in enumerate(sorted_tasks)} if sorted_tasks else {}
    )

    scored: List[Dict[str, Any]] = []
    n_skip = {"no_tokens": 0, "no_hint": 0, "no_im_end": 0, "no_assistant": 0, "too_long": 0}

    for rec in recs:
        task_id = rec.get("task_id") or (rec.get("extra_info") or {}).get("task_id")
        prompt_ids = rec.get("prompt_token_ids")
        response_ids = rec.get("response_ids")
        loss_mask = rec.get("loss_mask")
        if not prompt_ids or not response_ids or not loss_mask:
            n_skip["no_tokens"] += 1
            continue
        hint_text = hints.get(task_id)
        if not hint_text:
            n_skip["no_hint"] += 1
            continue
        hinted_prompt = build_hint_prompt(tok, prompt_ids, hint_text, args.inject_style, im_end_id)
        if hinted_prompt is None:  # splice found no <|im_end|>
            n_skip["no_im_end"] += 1
            continue
        # Score ONLY the student tokens: drop everything after the last student token (trailing obs/eos
        # are never scored and don't condition any student token), then sum LM-head logprobs at the
        # student positions only.
        student_idx = [j for j, m in enumerate(loss_mask) if m]
        if not student_idx:
            n_skip["no_assistant"] += 1
            continue
        last = student_idx[-1]
        resp = list(response_ids[: last + 1])
        lm = list(loss_mask[: last + 1])

        # Skip trajectories too long for a single HF forward (the with-hint seq is the longer one).
        if args.max_seq_len and len(hinted_prompt) + len(resp) > args.max_seq_len:
            n_skip["too_long"] += 1
            continue

        base_seq = list(prompt_ids) + resp
        hint_seq = hinted_prompt + resp
        base_pos = assistant_positions(len(prompt_ids), lm)
        hint_pos = assistant_positions(len(hinted_prompt), lm)

        lp_base, n_b = scorer.sum_logprob_at(base_seq, base_pos)
        lp_hint, n_h = scorer.sum_logprob_at(hint_seq, hint_pos)
        n = min(n_b, n_h) or 1
        row = {
            "uid": rec.get("uid"),
            "task_id": task_id,
            "repetition_id": rec.get("repetition_id", 0),
            "success": is_success(rec),
            "reward": float(rec.get("reward", 0.0) or 0.0),
            "n_assistant": n,
            "lp_base": lp_base,
            "lp_hint": lp_hint,
            "dlp_sum": lp_hint - lp_base,
            "dlp_mean": (lp_hint - lp_base) / n,
        }
        if args.ablation and mismatch_of:
            mtext = hints.get(mismatch_of.get(task_id))
            if mtext:
                mis_prompt = build_hint_prompt(tok, prompt_ids, mtext, args.inject_style, im_end_id)
                if mis_prompt is not None:
                    lp_mis, _ = scorer.sum_logprob_at(mis_prompt + resp, assistant_positions(len(mis_prompt), lm))
                    row["dlp_sum_mismatch"] = lp_mis - lp_base
                    row["dlp_mean_mismatch"] = (lp_mis - lp_base) / n
        scored.append(row)

    with (out_dir / "scores.jsonl").open("w") as f:
        for row in scored:
            f.write(json.dumps(row) + "\n")

    labels = [r["success"] for r in scored]
    report: Dict[str, Any] = {
        "hints": str(Path(args.hints).expanduser()),
        "model_path": args.model_path,
        "inject_style": args.inject_style,
        "input_format": args.input_format,
        "pi_format": args.pi_format,
        "n_scored": len(scored),
        "n_success": sum(labels),
        "n_fail": len(labels) - sum(labels),
        "skipped": n_skip,
        "signals": {},
    }
    signal_keys = ["dlp_sum", "dlp_mean", "lp_hint"]
    if args.ablation:
        signal_keys += [k for k in ("dlp_sum_mismatch", "dlp_mean_mismatch") if any(k in r for r in scored)]
    for key in signal_keys:
        rows = [r for r in scored if key in r]
        s = [r[key] for r in rows]
        y = [r["success"] for r in rows]
        report["signals"][key] = {
            "n": len(rows),
            "auc": roc_auc(s, y),
            "spearman": spearman(s, y),
            "mean_pos": (sum(v for v, yy in zip(s, y) if yy == 1) / max(1, sum(y))),
            "mean_neg": (sum(v for v, yy in zip(s, y) if yy == 0) / max(1, len(y) - sum(y))),
        }

    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in scored:
        by_task[r["task_id"]].append(r)
    selected: List[Dict[str, Any]] = []
    for tid, rows in by_task.items():
        rows.sort(key=lambda r: r["dlp_sum"], reverse=True)
        selected.extend(rows[: max(1, args.top_k)])
    with (out_dir / "selected.jsonl").open("w") as f:
        for row in selected:
            f.write(json.dumps(row) + "\n")
    report["selected_per_task"] = args.top_k
    report["n_selected"] = len(selected)
    report["selected_success_rate"] = (sum(r["success"] for r in selected) / len(selected)) if selected else None
    report["overall_success_rate"] = (sum(labels) / len(labels)) if labels else None

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    logger.info("DONE -> %s\n%s", out_dir, json.dumps(report, indent=2))
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

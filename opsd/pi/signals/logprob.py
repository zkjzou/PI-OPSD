"""Local HF log-prob scorer for PI signals (Phase 1 extraction from ``opsd/trainer/pi/score_trajectories.py``).

``HFLogprobScorer`` runs a full-sequence backbone forward (for correct conditioning) but applies the
large vocab LM head ONLY at the student (loss_mask=1) positions — fixing the vLLM prompt_logprobs OOM
on long ALFWorld trajectories and making scoring ~3x cheaper. torch/transformers are imported lazily
inside ``__init__`` so importing this module (and the shim) stays cheap on CPU-only paths.
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

logger = logging.getLogger("pi.signals.logprob")


class HFLogprobScorer:
    def __init__(self, model_path: str, device: str = "cuda:0", lm_head_chunk: int = 512):
        import os

        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        self.torch = torch
        self.device = device
        self.chunk = lm_head_chunk
        # cuDNN's fused-attention (SDPA) MHA-graph kernel fails ("mha_graph.execute ... is_good() false")
        # on the very long ALFWorld trajectories (~64k tokens). Disable the cuDNN SDPA backend so SDPA
        # falls back to flash/mem-efficient/math, which handle long sequences fine.
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except Exception:  # older torch without the toggle
            pass
        logger.info("loading %s on %s ...", model_path, device)
        kw = {}
        # MoE ONLY: transformers' default `eager` experts path is a Python for-loop over the experts hit by
        # the batch (Qwen3MoeExperts.forward), so a long sequence touches all 128 experts in all 48 layers
        # => ~6k small GEMMs per forward. `grouped_mm` batches them into one grouped matmul. Requires
        # torch's grouped_mm (sm90+); fall back silently to eager so a 4B/dense run is byte-identical and
        # an unsupported GPU still works, just slowly.
        if getattr(AutoConfig.from_pretrained(model_path), "num_experts", None):
            has_grouped_mm = hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm")
            if has_grouped_mm and os.environ.get("PI_EXPERTS_IMPL", "grouped_mm") != "eager":
                kw["experts_implementation"] = os.environ.get("PI_EXPERTS_IMPL", "grouped_mm")
                logger.info("MoE detected: experts_implementation=%s", kw["experts_implementation"])
            else:
                logger.warning("MoE detected but using the eager experts loop -- expect a large slowdown")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map={"": device}, **kw
        ).eval()
        # Base decoder (no LM head) — so the full-vocab logits are NEVER computed over all positions.
        self.backbone = getattr(self.model, "model", None)
        self.lm_head = self.model.get_output_embeddings()
        if self.backbone is None:
            logger.warning("model has no .model backbone; falling back to output_hidden_states")

    def _hidden_states(self, ids):
        """Final-layer hidden states [seq, hidden] for the full sequence (no LM head applied)."""
        torch = self.torch
        with torch.no_grad():
            if self.backbone is not None:
                out = self.backbone(input_ids=ids, use_cache=False)
                return out.last_hidden_state[0]
            # fallback: full forward, request only 1 logit position so the LM head isn't run over all
            out = self.model(input_ids=ids, use_cache=False, output_hidden_states=True, logits_to_keep=1)
            return out.hidden_states[-1][0]

    def _hidden_states_batch(self, seqs, pad_id: int):
        """Final-layer hidden states for a BATCH: list of [len_b, hidden], unpadded, one per sequence.

        WHY THIS EXISTS. `analyze_pi_positions.py` needs the full hidden ROWS (it computes full-vocab KL /
        entropy / top-k overlap at every student position, not just a summed logprob), so it could not use
        `sum_logprob_at_batch` and ran ONE sequence per forward. At batch 1 a short trajectory (~1.6k
        tokens on ALFWorld 4B-Instruct) leaves an H100 largely idle. Each trajectory issues 1 base + 6
        hint forwards whose lengths differ only by the hint prefix, so they batch almost perfectly.

        Right-padding is the scheme `sum_logprob_at_batch` already uses: real tokens stay at 0..len-1 so
        RoPE positions and the causal mask are unchanged, and the pad tail is attention-masked. Slicing
        `hs[b, :len]` therefore returns exactly what the batch-1 path returns -- verify with
        PI_ASSERT_BATCH_PARITY=1, which re-runs each sequence unbatched and compares.
        """
        torch = self.torch
        B = len(seqs)
        max_len = max(len(s) for s in seqs)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=self.device)
        attn = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
        for b, s in enumerate(seqs):
            input_ids[b, : len(s)] = torch.tensor(s, dtype=torch.long, device=self.device)
            attn[b, : len(s)] = 1
        with torch.no_grad():
            if self.backbone is not None:
                hs = self.backbone(input_ids=input_ids, attention_mask=attn, use_cache=False).last_hidden_state
            else:
                hs = self.model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    use_cache=False,
                    output_hidden_states=True,
                    logits_to_keep=1,
                ).hidden_states[-1]
        return [hs[b, : len(seqs[b])] for b in range(B)]

    def _sum_from_hidden(self, hs_row, token_ids: Sequence[int], positions: Sequence[int]) -> Tuple[float, int]:
        """Sum log pi(token_p | <p) at the given positions, from one sequence's hidden states [seq,hidden].
        LM head applied ONLY at the (student) positions, chunked."""
        torch = self.torch
        import torch.nn.functional as F

        pred_idx = [p - 1 for p in positions if p >= 1]
        targets = [token_ids[p] for p in positions if p >= 1]
        total, n = 0.0, 0
        for i in range(0, len(pred_idx), self.chunk):
            idx = pred_idx[i : i + self.chunk]
            tgt = targets[i : i + self.chunk]
            logits = self.lm_head(hs_row[idx])  # [k, vocab] bf16 — LM head ONLY here
            logp = F.log_softmax(logits.float(), dim=-1)
            tgt_t = torch.tensor(tgt, device=self.device)
            total += logp[torch.arange(len(idx)), tgt_t].sum().item()
            n += len(idx)
        return total, n

    def sum_logprob_at(self, token_ids: Sequence[int], positions: Sequence[int]) -> Tuple[float, int]:
        """Single-sequence: sum log pi over the given (student) positions. LM head applied only there."""
        if not positions:
            return 0.0, 0
        ids = self.torch.tensor([list(token_ids)], device=self.device)
        hs = self._hidden_states(ids)  # [seq, hidden], bf16
        return self._sum_from_hidden(hs, token_ids, positions)

    def sum_logprob_at_batch(self, seqs, positions_list, pad_id: int) -> List[Tuple[float, int]]:
        """Batched: one right-padded backbone forward for the whole batch, then per-sequence student-only
        LM-head gather. Right-padding keeps real tokens at 0..len-1 (correct RoPE + causal), pad masked."""
        torch = self.torch
        B = len(seqs)
        max_len = max(len(s) for s in seqs)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=self.device)
        attn = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
        for b, s in enumerate(seqs):
            input_ids[b, : len(s)] = torch.tensor(s, dtype=torch.long, device=self.device)
            attn[b, : len(s)] = 1
        with torch.no_grad():
            if self.backbone is not None:
                hs = self.backbone(input_ids=input_ids, attention_mask=attn).last_hidden_state  # [B,max,H]
            else:
                hs = self.model(
                    input_ids=input_ids, attention_mask=attn, output_hidden_states=True, logits_to_keep=1
                ).hidden_states[-1]
        return [
            (self._sum_from_hidden(hs[b], seqs[b], pos) if pos else (0.0, 0)) for b, pos in enumerate(positions_list)
        ]


__all__ = ["HFLogprobScorer"]

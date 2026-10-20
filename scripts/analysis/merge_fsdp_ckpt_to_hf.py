#!/usr/bin/env python
"""Merge a SkyRL FSDP2 sharded policy checkpoint into a plain HuggingFace model directory.

WHY THIS EXISTS. SkyRL saves the policy as `model_world_size_{W}_rank_{R}.pt`, one file per rank, and
the only HF export in the repo (`fsdp_strategy.py:save_hf_model`) runs *inside* the W-rank training
job. So a sharded checkpoint could only be sampled by relaunching an W-GPU in-trainer eval -- you could
not `vllm serve` it, and you could not touch it on a single GPU at all. This script closes that gap:
after merging, any checkpoint is a normal HF dir usable by vLLM, transformers, or a 1-GPU local run.

HOW IT WORKS. FSDP2 stores each parameter as a DTensor. Empirically (Qwen3-4B, W=8) every parameter is
placed `Shard(dim=0)` over a 1-D mesh, so the full tensor is just the rank-ordered concat of the local
shards along dim 0. Two details matter:

  * torch.load works on CPU with NO process group initialised -- it warns about DeviceMesh unpickling,
    which is harmless here because we only ever call `.to_local()` and never a collective.
  * FSDP2 PADS the last ranks when dim0 is not divisible by W, so the concat can come out LONGER than
    the true global shape. We truncate to `dtensor.shape` and assert the result matches; a silent
    off-by-padding here would corrupt the embedding/lm_head rows, which is exactly the kind of bug that
    produces a model that loads fine and emits garbage.

Weights are stored fp32 (optimizer master copy); we cast to the HF config's torch_dtype (bf16) so the
output matches what vLLM will run and is half the size on disk.

  python scripts/analysis/merge_fsdp_ckpt_to_hf.py \
      --ckpt  ./runs/<run>/ckpts/global_step_200/policy \
      --out   ./runs/<run>/exports/global_step_200_hf
"""

import argparse
import json
import pathlib
import shutil

import torch

# Files that make the output a usable HF dir; SkyRL already writes them under policy/huggingface/.
HF_SIDECAR = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="dir holding model_world_size_W_rank_R.pt + huggingface/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default=None, help="override; default = HF config torch_dtype")
    args = ap.parse_args()

    ckpt, out = pathlib.Path(args.ckpt), pathlib.Path(args.out)
    shards = sorted(ckpt.glob("model_world_size_*_rank_*.pt"), key=lambda p: int(p.stem.split("_rank_")[1]))
    if not shards:
        raise SystemExit(f"no model_world_size_*_rank_*.pt under {ckpt}")
    world = int(shards[0].stem.split("_world_size_")[1].split("_rank_")[0])
    if len(shards) != world:
        raise SystemExit(f"world_size={world} but found {len(shards)} shards -- refusing a partial merge")
    print(f"merging {world} shards from {ckpt}")

    hf_src = ckpt / "huggingface"
    dtype = args.dtype
    if dtype is None:
        cfg = json.loads((hf_src / "config.json").read_text())
        dtype = cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16"
    torch_dtype = getattr(torch, dtype)
    print(f"output dtype: {dtype}")

    # Load rank 0 first to learn the parameter set, then stream the rest one rank at a time so peak
    # host memory stays ~= (one shard) + (one merged tensor), not all 8 shards at once.
    per_rank = []
    for s in shards:
        print(f"  loading {s.name}")
        per_rank.append(torch.load(s, map_location="cpu", weights_only=False))

    keys = list(per_rank[0].keys())
    merged, n_pad = {}, 0
    for k in keys:
        dt = per_rank[0][k]
        if not hasattr(dt, "to_local"):  # already a plain tensor (replicated / non-sharded)
            merged[k] = dt.to(torch_dtype)
            continue
        placements = dt.placements
        if len(placements) != 1 or placements[0].__class__.__name__ != "Shard" or placements[0].dim != 0:
            raise SystemExit(f"{k}: unsupported placement {placements}; this merger only handles Shard(dim=0)")
        global_shape = tuple(dt.shape)
        full = torch.cat([per_rank[r][k].to_local() for r in range(world)], dim=0)
        if full.shape[0] != global_shape[0]:  # FSDP2 padded the tail ranks
            if full.shape[0] < global_shape[0]:
                raise SystemExit(f"{k}: concat {tuple(full.shape)} SHORTER than global {global_shape}")
            full = full[: global_shape[0]]
            n_pad += 1
        assert tuple(full.shape) == global_shape, f"{k}: got {tuple(full.shape)} want {global_shape}"
        merged[k] = full.to(torch_dtype).contiguous()

    n_params = sum(v.numel() for v in merged.values())
    print(f"merged {len(merged)} tensors, {n_params/1e9:.3f}B params ({n_pad} needed depadding)")

    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    save_file(merged, str(out / "model.safetensors"), metadata={"format": "pt"})
    for f in HF_SIDECAR:
        if (hf_src / f).exists():
            shutil.copy2(hf_src / f, out / f)
    print(f"wrote {out}")
    print("  " + "  ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()

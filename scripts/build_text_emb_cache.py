"""
Pre-compute and cache LLM2Vec text embeddings for every prompt in a
subset.jsonl. Saves a single .pt file with `{text: tensor[1, llm_dim]}`.

Once the cache exists, `run_kimodo_batched.py --emb-cache <path>` swaps the
model's `text_encoder` for a cached lookup, so the Llama-3 8B base model is
never loaded during inference. That removes the ~30 s load + per-batch encode
cost (and the 16 GB VRAM allocation for Llama).

Usage:
    python scripts/build_text_emb_cache.py \
        --subset runs/eval_subset/t2m_overview_full.jsonl \
        --out runs/eval_subset/t2m_overview_text_emb.pt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dim", type=int, default=4096,
                   help="Llama-3-8B-Instruct hidden dim")
    p.add_argument("--dtype", default="float16",
                   help="encoder dtype; embeddings are then upcast to fp32 for cache")
    args = p.parse_args()

    os.environ.setdefault("TEXT_ENCODER_DEVICE", args.device)

    rows = [json.loads(l) for l in open(args.subset) if l.strip()]
    texts = [r["text"] for r in rows if r.get("text", "").strip()]
    uniq = sorted(set(texts))
    print(f"[cache] {len(rows)} rows, {len(uniq)} unique texts")

    # Lazy import after env vars set.
    from kimodo.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder

    enc = LLM2VecEncoder(
        base_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        dtype=args.dtype,
        llm_dim=args.llm_dim,
        device=args.device,
    )
    print("[cache] encoder loaded")

    cache: dict[str, torch.Tensor] = {}
    for i, t in enumerate(uniq):
        feat, lengths = enc(t)  # feat shape: (1, llm_dim) for single string
        # `enc` returns single-text shape (1, llm_dim) when input is a str (per
        # llm2vec_wrapper.py: encoded_text = encoded_text[0] → (1, llm_dim))
        cache[t] = feat.detach().to(torch.float32).cpu()
        if (i + 1) % 50 == 0 or i + 1 == len(uniq):
            print(f"[cache] {i+1}/{len(uniq)} encoded; "
                  f"shape={tuple(feat.shape)} dtype={feat.dtype}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embeddings": cache,
        "llm_dim": args.llm_dim,
        "shape_note": "value is (1, llm_dim) — kimodo expects a 'length' axis",
    }, out)
    print(f"[cache] wrote {out}  ({len(cache)} entries)")


if __name__ == "__main__":
    main()

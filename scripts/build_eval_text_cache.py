"""
Build an LLM2Vec embedding cache for every prompt in the text2motion
testsuite groups, so TMR-G1 retrieval eval can run with CachedTextEncoder
(no live Llama-3 at eval time). Same blob layout as the training cache.

Usage:
    python scripts/build_eval_text_cache.py \
        --testsuite data/kimodo_benchmark/testsuite \
        --out data/bones_seed/eval_text_emb.pt --device cuda
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--testsuite", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--llm-dim", type=int, default=4096)
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args()

    os.environ.setdefault("TEXT_ENCODER_DEVICE", args.device)

    base = Path(args.testsuite)
    metas = []
    for split in ("content", "repetition"):
        for grp in ("overview", "timeline_single", "timeline_multi"):
            metas += glob.glob(str(base / split / "text2motion" / grp / "*" / "meta.json"))
    texts = []
    for m in metas:
        t = json.load(open(m)).get("text", "")
        if isinstance(t, str) and t.strip():
            texts.append(t.strip())
    uniq = sorted(set(texts))
    print(f"[eval-text-cache] {len(metas)} meta.json, {len(texts)} prompts, {len(uniq)} unique")

    from kimodo.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder
    enc = LLM2VecEncoder(
        base_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        dtype=args.dtype, llm_dim=args.llm_dim, device=args.device,
    )
    feats = {}
    t0 = time.time()
    for i, cap in enumerate(uniq):
        f, _ = enc(cap)
        feats[cap] = f.detach().to(torch.float32).cpu()
        if (i + 1) % args.log_every == 0 or i + 1 == len(uniq):
            r = (i + 1) / max(1e-6, time.time() - t0)
            print(f"[eval-text-cache] {i+1}/{len(uniq)} ({r:.1f}/s)", flush=True)

    captions = uniq
    features = torch.stack([feats[c] for c in captions], dim=0).squeeze(1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"captions": captions, "features": features,
                "meta": {"encoder_type": "llm2vec", "dim": int(features.shape[1])}}, args.out)
    print(f"[eval-text-cache] wrote {args.out} shape={tuple(features.shape)}")


if __name__ == "__main__":
    main()

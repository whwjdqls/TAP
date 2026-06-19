"""
Build a CachedTextEncoder-compatible LLM2Vec embedding cache for every text
in the Bones-SEED corpus, drawn from three sources:

  1. ``natural`` — content_natural_desc_{1..4} columns of
     ``seed_metadata_v004.csv``.
  2. ``single`` — events[].description in
     ``seed_metadata_v002_temporal_labels.jsonl``.
  3. ``multi``  — merged_description in ``multi_timeline.jsonl``.

The cache is restricted to motions whose ``move_name`` appears in any of the
benchmark splits (train / test_content / test_repetition). Captions are
deduplicated so the LLM2Vec pass only runs for unique strings.

Output blob (same layout as ``kimodo.scripts.precompute_text_embeddings``):

    {
      "captions": List[str],          # N unique captions
      "features": Tensor(N, llm_dim), # float32 pooled embeddings
      "meta":     {"encoder_type": "llm2vec",
                   "model": "...mntp/...mntp-supervised",
                   "dim": llm_dim,
                   "sources": ["natural","single","multi"],
                   "filename_count": int,
                   "include_mirrored": bool}
    }

Drop the resulting .pt into ``kimodo.model.cached_text.CachedTextEncoder``
to skip the live encoder during training.

Usage:
    python scripts/build_g1_text_emb_cache.py \
        --natural-csv data/bones_seed/metadata/metadata/seed_metadata_v004.csv \
        --single-jsonl data/bones_seed/metadata/metadata/seed_metadata_v002_temporal_labels.jsonl \
        --multi-jsonl data/bones_seed/multi_timeline.jsonl \
        --splits-dir data/kimodo_benchmark/splits \
        --out data/bones_seed/g1_text_emb.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import torch


def _load_names_from_split(splits_dir: Path) -> set[str]:
    names = set()
    for fname in ("train_split_paths.txt", "test_content_split_paths.txt",
                  "test_repetition_split_paths.txt"):
        fp = splits_dir / fname
        if not fp.is_file():
            print(f"WARN: missing split file {fp}", file=sys.stderr)
            continue
        with open(fp) as f:
            for line in f:
                rel = line.strip()
                if not rel:
                    continue
                # rel = "<session>/<move_name>"
                base = rel.rsplit("/", 1)[-1]
                names.add(base)
    return names


def _iter_natural(csv_path: Path, allowed: set[str], include_mirrored: bool) -> Iterable[str]:
    cols = ("content_natural_desc_1", "content_natural_desc_2",
            "content_natural_desc_3", "content_natural_desc_4",
            "content_short_description", "content_short_description_2")
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            n = row.get("move_name") or row.get("filename") or ""
            if n not in allowed:
                continue
            if not include_mirrored and n.endswith("_M"):
                continue
            for c in cols:
                v = row.get(c, "")
                if isinstance(v, str):
                    v = v.strip()
                if v:
                    yield v


def _iter_single(jsonl_path: Path, allowed: set[str], include_mirrored: bool) -> Iterable[str]:
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            n = obj["filename"]
            if n not in allowed:
                continue
            if not include_mirrored and n.endswith("_M"):
                continue
            for ev in obj.get("events", []):
                t = ev.get("description")
                if isinstance(t, str) and t.strip():
                    yield t.strip()


def _iter_multi(jsonl_path: Path, allowed: set[str], include_mirrored: bool) -> Iterable[str]:
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            n = obj["filename"]
            if n not in allowed:
                continue
            if not include_mirrored and n.endswith("_M"):
                continue
            t = obj.get("merged_description")
            if isinstance(t, str) and t.strip():
                yield t.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--natural-csv", required=True)
    p.add_argument("--single-jsonl", required=True)
    p.add_argument("--multi-jsonl", required=True)
    p.add_argument("--splits-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--include-mirrored", action="store_true", default=True)
    p.add_argument("--llm-dim", type=int, default=4096)
    p.add_argument("--dtype", default="bfloat16",
                   help="encoder dtype; embeddings always saved fp32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true",
                   help="if --out exists, skip captions already cached")
    p.add_argument("--limit", type=int, default=0,
                   help="cap unique captions (for smoke tests)")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=2000,
                   help="periodically flush partial cache to disk for resume")
    args = p.parse_args()

    os.environ.setdefault("TEXT_ENCODER_DEVICE", args.device)

    allowed = _load_names_from_split(Path(args.splits_dir))
    print(f"[g1-text-cache] allowed move_names from splits: {len(allowed):,}")

    counts = {"natural": 0, "single": 0, "multi": 0}
    captions: set[str] = set()
    for c in _iter_natural(Path(args.natural_csv), allowed, args.include_mirrored):
        captions.add(c); counts["natural"] += 1
    print(f"[g1-text-cache] natural captions raw={counts['natural']:,} unique-running={len(captions):,}")
    for c in _iter_single(Path(args.single_jsonl), allowed, args.include_mirrored):
        captions.add(c); counts["single"] += 1
    print(f"[g1-text-cache] +single captions raw={counts['single']:,} unique-running={len(captions):,}")
    for c in _iter_multi(Path(args.multi_jsonl), allowed, args.include_mirrored):
        captions.add(c); counts["multi"] += 1
    print(f"[g1-text-cache] +multi captions raw={counts['multi']:,} unique-running={len(captions):,}")

    uniq = sorted(captions)
    if args.limit:
        uniq = uniq[: args.limit]
    print(f"[g1-text-cache] writing cache for {len(uniq):,} unique captions")

    # Resume support.
    existing: dict[str, torch.Tensor] = {}
    out_path = Path(args.out)
    if args.resume and out_path.is_file():
        blob = torch.load(str(out_path), map_location="cpu", weights_only=False)
        for cap, feat in zip(blob["captions"], blob["features"]):
            existing[cap] = feat
        print(f"[g1-text-cache] resume: {len(existing):,} already cached")

    to_encode = [c for c in uniq if c not in existing]
    print(f"[g1-text-cache] {len(to_encode):,} to encode")

    if to_encode:
        # Lazy import after env vars set.
        from kimodo.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder
        enc = LLM2VecEncoder(
            base_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            dtype=args.dtype,
            llm_dim=args.llm_dim,
            device=args.device,
        )
        t0 = time.time()
        for i, cap in enumerate(to_encode):
            feat, _ = enc(cap)  # (1, llm_dim)
            existing[cap] = feat.detach().to(torch.float32).cpu()
            if (i + 1) % args.log_every == 0 or i + 1 == len(to_encode):
                rate = (i + 1) / max(1e-6, time.time() - t0)
                eta = (len(to_encode) - (i + 1)) / max(1e-6, rate) / 60
                print(f"[g1-text-cache] {i+1:,}/{len(to_encode):,} "
                      f"({rate:.1f}/s, ETA {eta:.1f} min)", flush=True)
            if (i + 1) % args.save_every == 0:
                # Partial flush in cache layout (sorted captions, fp32 features).
                captions_list = sorted(existing.keys())
                features = torch.stack([existing[c] for c in captions_list], dim=0).squeeze(1)
                tmp = out_path.with_suffix(".pt.partial")
                torch.save({
                    "captions": captions_list, "features": features,
                    "meta": {"encoder_type": "llm2vec",
                             "model": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
                             "dim": int(features.shape[1])},
                }, tmp)
                tmp.replace(out_path)
                print(f"[g1-text-cache] partial save: {len(captions_list):,} entries", flush=True)

    # Stack in deterministic order matching `captions` list.
    captions_list = uniq
    features = torch.stack([existing[c] for c in captions_list], dim=0).squeeze(1)
    # features is shape (N, llm_dim) fp32 (cached as (1, llm_dim) per row → squeezed).

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "captions": captions_list,
        "features": features,
        "meta": {
            "encoder_type": "llm2vec",
            "model": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            "dim": int(features.shape[1]),
            "sources": ["natural", "single", "multi"],
            "filename_count": len(allowed),
            "include_mirrored": args.include_mirrored,
            "raw_counts": counts,
        },
    }, out_path)
    print(f"[g1-text-cache] wrote {out_path}  shape={tuple(features.shape)} dtype={features.dtype}")


if __name__ == "__main__":
    main()

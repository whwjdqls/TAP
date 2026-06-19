"""Cached text encoder — serves precomputed LLM2Vec embeddings from a .pt
cache with the same call signature as the live encoders.

    enc = CachedTextEncoder(cache_path=..., device="cuda")
    feats, lengths = enc(list_of_strings)   # feats: (B, 1, D), lengths: [1]*B

Cache blob layout (produced by scripts/build_g1_text_emb_cache.py):
    {"captions": List[str], "features": Tensor(N, D), "meta": {...}}

Cache misses raise KeyError — no live-encoder fallback by design (point is to
never load Llama-3 during training). Adapted from kimodo_open's
kimodo.model.cached_text.cached_encoder.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import torch


class CachedTextEncoder:
    def __init__(self, cache_path: Union[str, Path], device: str = "cpu"):
        cache_path = Path(cache_path)
        if not cache_path.is_file():
            raise FileNotFoundError(f"text-embedding cache not found: {cache_path}")
        try:
            blob = torch.load(str(cache_path), map_location="cpu",
                              weights_only=False, mmap=True)
        except (RuntimeError, TypeError):
            blob = torch.load(str(cache_path), map_location="cpu", weights_only=False)

        captions: List[str] = list(blob["captions"])
        features: torch.Tensor = blob["features"]
        if features.dtype != torch.float32:
            features = features.to(torch.float32)
        if features.dim() != 2 or features.shape[0] != len(captions):
            raise ValueError(f"cache shape mismatch: features={tuple(features.shape)} "
                             f"vs #captions={len(captions)}")

        self._captions = captions
        self._caption_to_idx = {c: i for i, c in enumerate(captions)}
        self._features = features
        self._device = torch.device(device)
        self._cache_path = cache_path
        self.text_dim = int(features.shape[1])
        self.llm_dim = self.text_dim
        self.meta = dict(blob.get("meta", {}))

    def to(self, device):
        self._device = torch.device(device)
        return self

    def eval(self):
        return self

    def get_device(self):
        return self._device

    def __len__(self) -> int:
        return len(self._captions)

    def __contains__(self, caption: str) -> bool:
        return caption in self._caption_to_idx

    @torch.no_grad()
    def __call__(self, text: Union[str, List[str]]) -> Tuple[torch.Tensor, List[int]]:
        is_string = isinstance(text, str)
        if is_string:
            text = [text]
        missing, idxs = [], []
        for cap in text:
            i = self._caption_to_idx.get(cap)
            if i is None:
                missing.append(cap)
            else:
                idxs.append(i)
        if missing:
            raise KeyError(f"{len(missing)} caption(s) missing from cache "
                           f"({self._cache_path}). Examples: {missing[:3]}")
        feats = self._features[torch.tensor(idxs, dtype=torch.long)]  # (B, D)
        feats = feats.unsqueeze(1).to(self._device, non_blocking=True)  # (B, 1, D)
        lengths = [1] * feats.shape[0]
        if is_string:
            feats = feats[0]
            lengths = lengths[0]
        return feats, lengths

"""Bones-SEED G1 text->motion dataset for g1_rep_v1 (142-D).

Mirrors HumanML3DNativeTextMotionDataset's batch contract (so kimodo's
``train_one_step`` + ``build_collate_fn`` work unchanged): reads precomputed
g1_rep_v1 feature ``.npy`` (T,142), z-scores with the flat Mean/Std, random-
windows + right-pads. Text comes from a prebuilt {<sess>/<move>: [captions]}
index (natural Bones-SEED descriptions, all present in the LLM2Vec cache).

Item dict: {motion (max_T,142) z-scored, length int, text str,
first_heading_angle 0.0, filename str}.
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from kimodo.data.humanml3d_text_motion import build_collate_fn  # reused collate

import g1_rep_v1

log = logging.getLogger(__name__)
FEAT_DIM = g1_rep_v1.FEAT_DIM


class G1RepV1TextMotionDataset(Dataset):
    def __init__(
        self,
        feat_root: str | Path,
        text_index: str | Path,
        mean: np.ndarray,
        std: np.ndarray,
        fps: int = 20,
        window_size: int = 200,
        max_motion_length: int = 200,
        min_motion_len: int = 24,
        unit_length: int = 4,
        clip_normalized: Optional[float] = None,
        motion_rep=None,
        skeleton=None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.feat_root = Path(feat_root)
        self.fps = int(fps)
        self.window_size = int(window_size)
        self.max_motion_length = int(max_motion_length)
        self.min_motion_len = int(min_motion_len)
        self.unit_length = int(unit_length)
        self.clip_normalized = float(clip_normalized) if clip_normalized else None
        self.skeleton = skeleton
        self.motion_rep = motion_rep

        if mean.shape != (FEAT_DIM,) or std.shape != (FEAT_DIM,):
            raise ValueError(f"mean/std must be ({FEAT_DIM},), got {mean.shape}/{std.shape}")
        std = np.where(std < 1e-4, np.float32(1.0), std).astype(np.float32)
        self.mean, self.std = mean.astype(np.float32), std

        self._rng = random.Random(seed)
        with open(text_index) as f:
            index = json.load(f)

        # (key, caption) samples; cache each motion's feats in memory (small).
        self.feats: dict = {}
        self.samples: List[Tuple[str, str]] = []
        n_missing = n_short = 0
        for key, caps in index.items():
            fp = self.feat_root / f"{key}.npy"
            if not fp.is_file():
                n_missing += 1
                continue
            try:
                arr = np.load(fp).astype(np.float32)
            except Exception:
                n_missing += 1
                continue
            if arr.ndim != 2 or arr.shape[1] != FEAT_DIM or arr.shape[0] < self.min_motion_len:
                n_short += 1
                continue
            self.feats[key] = arr
            for c in caps:
                self.samples.append((key, c))
        if not self.samples:
            raise RuntimeError(f"No samples from {text_index} / {feat_root}")
        log.info("G1RepV1: %d motions, %d (motion,caption) samples (missing=%d short<%d=%d)",
                 len(self.feats), len(self.samples), n_missing, self.min_motion_len, n_short)

    def __len__(self) -> int:
        return len(self.samples)

    def _pick_window(self, total_T: int) -> Tuple[int, int]:
        if self.unit_length < 10:
            coin = self._rng.choice(["single", "single", "double"])
        else:
            coin = "single"
        cap = min(total_T, self.window_size)
        if coin == "double":
            m_len = max(self.unit_length, (cap // self.unit_length - 1) * self.unit_length)
        else:
            m_len = (cap // self.unit_length) * self.unit_length
        m_len = max(min(m_len, total_T), self.min_motion_len)
        m_len = min(m_len, total_T)
        start = self._rng.randint(0, max(0, total_T - m_len))
        return start, m_len

    def __getitem__(self, item: int) -> dict:
        key, caption = self.samples[item]
        full = self.feats[key]
        start, m_len = self._pick_window(full.shape[0])
        feats = full[start:start + m_len].copy()
        feats = (feats - self.mean) / self.std
        if self.clip_normalized is not None:
            feats = np.clip(feats, -self.clip_normalized, self.clip_normalized)
        motion = np.zeros((self.max_motion_length, FEAT_DIM), dtype=np.float32)
        motion[:feats.shape[0]] = feats
        return {
            "motion": torch.from_numpy(motion),
            "length": int(feats.shape[0]),
            "text": caption,
            "first_heading_angle": 0.0,   # rep is heading-canonical at frame 0
            "filename": key,
        }


def build_g1_collate_fn():
    return build_collate_fn()

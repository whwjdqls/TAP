"""TMR training losses: KL + reconstruction + symmetric InfoNCE."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean KL( N(mu, sigma^2) || N(0, I) ) over batch."""
    # 0.5 * sum_d ( exp(logvar) + mu^2 - 1 - logvar )
    return 0.5 * torch.mean(torch.sum(logvar.exp() + mu.pow(2) - 1.0 - logvar, dim=-1))


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """L2 between predicted and target features, optionally masked."""
    diff = (pred - target).pow(2).sum(-1)  # (B, T)
    if mask is None:
        return diff.mean()
    m = mask.float()
    n = m.sum().clamp(min=1.0)
    return (diff * m).sum() / n


def info_nce(
    motion_emb: torch.Tensor,   # (B, D), L2-normalized
    text_emb: torch.Tensor,     # (B, D), L2-normalized
    temperature: float = 0.1,
    dup_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Symmetric InfoNCE with optional duplicate-text masking.

    `dup_mask[i, j] = True` if text_i and text_j refer to the same caption (or
    similar enough that we shouldn't treat j as a negative for i). The
    diagonal is the positive and is always preserved.
    """
    sim = motion_emb @ text_emb.t() / temperature  # (B, B)
    if dup_mask is not None:
        # Don't penalize the model for matching duplicate-text negatives.
        # Set masked off-diagonal entries to -inf so they vanish in softmax.
        B = sim.shape[0]
        eye = torch.eye(B, dtype=torch.bool, device=sim.device)
        bad = dup_mask & ~eye
        sim = sim.masked_fill(bad, float("-inf"))

    # m2t: row-wise (motion -> text); t2m: column-wise (text -> motion)
    targets = torch.arange(sim.shape[0], device=sim.device)
    loss_m2t = F.cross_entropy(sim, targets)
    loss_t2m = F.cross_entropy(sim.t(), targets)
    return 0.5 * (loss_m2t + loss_t2m)


def info_nce_membank(
    zm: torch.Tensor,            # (B, D) L2-normalized motion emb (with grad)
    zt: torch.Tensor,            # (B, D) L2-normalized text emb (with grad)
    queue_zm: torch.Tensor = None,   # (Q, D) detached past motion emb
    queue_zt: torch.Tensor = None,   # (Q, D) detached past text emb
    temperature: float = 0.1,
    text_dup_threshold: float = 0.9,
) -> torch.Tensor:
    """Symmetric InfoNCE with a memory-bank of extra negatives + projected
    text-similarity false-negative masking.

    Negatives = other batch items + the queue. A candidate key is masked
    (excluded from the softmax) when its text is near-duplicate of the query's
    text (cosine in the projected 256-d space > threshold) — except the
    positive, which is always kept. This mirrors the benchmark's 0.99 text-dedup
    so we don't penalize the model for semantically-equivalent captions.
    """
    B = zm.shape[0]
    device = zm.device
    if queue_zm is not None and queue_zm.numel() > 0:
        keys_m = torch.cat([zm, queue_zm], dim=0)  # (B+Q, D)
        keys_t = torch.cat([zt, queue_zt], dim=0)
    else:
        keys_m, keys_t = zm, zt

    pos = torch.arange(B, device=device)
    # text-text cosine between batch queries and all keys (projected space).
    tt = zt @ keys_t.t()                       # (B, B+Q)
    dup = tt > text_dup_threshold              # mask these as false negatives
    dup[pos, pos] = False                      # never mask the positive
    neg_inf = torch.finfo(zm.dtype).min

    logit_t2m = (zt @ keys_m.t()) / temperature   # text query -> motion keys
    logit_t2m = logit_t2m.masked_fill(dup, neg_inf)
    loss_t2m = F.cross_entropy(logit_t2m, pos)

    logit_m2t = (zm @ keys_t.t()) / temperature   # motion query -> text keys
    logit_m2t = logit_m2t.masked_fill(dup, neg_inf)
    loss_m2t = F.cross_entropy(logit_m2t, pos)
    return 0.5 * (loss_t2m + loss_m2t)


def duplicate_mask(texts: list[str]) -> torch.Tensor:
    """Boolean mask of identical-text pairs (within-batch dedup).

    Cheap proxy for the benchmark's 0.99-text-sim grouping; for richer dup
    detection we'd run all batch texts through the (frozen) text encoder and
    threshold cosine similarity, but exact-string already handles paraphrase
    repetition in BONES-SEED.
    """
    B = len(texts)
    out = torch.zeros(B, B, dtype=torch.bool)
    by_text: dict[str, list[int]] = {}
    for i, t in enumerate(texts):
        by_text.setdefault(t, []).append(i)
    for idxs in by_text.values():
        if len(idxs) < 2:
            continue
        for i in idxs:
            for j in idxs:
                out[i, j] = True
    return out


__all__ = ["kl_to_standard_normal", "reconstruction_loss",
           "info_nce", "duplicate_mask"]

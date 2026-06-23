"""Motion-VAE training losses: reconstruction + KL.

These are the exact same two terms used by `tmr_g1.training.losses`; the
contrastive (InfoNCE) loss is intentionally absent — a motion VAE has no text
branch to contrast against.
"""
from __future__ import annotations

from typing import Optional

import torch


def kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean KL( N(mu, sigma^2) || N(0, I) ) over batch."""
    # 0.5 * sum_d ( exp(logvar) + mu^2 - 1 - logvar )
    return 0.5 * torch.mean(torch.sum(logvar.exp() + mu.pow(2) - 1.0 - logvar, dim=-1))


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """L2 between predicted and target features, optionally masked over time."""
    diff = (pred - target).pow(2).sum(-1)  # (B, T)
    if mask is None:
        return diff.mean()
    m = mask.float()
    n = m.sum().clamp(min=1.0)
    return (diff * m).sum() / n


__all__ = ["kl_to_standard_normal", "reconstruction_loss"]

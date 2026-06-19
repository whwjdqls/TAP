"""Build a TMR model wired to G1Skeleton34 + a frozen CachedTextEncoder.

Returns a `kimodo.model.tmr.TMR` instance ready for training:
  - motion encoder consumes `TMRMotionRep(G1Skeleton34, fps=20)` features (210-d)
  - top text encoder consumes `(B, 1, llm_dim)` lookups from the cache
  - VAE-style; outputs are L2-normalized at retrieval time

Also exposes an ACTOR-style motion decoder for the reconstruction loss
(architecturally mirrors the encoder, predicts the same `motion_rep` features
that the encoder consumes).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import repeat

from kimodo.model.tmr import ACTORStyleEncoder, PositionalEncoding, TMR
from kimodo.motion_rep.reps.tmr_motionrep import TMRMotionRep
from kimodo.skeleton import G1Skeleton34


class ACTORStyleDecoder(nn.Module):
    """Symmetric ACTOR-style transformer decoder.

    Given a latent token `z [B, latent_dim]` and a target length `T`, produce
    motion features `[B, T, nfeats]`. Architecture mirrors the encoder so the
    parameter budget stays close to the SOMA TMR's motion_decoder.pt
    (~similar scale).
    """

    def __init__(self,
                 nfeats: int,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu") -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.nfeats = nfeats

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout=dropout, batch_first=True)

        seq_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.seqTransDecoder = nn.TransformerDecoder(seq_layer, num_layers=num_layers)
        self.final_layer = nn.Linear(latent_dim, nfeats)

    def forward(self, z: torch.Tensor, T: int) -> torch.Tensor:
        """z: (B, latent_dim) -> (B, T, nfeats)."""
        B = z.shape[0]
        device = z.device
        # Memory = the single latent token, broadcast.
        memory = z.unsqueeze(1)  # (B, 1, latent_dim)
        # Query = positional encoding for each output frame.
        tgt = torch.zeros(B, T, self.latent_dim, device=device, dtype=z.dtype)
        tgt = self.sequence_pos_encoding(tgt)
        out = self.seqTransDecoder(tgt=tgt, memory=memory)  # (B, T, latent_dim)
        return self.final_layer(out)


def build_tmr_g1(
    text_emb_cache_path: str,
    stats_path: Optional[str] = None,
    fps: int = 20,
    latent_dim: int = 256,
    ff_size: int = 1024,
    num_layers: int = 6,
    num_heads: int = 4,
    dropout: float = 0.1,
    vae: bool = True,
    activation: str = "gelu",
    device: str = "cuda",
) -> Tuple[TMR, TMRMotionRep, ACTORStyleDecoder]:
    """Construct (tmr, motion_rep, motion_decoder) for training."""
    from tmr_g1.model.cached_text import CachedTextEncoder

    skeleton = G1Skeleton34()
    motion_rep = TMRMotionRep(skeleton=skeleton, fps=fps, stats_path=stats_path)

    motion_encoder = ACTORStyleEncoder(
        motion_rep=motion_rep,
        llm_shape=None,
        vae=vae,
        latent_dim=latent_dim,
        ff_size=ff_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        activation=activation,
        ckpt_path=None,
    ).to(device)

    text_encoder_cache = CachedTextEncoder(cache_path=text_emb_cache_path, device=device)
    llm_dim = text_encoder_cache.text_dim
    top_text_encoder = ACTORStyleEncoder(
        motion_rep=None,
        llm_shape=(1, llm_dim),
        vae=vae,
        latent_dim=latent_dim,
        ff_size=ff_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        activation=activation,
        ckpt_path=None,
    ).to(device)

    tmr = TMR(
        motion_encoder=motion_encoder,
        top_text_encoder=top_text_encoder,
        vae=vae,
        text_encoder=text_encoder_cache,
        device=device,
        sample_mean=False,  # sample during training; mean at eval
        unit_vector=True,
        compute_grads=True,
    ).to(device)

    motion_decoder = ACTORStyleDecoder(
        nfeats=motion_rep.motion_rep_dim,
        latent_dim=latent_dim,
        ff_size=ff_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        activation=activation,
    ).to(device)
    return tmr, motion_rep, motion_decoder


__all__ = ["build_tmr_g1", "ACTORStyleDecoder"]

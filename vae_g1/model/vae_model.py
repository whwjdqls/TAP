"""Build a motion VAE wired to G1Skeleton34 + TMRMotionRep features.

Mirrors `tmr_g1.model.tmr_model` but keeps ONLY the motion path:
  - ACTOR-style transformer encoder consumes `TMRMotionRep(G1Skeleton34, fps=20)`
    features (210-d) -> (mu, logvar) over a `latent_dim` latent.
  - ACTOR-style transformer decoder maps a latent token + target length back to
    the same `motion_rep` features (the reconstruction target).

There is no text encoder and no contrastive head — training is reconstruction
+ KL only (see `vae_g1.training.losses`). The encoder is byte-for-byte the same
module class (`kimodo.model.tmr.ACTORStyleEncoder`) that TMR-G1 uses, so a
trained VAE encoder is a drop-in motion feature extractor.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from kimodo.model.tmr import ACTORStyleEncoder, PositionalEncoding
from kimodo.motion_rep.reps.tmr_motionrep import TMRMotionRep
from kimodo.skeleton import G1Skeleton34


class ACTORStyleDecoder(nn.Module):
    """Symmetric ACTOR-style transformer decoder.

    Given a latent token `z [B, latent_dim]` and a target length `T`, produce
    motion features `[B, T, nfeats]`. Architecture mirrors the encoder.
    """

    def __init__(self,
                 nfeats: int,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 6,
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
        memory = z.unsqueeze(1)  # (B, 1, latent_dim)
        tgt = torch.zeros(B, T, self.latent_dim, device=device, dtype=z.dtype)
        tgt = self.sequence_pos_encoding(tgt)
        out = self.seqTransDecoder(tgt=tgt, memory=memory)  # (B, T, latent_dim)
        return self.final_layer(out)


class MotionVAE(nn.Module):
    """ACTOR-style motion VAE: encoder -> (mu, logvar) -> reparam -> decoder."""

    def __init__(self, encoder: ACTORStyleEncoder, decoder: ACTORStyleDecoder,
                 vae: bool = True) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vae = vae

    def encode(self, feats: torch.Tensor, mask: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """feats: (B, T, nfeats), mask: (B, T) -> (mu, logvar) each (B, latent)."""
        out = self.encoder({"x": feats, "mask": mask})  # (B, 2, latent) if vae
        if self.vae:
            mu, logvar = out.unbind(1)
        else:
            mu, logvar = out[:, 0], torch.zeros_like(out[:, 0])
        return mu, logvar

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor, training: bool
                ) -> torch.Tensor:
        if not training:
            return mu
        std = (0.5 * logvar).exp()
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, T: int) -> torch.Tensor:
        return self.decoder(z, T)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor, training: bool):
        mu, logvar = self.encode(feats, mask)
        z = self.reparam(mu, logvar, training)
        pred = self.decode(z, feats.shape[1])
        return pred, mu, logvar, z


def build_vae_g1(
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
) -> Tuple[MotionVAE, TMRMotionRep]:
    """Construct (vae, motion_rep) for training."""
    skeleton = G1Skeleton34()
    motion_rep = TMRMotionRep(skeleton=skeleton, fps=fps, stats_path=stats_path)

    encoder = ACTORStyleEncoder(
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

    decoder = ACTORStyleDecoder(
        nfeats=motion_rep.motion_rep_dim,
        latent_dim=latent_dim,
        ff_size=ff_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        activation=activation,
    ).to(device)

    vae_model = MotionVAE(encoder=encoder, decoder=decoder, vae=vae).to(device)
    return vae_model, motion_rep


__all__ = ["build_vae_g1", "MotionVAE", "ACTORStyleDecoder"]

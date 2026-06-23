"""In-training reconstruction evaluator for the motion VAE.

Counterpart of `tmr_g1.training.inline_eval.GroupEvaluator`, but instead of
text->motion retrieval it measures, on a held-out testsuite group:

  - recon_mse   : masked L2 reconstruction error (normalized feature units)
  - kl          : mean KL(q(z|motion) || N(0,I))
  - mu_std      : mean per-dim std of the posterior means across the eval set
                  (a proxy for how much of the latent space the encoder uses;
                  near-0 => posterior collapse / poor motion-space coverage)
  - active_units: # latent dims whose mu-std exceeds 1e-2 (effective dimension)

mu_std / active_units are the signals that matter when the encoder is meant to
double as a feature extractor for FID: a collapsed (tightly-clustered) latent
gives misleadingly small FID regardless of motion quality.

Motion features are precomputed ONCE at construction; each `evaluate(vae)` only
re-runs the (cheap) encoder + decoder.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch


class ReconEvaluator:
    def __init__(self, testsuite, g1_npz_root, motion_rep,
                 split="content", grp="overview", fps=20, canonicalize=True,
                 device="cuda", max_cases=0):
        self.device = device

        group_dir = Path(testsuite) / split / "text2motion" / grp
        scale = fps / 30.0  # crop indices are 30 fps; NPZs are `fps`
        feats_list = []
        miss = 0
        for cdir in sorted(glob.glob(str(group_dir / "*"))):
            cdir = Path(cdir)
            sp = cdir / "seed_motion.json"
            if not sp.is_file():
                continue
            seed = json.load(open(sp))
            bvh = seed.get("bvh_path", "")
            session = bvh.split("/")[1] if bvh else ""
            npz = Path(g1_npz_root) / session / f"{seed['move_name']}.npz"
            if not npz.is_file():
                miss += 1
                continue
            try:
                with np.load(npz, allow_pickle=False) as d:
                    posed = np.asarray(d["posed_joints"]).astype(np.float32)
            except Exception:  # noqa: BLE001
                miss += 1
                continue
            n = posed.shape[0]
            a = max(0, int(round(seed["crop_start_frame_index"] * scale)))
            b = min(n, int(round(seed["crop_end_frame_index"] * scale)))
            if b - a < 4:
                miss += 1
                continue
            clip = posed[a:b]
            with torch.no_grad():
                f = motion_rep(
                    posed_joints=torch.from_numpy(clip).unsqueeze(0),
                    to_normalize=True, to_canonicalize=canonicalize,
                    lengths=torch.tensor([clip.shape[0]]),
                )[0]  # (T, D)
            feats_list.append(f.float())
            if max_cases and len(feats_list) >= max_cases:
                break

        self.n = len(feats_list)
        self.miss = miss
        T_max = max(f.shape[0] for f in feats_list)
        D = feats_list[0].shape[1]
        self.feats = torch.zeros(self.n, T_max, D)
        self.mask = torch.zeros(self.n, T_max, dtype=torch.bool)
        for i, f in enumerate(feats_list):
            self.feats[i, : f.shape[0]] = f
            self.mask[i, : f.shape[0]] = True

    @torch.no_grad()
    def evaluate(self, vae, chunk=128):
        was_training = vae.training
        vae.eval()
        dev = self.device
        sq_sum = 0.0
        n_elem = 0.0
        kl_sum = 0.0
        mus = []
        for s in range(0, self.n, chunk):
            f = self.feats[s:s + chunk].to(dev)
            msk = self.mask[s:s + chunk].to(dev)
            mu, logvar = vae.encode(f, msk)
            pred = vae.decode(mu, f.shape[1])  # deterministic recon at eval
            diff = (pred - f).pow(2).sum(-1)   # (b, T)
            m = msk.float()
            sq_sum += float((diff * m).sum().item())
            n_elem += float(m.sum().item())
            kl_sum += float(
                (0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1.0 - logvar, dim=-1)).sum().item()
            )
            mus.append(mu.cpu())

        if was_training:
            vae.train()

        mu_all = torch.cat(mus, dim=0).numpy()           # (N, latent)
        per_dim_std = mu_all.std(0)                       # (latent,)
        recon_mse = sq_sum / max(1.0, n_elem)
        return {
            "recon_mse": recon_mse,
            "kl": kl_sum / max(1, self.n),
            "mu_std": float(per_dim_std.mean()),
            "active_units": int((per_dim_std > 1e-2).sum()),
        }

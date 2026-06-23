"""
VAE-G1 reconstruction eval on the kimodo text2motion testsuite.

Counterpart of `tmr_g1.eval_retrieval`. For each group (content|repetition x
overview|timeline_single|timeline_multi) it loads the GT G1 motions (unified
NPZ, cropped to the test window), runs them through the trained motion VAE, and
reports:

  - recon_mse    : masked L2 reconstruction error (normalized feature units)
  - recon_rmse   : sqrt of the above (same units, easier to read)
  - kl           : mean KL(q(z|motion) || N(0,I))
  - mu_std       : mean per-dim std of posterior means across the pool
  - active_units : # latent dims with mu-std > 1e-2 (effective dimension)

mu_std / active_units quantify how much of the latent space the encoder uses —
the property that determines whether the encoder is a usable FID feature
extractor (a collapsed latent gives a misleadingly small FID).

Encode motions EXACTLY as training did: `motion_rep(posed_joints,
to_normalize=True, to_canonicalize=True)`, then the VAE encoder mean (mu).

Usage:
    python vae_g1/eval_recon.py \
        --ckpt runs/vae_g1/v0/last.pt \
        --stats-path data/bones_seed/tmr_g1_stats_v3 \
        --g1-npz-root data/bones_seed/g1_unified_npz \
        --testsuite data/kimodo_benchmark/testsuite \
        --out runs/vae_g1/v0/recon.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch

from vae_g1.model.vae_model import build_vae_g1

GROUPS = ["overview", "timeline_single", "timeline_multi"]
SPLITS = ["content", "repetition"]


def load_model(args):
    vae, motion_rep = build_vae_g1(
        stats_path=args.stats_path,
        fps=args.fps,
        latent_dim=args.latent_dim,
        ff_size=args.ff_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        device=args.device,
    )
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    vae.encoder.load_state_dict(ckpt["motion_encoder"])
    vae.decoder.load_state_dict(ckpt["motion_decoder"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, motion_rep


def load_cases(group_dir: Path):
    cases = []
    for cdir in sorted(glob.glob(str(group_dir / "*"))):
        cdir = Path(cdir)
        sp = cdir / "seed_motion.json"
        if not sp.is_file():
            continue
        seed = json.load(open(sp))
        bvh = seed.get("bvh_path", "")
        session = bvh.split("/")[1] if bvh else ""
        cases.append({
            "case_id": cdir.name,
            "move_name": seed["move_name"],
            "session": session,
            "crop_start": int(seed["crop_start_frame_index"]),
            "crop_end": int(seed["crop_end_frame_index"]),
        })
    return cases


@torch.no_grad()
def eval_group(vae, motion_rep, args, split, grp):
    group_dir = Path(args.testsuite) / split / "text2motion" / grp
    cases = load_cases(group_dir)
    npz_root = Path(args.g1_npz_root)
    scale = args.fps / 30.0  # crop indices are 30 fps; our NPZs are args.fps
    dev = args.device

    sq_sum = 0.0
    n_elem = 0.0
    kl_sum = 0.0
    mus = []
    kept = miss = 0
    for c in cases:
        npz = npz_root / c["session"] / f"{c['move_name']}.npz"
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
        a = max(0, int(round(c["crop_start"] * scale)))
        b = min(n, int(round(c["crop_end"] * scale)))
        if b - a < 4:
            miss += 1
            continue
        clip = posed[a:b]
        T = clip.shape[0]
        feats = motion_rep(
            posed_joints=torch.from_numpy(clip).unsqueeze(0).to(dev),
            to_normalize=True, to_canonicalize=args.canonicalize,
            lengths=torch.tensor([T], device=dev),
        )  # (1, T, D)
        mask = torch.ones(1, T, dtype=torch.bool, device=dev)
        mu, logvar = vae.encode(feats, mask)
        pred = vae.decode(mu, T)
        diff = (pred - feats).pow(2).sum(-1)  # (1, T)
        sq_sum += float(diff.sum().item())
        n_elem += float(T)
        kl_sum += float(
            (0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1.0 - logvar, dim=-1)).sum().item()
        )
        mus.append(mu.squeeze(0).cpu())
        kept += 1

    mu_all = torch.stack(mus, dim=0).numpy()  # (kept, latent)
    per_dim_std = mu_all.std(0)
    recon_mse = sq_sum / max(1.0, n_elem)
    return {
        "recon_mse": recon_mse,
        "recon_rmse": float(np.sqrt(recon_mse)),
        "kl": kl_sum / max(1, kept),
        "mu_std": float(per_dim_std.mean()),
        "active_units": int((per_dim_std > 1e-2).sum()),
        "pool": kept,
        "missing": miss,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stats-path", required=True)
    p.add_argument("--g1-npz-root", required=True)
    p.add_argument("--testsuite", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--ff-size", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--canonicalize", action="store_true", default=True)
    p.add_argument("--no-canonicalize", dest="canonicalize", action="store_false")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    vae, motion_rep = load_model(args)

    results = {}
    print(f"\n{'group':<32} {'pool':>6} {'recon_mse':>10} {'rmse':>8} "
          f"{'kl':>8} {'mu_std':>8} {'act_dim':>8}")
    print("-" * 90)
    for split in SPLITS:
        for grp in GROUPS:
            r = eval_group(vae, motion_rep, args, split, grp)
            key = f"{split}/{grp}"
            results[key] = r
            print(f"{key:<32} {r['pool']:>6} {r['recon_mse']:>10.4f} "
                  f"{r['recon_rmse']:>8.4f} {r['kl']:>8.2f} {r['mu_std']:>8.4f} "
                  f"{r['active_units']:>8}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"results": results, "ckpt": args.ckpt}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

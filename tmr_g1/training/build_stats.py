"""Compute per-feature mean/std for TMR-G1 normalization.

Streams a random sample of train motions through `TMRMotionRep(G1Skeleton30)`
and dumps the split-layout stats expected by `MotionRepBase`:

    <out>/global_root/{mean,std}.npy
    <out>/local_root/{mean,std}.npy
    <out>/body/{mean,std}.npy
    <out>/{mean,std}.npy      # global concat, for parity with kimodo's TMR

Usage:
    python tmr_g1/training/build_stats.py \
        --data-root data/bones_seed/g1_20fps_npz \
        --train-split data/kimodo_benchmark/splits/train_split_paths.txt \
        --out data/bones_seed/tmr_g1_stats \
        --n-motions 5000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from kimodo.skeleton import G1Skeleton34
from kimodo.motion_rep.reps.tmr_motionrep import TMRMotionRep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--train-split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--n-motions", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=200)  # 10s @ 20fps
    p.add_argument("--canonicalize", action="store_true", default=False,
                   help="build stats on canonicalized features (must match training)")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    data_root = Path(args.data_root)
    with open(args.train_split) as f:
        rels = [ln.strip() for ln in f if ln.strip()]
    rng.shuffle(rels)
    skel = G1Skeleton34()
    mr = TMRMotionRep(skeleton=skel, fps=args.fps, stats_path=None)

    n_taken = 0
    n_global_root = mr.global_root_dim
    n_local_root = mr.local_root_dim
    n_body = mr.body_dim
    n_total = mr.motion_rep_dim
    # Welford for global (cat) only is enough; split blocks slice from global.
    # But the loader (`_require_split_stats_layout`) requires each split to be
    # present, so we compute them from the global vector via the rep's slices.

    s = torch.zeros(n_total, dtype=torch.float64)
    ss = torch.zeros(n_total, dtype=torch.float64)
    count = 0

    for rel in rels:
        if n_taken >= args.n_motions:
            break
        npz_path = data_root / (rel + ".npz")
        if not npz_path.is_file():
            continue
        with np.load(npz_path, mmap_mode="r") as d:
            posed = np.asarray(d["posed_joints"]).astype(np.float32)
        T = posed.shape[0]
        if T < 20:
            continue
        if T > args.max_frames:
            offset = int(rng.integers(0, T - args.max_frames + 1))
            posed = posed[offset:offset + args.max_frames]
            T = posed.shape[0]
        with torch.no_grad():
            feats = mr(
                posed_joints=torch.from_numpy(posed).unsqueeze(0),  # (1, T, J, 3)
                to_normalize=False,
                to_canonicalize=args.canonicalize,
                lengths=torch.tensor([T]),
            )[0]  # (T, n_total)
        s += feats.double().sum(0)
        ss += feats.double().pow(2).sum(0)
        count += T
        n_taken += 1
        if n_taken % 200 == 0:
            print(f"[stats] {n_taken}/{args.n_motions}  frames={count}")

    mean = (s / count).float()
    var = (ss / count).float() - mean.pow(2)
    std = var.clamp(min=1e-8).sqrt()
    print(f"[stats] {n_taken} motions, {count} frames, dim={n_total}")

    out = Path(args.out)
    (out / "global_root").mkdir(parents=True, exist_ok=True)
    (out / "local_root").mkdir(parents=True, exist_ok=True)
    (out / "body").mkdir(parents=True, exist_ok=True)

    # global concat
    np.save(out / "mean.npy", mean.numpy())
    np.save(out / "std.npy", std.numpy())

    # split layout: global_root + body together cover the full vector.
    np.save(out / "global_root" / "mean.npy", mean[:n_global_root].numpy())
    np.save(out / "global_root" / "std.npy",  std[:n_global_root].numpy())
    np.save(out / "body" / "mean.npy", mean[n_global_root:n_global_root + n_body].numpy())
    np.save(out / "body" / "std.npy",  std[n_global_root:n_global_root + n_body].numpy())

    # local_root is a sub-rep used in some codepaths; we don't have it in the
    # TMRMotionRep feature concat, but the stats loader expects the files.
    # Reuse global_root's stats with zero-padding to local_root_dim (it's never
    # consumed at retrieval time).
    pad_mean = torch.zeros(n_local_root)
    pad_std = torch.ones(n_local_root)
    np.save(out / "local_root" / "mean.npy", pad_mean.numpy())
    np.save(out / "local_root" / "std.npy",  pad_std.numpy())

    print(f"[stats] wrote {out}")


if __name__ == "__main__":
    main()

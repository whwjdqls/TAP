"""Compute flat per-dim Mean/Std over precomputed g1_rep_v1 feature files.

Streams every <feat-root>/**/*.npy, accumulates float64 count/sum/sumsq per
feature dim, writes Mean.npy / Std.npy (shape (136,)). Constant dims (std <
1e-6) are set to std=1 so normalization leaves them as (x-mean).

    python compute_stats.py --feat-root /home/jungbin_cho/seed/g1_rep_v1_feats \
        --out-dir /home/jungbin_cho/seed/g1_rep_v1_stats
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "/home/jungbin_cho/TAP/kinematic_planner")
import g1_rep_v1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat-root", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    D = g1_rep_v1.FEAT_DIM
    n = 0
    s = np.zeros(D, np.float64)
    ss = np.zeros(D, np.float64)
    files = sorted(Path(args.feat_root).rglob("*.npy"))
    print(f"[stats] {len(files)} feature files", flush=True)
    for i, f in enumerate(files):
        x = np.load(f).astype(np.float64)            # (T, D)
        if x.ndim != 2 or x.shape[1] != D:
            continue
        n += x.shape[0]
        s += x.sum(0)
        ss += (x * x).sum(0)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(files)} frames={n}", flush=True)

    mean = (s / n).astype(np.float32)
    var = (ss / n) - (s / n) ** 2
    var = np.clip(var, 0, None)
    std = np.sqrt(var).astype(np.float32)
    std[std < 1e-6] = 1.0

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "Mean.npy", mean)
    np.save(out / "Std.npy", std)
    print(f"[stats] frames={n} -> {out}/Mean.npy,Std.npy", flush=True)
    # per-block summary
    for name, sl in g1_rep_v1.feature_layout().items():
        print(f"   {name:14s} mean|{np.abs(mean[sl]).mean():.4f}  std~{std[sl].mean():.4f}", flush=True)


if __name__ == "__main__":
    main()

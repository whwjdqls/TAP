"""Render g1_rep_v1 samples with kimodo's training-viz renderer (render_soma).

Uses kimodo.scripts.render_soma.render_single (floor + grid + tracking camera +
pelvis trail), the same renderer train.py uses for its viz — skeleton-agnostic
via joint_parents, pure matplotlib/CPU (works headless on this cluster).

    python render_samples_kimodo.py --samples-dir runs/.../samples --fps 20
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

os.chdir("/tmp")  # kimodo namespace guard
sys.path.insert(0, "/home/jungbin_cho/TAP/kinematic_planner")

import numpy as np
import imageio.v3 as iio
from kimodo.scripts.render_soma import render_single
from kimodo.skeleton import G1Skeleton34


def g1_joint_parents():
    s = G1Skeleton34()
    names = [str(n) for n in s.bone_order_names]
    idx = {n: i for i, n in enumerate(names)}
    bp = s.bone_parents
    if isinstance(bp, (list, tuple)):
        bp = bp[0]
    return [idx[bp[n]] if bp.get(n) else -1 for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default="/home/jungbin_cho/TAP/runs/mdm_g1_rep_v1_full/samples")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--which", default="joints_exec", choices=["joints_exec", "joints_ric"])
    ap.add_argument("--camera", default="fixed", choices=["fixed", "follow"])
    args = ap.parse_args()

    jp = g1_joint_parents()
    for f in sorted(glob.glob(f"{args.samples_dir}/*.npz")):
        z = np.load(f, allow_pickle=True)
        joints = np.asarray(z[args.which], dtype=np.float32)        # (T,34,3) Y-up
        frames = render_single(
            joints, jp, caption=str(z["prompt"]), color="tab:red",
            camera=args.camera, width=700, height=700, dpi=120, max_frames=None,
        )
        out = Path(f).with_suffix(".mp4")
        iio.imwrite(out, frames, fps=args.fps, codec="h264", plugin="pyav")
        print(f"  wrote {out.name}  ({frames.shape[0]} frames)", flush=True)
    print("done")


if __name__ == "__main__":
    main()

"""Render g1_rep_v1 sample npzs to mp4 — nicer 3D skeleton (matplotlib, no GL).

Thick colored limbs + joint spheres, a ground plane with grid, a drop shadow,
and a fading root trail. Pure CPU; works headless without any OpenGL.

    python render_samples_mpl.py --samples-dir runs/.../samples --fps 20
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import imageio.v3 as iio3


def _skeleton_edges():
    os.chdir("/tmp")
    from kimodo.skeleton import G1Skeleton34
    s = G1Skeleton34()
    names = [str(n) for n in s.bone_order_names]
    idx = {n: i for i, n in enumerate(names)}
    bp = s.bone_parents[0] if isinstance(s.bone_parents, (list, tuple)) else s.bone_parents
    edges, side = [], []
    for child, parent in bp.items():
        if parent is None or child not in idx or parent not in idx:
            continue
        edges.append((idx[parent], idx[child]))
        if "left" in child:
            side.append("L")
        elif "right" in child:
            side.append("R")
        else:
            side.append("C")
    return edges, side


COL = {"L": "#2a6fd6", "R": "#d6452a", "C": "#2b2b2b"}


def _draw_frame(ax, p, edges, side, cx, cz, rng, root_trail):
    # ground plane + grid
    g = np.linspace(-rng, rng, 9)
    for v in g:
        ax.plot([cx - rng, cx + rng], [cz + v, cz + v], [0, 0], color="#e3e3e3", lw=0.6, zorder=0)
        ax.plot([cx + v, cx + v], [cz - rng, cz + rng], [0, 0], color="#e3e3e3", lw=0.6, zorder=0)
    # drop shadow (joints projected to floor)
    ax.scatter(p[:, 0], p[:, 2], np.zeros(len(p)), s=18, c="#cccccc", alpha=0.5, zorder=1, depthshade=False)
    segs_sh = [[(p[a, 0], p[a, 2], 0.0), (p[b, 0], p[b, 2], 0.0)] for a, b in edges]
    ax.add_collection3d(Line3DCollection(segs_sh, colors="#d0d0d0", linewidths=3, zorder=1))
    # root trail
    if len(root_trail) > 1:
        rt = np.asarray(root_trail)
        ax.plot(rt[:, 0], rt[:, 2], np.zeros(len(rt)), color="#9bbcf0", lw=1.5, zorder=2)
    # limbs (thick) + joints
    segs = [[(p[a, 0], p[a, 2], p[a, 1]), (p[b, 0], p[b, 2], p[b, 1])] for a, b in edges]
    cols = [COL[s] for s in side]
    ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=5.5, zorder=5))
    ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=26, c="#111111", zorder=6, depthshade=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default="/home/jungbin_cho/TAP/runs/mdm_g1_rep_v1_full/samples")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--which", default="joints_exec", choices=["joints_exec", "joints_ric"])
    args = ap.parse_args()

    edges, side = _skeleton_edges()

    for f in sorted(glob.glob(f"{args.samples_dir}/*.npz")):
        z = np.load(f, allow_pickle=True)
        J = np.asarray(z[args.which], dtype=np.float32)      # (T,34,3) Y-up
        prompt = str(z["prompt"])
        T = J.shape[0]
        X, Z = J[..., 0], J[..., 2]
        cx, cz = (X.min() + X.max()) / 2, (Z.min() + Z.max()) / 2
        rng = max(X.max() - X.min(), Z.max() - Z.min(), 0.8) * 0.55 + 0.6
        frames = []
        fig = plt.figure(figsize=(5, 5), dpi=120)
        for t in range(T):
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor("white")
            _draw_frame(ax, J[t], edges, side, cx, cz, rng, J[:t + 1, 0])
            ax.set_xlim(cx - rng, cx + rng); ax.set_ylim(cz - rng, cz + rng); ax.set_zlim(0, 1.9)
            ax.set_box_aspect((2 * rng, 2 * rng, 1.9))
            ax.view_init(elev=8, azim=-65)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.grid(False)
            try:
                ax.xaxis.pane.set_visible(False); ax.yaxis.pane.set_visible(False); ax.zaxis.pane.set_visible(False)
            except Exception:
                pass
            ax.set_title(f"{prompt}", fontsize=10, color="#222")
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
            frames.append(img.copy())
            fig.clf()
        plt.close(fig)
        out = Path(f).with_suffix(".mp4")
        iio3.imwrite(out, np.stack(frames), fps=args.fps, codec="h264", plugin="pyav")
        print(f"  wrote {out.name}  ({T} frames)", flush=True)
    print("done")


if __name__ == "__main__":
    main()

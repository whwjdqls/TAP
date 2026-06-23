"""Render g1_rep_v1 sample npzs (their MuJoCo `qpos`) to mp4 videos.

Offscreen EGL render of the G1 in MuJoCo, one mp4 per sample. Run on a GPU
node. Loads the model once and renders all samples.

    python render_samples.py --samples-dir runs/.../samples --fps 20
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import imageio.v2 as imageio
import mujoco

XML = "/home/jungbin_cho/TAP/kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default="/home/jungbin_cho/TAP/runs/mdm_g1_rep_v1_full/samples")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--xml", default=XML)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.azimuth, cam.elevation = 3.2, 135.0, -10.0
    print(f"model.nq={model.nq}")

    files = sorted(glob.glob(f"{args.samples_dir}/*.npz"))
    for f in files:
        z = np.load(f, allow_pickle=True)
        qpos = np.asarray(z["qpos"], dtype=np.float64)        # (T,36)
        if qpos.shape[1] != model.nq:
            print(f"  SKIP {Path(f).name}: qpos cols {qpos.shape[1]} != nq {model.nq}")
            continue
        out = Path(f).with_suffix(".mp4")
        w = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                               macro_block_size=1)
        for frame in qpos:
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            cam.lookat[:] = frame[:3]
            renderer.update_scene(data, camera=cam)
            w.append_data(renderer.render())
        w.close()
        print(f"  wrote {out.name}  ({qpos.shape[0]} frames)")
    renderer.close()
    print("done")


if __name__ == "__main__":
    main()

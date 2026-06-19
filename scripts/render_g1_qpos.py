"""
Headless mp4 renderer for a G1 qpos CSV (kimodo / SONIC / SOMA-retarget output).

Loads `g1skel34/xml/g1.xml` from the kimodo assets, replays `qpos.csv`
frame-by-frame using `mujoco.Renderer` (offscreen), and writes an mp4.

Usage:
    python scripts/render_g1_qpos.py --csv runs/kimodo_out/walk.csv \
        --out runs/kimodo_out/walk.mp4 --fps 30
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import mujoco

# Default XML lives inside the kimodo checkout. Hardcode the absolute path so
# this script can run from any env (env_isaaclab doesn't have kimodo).
_DEFAULT_XML = "/nfsdata/home/jungbin.cho/TAP/kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--xml", default=_DEFAULT_XML,
                        help="MuJoCo XML for the G1 skeleton.")
    parser.add_argument("--camera_dist", type=float, default=3.5)
    parser.add_argument("--camera_az", type=float, default=135.0)
    parser.add_argument("--camera_el", type=float, default=-10.0)
    args = parser.parse_args()

    qpos = np.loadtxt(args.csv, delimiter=",")
    if qpos.ndim == 1:
        qpos = qpos[None]
    print(f"[render] qpos shape: {qpos.shape}  xml: {args.xml}")

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    if qpos.shape[1] != model.nq:
        raise ValueError(f"CSV has {qpos.shape[1]} qpos cols but model.nq={model.nq}")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = args.camera_dist
    cam.azimuth = args.camera_az
    cam.elevation = args.camera_el

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)

    for i, frame in enumerate(qpos):
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        cam.lookat[:] = frame[:3]
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        writer.append_data(img)

    writer.close()
    renderer.close()
    print(f"[render] wrote {args.out}  ({qpos.shape[0]} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()

"""
Inverse of `kimodo_csv_to_bones_seed.py`: take a Bones-SEED-format CSV (header
row, root pos in cm, root rot as Euler-XYZ deg, 29 joint angles in deg) and
emit a headerless MuJoCo G1 qpos CSV (m, quat-wxyz, rad) that `render_g1_qpos.py`
can play back.

Usage:
    python scripts/bones_seed_to_qpos.py --in <bones.csv> --out <qpos.csv>
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.inp)
    root_pos_m = np.stack([df["root_translateX"], df["root_translateY"], df["root_translateZ"]], axis=1) / 100.0
    euler_deg = np.stack([df["root_rotateX"], df["root_rotateY"], df["root_rotateZ"]], axis=1)
    root_quat_xyzw = Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat()
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    joint_cols = [c for c in df.columns if c.endswith("_dof")]
    assert len(joint_cols) == 29, f"got {len(joint_cols)} joints"
    joint_rad = np.deg2rad(df[joint_cols].values)
    qpos = np.concatenate([root_pos_m, root_quat_wxyz, joint_rad], axis=1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savetxt(args.out, qpos, delimiter=",", fmt="%.6f")
    print(f"[bones->qpos] wrote {args.out}  ({qpos.shape[0]} frames, {qpos.shape[1]} cols)")


if __name__ == "__main__":
    main()

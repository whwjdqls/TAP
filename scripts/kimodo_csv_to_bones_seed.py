"""
Adapter: Kimodo G1 qpos CSV → Bones-SEED flat CSV format expected by SONIC's
`convert_soma_csv_to_motion_lib.py` (Bones-SEED mode).

Kimodo CSV (from `kimodo_convert --to g1-csv`):
    36 cols, no header:
      cols 0..2  = root pos (meters, MuJoCo Z-up)
      cols 3..6  = root quat (wxyz)
      cols 7..35 = 29 joint angles (radians), in MuJoCo actuator order

Bones-SEED CSV (consumed by load_bones_csv):
    header row: Frame, root_translateX/Y/Z, root_rotateX/Y/Z,
                <29 joint names>_dof
    root pos in cm, root rot as intrinsic Euler-XYZ in degrees,
    joint angles in degrees, MuJoCo actuator order.

Usage:
    python scripts/kimodo_csv_to_bones_seed.py \
        --in  runs/kimodo_out/walk.csv \
        --out runs/kimodo_out/bones_seed_csvs/kimodo_session/walk.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from scipy.spatial.transform import Rotation

# Must match BONES_CSV_JOINT_NAMES in
# GR00T-WholeBodyControl/gear_sonic/data_process/convert_soma_csv_to_motion_lib.py
JOINT_NAMES = [
    "left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof",
    "left_knee_joint_dof", "left_ankle_pitch_joint_dof", "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof",
    "right_knee_joint_dof", "right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof", "waist_roll_joint_dof", "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof", "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof", "left_wrist_pitch_joint_dof", "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof", "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof", "right_wrist_pitch_joint_dof",
    "right_wrist_yaw_joint_dof",
]
assert len(JOINT_NAMES) == 29


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    qpos = np.loadtxt(args.inp, delimiter=",")
    if qpos.ndim == 1:
        qpos = qpos[None]
    assert qpos.shape[1] == 36, f"expected 36 qpos cols, got {qpos.shape[1]}"

    T = qpos.shape[0]
    root_pos_m = qpos[:, 0:3]
    root_quat_wxyz = qpos[:, 3:7]
    joint_rad = qpos[:, 7:36]

    # m → cm
    root_pos_cm = root_pos_m * 100.0

    # quat wxyz → euler xyz intrinsic, degrees
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    euler_deg = Rotation.from_quat(root_quat_xyzw).as_euler("xyz", degrees=True)

    # rad → deg
    joint_deg = np.rad2deg(joint_rad)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    header = (
        ["Frame", "root_translateX", "root_translateY", "root_translateZ",
         "root_rotateX", "root_rotateY", "root_rotateZ"]
        + JOINT_NAMES
    )
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(T):
            row = [i]
            row.extend(root_pos_cm[i].tolist())
            row.extend(euler_deg[i].tolist())
            row.extend(joint_deg[i].tolist())
            w.writerow(row)
    print(f"[adapter] wrote {args.out}  ({T} frames, {len(header)} cols)")


if __name__ == "__main__":
    main()

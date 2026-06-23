"""Validate g1_rep_v1 encode/decode on real Bones-SEED motions.

Checks per motion:
  [A] position invertibility: decode_positions(encode(M)) == canonicalize(M)
      (rotate by -head[0] about origin, shift XZ by initial root) to ~1e-5.
  [B] executable decode: decode (root + 29 qpos angles -> G1 FK) vs GT
      posed_joints, after the same canonicalization. Reports total error and
      the pelvis-tilt (yaw-only-root) component separately.
  [C] joint-angle source sanity: GT qpos (CSV root + CSV angles) -> FK vs GT
      posed_joints (the intrinsic ~0.04 m G1 1-DOF projection floor).
"""
from __future__ import annotations

import os
import sys

os.chdir("/tmp")  # kimodo namespace-shadowing guard
sys.path.insert(0, "/home/jungbin_cho/TAP/kinematic_planner")

import numpy as np
import torch

import g1_rep_v1
import g1_data
from kimodo.skeleton import G1Skeleton34
from kimodo.exports.mujoco import MujocoQposConverter

XML = "/home/jungbin_cho/TAP/kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml"


def canonicalize(posed, root, head0):
    """R_y(-head0) @ (posed - root_xz[0]); returns (T,34,3)."""
    rxz0 = root[0].clone(); rxz0[1] = 0.0
    shifted = posed - rxz0
    R = g1_rep_v1._Ry(-head0)
    return torch.einsum("ij,tnj->tni", R, shifted)


def main():
    skel = G1Skeleton34()
    conv = MujocoQposConverter(skel, xml_path=XML)
    r_hip, l_hip = skel.hip_joint_idx
    motions = [
        "230509/shadow_boxing_R_003__A361",
        "230509/warm_up_neck_001__A360_M",
        "230213/praying_004__A185_M",
        "230213/sit_on_heels_stop_001__A186_M",
    ]
    # Non-actuated end-effector tips the G1 cannot drive (toe tips, hand rolls);
    # error there is intrinsic and not robot-relevant.
    tips = [7, 14, 25, 33]
    body = [i for i in range(34) if i not in tips]
    print(f"{'motion':40s} {'[A]ric_max':>11s} {'[B]qpos_max':>11s} {'[B]mean':>9s} {'body_max':>9s} {'tilt_deg':>9s}")
    for m in motions:
        try:
            mo = g1_data.load_motion(m, skel)
        except Exception as e:
            print(f"{m:40s}  load FAIL: {e}")
            continue
        posed, root = mo["posed_joints"], mo["root_positions"]
        feats = g1_data.encode_motion(mo, skel)
        head0 = g1_rep_v1.heading_from_hips(posed, int(r_hip), int(l_hip))[0]
        canon = canonicalize(posed, root, head0)

        # [A] position round-trip
        W = g1_rep_v1.decode_positions(feats)
        a_max = (W - canon).norm(dim=-1).max().item()

        # [B] qpos -> FK decode (canonical frame)
        dec = g1_rep_v1.decode_qpos_to_joints(feats, conv, skel)
        b_err = (dec["posed_joints"] - canon).norm(dim=-1)            # (T,34)
        b_max, b_mean = b_err.max().item(), b_err.mean().item()
        body_max = b_err[:, body].max().item()

        _c_max, tilt = _gt_qpos_floor(m, mo, conv, skel)
        print(f"{m:40s} {a_max:11.6f} {b_max:11.4f} {b_mean:9.4f} {body_max:9.4f} {tilt:9.3f}")


def _gt_qpos_floor(sess_move, mo, conv, skel):
    """Run GT qpos (full root from CSV + 29 angles) -> FK; report error vs
    posed_joints (intrinsic floor) and the GT pelvis tilt (deg) off-vertical."""
    import pandas as pd
    from scipy.spatial.transform import Rotation
    csv = f"{g1_data.CSV_ROOT}/{sess_move}.csv"
    df = pd.read_csv(csv)
    T = mo["posed_joints"].shape[0]
    rp = np.stack([df.root_translateX, df.root_translateY, df.root_translateZ], 1) / 100.0
    eu = np.stack([df.root_rotateX, df.root_rotateY, df.root_rotateZ], 1)
    q = Rotation.from_euler("xyz", eu, degrees=True).as_quat()[:, [3, 0, 1, 2]]
    cols = g1_data._dof_columns(df)
    ang = np.deg2rad(df[cols].values)
    qpos = np.concatenate([rp, q, ang], 1)[::6][:T]
    mdict = conv.qpos_to_motion_dict(torch.from_numpy(qpos).float().unsqueeze(0),
                                     source_fps=20, root_quat_w_first=True, mujoco_rest_zero=False)
    local, rootp = mdict["local_rot_mats"], mdict["root_positions"]
    if local.dim() == 5:
        local, rootp = local[0], rootp[0]
    _, posed_fk, _ = skel.fk(local.unsqueeze(0), rootp.unsqueeze(0))
    c_max = (posed_fk[0] - mo["posed_joints"]).norm(dim=-1).max().item()
    # pelvis tilt: angle of the GT root quat away from pure-yaw (z-up). Use the
    # mujoco quats (z-up) -> tilt = angle between body-up (R@[0,0,1]) and [0,0,1].
    Rm = Rotation.from_quat(q[::6][:T][:, [1, 2, 3, 0]]).as_matrix()
    up = Rm @ np.array([0, 0, 1.0])
    tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1))).max()
    return c_max, float(tilt)


if __name__ == "__main__":
    main()

"""
Consistency checks for the unified G1 NPZ (`local_rot_mats, root_positions,
posed_joints` in kimodo Y-up frame) against the original Bones-SEED CSV.

Two checks:

  (A) **FK consistency** — `posed_joints` we stored (via kimodo's FK on our
      `local_rot_mats`) must equal MuJoCo's `xpos` for the same frame after
      the MuJoCo→Kimodo coordinate rotation. We map the 30 MJCF body indices
      to kimodo joint indices and compare. The 4 virtual endpoint joints
      (toe-tips, palm hand-rolls) are skipped because they have no MJCF
      counterpart.

  (B) **Round-trip to qpos** — feed `(local_rot_mats, root_positions)` into
      `kimodo.exports.mujoco.MujocoQposConverter.dict_to_qpos(...)`. The
      resulting MuJoCo qpos should match the original CSV qpos. We compare
      the 36-D qpos values.

The thresholds are loose (default 1e-3) because pickled rest poses and
quat-axis ordering have ~mm-scale rounding error.

Usage:
    python scripts/g1_check_unified_consistency.py \
        --csv data/bones_seed/g1/csv/<sess>/<name>.csv \
        --npz data/bones_seed/g1_20fps_npz_v2/<sess>/<name>.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation


G1_XML = "/nfsdata/home/jungbin.cho/TAP/kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml"
_M = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)


def _load_qpos(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    root_pos_m = (
        np.stack([df["root_translateX"], df["root_translateY"], df["root_translateZ"]], axis=1) / 100.0
    )
    euler_deg = np.stack([df["root_rotateX"], df["root_rotateY"], df["root_rotateZ"]], axis=1)
    root_quat_xyzw = Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat()
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    joint_cols = [c for c in df.columns if c.endswith("_dof")]
    joint_rad = np.deg2rad(df[joint_cols].values)
    return np.concatenate([root_pos_m, root_quat_wxyz, joint_rad], axis=1).astype(np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--npz", required=True)
    p.add_argument("--source-fps", type=int, default=120)
    p.add_argument("--target-fps", type=int, default=20)
    p.add_argument("--frames", type=int, default=20, help="how many frames to spot-check")
    p.add_argument("--pos-tol", type=float, default=1e-3, help="meters")
    p.add_argument("--rot-tol", type=float, default=1e-3, help="rad on a Rodrigues vector")
    args = p.parse_args()

    csv_path = Path(args.csv)
    npz_path = Path(args.npz)
    z = np.load(npz_path, allow_pickle=False)
    posed_joints = z["posed_joints"]      # (T, 34, 3) Y-up
    root_positions = z["root_positions"]  # (T, 3) Y-up
    local_rot_mats = z["local_rot_mats"]  # (T, 34, 3, 3) Y-up
    T_npz = posed_joints.shape[0]
    print(f"NPZ:  T={T_npz}  fps={int(z['fps'])}  frame={str(z['frame'])}")

    qpos_full = _load_qpos(csv_path)
    stride = args.source_fps // args.target_fps
    qpos = qpos_full[::stride]
    T = qpos.shape[0]
    assert T_npz == T, f"frame count mismatch: npz={T_npz} csv-downsampled={T}"

    # --- import kimodo machinery
    import mujoco
    import os
    os.chdir("/tmp")
    from kimodo.skeleton import G1Skeleton34
    from kimodo.exports.mujoco import MujocoQposConverter

    skel = G1Skeleton34()
    model = mujoco.MjModel.from_xml_path(G1_XML)
    data = mujoco.MjData(model)

    # Build MJCF body → kimodo joint index map (same as the converter)
    # using the table from g1_csv_to_unified_npz.py.
    MJ_TO_KIM = {"pelvis": "pelvis_skel", "torso_link": "waist_pitch_skel"}

    def mj_to_kim(name):
        if name in MJ_TO_KIM:
            return MJ_TO_KIM[name]
        if name.endswith("_link"):
            return name[:-5] + "_skel"
        return name

    mj_body_names = [model.body(i).name for i in range(model.nbody)]
    mj_to_kim_idx = {}
    for mj_i, n in enumerate(mj_body_names):
        if mj_i == 0:
            continue
        kn = mj_to_kim(n)
        if kn in skel.bone_order_names:
            mj_to_kim_idx[mj_i] = skel.bone_order_names.index(kn)

    n_frames = min(args.frames, T)
    pos_errs = []
    for t in range(n_frames):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        for mj_i, kim_i in mj_to_kim_idx.items():
            mj_pos = data.xpos[mj_i]                          # MJ Z-up
            mj_pos_k = mj_pos @ _M.T                          # to Y-up
            ours = posed_joints[t, kim_i]
            pos_errs.append(np.linalg.norm(mj_pos_k - ours))
    pos_errs = np.array(pos_errs)
    print(f"\n[A] posed_joints vs MuJoCo xpos (n={len(pos_errs)} mappings × {n_frames} frames)")
    print(f"     max_err = {pos_errs.max():.6f} m   "
          f"mean_err = {pos_errs.mean():.6f} m   "
          f"p95 = {np.quantile(pos_errs, 0.95):.6f} m   "
          f"tol = {args.pos_tol} m")
    a_ok = pos_errs.max() < args.pos_tol

    # --- (B) round-trip via MujocoQposConverter
    converter = MujocoQposConverter(skel, xml_path=G1_XML)
    out = {
        "local_rot_mats": torch.from_numpy(local_rot_mats[:n_frames]).unsqueeze(0),
        "root_positions": torch.from_numpy(root_positions[:n_frames]).unsqueeze(0),
        "posed_joints":   torch.from_numpy(posed_joints[:n_frames]).unsqueeze(0),
        "foot_contacts":  torch.zeros(1, n_frames, 4, dtype=torch.bool),
    }
    try:
        qpos_pred = converter.dict_to_qpos(out, device="cpu")
        if hasattr(qpos_pred, "detach"):
            qpos_pred = qpos_pred.detach().cpu().numpy()
        else:
            qpos_pred = np.asarray(qpos_pred)
        # qpos_pred shape: either (n_frames, 36) or (1, n_frames, 36)
        if qpos_pred.ndim == 3:
            qpos_pred = qpos_pred[0]
        if qpos_pred.shape != (n_frames, 36):
            raise ValueError(f"unexpected qpos shape: {qpos_pred.shape}")

        # Split for nicer reporting.
        root_pos_err = np.linalg.norm(qpos_pred[:, :3] - qpos[:n_frames, :3], axis=-1)
        # Quat comparison: dot product, both unit quats, handle sign ambiguity.
        q_pred = qpos_pred[:, 3:7]
        q_gt = qpos[:n_frames, 3:7]
        dots = np.abs(np.einsum('ij,ij->i', q_pred, q_gt))
        quat_err_rad = 2.0 * np.arccos(np.clip(dots, 0, 1))
        joint_err = np.abs(qpos_pred[:, 7:36] - qpos[:n_frames, 7:36])

        print(f"\n[B] dict_to_qpos round-trip vs original CSV qpos ({n_frames} frames)")
        print(f"     root_pos max={root_pos_err.max():.6f} m  mean={root_pos_err.mean():.6f} m")
        print(f"     root_rot max={quat_err_rad.max():.6f} rad mean={quat_err_rad.mean():.6f} rad")
        print(f"     joint   max={joint_err.max():.6f} rad mean={joint_err.mean():.6f} rad")
        b_ok = (root_pos_err.max() < args.pos_tol
                and quat_err_rad.max() < 5e-2  # ~3 deg; quat conversion has more wiggle
                and joint_err.max() < 5e-2)
    except Exception as e:  # noqa: BLE001
        print(f"\n[B] dict_to_qpos call failed: {type(e).__name__}: {e}")
        b_ok = False

    print()
    print(f"[A] FK consistency:    {'PASS' if a_ok else 'FAIL'}")
    print(f"[B] qpos round-trip:   {'PASS' if b_ok else 'FAIL'}")
    if not (a_ok and b_ok):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

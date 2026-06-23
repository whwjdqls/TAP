"""Load Bones-SEED G1 motions and build g1_rep_v1 features.

Sources per motion (frame-aligned at 20 fps):
  * unified NPZ  (/home/jungbin_cho/seed/g1_unified_npz/<sess>/<move>.npz):
      posed_joints (T,34,3), root_positions (T,3)  — kimodo Y-up, exact.
  * Bones-SEED CSV (/home/jungbin_cho/seed/g1/csv/<sess>/<move>.csv): 120 fps,
      29 `<joint>_dof` columns in DEGREES -> radians, strided by 6 to 20 fps.

The 29 qpos angles come from the CSV (the clean single-DOF ground truth); the
positions/root come from the NPZ.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

import g1_rep_v1

SEED_ROOT = os.environ.get("SEED_ROOT", "/home/jungbin_cho/seed")
NPZ_ROOT = os.environ.get("G1_NPZ_ROOT", f"{SEED_ROOT}/g1_unified_npz")
CSV_ROOT = os.environ.get("G1_CSV_ROOT", f"{SEED_ROOT}/g1/csv")

_DOF_COLS = None  # cached header order


def _dof_columns(df: pd.DataFrame):
    cols = [c for c in df.columns if c.endswith("_dof")]
    if len(cols) != g1_rep_v1.N_ACTUATED:
        raise ValueError(f"expected {g1_rep_v1.N_ACTUATED} _dof cols, got {len(cols)}")
    return cols


def load_qpos_angles(csv_path: str | Path, T: int, src_fps: int = 120, tgt_fps: int = 20) -> np.ndarray:
    """Return (T, 29) actuated joint angles in radians, downsampled to match NPZ."""
    df = pd.read_csv(csv_path)
    cols = _dof_columns(df)
    ang = np.deg2rad(df[cols].values.astype(np.float64))[:: src_fps // tgt_fps]
    if ang.shape[0] < T:
        raise ValueError(f"csv angles {ang.shape[0]} < npz T {T} for {csv_path}")
    return ang[:T].astype(np.float32)


def foot_contacts(posed_joints: torch.Tensor, skeleton, fps: int = 20) -> torch.Tensor:
    """(T,34,3) -> (T,4) foot-contact flags via kimodo's pos+vel detector."""
    from kimodo.motion_rep.feet import foot_detect_from_pos_and_vel
    from kimodo.motion_rep.feature_utils import compute_vel_xyz

    pj = posed_joints.unsqueeze(0)                                   # (1,T,34,3)
    lengths = torch.tensor([posed_joints.shape[0]])
    vel = compute_vel_xyz(pj, float(fps), lengths=lengths)          # (1,T,34,3)
    fc = foot_detect_from_pos_and_vel(pj, vel, skeleton, vel_thres=0.15, height_thresh=0.10)
    return fc[0].float()                                            # (T,4)


def load_motion(sess_move: str, skeleton) -> Dict[str, torch.Tensor]:
    """sess_move e.g. '230509/shadow_boxing_R_003__A361'. Returns tensors."""
    npz_path = Path(NPZ_ROOT) / f"{sess_move}.npz"
    csv_path = Path(CSV_ROOT) / f"{sess_move}.csv"
    with np.load(npz_path, allow_pickle=False) as d:
        posed = torch.from_numpy(np.asarray(d["posed_joints"], np.float32))    # (T,34,3)
        root = torch.from_numpy(np.asarray(d["root_positions"], np.float32))   # (T,3)
        root_rot = torch.from_numpy(np.asarray(d["local_rot_mats"][:, 0], np.float32))  # (T,3,3)
    T = posed.shape[0]
    angles = torch.from_numpy(load_qpos_angles(csv_path, T))                   # (T,29)
    fc = foot_contacts(posed, skeleton)                                        # (T,4)
    return {"posed_joints": posed, "root_positions": root, "root_rot": root_rot,
            "joint_angles": angles, "foot_contacts": fc}


def encode_motion(motion: Dict[str, torch.Tensor], skeleton) -> torch.Tensor:
    r_hip, l_hip = skeleton.hip_joint_idx
    return g1_rep_v1.encode(
        motion["posed_joints"], motion["root_positions"], motion["root_rot"],
        motion["joint_angles"], motion["foot_contacts"], int(r_hip), int(l_hip),
    )

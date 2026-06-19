"""
Bones-SEED G1 CSV (120 fps, MuJoCo Z-up) → unified 20-fps NPZ in kimodo's
Y-up frame containing:

    local_rot_mats  : (T, 34, 3, 3)  — kimodo G1Skeleton34 layout, Y-up
    root_positions  : (T, 3)          — pelvis xyz in Y-up
    posed_joints    : (T, 34, 3)      — kimodo FK output, Y-up
    fps             : int             — target fps (20)
    frame           : str             — "kimodo_y_up"

This NPZ is the canonical training input for **both** TMR-G1 (consumes
`posed_joints`) and a custom Kimodo-G1 trainer (consumes
`local_rot_mats + root_positions` and feeds `KimodoMotionRep(G1Skeleton34)`).

For Mujoco/SONIC playback we go through `kimodo.exports.mujoco.MujocoQposConverter`
exactly as upstream Kimodo-G1 does — same dict layout, same coordinate frame.

Approach (per frame):
  1. set MuJoCo qpos, run `mj_kinematics`.
  2. For each MJCF body i: `local_rot_mj[i] = xmat[parent_i].T @ xmat[i]`.
  3. Build `local_rot_mats[34, 3, 3]` indexed by kimodo joint order:
       - 30 entries come from MJCF bodies (mapped by name `*_joint` → `*_skel`).
       - 4 entries (left/right toe_base + left/right hand_roll_skel) are
         identity — they're virtual endpoints with no DOF.
  4. Rotate each `local_rot` into kimodo Y-up frame: `R_k = M R_mj M.T`.
     Rotate `root_positions` by `v_k = M v_mj` (i.e. `mj @ M.T` on rows).
  5. Run `G1Skeleton34.fk(local_rot_mats, root_positions)` → posed_joints.

Usage:
    python scripts/g1_csv_to_unified_npz.py \
        --csv data/bones_seed/g1/csv/<sess>/<name>.csv \
        --out /tmp/<name>.npz

Batch:
    python scripts/g1_csv_to_unified_npz.py \
        --csv-root data/bones_seed/g1/csv \
        --out-root data/bones_seed/g1_20fps_npz_v2 \
        --workers 16 --skip-existing
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# Pin BLAS/OMP to 1 thread per process *before* numpy import. With many
# multiprocessing workers, multi-threaded BLAS oversubscribes cores (load avg
# ~600 on a 32-core box). Single-threaded workers parallelize cleanly.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

G1_XML = "/nfsdata/home/jungbin.cho/TAP/kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml"

# MuJoCo Z-up, X-forward -> Kimodo Y-up, Z-forward (matches
# kimodo/exports/mujoco.py:_prepare_transforms). For row vectors:
#   v_kimodo = v_mj @ M.T
# For 3x3 rotations:
#   R_kimodo = M @ R_mj @ M.T
_MUJOCO_TO_KIMODO_NP = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32,
)


# ---------- CSV → qpos ---------------------------------------------------
def _load_qpos(csv_path: Path) -> np.ndarray:
    """Bones-SEED G1 CSV (header + cm/deg) → MuJoCo qpos (m, quat_wxyz, rad)."""
    df = pd.read_csv(csv_path)
    root_pos_m = (
        np.stack([df["root_translateX"], df["root_translateY"], df["root_translateZ"]], axis=1)
        / 100.0
    )
    euler_deg = np.stack(
        [df["root_rotateX"], df["root_rotateY"], df["root_rotateZ"]], axis=1
    )
    root_quat_xyzw = Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat()
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    joint_cols = [c for c in df.columns if c.endswith("_dof")]
    if len(joint_cols) != 29:
        raise ValueError(f"got {len(joint_cols)} _dof cols, want 29: {csv_path}")
    joint_rad = np.deg2rad(df[joint_cols].values)
    return np.concatenate([root_pos_m, root_quat_wxyz, joint_rad], axis=1).astype(np.float64)


# ---------- One-time MJCF body parent & name index ----------------------
_MJ_CACHE = {}


def _make_mj():
    """Build a per-process MuJoCo model + body index. Cheap once cached."""
    import mujoco
    pid = os.getpid()
    if pid in _MJ_CACHE:
        return _MJ_CACHE[pid]
    model = mujoco.MjModel.from_xml_path(G1_XML)
    data = mujoco.MjData(model)
    body_names = []
    for i in range(model.nbody):
        n = model.body(i).name
        body_names.append(n)
    # MJCF body name -> body index. `world` body is at index 0.
    name_to_bid = {n: i for i, n in enumerate(body_names)}
    parent_id = np.asarray(model.body_parentid, dtype=np.int64).copy()  # (nbody,)
    bundle = (mujoco, model, data, body_names, name_to_bid, parent_id)
    _MJ_CACHE[pid] = bundle
    return bundle


# ---------- Kimodo G1Skeleton34 (cached) --------------------------------
_KIM_CACHE = {}


def _make_kimodo_skeleton():
    pid = os.getpid()
    if pid in _KIM_CACHE:
        return _KIM_CACHE[pid]
    # IMPORTANT: cd to /tmp before importing kimodo (CLAUDE.md note 6).
    os.chdir("/tmp")
    from kimodo.skeleton import G1Skeleton34
    skel = G1Skeleton34()
    # Build kimodo-name → kimodo-index map.
    kimodo_names = list(skel.bone_order_names)
    kim_name_to_idx = {n: i for i, n in enumerate(kimodo_names)}
    _KIM_CACHE[pid] = (skel, kimodo_names, kim_name_to_idx)
    return _KIM_CACHE[pid]


# Manual MJCF body name → kimodo joint name table. Most follow the suffix
# convention `_link` ↔ `_skel`. The only exception is `torso_link`, whose
# kimodo counterpart is `waist_pitch_skel` (kimodo treats the torso as the
# third waist joint). The 4 virtual kimodo joints (`*_toe_base`,
# `*_hand_roll_skel`) have no MJCF counterpart and stay at identity.
_MJ_BODY_TO_KIM_NAME = {
    "pelvis": "pelvis_skel",
    "torso_link": "waist_pitch_skel",
}


def _mj_body_to_kim_name(mj_name: str) -> str:
    if mj_name in _MJ_BODY_TO_KIM_NAME:
        return _MJ_BODY_TO_KIM_NAME[mj_name]
    if mj_name.endswith("_link"):
        return mj_name[: -len("_link")] + "_skel"
    return mj_name


def _build_mj_to_kimodo_name_map(mj_body_names: list[str], kim_names: list[str]) -> dict[int, int]:
    """For each MJCF body index, return its kimodo joint index. World body
    is skipped. The 4 virtual kimodo joints with no MJCF counterpart are not
    in this map."""
    out: dict[int, int] = {}
    for mj_idx, mj_name in enumerate(mj_body_names):
        if mj_idx == 0:
            continue  # world
        kim_name = _mj_body_to_kim_name(mj_name)
        if kim_name not in kim_names:
            raise KeyError(f"MJCF body {mj_name!r} maps to kimodo joint "
                           f"{kim_name!r} which is not in skeleton.")
        out[mj_idx] = kim_names.index(kim_name)
    return out


# ---------- Main per-frame conversion -----------------------------------
def csv_to_unified_npz(csv_path: Path, out_path: Path,
                      source_fps: int = 120, target_fps: int = 20):
    qpos = _load_qpos(csv_path)
    stride = source_fps // target_fps
    qpos = qpos[::stride]
    T = qpos.shape[0]
    if T < 2:
        raise ValueError(f"too few frames: {T} in {csv_path}")

    (mujoco, model, data, mj_body_names, _name_to_bid, parent_id) = _make_mj()
    (kim_skel, kim_names, kim_name_to_idx) = _make_kimodo_skeleton()

    # Static MJCF body → kimodo joint index map.
    mj_to_kim = _build_mj_to_kimodo_name_map(mj_body_names, kim_names)
    nbody = model.nbody  # incl. world
    n_kim = len(kim_names)  # 34
    assert n_kim == 34, f"expected G1Skeleton34, got nbjoints={n_kim}"

    # Output arrays in kimodo space.
    local_rot_k = np.tile(np.eye(3, dtype=np.float32), (T, n_kim, 1, 1))  # default identity (for endpoints)
    root_positions = qpos[:, :3].astype(np.float32)  # MuJoCo frame, transform below

    M = _MUJOCO_TO_KIMODO_NP  # (3,3) MuJoCo -> Kimodo
    M_T = M.T

    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        xmat = data.xmat.reshape(nbody, 3, 3)  # global rotation per body in MJ frame

        # Local-rotation of each MJCF body relative to its parent:
        # R_local_mj[i] = R_parent_mj.T @ R_world_mj
        for mj_i, kim_i in mj_to_kim.items():
            R_world = xmat[mj_i].astype(np.float32)
            R_parent = xmat[parent_id[mj_i]].astype(np.float32)
            R_local_mj = R_parent.T @ R_world
            # Rotate into kimodo frame.
            R_local_k = M @ R_local_mj @ M_T
            local_rot_k[t, kim_i] = R_local_k

    # Transform root_positions to kimodo Y-up frame.
    root_positions_k = (root_positions @ M_T).astype(np.float32)

    # Run kimodo FK to get posed_joints in Y-up frame.
    with torch.no_grad():
        local_rot_t = torch.from_numpy(local_rot_k).unsqueeze(0)        # (1, T, 34, 3, 3)
        root_pos_t = torch.from_numpy(root_positions_k).unsqueeze(0)    # (1, T, 3)
        _, posed_joints, _ = kim_skel.fk(local_rot_t, root_pos_t)
        posed_joints = posed_joints[0].cpu().numpy().astype(np.float32)  # (T, 34, 3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        local_rot_mats=local_rot_k,            # (T, 34, 3, 3) fp32 — kimodo Y-up
        root_positions=root_positions_k,        # (T, 3) fp32
        posed_joints=posed_joints,              # (T, 34, 3) fp32
        fps=np.int32(target_fps),
        frame=np.array("kimodo_y_up"),
    )


# ---------- Driver ------------------------------------------------------
def _walk_csvs(csv_root: Path):
    for p in csv_root.rglob("*.csv"):
        yield p


def _worker(arg):
    chunk, source_fps, target_fps = arg
    ok = fail = 0
    for src, dst in chunk:
        try:
            csv_to_unified_npz(src, dst, source_fps, target_fps)
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"FAIL {src}: {e}", file=sys.stderr)
    return ok, fail


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--csv-root", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--source-fps", type=int, default=120)
    p.add_argument("--target-fps", type=int, default=20)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    if args.csv and args.out:
        csv_to_unified_npz(Path(args.csv), Path(args.out),
                           args.source_fps, args.target_fps)
        return

    assert args.csv_root and args.out_root
    csv_root = Path(args.csv_root)
    out_root = Path(args.out_root)

    todo = []
    for src in _walk_csvs(csv_root):
        rel = src.relative_to(csv_root)
        dst = out_root / rel.with_suffix(".npz")
        if args.skip_existing and dst.exists():
            continue
        todo.append((src, dst))
        if args.limit and len(todo) >= args.limit:
            break
    print(f"[unified] {len(todo)} CSVs to process, workers={args.workers}")
    if not todo:
        return

    if args.workers <= 1:
        _worker((todo, args.source_fps, args.target_fps))
        return

    chunks = [todo[i::args.workers] for i in range(args.workers)]
    work_args = [(c, args.source_fps, args.target_fps) for c in chunks]
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(_worker, work_args)
    ok = sum(r[0] for r in results)
    fail = sum(r[1] for r in results)
    dt = time.time() - t0
    print(f"[unified] FINAL ok={ok} fail={fail}  elapsed={dt/60:.1f} min "
          f"({ok/max(1e-6,dt):.1f}/s)")


if __name__ == "__main__":
    main()

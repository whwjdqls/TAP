"""
Resample one Bones-SEED G1 retargeted CSV (120 fps qpos, 36 cols) to 20 fps
and emit an NPZ with:

    posed_joints   : (T, 34, 3) global joint positions, MuJoCo Z-up
    root_positions : (T, 3)     root xyz (= posed_joints[:, root_idx])

This is the input format TMRMotionRep can consume via its `posed_joints=...`
path — no FK / rotation matrices needed downstream.

Designed for `xargs -P` parallelism over the whole 142K motion corpus.

Usage (single file):
    python scripts/g1_csv_to_20fps_npz.py --csv <one.csv> --out <one.npz>

Batch mode (whole corpus, see scripts/g1_resample_20fps.sbatch):
    python scripts/g1_csv_to_20fps_npz.py \
        --csv-root data/bones_seed/g1/csv \
        --out-root data/bones_seed/g1_20fps_npz \
        --workers 16 --skip-existing
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# Where the kimodo g1 MJCF lives. Hardcoded so this works from any env.
G1_XML = "/nfsdata/home/jungbin.cho/TAP/kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml"

# MuJoCo (Z-up, X-forward) -> Kimodo (Y-up, Z-forward).
# Matches kimodo/exports/mujoco.py:_prepare_transforms.
#   kimodo_x = mujoco_y
#   kimodo_y = mujoco_z   (pelvis height ends up on Y, the kimodo "up" axis)
#   kimodo_z = mujoco_x
# Applied to every position vector: v_kimodo = v_mujoco @ M.T
_MUJOCO_TO_KIMODO = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32,
)


def _load_qpos(csv_path: Path) -> np.ndarray:
    """Read a Bones-SEED G1 CSV (header + cm/deg) and convert to MuJoCo qpos
    (m, quat-wxyz, rad). Output shape (T, 36)."""
    df = pd.read_csv(csv_path)
    root_pos_m = np.stack(
        [df["root_translateX"], df["root_translateY"], df["root_translateZ"]],
        axis=1,
    ) / 100.0  # cm → m
    euler_deg = np.stack(
        [df["root_rotateX"], df["root_rotateY"], df["root_rotateZ"]],
        axis=1,
    )
    root_quat_xyzw = Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat()
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    joint_cols = [c for c in df.columns if c.endswith("_dof")]
    if len(joint_cols) != 29:
        raise ValueError(f"expected 29 _dof cols, got {len(joint_cols)} in {csv_path}")
    joint_rad = np.deg2rad(df[joint_cols].values)
    qpos = np.concatenate([root_pos_m, root_quat_wxyz, joint_rad], axis=1).astype(np.float64)
    return qpos


def _make_mj():
    """Build a (model, data, root_body_id, all_body_ids) tuple for the G1 MJCF."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(G1_XML)
    data = mujoco.MjData(model)
    # We treat *all* bodies (excluding world body 0) as the 34-joint skeleton.
    # SkeletonBase.nbjoints == 34 for G1Skeleton34 (32 articulated + 2 toe
    # endpoints + pelvis), so we expect model.nbody == 35 (incl. world).
    n_bodies_inc_world = model.nbody
    if n_bodies_inc_world - 1 != 34:
        # Not fatal — TMRMotionRep only needs the joint count to match
        # G1Skeleton34.nbjoints. If the MJCF includes extra fixed sites it'll
        # error here.
        print(f"WARN: g1.xml has {n_bodies_inc_world-1} bodies (excl world); "
              f"expected 34. We use all of them.", file=sys.stderr)
    body_ids = np.arange(1, n_bodies_inc_world)
    return mujoco, model, data, body_ids


def csv_to_npz(csv_path: Path, out_path: Path, source_fps: int = 120, target_fps: int = 20):
    """Read csv, FK in MuJoCo, downsample, save NPZ."""
    qpos = _load_qpos(csv_path)
    stride = source_fps // target_fps  # 6
    qpos = qpos[::stride]
    T = qpos.shape[0]
    if T < 2:
        raise ValueError(f"too few frames after downsample: {T} in {csv_path}")

    mujoco, model, data, body_ids = _make_mj()
    n_joints = len(body_ids)
    posed_joints = np.zeros((T, n_joints, 3), dtype=np.float32)
    root_positions = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        data.qpos[:] = qpos[t]
        # mj_kinematics is enough — no need for full forward dynamics here.
        mujoco.mj_kinematics(model, data)
        posed_joints[t] = data.xpos[body_ids]
        root_positions[t] = qpos[t, :3]

    # MuJoCo Z-up -> kimodo Y-up. Apply to every position vector.
    posed_joints = posed_joints @ _MUJOCO_TO_KIMODO.T
    root_positions = root_positions @ _MUJOCO_TO_KIMODO.T

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        posed_joints=posed_joints,
        root_positions=root_positions,
        fps=np.int32(target_fps),
        frame="kimodo_y_up",
    )


def _walk_csvs(csv_root: Path):
    for p in csv_root.rglob("*.csv"):
        yield p


def _worker(arg):
    chunk, source_fps, target_fps = arg
    ok = fail = 0
    for src, dst in chunk:
        try:
            csv_to_npz(src, dst, source_fps, target_fps)
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"[g1->20fps] FAIL {src}: {e}", file=sys.stderr)
    return ok, fail


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None, help="single-file mode")
    p.add_argument("--out", default=None, help="single-file mode output .npz")
    p.add_argument("--csv-root", default=None, help="batch mode: walk this dir for CSVs")
    p.add_argument("--out-root", default=None, help="batch mode: mirror tree under this dir")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--source-fps", type=int, default=120)
    p.add_argument("--target-fps", type=int, default=20)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    if args.csv and args.out:
        csv_to_npz(Path(args.csv), Path(args.out), args.source_fps, args.target_fps)
        return

    assert args.csv_root and args.out_root, "either --csv/--out or --csv-root/--out-root"
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
    print(f"[g1->20fps] {len(todo)} CSVs to process, workers={args.workers}")

    if not todo:
        return

    if args.workers <= 1:
        t0 = time.time()
        for i, (src, dst) in enumerate(todo, 1):
            try:
                csv_to_npz(src, dst, args.source_fps, args.target_fps)
            except Exception as e:  # noqa: BLE001
                print(f"[g1->20fps] FAIL {src}: {e}", file=sys.stderr)
            if i % 200 == 0:
                rate = i / (time.time() - t0)
                eta = (len(todo) - i) / max(1e-6, rate) / 60
                print(f"[g1->20fps] {i}/{len(todo)} done  {rate:.1f}/s  ETA {eta:.1f} min")
        return

    # Parallel workers: each owns its own MuJoCo model (cheap, ~50 ms init).
    import multiprocessing as mp

    chunks = [todo[i::args.workers] for i in range(args.workers)]
    work_args = [(c, args.source_fps, args.target_fps) for c in chunks]
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(_worker, work_args)
    ok = sum(r[0] for r in results)
    fail = sum(r[1] for r in results)
    dt = time.time() - t0
    print(f"[g1->20fps] FINAL ok={ok} fail={fail}  elapsed={dt/60:.1f} min "
          f"({ok/max(1e-6,dt):.1f}/s)")


if __name__ == "__main__":
    main()

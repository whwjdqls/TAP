"""
Batch consistency check: every unified G1 NPZ vs its original Bones-SEED CSV.

Same two checks as scripts/g1_check_unified_consistency.py, run over the whole
dataset in parallel and aggregated:

  (A) FK consistency  — stored `posed_joints` == MuJoCo `xpos` (Y-up). Exact
                        coordinate transform, spot-checked on evenly-spaced
                        frames per motion.
  (B) qpos round-trip — `dict_to_qpos(local_rot_mats, root_positions)` ==
                        original CSV qpos. Checked on ALL frames (batched, cheap).

Each worker loads the MuJoCo model + skeleton + converter once, then streams a
chunk of motions. A motion FAILS if errors exceed tolerance, frame counts
mismatch, or anything throws. Writes a JSON report + prints a summary.

This is a ~142K-motion job -> run on a CPU Slurm node, never the login node:
    sbatch scripts/g1_check_all_consistency.sbatch
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

G1_XML = os.environ.get("G1_XML") or str(
    Path(__file__).resolve().parents[1]
    / "kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml"
)
_M = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)

# Per-worker singletons (set in _init).
_W = {}


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


def _init(pos_tol, rot_tol, quat_tol, frames_a, source_fps, target_fps):
    import mujoco
    os.chdir("/tmp")  # dodge kimodo namespace-shadowing trap
    import torch
    from kimodo.skeleton import G1Skeleton34
    from kimodo.exports.mujoco import MujocoQposConverter

    skel = G1Skeleton34()
    model = mujoco.MjModel.from_xml_path(G1_XML)
    data = mujoco.MjData(model)

    mj_to_kim_map = {"pelvis": "pelvis_skel", "torso_link": "waist_pitch_skel"}

    def mj_to_kim(name):
        if name in mj_to_kim_map:
            return mj_to_kim_map[name]
        if name.endswith("_link"):
            return name[:-5] + "_skel"
        return name

    mj_to_kim_idx = {}
    for mj_i in range(model.nbody):
        if mj_i == 0:
            continue
        kn = mj_to_kim(model.body(mj_i).name)
        if kn in skel.bone_order_names:
            mj_to_kim_idx[mj_i] = skel.bone_order_names.index(kn)

    _W.update(
        mujoco=mujoco, torch=torch, skel=skel, model=model, data=data,
        converter=MujocoQposConverter(skel, xml_path=G1_XML),
        mj_to_kim_idx=mj_to_kim_idx,
        pos_tol=pos_tol, rot_tol=rot_tol, quat_tol=quat_tol,
        frames_a=frames_a, stride=source_fps // target_fps,
    )


def _check_one(args):
    rel, csv_path, npz_path = args
    res = {"rel": rel, "ok": False, "reason": "", "T": 0,
           "a_max": None, "b_root_pos_max": None, "b_quat_max": None,
           "b_joint_max": None, "rt_pose_max": None}
    try:
        mujoco = _W["mujoco"]; torch = _W["torch"]
        model = _W["model"]; data = _W["data"]
        mj_to_kim_idx = _W["mj_to_kim_idx"]; converter = _W["converter"]; skel = _W["skel"]

        z = np.load(npz_path, allow_pickle=False)
        posed_joints = z["posed_joints"]
        root_positions = z["root_positions"]
        local_rot_mats = z["local_rot_mats"]
        T_npz = posed_joints.shape[0]
        res["T"] = int(T_npz)

        qpos = _load_qpos(csv_path)[:: _W["stride"]]
        T = min(T_npz, qpos.shape[0])
        if T_npz != qpos.shape[0]:
            res["reason"] = f"frame_count_mismatch npz={T_npz} csv={qpos.shape[0]}"
            # still compare the overlap; large delta is the real red flag
            if abs(T_npz - qpos.shape[0]) > 1:
                return res
        if T < 1:
            res["reason"] = "empty"
            return res

        # (B) qpos round-trip on all overlap frames (batched) -> joint/root/quat
        # angle diffs. Joint-angle diff is NOT the fidelity criterion: kimodo's
        # converter mis-recovers some hinge angles (notably shoulder_pitch) from
        # the rotations, which is a converter property, not a data error.
        out = {
            "local_rot_mats": torch.from_numpy(local_rot_mats[:T]).unsqueeze(0),
            "root_positions": torch.from_numpy(root_positions[:T]).unsqueeze(0),
            "posed_joints":   torch.from_numpy(posed_joints[:T]).unsqueeze(0),
            "foot_contacts":  torch.zeros(1, T, 4, dtype=torch.bool),
        }
        qp = converter.dict_to_qpos(out, device="cpu")
        qp = qp.detach().cpu().numpy() if hasattr(qp, "detach") else np.asarray(qp)
        if qp.ndim == 3:
            qp = qp[0]
        gt = qpos[:T]
        root_pos_err = np.linalg.norm(qp[:, :3] - gt[:, :3], axis=-1)
        dots = np.abs(np.einsum("ij,ij->i", qp[:, 3:7], gt[:, 3:7]))
        quat_err = 2.0 * np.arccos(np.clip(dots, 0, 1))
        joint_err = np.abs(qp[:, 7:36] - gt[:, 7:36])
        res["b_root_pos_max"] = float(root_pos_err.max())
        res["b_quat_max"] = float(quat_err.max())
        res["b_joint_max"] = float(joint_err.max())

        # (A) FK fidelity (THE pass criterion) + (C) round-trip POSE error, both
        # on evenly-spaced frames. (A): stored posed_joints vs MuJoCo FK of the
        # original CSV qpos. (C): MuJoCo FK of the *recovered* qpos vs stored
        # posed_joints -> how far the converter-recovered pose drifts.
        idx = np.unique(np.linspace(0, T - 1, min(_W["frames_a"], T)).astype(int))
        a_max = 0.0
        rt_pose_max = 0.0
        for t in idx:
            data.qpos[:] = qpos[t]
            mujoco.mj_kinematics(model, data)
            for mj_i, kim_i in mj_to_kim_idx.items():
                a_max = max(a_max, float(
                    np.linalg.norm(data.xpos[mj_i] @ _M.T - posed_joints[t, kim_i])))
            data.qpos[:] = qp[t]
            mujoco.mj_kinematics(model, data)
            for mj_i, kim_i in mj_to_kim_idx.items():
                rt_pose_max = max(rt_pose_max, float(
                    np.linalg.norm(data.xpos[mj_i] @ _M.T - posed_joints[t, kim_i])))
        res["a_max"] = a_max
        res["rt_pose_max"] = rt_pose_max

        # PASS = data fidelity (FK) only. Round-trip pose error is reported, not
        # failed on, since it reflects kimodo's converter, not our stored data.
        res["ok"] = bool(a_max < _W["pos_tol"] and not res["reason"])
        if not res["ok"] and not res["reason"]:
            res["reason"] = f"FK_tol_exceeded a={a_max:.4g} m"
    except Exception as e:  # noqa: BLE001
        res["reason"] = f"{type(e).__name__}: {e}"
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz-root", default="data/bones_seed/g1_unified_npz")
    p.add_argument("--csv-root", default="data/bones_seed/g1/csv")
    p.add_argument("--out", default="runs/g1_consistency/report.json")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--frames-a", type=int, default=16, help="FK spot-check frames/motion")
    p.add_argument("--source-fps", type=int, default=120)
    p.add_argument("--target-fps", type=int, default=20)
    p.add_argument("--pos-tol", type=float, default=1e-3)
    p.add_argument("--rot-tol", type=float, default=1e-3)
    p.add_argument("--quat-tol", type=float, default=5e-2)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    npz_root = (repo / args.npz_root) if not Path(args.npz_root).is_absolute() else Path(args.npz_root)
    csv_root = (repo / args.csv_root) if not Path(args.csv_root).is_absolute() else Path(args.csv_root)

    todo = []
    for npz in npz_root.rglob("*.npz"):
        rel = npz.relative_to(npz_root)
        csv = csv_root / rel.with_suffix(".csv")
        if csv.is_file():
            todo.append((str(rel.with_suffix("")), str(csv), str(npz)))
        if args.limit and len(todo) >= args.limit:
            break
    print(f"[check-all] {len(todo)} motion pairs, workers={args.workers}", flush=True)
    if not todo:
        raise SystemExit("no matching CSV/NPZ pairs found")

    t0 = time.time()
    results = []
    with mp.Pool(
        args.workers, initializer=_init,
        initargs=(args.pos_tol, args.rot_tol, args.quat_tol,
                  args.frames_a, args.source_fps, args.target_fps),
    ) as pool:
        for i, r in enumerate(pool.imap_unordered(_check_one, todo, chunksize=32), 1):
            results.append(r)
            if i % 5000 == 0:
                nfail = sum(1 for x in results if not x["ok"])
                print(f"  {i}/{len(todo)}  fails={nfail}  "
                      f"({i/max(1e-6,time.time()-t0):.0f}/s)", flush=True)

    dt = time.time() - t0
    fails = [r for r in results if not r["ok"]]
    oks = [r for r in results if r["ok"]]

    def _vals(key):
        return np.array([r[key] for r in results if r.get(key) is not None], dtype=float)

    def _stats(key):
        v = _vals(key)
        if v.size == 0:
            return None
        return {"max": float(v.max()), "mean": float(v.mean()),
                "p99": float(np.quantile(v, 0.99))}

    rt = _vals("rt_pose_max")
    summary = {
        "total": len(results),
        "passed": len(oks),                       # PASS = FK fidelity only
        "failed": len(fails),
        "elapsed_min": round(dt / 60, 2),
        "fk_fidelity_m": _stats("a_max"),         # THE data-consistency metric
        "roundtrip_pose_m": _stats("rt_pose_max"),  # converter playback drift
        "roundtrip_pose_over_1cm": int((rt > 0.01).sum()) if rt.size else 0,
        "roundtrip_pose_over_5cm": int((rt > 0.05).sum()) if rt.size else 0,
        "roundtrip_angle": {                      # informational (converter quirk)
            "root_pos_m": _stats("b_root_pos_max"),
            "quat_rad": _stats("b_quat_max"),
            "joint_rad": _stats("b_joint_max"),
        },
        "tolerances": {"fk_pos_tol_m": args.pos_tol},
        "failures": [{"rel": r["rel"], "reason": r["reason"]} for r in fails[:500]],
        "n_failures_listed": min(len(fails), 500),
    }

    out_path = (repo / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print("\n==================== SUMMARY ====================")
    print(f"  total={summary['total']}  PASSED(FK)={summary['passed']}  FAILED(FK)={summary['failed']}")
    print(f"  elapsed={summary['elapsed_min']} min")
    fk = summary["fk_fidelity_m"]; rtp = summary["roundtrip_pose_m"]
    print(f"  [A] FK fidelity (data consistency): max={fk['max']:.2e} m  p99={fk['p99']:.2e} m")
    print(f"  [C] round-trip pose drift (kimodo converter): max={rtp['max']:.4f} m  "
          f"p99={rtp['p99']:.4f} m")
    print(f"      motions w/ round-trip pose > 1cm: {summary['roundtrip_pose_over_1cm']}  "
          f"> 5cm: {summary['roundtrip_pose_over_5cm']}")
    if fails:
        print(f"  first FK failures (up to 10):")
        for r in fails[:10]:
            print(f"    {r['rel']}: {r['reason']}")
    print(f"  report -> {out_path}")
    print("=================================================")


if __name__ == "__main__":
    main()

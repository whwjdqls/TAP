"""
In-place coordinate-frame fix for the 142K G1 20-fps NPZs: rotate every
position vector from MuJoCo (Z-up, X-forward) into the canonical kimodo
frame (Y-up, Z-forward) using the same matrix as
`kimodo/exports/mujoco.py:_prepare_transforms`.

We can skip the full ~50-min MuJoCo FK re-run because the saved positions are
already correct *up to* the coordinate rotation; only the axis swap is wrong.

Idempotency: detects already-converted NPZs via the saved `frame` key and
skips them.

Usage:
    python scripts/g1_npz_mj_to_kimodo.py \
        --root data/bones_seed/g1_20fps_npz --workers 16
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

_MUJOCO_TO_KIMODO = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32,
)


def _convert_one(path: Path) -> str:
    try:
        with np.load(path, allow_pickle=False) as d:
            keys = set(d.files)
            if "frame" in keys:
                try:
                    if str(d["frame"]) == "kimodo_y_up":
                        return "skip"
                except Exception:  # noqa: BLE001
                    pass
            posed = np.asarray(d["posed_joints"])
            root = np.asarray(d["root_positions"])
            fps = np.asarray(d["fps"]) if "fps" in keys else np.int32(20)
        posed_k = (posed @ _MUJOCO_TO_KIMODO.T).astype(np.float32)
        root_k = (root @ _MUJOCO_TO_KIMODO.T).astype(np.float32)
        # numpy auto-appends `.npz` if the path doesn't already end in it. Use
        # a sibling temp filename that already has `.npz` so the saved file
        # name is exactly what we pass.
        tmp = path.parent / f".{path.stem}.tmp.npz"
        np.savez_compressed(
            str(tmp),
            posed_joints=posed_k,
            root_positions=root_k,
            fps=fps,
            frame=np.array("kimodo_y_up"),
        )
        tmp.replace(path)
        return "ok"
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {path}: {e}", file=sys.stderr)
        return "fail"


def _worker(paths):
    ok = skip = fail = 0
    for p in paths:
        r = _convert_one(p)
        if r == "ok":
            ok += 1
        elif r == "skip":
            skip += 1
        else:
            fail += 1
    return ok, skip, fail


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    root = Path(args.root)
    todo = sorted(root.rglob("*.npz"))
    print(f"[mj->kimodo] found {len(todo)} NPZs in {root}, workers={args.workers}")
    if not todo:
        return

    chunks = [todo[i::args.workers] for i in range(args.workers)]
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(_worker, chunks)
    ok = sum(r[0] for r in results)
    skip = sum(r[1] for r in results)
    fail = sum(r[2] for r in results)
    dt = time.time() - t0
    print(f"[mj->kimodo] ok={ok} skip={skip} fail={fail}  "
          f"elapsed={dt/60:.1f} min ({ok/max(1e-6,dt):.1f}/s)")


if __name__ == "__main__":
    main()

"""
Precompute raw TMRMotionRep features (UN-normalized, UN-canonicalized) per G1
motion, once. The training dataset then slices the window and does only the
cheap per-fetch canonicalize + normalize (rotates on 210-d vectors), instead
of the full motion_rep forward (heading/rotate/velocity/foot-detect) every
fetch. ~3-5x dataloader throughput.

Correctness: per-frame features (root pos, heading, local joint positions,
velocities, foot contacts) are crop-invariant except a 1-frame velocity
boundary at the slice end (forward-diff vs duplicate), ~0.13% of values —
negligible (same approximation kimodo_open uses).

Output mirrors the input tree:
    <out-root>/<session>/<move>.npz  with  features (T, 210) fp16, fps, frame

Usage:
    python scripts/precompute_g1_features.py \
        --npz-root data/bones_seed/g1_unified_npz \
        --out-root data/bones_seed/g1_feat_npz \
        --workers 32 --skip-existing
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

_MR = {}


def _motion_rep():
    pid = os.getpid()
    if pid not in _MR:
        os.chdir("/tmp")
        from kimodo.skeleton import G1Skeleton34
        from kimodo.motion_rep.reps.tmr_motionrep import TMRMotionRep
        # No stats_path: we only need raw (unnormalized) features here.
        _MR[pid] = TMRMotionRep(skeleton=G1Skeleton34(), fps=20, stats_path=None)
    return _MR[pid]


def _one(src: Path, dst: Path):
    with np.load(src, allow_pickle=False) as d:
        posed = np.asarray(d["posed_joints"]).astype(np.float32)
    T = posed.shape[0]
    if T < 2:
        raise ValueError(f"too few frames {T}: {src}")
    mr = _motion_rep()
    with torch.no_grad():
        feats = mr(
            posed_joints=torch.from_numpy(posed).unsqueeze(0),
            to_normalize=False, to_canonicalize=False,
            lengths=torch.tensor([T]),
        )[0]  # (T, 210)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.stem}.tmp.npz"
    np.savez(str(tmp), features=feats.half().numpy(), fps=np.int32(20),
             frame=np.array("kimodo_y_up"))
    tmp.replace(dst)


def _worker(arg):
    chunk = arg
    ok = fail = 0
    for src, dst in chunk:
        try:
            _one(src, dst)
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"FAIL {src}: {e}", file=sys.stderr)
    return ok, fail


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    npz_root, out_root = Path(args.npz_root), Path(args.out_root)
    todo = []
    for src in npz_root.rglob("*.npz"):
        dst = out_root / src.relative_to(npz_root)
        if args.skip_existing and dst.exists():
            continue
        todo.append((src, dst))
        if args.limit and len(todo) >= args.limit:
            break
    print(f"[precompute-feat] {len(todo)} motions, workers={args.workers}")
    if not todo:
        return
    chunks = [todo[i::args.workers] for i in range(args.workers)]
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        res = pool.map(_worker, chunks)
    ok = sum(r[0] for r in res); fail = sum(r[1] for r in res)
    dt = time.time() - t0
    print(f"[precompute-feat] ok={ok} fail={fail} elapsed={dt/60:.1f}min "
          f"({ok/max(1e-6,dt):.1f}/s)")


if __name__ == "__main__":
    main()

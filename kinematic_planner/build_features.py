"""Precompute g1_rep_v1 features for a Bones-SEED split.

For each `<sess>/<move>` id in --split, encode g1_rep_v1 (T,136) and save to
--out-root/<sess>/<move>.npy. Multi-worker; skips missing/short/failed.

    python build_features.py --split .../train_split_paths_small.txt \
        --out-root /home/jungbin_cho/seed/g1_rep_v1_feats --workers 16
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, "/home/jungbin_cho/TAP/kinematic_planner")

_SKEL = None


def _skeleton():
    global _SKEL
    if _SKEL is None:
        os.chdir("/tmp")  # kimodo namespace guard
        from kimodo.skeleton import G1Skeleton34
        _SKEL = G1Skeleton34()
    return _SKEL


def _one(args):
    mid, out_root, min_frames = args
    import torch  # noqa
    import g1_data
    skel = _skeleton()
    try:
        mo = g1_data.load_motion(mid, skel)
        if mo["posed_joints"].shape[0] < min_frames:
            return ("short", mid)
        feats = g1_data.encode_motion(mo, skel).cpu().numpy().astype(np.float32)
        if not np.isfinite(feats).all():
            return ("nonfinite", mid)
        dst = Path(out_root) / f"{mid}.npy"
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, feats)
        return ("ok", mid)
    except Exception as e:  # noqa: BLE001
        return (f"err:{type(e).__name__}:{e}", mid)


def _worker(chunk):
    return [_one(a) for a in chunk]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--min-frames", type=int, default=10)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    with open(args.split) as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    if args.limit:
        ids = ids[: args.limit]
    todo = [m for m in ids if not (args.skip_existing and (Path(args.out_root) / f"{m}.npy").exists())]
    print(f"[feats] {len(todo)}/{len(ids)} to encode, workers={args.workers}", flush=True)

    chunks = [todo[i::args.workers] for i in range(args.workers)]
    work = [[(m, args.out_root, args.min_frames) for m in c] for c in chunks]
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(_worker, work)
    flat = [r for sub in results for r in sub]
    ok = sum(1 for s, _ in flat if s == "ok")
    bad = [(s, m) for s, m in flat if s != "ok"]
    dt = time.time() - t0
    print(f"[feats] ok={ok} bad={len(bad)} elapsed={dt/60:.1f}min ({ok/max(1e-6,dt):.1f}/s)", flush=True)
    for s, m in bad[:20]:
        print(f"   {s}  {m}", flush=True)
    # reason histogram
    from collections import Counter
    print("[feats] bad reasons:", Counter(s.split(':')[0] for s, _ in bad), flush=True)


if __name__ == "__main__":
    main()

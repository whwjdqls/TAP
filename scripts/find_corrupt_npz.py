"""Scan NPZs for corruption (truncated/partial writes); list bad paths.

Usage:
    python scripts/find_corrupt_npz.py --root <dir> --workers 32 [--delete]
"""
from __future__ import annotations
import argparse, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np


def _check(path):
    try:
        with np.load(path, allow_pickle=False) as d:
            _ = d["posed_joints"].shape
            _ = d["root_positions"].shape
        return None
    except Exception:  # noqa: BLE001
        return str(path)


def _worker(paths):
    return [p for p in (_check(x) for x in paths) if p is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--out", default=None, help="write bad paths list here")
    args = ap.parse_args()

    paths = sorted(Path(args.root).rglob("*.npz"))
    print(f"[scan] {len(paths)} NPZs, workers={args.workers}")
    chunks = [paths[i::args.workers] for i in range(args.workers)]
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(_worker, chunks)
    bad = [p for sub in results for p in sub]
    print(f"[scan] {len(bad)} corrupt  ({time.time()-t0:.1f}s)")
    if args.out:
        Path(args.out).write_text("\n".join(bad) + ("\n" if bad else ""))
        print(f"[scan] wrote list -> {args.out}")
    for p in bad[:20]:
        print("  BAD", p)
    if args.delete:
        for p in bad:
            Path(p).unlink(missing_ok=True)
        print(f"[scan] deleted {len(bad)} corrupt files")


if __name__ == "__main__":
    main()

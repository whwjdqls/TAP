"""
For each row in subset.jsonl, locate the corresponding G1 retargeted CSV in
Bones-SEED, crop to [crop_start, crop_end], and write it to a session-tree of
Bones-SEED-format CSVs that SONIC's `convert_soma_csv_to_motion_lib.py`
consumes in `--individual` mode.

Layout produced:
    <out_csvs>/<session>/<move_name>.csv

Then optionally invokes SONIC's converter to produce:
    <out_pkls>/<session>/<move_name>.pkl

Usage:
    python scripts/build_gt_motion_lib.py \
        --subset runs/eval_subset/text2motion_overview.jsonl \
        --metadata data/bones_seed/metadata/metadata/seed_metadata_v004.parquet \
        --g1-root  data/bones_seed/g1 \
        --out-csvs runs/eval_gt/bones_csvs \
        --out-pkls runs/eval_gt/motion_lib \
        --fps 30 --fps-source 120 \
        --convert
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--g1-root", required=True, help="dir containing csv/<session>/<move>.csv")
    p.add_argument("--out-csvs", required=True)
    p.add_argument("--out-pkls", required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--fps-source", type=int, default=120)
    p.add_argument("--crop-fps", type=int, default=30,
                   help="FPS the benchmark's crop_start/end indices are expressed in. "
                        "Bones-SEED G1 CSVs are 120 fps but the kimodo benchmark "
                        "stores crop indices in 30-fps SOMA frames, so we scale by "
                        "fps_source/crop_fps before slicing.")
    p.add_argument("--convert", action="store_true",
                   help="invoke SONIC convert_soma_csv_to_motion_lib.py after writing CSVs")
    p.add_argument("--sonic-converter", default=str(
        Path(__file__).resolve().parents[1]
        / "GR00T-WholeBodyControl" / "gear_sonic" / "data_process"
        / "convert_soma_csv_to_motion_lib.py"))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    g1_root = Path(args.g1_root) / "csv"

    rows = []
    with open(args.subset) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"[gt] subset rows: {len(rows)}")

    out_csvs = Path(args.out_csvs)
    if out_csvs.exists():
        shutil.rmtree(out_csvs)
    out_csvs.mkdir(parents=True)

    n_ok, n_missing, n_empty = 0, 0, 0
    for r in rows:
        move = r["move_name"]
        session = r.get("session") or ""
        if not session:
            n_missing += 1
            continue
        src = g1_root / session / f"{move}.csv"
        if not src.exists():
            print(f"[gt] missing csv: {src}")
            n_missing += 1
            continue

        # Read header + cropped frames.
        with open(src) as f:
            header = f.readline()
            lines = f.readlines()
        # Scale crop indices from benchmark fps (30) -> CSV fps (120).
        scale = args.fps_source // args.crop_fps
        a = max(0, r["crop_start"] * scale)
        b = min(len(lines), r["crop_end"] * scale)
        if b - a < 2:
            n_empty += 1
            continue

        # Session is the parent dir name of the CSV.
        session = src.parent.name
        dst_dir = out_csvs / session
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        with open(dst, "w") as f:
            f.write(header)
            f.writelines(lines[a:b])
        n_ok += 1

    print(f"[gt] wrote {n_ok} CSVs ({n_missing} missing, {n_empty} too short)")

    if not args.convert:
        return

    out_pkls = Path(args.out_pkls)
    if out_pkls.exists():
        shutil.rmtree(out_pkls)
    out_pkls.mkdir(parents=True)
    cmd = [
        sys.executable, args.sonic_converter,
        "--input", str(out_csvs),
        "--output", str(out_pkls),
        "--individual",
        "--fps", str(args.fps),
        "--fps_source", str(args.fps_source),
        "--num_workers", str(args.workers),
    ]
    print(f"[gt] running converter:\n  {' '.join(cmd)}")
    subprocess.check_call(cmd)
    n = sum(1 for _ in out_pkls.rglob("*.pkl"))
    print(f"[gt] motion_lib has {n} PKLs at {out_pkls}")


if __name__ == "__main__":
    main()

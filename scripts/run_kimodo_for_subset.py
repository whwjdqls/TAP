"""
For each row in subset.jsonl, generate a motion with Kimodo-G1-SEED-v1,
convert to a Bones-SEED-format CSV (units: m→cm, quat→Euler-deg, rad→deg),
and emit a session-tree of CSVs that SONIC's converter can ingest.

Each output CSV is named `<case_id_slug>.csv` and placed under
`<out_csvs>/kimodo/<case_id_slug>.csv` so the converter walks a single
session named "kimodo".

Then optionally runs SONIC's converter to build the motion_lib PKLs at
`<out_pkls>/kimodo/<case_id_slug>.pkl`.

Usage:
    python scripts/run_kimodo_for_subset.py \
        --subset runs/eval_subset/text2motion_overview.jsonl \
        --out-npzs runs/eval_kimodo/npzs \
        --out-csvs runs/eval_kimodo/bones_csvs \
        --out-pkls runs/eval_kimodo/motion_lib \
        --model Kimodo-G1-SEED-v1 \
        --diffusion-steps 50 \
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


def slugify(case_id: str) -> str:
    return case_id.replace("/", "__")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--out-npzs", required=True)
    p.add_argument("--out-csvs", required=True)
    p.add_argument("--out-pkls", required=True)
    p.add_argument("--model", default="Kimodo-G1-SEED-v1")
    p.add_argument("--diffusion-steps", type=int, default=50)
    p.add_argument("--text-encoder-device", default="cpu")
    p.add_argument("--convert", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--sonic-converter", default=str(
        Path(__file__).resolve().parents[1]
        / "GR00T-WholeBodyControl" / "gear_sonic" / "data_process"
        / "convert_soma_csv_to_motion_lib.py"))
    p.add_argument("--adapter", default=str(
        Path(__file__).resolve().parent / "kimodo_csv_to_bones_seed.py"))
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    rows = []
    with open(args.subset) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    print(f"[kimodo] {len(rows)} cases")

    out_npzs = Path(args.out_npzs)
    out_csvs = Path(args.out_csvs)
    for d in (out_npzs, out_csvs):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    session_dir = out_csvs / "kimodo"
    session_dir.mkdir(parents=True)

    env = os.environ.copy()
    env.setdefault("TEXT_ENCODER_DEVICE", args.text_encoder_device)

    n_ok, n_fail = 0, 0
    for i, r in enumerate(rows):
        slug = slugify(r["case_id"])
        out_stem = out_npzs / slug
        # 1) generation
        gen_cmd = [
            "kimodo_gen", r["text"],
            "--model", args.model,
            "--duration", f"{max(0.5, r['duration']):.2f}",
            "--num_samples", "1",
            "--diffusion_steps", str(args.diffusion_steps),
            "--seed", str(r.get("seed", 0)),
            "--output", str(out_stem),
        ]
        print(f"[kimodo {i+1}/{len(rows)}] {r['case_id']}  text={r['text']!r}  dur={r['duration']}")
        try:
            subprocess.check_call(gen_cmd, env=env)
        except subprocess.CalledProcessError as e:
            print(f"[kimodo] gen FAILED for {slug}: {e}")
            n_fail += 1
            continue

        npz = out_npzs / f"{slug}.npz"
        if not npz.exists():
            print(f"[kimodo] missing NPZ: {npz}")
            n_fail += 1
            continue

        # 2) kimodo NPZ → G1 qpos CSV (radians)
        qpos_csv = out_npzs / f"{slug}_qpos.csv"
        subprocess.check_call([
            "kimodo_convert", str(npz), str(qpos_csv),
            "--from", "kimodo", "--to", "g1-csv",
        ], env=env)

        # 3) qpos CSV → Bones-SEED-format CSV (cm/deg with header)
        dst = session_dir / f"{slug}.csv"
        subprocess.check_call([
            sys.executable, args.adapter,
            "--in", str(qpos_csv), "--out", str(dst),
        ], env=env)
        n_ok += 1

    print(f"[kimodo] {n_ok} ok, {n_fail} failed")
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
        "--fps", "30", "--fps_source", "30",
        "--num_workers", str(args.workers),
    ]
    print(f"[kimodo] converter:\n  {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)
    n = sum(1 for _ in out_pkls.rglob("*.pkl"))
    print(f"[kimodo] motion_lib has {n} PKLs at {out_pkls}")


if __name__ == "__main__":
    main()

"""
Walk a Kimodo-Motion-Gen-Benchmark testsuite dir, pull (text prompt, duration,
seed) from each `meta.json` and (move_name, crop_start, crop_end) from each
`seed_motion.json`, optionally subsample, and emit a JSONL.

Each output row:
    {
      "case_id": "content/text2motion/overview/0042",
      "text": "...",
      "duration": 3.0,
      "seed": 12,
      "move_name": "..._001__A057",
      "crop_start": 55,
      "crop_end": 145
    }

Usage:
    python scripts/build_eval_subset.py \
        --testsuite data/kimodo_benchmark/testsuite/content/text2motion/overview \
        --out runs/eval_subset/text2motion_overview.jsonl \
        --max 50 --require-text
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--testsuite", required=True, help="directory containing ####/ subdirs")
    p.add_argument("--out", required=True)
    p.add_argument("--max", type=int, default=0, help="0 = keep all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--require-text", action="store_true",
                   help="skip cases whose meta.json has empty text (constraints_notext task)")
    args = p.parse_args()

    root = Path(args.testsuite)
    cases = sorted([d for d in root.iterdir() if d.is_dir() and d.name.isdigit()])
    print(f"[subset] found {len(cases)} index dirs under {root}")

    rows = []
    skipped = 0
    for d in cases:
        meta_p = d / "meta.json"
        seed_p = d / "seed_motion.json"
        if not meta_p.exists() or not seed_p.exists():
            skipped += 1
            continue
        with open(meta_p) as f:
            meta = json.load(f)
        with open(seed_p) as f:
            seed = json.load(f)
        text = (meta.get("text") or "").strip()
        if args.require_text and not text:
            skipped += 1
            continue
        bvh_path = seed.get("bvh_path", "")
        # bvh_path format: "BVH/<session>/<move_name>.bvh"
        session = bvh_path.split("/")[1] if bvh_path else ""
        rows.append({
            "case_id": str(d.relative_to(root.parent.parent.parent)),
            "text": text,
            "duration": float(meta.get("duration", 0.0)),
            "seed": int(meta.get("seed", 0)),
            "move_name": seed["move_name"],
            "session": session,
            "crop_start": int(seed["crop_start_frame_index"]),
            "crop_end": int(seed["crop_end_frame_index"]),
        })

    print(f"[subset] loaded {len(rows)} rows, skipped {skipped}")
    if args.max and len(rows) > args.max:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = sorted(rows[:args.max], key=lambda r: r["case_id"])
    print(f"[subset] keeping {len(rows)}")

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[subset] wrote {args.out}")


if __name__ == "__main__":
    main()

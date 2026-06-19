"""Split a subset.jsonl into N evenly-sized shards.

Usage:
    python scripts/shard_subset.py --in subset.jsonl --out-dir shards --n 8
Produces shards/shard_0.jsonl ... shards/shard_{N-1}.jsonl
"""
import argparse, json, os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", required=True)
p.add_argument("--out-dir", required=True)
p.add_argument("--n", type=int, required=True)
args = p.parse_args()

rows = [json.loads(l) for l in open(args.inp) if l.strip()]
out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
shards = [[] for _ in range(args.n)]
for i, r in enumerate(rows):
    shards[i % args.n].append(r)
for i, s in enumerate(shards):
    fp = out / f"shard_{i}.jsonl"
    with open(fp, "w") as f:
        for r in s:
            f.write(json.dumps(r) + "\n")
    print(f"shard_{i}: {len(s)} rows -> {fp}")

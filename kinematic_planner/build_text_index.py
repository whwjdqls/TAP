"""Build a {<sess>/<move>: [captions]} index for a Bones-SEED split.

Captions = content_natural_desc_{1..4} from seed_metadata_v004.csv (keyed by
move_g1_path = g1/csv/<sess>/<move>.csv). Captions are intersected with the
LLM2Vec cache so the trainer's CachedTextEncoder never misses.

    python build_text_index.py --split .../train_split_paths_small.txt \
        --cache /home/jungbin_cho/kimodo_caches/bones_seed_llm2vec_small.pt \
        --out /home/jungbin_cho/seed/g1_rep_v1_text_small.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

META = "/home/jungbin_cho/seed/metadata/seed_metadata_v004.csv"
NAT_COLS = [f"content_natural_desc_{i}" for i in (1, 2, 3, 4)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--feat-root", default="/home/jungbin_cho/seed/g1_rep_v1_feats")
    args = p.parse_args()

    with open(args.split) as f:
        ids = {ln.strip() for ln in f if ln.strip()}
    cache_caps = set(torch.load(args.cache, map_location="cpu", weights_only=False)["captions"])
    print(f"[text] split={len(ids)} motions, cache={len(cache_caps)} captions")

    df = pd.read_csv(META, usecols=["move_g1_path"] + NAT_COLS)
    df["key"] = df["move_g1_path"].str.replace(r"^g1/csv/", "", regex=True).str.replace(r"\.csv$", "", regex=True)

    index, n_cap, n_drop_caps = {}, 0, 0
    for _, row in df.iterrows():
        key = row["key"]
        if key not in ids:
            continue
        caps = []
        for c in NAT_COLS:
            v = row[c]
            if isinstance(v, str) and v.strip():
                if v.strip() in cache_caps:
                    caps.append(v.strip())
                else:
                    n_drop_caps += 1
        if caps:
            index[key] = caps
            n_cap += len(caps)

    # also require the feature file to exist
    feat_root = Path(args.feat_root)
    have_feat = {k: v for k, v in index.items() if (feat_root / f"{k}.npy").exists()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(have_feat, f)
    print(f"[text] motions with >=1 cached caption: {len(index)}; with feature file: {len(have_feat)}")
    print(f"[text] total (motion,caption) pairs: {sum(len(v) for v in have_feat.values())}  "
          f"(dropped {n_drop_caps} uncached captions)")
    print(f"[text] wrote {args.out}")


if __name__ == "__main__":
    main()

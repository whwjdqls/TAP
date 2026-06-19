"""
TMR-G1 retrieval eval on the kimodo text2motion testsuite.

For each group (content|repetition × overview|timeline_single|timeline_multi):
  - For each test case: GT G1 motion (unified NPZ, cropped to the test window)
    + its text prompt.
  - Encode motion with the trained TMR-G1 motion encoder; encode text with the
    TMR-G1 text encoder (via a precomputed LLM2Vec cache).
  - Compute text→motion R-precision (R@1/2/3/5/10, MedR) over the pool, with
    the benchmark's 0.99 text-dedup, using kimodo's own metric code.

We are benchmarking the *TMR model itself* on GT motions, so these numbers are
directly comparable to the published "Ground Truth" R@3 rows
(https://research.nvidia.com/labs/sil/projects/kimodo/docs/benchmark/results.html).

IMPORTANT: encode motions the same way training did — `motion_rep(posed_joints,
to_normalize=True)` (NO to_canonicalize), then motion_encoder, take the mean
(mu). Mismatching this vs training silently tanks retrieval.

Usage:
    python tmr_g1/eval_retrieval.py \
        --ckpt runs/tmr_g1/v1/last.pt \
        --stats-path data/bones_seed/tmr_g1_stats_v2 \
        --text-emb-cache data/bones_seed/eval_text_emb.pt \
        --g1-npz-root data/bones_seed/g1_unified_npz \
        --testsuite data/kimodo_benchmark/testsuite \
        --out runs/tmr_g1/v1/retrieval.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn.functional as F

from tmr_g1.model.tmr_model import build_tmr_g1
from kimodo.metrics.tmr import compute_tmr_retrieval_metrics

GROUPS = ["overview", "timeline_single", "timeline_multi"]
SPLITS = ["content", "repetition"]
# Published Ground-Truth R@3 from the kimodo benchmark results page, for
# reference printing only. These are the SOMA TMR (trained on full Rigplay
# incl. test) retrieving on SOMA GT motions.
PUBLISHED_GT_R3 = {
    "content/overview": 89.09,
    "content/timeline_single": 86.26,
    "content/timeline_multi": 88.47,
    "repetition/overview": 93.91,
    "repetition/timeline_single": 90.13,
    "repetition/timeline_multi": 94.49,
}


def load_model(args):
    tmr, motion_rep, _decoder = build_tmr_g1(
        text_emb_cache_path=args.text_emb_cache,
        stats_path=args.stats_path,
        fps=args.fps,
        device=args.device,
    )
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    tmr.motion_encoder.load_state_dict(ckpt["motion_encoder"])
    tmr.text_encoder.load_state_dict(ckpt["text_encoder"])
    tmr.eval()
    for p in tmr.parameters():
        p.requires_grad_(False)
    return tmr, motion_rep


@torch.no_grad()
def encode_motion(tmr, motion_rep, posed_joints, device, canonicalize=True):
    # posed_joints: (T, 34, 3) numpy. MUST match how training encoded motions.
    T = posed_joints.shape[0]
    pj = torch.from_numpy(posed_joints).unsqueeze(0).to(device)  # (1, T, 34, 3)
    lengths = torch.tensor([T], device=device)
    feats = motion_rep(posed_joints=pj, to_normalize=True,
                       to_canonicalize=canonicalize, lengths=lengths)  # (1, T, D)
    mask = torch.ones(1, T, dtype=torch.bool, device=device)
    out = tmr.motion_encoder({"x": feats, "mask": mask})  # (1, 2, latent) if vae
    mu = out.unbind(1)[0]                                   # (1, latent)
    return F.normalize(mu, dim=-1)[0].cpu().numpy()


@torch.no_grad()
def encode_text(tmr, text, device):
    raw_feat, _ = tmr.raw_text_encoder([text])  # (1, 1, llm_dim)
    mask = torch.ones(1, 1, dtype=torch.bool, device=device)
    out = tmr.text_encoder({"x": raw_feat.to(device), "mask": mask})
    mu = out.unbind(1)[0]
    return F.normalize(mu, dim=-1)[0].cpu().numpy()


def load_cases(group_dir: Path):
    cases = []
    for cdir in sorted(glob.glob(str(group_dir / "*"))):
        cdir = Path(cdir)
        mp = cdir / "meta.json"
        sp = cdir / "seed_motion.json"
        if not (mp.is_file() and sp.is_file()):
            continue
        meta = json.load(open(mp))
        seed = json.load(open(sp))
        text = (meta.get("text") or "").strip()
        if not text:
            continue
        bvh = seed.get("bvh_path", "")
        session = bvh.split("/")[1] if bvh else ""
        cases.append({
            "case_id": cdir.name,
            "text": text,
            "move_name": seed["move_name"],
            "session": session,
            "crop_start": int(seed["crop_start_frame_index"]),
            "crop_end": int(seed["crop_end_frame_index"]),
        })
    return cases


def eval_group(tmr, motion_rep, args, split, grp):
    group_dir = Path(args.testsuite) / split / "text2motion" / grp
    cases = load_cases(group_dir)
    npz_root = Path(args.g1_npz_root)
    scale = args.fps / 30.0  # crop indices are 30 fps; our NPZs are args.fps

    motion_embs, text_embs, kept = [], [], 0
    miss = 0
    for c in cases:
        npz = npz_root / c["session"] / f"{c['move_name']}.npz"
        if not npz.is_file():
            miss += 1
            continue
        try:
            with np.load(npz, allow_pickle=False) as d:
                posed = np.asarray(d["posed_joints"]).astype(np.float32)
        except Exception:  # noqa: BLE001
            miss += 1
            continue
        n = posed.shape[0]
        a = max(0, int(round(c["crop_start"] * scale)))
        b = min(n, int(round(c["crop_end"] * scale)))
        if b - a < 4:
            miss += 1
            continue
        clip = posed[a:b]
        try:
            m = encode_motion(tmr, motion_rep, clip, args.device, args.canonicalize)
            t = encode_text(tmr, c["text"], args.device)
        except KeyError:
            miss += 1
            continue
        motion_embs.append(m)
        text_embs.append(t)
        kept += 1

    motion_embs = np.stack(motion_embs)
    text_embs = np.stack(text_embs)
    metrics = compute_tmr_retrieval_metrics(motion_embs, text_embs)
    # Pull just the t2m retrieval keys.
    out = {k.replace("TMR/t2m_R/", ""): v for k, v in metrics.items() if k.startswith("TMR/t2m_R/")}
    out["pool"] = kept
    out["missing"] = miss
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stats-path", required=True)
    p.add_argument("--text-emb-cache", required=True)
    p.add_argument("--g1-npz-root", required=True)
    p.add_argument("--testsuite", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--canonicalize", action="store_true", default=True)
    p.add_argument("--no-canonicalize", dest="canonicalize", action="store_false")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tmr, motion_rep = load_model(args)

    results = {}
    print(f"\n{'group':<32} {'pool':>6} {'R@1':>7} {'R@2':>7} {'R@3':>7} {'R@5':>7} {'R@10':>7} {'MedR':>6}  {'GT R@3':>7}")
    print("-" * 100)
    for split in SPLITS:
        for grp in GROUPS:
            r = eval_group(tmr, motion_rep, args, split, grp)
            key = f"{split}/{grp}"
            results[key] = r
            gt = PUBLISHED_GT_R3.get(key, None)
            gt_s = f"{gt:.2f}" if gt is not None else "-"
            print(f"{key:<32} {r['pool']:>6} {r.get('R01',0):>7.2f} {r.get('R02',0):>7.2f} "
                  f"{r.get('R03',0):>7.2f} {r.get('R05',0):>7.2f} {r.get('R10',0):>7.2f} "
                  f"{r.get('MedR',0):>6.1f}  {gt_s:>7}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"results": results, "published_gt_r3": PUBLISHED_GT_R3,
                       "ckpt": args.ckpt}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

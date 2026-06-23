"""Evaluate the g1_rep_v1 text->motion MDM on the Kimodo motion-gen benchmark,
scored by the TMR-G1 retrieval model — the G1 analog of kimodo/benchmark.

For each text2motion case: generate a motion with the MDM from the prompt,
decode to G1 joint positions, and score with TMR-G1 (text-alignment retrieval +
similarity + FID) plus foot-skate / contact-consistency. GT motions come from
g1_unified_npz (cropped to the case window). **No generated motion is saved** —
only 256-d TMR embeddings + per-case scalars are kept in memory.

Metrics per group (content/repetition x overview/timeline_single/timeline_multi):
  TMR/t2m_R/{R01,R02,R03,R05,R10,MedR}  text -> GENERATED motion  (the MDM score)
  TMR/t2m_gt_R/*                         text -> GT motion         (TMR-G1 upper bound)
  TMR/m2m_R/*                            generated -> GT motion
  TMR/FID/{gen_text,gen_gt,gt_text}
  TMR/{t2m_sim,m2m_sim,t2m_gt_sim}
  foot_skate_from_pred_contacts, foot_contact_consistency  (generated)

    python eval_mdm.py --mdm-ckpt runs/.../ckpt_final.pt \
        --tmr-ckpt runs/tmr_g1/v6/step_00015000.pt \
        --tmr-stats runs/tmr_g1/v6/tmr_g1_stats_v3 \
        --text-cache /home/jungbin_cho/kimodo_caches/benchmark_llm2vec.pt \
        --testsuite /home/jungbin_cho/Kimodo-Motion-Gen-Benchmark-20fps/testsuite \
        --g1-npz-root /home/jungbin_cho/seed/g1_unified_npz \
        --out runs/.../benchmark_eval.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

os.chdir("/tmp")  # kimodo namespace guard
sys.path.insert(0, "/home/jungbin_cho/TAP/kinematic_planner")

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import g1_rep_v1
from kimodo.scripts.train import build_denoiser_from_model_config, build_text_encoder, encode_texts, load_config
from kimodo.scripts.train_w_hml3d import _cfg_sample
from kimodo.model.diffusion import Diffusion
from kimodo.skeleton import G1Skeleton34
from kimodo.metrics.tmr import compute_tmr_retrieval_metrics, get_scores_unit
from kimodo.metrics.foot_skate import FootSkateFromContacts, FootContactConsistency
from tmr_g1.model.tmr_model import build_tmr_g1

FPS = 20
GROUPS = ["overview", "timeline_single", "timeline_multi"]
SPLITS = ["content", "repetition"]


# ----------------------------------------------------------------------------
# MDM (generator)
# ----------------------------------------------------------------------------
def load_mdm(cfg, ckpt_path, device, use_ema=True):
    den = build_denoiser_from_model_config(
        cfg.model_config_path, cfg.get("stats_path", ""), fps_override=cfg.get("denoiser_fps_override"),
    ).to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = dict(ck["denoiser"])
    if use_ema and ck.get("ema"):
        for k, v in ck["ema"].items():
            if k in sd and sd[k].shape == v.shape:
                sd[k] = v
    den.load_state_dict(sd, strict=True)
    den.eval()
    for p in den.parameters():
        p.requires_grad_(False)
    return den


@torch.no_grad()
def mdm_generate(den, diffusion, text_encoder, texts, n_frames, mean, std, device, n_steps, cfg_scale):
    """Generate a batch of motions. Returns list of (T_i, 34, 3) joints + (T_i,4) contacts."""
    B = len(texts)
    maxT = int(max(n_frames))
    D = den.motion_rep.motion_rep_dim
    text_feat, text_pad = encode_texts(text_encoder, texts, device)
    pad_mask = torch.zeros(B, maxT, dtype=torch.bool, device=device)
    for i, n in enumerate(n_frames):
        pad_mask[i, :n] = True
    first_heading = torch.zeros(B, device=device)
    motion_mask = torch.zeros(B, maxT, D, device=device)
    observed = torch.zeros(B, maxT, D, device=device)
    gen = _cfg_sample(den, diffusion, text_feat=text_feat, text_pad_mask=text_pad, pad_mask=pad_mask,
                      first_heading=first_heading, motion_mask=motion_mask, observed=observed,
                      n_steps=n_steps, cfg_scale=cfg_scale, device=device)  # (B,maxT,D) normalized
    gen = gen.float() * std + mean                                          # unnormalize
    out = []
    fc_sl = g1_rep_v1.SLICE_DICT["foot_contacts"]
    for i in range(B):
        n = int(n_frames[i])
        feats = gen[i, :n].cpu()
        joints = g1_rep_v1.decode_positions(feats).numpy()                  # (n,34,3)
        contacts = (feats[:, fc_sl] > 0.5).float().numpy()                  # (n,4)
        out.append((joints, contacts))
    return out


# ----------------------------------------------------------------------------
# TMR-G1 (scorer)
# ----------------------------------------------------------------------------
def load_tmr(ckpt_path, stats_path, text_cache, device):
    tmr, motion_rep, _dec = build_tmr_g1(text_emb_cache_path=text_cache, stats_path=stats_path,
                                         fps=FPS, device=device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    tmr.motion_encoder.load_state_dict(ck["motion_encoder"])
    tmr.text_encoder.load_state_dict(ck["text_encoder"])
    tmr.eval()
    for p in tmr.parameters():
        p.requires_grad_(False)
    return tmr, motion_rep


@torch.no_grad()
def tmr_encode_motion(tmr, motion_rep, posed_joints, device):
    T = posed_joints.shape[0]
    pj = torch.from_numpy(np.asarray(posed_joints, np.float32)).unsqueeze(0).to(device)
    feats = motion_rep(posed_joints=pj, to_normalize=True, to_canonicalize=True,
                       lengths=torch.tensor([T], device=device))
    out = tmr.motion_encoder({"x": feats, "mask": torch.ones(1, T, dtype=torch.bool, device=device)})
    return F.normalize(out.unbind(1)[0], dim=-1)[0].cpu().numpy()


@torch.no_grad()
def tmr_encode_text(tmr, text, device):
    raw, _ = tmr.raw_text_encoder([text])
    out = tmr.text_encoder({"x": raw.to(device), "mask": torch.ones(1, 1, dtype=torch.bool, device=device)})
    return F.normalize(out.unbind(1)[0], dim=-1)[0].cpu().numpy()


# ----------------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------------
def load_cases(testsuite, split, group, g1_root, cache_caps):
    gdir = Path(testsuite) / split / "text2motion" / group
    cases, n_no_text, n_no_gt = [], 0, 0
    for cdir in sorted(glob.glob(str(gdir / "*"))):
        cdir = Path(cdir)
        mp, sp = cdir / "meta.json", cdir / "seed_motion.json"
        if not (mp.is_file() and sp.is_file()):
            continue
        meta, seed = json.load(open(mp)), json.load(open(sp))
        text = (meta.get("text") or "").strip()
        if not text or text not in cache_caps:
            n_no_text += 1
            continue
        bvh = seed.get("bvh_path", "")
        session = bvh.split("/")[1] if "/" in bvh else ""
        npz = Path(g1_root) / session / f"{seed['move_name']}.npz"
        if not npz.is_file():
            n_no_gt += 1
            continue
        cases.append({
            "id": cdir.name, "text": text,
            "n_frames": max(4, int(round(float(meta["duration"]) * FPS))),
            "npz": str(npz), "seed": int(meta.get("seed", 0)),
            "crop_start": int(seed["crop_start_frame_index"]),
            "crop_end": int(seed["crop_end_frame_index"]),
        })
    return cases, n_no_text, n_no_gt


def load_gt_joints(case):
    with np.load(case["npz"], allow_pickle=False) as d:
        posed = np.asarray(d["posed_joints"], np.float32)
    n = posed.shape[0]
    s = FPS / 30.0  # crop indices are at 30fps; npz is 20fps
    a = max(0, int(round(case["crop_start"] * s)))
    b = min(n, int(round(case["crop_end"] * s)))
    if b - a < 4:
        return None
    return posed[a:b]


# ----------------------------------------------------------------------------
# Eval one group
# ----------------------------------------------------------------------------
def eval_group(den, diffusion, mdm_text_enc, mean, std, tmr, motion_rep, fsc, fcc,
               cases, device, n_steps, cfg_scale, batch):
    gen_E, gt_E, txt_E, ids = [], [], [], []
    t2m_sim, m2m_sim, t2m_gt_sim, skate, contact = [], [], [], [], []
    for i in range(0, len(cases), batch):
        chunk = cases[i:i + batch]
        gens = mdm_generate(den, diffusion, mdm_text_enc, [c["text"] for c in chunk],
                            [c["n_frames"] for c in chunk], mean, std, device, n_steps, cfg_scale)
        for c, (gj, gc) in zip(chunk, gens):
            gt = load_gt_joints(c)
            if gt is None:
                continue
            ge = tmr_encode_motion(tmr, motion_rep, gj, device)
            te = tmr_encode_text(tmr, c["text"], device)
            gte = tmr_encode_motion(tmr, motion_rep, gt, device)
            gen_E.append(ge); gt_E.append(gte); txt_E.append(te); ids.append(c["id"])
            t2m_sim.append(float(get_scores_unit(ge, te)))
            m2m_sim.append(float(get_scores_unit(ge, gte)))
            t2m_gt_sim.append(float(get_scores_unit(gte, te)))
            # ensure_batched adds the batch dim — pass everything UNbatched + 0-d length.
            L = torch.tensor(gj.shape[0])
            gjt = torch.from_numpy(gj); gct = torch.from_numpy(gc)
            skate.append(float(fsc(posed_joints=gjt, foot_contacts=gct, lengths=L)["foot_skate_from_pred_contacts"]))
            contact.append(float(fcc(posed_joints=gjt, foot_contacts=gct, lengths=L)["foot_contact_consistency"]))
    if not gen_E:
        return None
    M, T, G = np.stack(gen_E), np.stack(txt_E), np.stack(gt_E)
    m = compute_tmr_retrieval_metrics(M, T, gt_motion_emb=G, rounding=2)
    m["TMR/t2m_sim"] = round(float(np.mean(t2m_sim)), 4)
    m["TMR/m2m_sim"] = round(float(np.mean(m2m_sim)), 4)
    m["TMR/t2m_gt_sim"] = round(float(np.mean(t2m_gt_sim)), 4)
    m["foot_skate_from_pred_contacts"] = round(float(np.mean(skate)), 5)
    m["foot_contact_consistency"] = round(float(np.mean(contact)), 4)
    m["pool"] = len(gen_E)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdm-config", default="/home/jungbin_cho/TAP/kinematic_planner/configs/train_g1_rep_v1.yaml")
    ap.add_argument("--mdm-ckpt", default="/home/jungbin_cho/TAP/runs/mdm_g1_rep_v1_full/ckpt_final.pt")
    ap.add_argument("--tmr-ckpt", default="/home/jungbin_cho/TAP/runs/tmr_g1/v6/step_00015000.pt")
    ap.add_argument("--tmr-stats", default="/home/jungbin_cho/TAP/runs/tmr_g1/v6/tmr_g1_stats_v3")
    ap.add_argument("--text-cache", default="/home/jungbin_cho/kimodo_caches/benchmark_llm2vec.pt")
    ap.add_argument("--testsuite", default="/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark-20fps/testsuite")
    ap.add_argument("--g1-npz-root", default="/home/jungbin_cho/seed/g1_unified_npz")
    ap.add_argument("--out", default="/home/jungbin_cho/TAP/runs/mdm_g1_rep_v1_full/benchmark_eval.json")
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--cfg-scale", type=float, default=2.5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--groups", nargs="*", default=GROUPS)
    ap.add_argument("--limit", type=int, default=0, help="cap cases per group (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.mdm_config, []); OmegaConf.resolve(cfg)
    # shared cached text encoder for the MDM (benchmark cache)
    mdm_text_enc = build_text_encoder(OmegaConf.create({"type": "llm2vec", "cache_path": args.text_cache}), device=device)
    cache_caps = set(torch.load(args.text_cache, map_location="cpu", weights_only=False)["captions"])
    den = load_mdm(cfg, args.mdm_ckpt, device)
    diffusion = Diffusion(num_base_steps=int(cfg.get("num_base_steps", 1000))).to(device)
    mean = torch.from_numpy(np.load(cfg.data.mean_path).astype(np.float32)).to(device)
    std_raw = np.load(cfg.data.std_path).astype(np.float32)
    std = torch.from_numpy(np.where(std_raw < 1e-4, np.float32(1.0), std_raw)).to(device)
    print(f"MDM loaded (dim {den.motion_rep.motion_rep_dim})", flush=True)

    tmr, motion_rep = load_tmr(args.tmr_ckpt, args.tmr_stats, args.text_cache, device)
    skel = G1Skeleton34()
    fsc = FootSkateFromContacts(skeleton=skel, fps=float(FPS))
    fcc = FootContactConsistency(skeleton=skel, fps=float(FPS))
    print("TMR-G1 loaded", flush=True)

    results = {}
    for split in args.splits:
        for group in args.groups:
            cases, n_nt, n_ng = load_cases(args.testsuite, split, group, args.g1_npz_root, cache_caps)
            if args.limit:
                cases = cases[:args.limit]
            t0 = time.time()
            m = eval_group(den, diffusion, mdm_text_enc, mean, std, tmr, motion_rep, fsc, fcc,
                           cases, device, args.n_steps, args.cfg_scale, args.batch)
            key = f"{split}/{group}"
            results[key] = m
            if m:
                print(f"[{key}] pool={m['pool']} (skip txt={n_nt} gt={n_ng})  "
                      f"t2m R@3={m.get('TMR/t2m_R/R03')} (gt R@3={m.get('TMR/t2m_gt_R/R03')})  "
                      f"FID gen_text={m.get('TMR/FID/gen_text')}  skate={m.get('foot_skate_from_pred_contacts')}  "
                      f"{(time.time()-t0)/60:.1f}min", flush=True)

    # weighted aggregate over groups
    def agg(keys):
        tot = {}; w = 0
        for k in keys:
            m = results.get(k)
            if not m:
                continue
            n = m["pool"]; w += n
            for mk, mv in m.items():
                if mk == "pool" or not isinstance(mv, (int, float)):
                    continue
                tot[mk] = tot.get(mk, 0.0) + mv * n
        return {mk: round(mv / max(1, w), 3) for mk, mv in tot.items()} | {"pool": w} if w else None
    summary = {"per_group": results,
               "per_split": {sp: agg([f"{sp}/{g}" for g in args.groups]) for sp in args.splits},
               "overall": agg([f"{sp}/{g}" for sp in args.splits for g in args.groups]),
               "config": {"mdm_ckpt": args.mdm_ckpt, "tmr_ckpt": args.tmr_ckpt,
                          "n_steps": args.n_steps, "cfg_scale": args.cfg_scale}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

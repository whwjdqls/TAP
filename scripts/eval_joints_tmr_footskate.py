"""Score a set of G1 motions (joints) with the TMR-G1 critic (v14jit) + foot-skate.

Generic: used for BOTH the raw-MDM output (Stage 2) and the SONIC-tracked output
(Stage 5), so the numbers are directly comparable. Reads a dir of per-case npz
(produced by mdm_gen_for_cases.py or sonic_traj_to_joints.py) + their manifest.jsonl.

Each npz must hold:
  joints (T,34,3)  -- key auto-detected: 'joints' then 'joints_exec'
  prompt/text, and GT refs: gt_npz, crop_start, crop_end   (for retrieval/FID vs GT)
  contacts (T,4)   -- OPTIONAL; enables the contact-based foot-skate metrics

Metrics per group (kimodo metric code, comparable to published GT rows):
  TMR/t2m_R/*      text -> motion (the score)          TMR/t2m_gt_R/*  text -> GT
  TMR/m2m_R/*      motion -> GT                         TMR/FID/*       gen/gt/text FID
  TMR/{t2m_sim,m2m_sim,t2m_gt_sim}
  foot_skate_from_height, foot_skate_ratio              (contact-FREE, both raw & tracked)
  foot_skate_from_pred_contacts, foot_contact_consistency  (only if contacts present)

Uses TAP/kimodo (how v14jit was trained) — run as its own process, NOT with the
MDM generator (which uses kimodo_open).

    python scripts/eval_joints_tmr_footskate.py \
        --npz-dir runs/mdm_sonic/01_mdm_gen/eval \
        --tmr-ckpt runs/tmr_g1/v14jit/step_00024000.pt \
        --out runs/mdm_sonic/04_eval/mdm_raw.json --tag mdm_raw
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.chdir("/tmp")  # kimodo namespace guard (TAP/kimodo)
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn.functional as F

REPO = "/nfsdata/home/jungbin.cho/TAP"
FPS = 20


def load_tmr(ckpt, stats, text_cache, device):
    from tmr_g1.model.tmr_model import build_tmr_g1
    tmr, motion_rep, _dec = build_tmr_g1(text_emb_cache_path=text_cache,
                                         stats_path=stats, fps=FPS, device=device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    tmr.motion_encoder.load_state_dict(ck["motion_encoder"])
    tmr.text_encoder.load_state_dict(ck["text_encoder"])
    tmr.eval()
    for p in tmr.parameters():
        p.requires_grad_(False)
    return tmr, motion_rep


@torch.no_grad()
def enc_motion(tmr, mr, joints, device):
    T = joints.shape[0]
    pj = torch.from_numpy(np.asarray(joints, np.float32)).unsqueeze(0).to(device)
    feats = mr(posed_joints=pj, to_normalize=True, to_canonicalize=True,
               lengths=torch.tensor([T], device=device))
    out = tmr.motion_encoder({"x": feats, "mask": torch.ones(1, T, dtype=torch.bool, device=device)})
    return F.normalize(out.unbind(1)[0], dim=-1)[0].cpu().numpy()


@torch.no_grad()
def enc_text(tmr, text, device):
    raw, _ = tmr.raw_text_encoder([text])
    out = tmr.text_encoder({"x": raw.to(device), "mask": torch.ones(1, 1, dtype=torch.bool, device=device)})
    return F.normalize(out.unbind(1)[0], dim=-1)[0].cpu().numpy()


def load_gt(rec):
    if "gt_npz" not in rec:
        return None
    with np.load(rec["gt_npz"], allow_pickle=False) as d:
        posed = np.asarray(d["posed_joints"], np.float32)
    n = posed.shape[0]
    s = FPS / 30.0  # crop indices at 30fps; npz at 20fps
    a = max(0, int(round(rec["crop_start"] * s)))
    b = min(n, int(round(rec["crop_end"] * s)))
    return posed[a:b] if b - a >= 4 else None


def get_joints(d):
    for k in ("joints", "joints_exec"):
        if k in d:
            return np.asarray(d[k], np.float32)
    raise KeyError("npz has no 'joints' or 'joints_exec'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--tmr-ckpt", default=f"{REPO}/runs/tmr_g1/v14jit/step_00024000.pt")
    ap.add_argument("--tmr-stats", default=f"{REPO}/data/bones_seed/tmr_g1_stats_v3")
    ap.add_argument("--text-cache", default=f"{REPO}/data/bones_seed/eval_text_emb.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--per-sample-out", default=None,
                    help="jsonl of per-sample {case_id,text,t2m_sim,m2m_sim,t2m_gt_sim,foot_skate}")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    npz_dir = args.npz_dir if os.path.isabs(args.npz_dir) else os.path.join(REPO, args.npz_dir)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)

    from kimodo.metrics.tmr import compute_tmr_retrieval_metrics, get_scores_unit
    from kimodo.metrics.foot_skate import (
        FootSkateFromHeight, FootSkateRatio, FootSkateFromContacts, FootContactConsistency,
    )
    from kimodo.skeleton import G1Skeleton34

    tmr, mr = load_tmr(args.tmr_ckpt, args.tmr_stats, args.text_cache, device)
    skel = G1Skeleton34()
    fsh = FootSkateFromHeight(skeleton=skel, fps=float(FPS))
    fsh50 = FootSkateFromHeight(skeleton=skel, fps=50.0)   # native SONIC rate (no resample smoothing)
    fsr = FootSkateRatio(skeleton=skel, fps=float(FPS))
    fsc = FootSkateFromContacts(skeleton=skel, fps=float(FPS))
    fcc = FootContactConsistency(skeleton=skel, fps=float(FPS))

    manifest = os.path.join(npz_dir, "manifest.jsonl")
    recs = [json.loads(l) for l in open(manifest) if l.strip()]
    if args.limit:
        recs = recs[:args.limit]

    gen_E, gt_E, txt_E = [], [], []
    t2m_sim, m2m_sim, t2m_gt_sim = [], [], []
    _psamp = []
    skate_h, skate_r, skate_c, contact_c = [], [], [], []
    skate_h50 = []
    n_no_gt = 0
    for i, rec in enumerate(recs):
        d = np.load(os.path.join(npz_dir, rec["file"]), allow_pickle=True)
        joints = get_joints(d)
        text = rec.get("text") or str(d.get("prompt", ""))
        gt = load_gt(rec)
        if gt is None:
            n_no_gt += 1
            continue
        ge = enc_motion(tmr, mr, joints, device)
        te = enc_text(tmr, text, device)
        gte = enc_motion(tmr, mr, gt, device)
        gen_E.append(ge); txt_E.append(te); gt_E.append(gte)
        t2m_sim.append(float(get_scores_unit(ge, te)))
        m2m_sim.append(float(get_scores_unit(ge, gte)))
        t2m_gt_sim.append(float(get_scores_unit(gte, te)))
        if args.per_sample_out:
            _psamp.append({"case_id": rec.get("case_id", Path(rec["file"]).stem),
                           "file": rec["file"], "text": text,
                           "t2m_sim": t2m_sim[-1], "m2m_sim": m2m_sim[-1], "t2m_gt_sim": t2m_gt_sim[-1]})
        jt = torch.from_numpy(joints); L = torch.tensor(joints.shape[0])
        skate_h.append(float(fsh(posed_joints=jt, lengths=L)["foot_skate_from_height"]))
        if "joints_50fps" in d:
            j50 = torch.from_numpy(np.asarray(d["joints_50fps"], np.float32))
            skate_h50.append(float(fsh50(posed_joints=j50,
                                         lengths=torch.tensor(j50.shape[0]))["foot_skate_from_height"]))
        if "contacts" in d:
            ct = torch.from_numpy(np.asarray(d["contacts"], np.float32))
            skate_r.append(float(fsr(posed_joints=jt, foot_contacts=ct, lengths=L)["foot_skate_ratio"]))
            skate_c.append(float(fsc(posed_joints=jt, foot_contacts=ct, lengths=L)["foot_skate_from_pred_contacts"]))
            contact_c.append(float(fcc(posed_joints=jt, foot_contacts=ct, lengths=L)["foot_contact_consistency"]))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(recs)}", flush=True)

    M, T, G = np.stack(gen_E), np.stack(txt_E), np.stack(gt_E)
    m = compute_tmr_retrieval_metrics(M, T, gt_motion_emb=G, rounding=2)
    m["TMR/t2m_sim"] = round(float(np.mean(t2m_sim)), 4)
    m["TMR/m2m_sim"] = round(float(np.mean(m2m_sim)), 4)
    m["TMR/t2m_gt_sim"] = round(float(np.mean(t2m_gt_sim)), 4)
    m["foot_skate_from_height"] = round(float(np.mean(skate_h)), 5)
    if skate_h50:
        m["foot_skate_from_height_50fps"] = round(float(np.mean(skate_h50)), 5)
    if skate_c:
        m["foot_skate_ratio"] = round(float(np.mean(skate_r)), 5)
        m["foot_skate_from_pred_contacts"] = round(float(np.mean(skate_c)), 5)
        m["foot_contact_consistency"] = round(float(np.mean(contact_c)), 4)
    m["pool"] = len(gen_E)
    m["skipped_no_gt"] = n_no_gt

    out = {"tag": args.tag, "npz_dir": npz_dir, "tmr_ckpt": args.tmr_ckpt, "metrics": m}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    if args.per_sample_out:
        pp = args.per_sample_out if os.path.isabs(args.per_sample_out) else os.path.join(REPO, args.per_sample_out)
        Path(pp).parent.mkdir(parents=True, exist_ok=True)
        with open(pp, "w") as f:
            for r in _psamp:
                f.write(json.dumps(r) + "\n")
        print(f"  per-sample scores ({len(_psamp)}) -> {pp}", flush=True)
    print(f"\n[{args.tag}] pool={m['pool']} (no_gt={n_no_gt})  "
          f"t2m R@3={m.get('TMR/t2m_R/R03')} (gt R@3={m.get('TMR/t2m_gt_R/R03')})  "
          f"FID gen_text={m.get('TMR/FID/gen_text')}  "
          f"skate_h={m.get('foot_skate_from_height')}", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

"""Sample G1 motions from text prompts with a trained g1_rep_v1 MDM.

Loads a checkpoint (EMA weights overlaid), encodes prompts with the live
LLM2Vec encoder (arbitrary text, not just cached captions), runs CFG-guided
DDIM sampling (default 50 steps), then decodes to:
  * ric joint positions (exact),
  * executable G1 pose via root + 29 qpos angles -> FK,
  * MuJoCo qpos (T,36) for rendering / SONIC tracking.

    python sample_g1.py --config configs/train_g1_rep_v1.yaml \
        --ckpt /home/jungbin_cho/TAP/runs/mdm_g1_rep_v1_full/ckpt_final.pt \
        --n-steps 50 --num-frames 120 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from omegaconf import OmegaConf

from kimodo.scripts.train import build_denoiser_from_model_config, build_text_encoder, encode_texts, load_config
from kimodo.scripts.train_w_hml3d import _cfg_sample
from kimodo.model.diffusion import Diffusion
from kimodo.exports.mujoco import MujocoQposConverter

import g1_rep_v1

XML = "/home/jungbin_cho/TAP/kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml"

DEFAULT_PROMPTS = [
    "a person walks forward",
    "a person walks backward slowly",
    "a person turns around and sits down",
    "a person waves their right hand",
    "a person raises both arms above their head",
    "a person kicks with the right leg",
    "a person crouches down and stands back up",
    "a person jumps",
]


def load_denoiser(cfg, ckpt_path, device, use_ema=True):
    den = build_denoiser_from_model_config(
        cfg.model_config_path, cfg.get("stats_path", ""), fps_override=cfg.get("denoiser_fps_override"),
    ).to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = dict(ck["denoiser"])
    n_ema = 0
    if use_ema and ck.get("ema"):
        for k, v in ck["ema"].items():
            if k in sd and sd[k].shape == v.shape:
                sd[k] = v; n_ema += 1
    den.load_state_dict(sd, strict=True)
    den.eval()
    for p in den.parameters():
        p.requires_grad_(False)
    print(f"loaded {ckpt_path} (step={ck.get('step')}), EMA overlaid on {n_ema} tensors")
    return den


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts", nargs="*", default=None, help="prompts (default: a built-in set)")
    ap.add_argument("--prompts-file", default=None, help="text file, one prompt per line")
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--num-frames", type=int, default=120)
    ap.add_argument("--cfg-scale", type=float, default=2.5)
    ap.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/home/jungbin_cho/TAP/runs/mdm_g1_rep_v1_full/samples")
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config, [])
    OmegaConf.resolve(cfg)

    prompts = args.prompts
    if args.prompts_file:
        prompts = [ln.strip() for ln in open(args.prompts_file) if ln.strip()]
    if not prompts:
        prompts = DEFAULT_PROMPTS

    den = load_denoiser(cfg, args.ckpt, device, use_ema=not args.no_ema)
    diffusion = Diffusion(num_base_steps=int(cfg.get("num_base_steps", 1000))).to(device)
    motion_rep = den.motion_rep
    D = motion_rep.motion_rep_dim
    # Decode (FK + qpos conversion) runs on CPU — kimodo's FK builds its joint
    # index mask on CPU, so a fresh CPU skeleton/converter is required (the
    # model's skeleton lives on the GPU).
    from kimodo.skeleton import G1Skeleton34
    skel = G1Skeleton34()
    conv = MujocoQposConverter(skel, xml_path=XML)

    mean = torch.from_numpy(np.load(cfg.data.mean_path).astype(np.float32)).to(device)
    std_raw = np.load(cfg.data.std_path).astype(np.float32)
    std = torch.from_numpy(np.where(std_raw < 1e-4, np.float32(1.0), std_raw)).to(device)

    print("building live LLM2Vec text encoder (loads the 8B model)...")
    text_encoder = build_text_encoder(OmegaConf.create({"type": "llm2vec", "device": "auto"}), device=device)

    B = len(prompts)
    T = int(args.num_frames)
    text_feat, text_pad = encode_texts(text_encoder, prompts, device)
    pad_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    first_heading = torch.zeros(B, device=device)
    motion_mask = torch.zeros(B, T, D, device=device)
    observed = torch.zeros(B, T, D, device=device)

    print(f"sampling {B} prompts x {T} frames, {args.n_steps} steps ({args.sampler}), cfg={args.cfg_scale}")
    gen = _cfg_sample(
        den, diffusion, text_feat=text_feat, text_pad_mask=text_pad, pad_mask=pad_mask,
        first_heading=first_heading, motion_mask=motion_mask, observed=observed,
        n_steps=args.n_steps, cfg_scale=args.cfg_scale, device=device, sampler=args.sampler,
    )                                                  # (B,T,D) normalized
    gen = gen.float() * std + mean                     # unnormalize -> raw g1_rep_v1

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, prompt in enumerate(prompts):
        feats = gen[i].cpu()                            # (T,142)
        joints_ric = g1_rep_v1.decode_positions(feats).numpy()           # (T,34,3) exact positions
        dec = g1_rep_v1.decode_qpos_to_joints(feats, conv, skel)         # executable
        joints_exec = dec["posed_joints"].numpy()
        qpos = conv.to_qpos(dec["local_rot_mats"].unsqueeze(0), dec["root_positions"].unsqueeze(0),
                            root_quat_w_first=True, mujoco_rest_zero=False)
        qpos = (qpos[0] if qpos.dim() == 3 else qpos).cpu().numpy()      # (T,36) mujoco qpos
        slug = prompt.lower().replace(" ", "_")[:40]
        fn = out_dir / f"{i:02d}_{slug}.npz"
        np.savez(fn, prompt=prompt, features=feats.numpy(), joints_ric=joints_ric,
                 joints_exec=joints_exec, qpos=qpos,
                 joint_angles=feats[:, g1_rep_v1.SLICE_DICT["joint_angles"]].numpy())
        manifest.append({"idx": i, "prompt": prompt, "file": fn.name, "frames": T})
        print(f"  [{i}] {prompt!r:55s} -> {fn.name}")
    json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=2)
    print(f"\nwrote {B} samples to {out_dir}")


if __name__ == "__main__":
    main()

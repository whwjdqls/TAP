"""Train an MDM one-stage denoiser on g1_rep_v1 (142-D), text-conditioned.

Reuses kimodo's Bones-SEED MDM machinery (denoiser build, diffusion, text
encoder, train_one_step, EMA, checkpointing) unchanged; swaps in the G1
dataset + the flat MDM masked-L2 loss. Run from this directory:

    python train_g1.py --config configs/train_g1_rep_v1.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # g1_* importable; kimodo from site-packages

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from kimodo.scripts.train import (
    ModelEMA, build_denoiser_from_model_config, build_text_encoder,
    load_checkpoint, load_config, maybe_barrier, save_checkpoint,
    setup_distributed, train_one_step, _resolve_num_steps, cleanup_distributed,
)
from kimodo.scripts.train_w_hml3d_native import MDMNativeL2Loss
from kimodo.model.diffusion import Diffusion

import g1_rep_v1
from g1_dataset import G1RepV1TextMotionDataset, build_g1_collate_fn

log = logging.getLogger("train_g1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    OmegaConf.resolve(cfg)

    env = setup_distributed()
    seed = int(cfg.trainer.seed) + env.rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda", env.local_rank) if torch.cuda.is_available() else torch.device("cpu")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(output_dir / f"train_rank{env.rank}.log"), logging.StreamHandler(sys.stdout)],
    )
    if env.is_main:
        OmegaConf.save(cfg, output_dir / "config.yaml")
        log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # ---- denoiser + diffusion ----
    denoiser = build_denoiser_from_model_config(
        cfg.model_config_path, cfg.get("stats_path", ""),
        fps_override=cfg.get("denoiser_fps_override"),
    ).to(device)
    diffusion = Diffusion(num_base_steps=int(_resolve_num_steps(cfg))).to(device)
    motion_rep = denoiser.motion_rep
    assert motion_rep.motion_rep_dim == g1_rep_v1.FEAT_DIM, (
        f"expected motion_rep_dim={g1_rep_v1.FEAT_DIM}, got {motion_rep.motion_rep_dim}")
    log.info("Denoiser built: motion_rep_dim=%d nbjoints=%d", motion_rep.motion_rep_dim, motion_rep.nbjoints)

    # ---- dataset / loader ----
    mean = np.load(cfg.data.mean_path).astype(np.float32)
    std = np.load(cfg.data.std_path).astype(np.float32)

    def _ds():
        return G1RepV1TextMotionDataset(
            feat_root=cfg.data.feat_root, text_index=cfg.data.text_index,
            mean=mean, std=std, fps=int(cfg.data.fps),
            window_size=int(cfg.data.window_size), max_motion_length=int(cfg.data.max_motion_length),
            min_motion_len=int(cfg.data.min_motion_len), unit_length=int(cfg.data.unit_length),
            clip_normalized=cfg.data.get("clip_normalized"),
            motion_rep=motion_rep, skeleton=motion_rep.skeleton, seed=seed,
        )
    if env.is_main:
        dataset = _ds()
    maybe_barrier()
    if not env.is_main:
        dataset = _ds()
    maybe_barrier()

    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True, seed=int(cfg.trainer.seed)) if env.is_distributed else None
    loader = DataLoader(
        dataset, batch_size=int(cfg.trainer.batch_size), shuffle=(sampler is None), sampler=sampler,
        num_workers=int(cfg.data.num_workers), pin_memory=bool(cfg.data.pin_memory), drop_last=True,
        collate_fn=build_g1_collate_fn(), persistent_workers=bool(cfg.data.num_workers),
    )

    # ---- optim / sched / amp / ema ----
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=float(cfg.trainer.lr),
                                  betas=tuple(cfg.trainer.betas), weight_decay=float(cfg.trainer.weight_decay))
    warmup = int(cfg.trainer.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: float(s) / max(1, warmup) if warmup > 0 and s < warmup else 1.0)
    mp = str(cfg.trainer.mixed_precision).lower()
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mp, None)
    scaler = GradScaler() if mp == "fp16" else None
    ema = ModelEMA(denoiser, decay=float(cfg.trainer.ema_decay)) if float(cfg.trainer.ema_decay) > 0 else None

    # ---- text encoder (cached LLM2Vec) ----
    text_encoder = build_text_encoder(cfg.text_encoder, device=device)
    log.info("Text encoder: %s", type(text_encoder).__name__)

    if env.is_distributed:
        denoiser = DDP(denoiser, device_ids=[env.local_rank], find_unused_parameters=False)
    loss_fn = MDMNativeL2Loss(motion_rep).to(device)

    start_step = 0
    if cfg.trainer.get("resume_from"):
        start_step = load_checkpoint(Path(cfg.trainer.resume_from), denoiser, optimizer, scheduler, scaler, ema, device)

    tb = None
    if env.is_main:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=str(output_dir / "tb"))

    num_steps = int(cfg.trainer.num_steps)
    grad_accum = int(cfg.trainer.grad_accum)
    log_every, print_every = int(cfg.trainer.log_every), int(cfg.trainer.get("print_every", 50))
    ckpt_every = int(cfg.trainer.ckpt_every)
    grad_clip = float(cfg.trainer.grad_clip)
    grad_guard = float(cfg.trainer.get("grad_guard_max", 100.0))
    ema_every = int(cfg.trainer.ema_every)
    text_drop = float(cfg.trainer.text_drop_prob)

    log.info("Training %d -> %d steps | %d samples | batch %d", start_step, num_steps, len(dataset), int(cfg.trainer.batch_size))
    denoiser.train()
    step, epoch = start_step, 0
    while step < num_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            agg: Dict[str, float] = {}
            for _ in range(grad_accum):
                losses = train_one_step(denoiser, diffusion, loss_fn, batch, text_encoder, device,
                                        text_drop, autocast_dtype, constraint_sampler=None)
                lo = losses["loss"] / grad_accum
                (scaler.scale(lo).backward() if scaler is not None else lo.backward())
                for k, v in losses.items():
                    agg[k] = agg.get(k, 0.0) + float(v.detach()) / grad_accum
            if scaler is not None:
                scaler.unscale_(optimizer)
            params = denoiser.module.parameters() if hasattr(denoiser, "module") else denoiser.parameters()
            gnorm = torch.nn.utils.clip_grad_norm_(params, grad_clip if grad_clip > 0 else float("inf"))
            skip = (not torch.isfinite(gnorm)) or float(gnorm) > grad_guard
            agg["grad_norm"] = float(gnorm)
            if skip:
                log.warning("step %d skip (grad_norm=%.3e)", step + 1, float(gnorm))
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.update()
            else:
                if scaler is not None:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
            if ema is not None and step % ema_every == 0 and not skip:
                ema.update(denoiser.module if hasattr(denoiser, "module") else denoiser)
            step += 1
            dt = time.time() - t0

            if env.is_main and print_every and step % print_every == 0:
                try:
                    print(f"[step {step:>7d}] loss={agg['loss']:.4f} grad={agg['grad_norm']:.2f} dt={dt:.2f}s", flush=True)
                except OSError:
                    pass
            if env.is_main and step % log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                log.info("step=%d loss=%.4f grad=%.2f lr=%.2e dt=%.2fs", step, agg["loss"], agg["grad_norm"], lr_now, dt)
                if tb is not None:
                    for k, v in agg.items():
                        tb.add_scalar(f"train/{k}", v, step)
                    tb.add_scalar("train/lr", lr_now, step)
                    tb.add_scalar("train/step_time", dt, step)
            if env.is_main and ckpt_every and step % ckpt_every == 0:
                save_checkpoint(output_dir / f"ckpt_step{step:07d}.pt", denoiser, optimizer, scheduler, scaler, ema, step, cfg)
                latest = output_dir / "latest.pt"
                try:
                    if latest.exists() or latest.is_symlink():
                        latest.unlink()
                    latest.symlink_to(f"ckpt_step{step:07d}.pt")
                except OSError:
                    shutil.copy2(output_dir / f"ckpt_step{step:07d}.pt", latest)
            if step >= num_steps:
                break
        epoch += 1

    if env.is_main:
        save_checkpoint(output_dir / "ckpt_final.pt", denoiser, optimizer, scheduler, scaler, ema, step, cfg)
    cleanup_distributed(env)


if __name__ == "__main__":
    main()

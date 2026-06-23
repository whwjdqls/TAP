"""Minimal motion-VAE trainer (G1 / Bones-SEED).

Counterpart of `tmr_g1.training.train` with the text branch removed.

Steps per batch (B motion clips):
  1. Load precomputed (normalized + canonicalized) `TMRMotionRep` features
     (B, T_max, 210) + mask from the dataset worker.
  2. Motion encoder -> mu, logvar -> reparam -> z.
  3. Decoder(z, T) -> recon features; masked L2 vs the input features.
  4. KL(mu, logvar) to the standard normal prior.
  5. loss = lambda_recon * recon + lambda_kl * kl.
  6. Step AdamW; cosine/constant LR with warmup; grad clip.

The dataloader is identical to TMR-G1's (the same natural/single/multi caption
pools define the motion *windows*); captions are simply ignored here.

Usage:
    python vae_g1/training/train.py \
        --data-root  data/bones_seed/g1_unified_npz \
        --feat-root  data/bones_seed/g1_feat_npz \
        --natural-csv  data/bones_seed/metadata/metadata/seed_metadata_v004.csv \
        --single-jsonl data/bones_seed/metadata/metadata/seed_metadata_v002_temporal_labels.jsonl \
        --multi-jsonl  data/bones_seed/multi_timeline.jsonl \
        --train-split  data/kimodo_benchmark/splits/train_split_paths.txt \
        --stats-path   data/bones_seed/tmr_g1_stats_v3 \
        --out-dir runs/vae_g1/v0
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from torch.utils.data import DataLoader

from vae_g1.data.g1_dataset import G1BonesSeedDataset, collate_motion_batch
from vae_g1.data.motion_dataset import G1MotionDataset
from vae_g1.model.vae_model import build_vae_g1
from vae_g1.training.losses import kl_to_standard_normal, reconstruction_loss

log = logging.getLogger("vae_g1")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _cosine_lr(step, warmup, total, base_lr, min_lr=1e-6):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * p))


def _constant_lr(step, warmup, total, base_lr, min_lr=1e-6):
    """Linear warmup, then hold base_lr flat (no decay)."""
    if step < warmup:
        return base_lr * step / max(1, warmup)
    return base_lr


def _lr_at(schedule, step, warmup, total, base_lr):
    if schedule == "constant":
        return _constant_lr(step, warmup, total, base_lr)
    return _cosine_lr(step, warmup, total, base_lr)


def _write_config(args, out_dir: Path) -> None:
    import json
    import subprocess

    def _git(cmd):
        try:
            return subprocess.check_output(
                ["git", "-C", str(Path(__file__).resolve().parent)] + cmd,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:  # noqa: BLE001
            return None

    cfg = {
        "model_name": "VAE-G1-SEED-v1",
        "architecture": {
            "type": "Motion VAE (ACTOR-style transformer encoder + decoder)",
            "skeleton": "G1Skeleton34",
            "motion_rep": "TMRMotionRep",
            "motion_rep_dim": 210,
            "latent_dim": args.latent_dim,
            "ff_size": args.ff_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "vae": True,
            "activation": "gelu",
        },
        "data": {
            "fps": args.fps,
            "canonicalize": bool(args.canonicalize),
            "data_root": args.data_root,
            "natural_csv": args.natural_csv,
            "single_jsonl": args.single_jsonl,
            "multi_jsonl": args.multi_jsonl,
            "train_split": args.train_split,
            "stats_path": args.stats_path,
            "window_mode": args.window_mode,
            "sampler_mode": args.sampler_mode,
            "min_clip_sec": args.min_clip_sec,
            "max_clip_sec": args.max_clip_sec,
            "oversample": args.oversample,
            "aug_feat_noise_std": args.aug_feat_noise_std,
            "aug_time_jitter_sec": args.aug_time_jitter_sec,
        },
        "optim": {
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lr_schedule": f"{args.lr_schedule} with warmup",
            "lr_total": args.lr_total if args.lr_total > 0 else args.max_steps,
            "warmup": args.warmup,
            "max_steps": args.max_steps,
            "batch": args.batch,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
        },
        "loss": {
            "lambda_recon": args.lambda_recon,
            "lambda_kl": args.lambda_kl,
        },
        "provenance": {
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        },
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    log.info("wrote %s", out_dir / "config.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--train-split", required=True)
    p.add_argument("--stats-path", required=True)
    # Caption metadata — only needed for --window-mode caption.
    p.add_argument("--natural-csv", default="")
    p.add_argument("--single-jsonl", default="")
    p.add_argument("--multi-jsonl", default="")
    p.add_argument("--window-mode", default="random", choices=["random", "caption"],
                   help="random: arbitrary random crops over ALL motions (no "
                        "captions needed); caption: caption-defined windows "
                        "(natural/single/multi pools, like tmr_g1).")
    p.add_argument("--min-clip-sec", type=float, default=2.0,
                   help="[random mode] min random-window length")
    p.add_argument("--max-clip-sec", type=float, default=10.0,
                   help="[random mode] max random-window length")
    p.add_argument("--oversample", type=int, default=1,
                   help="[random mode] random crops per motion per epoch")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--warmup", type=int, default=2_000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-schedule", default="cosine", choices=["cosine", "constant"],
                   help="constant = warmup then hold base_lr flat (no decay)")
    p.add_argument("--lr-total", type=int, default=0,
                   help="cosine decay horizon; 0 -> use max_steps.")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--lambda-kl", type=float, default=1e-4)
    p.add_argument("--lambda-recon", type=float, default=1.0)
    p.add_argument("--feat-root", default="",
                   help="precomputed-feature mirror tree; speeds up dataloader")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--canonicalize", action="store_true", default=True,
                   help="canonicalize motion features (heading/position invariant)")
    p.add_argument("--no-canonicalize", dest="canonicalize", action="store_false")
    p.add_argument("--sampler-mode", default="round_robin",
                   choices=["round_robin", "proportional"],
                   help="round_robin: 1:1:1 source mix; proportional: each "
                        "unique window ~once/epoch (mix follows pool sizes).")
    # Model architecture (matches the TMR-G1 motion encoder).
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--ff-size", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--aug-feat-noise-std", type=float, default=0.0,
                   help="train-time additive Gaussian noise (normalized feat space); 0=off")
    p.add_argument("--aug-time-jitter-sec", type=float, default=0.0,
                   help="train-time +-jitter on slice start/end boundaries (sec); 0=off")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=5_000)
    p.add_argument("--eval-every", type=int, default=5_000,
                   help="run reconstruction eval every N steps (0=off)")
    p.add_argument("--eval-testsuite", default="/nfsdata/home/jungbin.cho/TAP/data/kimodo_benchmark/testsuite")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--resume-from", default="",
                   help="checkpoint to resume encoder/decoder/optimizer + step")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_config(args, out_dir)

    torch.manual_seed(args.seed)

    # CPU motion_rep for the dataset workers (features computed per-sample at
    # batch=1 so canonicalize broadcasts correctly).
    from kimodo.motion_rep.reps.tmr_motionrep import TMRMotionRep
    from kimodo.skeleton import G1Skeleton34
    cpu_motion_rep = TMRMotionRep(
        skeleton=G1Skeleton34(), fps=args.fps, stats_path=args.stats_path,
    )

    if args.window_mode == "random":
        dataset = G1MotionDataset(
            data_root=args.data_root,
            split_paths=args.train_split,
            fps=args.fps,
            seed=args.seed,
            motion_rep=cpu_motion_rep,
            canonicalize=args.canonicalize,
            feat_root=(args.feat_root or None),
            min_clip_sec=args.min_clip_sec,
            max_clip_sec=args.max_clip_sec,
            oversample=args.oversample,
            aug_feat_noise_std=args.aug_feat_noise_std,
        )
    else:
        if not (args.natural_csv and args.single_jsonl and args.multi_jsonl):
            raise SystemExit("--window-mode caption requires "
                             "--natural-csv/--single-jsonl/--multi-jsonl")
        dataset = G1BonesSeedDataset(
            data_root=args.data_root,
            natural_csv_path=args.natural_csv,
            temporal_labels_path=args.single_jsonl,
            multi_timeline_path=args.multi_jsonl,
            split_paths=args.train_split,
            fps=args.fps,
            seed=args.seed,
            motion_rep=cpu_motion_rep,
            canonicalize=args.canonicalize,
            sampler_mode=args.sampler_mode,
            feat_root=(args.feat_root or None),
            aug_feat_noise_std=args.aug_feat_noise_std,
            aug_time_jitter_sec=args.aug_time_jitter_sec,
        )
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
        collate_fn=collate_motion_batch, persistent_workers=args.num_workers > 0,
    )

    vae, motion_rep = build_vae_g1(
        stats_path=args.stats_path,
        fps=args.fps,
        latent_dim=args.latent_dim,
        ff_size=args.ff_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        device=args.device,
    )
    vae.train()
    params = list(vae.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(log_dir=str(out_dir / "tb"))

    evaluator = None
    if args.eval_every and Path(args.eval_testsuite).is_dir():
        from vae_g1.training.inline_eval import ReconEvaluator
        evaluator = ReconEvaluator(
            testsuite=args.eval_testsuite,
            g1_npz_root=args.data_root,
            motion_rep=cpu_motion_rep,
            split="content", grp="overview",
            fps=args.fps, canonicalize=args.canonicalize, device=args.device,
        )
        log.info("inline recon eval ready: content/overview pool=%d (missing=%d)",
                 evaluator.n, evaluator.miss)

    start_step = 0
    if args.resume_from and Path(args.resume_from).is_file():
        ck = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        vae.encoder.load_state_dict(ck["motion_encoder"])
        vae.decoder.load_state_dict(ck["motion_decoder"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start_step = int(ck.get("step", 0))
        log.info("resumed from %s at step %d", args.resume_from, start_step)

    step = start_step
    t0 = time.time()
    for epoch in range(10_000):  # virtual, capped by max_steps
        for batch in loader:
            feats = batch["features"].to(args.device, non_blocking=True)
            mask = batch["mask"].to(args.device, non_blocking=True)

            pred, mu, logvar, z = vae(feats, mask, training=True)
            l_recon = reconstruction_loss(pred, feats, mask)
            l_kl = kl_to_standard_normal(mu, logvar)
            loss = args.lambda_recon * l_recon + args.lambda_kl * l_kl

            lr_total = args.lr_total if args.lr_total > 0 else args.max_steps
            lr = _lr_at(args.lr_schedule, step, args.warmup, lr_total, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            step += 1

            if step % args.log_every == 0:
                dt = time.time() - t0
                log.info(
                    "step=%d lr=%.2e loss=%.4f recon=%.4f kl=%.4f  rate=%.1f steps/s",
                    step, lr, loss.item(), l_recon.item(), l_kl.item(),
                    step / max(1, dt),
                )
                tb.add_scalar("loss/total", loss.item(), step)
                tb.add_scalar("loss/recon", l_recon.item(), step)
                tb.add_scalar("loss/kl_motion", l_kl.item(), step)
                tb.add_scalar("train/lr", lr, step)
                tb.add_scalar("train/steps_per_sec", step / max(1, dt), step)
            if evaluator is not None and step % args.eval_every == 0:
                em = evaluator.evaluate(vae)
                log.info(
                    "EVAL[content/overview] step=%d recon_mse=%.4f kl=%.2f "
                    "mu_std=%.4f active_units=%d (pool=%d)",
                    step, em["recon_mse"], em["kl"], em["mu_std"],
                    em["active_units"], evaluator.n,
                )
                for k, v in em.items():
                    tb.add_scalar(f"eval_content_overview/{k}", v, step)
            if step % args.ckpt_every == 0:
                ckpt = {
                    "step": step,
                    "motion_encoder": vae.encoder.state_dict(),
                    "motion_decoder": vae.decoder.state_dict(),
                    "optimizer": opt.state_dict(),
                    "args": vars(args),
                }
                torch.save(ckpt, out_dir / f"step_{step:08d}.pt")
                torch.save(ckpt, out_dir / "last.pt")
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    log.info("done after %d steps", step)


if __name__ == "__main__":
    main()

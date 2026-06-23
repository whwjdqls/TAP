"""Minimal TMR-G1 trainer.

Steps per batch (B motion clips):
  1. Load posed_joints (B, T_max, 30, 3) + mask + raw text.
  2. Motion features: `TMRMotionRep(G1Skeleton30, fps=20)(posed_joints, to_normalize=True)`.
  3. Motion encoder → mu_m, logvar_m → reparam → z_m.
  4. Text features:   `CachedTextEncoder(text)` → (B, 1, llm_dim) → top text encoder
                     → mu_t, logvar_t → reparam → z_t.
  5. Decoder(z_m, T) → recon features; L2 vs motion features (masked).
  6. KL(mu_m, logvar_m) + KL(mu_t, logvar_t).
  7. InfoNCE between L2-normalized z_m and z_t with duplicate-text masking.
  8. Step AdamW; cosine LR with warmup.

Usage:
    python tmr_g1/training/train.py \
        --data-root data/bones_seed/g1_20fps_npz \
        --natural-csv data/bones_seed/metadata/metadata/seed_metadata_v004.csv \
        --single-jsonl data/bones_seed/metadata/metadata/seed_metadata_v002_temporal_labels.jsonl \
        --multi-jsonl data/bones_seed/multi_timeline.jsonl \
        --train-split data/kimodo_benchmark/splits/train_split_paths.txt \
        --text-emb-cache data/bones_seed/g1_text_emb.pt \
        --stats-path data/bones_seed/tmr_g1_stats \
        --out-dir runs/tmr_g1/v0
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tmr_g1.data.g1_dataset import G1BonesSeedDataset, collate_motion_batch
from tmr_g1.model.tmr_model import build_tmr_g1
from tmr_g1.training.losses import (
    info_nce_membank, kl_to_standard_normal, reconstruction_loss,
)

log = logging.getLogger("tmr_g1")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _reparam(mu: torch.Tensor, logvar: torch.Tensor, training: bool) -> torch.Tensor:
    if not training:
        return mu
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


def _encode_motion(tmr, feats, mask, training):
    # feats: (B, T_max, motion_rep_dim) — precomputed (normalized + canonicalized)
    # per-sample in the dataset worker. mask: (B, T_max).
    motion_inputs = {"x": feats, "mask": mask}
    out = tmr.motion_encoder(motion_inputs)  # (B, 2, latent) if vae
    mu, logvar = out.unbind(1)
    z = _reparam(mu, logvar, training)
    return z, mu, logvar, feats


def _encode_text(tmr, texts, training):
    # tmr.raw_text_encoder is the CachedTextEncoder.
    raw_feat, raw_len = tmr.raw_text_encoder(texts)  # (B, 1, llm_dim)
    mask = torch.ones(raw_feat.shape[0], 1, dtype=torch.bool, device=raw_feat.device)
    out = tmr.text_encoder({"x": raw_feat, "mask": mask})
    mu, logvar = out.unbind(1)
    z = _reparam(mu, logvar, training)
    return z, mu, logvar


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
    """Write a config.json capturing all hyperparameters + model architecture
    + data provenance at the start of the run."""
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
        "model_name": "TMR-G1-SEED-v1",
        # Architecture — identical to nvidia/TMR-SOMA-RP-v1 except skeleton+fps.
        "architecture": {
            "type": "TMR dual-encoder (ACTOR-style transformer, VAE)",
            "skeleton": "G1Skeleton34",
            "motion_rep": "TMRMotionRep",
            "motion_rep_dim": 210,
            "text_encoder_input": "LLM2Vec Meta-Llama-3-8B-Instruct (mntp-supervised), dim 4096, frozen",
            "latent_dim": args.latent_dim,
            "ff_size": args.ff_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "vae": True,
            "activation": "gelu",
            "unit_vector": True,
        },
        "data": {
            "fps": args.fps,
            "canonicalize": bool(args.canonicalize),
            "data_root": args.data_root,
            "natural_csv": args.natural_csv,
            "single_jsonl": args.single_jsonl,
            "multi_jsonl": args.multi_jsonl,
            "train_split": args.train_split,
            "text_emb_cache": args.text_emb_cache,
            "stats_path": args.stats_path,
            "sources": ["natural", "single", "multi"],
            "sampler_mode": args.sampler_mode,
            "max_clip_sec": 10.0,
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
            "cross_modal_recon": args.cross_modal_recon,
            "lambda_kl": args.lambda_kl,
            "lambda_contrastive": args.lambda_contrastive,
            "info_nce_temp": args.info_nce_temp,
            "queue_size": args.queue_size,
            "text_dup_threshold": args.text_dup_threshold,
            "info_nce_dup_mask": "projected text-cos > text_dup_threshold (batch+queue)",
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
    p.add_argument("--natural-csv", required=True)
    p.add_argument("--single-jsonl", required=True)
    p.add_argument("--multi-jsonl", required=True)
    p.add_argument("--train-split", required=True)
    p.add_argument("--text-emb-cache", required=True)
    p.add_argument("--stats-path", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--warmup", type=int, default=2_000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-schedule", default="cosine", choices=["cosine", "constant"],
                   help="constant = warmup then hold base_lr flat (no decay)")
    p.add_argument("--lr-total", type=int, default=0,
                   help="cosine decay horizon; 0 -> use max_steps. Set > max_steps "
                        "to keep a gentle decay while stopping early.")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--lambda-kl", type=float, default=1e-4)
    p.add_argument("--lambda-recon", type=float, default=1.0)
    p.add_argument("--cross-modal-recon", action="store_true", default=False,
                   help="also reconstruct motion from the text latent (TMR-style "
                        "text<->motion alignment); keeps generation")
    p.add_argument("--lambda-contrastive", type=float, default=1e-1)
    p.add_argument("--info-nce-temp", type=float, default=0.1)
    p.add_argument("--queue-size", type=int, default=0,
                   help="memory-bank size (extra InfoNCE negatives); 0=off")
    p.add_argument("--text-dup-threshold", type=float, default=0.9,
                   help="mask negatives whose projected text-cos > this")
    p.add_argument("--feat-root", default="",
                   help="precomputed-feature mirror tree; speeds up dataloader")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--canonicalize", action="store_true", default=True,
                   help="canonicalize motion features (heading/position invariant)")
    p.add_argument("--no-canonicalize", dest="canonicalize", action="store_false")
    p.add_argument("--sampler-mode", default="round_robin",
                   choices=["round_robin", "proportional"],
                   help="round_robin: 1:1:1 source mix (upsamples natural). "
                        "proportional: each unique pair ~once/epoch "
                        "(mix follows pool sizes; more timeline exposure).")
    # Model architecture (matches nvidia/TMR-SOMA-RP-v1 config.yaml).
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--ff-size", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=5_000)
    # In-training retrieval eval (content/overview only, for speed).
    p.add_argument("--eval-every", type=int, default=5_000,
                   help="run content/overview retrieval eval every N steps (0=off)")
    p.add_argument("--eval-testsuite", default="/nfsdata/home/jungbin.cho/TAP/data/kimodo_benchmark/testsuite")
    p.add_argument("--eval-text-cache", default="/nfsdata/home/jungbin.cho/TAP/data/bones_seed/eval_text_emb.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--resume-from", default="",
                   help="checkpoint to resume encoder/decoder/optimizer + step")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dump full run config for provenance/reproducibility.
    _write_config(args, out_dir)

    torch.manual_seed(args.seed)

    # ---- CPU motion_rep for the dataset workers (features computed per-sample
    # in the worker, so canonicalize broadcasts correctly at batch=1).
    from kimodo.motion_rep.reps.tmr_motionrep import TMRMotionRep
    from kimodo.skeleton import G1Skeleton34
    cpu_motion_rep = TMRMotionRep(
        skeleton=G1Skeleton34(), fps=args.fps, stats_path=args.stats_path,
    )

    # ---- Dataset / loader
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
    )
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
        collate_fn=collate_motion_batch, persistent_workers=args.num_workers > 0,
    )

    # ---- Model
    tmr, motion_rep, decoder = build_tmr_g1(
        text_emb_cache_path=args.text_emb_cache,
        stats_path=args.stats_path,
        fps=args.fps,
        device=args.device,
    )
    tmr.train(); decoder.train()
    params = list(tmr.motion_encoder.parameters()) + list(tmr.text_encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # ---- TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(log_dir=str(out_dir / "tb"))

    # ---- In-training evaluator (content/overview)
    evaluator = None
    if args.eval_every and Path(args.eval_text_cache).is_file():
        from tmr_g1.training.inline_eval import GroupEvaluator
        evaluator = GroupEvaluator(
            testsuite=args.eval_testsuite,
            g1_npz_root=args.data_root,
            text_cache_path=args.eval_text_cache,
            motion_rep=cpu_motion_rep,
            split="content", grp="overview",
            fps=args.fps, canonicalize=args.canonicalize, device=args.device,
        )
        log.info("inline eval ready: content/overview pool=%d (missing=%d)",
                 evaluator.n, evaluator.miss)

    # ---- Resume (encoder/decoder/optimizer/step)
    start_step = 0
    if args.resume_from and Path(args.resume_from).is_file():
        ck = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        tmr.motion_encoder.load_state_dict(ck["motion_encoder"])
        tmr.text_encoder.load_state_dict(ck["text_encoder"])
        decoder.load_state_dict(ck["motion_decoder"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start_step = int(ck.get("step", 0))
        log.info("resumed from %s at step %d", args.resume_from, start_step)

    # ---- Memory bank (FIFO of detached embeddings)
    queue_zm = queue_zt = None

    # ---- Train loop
    step = start_step
    t0 = time.time()
    for epoch in range(10_000):  # virtual, we cap by max_steps
        for batch in loader:
            feats_in = batch["features"].to(args.device, non_blocking=True)
            mask = batch["mask"].to(args.device, non_blocking=True)
            texts = batch["text"]

            z_m, mu_m, logvar_m, feats = _encode_motion(
                tmr, feats_in, mask, training=True)
            z_t, mu_t, logvar_t = _encode_text(tmr, texts, training=True)

            # Decoder reconstructs the motion features. Always from the motion
            # latent; with --cross-modal-recon, ALSO from the text latent so the
            # text embedding must decode to the motion (TMR's alignment
            # mechanism — keeps generation AND makes it serve retrieval).
            T_max = feats.shape[1]
            pred_m = decoder(z_m, T_max)  # (B, T_max, motion_rep_dim)
            l_recon_m = reconstruction_loss(pred_m, feats, mask)
            if args.cross_modal_recon:
                pred_t = decoder(z_t, T_max)
                l_recon_t = reconstruction_loss(pred_t, feats, mask)
                l_recon = 0.5 * (l_recon_m + l_recon_t)
            else:
                l_recon_t = torch.zeros((), device=args.device)
                l_recon = l_recon_m
            l_kl_m = kl_to_standard_normal(mu_m, logvar_m)
            l_kl_t = kl_to_standard_normal(mu_t, logvar_t)

            zm_n = F.normalize(z_m, dim=-1)
            zt_n = F.normalize(z_t, dim=-1)
            l_nce = info_nce_membank(
                zm_n, zt_n, queue_zm, queue_zt,
                temperature=args.info_nce_temp,
                text_dup_threshold=args.text_dup_threshold,
            )

            loss = (args.lambda_recon * l_recon
                    + args.lambda_kl * (l_kl_m + l_kl_t)
                    + args.lambda_contrastive * l_nce)

            # Enqueue detached embeddings for the memory bank.
            if args.queue_size > 0:
                with torch.no_grad():
                    qm, qt = zm_n.detach(), zt_n.detach()
                    if queue_zm is None:
                        queue_zm, queue_zt = qm, qt
                    else:
                        queue_zm = torch.cat([queue_zm, qm], 0)[-args.queue_size:]
                        queue_zt = torch.cat([queue_zt, qt], 0)[-args.queue_size:]

            # LR schedule. lr_total decouples the cosine horizon from max_steps
            # so we can stop early while keeping a gentle (e.g. 200k) decay.
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
                # Quick retrieval-accuracy probe on this batch:
                with torch.no_grad():
                    sim = zm_n @ zt_n.t()
                    r_at_1 = (sim.argmax(1) == torch.arange(sim.shape[0], device=sim.device)).float().mean().item()
                dt = time.time() - t0
                log.info(
                    "step=%d lr=%.2e loss=%.4f recon=%.4f recon_t=%.4f kl_m=%.4f kl_t=%.4f nce=%.4f "
                    "batch_R@1=%.3f  rate=%.1f steps/s",
                    step, lr, loss.item(), l_recon.item(), float(l_recon_t),
                    l_kl_m.item(), l_kl_t.item(), l_nce.item(), r_at_1, step / max(1, dt),
                )
                tb.add_scalar("loss/total", loss.item(), step)
                tb.add_scalar("loss/recon", l_recon.item(), step)
                tb.add_scalar("loss/recon_text", float(l_recon_t), step)
                tb.add_scalar("loss/kl_motion", l_kl_m.item(), step)
                tb.add_scalar("loss/kl_text", l_kl_t.item(), step)
                tb.add_scalar("loss/infonce", l_nce.item(), step)
                tb.add_scalar("train/batch_R@1", r_at_1, step)
                tb.add_scalar("train/lr", lr, step)
                tb.add_scalar("train/steps_per_sec", step / max(1, dt), step)
            if evaluator is not None and step % args.eval_every == 0:
                em = evaluator.evaluate(tmr)
                log.info(
                    "EVAL[content/overview] step=%d R@1=%.2f R@3=%.2f R@5=%.2f "
                    "R@10=%.2f MedR=%.1f (pool=%d)",
                    step, em.get("R01", 0), em.get("R03", 0), em.get("R05", 0),
                    em.get("R10", 0), em.get("MedR", 0), evaluator.n,
                )
                for k in ("R01", "R02", "R03", "R05", "R10", "MedR"):
                    if k in em:
                        tb.add_scalar(f"eval_content_overview/{k}", em[k], step)
            if step % args.ckpt_every == 0:
                ckpt = {
                    "step": step,
                    "motion_encoder": tmr.motion_encoder.state_dict(),
                    "text_encoder": tmr.text_encoder.state_dict(),
                    "motion_decoder": decoder.state_dict(),
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

# VAE-G1 — motion VAE for the G1 embodiment

A from-scratch **motion VAE** trained on Bones-SEED G1 motions. It is the
architectural sibling of [`tmr_g1`](../tmr_g1): the **same** ACTOR-style
transformer motion encoder and the **same** `TMRMotionRep(G1Skeleton34)` 210-d
feature space, but with the **text branch and contrastive loss removed**.
Training objective is reconstruction + KL only.

```
text→motion retrieval (tmr_g1)         motion autoencoding (vae_g1)
┌───────────┐   ┌───────────┐          ┌───────────┐   ┌───────────┐
│  motion   │   │   text    │          │  motion   │   │  motion   │
│  encoder  │   │  encoder  │          │  encoder  │   │  decoder  │
└─────┬─────┘   └─────┬─────┘          └─────┬─────┘   └─────▲─────┘
   z_m│            z_t│                    z │              z │
      └──InfoNCE──────┘                      └──────recon────┘
   + recon + KL                              + KL
```

## Why a separate model

`tmr_g1`'s encoder was tuned for retrieval, which (with `λ_recon=0.1`) packs
motions into a tight, text-aligned latent. That makes it a **weak FID feature
extractor**: a collapsed/clustered latent yields a misleadingly small FID
regardless of motion quality. A motion VAE trained with a proper reconstruction
weight is forced to spread the latent to cover the motion manifold, giving an
embedding that is **discriminative for FID / motion-space metrics**. The two
models share a feature space and encoder class, so they remain swappable.

## What it produces

A trained motion encoder (`mu` of `q(z|motion)`) usable as:
- a **reconstruction-grade autoencoder** (encode → decode → G1 motion features), and
- a **feature extractor** for FID / precision-recall of G1 motion generators
  (e.g. the `kinematic_planner` MDM).

## Layout (mirrors `tmr_g1`)

| file | role |
|---|---|
| `model/vae_model.py` | `build_vae_g1()` → `(MotionVAE, motion_rep)`; `MotionVAE` = ACTOR encoder + `ACTORStyleDecoder` |
| `training/losses.py` | `kl_to_standard_normal`, `reconstruction_loss` (no InfoNCE) |
| `training/train.py` | argparse trainer; loss = `λ_recon·recon + λ_kl·kl` |
| `training/inline_eval.py` | `ReconEvaluator` — recon_mse / kl / mu_std / active_units during training |
| `training/build_stats.py` | per-feature mean/std (identical to `tmr_g1`; reuse `tmr_g1_stats_v3`) |
| `data/g1_dataset.py` | **identical** to `tmr_g1`'s dataset (captions define the windows; text is ignored) |
| `eval_recon.py` | standalone recon + latent-spread eval over all 6 testsuite groups |
| `scripts/train_vae_g1.sbatch` | parameterized Slurm launcher |

## Key differences from `tmr_g1`

- **No text encoder / no LLM2Vec / no `--text-emb-cache`.** The dataloader still
  builds the natural/single/multi pools (they define the motion windows) but the
  captions are dropped in the train loop.
- **Loss = recon + KL only.** No InfoNCE, no memory-bank queue, no dup-mask.
- **`λ_recon` defaults to 1.0** (vs 0.1 in TMR) — we *want* faithful
  reconstruction and a well-used latent here.
- **Eval metric is reconstruction + latent spread**, not retrieval R@k. Watch
  `mu_std` / `active_units`: near-zero ⇒ posterior collapse ⇒ useless as an FID
  extractor.
- Inherits the same augmentation knobs added during the TMR overfitting work:
  `--aug-feat-noise-std` and `--aug-time-jitter-sec`.

## Install

`vae_g1` is registered in the top-level `pyproject.toml`
(`include = ["tmr_g1*", "vae_g1*"]`). Make it importable the same way `tmr_g1`
is (an editable finder, so `import kimodo` is **not** shadowed):

```bash
conda activate kimodo_soma
cd /nfsdata/home/jungbin.cho/TAP && pip install -e . --no-deps
```

## Train

Reuses the existing TMR-G1 stats (`tmr_g1_stats_v3`) and feature cache
(`g1_feat_npz`) — same motion rep, so no rebuild needed.

```bash
# defaults: lr 3e-4 cosine, batch 256, λ_recon 1.0, λ_kl 1e-4, 100k steps,
#           proportional sampler, eval/ckpt every 2k
TAG=v0 sbatch vae_g1/scripts/train_vae_g1.sbatch

# e.g. a higher-KL / regularized run
TAG=v1_kl3 KL=3e-4 DROPOUT=0.2 sbatch vae_g1/scripts/train_vae_g1.sbatch
```

Override knobs (env vars): `LR BS KL RECON SEED DROPOUT GRADCLIP WARMUP NOISE
JITTER MAXSTEPS LRTOTAL SAMPLER`.

## Eval

```bash
cd /tmp && python /nfsdata/home/jungbin.cho/TAP/vae_g1/eval_recon.py \
  --ckpt /nfsdata/home/jungbin.cho/TAP/runs/vae_g1/v0/last.pt \
  --stats-path /nfsdata/home/jungbin.cho/TAP/data/bones_seed/tmr_g1_stats_v3 \
  --g1-npz-root /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1_unified_npz \
  --testsuite /nfsdata/home/jungbin.cho/TAP/data/kimodo_benchmark/testsuite \
  --out /nfsdata/home/jungbin.cho/TAP/runs/vae_g1/v0/recon.json
```

Reports per-group `recon_mse / recon_rmse / kl / mu_std / active_units`.

## Notes

- Run Python from a cwd **without** a `kimodo/` dir (the scripts `cd /tmp`) to
  avoid the namespace-shadowing trap (`docs/notes.md` §6).
- The encoder checkpoint key is `motion_encoder` (same as TMR-G1), so a VAE-G1
  encoder can be loaded by any code expecting a TMR-G1 motion encoder.

#!/bin/bash
#SBATCH --job-name=tmr_g1_v8
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1_v8_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1_v8_%j.err

set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
source /lab-vision/human_motion/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate /nfsdata/home/jungbin.cho/miniconda3.broken/envs/kimodo_soma
cd /tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TEXT_ENCODER_DEVICE=cuda

OUT=${OUT:-/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1/v8}

# v8 = v6 recipe (recon 0.1 breakthrough) + anti-overfit fixes.
# This model is ALSO the frozen feature extractor for FID + R-precision eval,
# so it needs a well-converged, stable latent space -- not an aggressive
# early-stop. v6 peaked at content/overview R@3=85.7 @ step 15k then declined to
# ~67 by 80k because the VAE KL blew up to ~1000 (lambda_kl 1e-4 = no
# constraint) and the latent drifted. lambda_kl 1e-3 is what lets us train long
# WITHOUT that decline.
# Changes vs v6:
#   --lr-schedule constant : user prefers peak LR (gave best metrics). Hold 3e-4
#                            after warmup; pick best/most-stable ckpt via dense
#                            eval. (Caveat: constant LR keeps the space jittery.)
#   --lambda-kl 1e-3       : 10x v6 -> real pressure on the latent (KL ~1000 was
#                            unconstrained). The actual fix for the decline.
#   --max-steps 100000     : converged evaluator for FID, not a 15k early-stop.
#   --eval-every 2500      : dense eval to watch stability under constant LR.
#   --ckpt-every 5000      : 20 checkpoints over 100k to pick the best from.
python -u /nfsdata/home/jungbin.cho/TAP/tmr_g1/training/train.py \
  --data-root    /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1_unified_npz \
  --feat-root    /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1_feat_npz \
  --natural-csv  /nfsdata/home/jungbin.cho/TAP/data/bones_seed/metadata/metadata/seed_metadata_v004.csv \
  --single-jsonl /nfsdata/home/jungbin.cho/TAP/data/bones_seed/metadata/metadata/seed_metadata_v002_temporal_labels.jsonl \
  --multi-jsonl  /nfsdata/home/jungbin.cho/TAP/data/bones_seed/multi_timeline.jsonl \
  --train-split  /nfsdata/home/jungbin.cho/TAP/data/kimodo_benchmark/splits/train_split_paths.txt \
  --text-emb-cache /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1_text_emb.pt \
  --stats-path   /nfsdata/home/jungbin.cho/TAP/data/bones_seed/tmr_g1_stats_v3 \
  --out-dir      "${OUT}" \
  --batch 256 --num-workers 24 \
  --max-steps 100000 --warmup 2000 \
  --lr 3e-4 --lr-schedule constant --weight-decay 1e-2 \
  --lambda-recon 0.1 --lambda-kl 1e-3 --lambda-contrastive 1.0 \
  --info-nce-temp 0.1 --queue-size 8192 --text-dup-threshold 0.9 \
  --canonicalize --sampler-mode proportional \
  --eval-every 2500 --log-every 100 --ckpt-every 5000

echo "Done at $(date)"

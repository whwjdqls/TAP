#!/bin/bash
#SBATCH --job-name=tmr_g1
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1_%j.err

set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

source /lab-vision/human_motion/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate /nfsdata/home/jungbin.cho/miniconda3.broken/envs/kimodo_soma
cd /tmp

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TEXT_ENCODER_DEVICE=cuda

OUT=${OUT:-/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1/v2}

# Hyperparameters (v2):
#   batch 256  — InfoNCE retrieval quality scales with #in-batch negatives;
#                256 >> v1's 64. The model is tiny so memory is a non-issue;
#                throughput handled by 20 dataloader workers (feature compute
#                is per-sample on CPU).
#   lr 3e-4    — scaled ~1.5x from v1's 2e-4 for the 4x larger batch (between
#                sqrt and linear scaling); stable in smoke tests.
#   canonicalize ON (default) — heading/position-invariant features; the key
#                fix over v1. Stats are tmr_g1_stats_v3 (also canonicalized).
python -u /nfsdata/home/jungbin.cho/TAP/tmr_g1/training/train.py \
  --data-root    /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1_unified_npz \
  --natural-csv  /nfsdata/home/jungbin.cho/TAP/data/bones_seed/metadata/metadata/seed_metadata_v004.csv \
  --single-jsonl /nfsdata/home/jungbin.cho/TAP/data/bones_seed/metadata/metadata/seed_metadata_v002_temporal_labels.jsonl \
  --multi-jsonl  /nfsdata/home/jungbin.cho/TAP/data/bones_seed/multi_timeline.jsonl \
  --train-split  /nfsdata/home/jungbin.cho/TAP/data/kimodo_benchmark/splits/train_split_paths.txt \
  --text-emb-cache /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1_text_emb.pt \
  --stats-path   /nfsdata/home/jungbin.cho/TAP/data/bones_seed/tmr_g1_stats_v3 \
  --out-dir      "${OUT}" \
  --batch 256 --num-workers 20 \
  --max-steps 100000 --warmup 2000 \
  --lr 3e-4 --weight-decay 1e-2 \
  --lambda-recon 1.0 --lambda-kl 1e-5 --lambda-contrastive 1.0 \
  --info-nce-temp 0.1 \
  --canonicalize \
  --log-every 100 --ckpt-every 5000

echo "Done at $(date)"

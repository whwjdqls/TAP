#!/bin/bash
#SBATCH --job-name=tmr_g1_v9a
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1_v9a_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1_v9a_%j.err

set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo_soma
cd /tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TEXT_ENCODER_DEVICE=cuda

OUT=${OUT:-/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1/v9a}

# v9a = v6's PROVEN recipe (the run that gave real R@3 85.7), cosine decay.
# v8 collapsed because lambda_kl 1e-3 caused text-encoder posterior collapse;
# v9a keeps lambda_kl at v6's 1e-4 (the value contrastive learning needs).
# Moderate 60k horizon; pick best checkpoint by the standalone eval.
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
  --max-steps 60000 --warmup 2000 \
  --lr 3e-4 --lr-schedule cosine --weight-decay 1e-2 \
  --lambda-recon 0.1 --lambda-kl 1e-4 --lambda-contrastive 1.0 \
  --info-nce-temp 0.1 --queue-size 8192 --text-dup-threshold 0.9 \
  --canonicalize --sampler-mode proportional \
  --eval-every 2500 --log-every 100 --ckpt-every 5000

echo "Done at $(date)"

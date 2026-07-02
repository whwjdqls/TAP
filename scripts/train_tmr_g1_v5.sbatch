#!/bin/bash
#SBATCH --job-name=tmr_g1_v5
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1_v5_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1_v5_%j.err

set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
source /lab-vision/human_motion/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate /nfsdata/home/jungbin.cho/miniconda3.broken/envs/kimodo_soma
cd /tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TEXT_ENCODER_DEVICE=cuda

OUT=${OUT:-/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1/v5}

# v5 = Tier-1 stack on top of the v2/v3 recipe (canon, proportional, stats_v3):
#   --feat-root        precomputed raw features -> cheap per-fetch (throughput).
#   --queue-size 8192  memory-bank: ~8.4k InfoNCE negatives at batch 256,
#                      decoupling #negatives from batch size (v4 needed bs512
#                      for just 512). Should help R@1/R@3 esp. on repetition.
#   --text-dup-threshold 0.9  mask near-duplicate-caption negatives (projected
#                      text-cos), matching the benchmark's dedup; cleans signal.
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
  --lr 3e-4 --weight-decay 1e-2 \
  --lambda-recon 1.0 --lambda-kl 1e-5 --lambda-contrastive 1.0 \
  --info-nce-temp 0.1 --queue-size 8192 --text-dup-threshold 0.9 \
  --canonicalize --sampler-mode proportional \
  --eval-every 5000 --log-every 100 --ckpt-every 5000

echo "Done at $(date)"

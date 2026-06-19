#!/bin/bash
#SBATCH --job-name=tmr_g1_v4
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1_v4_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1_v4_%j.err

set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo_soma
cd /tmp

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TEXT_ENCODER_DEVICE=cuda

OUT=${OUT:-/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1/v4}

# v4 vs v3: same config (proportional sampler, canonicalize, lr-scaled),
#   batch 512 (2x v3's 256 -> more InfoNCE negatives), max_steps 200000.
#   lr 4e-4 (~sqrt-scaled from v3's 3e-4 @ bs256). 40 dataloader workers since
#   per-sample feature computation is the throughput bottleneck.
#   Inline content/overview retrieval eval every 5000 steps -> TensorBoard
#   (eval_content_overview/R@k) + console.
python -u /nfsdata/home/jungbin.cho/TAP/tmr_g1/training/train.py \
  --data-root    /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1_unified_npz \
  --natural-csv  /nfsdata/home/jungbin.cho/TAP/data/bones_seed/metadata/metadata/seed_metadata_v004.csv \
  --single-jsonl /nfsdata/home/jungbin.cho/TAP/data/bones_seed/metadata/metadata/seed_metadata_v002_temporal_labels.jsonl \
  --multi-jsonl  /nfsdata/home/jungbin.cho/TAP/data/bones_seed/multi_timeline.jsonl \
  --train-split  /nfsdata/home/jungbin.cho/TAP/data/kimodo_benchmark/splits/train_split_paths.txt \
  --text-emb-cache /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1_text_emb.pt \
  --stats-path   /nfsdata/home/jungbin.cho/TAP/data/bones_seed/tmr_g1_stats_v3 \
  --out-dir      "${OUT}" \
  --batch 512 --num-workers 40 \
  --max-steps 200000 --warmup 2000 \
  --lr 4e-4 --weight-decay 1e-2 \
  --lambda-recon 1.0 --lambda-kl 1e-5 --lambda-contrastive 1.0 \
  --info-nce-temp 0.1 \
  --canonicalize --sampler-mode proportional \
  --eval-every 5000 --log-every 100 --ckpt-every 5000

echo "Done at $(date)"

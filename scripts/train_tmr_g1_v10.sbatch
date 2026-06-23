#!/bin/bash
#SBATCH --job-name=tmr_g1_v10
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1_v10_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1_v10_%j.err

set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)  TAG=${TAG} QUEUE=${QUEUE} LR=${LR}"
source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo_soma
cd /tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TEXT_ENCODER_DEVICE=cuda

# Parameterized ablation on v6's proven base (recon 0.1, kl 1e-4, dup 0.9,
# bs256, canon, proportional). Tests: drop the memory-bank buffer (QUEUE=0)
# and/or raise peak LR. Cosine horizon held at 200k (like v6, which peaked in
# the 10-20k high-LR window) via --lr-total, but stop at 40k. Dense eval/ckpt
# every 2000 to catch the peak.
#   TAG=v10a QUEUE=0    LR=3e-4   (no buffer)
#   TAG=v10b QUEUE=8192 LR=5e-4   (higher peak LR)
#   TAG=v10c QUEUE=0    LR=5e-4   (both)
: "${TAG:?set TAG}" "${QUEUE:?set QUEUE}" "${LR:?set LR}"
OUT=/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1/${TAG}

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
  --max-steps 40000 --warmup 2000 \
  --lr "${LR}" --lr-schedule cosine --lr-total 200000 --weight-decay 1e-2 \
  --lambda-recon 0.1 --lambda-kl 1e-4 --lambda-contrastive 1.0 \
  --info-nce-temp 0.1 --queue-size "${QUEUE}" --text-dup-threshold 0.9 \
  --canonicalize --sampler-mode proportional \
  --eval-every 2000 --log-every 100 --ckpt-every 2000

echo "Done at $(date)"

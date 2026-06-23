#!/bin/bash
#SBATCH --job-name=tmr_g1_v9b
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1_v9b_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1_v9b_%j.err

set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo_soma
cd /tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TEXT_ENCODER_DEVICE=cuda

OUT=${OUT:-/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1/v9b}

# v9b = constant-LR variant (user preference), done SAFELY:
#   lambda_kl 1e-4 (NOT 1e-3 -> avoids the v8 text-encoder collapse)
#   constant lr 1e-4 (NOT 3e-4 -> a held-flat 3e-4 never settles and risks
#   the instability we saw; lower flat lr is the safe way to keep peak-ish LR).
# Same 60k horizon as v9a; compare the two and pick the best by standalone eval.
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
  --lr 1e-4 --lr-schedule constant --weight-decay 1e-2 \
  --lambda-recon 0.1 --lambda-kl 1e-4 --lambda-contrastive 1.0 \
  --info-nce-temp 0.1 --queue-size 8192 --text-dup-threshold 0.9 \
  --canonicalize --sampler-mode proportional \
  --eval-every 2500 --log-every 100 --ckpt-every 5000

echo "Done at $(date)"

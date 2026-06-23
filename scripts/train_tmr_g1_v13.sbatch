#!/bin/bash
#SBATCH --job-name=tmr_g1_v13
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1_v13_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1_v13_%j.err

set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)  TAG=${TAG}"
source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo_soma
cd /tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TEXT_ENCODER_DEVICE=cuda

# Anti-overfit / FID-coverage round. Defaults = v6 baseline; override via --export.
# New lever: NOISE = train-time Gaussian feature-noise std (alignment-safe aug).
: "${TAG:?set TAG}"
LR=${LR:-3e-4}; QUEUE=${QUEUE:-8192}; BS=${BS:-256}; KL=${KL:-1e-4}
TEMP=${TEMP:-0.1}; DUP=${DUP:-0.9}; RECON=${RECON:-0.1}
SEED=${SEED:-0}; DROPOUT=${DROPOUT:-0.1}; GRADCLIP=${GRADCLIP:-1.0}; WARMUP=${WARMUP:-2000}
NOISE=${NOISE:-0.0}; JITTER=${JITTER:-0.0}
OUT=/nfsdata/home/jungbin.cho/TAP/runs/tmr_g1/${TAG}
echo "  LR=$LR QUEUE=$QUEUE BS=$BS KL=$KL TEMP=$TEMP DUP=$DUP RECON=$RECON SEED=$SEED DROPOUT=$DROPOUT GRADCLIP=$GRADCLIP WARMUP=$WARMUP NOISE=$NOISE JITTER=$JITTER"

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
  --batch "${BS}" --num-workers 16 \
  --max-steps 40000 --warmup "${WARMUP}" \
  --lr "${LR}" --lr-schedule cosine --lr-total 200000 --weight-decay 1e-2 \
  --lambda-recon "${RECON}" --lambda-kl "${KL}" --lambda-contrastive 1.0 \
  --info-nce-temp "${TEMP}" --queue-size "${QUEUE}" --text-dup-threshold "${DUP}" \
  --dropout "${DROPOUT}" --seed "${SEED}" --grad-clip "${GRADCLIP}" \
  --aug-feat-noise-std "${NOISE}" --aug-time-jitter-sec "${JITTER}" \
  --canonicalize --sampler-mode proportional \
  --eval-every 2000 --log-every 100 --ckpt-every 2000

echo "Done at $(date)"

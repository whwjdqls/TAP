#!/bin/bash
#SBATCH --job-name=vae_g1
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/vae_g1_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/vae_g1_%j.err

set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)  TAG=${TAG}"
source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo_soma
# cd out of the repo root so `import kimodo` resolves to the installed package
# (the repo's kimodo/ dir shadows it; see docs/notes.md S6). vae_g1 is found via
# its editable finder, not the cwd, so this is safe.
cd /tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Motion VAE (recon + KL). Defaults below; override via --export.
# Default WINMODE=random: arbitrary random crops over ALL motions (no captions).
: "${TAG:?set TAG}"
LR=${LR:-3e-4}; BS=${BS:-256}; KL=${KL:-1e-4}; RECON=${RECON:-1.0}
SEED=${SEED:-0}; DROPOUT=${DROPOUT:-0.1}; GRADCLIP=${GRADCLIP:-1.0}; WARMUP=${WARMUP:-2000}
NOISE=${NOISE:-0.0}; JITTER=${JITTER:-0.0}; MAXSTEPS=${MAXSTEPS:-100000}; LRTOTAL=${LRTOTAL:-100000}
WINMODE=${WINMODE:-random}; MINCLIP=${MINCLIP:-2.0}; MAXCLIP=${MAXCLIP:-10.0}; OVERSAMPLE=${OVERSAMPLE:-1}
OUT=/nfsdata/home/jungbin.cho/TAP/runs/vae_g1/${TAG}
echo "  LR=$LR BS=$BS KL=$KL RECON=$RECON SEED=$SEED DROPOUT=$DROPOUT GRADCLIP=$GRADCLIP WARMUP=$WARMUP NOISE=$NOISE JITTER=$JITTER MAXSTEPS=$MAXSTEPS WINMODE=$WINMODE CLIP=[$MINCLIP,$MAXCLIP] OVERSAMPLE=$OVERSAMPLE"

python -u /nfsdata/home/jungbin.cho/TAP/vae_g1/training/train.py \
  --data-root    /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1_unified_npz \
  --feat-root    /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1_feat_npz \
  --train-split  /nfsdata/home/jungbin.cho/TAP/data/kimodo_benchmark/splits/train_split_paths.txt \
  --stats-path   /nfsdata/home/jungbin.cho/TAP/data/bones_seed/tmr_g1_stats_v3 \
  --out-dir      "${OUT}" \
  --window-mode "${WINMODE}" --min-clip-sec "${MINCLIP}" --max-clip-sec "${MAXCLIP}" --oversample "${OVERSAMPLE}" \
  --batch "${BS}" --num-workers 16 \
  --max-steps "${MAXSTEPS}" --warmup "${WARMUP}" \
  --lr "${LR}" --lr-schedule cosine --lr-total "${LRTOTAL}" --weight-decay 1e-2 \
  --lambda-recon "${RECON}" --lambda-kl "${KL}" \
  --dropout "${DROPOUT}" --seed "${SEED}" --grad-clip "${GRADCLIP}" \
  --aug-feat-noise-std "${NOISE}" --aug-time-jitter-sec "${JITTER}" \
  --canonicalize \
  --eval-every 2000 --log-every 100 --ckpt-every 2000

echo "Done at $(date)"

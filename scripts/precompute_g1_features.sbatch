#!/bin/bash
#SBATCH --job-name=g1_feat
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/g1_feat_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/g1_feat_%j.err
set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo_soma
cd /tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u /nfsdata/home/jungbin.cho/TAP/scripts/precompute_g1_features.py \
  --npz-root /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1_unified_npz \
  --out-root /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1_feat_npz \
  --workers 32 --skip-existing
echo "count: $(find /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1_feat_npz -name '*.npz' | wc -l)"
du -sh /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1_feat_npz
echo "Done at $(date)"

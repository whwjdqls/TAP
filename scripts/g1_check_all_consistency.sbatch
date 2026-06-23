#!/bin/bash
#SBATCH --job-name=g1_consist
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/g1_consist_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/g1_consist_%j.err

set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo_soma
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -u /nfsdata/home/jungbin.cho/TAP/scripts/g1_check_all_consistency.py \
  --workers 64 \
  --out /nfsdata/home/jungbin.cho/TAP/runs/g1_consistency/report_${SLURM_JOB_ID:-local}.json

echo "Done at $(date)"

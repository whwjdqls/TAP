#!/bin/bash
#SBATCH --job-name=g1_yup
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/g1_yup_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/g1_yup_%j.err

set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo_soma
cd /tmp

python /nfsdata/home/jungbin.cho/TAP/scripts/g1_npz_mj_to_kimodo.py \
  --root /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1_20fps_npz \
  --workers 8

echo "Done at $(date)"

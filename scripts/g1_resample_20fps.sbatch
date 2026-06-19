#!/bin/bash
#SBATCH --job-name=g1_20fps
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/g1_20fps_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/g1_20fps_%j.err

set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo_soma
cd /nfsdata/home/jungbin.cho/TAP

python scripts/g1_csv_to_20fps_npz.py \
  --csv-root data/bones_seed/g1/csv \
  --out-root data/bones_seed/g1_20fps_npz \
  --workers 8 \
  --skip-existing

echo "Done at $(date)"
echo "NPZ count: $(find data/bones_seed/g1_20fps_npz -name '*.npz' | wc -l)"
du -sh data/bones_seed/g1_20fps_npz

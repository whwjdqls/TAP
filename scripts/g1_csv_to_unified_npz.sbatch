#!/bin/bash
#SBATCH --job-name=g1_unified
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/g1_unified_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/g1_unified_%j.err

set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo_soma
cd /tmp

# Pin each worker process to a single BLAS/OMP thread. Without this, 32
# worker processes each spawn ~20 numpy/MuJoCo threads -> load avg ~600 on a
# 32-core allocation, catastrophic contention. Single-threaded workers
# parallelize cleanly across the 32 cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python -u /nfsdata/home/jungbin.cho/TAP/scripts/g1_csv_to_unified_npz.py \
  --csv-root /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1/csv \
  --out-root /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1_unified_npz \
  --workers 32 \
  --skip-existing

echo "Done at $(date)"
echo "NPZ count: $(find /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1_unified_npz -name '*.npz' | wc -l)"
du -sh /nfsdata/home/jungbin.cho/TAP/data/bones_seed/g1_unified_npz

#!/bin/bash
#SBATCH --job-name=kimodo_gen
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/kimodo_gen_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/kimodo_gen_%j.err

# Generate Kimodo-G1-SEED-v1 motions for each row in $SUBSET.
# Required env vars:
#   SUBSET     : path to subset.jsonl
#   OUT_NPZS, OUT_CSVS, OUT_PKLS : output dirs
# Optional:
#   DIFFUSION_STEPS  (default 50)
#   MODEL            (default Kimodo-G1-SEED-v1)

set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
: "${SUBSET:?set SUBSET=/path/to/subset.jsonl}"
: "${OUT_NPZS:?set OUT_NPZS=/path}"
: "${OUT_CSVS:?set OUT_CSVS=/path}"
: "${OUT_PKLS:?set OUT_PKLS=/path}"

source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo_soma
cd /nfsdata/home/jungbin.cho/TAP

python scripts/run_kimodo_for_subset.py \
  --subset "${SUBSET}" \
  --out-npzs "${OUT_NPZS}" \
  --out-csvs "${OUT_CSVS}" \
  --out-pkls "${OUT_PKLS}" \
  --model "${MODEL:-Kimodo-G1-SEED-v1}" \
  --diffusion-steps "${DIFFUSION_STEPS:-50}" \
  --text-encoder-device cpu \
  --convert --workers 4

echo "Job finished at $(date)"

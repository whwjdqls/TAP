#!/bin/bash
#SBATCH --job-name=kimodo_gen_batched
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/kimodo_gen_batched_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/kimodo_gen_batched_%j.err

set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
: "${SUBSET:?set SUBSET=}"
: "${OUT_CSVS:?set OUT_CSVS=}"
: "${OUT_PKLS:?set OUT_PKLS=}"
: "${EMB_CACHE:?set EMB_CACHE=}"

source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo_soma
cd /nfsdata/home/jungbin.cho/TAP

python scripts/run_kimodo_batched.py \
  --subset "${SUBSET}" \
  --out-csvs "${OUT_CSVS}" \
  --emb-cache "${EMB_CACHE}" \
  --model "${MODEL:-Kimodo-G1-SEED-v1}" \
  --diffusion-steps "${DIFFUSION_STEPS:-50}" \
  --batch "${BATCH:-16}" \
  --skip-existing

conda activate env_isaaclab
python GR00T-WholeBodyControl/gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
  --input "${OUT_CSVS}" --output "${OUT_PKLS}" \
  --individual --fps 30 --fps_source 30 --num_workers 4

echo "Done at $(date). PKL count: $(find "${OUT_PKLS}" -name '*.pkl' | wc -l)"

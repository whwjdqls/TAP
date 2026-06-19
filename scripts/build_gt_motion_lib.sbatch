#!/bin/bash
#SBATCH --job-name=gt_build
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/gt_build_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/gt_build_%j.err

set -eo pipefail
: "${SUBSET:?}"; : "${OUT_CSVS:?}"; : "${OUT_PKLS:?}"
source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /nfsdata/home/jungbin.cho/TAP

python scripts/build_gt_motion_lib.py \
  --subset "${SUBSET}" \
  --g1-root data/bones_seed/g1 \
  --out-csvs "${OUT_CSVS}" \
  --out-pkls "${OUT_PKLS}" \
  --fps 30 --fps-source 120 --convert --workers 8

#!/bin/bash
#SBATCH --job-name=render_cmp
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/render_cmp_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/render_cmp_%j.err
set -eo pipefail
: "${CASES:?}"
source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /nfsdata/home/jungbin.cho/TAP
python scripts/render_compare.py \
  --subset    runs/eval_subset/t2m_overview_full.jsonl \
  --gt-csvs   runs/eval_gt/t2m_overview_full_csvs \
  --kim-csvs  runs/eval_kimodo/t2m_overview_full_csvs \
  --gt-pkls   runs/eval_gt/t2m_overview_full_pkls \
  --kim-pkls  runs/eval_kimodo/t2m_overview_full_pkls \
  --out-dir   runs/eval_compare/videos \
  --cases     "${CASES}"

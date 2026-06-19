#!/bin/bash
#SBATCH --job-name=g1_text_emb
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/g1_text_emb_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/g1_text_emb_%j.err

set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate kimodo_soma
cd /nfsdata/home/jungbin.cho/TAP

python -u scripts/build_g1_text_emb_cache.py \
  --natural-csv  data/bones_seed/metadata/metadata/seed_metadata_v004.csv \
  --single-jsonl data/bones_seed/metadata/metadata/seed_metadata_v002_temporal_labels.jsonl \
  --multi-jsonl  data/bones_seed/multi_timeline.jsonl \
  --splits-dir   data/kimodo_benchmark/splits \
  --out          data/bones_seed/g1_text_emb.pt \
  --device cuda --dtype bfloat16 --resume \
  --log-every 50 --save-every 2000

echo "Done at $(date)"
ls -lh data/bones_seed/g1_text_emb.pt

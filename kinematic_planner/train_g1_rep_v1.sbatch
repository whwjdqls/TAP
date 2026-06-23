#!/bin/bash
#SBATCH --job-name=g1mdm
#SBATCH --partition=a2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jungbin_cho/TAP/runs/mdm_g1_rep_v1_full/slurm_%j.out
#SBATCH --error=/home/jungbin_cho/TAP/runs/mdm_g1_rep_v1_full/slurm_%j.err

# Train the MDM text->motion planner on g1_rep_v1 (Bones-SEED G1 GT), LLM2Vec
# cached text. Single GPU. Resumable via trainer.resume_from=.../latest.pt.
set -eo pipefail
echo "[node] $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kimodo
cd /tmp   # kimodo namespace-shadowing guard

python /home/jungbin_cho/TAP/kinematic_planner/train_g1.py \
  --config /home/jungbin_cho/TAP/kinematic_planner/configs/train_g1_rep_v1.yaml \
  output_dir=/home/jungbin_cho/TAP/runs/mdm_g1_rep_v1_full \
  "$@"

echo "[done] $(date)"

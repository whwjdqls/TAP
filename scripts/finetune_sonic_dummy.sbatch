#!/bin/bash
#SBATCH --job-name=sonic_ft_dummy
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/sonic_ft_dummy_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/sonic_ft_dummy_%j.err

# End-to-end smoke job for SONIC on PILAB:
#   step A: 5-iter eval (inference on 2 sample motions) → metrics + log
#   step B: 5-iter fine-tune of released checkpoint on the same sample motions
# Real fine-tuning needs Bones-SEED (~130K motions) + multi-GPU. See
# GR00T-WholeBodyControl/docs/source/user_guide/training.md for the full
# config.

set -eo pipefail
# (no -u: conda activate hooks reference unbound vars like LD_LIBRARY_PATH)

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname) at $(date)"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "CPUs per task : ${SLURM_CPUS_PER_TASK:-local}"
nvidia-smi || true
echo

TAP_ROOT="/nfsdata/home/jungbin.cho/TAP"
REPO="${TAP_ROOT}/GR00T-WholeBodyControl"
CKPT="${REPO}/sonic_release/last.pt"
OUT_BASE="${TAP_ROOT}/runs"
JOB_TAG="job_${SLURM_JOB_ID:-local}"
EVAL_DIR="${OUT_BASE}/sonic_eval_${JOB_TAG}"
mkdir -p "${EVAL_DIR}"

source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

cd "${REPO}"

# Sanity: confirm checkpoint + sample data are in place.
test -f "${CKPT}" || { echo "Missing checkpoint: ${CKPT}" >&2; exit 1; }
test -d "${REPO}/sample_data/robot_filtered" || { echo "Missing sample_data/robot_filtered" >&2; exit 1; }
test -d "${REPO}/sample_data/smpl_filtered"  || { echo "Missing sample_data/smpl_filtered"  >&2; exit 1; }

# Keep W&B offline for a dummy/smoke job — comment out to log online.
export WANDB_MODE=offline

echo "============================================================"
echo "[A] Inference / eval on sample motions"
echo "============================================================"
python gear_sonic/eval_agent_trl.py \
  "+checkpoint=${CKPT}" \
  "+headless=True" \
  "++eval_callbacks=im_eval" \
  "++run_eval_loop=False" \
  "++num_envs=8" \
  "+manager_env/terminations=tracking/eval" \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=${REPO}/sample_data/robot_filtered" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=${REPO}/sample_data/smpl_filtered" \
  "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=2" \
  "++hydra.run.dir=${EVAL_DIR}"

echo
echo "============================================================"
echo "[B] Dummy fine-tune (5 PPO iterations, 16 envs)"
echo "============================================================"
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  +checkpoint="${CKPT}" \
  num_envs=16 \
  headless=True \
  ++algo.config.num_learning_iterations=5 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file="${REPO}/sample_data/robot_filtered" \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="${REPO}/sample_data/smpl_filtered" \
  base_dir="${OUT_BASE}" \
  project_name=sonic_ft_dummy \
  experiment_name="${JOB_TAG}"

echo
echo "Eval dir : ${EVAL_DIR}"
echo "Train dir: ${OUT_BASE}/sonic_ft_dummy/${JOB_TAG}-*"
echo "Job finished at $(date)"

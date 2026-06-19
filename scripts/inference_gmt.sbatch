#!/bin/bash
#SBATCH --job-name=gmt_smoke
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/gmt_smoke_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/gmt_smoke_%j.err

# GMT (humanoid-general-motion-tracking) inference smoke test.
# Runs the pretrained Mujoco policy on each example motion for 500 sim steps
# and prints action stats / final qpos for each. CPU-only: GMT's pinned torch
# 2.4.1+cu121 does not support the cluster's Blackwell sm_120 GPUs, and the
# Mujoco sim is CPU-bound anyway. Upstream releases NO training code, so this
# repo only supports inference / sim2sim eval.

set -eo pipefail

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname) at $(date)"
echo "CPUs per task : ${SLURM_CPUS_PER_TASK:-local}"
echo

TAP_ROOT="/nfsdata/home/jungbin.cho/TAP"

source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate gmt

cd "${TAP_ROOT}"

MOTIONS=(
  walk_stand.pkl
  basic_walk.pkl
  crouchwalk_stand.pkl
  squat.pkl
  dance.pkl
  dance_waltz.pkl
  kick_walk.pkl
  airkick_stand.pkl
)

for m in "${MOTIONS[@]}"; do
  echo "============================================================"
  echo "Motion: ${m}"
  echo "============================================================"
  python scripts/gmt_smoke.py --motion "${m}" --steps 500
  echo
done

echo "Job finished at $(date)"

#!/bin/bash
#SBATCH --job-name=sonic_eval
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/sonic_eval_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/sonic_eval_%j.err

# Run SONIC eval (im_eval callback) on a motion_lib dir.
# Required env vars:
#   MOTION_LIB    : path to a directory of <session>/<motion>.pkl
#   EVAL_TAG      : short name used to create the eval output dir
# Optional:
#   NUM_ENVS      : default 32. Limited to # of motions in the set.
#   MAX_MOTIONS   : default unset; caps how many motions get loaded.
#   SMPL_MOTION   : default sample_data/smpl_filtered (only needed because the
#                   release config wires an SMPL encoder; values irrelevant for
#                   G1-only eval but the path must exist).
# Outputs:
#   runs/sonic_eval/<EVAL_TAG>-<jobid>/{metrics_eval.json, .hydra/, eval.log}

set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
: "${MOTION_LIB:?set MOTION_LIB=/path/to/motion_lib_dir}"
: "${EVAL_TAG:?set EVAL_TAG=short_name}"

TAP_ROOT=/nfsdata/home/jungbin.cho/TAP
REPO=${TAP_ROOT}/GR00T-WholeBodyControl
CKPT=${REPO}/sonic_release/last.pt
SMPL=${SMPL_MOTION:-${REPO}/sample_data/smpl_filtered}
NUM_ENVS=${NUM_ENVS:-32}
OUT=${TAP_ROOT}/runs/sonic_eval/${EVAL_TAG}-${SLURM_JOB_ID:-local}
mkdir -p "${OUT}"

source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd "${REPO}"

export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1

EXTRA=()
if [[ -n "${MAX_MOTIONS:-}" ]]; then
  EXTRA+=("++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=${MAX_MOTIONS}")
fi

python gear_sonic/eval_agent_trl.py \
  "+checkpoint=${CKPT}" \
  "+headless=True" \
  "++eval_callbacks=im_eval" \
  "++run_eval_loop=False" \
  "+eval_output_dir=${OUT}" \
  "++num_envs=${NUM_ENVS}" \
  "+manager_env/terminations=tracking/eval" \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=${MOTION_LIB}" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=${SMPL}" \
  "${EXTRA[@]}" \
  "++hydra.run.dir=${OUT}"

echo "metrics → ${OUT}/metrics_eval.json"
echo "Job finished at $(date)"

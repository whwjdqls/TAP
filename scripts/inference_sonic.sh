#!/bin/bash
# Run SONIC policy inference / evaluation on sample motions.
#
# Outputs metrics to TAP/runs/inference_<timestamp>/.
# Modes:
#   metrics  - success rate, MPJPE
#   render   - save .mp4 of policy rollouts
#
# Usage:
#   bash scripts/inference_sonic.sh metrics
#   bash scripts/inference_sonic.sh render
set -eo pipefail
# (no -u: conda activate hooks reference unbound vars like LD_LIBRARY_PATH)

MODE="${1:-metrics}"
TAP_ROOT="/nfsdata/home/jungbin.cho/TAP"
REPO="${TAP_ROOT}/GR00T-WholeBodyControl"
CKPT="${REPO}/sonic_release/last.pt"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${TAP_ROOT}/runs/inference_${MODE}_${TS}"
mkdir -p "${OUT}"

source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

cd "${REPO}"

COMMON_OVERRIDES=(
  "+checkpoint=${CKPT}"
  "+headless=True"
  "++eval_callbacks=im_eval"
  "++run_eval_loop=False"
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=${REPO}/sample_data/robot_filtered"
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=${REPO}/sample_data/smpl_filtered"
  "++hydra.run.dir=${OUT}"
)

if [[ "${MODE}" == "metrics" ]]; then
  python gear_sonic/eval_agent_trl.py \
    "${COMMON_OVERRIDES[@]}" \
    "++num_envs=8" \
    "+manager_env/terminations=tracking/eval" \
    "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=2"
elif [[ "${MODE}" == "render" ]]; then
  python gear_sonic/eval_agent_trl.py \
    "${COMMON_OVERRIDES[@]}" \
    "++num_envs=2" \
    "++manager_env.config.render_results=True" \
    "++manager_env.config.save_rendering_dir=${OUT}/renders" \
    "++manager_env.config.env_spacing=10.0" \
    "~manager_env/recorders=empty" \
    "+manager_env/recorders=render"
else
  echo "Unknown mode: ${MODE}. Use metrics|render." >&2
  exit 1
fi

echo
echo "Output dir: ${OUT}"

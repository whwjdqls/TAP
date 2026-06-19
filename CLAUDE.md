# CLAUDE.md

## WARNING
NEVER run heavy memory or compute jobs in login node!!!
if you need heavy compute just srun CPU or sbatch CPU.
If you have to debug with a GPU srun GPU and scancel it when you are done. If you are doing a sequence of debugging with GPU srun, then only keep for maximum 5 hours. scancel if you are done.
Only then run SBATCH jobs.

## Technical notes
**See `docs/notes.md`** for important facts that aren't obvious from code: kimodo-G1's 417-dim feature space; TMR's positions-only 186-dim feature; data sources shared with `kimodo_open`; LLM2Vec freezing & cache layout; G1 MJCF being 30 bodies (not 34); namespace-package shadowing under editable install; benchmark crop-index FPS mismatch; SONIC's 50 Hz tick; SONIC eval saving rules.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo layout

- `README.MD` — research proposal for **Tracker-Aware Kinematic Motion Planning for Humanoid Whole-Body Control** (ICRA 2027 target).
- `GR00T-WholeBodyControl/` — vendored checkout of `NVlabs/GR00T-WholeBodyControl` (Git LFS pulled). Houses the **SONIC** training/eval/deploy stacks. Do **not** edit files inside this directory; treat it as an upstream dependency.
- `GMT/` — vendored checkout of `zixuan417/humanoid-general-motion-tracking`. **Inference-only**: ships a MuJoCo `sim2sim.py` + pretrained `assets/pretrained_checkpoints/pretrained.pt` + 8 example motions in `assets/motions/`. Upstream has **not yet released training / retargeter code** ("coming soon" in README). Until then, fine-tuning GMT requires writing the training loop from scratch around `utils/motion_lib.py` and the policy — not currently in scope.
- `scripts/` — user-owned launch wrappers. New harness code (inference, fine-tune, sweep, sbatch) belongs here, not inside the vendored repos.
- `runs/` — Slurm logs and Hydra-managed train/eval output dirs.

## Environments

Conda envs live at `/nfsdata/home/jungbin.cho/miniconda3/envs/`.

### `gmt` — for GMT inference

Python 3.8, torch 2.4.1+cu121, MuJoCo 3.2.3, numpy 1.23.0, `mujoco-python-viewer 0.1.4`. Activate with `conda activate gmt`. **CPU-only on this cluster**: GMT's pinned torch 2.4.1 lacks sm_120 kernels, so it errors on the Blackwell RTX Pro 6000 GPUs. MuJoCo runs on CPU anyway and the JIT policy is tiny — CPU is fine.

The upstream `mujoco_viewer 0.1.4` requires a GLFW display even in `'offscreen'` mode. Headless nodes have none and there's no `xvfb` on PATH. `scripts/gmt_smoke.py` works around this by stubbing `mujoco_viewer.MujocoViewer` with a no-op; use it (not `GMT/sim2sim.py` directly) for headless smoke / inference.

### `env_isaaclab` — for SONIC training/inference

**For all SONIC work, activate `env_isaaclab`:**

```bash
source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
```

`env_isaaclab` has Python 3.11.15, PyTorch 2.7.0+cu128, Isaac Lab 2.3.0, Isaac Sim 5.1.0.0, and the SONIC training extras (`trl==0.28.0`, `accelerate`, `smpl_sim`, `hydra-core`, `wandb`, …). `gear_sonic` is installed in editable mode pointing at `GR00T-WholeBodyControl/gear_sonic`. `git-lfs` is in the `base` env (not on PATH globally) — re-pull LFS from `base`.

**Missing-from-`pyproject` runtime deps**, manually pip-installed: `open3d`, `vector_quantize_pytorch`. SONIC's universal-token model and motion-lib import these at startup but the upstream `gear_sonic[training]` extra forgets to declare them. If a fresh env is built, add these.

> SONIC's `sonic_release` config officially targets Isaac Lab **2.3.2**; we're on **2.3.0**. Smoke-tested fine, but inspect for API drift if training silently misbehaves.

CUDA only shows on a GPU node, not on the login node. All SONIC runs need a GPU — submit via Slurm.

## Slurm

Use `--partition=pilab --qos=qos-pi-top` for the user's research jobs. Each pilab node has 8× RTX Pro 6000 Blackwell, 128 CPUs, ~2 TB RAM (defaults: 16 CPUs / 240 GB per GPU). See `scripts/finetune_sonic_dummy.sbatch` for the template.

## SONIC: where the entry points live

All paths below are inside `GR00T-WholeBodyControl/`:

- **Train / fine-tune** — `gear_sonic/train_agent_trl.py` (Hydra). The "release" recipe is `+exp=manager/universal_token/all_modes/sonic_release` (3 encoders: G1, teleop, SMPL). Fine-tune by adding `+checkpoint=sonic_release/last.pt`.
- **Eval / inference** — `gear_sonic/eval_agent_trl.py`. Two modes: metrics (success rate, MPJPE) and render (mp4 rollouts). Pass `++eval_callbacks=im_eval ++run_eval_loop=False` plus motion-path overrides; the released checkpoint's embedded `config.yaml` points at internal training paths and must be overridden.
- **Continuous eval daemon** — `gear_sonic/eval_exp.py` (watches an experiment dir for new checkpoints).
- **Multi-GPU** — wrap `train_agent_trl.py` in `accelerate launch --num_processes=N`. Real fine-tuning is meant for 64+ GPUs.
- **Data processing** — `gear_sonic/data_process/{convert_soma_csv_to_motion_lib,filter_and_copy_bones_data,extract_soma_joints_from_bvh,split_pkl_files}.py`. These don't need Isaac Lab.
- **Checkpoint + data download** — top-level `download_from_hf.py` (`--sample`, `--training`, `--training --no-smpl` to skip the 30 GB SMPL tarball).
- **Pre-flight check** — top-level `check_environment.py --training`.

The full guide is `GR00T-WholeBodyControl/docs/source/user_guide/training.md`.

## Data already on disk

- `GR00T-WholeBodyControl/sonic_release/last.pt` — released SONIC PyTorch checkpoint (~450 MB).
- `GR00T-WholeBodyControl/sonic_release/config.yaml` — paired Hydra config (paths inside it point at internal NV dirs; override `motion_lib_cfg.motion_file` and `.smpl_motion_file` when loading).
- `GR00T-WholeBodyControl/sample_data/{robot_filtered,smpl_filtered,soma_filtered}/` — 2 walking motions; usable as "dummy data" for smoke runs and small fine-tunes.

The full Bones-SEED motion library (~130K motions) and 30 GB SMPL tarball are **not** downloaded — pull with `python download_from_hf.py --training` (no `--no-smpl`) only when you need them.

## Common commands

From `/nfsdata/home/jungbin.cho/TAP`:

```bash
# --- SONIC ---
# Pre-flight check (after activating env_isaaclab)
cd GR00T-WholeBodyControl && python check_environment.py --training

# Local inference on sample motions (needs a GPU node)
bash scripts/inference_sonic.sh metrics    # success rate + MPJPE
bash scripts/inference_sonic.sh render     # .mp4 rollouts

# Dummy fine-tune on PILAB (inference + 5-iter PPO on 2 sample motions)
sbatch scripts/finetune_sonic_dummy.sbatch

# --- GMT (inference only — no upstream training code yet) ---
conda activate gmt
python scripts/gmt_smoke.py --motion walk_stand.pkl --steps 500   # headless
sbatch scripts/inference_gmt.sbatch                               # all 8 motions on pilab
```

## When adding code

- The planner, tracker, and trackability critic from the research proposal should be independently swappable — the experimental matrix in `README.MD` §6.1 depends on freezing one while training another.
- Add new launch / harness / fine-tune scripts under `scripts/`, not inside `GR00T-WholeBodyControl/`.
- Use `gear_sonic`'s Hydra override syntax (`++path.to.key=value`) rather than editing config files in the vendored repo.
- Real SONIC fine-tuning needs Bones-SEED + multi-GPU; the dummy sbatch is only a smoke test.

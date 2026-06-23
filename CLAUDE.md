# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## WARNING
NEVER run heavy memory or compute jobs in login node!!!
if you need heavy compute just srun CPU or sbatch CPU.
If you have to debug with a GPU srun GPU and scancel it when you are done. If you are doing a sequence of debugging with GPU srun, then only keep for maximum 5 hours. scancel if you are done.
Only then run SBATCH jobs.

## Two machines (READ FIRST — translate paths & env names per machine)

This project runs on **two machines**, both on the `pilab` Slurm cluster. The committed `scripts/*` and `README.MD` are written for **Machine A**; identify which machine you're on (`hostname`, `echo $HOME`) and translate paths/env-names accordingly before running anything.

| | **Machine A** — what the committed scripts/`README.MD` assume | **Machine B** — `pilabslurm-slurm-login-001` (this one) |
|---|---|---|
| Repo root | `/nfsdata/home/jungbin.cho/TAP` | `/home/jungbin_cho/TAP` |
| Conda prefix | `/nfsdata/home/jungbin.cho/miniconda3` | `/home/jungbin_cho/miniconda3` |
| SONIC env | `env_isaaclab` | `isaaclab` (Isaac Lab 2.3.0, py3.11; **`gear_sonic`/`trl`/`smpl_sim` extras NOT confirmed installed** — may be a bare Isaac Lab env) |
| kimodo + TMR-G1 env | `kimodo_soma` | `kimodo` (py3.10, torch 2.4.0; `kimodo` installed editable from **`/home/jungbin_cho/kimodo_open`**, NOT `TAP/kimodo`; `tmr_g1` not installed yet) |
| GMT env | `gmt` | **absent** |

This (Machine B) is a **login node** — no GPU (`nvidia-smi` fails); `sbatch`/`srun`/`sinfo` work, so the WARNING above applies. **Not yet present on Machine B**: the vendored `GR00T-WholeBodyControl/`, `GMT/`, `SOMA-X/` checkouts; the `data/` and `runs/` trees; the `gmt` env. Only `TAP/` (this repo: `docs/ kimodo/ scripts/ tmr_g1/`) and a sibling `/home/jungbin_cho/kimodo_open` exist here.

Working across both:
- **Before running a committed `scripts/*.sbatch` / `*.sh` on Machine B, translate the `/nfsdata/...` paths, conda prefix, env names, and `cd` targets** (or they fail). Prefer making scripts machine-agnostic (derive root from `$(git rev-parse --show-toplevel)` / `$HOME`, resolve the conda base dynamically) so the same script runs on both — don't fork per-machine copies.
- On Machine B, `import kimodo` resolves to **`/home/jungbin_cho/kimodo_open`**, not `TAP/kimodo` — confirm you're getting the `TMRMotionRep`/`G1Skeleton34` you intend. To use `TAP/kimodo` + `tmr_g1`, `pip install -e ./kimodo` and `pip install -e .` into the chosen env (still honoring the namespace-shadowing `cd`-out-of-repo trap, `docs/notes.md` §6).
- SONIC / GMT / full-data work on Machine B needs the vendored repos cloned + data pulled (`README.MD` §1–4). Either way, jobs use `--partition=pilab --qos=qos-pi-top`.

## Technical notes
**See `docs/notes.md`** for important facts that aren't obvious from code: kimodo-G1's 417-dim feature space; TMR's positions-only 186/210-dim feature; data sources shared with `kimodo_open`; LLM2Vec freezing & cache layout; G1 MJCF being 30 bodies (not 34); namespace-package shadowing under editable install; benchmark crop-index FPS mismatch; SONIC's 50 Hz tick; SONIC eval saving rules; the unified G1 NPZ format; and §9.10's down-weighted-generation-loss breakthrough. **See `docs/tmr_g1_plan.md`** for the full TMR-G1 training design and `docs/tmr_g1_improvement_plan.md` for the tuning log.

## What this repo is

A research scaffold for **Tracker-Aware Kinematic Motion Planning** (`README.MD`, ICRA 2027 target). The thesis: a kinematic motion **planner** (Kimodo) should be optimized not only for human-motion realism but for downstream **trackability** under a frozen whole-body **tracker** (SONIC / GMT). The core experiment (`README.MD` §6.1) is a controlled matrix freezing one of {planner, tracker} while adapting the other.

The planner, tracker, and trackability critic must stay **independently swappable** — never hard-wire one to another.

## Repo layout

This repo (`TAP/`) holds only the research scaffolding that is committed: `scripts/`, the `tmr_g1/` + `kinematic_planner/` packages, `docs/`, `pyproject.toml`. The upstream repos and all data are **gitignored** (large, LFS-backed, own git history — see `.gitignore`) and may or may not be checked out on a given node.

- `kinematic_planner/` — **committed**. The TAP **planner**: a from-scratch text→G1-motion **MDM diffusion model** trained on Bones-SEED G1 GT in a new HumanML3D-style representation (`g1_rep_v1`), decoding to executable G1 qpos. This is the generative motion prior the trackability work adapts. See "Kinematic planner" below + `kinematic_planner/NOTES_g1_rep_v1.md`.
- `tmr_g1/` — **committed, editable-installed package** (`pip install -e .`). A from-scratch **TMR (Text-Motion Retrieval) model trained on the G1 embodiment** from Bones-SEED. This is the project's main original code (the SOMA-trained TMR's training code was never released). Serves as the text-alignment / semantic-preservation critic. See "TMR-G1" below.
- `scripts/` — user-owned launch wrappers + the data-prep / generation / eval pipeline. **All new harness code goes here**, never inside a vendored repo.
- `docs/` — `notes.md` (gotchas), `tmr_g1_plan.md`, `tmr_g1_improvement_plan.md`.
- `kimodo/` — vendored `nv-tlabs/kimodo`: the **kinematic motion planner** (diffusion, MDM-like) + the `TMRMotionRep`/`KimodoMotionRep` motion-representation library + `G1Skeleton34`/`G1Skeleton30` + the benchmark/metric code that `tmr_g1` imports. Treat as a dependency; do not edit.
- `GR00T-WholeBodyControl/` — vendored `NVlabs/GR00T-WholeBodyControl`: the **SONIC** whole-body tracker (train/eval/deploy). Do not edit.
- `GMT/` — vendored `zixuan417/humanoid-general-motion-tracking`: a second tracker, **inference-only** (no upstream training/retargeter code yet).
- `SOMA-X/` — vendored `NVlabs/SOMA-X`: SOMA parameter generation / retargeting, used to turn Kimodo SOMA output into trackable motions and for render comparisons.
- `runs/` — Slurm logs + Hydra/training output dirs (gitignored). TMR-G1 runs land in `runs/tmr_g1/<ver>/`; SONIC evals in `runs/sonic_eval/<tag>-<jobid>/`.
- `data/` — gitignored data tree (see "Data on disk").

> Vendored-repo setup (clone + LFS pull + env) lives in `README.MD` §0–4. Clone the four upstream repos as siblings inside `TAP/` with the exact directory names above; scripts assume them.

## Environments

**Three envs, one per stack** — do not mix. Names below are the **Machine A** names the scripts use; on Machine B substitute per the "Two machines" table (`kimodo_soma`→`kimodo`, `env_isaaclab`→`isaaclab`, `gmt`→absent), and source conda from `$HOME/miniconda3` instead of `/nfsdata/...`.

### `kimodo_soma` (Machine B: `kimodo`) — kimodo planner + SOMA-X + **TMR-G1 training/eval**

The most-used env for current work. Shared by `kimodo`, `SOMA-X`, and `tmr_g1`. Python ≥3.10. `pip install -e ./kimodo`, `pip install -e ./SOMA-X`, `pip install -e .` (for `tmr_g1`).

> **Namespace-shadowing trap** (`docs/notes.md` §6): `kimodo/` ships both `kimodo/assets.py` and a `kimodo/assets/` dir, so `import kimodo` breaks whenever the cwd contains a `kimodo/` directory. **Always launch Python from a cwd without one** — every `kimodo_soma` sbatch does `cd /tmp` (or `cd` out of the repo root) before running. The training/eval entry points are invoked by absolute path for this reason.

### `env_isaaclab` (Machine B: `isaaclab`) — for SONIC training/inference

Python 3.11.15, PyTorch 2.7.0+cu128, Isaac Lab 2.3.0, Isaac Sim 5.1.0.0, SONIC training extras (`trl==0.28.0`, `accelerate`, `smpl_sim`, `hydra-core`, `wandb`, …). `gear_sonic` installed editable. Activate:

```bash
source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
```

**Missing-from-`pyproject` runtime deps**, manually pip-installed: `open3d`, `vector_quantize_pytorch` — re-add if rebuilding the env. `git-lfs` lives in `base` (not on PATH globally); re-pull LFS from `base`. SONIC's `sonic_release` config officially targets Isaac Lab **2.3.2**; we're on **2.3.0** (smoke-tested fine — inspect for API drift if training silently misbehaves). CUDA only shows on a GPU node — all SONIC runs need a GPU via Slurm.

### `gmt` (Machine B: absent — create if needed) — for GMT inference

Python 3.8, torch 2.4.1+cu121, MuJoCo 3.2.3, numpy 1.23.0, `mujoco-python-viewer 0.1.4`. **CPU-only on this cluster**: GMT's pinned torch 2.4.1 lacks sm_120 kernels for the Blackwell GPUs; MuJoCo runs on CPU and the JIT policy is tiny, so CPU is fine. The upstream viewer needs a GLFW display even in `'offscreen'` mode and headless nodes have none — use `scripts/gmt_smoke.py` (stubs the viewer), **not** `GMT/sim2sim.py` directly.

> Paths/env-names hardcoded in `scripts/` are **Machine A's**; see the "Two machines" section for the Machine B translation. Prefer making edited scripts machine-agnostic over forking per-machine copies.

## Slurm

Use `--partition=pilab --qos=qos-pi-top` for research jobs. Each pilab node has 8× RTX Pro 6000 Blackwell, 128 CPUs, ~2 TB RAM (defaults: 16 CPUs / 240 GB per GPU). Templates: `scripts/finetune_sonic_dummy.sbatch` (SONIC), `scripts/train_tmr_g1_v7.sbatch` (TMR-G1), `scripts/eval_sonic_set.sbatch` (env-var-driven SONIC eval).

## The TAP experiment pipeline (planner → tracker → metrics)

The README §6.1 matrix is run as a per-subset pipeline of `scripts/`. The unit of work is a **subset.jsonl** (one row per benchmark test case: `text`, `move_name`, crop window, seed):

1. `build_eval_subset.py` — walk a kimodo benchmark testsuite → `subset.jsonl`.
2. **GT branch** — `build_gt_motion_lib.py`: crop the matching Bones-SEED G1 CSV → session-tree of Bones-SEED CSVs → (via SONIC's `convert_soma_csv_to_motion_lib.py`) motion_lib PKLs. This is the per-case upper bound.
3. **Planner branch** — `run_kimodo_for_subset.py` (sbatch: `kimodo_gen_subset.sbatch` / `kimodo_gen_batched.sbatch`): generate motion with **Kimodo-G1-SEED-v1**, adapt to Bones-SEED CSV (`kimodo_csv_to_bones_seed.py`: m→cm, quat→Euler-deg, rad→deg, MuJoCo actuator order), → motion_lib PKLs.
4. **Tracker eval** — `eval_sonic_set.sbatch` (env vars `MOTION_LIB`, `EVAL_TAG`): runs SONIC's `im_eval` over a motion_lib dir → `metrics_eval.json` (success/MPJPE/fall — the **trackability signal**). Note: SONIC only writes `metrics_eval.json` when `+eval_output_dir=...` is passed (`docs/notes.md` §10).
5. **Compare** — `compare_metrics.py` (GT vs Kimodo metric tables) and `render_compare.py` (2×2 side-by-side mp4: Kimodo ref / SONIC-tracking-Kimodo / GT ref / SONIC-tracking-GT).

Text-alignment / semantic preservation is graded separately by **TMR-G1** retrieval (below). Sharding helpers: `shard_subset.py`, `split_pkl_files.py`.

## Kinematic planner: text→G1 motion MDM (`kinematic_planner/`)

The TAP **planner**. An MDM (one-stage diffusion) text→motion model trained on **Bones-SEED G1 GT** in a new representation **`g1_rep_v1`**, generating motion that decodes to **executable G1 qpos** (the input a tracker like SONIC would track). Reuses kimodo's MDM stack by import (`OnestageDenoiser` + `Diffusion` + LLM2Vec `CachedTextEncoder` + flat MDM masked-L2 loss); only the rep, dataset, and configs are new.

- **Rep `g1_rep_v1.py`** (142-D): `rot_velocity(1) + lin_velocity(2) + root_height(1) + root_orient_6d(6) + joint_angles(29, qpos) + ric_data(99) + foot_contacts(4)`. Heading-canonical, heading-invariant. Decode = integrate root → **root + 29 qpos angles → G1 FK**. **Validated lossless**: positions and the executable G1 pose both reconstruct to **0 m**; the residual vs raw mocap is only the G1's intrinsic 1-DOF hardware limit. Full design + the root-orientation fix: `kinematic_planner/NOTES_g1_rep_v1.md`.
- **Pipeline** (run each from the `kinematic_planner/` dir, `kimodo` env — running `python kinematic_planner/<x>.py` keeps `g1_*` importable while `import kimodo` resolves to the installed `kimodo_open`, sidestepping the namespace trap):
  1. `build_features.py --split <split> --out-root <feats>` — encode g1_rep_v1 from unified NPZ + CSV angles (multi-worker, a2 CPU).
  2. `compute_stats.py --feat-root <feats> --out-dir <stats>` — flat Mean/Std.
  3. `build_text_index.py --split <split> --cache <llm2vec.pt> --out <json>` — natural Bones-SEED captions ∩ LLM2Vec cache (98% coverage).
  4. `sbatch train_g1_rep_v1.sbatch` — MDM train (Bones-SEED recipe: lr 2e-5, bf16, EMA 0.995, CFG 0.1, batch 128, 200k steps; TensorBoard). Single GPU on a2.
  5. `sample_g1.py --ckpt <ckpt> --n-steps 50 [--prompts ...]` — text → 50-step CFG DDIM → g1_rep_v1 → joints + **MuJoCo qpos** npz (live LLM2Vec encoder, GPU).
  6. `render_samples_kimodo.py` — viz via kimodo's `render_soma` (skeleton, matplotlib/CPU — see render caveat).
- **Data artifacts** (Machine B, under `/home/jungbin_cho/seed/`, **gitignored — rebuild per machine**): `g1_unified_npz/` (the §9.5 unified NPZ), `g1_rep_v1_feats/`, `g1_rep_v1_stats/`, `g1_rep_v1_text_{full,small}.json`. Text cache: `/home/jungbin_cho/kimodo_caches/bones_seed_llm2vec_small.pt` (227K captions, the full-corpus cache despite the name). Splits: `/home/jungbin_cho/Kimodo-Motion-Gen-Benchmark/splits/train_split_paths{,_small,_medium}.txt` (128K / 12.8K / 38.5K).
- **Trained checkpoint** (Machine B, gitignored): `runs/mdm_g1_rep_v1_full/ckpt_final.pt` (200k steps, full split; loss 1.62→0.046). The other machine must retrain to obtain it.
- **Machine-A note**: paths are hardcoded to `/home/jungbin_cho/...`; `g1_data.py` honors `$SEED_ROOT`, but the configs + the `G1_XML` in sample/render/test hardcode Machine-B paths — adjust per the "Two machines" table.
- **Render caveat**: MuJoCo headless GL does **not** work on this cluster (compute-only GPU driver: no `libEGL_nvidia`/NVIDIA EGL ICD → `eglQueryDevices`=0; conda `mesalib` ships no `libOSMesa`). So `render_samples.py` (MuJoCo, EGL/OSMesa) fails here — use **`render_samples_kimodo.py`** (kimodo `render_soma`, pure matplotlib/CPU). A textured-robot mesh render needs a machine with real graphics libs (the qpos in each sample npz is portable to `scripts/render_g1_qpos.py` there).

## TMR-G1: the from-scratch retrieval model (`tmr_g1/`)

A TMR dual-encoder (ACTOR-style transformer, VAE) re-implemented because NVIDIA released only the SOMA-trained weights, not the trainer. Architecture mirrors `nvidia/TMR-SOMA-RP-v1` (`latent_dim=256, ff_size=1024, num_layers=6, num_heads=4`) but on `G1Skeleton34` + `TMRMotionRep` (G1) features. Text side is a **frozen LLM2Vec** (Llama-3-8B-Instruct, mntp-supervised, 4096-d) consumed via a **precomputed cache** (`CachedTextEncoder`) — never encode text live during training.

- **Train** — `tmr_g1/training/train.py` (argparse, not Hydra). Loss = recon + `λ_kl`·KL + `λ_contrastive`·InfoNCE; `--cross-modal-recon` decodes motion from the text latent too (TMR-style alignment). Writes `config.json` + `tb/` + `step_*.pt`/`last.pt` to `--out-dir`. Reference launch: `scripts/train_tmr_g1_v7.sbatch` (the latest; v1→v7 iterate the recipe — see `docs/notes.md` §9.9–9.10).
- **Eval** — `tmr_g1/eval_retrieval.py`: text→motion R@{1,2,3,5,10}/MedR on the kimodo testsuite, comparable to the published GT rows.
- **GOTCHA**: encode motions at eval **exactly as training did** — `TMRMotionRep(posed_joints, to_normalize=True, to_canonicalize=...)` then take the encoder **mean (mu)**. A mismatch silently tanks retrieval. Crop indices in the testsuite are **30 fps**; the NPZs are 20 fps — rescale (`scale = fps/30`).
- **Architecture/build**: `tmr_g1/model/tmr_model.py` (`build_tmr_g1` returns `(tmr, motion_rep, decoder)`); skeleton helper `tmr_g1/skeleton_g1_30.py`; cached text `tmr_g1/model/cached_text.py`; losses `tmr_g1/training/losses.py`; inline eval during training `tmr_g1/training/inline_eval.py`; stats `tmr_g1/training/build_stats.py`.

## SONIC: where the entry points live

All paths inside `GR00T-WholeBodyControl/`:

- **Train / fine-tune** — `gear_sonic/train_agent_trl.py` (Hydra). Release recipe: `+exp=manager/universal_token/all_modes/sonic_release` (3 encoders: G1, teleop, SMPL). Fine-tune by adding `+checkpoint=sonic_release/last.pt`. Multi-GPU: wrap in `accelerate launch --num_processes=N` (real fine-tuning targets 64+ GPUs).
- **Eval / inference** — `gear_sonic/eval_agent_trl.py`. Pass `++eval_callbacks=im_eval ++run_eval_loop=False` + motion-path overrides; the released checkpoint's embedded `config.yaml` points at internal NV paths and must be overridden (`++manager_env.commands.motion.motion_lib_cfg.{motion_file,smpl_motion_file}=...`).
- **Continuous eval daemon** — `gear_sonic/eval_exp.py`.
- **Data processing** (no Isaac Lab needed) — `gear_sonic/data_process/{convert_soma_csv_to_motion_lib,filter_and_copy_bones_data,extract_soma_joints_from_bvh,split_pkl_files}.py`.
- **Download / preflight** — top-level `download_from_hf.py` (`--sample`, `--training`, `--training --no-smpl`), `check_environment.py --training`.

Full guide: `GR00T-WholeBodyControl/docs/source/user_guide/training.md`.

## Data on disk

Under `data/` (gitignored). Central to current work:

- `data/bones_seed/g1_unified_npz/<session>/<move>.npz` — **canonical 20-fps G1 motions** (`posed_joints`, `local_rot_mats`, `root_positions`, kimodo Y-up). Input to **both** TMR-G1 (uses `posed_joints`) and a custom Kimodo-G1 trainer. Built by `g1_csv_to_unified_npz.py` from the raw 120-fps `data/bones_seed/g1/csv/` (142 K Bones-SEED G1 retargets, 29-DOF qpos).
- `data/bones_seed/g1_feat_npz/` — precomputed `TMRMotionRep` features (`--feat-root`, speeds the dataloader).
- `data/bones_seed/metadata/metadata/{seed_metadata_v004.csv, seed_metadata_v002_temporal_labels.jsonl}` + `data/bones_seed/multi_timeline.jsonl` — the three text sources: **natural** (`content_natural_desc_1..4`), **single** (temporal-segment labels), **multi** (merged timelines).
- `data/bones_seed/{g1_text_emb.pt, eval_text_emb.pt}` — LLM2Vec caches (train / eval) built by `build_g1_text_emb_cache.py` / `build_eval_text_cache.py`.
- `data/bones_seed/tmr_g1_stats_v3/` — per-feature mean/std for motion-rep normalization.
- `data/kimodo_benchmark/splits/{train,test_content,test_repetition}_split_paths.txt` — official 128 K / 7 K / 7 K split; `data/kimodo_benchmark/testsuite/` — text2motion eval cases.
- `data/tmr_soma_rp_v1/` — released SOMA TMR (architecture/stats reference, warm-start source).

SONIC assets (under `GR00T-WholeBodyControl/`): `sonic_release/{last.pt,config.yaml}` (released checkpoint; override the internal paths in `config.yaml` when loading); `sample_data/{robot_filtered,smpl_filtered,soma_filtered}/` (2 walking motions for smoke runs). The full Bones-SEED motion library (~130 K) + 30 GB SMPL tarball are not downloaded — `python download_from_hf.py --training` only when needed.

## Common commands

Run from the repo root (`/nfsdata/home/jungbin.cho/TAP` on Machine A, `/home/jungbin_cho/TAP` on Machine B); env names below are Machine A's (`kimodo_soma`→`kimodo`, `env_isaaclab`→`isaaclab` on Machine B). Long jobs go through Slurm, never the login node.

```bash
# --- TMR-G1 (kimodo_soma env; cd out of repo root to dodge the kimodo shadowing) ---
sbatch scripts/train_tmr_g1_v7.sbatch                    # train (latest recipe)
python tmr_g1/eval_retrieval.py --ckpt runs/tmr_g1/v7/last.pt \
  --stats-path data/bones_seed/tmr_g1_stats_v3 \
  --text-emb-cache data/bones_seed/eval_text_emb.pt \
  --g1-npz-root data/bones_seed/g1_unified_npz \
  --testsuite data/kimodo_benchmark/testsuite --out runs/tmr_g1/v7/retrieval.json
sbatch scripts/build_g1_text_emb_cache.sbatch            # (re)build LLM2Vec cache
sbatch scripts/g1_csv_to_unified_npz.sbatch              # build unified G1 NPZs

# --- TAP planner→tracker pipeline ---
python scripts/build_eval_subset.py ...                  # testsuite -> subset.jsonl
SUBSET=... OUT_NPZS=... OUT_CSVS=... OUT_PKLS=... sbatch scripts/kimodo_gen_subset.sbatch
MOTION_LIB=runs/eval_kimodo/<tag>_pkls EVAL_TAG=kimodo sbatch scripts/eval_sonic_set.sbatch
python scripts/compare_metrics.py ...                    # GT vs Kimodo metrics
sbatch scripts/render_compare.sbatch                     # 2x2 side-by-side mp4

# --- SONIC (env_isaaclab; needs a GPU node) ---
cd GR00T-WholeBodyControl && python check_environment.py --training
bash scripts/inference_sonic.sh metrics                  # success rate + MPJPE
bash scripts/inference_sonic.sh render                   # .mp4 rollouts
sbatch scripts/finetune_sonic_dummy.sbatch               # 5-iter PPO smoke test

# --- GMT (gmt env; inference only) ---
python scripts/gmt_smoke.py --motion walk_stand.pkl --steps 500   # headless
sbatch scripts/inference_gmt.sbatch                               # all 8 motions
```

## When adding code

- Keep planner / tracker / trackability-critic **independently swappable** — §6.1's matrix freezes one while training another.
- New launch / harness / fine-tune scripts go under `scripts/`, never inside a vendored repo.
- Use Hydra override syntax (`++path.to.key=value`) for SONIC rather than editing vendored config files.
- Real SONIC fine-tuning needs the full Bones-SEED lib + multi-GPU; the dummy sbatch is only a smoke test.
- When you discover a non-obvious fact, append it to `docs/notes.md` (don't rewrite — older entries still hold).

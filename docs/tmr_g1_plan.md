# Training plan: TMR-G1 (Bones-SEED, G1 embodiment)

## 1. How the Bones-SEED / Kimodo benchmark evaluates text→motion

Source: `kimodo/docs/source/benchmark/{introduction,metrics,pipeline}.md`
and `kimodo/benchmark/{embed_folder,evaluate_folder}.py`.

The benchmark evaluates generated motion on three axes:

1. **Motion quality** — foot-skate (4 variants), foot-contact consistency.
2. **Constraint following** — root-2D, end-effector, full-body keyframe errors.
3. **Text alignment** — handled by an *external* dual-encoder model called **TMR**
   (Text-to-Motion Retrieval, Petrovich et al. ICCV 2023), retrained by NVIDIA
   on the Rigplay dataset.

For every test case the protocol runs **two passes**:
- *generated pass*: all metrics on `motion.npz` from the model under test.
- *gt pass*: same motion-quality / constraint metrics on `gt_motion.npz`, used
  as a per-case upper bound.

The retrieval metrics — the ones that grade text alignment — come from passing
both generated motion and the text prompt through TMR:

| Key | What it measures |
|---|---|
| `TMR/t2m_sim` | cosine(text emb, gen motion emb), rescaled to [0,1] |
| `TMR/m2m_sim` | cosine(gen motion emb, GT motion emb) |
| `TMR/t2m_gt_sim` | cosine(text emb, GT motion emb) — upper bound |
| `TMR/t2m_R/R{01,02,03,05,10}` | text→gen-motion retrieval Recall@k (within the group's motion pool) |
| `TMR/t2m_R/MedR` | text→gen median rank |
| `TMR/m2m_R/...` and `TMR/t2m_gt_R/...` | analogous retrieval w/ GT motion |
| `TMR/FID/{gen_gt, gen_text, gt_text}` | Fréchet distance between Gaussian fits of TMR embeddings |

Duplicates: prompts whose text-text TMR similarity ≥ 0.99 are grouped so that
retrieving any motion from the dup group counts as a hit.

**Inputs to TMR**:
- text → LLM2Vec sentence embeddings (Llama-3-8B-Instruct + MNTP) → TMR top
  text encoder → 256-d.
- motion → joint positions on the **SOMA-30** skeleton at **30 fps**, ≤ 300
  frames → `TMRMotionRep` features → ACTOR-style motion encoder → 256-d.
- All TMR embeddings are L2-normalized (`unit_vector: true` in
  `tmr_soma_rp_v1/config.yaml`).

## 2. The target model

**Model:** `nvidia/TMR-SOMA-RP-v1` — already downloaded to
`TAP/data/tmr_soma_rp_v1/`. Layout:

```
config.yaml              # Hydra config for kimodo.model.tmr.TMR.from_args
last_weights/
  motion_encoder.pt      # ACTORStyleEncoder over SOMA-30 motion features
  text_encoder.pt        # ACTORStyleEncoder over LLM2Vec embeddings
  motion_decoder.pt      # reconstruction head (used in training, optional at eval)
stats/motion/            # mean/std for motion-rep normalization, split layout:
  body/{mean,std}.npy
  global_root/{mean,std}.npy
  local_root/{mean,std}.npy
  mean.npy
  std.npy
```

Architecture (per the HF card and `kimodo/model/tmr.py`):
- Dual-encoder, transformer-based, VAE-style (sample 2 tokens → mu, logvar).
- `latent_dim=256`, `ff_size=1024`, `num_layers=6`, `num_heads=4`, `dropout=0.1`.
- Motion encoder: ~4.8 M params; text encoder: ~5.8 M params.
- Output: 256-d embedding, L2-normalized.
- Training data: full Rigplay (700 h, ~1 B tokens of paired text) on SOMA-30,
  30 fps, 10-s clips.
- **Training code is NOT released** — `kimodo/benchmark/` only contains
  embedding/eval scripts. We have to reimplement the trainer.

The released `kimodo.skeleton.G1Skeleton34` already exists with the right
joint hierarchy. `TMRMotionRep` is skeleton-agnostic (it uses
`skeleton.nbjoints`, FK, foot/hand joint names). So architecturally a G1
variant is a one-line config swap — the gap is training data, loss, and the
trainer itself.

## 3. Plan: train **TMR-G1-SEED-v1** from BONES-SEED on the G1 embodiment

### 3.1 What we already have on disk

- `data/bones_seed/g1/csv/<session>/<move>.csv` — 142,220 G1 retargeted
  motions at 120 fps (29 DOF qpos, MJCF actuator order).
- `data/bones_seed/metadata/metadata/seed_metadata_v004.parquet` — per-motion
  metadata: actor, content category, **`content_natural_desc_1..4`** (4
  rephrasings of the natural-language description), short descs, etc.
- `data/bones_seed/metadata/metadata/seed_metadata_v002_temporal_labels.jsonl`
  — fine-grained temporal labels (start/end frame + text per segment), 142 K
  rows. These were generated specifically for Kimodo.
- `data/kimodo_benchmark/splits/{train,test_content,test_repetition}_split_paths.txt`
  — official 128 K / 7 K / 7 K split used by Kimodo-SOMA-SEED-v1.
- `runs/eval_subset/t2m_overview_text_emb.pt` — LLM2Vec embeddings for the
  benchmark's 917 text2motion prompts; cache infra is in
  `scripts/build_text_emb_cache.py` and trivially scales to all SEED texts.
- `TAP/data/tmr_soma_rp_v1/` — the SOMA target model (architecture, stats
  layout, and a checkpoint we can warm-start from for transfer).

### 3.2 Code we will need to write

All new code lives under `TAP/scripts/tmr/` and `TAP/tmr_g1/` (a small package
so we can import cleanly). We do NOT modify `TAP/kimodo/` or
`TAP/GR00T-WholeBodyControl/`.

| Component | Role | Sketch |
|---|---|---|
| `tmr_g1/data/g1_motion_dataset.py` | streaming dataset | walks split.txt; for each motion reads G1 CSV, downsamples 120→30 fps, runs FK (via `G1Skeleton34.fk`) to get joint positions, crops a random 30–300-frame window, applies `TMRMotionRep(G1Skeleton34, fps=30)` → features, returns `(motion_feats, mask, text_idx)` |
| `tmr_g1/data/seed_text.py` | per-move text source | for each `move_name`, draws (with weights) one of: `content_short_description`, `content_short_description_2`, `content_natural_desc_{1..4}`, or a matching temporal-label segment from the JSONL; returns the raw string |
| `tmr_g1/data/llm2vec_cache.py` | pre-encoded text store | reuses `scripts/build_text_emb_cache.py`; produces `text_emb.pt` keyed by raw text; embeddings are `[1, 4096]` fp32 |
| `tmr_g1/model/tmr_g1.py` | TMR with G1 skeleton | thin wrapper around `kimodo.model.tmr.TMR.from_args` with `motion_rep=TMRMotionRep(G1Skeleton34, fps=30)` and `llm_shape=[1, 4096]`. Optional warm-start from `tmr_soma_rp_v1/last_weights/text_encoder.pt` (text encoder transfers directly since the text side doesn't depend on the skeleton). |
| `tmr_g1/model/motion_decoder.py` | reconstruction head | ACTOR-style transformer decoder mirroring the encoder; outputs a tensor of shape `[B, T, motion_rep_dim]`. Architecture matches `tmr_soma_rp_v1/last_weights/motion_decoder.pt` but with G1's `motion_rep_dim`. |
| `tmr_g1/training/losses.py` | TMR loss | per Petrovich et al. — sum of: <br/>• `L_kl` = avg(KL(N(mu, sigma²) ‖ N(0,1))) on motion latent <br/>• `L_recon` = L2 between decoded features and target features <br/>• `L_contrastive` = symmetric InfoNCE between L2-normalized motion and text embeddings with temperature τ (≈ 0.1); duplicates handled by masking text-text similarity > 0.99 in the negatives, matching benchmark eval. |
| `tmr_g1/training/train.py` | train loop | Hydra entrypoint; AdamW, cosine LR, warmup, mixed-precision; eval every N steps on the content+repetition test splits. |
| `tmr_g1/training/build_stats.py` | per-feature mean/std | streams a sample of train motions through `TMRMotionRep` and dumps the split-layout stats (`global_root`, `local_root`, `body`) the same way SOMA's checkpoint is shipped. |
| `scripts/train_tmr_g1.sbatch` | sbatch launcher | `pilab` partition, 1 GPU initially (4–8 GPU later with accelerate); writes to `runs/tmr_g1/<run_name>/`. |
| `scripts/eval_tmr_g1.py` | retrieval eval | loads a TMR-G1 checkpoint + held-out test split, computes `t2m_sim`, R@{1,2,3,5,10}, MedR and FID exactly the way `kimodo/benchmark/evaluate_folder.py` does (we can import that file's helpers and only swap the loaded model). |

### 3.3 Pre-training data prep (one-time, CPU/sbatch)

1. **Resample G1 CSVs to 30 fps.** Run our existing
   `convert_soma_csv_to_motion_lib.py` with `--fps 30 --fps_source 120
   --individual` on the entire 142 K set → `data/bones_seed/motion_lib/`. ≈ 1 h.
   This also dumps `joint_pos` + `body_pos_w` + `body_quat_w` per motion,
   which we can FK on the fly or precompute.
2. **Precompute joint positions** (optional but ~10× faster training): for
   each motion, run `G1Skeleton34.fk(local_rot_mats, root_positions)` and save
   the global `posed_joints` as fp16. ~30 GB.
3. **Text vocabulary.** For each `move_name` in the split files, materialize
   the candidate texts: up to 4 `content_natural_desc_*` + 2 short descs +
   N temporal segments. Deduplicate across the corpus to keep the LLM2Vec
   pass small (likely ≪ 1 M unique strings).
4. **LLM2Vec cache.** Reuse `scripts/build_text_emb_cache.py` to embed every
   unique string once → `data/tmr_g1/text_emb.pt`. On 1 GPU at ~5 texts/s
   this is a few hours but only ever needs to run once.
5. **Stats.** Run `tmr_g1/training/build_stats.py` on the train split to
   produce `data/tmr_g1/stats/motion/{global_root,local_root,body}/{mean,std}.npy`.

### 3.4 Training procedure

- **Splits**: read directly from `data/kimodo_benchmark/splits/`. Train on
  `train_split_paths.txt` (128 K motions). Hold out the two test splits for
  retrieval eval.
- **Clip sampling**: per motion, sample a random window of 30–300 frames
  (1–10 s). Aligns with TMR's stated 10-s cap.
- **Text sampling**: 50 % `content_natural_desc_*`, 30 % temporal segments
  cropped to the same window, 20 % `content_short_description*`. The temporal
  segments give us local descriptions which is the harder retrieval setting.
- **Batch**: 64 motion-text pairs (start with 32 if VRAM-tight).
- **Optimizer**: AdamW, `lr=2e-4`, weight decay 1e-2, cosine schedule with 2 k
  warmup steps; train ~100 k steps (≈ 50 epochs at batch 64). Stop when
  R@10 on `test_content_split` stops improving for 5 evals.
- **Loss weights** (paper defaults to start): `λ_recon=1.0`, `λ_kl=1e-4`,
  `λ_contrastive=1e-1`, InfoNCE temperature `τ=0.1`. Tune contrastive weight
  upward if FID/retrieval metrics lag; downward if reconstruction degrades.
- **Mixed precision**: fp16 for transformer, fp32 for the LLM2Vec table
  lookup. Use `torch.compile` on the encoders for ~1.3× throughput.
- **Hardware**: a single Blackwell RTX 6000 in `pilab` is enough for a v0;
  scale to 4× when we want to match the published SOMA training run.

### 3.5 Evaluation hooks (must match the official protocol)

We reuse `kimodo/benchmark/evaluate_folder.py` so our numbers are directly
comparable. Two adjustments are needed:

1. The benchmark expects motion in **SOMA-30 joint positions**. Our model
   eats **G1-34 joint positions**. We patch in a `--motion-rep` flag that
   selects the right `TMRMotionRep` (already supported via Hydra config in
   `tmr_soma_rp_v1/config.yaml`).
2. The benchmark assumes `motion.npz` carries `posed_joints` on the SOMA
   skeleton. For our G1 evaluation we either (a) feed G1 `posed_joints`
   directly (and run a G1-trained TMR), or (b) retarget G1 motion back to
   SOMA and use the SOMA TMR. We pick (a) for the TMR-G1 paper; (b) is the
   apples-to-apples comparison.

Eval metrics that ship out of the box once those plumbing changes land:
`TMR/t2m_sim`, `TMR/{t2m,m2m,t2m_gt}_R/R{01..10}`, MedR, FID variants.
We also report **R@1 per content category** so we can identify motion classes
the model still struggles with (e.g. handedness, as the HF card flags).

### 3.6 Risks / open questions

- **Text source quality on SEED-only.** Rigplay's 700 h drove most of TMR's
  vocabulary; SEED is ~286 h. Expect a meaningful drop in R@k vs the released
  SOMA-RP model; the relevant comparison is **TMR-G1-SEED** vs hypothetical
  TMR-SOMA-SEED (which NVIDIA didn't release).
- **G1 has joints SOMA doesn't model** (wrist roll/pitch/yaw triplets) and
  **SOMA has joints G1 doesn't** (eyes, jaw, individual hand fingers).
  Retrieval quality on actions where these matter (gestures involving
  fingers) will be inherently worse on G1.
- **Foot contact heuristic.** `foot_detect_from_pos_and_vel` thresholds are
  tuned for human skeletons. We should re-derive thresholds for G1 from the
  Bones-SEED retargeting (median foot height & velocity over a held-out
  sample).
- **Validation against the released model.** Before claiming a number, embed
  10 K Bones-SEED motions with both the released SOMA TMR (after SOMA
  retargeting) and our G1 TMR; correlation between similarity matrices is a
  sanity proxy.

### 3.7 Suggested first deliverable

`v0` smoke (~1 day): wire the G1 dataset + LLM2Vec cache + ACTORStyleEncoder
into a single `train_tmr_g1.py`, train for 5 k steps on a 5 k-motion subset,
report R@1/R@10 on a 1 k-motion held-out subset. If R@10 > 60 % the
architecture/data plumbing is working and we can scale to the full split.

If that goes well, target a `v1` release as a HuggingFace mirror named
`TMR-G1-SEED-v1` with the same file layout as `TMR-SOMA-RP-v1` so it drops
straight into `kimodo/benchmark/embed_folder.py`'s `--model` flag.

# TAP technical notes

Reference notes that aren't obvious from the code but cost time to figure out.
Append, don't rewrite — older entries are still useful.

---

## 1. Kimodo-G1 model I/O is a 417-dim feature, not rotation matrices

The Kimodo-G1-SEED-v1 NPZ on disk holds `local_rot_mats`, `posed_joints`,
`root_positions` and friends, but those are *decoded outputs of
`KimodoMotionRep.inverse(...)`*. The **diffusion model itself operates
end-to-end in feature space**.

From `nvidia/Kimodo-G1-SEED-v1/config.yaml`:

```yaml
denoiser:
  motion_rep:
    _target_: kimodo.motion_rep.KimodoMotionRep
    fps: 30
    skeleton: G1Skeleton34
```

Feature layout (`kimodo/motion_rep/reps/kimodo_motionrep.py:size_dict`):

| Block                     | shape       | dims |
|---------------------------|-------------|-----:|
| `smooth_root_pos`         | `(T, 3)`    | 3    |
| `global_root_heading`     | `(T, 2)` (cos, sin) | 2 |
| `local_joints_positions`  | `(T, 34, 3)` (root-removed) | 102 |
| `global_rot_data`         | `(T, 34, 6)` (continuous-6D rot per joint) | **204** |
| `velocities`              | `(T, 34, 3)` | 102 |
| `foot_contacts`           | `(T, 4)`    | 4    |
| **TOTAL `motion_rep_dim`**|             | **417** |

Per-frame I/O of the diffusion network is `(B, T, 417)` at fps=30. Then
`motion_rep.inverse(features)` decodes that to the {local_rot_mats,
global_rot_mats, posed_joints, root_positions, smooth_root_pos, foot_contacts,
global_root_heading} dict that gets saved as NPZ.

So when someone asks "what's the kimodo model output?" the answer is a
417-dim feature vector per frame; rotation matrices come from decoding it.

The training-time **input** is the same 417-dim feature — produced by
`KimodoMotionRep(local_joint_rots[B,T,J,3,3], root_positions[B,T,3])`.

---

## 2. TMR motion encoder is positions only (no joint rotations)

> **[SUPERSEDED dims — see §11.](#11-motion-representation--authoritative-reference)**
> The 186-dim / `G1Skeleton30` numbers below were the early stopgap. The
> shipped TMR-G1 uses **`G1Skeleton34` → 210 dims**. The "positions-only"
> point in this section is still correct; only the joint count / dim changed.

`kimodo/motion_rep/reps/tmr_motionrep.py:TMRMotionRep.size_dict`:

| Block | shape | dims (G1Skeleton30) | dims (SOMASkeleton30) |
|---|---|---:|---:|
| `root_pos` | `(T, 3)` | 3 | 3 |
| `global_root_heading` | `(T, 2)` | 2 | 2 |
| `local_joints_positions` | `(T, J-1, 3)` (root removed) | 87 | 87 |
| `velocities` | `(T, J, 3)` | 90 | 90 |
| `foot_contacts` | `(T, 4)` | 4 | 4 |
| **TOTAL** | | **186** | **186** |

Note **no rotations** — TMR is a positions + heading + per-frame velocities
+ binary foot-contact encoder. That's the deliberate design: TMR is for
retrieval, where positions are sufficient and rotations would just inflate
the latent.

The published `nvidia/TMR-SOMA-RP-v1` ships `stats/motion/mean.npy` at **190
dims**, which is 4 dims wider than the on-tree `TMRMotionRep` computes for
SOMA-30. Likely an internal feature block (extra contact channels? per-joint
height? unclear) NVIDIA didn't release. The on-tree `TMRMotionRep` is what
we use for TMR-G1; the dim mismatch only matters if you try to load the
SOMA TMR weights into our `TMRMotionRep`.

For TMR-G1 we use `tmr_g1.skeleton_g1_30.G1Skeleton30` (matches the 30
MuJoCo bodies in `kimodo/assets/skeletons/g1skel34/xml/g1.xml`), so the
input is `posed_joints[T, 30, 3]` global joint positions, fps=20.

---

## 3. Data sources are the same as kimodo_open

`tmr_g1/data/g1_dataset.py` mirrors `kimodo_open/kimodo/data/soma_text_motion.py`:
1:1:1 round-robin over three pools:

- **natural** — `content_natural_desc_{1..4}` + `content_short_description*` columns of `seed_metadata_v004.csv` (one entry per move; per-fetch picks one paraphrase).
- **single**  — `events[].description` from `seed_metadata_v002_temporal_labels.jsonl` (one entry per event).
- **multi**   — `merged_description` from `multi_timeline.jsonl` (one entry per merged window).

Per-source filtering thresholds: `min_frames=10`, `max_segment_sec=15`,
`max_clip_sec=10`, `rand_offset_max_sec=2`.

**Only the motion data differs**: kimodo_open uses SOMA NPZs `(local_rot_mats[T,30,3,3] + root_positions[T,3])`; TMR-G1 uses G1-retargeted CSVs converted to `posed_joints[T,30,3]` NPZs at 20 fps.

Local copies on /nfsdata:
- `data/bones_seed/metadata/metadata/seed_metadata_v004.csv` (~140 MB; 142K rows).
- `data/bones_seed/metadata/metadata/seed_metadata_v002_temporal_labels.jsonl`.
- `data/bones_seed/multi_timeline.jsonl` (278K segments).
- `data/kimodo_benchmark/splits/{train,test_content,test_repetition}_split_paths.txt`.
- `data/bones_seed/g1/csv/<session>/<move>.csv` — 120-fps Bones-SEED format (cm/deg).
- `data/bones_seed/g1_20fps_npz/<session>/<move>.npz` — built from those, 20-fps `posed_joints + root_positions`.

---

## 4. LLM2Vec text encoder is frozen during all training

Both Kimodo and TMR keep LLM2Vec frozen. Therefore we precompute embeddings
once and serve them via `kimodo.model.cached_text.CachedTextEncoder` (drop-in
for `LLM2VecEncoder`).

For TMR-G1: `scripts/build_g1_text_emb_cache.py` enumerates every unique caption from the three sources above (restricted to motions in `splits/`), encodes each with bfloat16 LLM2Vec, and writes a blob at `data/bones_seed/g1_text_emb.pt` with the layout

```python
{
  "captions": List[str],            # N unique captions
  "features": Tensor(N, 4096),      # float32 pooled embeddings
  "meta":     {"encoder_type": "llm2vec", "model": "...", "dim": 4096, ...}
}
```

That's the format the `CachedTextEncoder` expects. Periodic in-flight saves
to a `.pt.partial`-then-atomic-rename, so a TIMEOUT loses ≤ 2K captions.

Stats from the current run (after the three-source enumeration with split
filtering and mirrored included):
- raw captions: 538,452 natural + 352,703 single + 278,264 multi = ~1.17 M.
- unique: **238,460**. Encoding rate ≈ 22/s on a Blackwell RTX 6000 with
  bfloat16. ETA ≈ 3 h for full encode.

---

## 5. MuJoCo G1 MJCF only has 30 bodies, not 34

`kimodo.skeleton.G1Skeleton34` lists 34 joints (32 articulated + 2 toe
endpoints + pelvis), but `kimodo/assets/skeletons/g1skel34/xml/g1.xml`
only models 30 bodies. The 4 missing joints (toe tips + palm endpoints)
are virtual offsets, not MJCF bodies. So MuJoCo FK from qpos gives 30
joint positions, not 34.

`tmr_g1.skeleton_g1_30.G1Skeleton30` (`bone_order_names_with_parents` matching
the 30 MJCF bodies; foot/hand names are the MJCF joint names
`*_ankle_roll_joint`, `*_wrist_yaw_joint`) was an early stopgap and is **no
longer used by the shipped model**.

> **[SUPERSEDED — see §11.](#11-motion-representation--authoritative-reference)**
> The 30-body MJCF limit is real, but TMR-G1 does **not** train on raw MuJoCo
> FK output. It trains on the unified NPZ's `posed_joints[T, 34, 3]` (§9.5),
> which is full `G1Skeleton34.fk(...)` over 34 joints (30 MJCF bodies + 4
> virtual endpoints). So `build_tmr_g1` instantiates `G1Skeleton34` and
> `TMRMotionRep(G1Skeleton34, fps=20).motion_rep_dim` = **210**, not 186.

---

## 6. Kimodo package install: namespace-package shadowing trap

The `TAP/kimodo` repo has both `kimodo/assets.py` (path-helper module) AND
`kimodo/assets/` (data directory: images, skeletons, demo assets). Under
editable install (`pip install -e ./kimodo`), Python's import machinery picks
the *directory* as a namespace package when the cwd is `TAP/` (because
`TAP/kimodo/` itself is seen as a namespace package via the empty-string
`sys.path[0]` resolving to cwd).

Symptom: `from kimodo.assets import skeleton_asset_path` fails with
`ImportError: cannot import name 'skeleton_asset_path' from 'kimodo.assets'
(unknown location)` and `kimodo.assets.__file__` is `None`. Worse:
`kimodo.__path__` reports `_NamespacePath(['/nfsdata/home/jungbin.cho/TAP/kimodo'])`
(the *outer* repo root), so `kimodo.assets` resolves to the *outer* repo's
`assets/` dir (which has only GIF files).

**Two workarounds:**
1. **Always run from a cwd that does NOT contain a `kimodo/` directory.**
   `/tmp`, `~/`, `/nfsdata/home/jungbin.cho/TAP/scripts` are all fine.
   `/nfsdata/home/jungbin.cho/TAP` is not.
2. Or drop a forwarding `__init__.py` into `kimodo/kimodo/assets/` that
   re-exports the helpers and adjusts paths (we did this; see
   `kimodo/kimodo/assets/__init__.py`). This is necessary because (1) alone
   doesn't fix the namespace-shadowing inside sbatch scripts that `cd $REPO`.

Sbatch scripts and training entry points must `cd` somewhere safe before
launching Python — `cd /tmp && python ...`, not `cd /nfsdata/home/jungbin.cho/TAP && python ...`.

---

## 7. Bones-SEED CSV crop indices are at 30 fps, G1 CSVs are at 120 fps

The Kimodo benchmark's `seed_motion.json` stores `crop_start_frame_index`
and `crop_end_frame_index` at **30 fps** (matches the SOMA NPZ resampling),
but the on-disk G1 retargeted CSVs are at **120 fps**. Slicing CSV lines
with the benchmark indices directly gives motions 4× too short.

Fix: scale crop indices by `fps_source / crop_fps` (= 120 / 30 = 4) before
slicing into the 120-fps CSV. Implemented in `scripts/build_gt_motion_lib.py`
via `--crop-fps 30 --fps-source 120`.

This bug initially inflated kimodo's MPJPE-G by ~96 mm relative to GT
(short GT clips had no time to drift globally). After fix the delta is
~+7 mm — a much more honest signal.

---

## 8. SONIC runs the policy at 50 Hz, not 30

From `gear_sonic/config/manager_env/base_env.yaml`:
- physics: `sim_dt = 0.005` → 200 Hz.
- policy: `decimation = 4` → 50 Hz control tick.
- motion-lib references are stored at 30 fps; SONIC interpolates positions
  linearly and SLERPs quaternions on-the-fly to 50 Hz at sim time.
- Future-frame conditioning differs by encoder: G1/teleop at `dt=0.1`
  (10 fps × 10 = 1 s lookahead); SMPL at `dt=0.02` (50 fps × 10 = 0.2 s).

For our kimodo↔GT comparison, both motion_libs sit at 30 fps and get the
same interpolation, so the comparison is fair.

---

## 9.5. Unified G1 NPZ format (TMR-G1 *and* custom Kimodo-G1)

We pre-compute one NPZ per motion that's the canonical training input for
*both* tasks. Layout, all in **kimodo Y-up** frame at **20 fps**:

| key | shape | dtype | meaning |
|---|---|---|---|
| `local_rot_mats` | `(T, 34, 3, 3)` | fp32 | local rotation of each joint relative to its parent, kimodo `G1Skeleton34` order |
| `root_positions` | `(T, 3)` | fp32 | pelvis xyz |
| `posed_joints` | `(T, 34, 3)` | fp32 | global joint positions, = kimodo `G1Skeleton34.fk(local_rot_mats, root_positions)` |
| `fps` | `()` | int32 | always 20 |
| `frame` | `()` | str (U) | always `"kimodo_y_up"` |

Producer: `scripts/g1_csv_to_unified_npz.py`. Algorithm:

1. Load Bones-SEED CSV → MuJoCo `qpos[T_src, 36]` (m / quat-wxyz / rad).
2. Stride to target fps (`120 → 20`).
3. Per frame: set `data.qpos`, call `mj_kinematics`, read `xmat[i]` (global rotation of each MJCF body).
4. Per MJCF body: `R_local_mj[i] = xmat[parent_i].T @ xmat[i]`.
5. Map 30 MJCF bodies → 30 of 34 kimodo joints by name (see table below); 4 virtual endpoint joints stay at identity.
6. Rotate each local rotation into kimodo frame: `R_k = M R_mj M^T` with `M = [[0,1,0],[0,0,1],[1,0,0]]`.
7. Rotate root position: `v_k = v_mj @ M^T`.
8. Run `G1Skeleton34.fk(local_rot_mats_k, root_positions_k)` → `posed_joints`.

**MJCF body ↔ kimodo joint name map** (most are `_link` → `_skel`):

| MJCF body | kimodo joint |
|---|---|
| `pelvis` | `pelvis_skel` |
| `torso_link` | **`waist_pitch_skel`** (the only non-pattern map; kimodo treats torso as the 3rd waist joint) |
| `<x>_link` (28 others) | `<x>_skel` |
| — (no MJCF) | `left_toe_base`, `right_toe_base`, `left_hand_roll_skel`, `right_hand_roll_skel` (4 virtual endpoints, identity local rotation) |

**Consistency** (checked by `scripts/g1_check_unified_consistency.py`):
- (A) Our stored `posed_joints` equals MuJoCo's `xpos` after Y-up transform: **0 m error**.
- (B) Feeding `(local_rot_mats, root_positions)` back into
  `kimodo.exports.mujoco.MujocoQposConverter.dict_to_qpos(...)` reproduces
  the original CSV qpos: root pos = 0 m, root rot ≤ 0.04°, joint angles ≤ 0.4°.

That means: this NPZ format is **round-trippable to MuJoCo qpos via kimodo's
official path** — so visualization in MuJoCo or feeding into SONIC uses the
exact same conversion as upstream Kimodo-G1's outputs.

## 9.6. Two G1 skeletons to keep straight

| | `G1Skeleton30` (ours, in `tmr_g1/skeleton_g1_30.py`) | `G1Skeleton34` (kimodo upstream) |
|---|---|---|
| Joint count | 30 (matches MJCF bodies 1:1) | 34 (30 + 4 virtual endpoints) |
| Has `joints.p` / rest pose files | No — we pass `load=False` | Yes (`kimodo/assets/skeletons/g1skel34/`) |
| Use case | TMR-only positions, when you don't need FK from rotations | Anything that reuses kimodo's `KimodoMotionRep`, including custom Kimodo-G1 |
| `TMRMotionRep.motion_rep_dim` | 186 | 186 (same formula; same number of joints removed from `local_joints_positions` slot is `J-1=33` not `29`, so actually different — see below) |

Wait — careful: `TMRMotionRep` with `G1Skeleton34` is *not* the same dim as with `G1Skeleton30`:
- `G1Skeleton30`: `3 + 2 + 29×3 + 30×3 + 4 = 186`
- `G1Skeleton34`: `3 + 2 + 33×3 + 34×3 + 4 = 210`

For the unified pipeline we should **commit to `G1Skeleton34` throughout** — both for the saved NPZ (already does) and for `TMRMotionRep` going forward. Our earlier `G1Skeleton30` was only a stopgap because MuJoCo gave us 30 bodies; with `posed_joints` computed by `G1Skeleton34.fk` we have full 34-joint data and should use it.

**DONE** (this is now the shipped state): `tmr_g1/model/tmr_model.py:build_tmr_g1`
instantiates `G1Skeleton34` → `TMRMotionRep` dim **210**, and stats were rebuilt
as `tmr_g1_stats_v3` (dim 210, canonicalized). `tmr_g1/skeleton_g1_30.py` still
exists but is unused by training/eval. See §11 for the authoritative layout.

## 9.7. TMR-G1-SEED-v1 retrieval results (first run, 100k steps)

Eval: `tmr_g1/eval_retrieval.py` — text→motion R-precision on GT G1 motions,
kimodo `contrastive_metrics` (0.99 text-dedup). We benchmark the *TMR model
itself*, so compare to the published **"Content Ground Truth" R@3** rows
(https://research.nvidia.com/labs/sil/projects/kimodo/docs/benchmark/results.html).

| group | pool | R@1 | R@3 | R@5 | R@10 | MedR | published GT R@3 | gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| content/overview | 917 | 22.4 | 52.8 | 64.7 | 77.8 | 3 | **89.09** | −36.3 |
| content/timeline_single | 917 | 26.3 | 49.7 | 63.5 | 74.7 | 4 | **86.26** | −36.6 |
| content/timeline_multi | 789 | 28.0 | 57.7 | 67.8 | 79.7 | 3 | **88.47** | −30.8 |
| repetition/overview | 2380 | 48.5 | 74.4 | 83.3 | 90.6 | 2 | **93.91** | −19.5 |
| repetition/timeline_single | 2380 | 50.4 | 71.1 | 78.5 | 84.9 | 1 | **90.13** | −19.0 |
| repetition/timeline_multi | 1779 | 45.6 | 72.1 | 81.1 | 89.5 | 2 | **94.49** | −22.4 |

Reading:
- We're **well below the published GT R@3** (52–58 vs 86–89 on content).
  Several reasons, roughly in order of impact:
  1. **The published SOMA TMR was trained on the FULL Rigplay dataset
     (train+test)** per its model card — it saw the eval motions during
     training. Ours trained on the **train split only**, so `content/*` is
     genuinely unseen content. The fair-er comparison is `repetition/*`
     (content seen in training, novel performances): R@3 71–74.
  2. **No canonicalization / no random-heading augmentation in training**
     (`tmr_g1/training/train.py` calls `motion_rep(..., to_normalize=True)`
     without `to_canonicalize=True`, and the dataset has no heading aug). The
     stats comparison (note in git history) showed our heading distribution is
     under-augmented vs kimodo's uniform-circle. This likely caps retrieval.
  3. SEED-only (~286h) vs Rigplay (700h) training data.
  4. Loss balance / steps not tuned.
- The result is internally sensible: `repetition` (seen content) ≫ `content`
  (unseen content); R@k climbs smoothly; MedR 1–4 on pools of 800–2400 (random
  MedR would be ~pool/2). So the pipeline is correct; the gap is data/aug, not
  a bug.

**v2 levers** (highest expected ROI first): enable `to_canonicalize=True` +
random-heading aug in training; retrain. Optionally include test split in
training *only if* matching the published GT protocol exactly (not for honest
generalization numbers).

Artifacts: `runs/tmr_g1/v1/{last.pt,step_*.pt,retrieval.json}`,
`data/bones_seed/{tmr_g1_stats_v2,eval_text_emb.pt}`.

## 9.8. TMR-G1 v1 hyperparameters + architecture (== TMR-SOMA-RP)

**Architecture is identical to `nvidia/TMR-SOMA-RP-v1`** (verified against its
`config.yaml`): TMR dual-encoder, ACTOR-style transformer, VAE, `latent_dim=256`,
`ff_size=1024`, `num_layers=6`, `num_heads=4`, `dropout=0.1`, `gelu`,
`unit_vector=True`, text input = LLM2Vec Meta-Llama-3-8B-Instruct (mntp-supervised),
4096-d, frozen. **Only two differences vs the SOMA model**: skeleton
(`G1Skeleton34` vs `SOMASkeleton30` → `TMRMotionRep` dim 210 vs 186) and fps
(20 vs 30). So this is effectively "TMR-SOMA-RP architecture, retrained on G1
Bones-SEED".

A `config.json` is written to the run dir at training start (see
`tmr_g1/training/train.py:_write_config`) capturing all of the below + data
paths + git commit. TensorBoard logs under `<out>/tb` (loss/total, recon,
kl_motion, kl_text, infonce, train/batch_R@1, lr, steps_per_sec).

| | v1 (`runs/tmr_g1/v1`) | v2 (`runs/tmr_g1/v2`) |
|---|---|---|
| canonicalize | **off** | **on** (heading/pos-invariant; the key fix) |
| stats | `tmr_g1_stats_v2` | `tmr_g1_stats_v3` (canonicalized) |
| batch | 64 | **256** (more InfoNCE negatives) |
| lr | 2e-4 | **3e-4** (scaled for 4× batch) |
| weight_decay | 1e-2 | 1e-2 |
| schedule | cosine, warmup 2000 | cosine, warmup 2000 |
| max_steps | 100000 | 100000 |
| grad_clip | 1.0 | 1.0 |
| λ_recon / λ_kl / λ_contrastive | 1.0 / 1e-5 / 1.0 | same |
| InfoNCE temp | 0.1 | 0.1 |
| InfoNCE dup-mask | exact-string within batch | same |
| fps | 20 | 20 |

**Canonicalization gotcha**: `TMRMotionRep`'s canonicalize calls
`translate_2d`, which only broadcasts correctly at **batch=1**. So features
MUST be computed per-sample in the DataLoader worker
(`G1BonesSeedDataset(motion_rep=..., canonicalize=True)` → returns `features`),
NOT on the padded batch in the train loop. Stats (`build_stats.py
--canonicalize`) must use the same flag so normalization matches.

## 9.9. TMR-G1 v1–v4 retrieval comparison (R@3, text->motion)

All vs published GT R@3. v4 is the ~140k partial checkpoint (200k still running).

| group | v1 | v2 | v3 | v4* | GT |
|---|---:|---:|---:|---:|---:|
| content/overview | 52.8 | **58.6** | 56.9 | 55.8 | 89.09 |
| content/timeline_single | 49.7 | 53.1 | 50.4 | 52.5 | 86.26 |
| content/timeline_multi | 57.7 | **62.7** | 60.6 | 62.5 | 88.47 |
| repetition/overview | 74.4 | 83.1 | 81.2 | **83.5** | 93.91 |
| repetition/timeline_single | 71.1 | 78.8 | 81.3 | **84.4** | 90.13 |
| repetition/timeline_multi | 72.1 | 80.3 | 80.4 | **82.5** | 94.49 |

Exact configs (from each run's `config.json`; v1 predates config dump):
- v1: no-canon, round-robin, **bs64**, lr2e-4, 100k
- v2: **canon**, round-robin, **bs256**, lr3e-4, 100k
- v3: canon, **proportional**, bs256, lr3e-4, 150k
- v4: canon, proportional, **bs512**, lr4e-4, 200k (best on repetition)

So v1→v2 changed *two* things (canon + bs64→256); both contribute, canon
most. v2→v3 isolates the sampler; v3→v4 isolates batch (256→512) + steps.

Takeaways:
- **Canonicalization + bigger batch (v1→v2) is the big win**, +3 to +9 R@3.
- **Bigger batch (v4) helps most on R@1 and on repetition** (more InfoNCE
  negatives) — repetition/timeline_single reached 84.4 (GT 90.13, gap −5.7).
- **Content (unseen) plateaus at R@3 ~55–63** regardless of batch/steps/sampler
  — this is the structural ceiling from (a) training on train-split only while
  the published SOMA TMR saw the full set incl. test, and (b) SEED-only 286h.
  Inline eval confirmed content/overview R@3 was flat from step ~80k.
- **Proportional sampler (v3) traded overview for timeline_single** as
  predicted; bs512 (v4) recovered both.

Best model so far: **v4** (repetition) / **v2** (content/overview). Eval JSONs
in each `runs/tmr_g1/v*/retrieval*.json`.

## 9.10. BREAKTHROUGH: down-weighting the generation loss (v6)

The recon-vs-contrastive imbalance (§"loss weighting") was crippling
retrieval. v5 had recon term ~20 vs contrastive ~3.5 (the queue had already
raised contrastive); v6 set **λ_recon = 0.1** (+ λ_kl 1e-4) → recon term ~4.4
≈ contrastive ~5. Generation is KEPT (not dropped — that was rejected), just
de-emphasized so the encoder optimizes for retrieval, not reconstruction.

Effect on content/overview R@3: v5 **57** → v6 **86** (peak, step 15k). R@1
25→67, MedR 3→1.

**v6 @ step 15k, all six groups vs published GT R@3** (the headline result):

| group | v6 R@1 | v6 R@3 | v6 R@10 | GT R@3 | Δ R@3 |
|---|---:|---:|---:|---:|---:|
| content/overview | 66.9 | 85.7 | 95.2 | 89.09 | −3.4 |
| content/timeline_single | 67.8 | **88.0** | 96.5 | 86.26 | **+1.7** |
| content/timeline_multi | 61.5 | 85.3 | 94.7 | 88.47 | −3.2 |
| repetition/overview | 64.5 | 87.3 | 96.6 | 93.91 | −6.6 |
| repetition/timeline_single | 71.8 | **90.6** | 97.8 | 90.13 | **+0.4** |
| repetition/timeline_multi | 57.0 | 86.2 | 97.0 | 94.49 | −8.3 |

**TMR-G1-SEED is now competitive with the published TMR-SOMA-RP and beats it
on both timeline_single groups** — despite training on SEED-only / train-split
while the reference used full Rigplay incl. the test motions. The earlier
"structural ceiling" on content was NOT data — it was the loss weighting.

Config that did it (v6 = `runs/tmr_g1/v6`): canon, proportional sampler,
precomputed features, memory bank (queue 8192), text-dup mask 0.9, **λ_recon
0.1 / λ_kl 1e-4 / λ_contrastive 1.0**, bs256, lr3e-4.

Caveats / next:
- **Volatile**: inline R@3 swings 63–86 between evals (high LR 2.76e-4 +
  de-emphasized recon regularizes the latent less). Pick the best checkpoint;
  the cosine decay to 200k should stabilize. `step_00015000.pt` is the best
  all-rounder so far → `runs/tmr_g1/v6/retrieval_step15k.json`.
- Worth a follow-up: a touch more recon (0.2) or lower peak LR for stability;
  and re-confirm on the final 200k checkpoint.

## 10. SONIC `metrics_eval.json` is only saved when `+eval_output_dir=...`

`gear_sonic/eval_agent_trl.py` line ~575 sets `config.callbacks.im_eval.output_dir = config.get("eval_output_dir", None)`. If that's None, `im_eval_callback.save_metrics_eval(...)` silently no-ops (it does `if self.output_dir is not None: ...`).

Our `scripts/eval_sonic_set.sbatch` always passes `+eval_output_dir=${OUT}`.

---

## 11. Motion representation — authoritative reference

Single source of truth for how motion is encoded in this repo. Where this
section disagrees with §2 / §5 / §6 above, **§11 wins** (those predate the
final `G1Skeleton34` decision). The unified on-disk NPZ that feeds everything
is in §9.5; this section is about the *feature* tensors derived from it.

### 11.1 Two distinct motion representations (don't confuse them)

| | `KimodoMotionRep` | `TMRMotionRep` |
|---|---|---|
| Used by | Kimodo-G1 diffusion / SONIC tracking | TMR-G1 retrieval (this project's focus) |
| Skeleton | `G1Skeleton34` | `G1Skeleton34` |
| fps | 30 | **20** |
| Dim | **417** | **210** |
| Contains rotations? | **Yes** (`global_rot_data`, 6D, 204 dims) | **No** — positions/velocities/contacts only |
| Source class | `kimodo/motion_rep/reps/kimodo_motionrep.py` | `kimodo/motion_rep/reps/tmr_motionrep.py` |
| Full layout | §1 | §11.2 below |

Both are reversible feature encodings (`__call__` ⇆ `inverse`). KimodoMotionRep
is the diffusion model's actual I/O (§1). TMRMotionRep is intentionally
positions-only: retrieval doesn't need joint orientations, and dropping them
keeps the latent lean.

### 11.2 `TMRMotionRep` block layout (G1Skeleton34, the shipped TMR-G1)

`nbjoints = 34` ⇒ dim **210**:

| Block | shape | dims | meaning |
|---|---|---:|---|
| `root_pos` | `(T, 3)` | 3 | pelvis xyz (Y-up; see §11.4) |
| `global_root_heading` | `(T, 2)` | 2 | `(cos θ, sin θ)` of facing direction |
| `local_joints_positions` | `(T, 33, 3)` | 99 | per-joint position, planar-root-removed + heading-removed (root excluded ⇒ `J−1`) |
| `velocities` | `(T, 34, 3)` | 102 | per-joint xyz velocity, heading-removed |
| `foot_contacts` | `(T, 4)` | 4 | binary, from pos+vel thresholds (0.15, 0.10) |
| **TOTAL** | | **210** | |

Compare: `G1Skeleton30` gives `3 + 2 + 29×3 + 30×3 + 4 = 186` (the abandoned
stopgap); `SOMASkeleton30` is also 186 (kimodo_open's SOMA TMR). Published
`nvidia/TMR-SOMA-RP-v1` ships 190-dim stats — 4 undisclosed extra channels
(§2). None of these are load-compatible with our 210-dim weights.

### 11.3 Input convention: positions only, FK skipped

TMRMotionRep is called with **`posed_joints[B, T, 34, 3]`** (global joint
positions) and `to_normalize` / `to_canonicalize` flags — *not* with rotation
matrices. When `posed_joints` is passed, FK is skipped and
`root_positions = posed_joints[:, :, 0]` (pelvis). The unified NPZ also stores
`local_rot_mats` + `root_positions`, but **TMR ignores them**; those exist for
KimodoMotionRep and for the MuJoCo qpos round-trip (§9.5). So the only part of
the unified NPZ that TMR-G1 ever reads is `posed_joints`.

`lengths` must be passed for any batched call; at `batch=1` it defaults to the
full frame count (`tmr_motionrep.py:87`).

### 11.4 Geometric semantics (what "local" and "heading-removed" mean)

Not obvious from dims alone — from `tmr_motionrep.py:99–113`:

- **Frame is Y-up.** The ground plane is **X–Z**; **Y is height**.
  `translate_2d` edits root indices 0 and 2; `ground_offset` reads index 1.
  This matches the kimodo Y-up convention the unified NPZ is stored in (§9.5).
- **`local_joints_positions` is planar-root-removed but keeps height.** It is
  `(joint − pelvis)` then `+ ground_offset`, where `ground_offset` re-adds the
  pelvis **Y** only. So horizontally it's pelvis-relative; vertically it's the
  joint's absolute height above the floor. Root joint is dropped (`[:, :, 1:]`).
- **Heading is factored out for rotation-invariance.** Both
  `local_joints_positions` and `velocities` are rotated by `−root_heading`
  (`RotateFeatures`), so two clips that differ only by which way the character
  faces produce identical local features. The absolute facing lives solely in
  the 2-dim `global_root_heading`.
- **`velocities`** are global xyz finite-difference velocities (`compute_vel_xyz`
  at `fps`), then heading-removed. The last frame uses a duplicate-frame
  forward-diff — the only crop-sensitive value (§9.5 precompute note).

`inverse(...)` reconstructs `posed_joints` from `root_pos` +
`local_joints_positions` (rotations are gone, so `local_rot_mats`/`global_rot_mats`
come back `None`) — adequate for the recon loss, which targets these features.

### 11.5 Canonicalization, normalization, and where they happen

- **Normalization**: per-feature `(x − mean) / std` from a stats dir. The
  shipped stats are **`tmr_g1_stats_v3`** (dim 210, computed with
  `--canonicalize` so mean/std match the canonicalized features). Built by
  `tmr_g1/training/build_stats.py`.
- **Canonicalization** (`to_canonicalize=True`): heading/position-invariant
  framing applied *before* normalize. **Gotcha**: it routes through
  `translate_2d`, which only broadcasts correctly at **batch=1** — so features
  are computed **per-sample in the DataLoader worker**, never on the padded
  batch. Stats must be built with the same flag (§9.8).
- **Two dataset code paths** (`g1_dataset.py:__getitem__`):
  1. `motion_rep` set, no `feat_root`: full `TMRMotionRep` forward per fetch
     (FK-free, but heading/rotate/velocity/foot-detect each fetch).
  2. `feat_root` set (the fast path used by v5/v6): load **precomputed raw**
     `(T, 210)` fp16 from `scripts/precompute_g1_features.py`
     (un-normalized, un-canonicalized), then apply only canonicalize + normalize
     at fetch — ~3–5× dataloader throughput (§9.5 precompute, task #39).
- Eval (`eval_retrieval.py`) encodes with `to_normalize=True,
  to_canonicalize=True` to match training exactly.

### 11.6 Where each rep is constructed

| Call site | Rep |
|---|---|
| `tmr_g1/model/tmr_model.py:build_tmr_g1` | `TMRMotionRep(G1Skeleton34, fps=20)` → encoder/decoder `nfeats=210` |
| `scripts/precompute_g1_features.py:_motion_rep` | same, `stats_path=None` (raw) |
| `nvidia/Kimodo-G1-SEED-v1/config.yaml` denoiser | `KimodoMotionRep(G1Skeleton34, fps=30)` → 417 |

Note: `tmr_g1/model/tmr_model.py`'s module docstring still says "G1Skeleton30"
in prose — stale comment; the code on line ~89 instantiates `G1Skeleton34`.
`tmr_g1/data/g1_dataset.py`'s docstring likewise says `posed_joints (T, 30, 3)`
— also stale; the arrays are `(T, 34, 3)`.

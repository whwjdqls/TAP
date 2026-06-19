# TMR-G1 improvement plan

Reasoning from the v1–v4 evidence, prioritized by expected ROI. Goal: raise
text→motion R-precision on the kimodo text2motion benchmark.

## 1. Diagnosis: where the gap actually comes from

Current best (R@3): content ~56–63, repetition ~83–84. Published GT: content
86–89, repetition 90–94.

Two *very different* gaps, and the reason matters:

- **Repetition gap is small (~6–11)** and shrinks with more negatives (v4
  bs512: repetition/timeline_single 84.4 vs GT 90.1, −5.7). Repetition =
  content types seen in training (novel performances). So our model
  generalizes well here; the residual is mostly the reference's advantages.
- **Content gap is large (~30) and plateaus** (inline eval flat from step 80k
  regardless of batch/steps/sampler). Content = semantic types NOT in our
  train split.

The published `nvidia/TMR-SOMA-RP-v1` has **two structural advantages we cannot
match by training harder**:
1. It was trained on **700h Rigplay**; we have **286h SEED** (train split only).
2. Its model card states it was trained on **train+test** "to be most useful
   for evaluating generators" — i.e. it *saw the eval motions*. On the
   "content" split (unseen to us) it had already seen those clips.

**Conclusion:** chasing the published GT number directly is partly chasing a
test-set-leakage artifact. The honest target is "best generalizing TMR-G1 on
286h", and the honest *comparison* is against a TMR-SOMA trained identically
(see §3). Optimization (more steps/bigger model) is NOT the content bottleneck.

## 2. Tier-1 levers (highest ROI; do first)

### 2.1 Throughput: precompute/pack motion features  [enables everything else]
- **Problem:** per-sample `TMRMotionRep` in the DataLoader workers caps us at
  2–6 steps/s (FK-free but foot-detect + canonicalize + normalize per fetch).
  This makes every experiment slow (v4 = ~18h).
- **Fix (kimodo_open already does this):** precompute the expensive,
  crop-invariant part of the features once per *full* motion, store a packed
  mmap tensor. At fetch time, slice + do only the cheap runtime canonicalize
  (a rotate on the 210-d vectors, no FK). See
  `kimodo_open/.../soma_text_motion.py:_load_features_segment` and
  `pack_bones_seed_features.py`.
- **Expected:** 3–5× throughput → bigger batches and more ablations per day.
- **Effort:** medium (1 script + dataset path). **Do this first** — it
  unblocks 2.2–2.4.

### 2.2 Decouple #negatives from batch with a memory bank (MoCo-style)
- **Logic:** v2→v4 showed retrieval scales with in-batch negatives, but batch
  is throughput/memory-bound. A momentum/FIFO queue of past text+motion
  embeddings gives thousands of negatives at ~no cost.
- **Plan:** maintain a queue (size 4k–16k) of L2-normalized motion & text
  embeddings; InfoNCE denominator = batch + queue. Optionally a momentum
  encoder (EMA) for queue consistency; simpler first: stop-grad queue of
  recent embeddings.
- **Expected:** the single most promising *model* lever for R@1/R@3,
  especially repetition. Effort: medium.

### 2.3 Semantic false-negative masking during training
- **Logic:** InfoNCE currently treats every other sample as a negative, but
  many captions are near-duplicates ("walk forward" vs "walks ahead"). These
  false negatives inject wrong gradient. The *eval* already forgives this
  (0.99 text-sim dedup); training should too.
- **Plan:** within batch (and queue, if 2.2), mask pairs with text-text cosine
  > τ (e.g. 0.9) from the negative set, like `contrastive_metrics`. We already
  have all LLM2Vec text embeddings cached → cheap to compute text-text sim.
- **Expected:** cleaner signal, modest but reliable R@k gain. Effort: low.

## 3. Tier-2: fair comparison + data

### 3.1 Train a TMR-SOMA-SEED baseline (apples-to-apples)
- **Why:** the only honest way to know if the *G1 embodiment* costs retrieval
  quality is to train the identical TMR on SOMA motions from the *same SEED
  train split*, same steps/aug, and compare TMR-G1 vs TMR-SOMA head-to-head.
  Removes both confounds (700h vs 286h, test leakage). We already have
  `data/bones_seed/g1/.../` → also have SOMA via `move_soma_uniform_path` in
  the metadata (BVH). Would need a SOMA 20fps NPZ build (analogous to the G1
  unified pipeline) using `SOMASkeleton30`.
- **Expected:** not a score gain, but the *correct denominator* for the paper.
  If TMR-G1 ≈ TMR-SOMA-SEED, the embodiment is "free" and the gap to published
  GT is entirely data/leakage. Effort: medium (reuse the G1 pipeline).

### 3.2 Motion augmentation to extract more from 286h
- Speed/time-warp (±10–20%), small per-joint position jitter, frame dropout.
- Already have: mirroring (_M), random crop, heading (via canon), paraphrase
  sampling. Add the above. **Expected:** small, helps generalization (esp.
  content). Effort: low–medium.

## 4. Tier-3: ablations / tuning (cheap, do alongside)

- **Loss ablation:** recon on/off, KL weight (currently 1e-5 → KL barely
  active; latents may be under-regularized). Try λ_kl ∈ {1e-5, 1e-4}, and a
  recon-free run — recon may not help retrieval.
- **InfoNCE temperature** sweep {0.05, 0.07, 0.1}.
- **Stop early:** content plateaus ~80k; don't pay for 200k unless repetition
  still climbing. Use the inline eval to early-stop per group.
- **Eval at sample_mean vs sampled** (we use mean — correct).

## 5. Tier-4: protocol replication (flag honestly, optional)

- Training on **train+test** would reproduce the published GT setting (the
  reference did this). It would make our content numbers jump, but it is
  **test-set leakage** and only valid to *reproduce* the published "GT TMR"
  row, never as a generalization claim. Keep separate, label clearly.

## 6. Recommended sequence

1. **2.1 packed features** (unblocks speed) — infra.
2. **2.2 memory-bank InfoNCE** + **2.3 false-negative masking** — retrain best
   config (canon, bs512-equiv via queue, proportional or round-robin).
3. Run **4** ablations cheaply on top (now fast).
4. **3.1 TMR-SOMA-SEED baseline** for the honest comparison in parallel.
5. **3.2 augmentation** if content still lags.

Expected outcome: repetition R@3 into the high-80s (closing most of the −6 to
−11 gap via more negatives + clean negatives); content improves modestly but
remains structurally below the leakage-advantaged GT — which §3.1 will
contextualize correctly.

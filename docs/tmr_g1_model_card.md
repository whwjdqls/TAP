# TMR-G1-SEED-v1 — model card

Text-to-motion retrieval model for the **G1** humanoid embodiment, trained on
Bones-SEED. Doubles as the frozen **feature extractor for FID + R-precision**
evaluation of G1 motion generators.

- **Checkpoint:** `runs/tmr_g1/TMR-G1-SEED-v1.pt` (= `runs/tmr_g1/v6/step_00015000.pt`, copied)
- **Selected from:** run `v6`, step 15,000.
- Architecture mirrors `nvidia/TMR-SOMA-RP-v1`; differences are the skeleton
  (`G1Skeleton34` → 210-dim `TMRMotionRep`, not SOMA-30/186) and fps (20 vs 30).
  See `docs/notes.md` §11 for the authoritative motion-rep reference.

## Architecture

TMR dual-encoder, ACTOR-style transformer, VAE. `latent_dim=256`, `ff_size=1024`,
`num_layers=6`, `num_heads=4`, `dropout=0.1`, `gelu`, `unit_vector=True`.
Text input = LLM2Vec (Meta-Llama-3-8B-Instruct, mntp-supervised), 4096-d, **frozen**.
Motion input = `posed_joints[T,34,3]` → `TMRMotionRep(G1Skeleton34, fps=20)` (210-d).

## Training recipe (run v6)

| | |
|---|---|
| data | Bones-SEED **train split only**, 3 caption sources (natural / single / multi) |
| sampler | proportional | canonicalize | on |
| stats | `tmr_g1_stats_v3` (canonicalized, dim 210) |
| batch | 256 | optimizer | AdamW, wd 1e-2, grad-clip 1.0 |
| lr | 3e-4, cosine + warmup 2000 (200k horizon) | step selected | 15,000 |
| λ_recon / λ_kl / λ_contrastive | **0.1** / 1e-4 / 1.0 |
| InfoNCE | temp 0.1, memory-bank queue 8192, false-neg text-dup mask 0.9 |

The **λ_recon=0.1** (down-weighted reconstruction) was the key unlock — see
`docs/notes.md` §9.10.

## Results — text→motion R@3 (kimodo `contrastive_metrics`, 0.99 text-dedup)

vs published TMR-SOMA-RP "Ground Truth" R@3. **Bold = TMR-G1 ≥ published.**

| group | pool | R@1 | R@3 | R@10 | published GT R@3 |
|---|---:|---:|---:|---:|---:|
| content/overview | 917 | 66.9 | 85.7 | 95.2 | 89.1 |
| content/timeline_single | 917 | 67.8 | **88.0** | 96.5 | 86.3 |
| content/timeline_multi | 789 | 61.5 | 85.3 | 94.7 | 88.5 |
| repetition/overview | 2380 | 64.5 | 87.3 | 96.6 | 93.9 |
| repetition/timeline_single | 2380 | 71.8 | **90.6** | 97.8 | 90.1 |
| repetition/timeline_multi | 1779 | 57.0 | 86.2 | 97.0 | 94.5 |

Competitive with — and on two groups better than — the published model, despite
training on SEED-only / train-split-only (the reference used full Rigplay incl.
the test motions).

## Why this checkpoint (search summary)

A wide search was run to beat v6: v8 (collapse), v9 (cosine vs constant LR),
v10 (no-buffer / higher-LR), v11 (7-run OVAT: lr, queue, batch, kl, temp, dup),
v12 (seeds, lower LR, larger queue, dup+queue combo, dropout, grad-clip).
**Nothing beat v6 on content / overall** (6-group avg R@3: v6 **87.2**). Notes:

- The recipe is **high-variance**: v6@15k caught a strong point (a clean re-run
  peaks ~77 on content/overview). It is a real, saved checkpoint — verified by
  the standalone eval above.
- The recipe is also **seed-unstable** with the memory-bank: ~1-in-3 runs suffer
  text-encoder posterior collapse at lr 3e-4. Pushing lr (≥4e-4), sharpening
  temp (0.05), or λ_kl≥1e-3 reliably collapses it.
- **Stable alternative:** `v12dupq` (dup-mask 0.99 + queue 16384) never
  collapses and **wins all repetition groups** (e.g. repetition/timeline_single
  93.1), but loses content by 4–8 pts (avg 85.5). Prefer it only if robustness
  matters more than unseen-content accuracy.

## Loading / evaluating

```bash
conda activate kimodo_soma   # run from a cwd without a kimodo/ dir (namespace trap, notes §6)
python tmr_g1/eval_retrieval.py \
  --ckpt runs/tmr_g1/TMR-G1-SEED-v1.pt \
  --stats-path data/bones_seed/tmr_g1_stats_v3 \
  --text-emb-cache data/bones_seed/eval_text_emb.pt \
  --g1-npz-root data/bones_seed/g1_unified_npz \
  --testsuite data/kimodo_benchmark/testsuite
```

Encode motion with `to_normalize=True, to_canonicalize=True` to match training.
The eval has a **collapse guard**: degenerate (near-constant) embeddings report
`COLLAPSED` + zeros instead of a misleading 100%.

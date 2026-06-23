"""VAE-G1: a motion VAE trained on the G1 embodiment from Bones-SEED.

Architecturally a sibling of `tmr_g1` with the text branch removed: an
ACTOR-style transformer motion encoder + decoder, trained with reconstruction
+ KL only (no contrastive loss, no LLM2Vec). Same `TMRMotionRep(G1Skeleton34)`
210-d feature space and the same dataloader, so its latent is directly
comparable to (and swappable with) the TMR-G1 motion encoder.
"""

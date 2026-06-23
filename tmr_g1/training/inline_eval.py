"""
In-training retrieval evaluator for a single testsuite group (default
content/text2motion/overview). Precomputes motion features + text LLM
embeddings ONCE at construction, then each `evaluate(tmr)` only re-runs the
(cheap) motion/text encoders + metric — a few seconds, so it's safe to call
periodically during training.

Reports text->motion R@1/2/3/5/10 + MedR via kimodo's `contrastive_metrics`
(same protocol as the standalone eval and the published GT numbers).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kimodo.metrics.tmr import compute_tmr_retrieval_metrics


class GroupEvaluator:
    def __init__(self, testsuite, g1_npz_root, text_cache_path, motion_rep,
                 split="content", grp="overview", fps=20, canonicalize=True,
                 device="cuda", max_cases=0):
        self.device = device

        blob = torch.load(text_cache_path, map_location="cpu", weights_only=False)
        cap2feat = {c: blob["features"][i] for i, c in enumerate(blob["captions"])}

        group_dir = Path(testsuite) / split / "text2motion" / grp
        scale = fps / 30.0  # crop indices are 30 fps; NPZs are `fps`
        feats_list, llm_list = [], []
        miss = 0
        for cdir in sorted(glob.glob(str(group_dir / "*"))):
            cdir = Path(cdir)
            mp, sp = cdir / "meta.json", cdir / "seed_motion.json"
            if not (mp.is_file() and sp.is_file()):
                continue
            meta = json.load(open(mp))
            seed = json.load(open(sp))
            text = (meta.get("text") or "").strip()
            if not text or text not in cap2feat:
                miss += 1
                continue
            bvh = seed.get("bvh_path", "")
            session = bvh.split("/")[1] if bvh else ""
            npz = Path(g1_npz_root) / session / f"{seed['move_name']}.npz"
            if not npz.is_file():
                miss += 1
                continue
            try:
                with np.load(npz, allow_pickle=False) as d:
                    posed = np.asarray(d["posed_joints"]).astype(np.float32)
            except Exception:  # noqa: BLE001
                miss += 1
                continue
            n = posed.shape[0]
            a = max(0, int(round(seed["crop_start_frame_index"] * scale)))
            b = min(n, int(round(seed["crop_end_frame_index"] * scale)))
            if b - a < 4:
                miss += 1
                continue
            clip = posed[a:b]
            with torch.no_grad():
                f = motion_rep(
                    posed_joints=torch.from_numpy(clip).unsqueeze(0),
                    to_normalize=True, to_canonicalize=canonicalize,
                    lengths=torch.tensor([clip.shape[0]]),
                )[0]  # (T, D)
            feats_list.append(f.float())
            llm_list.append(cap2feat[text].float())
            if max_cases and len(feats_list) >= max_cases:
                break

        self.n = len(feats_list)
        self.miss = miss
        # Pre-pad motion features to (N, T_max, D) + mask.
        T_max = max(f.shape[0] for f in feats_list)
        D = feats_list[0].shape[1]
        self.feats = torch.zeros(self.n, T_max, D)
        self.mask = torch.zeros(self.n, T_max, dtype=torch.bool)
        for i, f in enumerate(feats_list):
            self.feats[i, : f.shape[0]] = f
            self.mask[i, : f.shape[0]] = True
        # llm: (N, 1, 4096)
        self.llm = torch.stack(llm_list, dim=0)
        if self.llm.dim() == 2:
            self.llm = self.llm.unsqueeze(1)

    @torch.no_grad()
    def evaluate(self, tmr, chunk=256):
        was_training = tmr.training
        tmr.eval()
        dev = self.device
        m_emb, t_emb = [], []
        for s in range(0, self.n, chunk):
            f = self.feats[s:s + chunk].to(dev)
            msk = self.mask[s:s + chunk].to(dev)
            out = tmr.motion_encoder({"x": f, "mask": msk})
            mu = out.unbind(1)[0]
            m_emb.append(F.normalize(mu, dim=-1).cpu())

            lf = self.llm[s:s + chunk].to(dev)
            lmsk = torch.ones(lf.shape[0], 1, dtype=torch.bool, device=dev)
            tout = tmr.text_encoder({"x": lf, "mask": lmsk})
            tmu = tout.unbind(1)[0]
            t_emb.append(F.normalize(tmu, dim=-1).cpu())

        if was_training:
            tmr.train()
        motion_emb = torch.cat(m_emb).numpy()
        text_emb = torch.cat(t_emb).numpy()

        # Collapse guard: if the encoders degenerate (posterior collapse ->
        # near-constant embeddings), the similarity matrix is constant and
        # compute_tmr_retrieval_metrics returns a *fake* 100% (nothing strictly
        # beats the diagonal). Detect it and report 0 instead of being fooled.
        m_std = float(np.linalg.norm(motion_emb.std(0)))
        t_std = float(np.linalg.norm(text_emb.std(0)))
        if min(m_std, t_std) < 1e-4 or not np.isfinite([m_std, t_std]).all():
            return {"COLLAPSED": 1.0, "motion_emb_std": m_std, "text_emb_std": t_std,
                    "R@1": 0.0, "R@2": 0.0, "R@3": 0.0, "R@5": 0.0, "R@10": 0.0}

        metrics = compute_tmr_retrieval_metrics(motion_emb, text_emb)
        return {k.replace("TMR/t2m_R/", ""): v
                for k, v in metrics.items() if k.startswith("TMR/t2m_R/")}

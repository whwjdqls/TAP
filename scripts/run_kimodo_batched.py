"""
Batched Kimodo-G1 generation: load the model once and feed it batches of N
prompts so each diffusion call processes N motions in parallel.

Emits a Bones-SEED-format CSV per case (units cm/deg, with header) ready for
SONIC's `convert_soma_csv_to_motion_lib.py`.

Layout produced (single session "kimodo"):
    <out-csvs>/kimodo/<case_id_slug>.csv

Usage:
    python scripts/run_kimodo_batched.py \
        --subset runs/eval_subset/t2m_overview_full.jsonl \
        --out-csvs runs/eval_kimodo/t2m_overview_csvs \
        --batch 16 --diffusion-steps 50 \
        --model Kimodo-G1-SEED-v1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Bones-SEED CSV joint order (must match SONIC converter).
JOINT_NAMES = [
    "left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof",
    "left_knee_joint_dof", "left_ankle_pitch_joint_dof", "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof",
    "right_knee_joint_dof", "right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof", "waist_roll_joint_dof", "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof", "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof", "left_wrist_pitch_joint_dof", "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof", "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof", "right_wrist_pitch_joint_dof",
    "right_wrist_yaw_joint_dof",
]
HEADER = (
    ["Frame", "root_translateX", "root_translateY", "root_translateZ",
     "root_rotateX", "root_rotateY", "root_rotateZ"]
    + JOINT_NAMES
)
assert len(JOINT_NAMES) == 29


def slugify(case_id: str) -> str:
    return case_id.replace("/", "__")


def qpos_to_bones_seed_csv(qpos: np.ndarray, out_path: Path):
    """qpos: (T, 36) MuJoCo qpos = [root_pos(m), root_quat(wxyz), 29 joints(rad)]."""
    T = qpos.shape[0]
    assert qpos.shape[1] == 36, f"got {qpos.shape}"
    root_pos_cm = qpos[:, 0:3] * 100.0
    root_quat_wxyz = qpos[:, 3:7]
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    euler_deg = Rotation.from_quat(root_quat_xyzw).as_euler("xyz", degrees=True)
    joint_deg = np.rad2deg(qpos[:, 7:36])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for i in range(T):
            row = [i]
            row.extend(root_pos_cm[i].tolist())
            row.extend(euler_deg[i].tolist())
            row.extend(joint_deg[i].tolist())
            w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--out-csvs", required=True)
    p.add_argument("--model", default="Kimodo-G1-SEED-v1")
    p.add_argument("--diffusion-steps", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--text-encoder-device", default="cpu")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--emb-cache", default="",
                   help="path to a .pt cache from build_text_emb_cache.py; "
                        "if set, text_encoder is swapped for a cached lookup "
                        "and Llama-3 is never loaded.")
    args = p.parse_args()

    os.environ.setdefault("TEXT_ENCODER_DEVICE", args.text_encoder_device)

    rows = [json.loads(l) for l in open(args.subset) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[batched] {len(rows)} cases  batch={args.batch}  model={args.model}")

    out_dir = Path(args.out_csvs) / "kimodo"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing:
        before = len(rows)
        rows = [r for r in rows if not (out_dir / f"{slugify(r['case_id'])}.csv").exists()]
        print(f"[batched] skip-existing: {before - len(rows)} already done, {len(rows)} left")

    # Lazy import after env vars set.
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.model.load_model import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.emb_cache:
        # Build model WITHOUT instantiating Llama by swapping the text-encoder
        # entry in the model's hydra config BEFORE load_model resolves it.
        # The cached encoder mimics the interface: __call__(list[str]) ->
        # (feat[B, 1, llm_dim] tensor, lengths list).
        cache_blob = torch.load(args.emb_cache, map_location="cpu")
        emb_table = cache_blob["embeddings"]
        llm_dim = cache_blob.get("llm_dim")
        print(f"[batched] emb cache: {len(emb_table)} entries, llm_dim={llm_dim}")

        class CachedTextEncoder:
            def __init__(self, table, device, llm_dim):
                self.table = table
                self._device = device
                self.llm_dim = llm_dim

            def __call__(self, text):
                is_str = isinstance(text, str)
                texts = [text] if is_str else list(text)
                feats = []
                for t in texts:
                    if t not in self.table:
                        raise KeyError(f"text not in cache: {t!r}")
                    feats.append(self.table[t])
                feat = torch.stack(feats, dim=0).to(self._device)  # (B, 1, llm_dim)
                lengths = [1] * len(texts)
                if is_str:
                    feat = feat[0]
                    lengths = lengths[0]
                return feat, lengths

            def to(self, *a, **kw):
                return self
            def eval(self):
                return self

        cached_encoder = CachedTextEncoder(emb_table, device, llm_dim)
    else:
        cached_encoder = None

    t0 = time.time()
    model, resolved_model = load_model(
        args.model, device=device, default_family="Kimodo", return_resolved_name=True,
    )
    if cached_encoder is not None:
        # Replace the freshly-loaded text encoder with our cache lookup. The
        # base Llama-3 weights are still in VRAM at this point — free them.
        try:
            del model.text_encoder
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        model.text_encoder = cached_encoder
        print("[batched] swapped text_encoder for cache lookup")
    converter = MujocoQposConverter(model.skeleton)
    print(f"[batched] model loaded in {time.time()-t0:.1f}s on {device}  fps={model.fps}")

    n_ok, n_fail = 0, 0
    t_gen = 0.0
    for batch_start in range(0, len(rows), args.batch):
        batch = rows[batch_start : batch_start + args.batch]
        texts = [r["text"] for r in batch]
        # cap duration in case some are 0
        num_frames = [max(15, int(r["duration"] * model.fps)) for r in batch]
        # Use a deterministic per-batch seed from the first case's seed (kimodo
        # `seed_everything` is global; for batched diffusion we just want
        # reproducibility, not per-case independence — these are research runs).
        try:
            from kimodo.tools import seed_everything
            seed_everything(batch[0].get("seed", 0))
        except Exception:  # noqa: BLE001
            pass

        tt = time.time()
        try:
            with torch.no_grad():
                output = model(
                    texts,
                    num_frames,
                    num_denoising_steps=args.diffusion_steps,
                    multi_prompt=False,
                    return_numpy=True,
                )
        except Exception as e:  # noqa: BLE001
            print(f"[batched] batch {batch_start//args.batch} FAILED: {type(e).__name__}: {e}")
            n_fail += len(batch)
            continue
        t_gen += time.time() - tt

        # Convert each item in the batch to qpos and save.
        qpos_batch = converter.dict_to_qpos(output, device)  # (B, T_max, 36) tensor / list / ndarray

        def _to_np(x):
            if hasattr(x, "detach"):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        if isinstance(qpos_batch, list):
            qposes = [_to_np(q) for q in qpos_batch]
        else:
            arr = _to_np(qpos_batch)
            qposes = [arr[i] for i in range(len(batch))]

        for r, qpos, T in zip(batch, qposes, num_frames):
            qpos = qpos[:T]  # trim to per-case duration in case of padding
            out_path = out_dir / f"{slugify(r['case_id'])}.csv"
            qpos_to_bones_seed_csv(qpos, out_path)
            n_ok += 1

        elapsed = time.time() - t0
        done = n_ok + n_fail
        rate = done / max(1e-6, elapsed)
        eta = (len(rows) - done) / max(1e-6, rate)
        print(f"[batched] batch {batch_start//args.batch + 1}/"
              f"{(len(rows)+args.batch-1)//args.batch}  "
              f"done={done}/{len(rows)}  ok={n_ok} fail={n_fail}  "
              f"gen_avg={t_gen/max(1,done):.2f}s  eta={eta/60:.1f}min")

    print(f"[batched] FINAL ok={n_ok} fail={n_fail}  total_elapsed={(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

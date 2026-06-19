"""
For each (case_id) in a subset, render a 2×2 side-by-side comparison mp4:

    top-left : kimodo reference motion (qpos replay)
    top-right: SONIC tracking the kimodo motion
    bot-left : Bones-SEED ground-truth reference motion (qpos replay)
    bot-right: SONIC tracking the GT motion

Inputs (all paths derived from --root):
    runs/eval_gt/t2m_overview_full_csvs/<session>/<move>.csv      (Bones-SEED CSV)
    runs/eval_kimodo/t2m_overview_full_csvs/kimodo/<slug>.csv     (Bones-SEED CSV)
    runs/eval_gt/t2m_overview_full_pkls/<session>/<move>.pkl      (SONIC motion_lib)
    runs/eval_kimodo/t2m_overview_full_pkls/kimodo/<slug>.pkl     (SONIC motion_lib)

For each chosen case:
  1. bones_seed_to_qpos.py on both refs → qpos CSV.
  2. render_g1_qpos.py on each qpos CSV → reference mp4.
  3. Single-motion SONIC eval --render_results=True on each motion_lib subset
     containing only that motion → tracked mp4.
  4. ffmpeg xstack 2x2 → composite mp4.

Usage:
    python scripts/render_compare.py \
        --subset runs/eval_subset/t2m_overview_full.jsonl \
        --gt-csvs   runs/eval_gt/t2m_overview_full_csvs \
        --kim-csvs  runs/eval_kimodo/t2m_overview_full_csvs \
        --gt-pkls   runs/eval_gt/t2m_overview_full_pkls \
        --kim-pkls  runs/eval_kimodo/t2m_overview_full_pkls \
        --out-dir   runs/eval_compare/videos \
        --cases     content/text2motion/overview/0000,content/text2motion/overview/0016
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path("/nfsdata/home/jungbin.cho/TAP")
SONIC_REPO = REPO / "GR00T-WholeBodyControl"
CKPT = SONIC_REPO / "sonic_release" / "last.pt"
SMPL_SAMPLE = SONIC_REPO / "sample_data" / "smpl_filtered"


def slugify(case_id: str) -> str:
    return case_id.replace("/", "__")


def find_session_move(subset_rows: list, case_id: str) -> tuple[str, str]:
    for r in subset_rows:
        if r["case_id"] == case_id:
            return r["session"], r["move_name"]
    raise KeyError(case_id)


def run(cmd):
    print(">", " ".join(map(str, cmd)))
    subprocess.check_call([str(c) for c in cmd])


def render_qpos(qpos_csv: Path, out_mp4: Path, fps: int = 30):
    run([
        sys.executable, str(REPO / "scripts" / "render_g1_qpos.py"),
        "--csv", qpos_csv, "--out", out_mp4, "--fps", str(fps),
        "--width", "480", "--height", "360",
    ])


def render_sonic_single(motion_pkl: Path, out_dir: Path, *, tag: str) -> Path:
    """Run a single-motion SONIC eval with render_results=True. Returns mp4."""
    run_dir = out_dir / f"sonic_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # SONIC expects a directory of session/<move>.pkl. We give it a tmp dir
    # holding just this one motion.
    motion_root = run_dir / "single_motion_lib" / "session"
    motion_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(motion_pkl, motion_root / motion_pkl.name)

    cmd = [
        sys.executable, str(SONIC_REPO / "gear_sonic" / "eval_agent_trl.py"),
        f"+checkpoint={CKPT}",
        "+headless=True",
        "++eval_callbacks=im_eval",
        "++run_eval_loop=False",
        "++num_envs=1",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={motion_root.parent}",
        f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file={SMPL_SAMPLE}",
        "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=1",
        "++manager_env.config.render_results=True",
        f"++manager_env.config.save_rendering_dir={run_dir}/videos",
        "++manager_env.config.env_spacing=10.0",
        "~manager_env/recorders=empty",
        "+manager_env/recorders=render",
        f"++hydra.run.dir={run_dir}",
    ]
    print(">", " ".join(cmd))
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["HYDRA_FULL_ERROR"] = "1"
    subprocess.check_call(cmd, cwd=SONIC_REPO, env=env)
    mp4 = run_dir / "videos" / "000000.mp4"
    if not mp4.exists():
        raise FileNotFoundError(mp4)
    return mp4


def composite_2x2(tl: Path, tr: Path, bl: Path, br: Path, out: Path, labels: tuple[str, str, str, str]):
    """2x2 stitch via imageio + PIL — avoids needing ffmpeg on PATH."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    out.parent.mkdir(parents=True, exist_ok=True)

    readers = [imageio.get_reader(str(p)) for p in (tl, tr, bl, br)]
    fps_list = [r.get_meta_data().get("fps", 30) for r in readers]
    fps = int(min(fps_list))

    streams = [[f for f in r] for r in readers]
    for r in readers:
        r.close()
    Ts = [len(s) for s in streams]
    T = min(Ts)
    print(f"[stitch] frames per panel: {Ts}  -> T={T} @ {fps}fps")
    H = max(s[0].shape[0] for s in streams)
    W = max(s[0].shape[1] for s in streams)

    def resize(img, h, w):
        return np.asarray(Image.fromarray(img).resize((w, h)))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8)
    for t in range(T):
        panels = []
        for stream, lab in zip(streams, labels):
            frame = resize(stream[t], H, W)
            img = Image.fromarray(frame).convert("RGB")
            d = ImageDraw.Draw(img)
            d.rectangle((4, 4, 4 + 8 + d.textlength(lab, font=font), 28), fill=(0, 0, 0))
            d.text((8, 6), lab, fill=(255, 255, 255), font=font)
            panels.append(np.asarray(img))
        top = np.concatenate([panels[0], panels[1]], axis=1)
        bot = np.concatenate([panels[2], panels[3]], axis=1)
        grid = np.concatenate([top, bot], axis=0)
        writer.append_data(grid)
    writer.close()
    print(f"[stitch] wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--gt-csvs", required=True)
    p.add_argument("--kim-csvs", required=True)
    p.add_argument("--gt-pkls", required=True)
    p.add_argument("--kim-pkls", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cases", required=True,
                   help="comma-separated case_id list, e.g. content/text2motion/overview/0000,...")
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.subset) if l.strip()]
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    # SONIC eval runs from gear_sonic cwd, so all paths we pass to it must be
    # absolute; same goes for any paths we use in subprocess wrappers.
    out_dir = Path(args.out_dir).resolve()
    args.gt_csvs = str(Path(args.gt_csvs).resolve())
    args.kim_csvs = str(Path(args.kim_csvs).resolve())
    args.gt_pkls = str(Path(args.gt_pkls).resolve())
    args.kim_pkls = str(Path(args.kim_pkls).resolve())

    for case in cases:
        session, move = find_session_move(rows, case)
        slug = slugify(case)
        case_out = out_dir / slug
        case_out.mkdir(parents=True, exist_ok=True)

        gt_bones = Path(args.gt_csvs) / session / f"{move}.csv"
        kim_bones = Path(args.kim_csvs) / "kimodo" / f"{slug}.csv"
        gt_pkl = Path(args.gt_pkls) / session / f"{move}.pkl"
        kim_pkl = Path(args.kim_pkls) / "kimodo" / f"{slug}.pkl"
        for p_ in (gt_bones, kim_bones, gt_pkl, kim_pkl):
            if not p_.exists():
                raise FileNotFoundError(p_)

        # Reference mp4s (qpos replay).
        gt_qpos = case_out / "gt_qpos.csv"
        kim_qpos = case_out / "kim_qpos.csv"
        run([sys.executable, str(REPO / "scripts" / "bones_seed_to_qpos.py"),
             "--in", gt_bones, "--out", gt_qpos])
        run([sys.executable, str(REPO / "scripts" / "bones_seed_to_qpos.py"),
             "--in", kim_bones, "--out", kim_qpos])
        gt_ref_mp4 = case_out / "gt_ref.mp4"
        kim_ref_mp4 = case_out / "kim_ref.mp4"
        render_qpos(gt_qpos, gt_ref_mp4)
        render_qpos(kim_qpos, kim_ref_mp4)

        # Tracked mp4s (SONIC eval).
        gt_track_mp4 = render_sonic_single(gt_pkl, case_out, tag="gt")
        kim_track_mp4 = render_sonic_single(kim_pkl, case_out, tag="kim")

        # 2x2 composite.
        composite_2x2(
            tl=kim_ref_mp4, tr=kim_track_mp4,
            bl=gt_ref_mp4,  br=gt_track_mp4,
            out=case_out / "compare.mp4",
            labels=("Kimodo ref", "SONIC tracks Kimodo",
                    "GT ref",     "SONIC tracks GT"),
        )
        print(f"[render_compare] -> {case_out/'compare.mp4'}")


if __name__ == "__main__":
    main()

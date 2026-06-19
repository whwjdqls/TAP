"""
Read two SONIC eval `metrics_eval.json` files (e.g. GT vs Kimodo) and emit a
side-by-side comparison covering the headline metrics from the SONIC paper:
  Succ rate, MPJPE-L/G/PA, vel_dist, accel_dist, plus subset variants.

Usage:
    python scripts/compare_metrics.py \
        --gt    runs/sonic_eval/gt_text2motion-<jobid>/metrics_eval.json \
        --pred  runs/sonic_eval/kimodo_text2motion-<jobid>/metrics_eval.json \
        --out   runs/eval_compare/text2motion.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Headline metrics from the SONIC paper (Table 4):
#   Succ      — success_rate
#   MPJPE-L   — mpjpe_l (mm, local, 14 links)
#   MPJPE-G   — mpjpe_g (mm, global)
#   MPJPE-PA  — mpjpe_pa (mm, procrustes-aligned)
#   E_vel     — vel_dist (mm/frame)
#   E_acc     — accel_dist (mm/frame^2)
HEADLINE = [
    ("Succ",     "success_rate",                     1.0),  # higher better
    ("MPJPE-L",  "mpjpe_l",                          -1),
    ("MPJPE-G",  "mpjpe_g",                          -1),
    ("MPJPE-PA", "mpjpe_pa",                         -1),
    ("E_vel",    "vel_dist",                         -1),
    ("E_acc",    "accel_dist",                       -1),
]

# Eval keys are namespaced; the callback emits both "eval/success/<m>" (averaged
# over succeeded motions) and "eval/all/<m>" (averaged over all motions). We
# prefer "all" so failures are not silently dropped.
NS_ALL = "eval/all/"
NS_OK  = "eval/success/"


def load(path):
    with open(path) as f:
        return json.load(f)


def pick(d, key):
    for ns in (NS_ALL, NS_OK, ""):
        if (ns + key) in d:
            return d[ns + key]
    return None


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, (int, float)):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--gt-label", default="GT")
    p.add_argument("--pred-label", default="Kimodo")
    args = p.parse_args()

    gt = load(args.gt)
    pred = load(args.pred)

    lines = []
    lines.append(f"# SONIC eval comparison: {args.gt_label} vs {args.pred_label}\n")
    lines.append(f"- GT   : `{args.gt}`")
    lines.append(f"- Pred : `{args.pred}`\n")

    lines.append("## Headline metrics (paper table 4)\n")
    lines.append(f"| Metric  | {args.gt_label} | {args.pred_label} | Δ ({args.pred_label}−{args.gt_label}) |")
    lines.append("|---------|-----:|-----:|-----:|")
    for label, key, _ in HEADLINE:
        a, b = pick(gt, key), pick(pred, key)
        delta = (b - a) if (isinstance(a, (int, float)) and isinstance(b, (int, float))) else None
        lines.append(f"| {label} | {fmt(a)} | {fmt(b)} | {fmt(delta)} |")

    # Subset variants if present
    SUBS = ["legs", "vr_3points", "other_upper_bodies", "foot"]
    sub_rows = []
    for sub in SUBS:
        for label, key, _ in HEADLINE[1:]:  # skip Succ
            full = f"{key}_{sub}"
            a, b = pick(gt, full), pick(pred, full)
            if a is None and b is None:
                continue
            delta = (b - a) if (isinstance(a, (int, float)) and isinstance(b, (int, float))) else None
            sub_rows.append((sub, label, a, b, delta))
    if sub_rows:
        lines.append("\n## Subset metrics\n")
        lines.append(f"| Subset | Metric | {args.gt_label} | {args.pred_label} | Δ |")
        lines.append("|--------|--------|-----:|-----:|-----:|")
        for sub, label, a, b, d in sub_rows:
            lines.append(f"| {sub} | {label} | {fmt(a)} | {fmt(b)} | {fmt(d)} |")

    out = "\n".join(lines) + "\n"
    print(out)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(out)
        print(f"[compare] wrote {args.out}")


if __name__ == "__main__":
    main()

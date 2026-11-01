"""
Robustness sweep: colorization accuracy against input degradation severity.

Uses the 34 autochromes as ground truth. For each degradation type and severity
level, degrade the autochrome, colorize it, and measure chroma error against the
true colour original.

python robustness.py
python robustness.py --types fade noise blur --levels 0 0.5 1.0
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage import color as skcolor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "colorization"))
sys.path.insert(0, os.path.join(HERE, "scripts"))

import colorizers  # noqa: E402
from degrade import CORE, EXTRA, degrade, image_seed  # noqa: E402

IMG_DIR = "data/matson"
FIG_DIR = "figures"
CSV = "matson.csv"


def load_meta():
    with open(CSV, encoding="utf-8") as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


def colourfulness(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                 + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def delta_e_ab(a, b):
    la = skcolor.rgb2lab(a)[:, :, 1:]
    lb = skcolor.rgb2lab(b)[:, :, 1:]
    return float(np.sqrt(((la - lb) ** 2).sum(axis=2)).mean())


def colorize(net, rgb_img, device):
    tens_l_orig, tens_l_rs = colorizers.preprocess_img(rgb_img, HW=(256, 256))
    tens_l_rs = tens_l_rs.to(device)
    with torch.no_grad():
        out = colorizers.postprocess_tens(tens_l_orig, net(tens_l_rs).cpu())
    return np.clip(out, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="siggraph17", choices=["eccv16", "siggraph17"])
    ap.add_argument("--types", nargs="+", default=CORE + EXTRA)
    ap.add_argument("--levels", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=693)
    args = ap.parse_args()

    out_csv = f"results_robustness_{args.model}.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = (colorizers.siggraph17(pretrained=True) if args.model == "siggraph17"
           else colorizers.eccv16(pretrained=True)).eval().to(device)

    meta = load_meta()
    autos = sorted(i for i, m in meta.items()
                   if "color" in m.get("medium", "").lower()
                   and "hand-color" not in m.get("medium", "").lower()
                   and os.path.exists(os.path.join(IMG_DIR, i + ".jpg")))
    if args.limit:
        autos = autos[:args.limit]

    total = len(autos) * len(args.types) * len(args.levels)
    print(f"device: {device}  model: {args.model}")
    print(f"{len(autos)} autochromes x {len(args.types)} types x {len(args.levels)} levels "
          f"= {total} colorizations")

    rows = []
    t0 = time.time()
    done = 0

    for iid in autos:
        ref_u8 = np.asarray(Image.open(os.path.join(IMG_DIR, iid + ".jpg")).convert("RGB"))
        ref = ref_u8.astype(np.float32) / 255.0
        c_ref = colourfulness(ref)

        for dtype in args.types:
            for sev in args.levels:
                dgr, _, _ = degrade(ref_u8, image_seed(args.seed, iid), [dtype], sev)
                pred = colorize(net, dgr, device)
                rows.append({
                    "id": iid,
                    "type": dtype,
                    "severity": sev,
                    "delta_e_ab": delta_e_ab(ref, pred),
                    "colourfulness_pred": colourfulness(pred),
                    "colourfulness_ref": c_ref,
                })
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{total}  ({time.time() - t0:.0f}s)")

    print(f"\n{total} colorizations in {time.time() - t0:.0f}s")

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")

    by = defaultdict(list)
    for r in rows:
        by[(r["type"], r["severity"])].append(r)

    print(f"\nMEAN deltaE_ab BY TYPE AND SEVERITY")
    header = "type".ljust(16) + "".join(f"{s:>9.2f}" for s in args.levels) + "     slope"
    print(header)
    print("-" * len(header))

    slopes = {}
    for dtype in args.types:
        means = [np.mean([r["delta_e_ab"] for r in by[(dtype, s)]]) for s in args.levels]
        slope = np.polyfit(args.levels, means, 1)[0]
        slopes[dtype] = slope
        print(dtype.ljust(16) + "".join(f"{m:>9.2f}" for m in means) + f"{slope:>10.2f}")

    print("\nmost damaging to colorization (by slope):")
    for dtype, s in sorted(slopes.items(), key=lambda kv: -kv[1])[:3]:
        print(f"  {dtype}: +{s:.2f} deltaE_ab per unit severity")

    print("\nMEAN predicted/reference COLOURFULNESS BY TYPE AND SEVERITY")
    h2 = "type".ljust(16) + "".join(f"{s:>9.2f}" for s in args.levels)
    print(h2)
    print("-" * len(h2))
    for dtype in args.types:
        ratios = [np.mean([r["colourfulness_pred"] / max(r["colourfulness_ref"], 1e-6)
                           for r in by[(dtype, s)]]) for s in args.levels]
        print(dtype.ljust(16) + "".join(f"{x:>9.0%}" for x in ratios))

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    for dtype in args.types:
        means = [np.mean([r["delta_e_ab"] for r in by[(dtype, s)]]) for s in args.levels]
        ax[0].plot(args.levels, means, marker="o", label=dtype, lw=1.8)
    ax[0].set_xlabel("degradation severity")
    ax[0].set_ylabel("mean deltaE_ab")
    ax[0].set_title(f"colorization error vs input degradation ({args.model})")
    ax[0].legend(fontsize=8, ncol=2)
    ax[0].grid(alpha=0.3)

    for dtype in args.types:
        ratios = [np.mean([r["colourfulness_pred"] / max(r["colourfulness_ref"], 1e-6)
                           for r in by[(dtype, s)]]) for s in args.levels]
        ax[1].plot(args.levels, ratios, marker="o", label=dtype, lw=1.8)
    ax[1].axhline(1.0, color="k", ls="--", lw=1, label="true colourfulness")
    ax[1].set_xlabel("degradation severity")
    ax[1].set_ylabel("predicted / reference colourfulness")
    ax[1].set_title("saturation response to degradation")
    ax[1].legend(fontsize=8, ncol=2)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/robustness_sweep_{args.model}.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote {FIG_DIR}/robustness_sweep_{args.model}.png")


if __name__ == "__main__":
    main()
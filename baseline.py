"""
Baseline colorization over the Matson images, two evaluation tiers.

Tier A (paired):   autochromes. Convert to greyscale, colorize, compare to the true original. PSNR / SSIM / dE.
Tier B (unpaired): greyscale scans. No ground truth, so report colourfulness and a no-reference sharpness proxy.

python baseline.py
python baseline.py --model siggraph17 --limit 50
"""

import argparse
import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage import color as skcolor
from skimage import exposure, restoration
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "colorization"))
import colorizers  # noqa: E402

IMG_DIR = "data/matson"
OUT_DIR = "outputs/baseline"
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


def sharpness(grey):
    gy, gx = np.gradient(grey.astype(float))
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def delta_e(a, b):
    la = skcolor.rgb2lab(a)
    lb = skcolor.rgb2lab(b)
    return float(np.sqrt(((la - lb) ** 2).sum(axis=2)).mean())


def delta_e_ab(a, b):
    la = skcolor.rgb2lab(a)[:, :, 1:]
    lb = skcolor.rgb2lab(b)[:, :, 1:]
    return float(np.sqrt(((la - lb) ** 2).sum(axis=2)).mean())

PREPROCESS = False


def preprocess(grey_u8):
    g = grey_u8.astype(np.float32) / 255.0
    g = restoration.denoise_bilateral(g, sigma_color=0.08, sigma_spatial=3)
    g = exposure.equalize_adapthist(g, clip_limit=0.01)
    return (np.clip(g, 0, 1) * 255).astype(np.uint8)

def colorize(net, rgb_img, device):
    tens_l_orig, tens_l_rs = colorizers.preprocess_img(rgb_img, HW=(256, 256))
    if PREPROCESS:
        g = np.asarray(Image.fromarray(rgb_img).convert("L"))
        enhanced = np.stack([preprocess(g)] * 3, axis=2)
        _, tens_l_rs = colorizers.preprocess_img(enhanced, HW=(256, 256))
    tens_l_rs = tens_l_rs.to(device)
    with torch.no_grad():
        out = colorizers.postprocess_tens(tens_l_orig, net(tens_l_rs).cpu())
    return np.clip(out, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="siggraph17", choices=["eccv16", "siggraph17"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--preprocess", action="store_true")
    args = ap.parse_args()

    global OUT_DIR, PREPROCESS
    PREPROCESS = args.preprocess
    tag = f"{args.model}_pre" if args.preprocess else args.model
    OUT_DIR = f"outputs/baseline_{tag}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  model: {args.model}")

    net = (colorizers.siggraph17(pretrained=True) if args.model == "siggraph17"
           else colorizers.eccv16(pretrained=True)).eval().to(device)

    meta = load_meta()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
    if args.limit:
        files = files[:args.limit]

    tier_a, tier_b = [], []
    t0 = time.time()

    for i, fn in enumerate(files, 1):
        iid = fn[:-4]
        m = meta.get(iid, {})
        is_auto = "autochrome" in m.get("medium", "").lower()

        try:
            orig = np.asarray(Image.open(os.path.join(IMG_DIR, fn)).convert("RGB"))
        except Exception as exc:
            print(f"  skip {fn}: {exc}")
            continue

        if is_auto:
            grey_rgb = np.stack([np.asarray(Image.fromarray(orig).convert("L"))] * 3, axis=2)
            pred = colorize(net, grey_rgb, device)
            ref = orig.astype(np.float32) / 255.0
            tier_a.append({
                "id": iid,
                "psnr": peak_signal_noise_ratio(ref, pred, data_range=1.0),
                "ssim": structural_similarity(ref, pred, channel_axis=2, data_range=1.0),
                "delta_e": delta_e(ref, pred),
                "delta_e_ab": delta_e_ab(ref, pred),
                "colourfulness_pred": colourfulness(pred),
                "colourfulness_ref": colourfulness(ref),
                "title": m.get("title", ""),
            })
            Image.fromarray((pred * 255).astype(np.uint8)).save(
                os.path.join(OUT_DIR, f"{iid}_pred.jpg"))
        else:
            pred = colorize(net, orig, device)
            grey = np.asarray(Image.fromarray(orig).convert("L"))
            tier_b.append({
                "id": iid,
                "colourfulness": colourfulness(pred),
                "sharpness": sharpness(grey),
                "dynamic_range": float(np.percentile(grey, 99) - np.percentile(grey, 1)),
                "title": m.get("title", ""),
            })
            if len(tier_b) <= 40:
                Image.fromarray((pred * 255).astype(np.uint8)).save(
                    os.path.join(OUT_DIR, f"{iid}_pred.jpg"))

        if i % 25 == 0:
            print(f"  {i}/{len(files)}")

    elapsed = time.time() - t0
    print(f"\n{len(files)} images in {elapsed:.1f}s ({elapsed/max(len(files),1):.2f}s each)")

    print(f"\nTIER A - paired, autochromes (n={len(tier_a)})")
    if tier_a:
        for k in ["psnr", "ssim", "delta_e", "delta_e_ab"]:
            vals = [r[k] for r in tier_a]
            print(f"  {k}: mean {np.mean(vals):.3f}  median {np.median(vals):.3f}  "
                  f"std {np.std(vals):.3f}")
        cp = np.mean([r["colourfulness_pred"] for r in tier_a])
        cr = np.mean([r["colourfulness_ref"] for r in tier_a])
        print(f"  colourfulness: predicted {cp:.4f} vs reference {cr:.4f} "
              f"({cp/cr*100:.0f}% of original)")

    print(f"\nTIER B - unpaired, greyscale scans (n={len(tier_b)})")
    if tier_b:
        for k in ["colourfulness", "sharpness", "dynamic_range"]:
            vals = [r[k] for r in tier_b]
            print(f"  {k}: mean {np.mean(vals):.2f}  median {np.median(vals):.2f}")

    for name, rows in [("tier_a", tier_a), ("tier_b", tier_b)]:
        if rows:
            with open(f"results_{name}_{tag}.csv", "w", newline="",
                      encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    if tier_a:
        worst = sorted(tier_a, key=lambda r: r["delta_e"], reverse=True)[:3]
        best = sorted(tier_a, key=lambda r: r["delta_e"])[:3]
        fig, axes = plt.subplots(3, 4, figsize=(14, 10))
        for row, rec in enumerate(best[:1] + worst[:2]):
            orig = np.asarray(Image.open(os.path.join(IMG_DIR, rec["id"] + ".jpg")).convert("RGB"))
            grey = np.asarray(Image.fromarray(orig).convert("L"))
            pred = np.asarray(Image.open(os.path.join(OUT_DIR, rec["id"] + "_pred.jpg")))
            for col, (img, lab) in enumerate([
                (grey, "input (greyscale)"), (pred, "predicted"),
                (orig, "true autochrome"), (None, "")]):
                ax = axes[row][col]
                if img is None:
                    ax.text(0.5, 0.5, f"dE {rec['delta_e']:.1f}\nPSNR {rec['psnr']:.1f}",
                            ha="center", va="center", fontsize=11)
                    ax.axis("off")
                    continue
                ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
                ax.set_title(lab, fontsize=9)
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/baseline_qualitative_{tag}.png", dpi=150)
        plt.close(fig)
        print(f"\nWrote {FIG_DIR}/baseline_qualitative_{tag}.png")

    print("Wrote results_tier_a/b CSVs and outputs/baseline/")


if __name__ == "__main__":
    main()
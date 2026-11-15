"""Shared evaluation protocol for every colorizer (issue #15, check 3). Model-agnostic.

Any model (UNet #17, SD #16, baselines) writes one RGB image per test id to
outputs/<name>/<id>.png (any resolution). This script then, for every image:

  1. SAME IMAGES       only the ids of the chosen split in splits/colour_split.csv (default: test);
                       refuses to score if any prediction is missing
  2. SAME RESOLUTION   prediction resized (bicubic) to the original photo's size
  3. SAME LUMINANCE    only the prediction's chrominance (ab) is kept and recombined with the
                       ORIGINAL photo's L, so a model is never rewarded or punished for
                       brightness changes (e.g. the SD VAE's luminance drift)
  4. SAME METRICS      PSNR, SSIM, dE (CIE76), dE_ab, colourfulness pred/ref; formulas as in
                       baseline.py. Since L is the original, dE equals dE_ab up to gamut clipping.

Writes results/<name>_<split>.csv (one row per image, for paired tests) and prints the summary.

Usage
  python -m condcolor.protocol outputs/condunet_real            # score a folder on the test split
  python -m condcolor.protocol outputs/condunet_real --split val
"""

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

IMG_DIR = Path("data/matson/images")
SPLIT = Path("splits/colour_split.csv")
RESULTS = Path("results")


# ---------------------------------------------------------------- metrics (as in baseline.py)

def colourfulness(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def delta_e(la, lb):
    return float(np.sqrt(((la - lb) ** 2).sum(axis=2)).mean())


def delta_e_ab(la, lb):
    return float(np.sqrt(((la[..., 1:] - lb[..., 1:]) ** 2).sum(axis=2)).mean())


# ---------------------------------------------------------------- protocol steps

def split_ids(split, path=SPLIT):
    with open(path, encoding="utf-8") as f:
        return [r["id"] for r in csv.DictReader(f) if r["split"] == split]


def load_truth(iid):
    """Original colour photo, float RGB in [0,1] at its stored resolution."""
    return np.asarray(Image.open(IMG_DIR / f"{iid}.jpg").convert("RGB"), dtype=np.float32) / 255.0


def recombine(truth_rgb, pred_rgb):
    """Steps 2 + 3: resize the prediction to the truth's size, keep its ab, use the truth's L.
    Returns (final RGB in [0,1], truth Lab, final Lab)."""
    h, w = truth_rgb.shape[:2]
    if pred_rgb.shape[:2] != (h, w):
        p = Image.fromarray((np.clip(pred_rgb, 0, 1) * 255).round().astype(np.uint8))
        pred_rgb = np.asarray(p.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0
    lab_t = color.rgb2lab(truth_rgb)
    lab_p = color.rgb2lab(pred_rgb)
    lab = np.concatenate([lab_t[..., :1], lab_p[..., 1:]], axis=2)
    with warnings.catch_warnings():  # out-of-gamut Lab values are clipped; the warning is noise
        warnings.simplefilter("ignore")
        out = np.clip(color.lab2rgb(lab), 0, 1).astype(np.float32)
    return out, lab_t, color.rgb2lab(out)  # Lab of the clipped, displayable result


def score_image(truth_rgb, pred_rgb):
    out, lab_t, lab_o = recombine(truth_rgb, pred_rgb)
    return {"psnr": peak_signal_noise_ratio(truth_rgb, out, data_range=1.0),
            "ssim": structural_similarity(truth_rgb, out, channel_axis=2, data_range=1.0),
            "delta_e": delta_e(lab_t, lab_o),
            "delta_e_ab": delta_e_ab(lab_t, lab_o),
            "colourfulness_pred": colourfulness(out),
            "colourfulness_ref": colourfulness(truth_rgb)}


def summarise(rows):
    s = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "delta_e", "delta_e_ab")}
    s["sat"] = float(np.mean([r["colourfulness_pred"] for r in rows]) /
                     np.mean([r["colourfulness_ref"] for r in rows]))
    s["n"] = len(rows)
    return s


def fmt(s):
    return (f"n={s['n']}  PSNR {s['psnr']:.2f}  SSIM {s['ssim']:.3f}  dE {s['delta_e']:.2f}  "
            f"dE_ab {s['delta_e_ab']:.2f}  colourfulness {100 * s['sat']:.0f}% of truth")


def evaluate_dir(pred_dir, split="test"):
    pred_dir = Path(pred_dir)
    ids = split_ids(split)
    missing = [i for i in ids if not (pred_dir / f"{i}.png").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(ids)} {split} predictions missing in {pred_dir}, "
                         f"e.g. {missing[:3]}; every model must be scored on the same images")
    rows = []
    for iid in ids:
        pred = np.asarray(Image.open(pred_dir / f"{iid}.png").convert("RGB"), dtype=np.float32) / 255.0
        rows.append({"id": iid, **score_image(load_truth(iid), pred)})
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{pred_dir.name}_{split}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    s = summarise(rows)
    print(f"{pred_dir.name} [{split}]  {fmt(s)}")
    print(f"per-image results -> {out}")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred_dir")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()
    evaluate_dir(args.pred_dir, args.split)

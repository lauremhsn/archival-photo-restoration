"""
EDA over the downloaded Matson images plus matson.csv.
Writes figures/ and prints the numbers for the report.

python eda.py
"""

import csv
import os
import statistics
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color as skcolor

IMG_DIR = "data/matson"
FIG_DIR = "figures"
CSV = "matson.csv"


def load_meta():
    with open(CSV, encoding="utf-8") as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


def is_colour(arr, tol=0.40):
    if arr.ndim < 3 or arr.shape[2] < 3:
        return False
    a = arr.astype(np.float32) / 255.0
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    return float(np.percentile(sat, 95)) > tol


def scan():
    meta = load_meta()
    rows = []
    files = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
    print(f"Scanning {len(files)} images")

    for fn in files:
        path = os.path.join(IMG_DIR, fn)
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                arr = np.asarray(im)
        except Exception as exc:
            print(f"  skip {fn}: {exc}")
            continue

        iid = fn[:-4]
        m = meta.get(iid, {})
        grey = np.asarray(Image.fromarray(arr).convert("L"))
        lab = skcolor.rgb2lab(arr.astype(np.float32) / 255.0)
        tone = float(np.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2).mean())
        rows.append({
            "id": iid,
            "w": arr.shape[1],
            "h": arr.shape[0],
            "aspect": arr.shape[1] / arr.shape[0],
            "mean": float(grey.mean()),
            "std": float(grey.std()),
            "p01": float(np.percentile(grey, 1)),
            "p99": float(np.percentile(grey, 99)),
            "colour": "autochrome" in m.get("medium", "").lower(),
            "tone": tone,
            "autochrome": "autochrome" in m.get("medium", "").lower(),
            "date": m.get("sort_date", ""),
            "location": m.get("location", ""),
            "title": m.get("title", ""),
        })
    return rows


def figures(rows):
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].scatter([r["w"] for r in rows], [r["h"] for r in rows], s=8, alpha=0.5)
    ax[0].set_xlabel("width (px)")
    ax[0].set_ylabel("height (px)")
    ax[0].set_title("Resolution")

    ax[1].hist([r["aspect"] for r in rows], bins=40)
    ax[1].set_xlabel("aspect ratio (w/h)")
    ax[1].set_title("Aspect ratio")

    ax[2].hist([r["mean"] for r in rows], bins=40)
    ax[2].set_xlabel("mean luminance")
    ax[2].set_title("Brightness")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/basic_stats.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist([r["std"] for r in rows], bins=40)
    ax[0].set_xlabel("luminance std dev")
    ax[0].set_title("Contrast spread")

    dyn = [r["p99"] - r["p01"] for r in rows]
    ax[1].hist(dyn, bins=40)
    ax[1].set_xlabel("p99 - p01")
    ax[1].set_title("Dynamic range (fading proxy)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/contrast.png", dpi=150)
    plt.close(fig)

    years = []
    for r in rows:
        try:
            y = int(str(r["date"])[:4])
            if 1850 < y < 1960:
                years.append(y)
        except (ValueError, TypeError):
            pass
    if years:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(years, bins=range(min(years), max(years) + 2))
        ax.set_xlabel("year")
        ax.set_title("Date distribution")
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/dates.png", dpi=150)
        plt.close(fig)

    sample = [r for r in rows if r["colour"]][:4] + [r for r in rows if not r["colour"]][:4]
    if sample:
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        for a, r in zip(axes.flat, sample):
            with Image.open(os.path.join(IMG_DIR, r["id"] + ".jpg")) as im:
                a.imshow(im)
            a.set_title(("colour" if r["colour"] else "greyscale"), fontsize=9)
            a.axis("off")
        for a in axes.flat[len(sample):]:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/samples.png", dpi=150)
        plt.close(fig)


def report(rows):
    n = len(rows)
    print(f"\n{n} images analysed\n")

    ws = [r["w"] for r in rows]
    hs = [r["h"] for r in rows]
    print(f"width  median {int(statistics.median(ws))}  range {min(ws)}-{max(ws)}")
    print(f"height median {int(statistics.median(hs))}  range {min(hs)}-{max(hs)}")

    land = sum(1 for r in rows if r["aspect"] > 1)
    print(f"landscape {land} ({land/n*100:.0f}%)  portrait {n-land} ({(n-land)/n*100:.0f}%)")

    col = sum(1 for r in rows if r["colour"])
    auto = sum(1 for r in rows if r["autochrome"])
    print(f"\ncolour images {col} ({col/n*100:.1f}%)")
    print(f"autochromes per metadata {auto}")
    print(f"greyscale (colorization targets) {n-col}")

    tones = [r["tone"] for r in rows if not r["colour"]]
    print(f"\nsepia strength (mean LAB chroma, greyscale scans only)")
    print(f"  median {statistics.median(tones):.1f}  range {min(tones):.1f}-{max(tones):.1f}")

    dyn = [r["p99"] - r["p01"] for r in rows]
    faded = sum(1 for d in dyn if d < 180)
    print(f"\nmean luminance {statistics.mean([r['mean'] for r in rows]):.1f}")
    print(f"median dynamic range {statistics.median(dyn):.0f}/255")
    print(f"low dynamic range (<180, fading proxy) {faded} ({faded/n*100:.0f}%)")

    locs = Counter()
    for r in rows:
        for part in r["location"].split(";"):
            part = part.strip()
            if part:
                locs[part] += 1
    print("\ntop locations:")
    for loc, c in locs.most_common(8):
        print(f"  {loc}: {c}")

    print(f"\nFigures written to {FIG_DIR}/")


if __name__ == "__main__":
    rows = scan()
    if not rows:
        raise SystemExit("No images found. Run ingest.py --download first.")
    figures(rows)
    report(rows)
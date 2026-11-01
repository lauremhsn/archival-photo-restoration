"""
Failure analysis over the Tier A results.

python failures.py
"""

import csv
import glob
import os
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

FIG_DIR = "figures"
IMG_DIR = "data/matson"
OUT_ROOT = "outputs"

BUCKETS = {
    "vegetation": r"oleander|bloom|flower|garden|tree|grove|field|palm|orchard|vine",
    "stone/architecture": r"petra|rock|temple|tomb|wall|ruin|city|mosque|church|gate|street|khazneh",
    "water": r"sea|river|jordan|lake|spring|pool|well",
    "people": r"figure|man|woman|men|women|group|bedouin|arab|boy|girl|costume",
}


def load(path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("psnr", "ssim", "delta_e", "delta_e_ab",
                  "colourfulness_pred", "colourfulness_ref"):
            if k in r and r[k]:
                r[k] = float(r[k])
    return {r["id"]: r for r in rows}


def bucket(title):
    t = title.lower()
    hits = [name for name, pat in BUCKETS.items() if re.search(pat, t)]
    return hits or ["other"]


def main():
    configs = {}
    for path in sorted(glob.glob("results_tier_a_*.csv")):
        name = os.path.basename(path)[len("results_tier_a_"):-len(".csv")]
        configs[name] = load(path)
    if not configs:
        raise SystemExit("No results_tier_a_*.csv found.")

    print(f"configs: {', '.join(configs)}")
    best = min(configs, key=lambda c: statistics.mean(
        r["delta_e_ab"] for r in configs[c].values()))
    print(f"best by mean deltaE_ab: {best}\n")

    rows = list(configs[best].values())
    n = len(rows)

    print(f"WORST 10 BY deltaE_ab ({best}, n={n})")
    print(f"{'dE_ab':>7} {'PSNR':>7} {'pred/ref':>9}  title")
    print("-" * 78)
    for r in sorted(rows, key=lambda r: -r["delta_e_ab"])[:10]:
        ratio = r["colourfulness_pred"] / max(r["colourfulness_ref"], 1e-6)
        print(f"{r['delta_e_ab']:>7.1f} {r['psnr']:>7.1f} {ratio:>8.0%}  {r['title'][:46]}")

    print(f"\nBEST 5")
    print("-" * 78)
    for r in sorted(rows, key=lambda r: r["delta_e_ab"])[:5]:
        ratio = r["colourfulness_pred"] / max(r["colourfulness_ref"], 1e-6)
        print(f"{r['delta_e_ab']:>7.1f} {r['psnr']:>7.1f} {ratio:>8.0%}  {r['title'][:46]}")

    cref = np.array([r["colourfulness_ref"] for r in rows])
    cpred = np.array([r["colourfulness_pred"] for r in rows])
    de = np.array([r["delta_e_ab"] for r in rows])

    rho = float(np.corrcoef(cref, de)[0, 1])
    print(f"\ncorrelation between reference colourfulness and deltaE_ab: r = {rho:.3f}")
    if rho > 0.3:
        print("  -> the model fails hardest on the most colourful scenes")

    under = int((cpred < cref).sum())
    print(f"under-saturated on {under}/{n}, over-saturated on {n - under}/{n}")

    print("\nBY SCENE CONTENT (from title keywords)")
    print(f"{'bucket':<20} {'n':>4} {'mean dE_ab':>11} {'mean pred/ref':>14}")
    print("-" * 56)
    per_bucket = {}
    for r in rows:
        for b in bucket(r["title"]):
            per_bucket.setdefault(b, []).append(r)
    for b, rs in sorted(per_bucket.items(), key=lambda kv: -len(kv[1])):
        if len(rs) < 3:
            continue
        mde = statistics.mean(x["delta_e_ab"] for x in rs)
        mratio = statistics.mean(
            x["colourfulness_pred"] / max(x["colourfulness_ref"], 1e-6) for x in rs)
        print(f"{b:<20} {len(rs):>4} {mde:>11.1f} {mratio:>13.0%}")

    if len(configs) > 1:
        print("\nPAIRED COMPARISON (Wilcoxon signed-rank on deltaE_ab)")
        names = list(configs)
        base = configs[best]
        for other in names:
            if other == best:
                continue
            ids = [i for i in base if i in configs[other]]
            a = [base[i]["delta_e_ab"] for i in ids]
            b = [configs[other][i]["delta_e_ab"] for i in ids]
            wins = sum(1 for x, y in zip(a, b) if x < y)
            try:
                stat, p = wilcoxon(a, b)
                print(f"  {best} vs {other}: better on {wins}/{len(ids)}, p = {p:.2e}")
            except ValueError:
                print(f"  {best} vs {other}: better on {wins}/{len(ids)}")

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    lim = max(cref.max(), cpred.max()) * 1.1
    ax[0].scatter(cref, cpred, s=30, alpha=0.7)
    ax[0].plot([0, lim], [0, lim], "k--", lw=1, label="perfect")
    ax[0].set_xlim(0, lim)
    ax[0].set_ylim(0, lim)
    ax[0].set_xlabel("reference colourfulness")
    ax[0].set_ylabel("predicted colourfulness")
    ax[0].set_title(f"saturation calibration ({best})")
    ax[0].legend(fontsize=8)

    ax[1].scatter(cref, de, s=30, alpha=0.7)
    ax[1].set_xlabel("reference colourfulness")
    ax[1].set_ylabel("deltaE_ab")
    ax[1].set_title(f"error vs scene colourfulness (r={rho:.2f})")

    for name, d in configs.items():
        vals = sorted(r["delta_e_ab"] for r in d.values())
        ax[2].plot(vals, np.linspace(0, 1, len(vals)), label=name, lw=1.8)
    ax[2].set_xlabel("deltaE_ab")
    ax[2].set_ylabel("cumulative fraction")
    ax[2].set_title("error distribution by config")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/failure_analysis.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote {FIG_DIR}/failure_analysis.png")

    worst = sorted(rows, key=lambda r: -r["delta_e_ab"])[:3]
    pred_dir = os.path.join(OUT_ROOT, f"baseline_{best}")
    if os.path.isdir(pred_dir):
        from PIL import Image
        fig, axes = plt.subplots(3, 3, figsize=(11, 11))
        for i, r in enumerate(worst):
            src = os.path.join(IMG_DIR, r["id"] + ".jpg")
            pre = os.path.join(pred_dir, r["id"] + "_pred.jpg")
            if not (os.path.exists(src) and os.path.exists(pre)):
                continue
            orig = np.asarray(Image.open(src).convert("RGB"))
            grey = np.asarray(Image.open(src).convert("L"))
            pred = np.asarray(Image.open(pre))
            for j, (img, lab) in enumerate([(grey, "input"), (pred, "predicted"),
                                            (orig, "true autochrome")]):
                axes[i][j].imshow(img, cmap="gray" if img.ndim == 2 else None)
                axes[i][j].axis("off")
                if i == 0:
                    axes[i][j].set_title(lab, fontsize=10)
            axes[i][0].set_ylabel(f"dE {r['delta_e_ab']:.1f}")
        fig.suptitle(f"worst 3 by deltaE_ab ({best})", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/worst_cases.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {FIG_DIR}/worst_cases.png")


if __name__ == "__main__":
    main()
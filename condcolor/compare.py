"""Paired comparison of protocol results on the same images (for #17 / #18).

For each pair (A vs B) on the per-image CSVs in results/: mean difference in dE_ab with a
bootstrap 95% CI, how many images A beats B on, and an exact two-sided sign test.
Also writes an example figure: input / each condition / truth.

Usage
  python -m condcolor.compare condunet_real condunet_none condunet_shuffled grey_reference
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from condcolor.protocol import load_truth, recombine, split_ids

RESULTS = Path("results")


def load(name, split):
    with open(RESULTS / f"{name}_{split}.csv", encoding="utf-8") as f:
        return {r["id"]: float(r["delta_e_ab"]) for r in csv.DictReader(f)}


def sign_test(wins, n):
    """Exact two-sided binomial test with p = 0.5 (ties excluded)."""
    k = min(wins, n - wins)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def bootstrap_ci(d, n=10000, seed=693):
    rng = np.random.default_rng(seed)
    means = rng.choice(d, (n, len(d)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="first is the reference condition (e.g. condunet_real)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--figure", default="figures/condunet_test_examples.png")
    ap.add_argument("--n-examples", type=int, default=6)
    args = ap.parse_args()

    ref_name = args.names[0]
    ref = load(ref_name, args.split)
    ids = sorted(ref)
    print(f"dE_ab on {len(ids)} {args.split} images (lower is better)")
    print(f"{'condition':<22}{'mean':>7}{'median':>8}")
    for name in args.names:
        v = np.array([load(name, args.split)[i] for i in ids])
        print(f"{name:<22}{v.mean():7.2f}{np.median(v):8.2f}")
    print(f"\npaired vs {ref_name} (positive = {ref_name} is better)")
    for name in args.names[1:]:
        other = load(name, args.split)
        d = np.array([other[i] - ref[i] for i in ids])
        lo, hi = bootstrap_ci(d)
        wins, ties = int((d > 0).sum()), int((d == 0).sum())
        print(f"  vs {name:<20} diff {d.mean():+.2f} [95% CI {lo:+.2f}, {hi:+.2f}]  "
              f"{ref_name} better on {wins}/{len(ids) - ties}  sign test p={sign_test(wins, len(ids) - ties):.2g}")

    # example figure: images spread across the error range of the reference condition
    order = sorted(ids, key=lambda i: ref[i])
    pick = [order[round(k * (len(order) - 1) / (args.n_examples - 1))] for k in range(args.n_examples)]
    cols = ["input"] + args.names + ["truth"]
    fig, ax = plt.subplots(len(pick), len(cols), figsize=(2.3 * len(cols), 2.0 * len(pick)))
    for r, iid in enumerate(pick):
        truth = load_truth(iid)
        grey = np.repeat(truth.mean(axis=2, keepdims=True), 3, axis=2)
        imgs = [grey]
        for name in args.names:
            pred = np.asarray(Image.open(Path("outputs") / name / f"{iid}.png").convert("RGB"),
                              dtype=np.float32) / 255.0
            imgs.append(recombine(truth, pred)[0])
        imgs.append(truth)
        for c, img in enumerate(imgs):
            ax[r, c].imshow(img)
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(cols[c].replace("condunet_", "captions: "), fontsize=9)
        ax[r, 0].set_ylabel(f"{iid}\n{ref_name.split('_')[-1]} dE {ref[iid]:.1f}", fontsize=7)
    fig.tight_layout()
    Path(args.figure).parent.mkdir(exist_ok=True)
    fig.savefig(args.figure, dpi=100)
    print(f"\nwrote {args.figure}")


if __name__ == "__main__":
    main()

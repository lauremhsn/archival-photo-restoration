"""
compares the two overnight runs against each other and against the cnn baselines,
on exactly the same images.

python sd_compare.py
"""

import csv, glob, os
import numpy as np
from scipy import stats


def load(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


def main():
    runs = {}
    for p in sorted(glob.glob("results_tier_a_*.csv")):
        name = os.path.basename(p)[len("results_tier_a_"):-4]
        d = load(p)
        if d:
            runs[name] = d
    if not runs:
        raise SystemExit("no results_tier_a_*.csv here")

    sd = [k for k in runs if k.startswith("sd")]
    if len(sd) < 2:
        print("need both sd runs before the conditioning test\n")

    common = set.intersection(*[set(v) for v in runs.values()])
    print(f"{len(common)} images all runs have in common\n")
    print(f"{'run':<24} {'n':>6} {'dE_ab':>8} {'psnr':>7} {'ssim':>7} {'sat':>6}")
    for name, d in sorted(runs.items(), key=lambda kv: np.mean(
            [float(r["delta_e_ab"]) for r in kv[1].values()])):
        sel = [d[i] for i in common] if common else list(d.values())
        de = np.mean([float(r["delta_e_ab"]) for r in sel])
        pr = np.mean([float(r["psnr"]) for r in sel])
        ss = np.mean([float(r["ssim"]) for r in sel])
        sat = np.mean([float(r["colourfulness_pred"]) / max(float(r["colourfulness_ref"]), 1e-6)
                       for r in sel])
        print(f"{name:<24} {len(sel):>6} {de:>8.2f} {pr:>7.2f} {ss:>7.3f} {sat*100:>5.0f}%")

    if len(sd) >= 2:
        a = [k for k in sd if k.endswith("title")][0]
        b = [k for k in sd if k.endswith("generic")][0]
        ids = sorted(set(runs[a]) & set(runs[b]))
        x = np.array([float(runs[a][i]["delta_e_ab"]) for i in ids])
        y = np.array([float(runs[b][i]["delta_e_ab"]) for i in ids])
        w = stats.wilcoxon(x, y)
        wins = int((x < y).sum())
        print(f"\n=== does the catalogue title help? n={len(ids)} ===")
        print(f"  with title   dE_ab {x.mean():.3f}")
        print(f"  generic      dE_ab {y.mean():.3f}")
        print(f"  difference   {y.mean()-x.mean():+.3f}  (positive means the title helped)")
        print(f"  title better on {wins}/{len(ids)} ({wins/len(ids)*100:.1f}%)")
        print(f"  wilcoxon p = {w.pvalue:.3g}")
        if w.pvalue < 0.05:
            print("  -> significant. remember to report the win rate too, n is big")
        else:
            print("  -> not significant. zero shot, so this is a floor not a verdict")


if __name__ == "__main__":
    main()

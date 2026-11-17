"""
tests the whole reframe against HECD's human ratings instead of js asserting it.

HECD gives us 20 images with controlled chroma and hue modifications, each one rated by real people
(1281 of them). so we can actually check:
  1. does dE_ab prefer the desaturated version while humans prefer the saturated one
  2. does the 60 degree crossover show up in anything humans do
  3. does splitting into chroma and hue track humans better than the aggregate does
  4. and whether their own 'hc' numbers are low bc hue got handled wrong

usage:
  python hecd_validate.py --root path/to/HECD-main
"""

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats
from skimage import color, io

FLOOR = 5.0


def lab_of(path):
    img = io.imread(path)
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    return color.rgb2lab(img[:, :, :3].astype(np.float64) / 255.0)


def measures(lab_gt, lab_re):
    a1, b1 = lab_gt[:, :, 1], lab_gt[:, :, 2]
    a2, b2 = lab_re[:, :, 1], lab_re[:, :, 2]
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
    dE = np.sqrt((a2 - a1) ** 2 + (b2 - b1) ** 2)
    dC = c2 - c1
    dH = np.sqrt(np.maximum(dE ** 2 - dC ** 2, 0.0))
    h1, h2 = np.arctan2(b1, a1), np.arctan2(b2, a2)
    dh = np.degrees(np.abs(np.arctan2(np.sin(h2 - h1), np.cos(h2 - h1))))
    m = (c1 >= FLOOR) & (c2 >= FLOOR)
    cm = c1 >= FLOOR
    out = {
        "dE_ab": float(np.mean(dE)),
        "dC_abs": float(np.mean(np.abs(dC))),
        "dH": float(np.mean(dH[m])) if m.any() else float("nan"),
        "chroma_ratio": float(np.median(c2[cm] / c1[cm])) if cm.any() else float("nan"),
        "hue_err": float(np.sum(dh[m] * c1[m]) / np.sum(c1[m])) if m.any() else float("nan"),
        "cov": float(100.0 * m.mean()),
    }
    return out


def their_hue_channel_demo():
    """shows why their hc numbers come out near zero: hue is circular, they stored it linear in uint8."""
    print("\n--- why HECD's 'hc' space scores near zero ---")
    # two pixels either side of the +-pi wrap. same hue to the eye, opposite ends of a uint8 channel
    lab1 = np.dstack([[[50.0]], [[-40.0]], [[1.0]]])
    lab2 = np.dstack([[[50.0]], [[-40.0]], [[-1.0]]])
    h1 = math.atan2(1.0, -40.0)
    h2 = math.atan2(-1.0, -40.0)
    u1 = h1 / math.pi * 128 + 128
    u2 = h2 / math.pi * 128 + 128
    true_sep = math.degrees(abs(math.atan2(math.sin(h2 - h1), math.cos(h2 - h1))))
    m = measures(lab1, lab2)
    print(f"  two pixels {true_sep:.1f} deg apart in hue")
    print(f"  their uint8 hue channel puts them at {u1:.1f} and {u2:.1f}, a gap of {abs(u1-u2):.1f} out of 255")
    print(f"  real dE_ab between them: {m['dE_ab']:.3f}")
    print("  so a hue that barely moved reads as a near-max error, on every pixel near the wrap")
    print("  and theres no chroma mask on top of that, so near-grey pixels have basically random hue")


def their_colourfulness_bug(root):
    """their colourfulness indexes rows instead of channels. check it on a real image."""
    print("\n--- HECD's colourfulness indexing ---")
    gt = os.path.join(root, "HECDImages", "015004_gt.jpg")
    img = io.imread(gt)
    lab = color.rgb2lab(img.astype(np.float64) / 255.0)
    print(f"  image shape {lab.shape}")
    wrong_a = lab[1]
    right_a = lab[:, :, 1]
    print(f"  their 'gt_file[1]' grabs shape {wrong_a.shape} -> thats image ROW 1, not channel 1")
    print(f"  the channel they meant, gt_file[:,:,1], is shape {right_a.shape}")
    print(f"  std of row 1 (what they compute): {np.std(wrong_a):.3f}")
    print(f"  std of channel a (what they meant): {np.std(right_a):.3f}")
    print("  so their colourfulness comes from two single rows of pixels, not the image")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="hecd_results.csv")
    args = ap.parse_args()

    agg = os.path.join(args.root, "bokeh_app", "HumanAggregatedResults.csv")
    imgdir = os.path.join(args.root, "HECDImages")
    rows = list(csv.DictReader(open(agg, encoding="utf-8-sig")))
    print(f"{len(rows)} rated pairs in HECD")

    recs = []
    cache = {}
    for i, r in enumerate(rows, 1):
        gt_name, re_name = r["GTFileName"], r["ReColFileName"]
        gp, rp = os.path.join(imgdir, gt_name), os.path.join(imgdir, re_name)
        if not (os.path.exists(gp) and os.path.exists(rp)):
            continue
        try:
            z = float(r["zScore"])
        except (ValueError, KeyError):
            continue
        if gt_name not in cache:
            cache[gt_name] = lab_of(gp)
        lab_gt = cache[gt_name]
        lab_re = lab_of(rp)
        if lab_re.shape != lab_gt.shape:
            continue
        m = measures(lab_gt, lab_re)
        m.update({"gt": gt_name, "re": re_name, "z": z, "details": r["Details"].strip(),
                  "sota": r["SOTA"]})
        recs.append(m)
        if i % 300 == 0:
            print(f"  {i}/{len(rows)}")

    print(f"scored {len(recs)} pairs\n")
    keys = ["dE_ab", "dC_abs", "dH", "chroma_ratio", "hue_err"]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["gt", "re", "details", "sota", "z", "cov"] + keys)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"wrote {args.out}")

    print("\n--- how well each thing agrees with the human z-score, all pairs ---")
    z = np.array([r["z"] for r in recs])
    for k in keys:
        v = np.array([r[k] for r in recs], dtype=float)
        ok = np.isfinite(v) & np.isfinite(z)
        if ok.sum() < 20:
            print(f"  {k}: too few finite values")
            continue
        tau, pt = stats.kendalltau(v[ok], z[ok])
        rho, pr = stats.spearmanr(v[ok], z[ok])
        print(f"  {k:14s} n={ok.sum():4d}  kendall tau={tau:+.3f} (p={pt:.2g})  spearman={rho:+.3f} (p={pr:.2g})")

    print("\n--- the big one: chroma shifted toward and away from grey ---")
    fam = defaultdict(list)
    for r in recs:
        if r["details"].startswith("mean chroma shift"):
            fam[r["details"]].append(r)
    for d in sorted(fam):
        g = fam[d]
        print(f"  {d:34s} n={len(g):3d}  human z={np.mean([x['z'] for x in g]):+.3f}  "
              f"dE_ab={np.mean([x['dE_ab'] for x in g]):6.2f}  "
              f"chroma ratio={np.nanmean([x['chroma_ratio'] for x in g]):.3f}")
    if fam:
        print("  read it like this: if the -sd rows get BETTER dE_ab but WORSE human z than the +sd rows,")
        print("  then the metric is rewarding desaturation that people dont actually like. thats the claim.")

    print("\n--- controlled hue offsets, does 60 degrees mean anything to humans ---")
    hue_fam = [r for r in recs if r["details"].startswith("changing segment hue by")]
    buckets = defaultdict(list)
    for r in hue_fam:
        try:
            step = int(r["details"].split("changing segment hue by ")[1].split(" ")[0])
        except (IndexError, ValueError):
            continue
        buckets[step].append(r)
    print("  uint8 step -> degrees, human z, dE_ab, measured hue error")
    for step in sorted(buckets):
        g = buckets[step]
        deg = step / 256.0 * 360.0
        print(f"   {step:4d} -> {deg:6.1f} deg  n={len(g):3d}  z={np.mean([x['z'] for x in g]):+.3f}  "
              f"dE_ab={np.mean([x['dE_ab'] for x in g]):6.2f}  "
              f"hue_err={np.nanmean([x['hue_err'] for x in g]):6.1f}")

    print("\n--- the six real colourisers ---")
    sota = defaultdict(list)
    for r in recs:
        if r["details"].startswith("Colourised by"):
            sota[r["details"]].append(r)
    if sota:
        print("  method human_z dE_ab chroma_ratio hue_err")
        ranked = sorted(sota.items(), key=lambda kv: -np.mean([x["z"] for x in kv[1]]))
        for d, g in ranked:
            print(f"  {d:26s} {np.mean([x['z'] for x in g]):+7.3f} {np.mean([x['dE_ab'] for x in g]):7.2f} "
                  f"{np.nanmean([x['chroma_ratio'] for x in g]):13.3f} {np.nanmean([x['hue_err'] for x in g]):8.1f}")
        hz = [np.mean([x["z"] for x in g]) for _, g in ranked]
        hd = [np.mean([x["dE_ab"] for x in g]) for _, g in ranked]
        if len(hz) > 2:
            tau, p = stats.kendalltau(hd, hz)
            print(f"  dE_ab vs the human ranking across methods: kendall tau={tau:+.3f} (p={p:.2g})")
            print("  only 6 methods so p is big. report it as no detectable agreement, NOT as anti-correlated")

    their_hue_channel_demo()
    their_colourfulness_bug(args.root)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pull every result into one compact printout for the report."""

import csv
import glob
import os
import statistics as st
from collections import Counter, defaultdict

import numpy as np


def load(path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("id", "title", "type", "medium", "location", "subject",
                     "notes", "date", "image_url", "sort_date", "location_raw"):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


print("=" * 70)
print("DATASET")
print("=" * 70)
meta = load("matson.csv")
n = len(meta)
print(f"records: {n}")
for f in ["title", "subject", "location", "date", "notes", "image_url"]:
    vals = [r[f] for r in meta if r.get(f)]
    mw = int(st.median([len(str(v).split()) for v in vals])) if vals else 0
    print(f"  {f:<11} {len(vals):>5} ({len(vals)/n*100:>5.1f}%)  median {mw} words")

col = [r for r in meta if "color" in r["medium"].lower()
       and "hand-color" not in r["medium"].lower()]
auto = [r for r in col if "autochrome" in r["medium"].lower()]
hand = [r for r in meta if "hand-color" in r["medium"].lower()]
print(f"\ncolour originals: {len(col)} ({len(col)/n*100:.1f}%)")
print(f"  of which autochrome: {len(auto)}")
print(f"hand-coloured (excluded): {len(hand)}")
print(f"greyscale targets: {n - len(col) - len(hand)}")

usable = sum(1 for r in meta if len(r["title"].split()) >= 5 and r["image_url"])
print(f"usable for conditioning (title>=5w + image): {usable} ({usable/n*100:.1f}%)")

imgs = len(glob.glob("data/matson/*.jpg"))
print(f"images on disk: {imgs}")

locs = Counter()
for r in meta:
    for p in str(r.get("location", "")).split(";"):
        p = p.strip()
        if p:
            locs[p] += 1
print("top locations: " + ", ".join(f"{k} {v}" for k, v in locs.most_common(6)))

print()
print("=" * 70)
print("BASELINES - TIER A (paired, colour originals)")
print("=" * 70)
print(f"{'config':<20} {'n':>5} {'PSNR':>7} {'SSIM':>7} {'dE':>7} {'dE_ab':>7} {'sat':>7}")
print("-" * 70)
tier_a = {}
for p in sorted(glob.glob("results_tier_a_*.csv")):
    name = os.path.basename(p)[len("results_tier_a_"):-4]
    rows = load(p)
    tier_a[name] = rows
    if not rows:
        continue
    sat = np.mean([r["colourfulness_pred"] / max(r["colourfulness_ref"], 1e-9)
                   for r in rows])
    de_ab = np.mean([r.get("delta_e_ab", r["delta_e"]) for r in rows])
    print(f"{name:<20} {len(rows):>5} "
          f"{np.mean([r['psnr'] for r in rows]):>7.2f} "
          f"{np.mean([r['ssim'] for r in rows]):>7.3f} "
          f"{np.mean([r['delta_e'] for r in rows]):>7.2f} "
          f"{de_ab:>7.2f} {sat:>6.0%}")

print()
print("=" * 70)
print("BASELINES - TIER B (unpaired greyscale)")
print("=" * 70)
for p in sorted(glob.glob("results_tier_b_*.csv")):
    name = os.path.basename(p)[len("results_tier_b_"):-4]
    rows = load(p)
    if not rows:
        continue
    print(f"{name:<20} n={len(rows):<5} "
          f"colourfulness {np.mean([r['colourfulness'] for r in rows]):.3f}  "
          f"sharpness {np.mean([r['sharpness'] for r in rows]):.2f}  "
          f"dyn range {np.mean([r['dynamic_range'] for r in rows]):.1f}")

if tier_a:
    best = min(tier_a, key=lambda c: np.mean(
        [r.get("delta_e_ab", r["delta_e"]) for r in tier_a[c]]))
    rows = tier_a[best]
    print()
    print("=" * 70)
    print(f"FAILURE ANALYSIS ({best})")
    print("=" * 70)
    cref = np.array([r["colourfulness_ref"] for r in rows])
    cpred = np.array([r["colourfulness_pred"] for r in rows])
    de = np.array([r.get("delta_e_ab", r["delta_e"]) for r in rows])
    print(f"n = {len(rows)}")
    print(f"corr(reference colourfulness, dE_ab) = {np.corrcoef(cref, de)[0,1]:.3f}")
    under = int((cpred < cref).sum())
    print(f"under-saturated {under}/{len(rows)} ({under/len(rows)*100:.0f}%), "
          f"over {len(rows)-under}")
    print(f"dE_ab  mean {de.mean():.2f}  median {np.median(de):.2f}  "
          f"p10 {np.percentile(de,10):.2f}  p90 {np.percentile(de,90):.2f}")
    print("\nworst 5:")
    for r in sorted(rows, key=lambda r: -r.get("delta_e_ab", r["delta_e"]))[:5]:
        ratio = r["colourfulness_pred"] / max(r["colourfulness_ref"], 1e-9)
        print(f"  dE {r.get('delta_e_ab', r['delta_e']):>6.1f}  sat {ratio:>5.0%}  "
              f"{r['title'][:48]}")

    if len(tier_a) > 1:
        try:
            from scipy.stats import wilcoxon
            print("\npaired vs best:")
            base = {r["id"]: r for r in tier_a[best]}
            for other in tier_a:
                if other == best:
                    continue
                o = {r["id"]: r for r in tier_a[other]}
                ids = [i for i in base if i in o]
                a = [base[i].get("delta_e_ab", base[i]["delta_e"]) for i in ids]
                b = [o[i].get("delta_e_ab", o[i]["delta_e"]) for i in ids]
                wins = sum(1 for x, y in zip(a, b) if x < y)
                _, pv = wilcoxon(a, b)
                print(f"  {best} vs {other}: {wins}/{len(ids)}, p = {pv:.2e}")
        except Exception as exc:
            print(f"  (wilcoxon skipped: {exc})")

print()
print("=" * 70)
print("ROBUSTNESS (slope of dE_ab per unit severity)")
print("=" * 70)
for p in sorted(glob.glob("results_robustness_*.csv")):
    name = os.path.basename(p)[len("results_robustness_"):-4]
    rows = load(p)
    if not rows:
        continue
    by = defaultdict(list)
    for r in rows:
        by[(r["type"], r["severity"])].append(r)
    levels = sorted({r["severity"] for r in rows})
    types = sorted({r["type"] for r in rows})
    nimg = len({r["id"] for r in rows})
    print(f"\n{name}  (n={nimg} images, {len(rows)} runs)")
    print(f"{'type':<16}" + "".join(f"{s:>8.2f}" for s in levels) + f"{'slope':>9}{'sat@1':>8}")
    for t in types:
        means = [np.mean([r["delta_e_ab"] for r in by[(t, s)]]) for s in levels]
        slope = np.polyfit(levels, means, 1)[0]
        sat = np.mean([r["colourfulness_pred"] / max(r["colourfulness_ref"], 1e-9)
                       for r in by[(t, levels[-1])]])
        print(f"{t:<16}" + "".join(f"{m:>8.2f}" for m in means)
              + f"{slope:>9.2f}{sat:>7.0%}")
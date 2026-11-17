"""
pulls chroma ratio out of the degradation sweep so we stop js asserting that noise and blur push
the model toward grey and actually show it.

feed it the per-run csvs that decompose.py writes. it wants run names that have the damage type and
the severity in them, default pattern is <damage>_s<severity>.

usage:
  python sweep_chroma.py --results results --pattern "(?P<damage>[a-z]+)_s(?P<sev>[0-9]+)_decomposed.csv"
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--pattern", default=r"(?P<damage>[a-z0-9]+)_s(?P<sev>[0-9]+)_decomposed\.csv")
    ap.add_argument("--out", default="results/sweep_chroma.csv")
    args = ap.parse_args()

    rx = re.compile(args.pattern)
    buckets = defaultdict(list)
    matched, skipped = [], []

    for path in sorted(glob.glob(os.path.join(args.results, "*_decomposed.csv"))):
        m = rx.search(os.path.basename(path))
        if not m:
            skipped.append(os.path.basename(path))
            continue
        damage, sev = m.group("damage"), int(m.group("sev"))
        matched.append((damage, sev, os.path.basename(path)))
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    buckets[(damage, sev)].append(
                        (float(row["chroma_ratio"]), float(row["dE_ab"]), float(row.get("hue_err_deg") or "nan"))
                    )
                except (KeyError, ValueError):
                    continue

    print(f"matched {len(matched)} run files")
    for damage, sev, name in matched:
        print(f"  {damage} severity {sev} <- {name}")
    if skipped:
        print(f"ignored {len(skipped)}: {skipped}")
    if not buckets:
        raise SystemExit("nothing matched, fix --pattern so it matches how you named the sweep runs")

    rows = []
    for (damage, sev), vals in sorted(buckets.items()):
        arr = np.array(vals, dtype=float)
        rows.append({
            "damage": damage,
            "severity": sev,
            "n": len(arr),
            "chroma_ratio": float(np.median(arr[:, 0])),
            "dE_ab": float(np.mean(arr[:, 1])),
            "hue_err_deg": float(np.nanmean(arr[:, 2])),
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {args.out}\n")
    print("damage sev n chroma_ratio dE_ab hue_err")
    for r in rows:
        print(f"{r['damage']:<12} {r['severity']:>4} {r['n']:>4} {r['chroma_ratio']:>10.3f} {r['dE_ab']:>8.2f} {r['hue_err_deg']:>8.1f}")

    print("\nper damage type, does chroma drop as severity goes up, and does dE_ab drop with it:")
    for damage in sorted({r["damage"] for r in rows}):
        sub = sorted([r for r in rows if r["damage"] == damage], key=lambda r: r["severity"])
        if len(sub) < 3:
            print(f"  {damage}: only {len(sub)} severities, skipping the correlation")
            continue
        sev = np.array([r["severity"] for r in sub], dtype=float)
        cr = np.array([r["chroma_ratio"] for r in sub])
        de = np.array([r["dE_ab"] for r in sub])
        r_cr = np.corrcoef(sev, cr)[0, 1]
        r_de = np.corrcoef(sev, de)[0, 1]
        verdict = "fits the grey story" if (r_cr < 0 and r_de < 0) else "does NOT fit the grey story"
        print(f"  {damage}: chroma-vs-severity r={r_cr:+.2f}, dE-vs-severity r={r_de:+.2f} -> {verdict}")


if __name__ == "__main__":
    main()

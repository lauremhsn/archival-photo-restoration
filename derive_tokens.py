"""cast token per colour original: cool / neutral / warm from image-mean b*, tertiles on train only."""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from skimage import color

from condcolor.protocol import SPLIT, load_truth

TOKENS = ["cool", "neutral", "warm"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="splits/cast_tokens.csv")
    args = p.parse_args()

    rows = list(csv.DictReader(open(SPLIT, newline="", encoding="utf-8")))
    for r in rows:
        lab = color.rgb2lab(load_truth(r["id"]))
        r["a_mean"] = float(lab[..., 1].mean())
        r["b_mean"] = float(lab[..., 2].mean())

    tb = np.array([r["b_mean"] for r in rows if r["split"] == "train"])
    t1, t2 = np.quantile(tb, [1 / 3, 2 / 3])
    for r in rows:
        r["token"] = TOKENS[int(r["b_mean"] > t1) + int(r["b_mean"] > t2)]

    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "a_mean", "b_mean", "token"])
        for r in rows:
            w.writerow([r["id"], r["split"], f"{r['a_mean']:.3f}", f"{r['b_mean']:.3f}", r["token"]])

    print(f"thresholds from train: b* <= {t1:.2f} cool, <= {t2:.2f} neutral, else warm")
    for t in TOKENS:
        ids = [r for r in rows if r["split"] == "train" and r["token"] == t]
        print(f"{t} centre (a, b): ({np.mean([r['a_mean'] for r in ids]):.2f}, {np.mean([r['b_mean'] for r in ids]):.2f})")
    for s in ("train", "val", "test"):
        c = Counter(r["token"] for r in rows if r["split"] == s)
        print(f"{s}: " + " ".join(f"{t}={c[t]}" for t in TOKENS))
    print("token vs autochrome (all splits):")
    for ac in ("0", "1"):
        c = Counter(r["token"] for r in rows if r["autochrome"] == ac)
        print(f"autochrome={ac}: " + " ".join(f"{t}={c[t]}" for t in TOKENS))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Check a checkpoint for fixed-position colour artifacts.

Feeds blank inputs (black / mid-grey / white, two sizes) with the empty caption. A healthy
model has no reason to put strong colour anywhere on a blank input; a fixed-position artifact
shows up as a small patch of very high chroma at the same spot in every case. Also counts how
many validation images light up the same pixels.

Usage
  python -m condcolor.artifacts checkpoints/condunet/best.pt
"""

import argparse

import numpy as np
import torch

from condcolor import data, text
from condcolor.data import AB_SCALE
from condcolor.predict import load_model

HIGH = 40.0  # chroma in Lab units; real photos here have median chroma ~10-15


@torch.no_grad()
def chroma(net, L, feat, mask, dev):
    ab = net(L.to(dev), feat[None].to(dev), mask[None].to(dev))[0]
    return (ab.norm(dim=0) * AB_SCALE).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, c = load_model(args.ckpt, dev)
    table = text.load()
    feat, mask = table[None]
    print(f"{args.ckpt} (padding={net.padding})")

    worst = 0.0
    for name, val in (("black", -1.0), ("grey", 0.0), ("white", 0.9)):
        for h, w in ((256, 256), (256, 384)):
            ch = chroma(net, torch.full((1, 1, h, w), val), feat, mask, dev)
            y, x = np.unravel_index(ch.argmax(), ch.shape)
            worst = max(worst, ch.max())
            print(f"  blank {name:<5} {h}x{w}: max chroma {ch.max():6.1f} at (y={y}, x={x}); "
                  f"pixels > {HIGH:.0f}: {(ch > HIGH).sum()}")

    ids = data.split_ids("val")
    hits = np.zeros((256, 256))
    for iid in ids:
        L, _ = data.rgb_to_tensors(data.load_rgb(iid, 256, "center"))
        f, m = table[iid]
        hits += chroma(net, L[None], f, m, dev) > HIGH
    shared = int((hits > len(ids) / 2).sum())
    print(f"  val: pixels above {HIGH:.0f} in more than half of the {len(ids)} images: {shared}")
    print("  verdict:", "ARTIFACT (fixed-position patch)" if shared > 0 or worst > 80 else "clean")


if __name__ == "__main__":
    main()

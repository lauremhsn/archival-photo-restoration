"""Run a trained conditioned UNet on whole images and write protocol-ready predictions.

The model sees the full photo (not a crop): L is resized so the short side is --size and both
sides are multiples of 16 (the UNet halves the size four times); the predicted ab is then
upsampled to the original size and combined with the original L. The PNGs this writes go
through condcolor.protocol exactly like any other model's output.

Caption modes (#18): real | none (empty caption) | shuffled (another image's caption).

Usage
  python -m condcolor.predict checkpoints/condunet/best.pt --split test --captions real
  python -m condcolor.protocol outputs/condunet_real
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import color

from condcolor import text
from condcolor.data import AB_SCALE
from condcolor.model import CondUNet
from condcolor.protocol import load_truth, split_ids


def caption_keys(ids, mode):
    if mode == "real":
        return list(ids)
    if mode == "none":
        return [None] * len(ids)
    if mode == "shuffled":  # fixed derangement, same as data.ColourPairs
        return list(ids[1:]) + list(ids[:1])
    raise ValueError(mode)


@torch.no_grad()
def predict_rgb(net, truth_rgb, feat, mask, size=256, dev="cuda"):
    """Full-image colorization: returns RGB at the original resolution with the original L."""
    h, w = truth_rgb.shape[:2]
    lab = color.rgb2lab(truth_rgb)
    s = size / min(h, w)
    nh, nw = max(16, round(h * s / 16) * 16), max(16, round(w * s / 16) * 16)
    L = torch.from_numpy(lab[..., 0].astype(np.float32))[None, None] / 50.0 - 1.0
    L_small = F.interpolate(L, size=(nh, nw), mode="bilinear", align_corners=False).to(dev)
    ab = net(L_small, feat[None].to(dev), mask[None].to(dev))
    ab = F.interpolate(ab, size=(h, w), mode="bilinear", align_corners=False)[0].cpu().numpy()
    out = np.concatenate([lab[..., :1], ab.transpose(1, 2, 0) * AB_SCALE], axis=2)
    with warnings.catch_warnings():  # out-of-gamut Lab values are clipped; the warning is noise
        warnings.simplefilter("ignore")
        return np.clip(color.lab2rgb(out), 0, 1)


def load_model(ckpt, dev):
    c = torch.load(ckpt, map_location=dev)
    # checkpoints from before the padding option were trained with zero padding
    net = CondUNet(**{"padding": "zeros", **c["model_args"]}).to(dev).eval()
    net.load_state_dict(c["model"])
    return net, c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--captions", default="real", choices=["real", "none", "shuffled"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", help="default outputs/<run>_<captions>")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, c = load_model(args.ckpt, dev)
    out = Path(args.out or f"outputs/{c['run']}_{args.captions}")
    out.mkdir(parents=True, exist_ok=True)
    table = text.load()
    ids = split_ids(args.split)
    for iid, key in zip(ids, caption_keys(ids, args.captions)):
        feat, mask = table[key]
        rgb = predict_rgb(net, load_truth(iid), feat, mask, args.size, dev)
        Image.fromarray((rgb * 255).round().astype(np.uint8)).save(out / f"{iid}.png")
    print(f"wrote {len(ids)} predictions to {out}")


if __name__ == "__main__":
    main()

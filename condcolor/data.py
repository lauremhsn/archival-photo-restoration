"""Paired data for the conditioned colorizer: L in, ab out, caption features as the condition.

Images are resized so the short side is `size`, then cropped to size x size
(random crop for training, centre crop otherwise). Lab is normalised so both input and
target sit roughly in [-1, 1]:  L/50 - 1  and  ab/110.
"""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage import color
from torch.utils.data import Dataset

IMG_DIR = Path("data/matson/images")
SPLIT = Path("splits/colour_split.csv")
AB_SCALE = 110.0


def split_ids(split, path=SPLIT):
    with open(path, encoding="utf-8") as f:
        return [r["id"] for r in csv.DictReader(f) if r["split"] == split]


def load_rgb(iid, size, crop="center", rng=None):
    im = Image.open(IMG_DIR / f"{iid}.jpg").convert("RGB")
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = im.size
    if crop == "random":
        rng = rng or np.random
        x, y = rng.randint(0, w - size + 1), rng.randint(0, h - size + 1)
    else:
        x, y = (w - size) // 2, (h - size) // 2
    return np.asarray(im.crop((x, y, x + size, y + size)), dtype=np.float32) / 255.0


def rgb_to_tensors(rgb):
    lab = color.rgb2lab(rgb).astype(np.float32)
    L = torch.from_numpy(lab[..., :1]).permute(2, 0, 1) / 50.0 - 1.0
    ab = torch.from_numpy(lab[..., 1:]).permute(2, 0, 1) / AB_SCALE
    return L, ab


def tensors_to_rgb(L, ab):
    """Inverse of rgb_to_tensors for one image: (1,H,W), (2,H,W) -> HxWx3 float in [0,1]."""
    lab = torch.cat([(L + 1.0) * 50.0, ab * AB_SCALE]).permute(1, 2, 0).cpu().numpy()
    return np.clip(color.lab2rgb(lab.astype(np.float64)), 0, 1)


class ColourPairs(Dataset):
    """text: {id: (feat, mask)} from condcolor.text.load(). caption_mode:
    'real' (own caption), 'none' (empty caption), 'shuffled' (another image's caption)."""

    def __init__(self, ids, text, size=256, train=True, caption_mode="real", seed=693):
        self.ids, self.text, self.size, self.train = list(ids), text, size, train
        self.caption_mode = caption_mode
        self.rng = np.random.RandomState(seed)
        self.cache = {}
        if caption_mode == "shuffled":  # fixed derangement: nobody keeps their own caption
            perm = list(self.ids[1:]) + self.ids[:1]
            self.caption_of = dict(zip(self.ids, perm))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        iid = self.ids[i]
        if self.train:
            rgb = load_rgb(iid, self.size, "random", self.rng)
            if self.rng.rand() < 0.5:
                rgb = rgb[:, ::-1].copy()
            L, ab = rgb_to_tensors(rgb)
        else:  # deterministic: cache the converted tensors
            if iid not in self.cache:
                self.cache[iid] = rgb_to_tensors(load_rgb(iid, self.size, "center"))
            L, ab = self.cache[iid]
        key = {"real": iid, "none": None,
               "shuffled": getattr(self, "caption_of", {}).get(iid)}[self.caption_mode]
        feat, mask = self.text[key]
        return {"id": iid, "L": L, "ab": ab, "text": feat, "text_mask": mask}

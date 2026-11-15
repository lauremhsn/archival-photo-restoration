"""Frozen CLIP ViT-B/32 text features for every caption, computed once and cached.

CLIP never trains, so running it every step would only waste time and GPU memory.
Each caption becomes 77 token vectors of size 512 plus a mask of the real (non-padding) tokens.

Also stores the empty caption "" as the "no text" condition for the ablation (#18).

Usage
  python -m condcolor.text                       # writes data/cache/clip_text.pt
"""

import argparse
import csv
from pathlib import Path

import torch

CLIP_NAME = "openai/clip-vit-base-patch32"
MAX_TOKENS = 77
SPLIT = Path("splits/colour_split.csv")
CACHE = Path("data/cache/clip_text.pt")


def caption(row, field="title"):
    """The text the model reads. 'title' is the agreed default; 'title+notes' is longer but
    CLIP truncates at 77 tokens, so most of the notes would be cut anyway."""
    if field == "title":
        return row["title"].strip()
    return f"{row['title'].strip()}. {row['notes'].strip()}"


@torch.no_grad()
def encode(texts, device="cpu", batch=64):
    from transformers import CLIPTextModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(CLIP_NAME)
    enc = CLIPTextModel.from_pretrained(CLIP_NAME).eval().to(device)
    feats, masks = [], []
    for i in range(0, len(texts), batch):
        t = tok(texts[i:i + batch], padding="max_length", max_length=MAX_TOKENS,
                truncation=True, return_tensors="pt").to(device)
        feats.append(enc(**t).last_hidden_state.float().cpu())   # (B, 77, 512)
        masks.append(t["attention_mask"].bool().cpu())          # True = real token
    return torch.cat(feats), torch.cat(masks)


def build(field="title", device="cpu"):
    with open(SPLIT, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [r["id"] for r in rows]
    texts = [caption(r, field) for r in rows]
    feats, masks = encode(texts + [""], device)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids, "texts": texts, "field": field, "model": CLIP_NAME,
                "feats": feats[:-1], "masks": masks[:-1],
                "empty_feat": feats[-1], "empty_mask": masks[-1]}, CACHE)
    trunc = sum(int(m.sum()) == MAX_TOKENS for m in masks[:-1])
    print(f"cached {len(ids)} captions ({field}) -> {CACHE}; {trunc} hit the 77-token limit")


def load(path=CACHE):
    """{id: (feat (77,512), mask (77,))} plus the empty-caption entry under key None."""
    c = torch.load(path)
    table = {i: (f, m) for i, f, m in zip(c["ids"], c["feats"], c["masks"])}
    table[None] = (c["empty_feat"], c["empty_mask"])
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default="title", choices=["title", "title+notes"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    build(args.field, args.device)

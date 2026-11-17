"""
sd predictions in the shape rola's protocol wants: one png per id in outputs/<name>/.

caption modes match condcolor.predict so the ablation lines up:
  real | none | shuffled | generic

  python sd_predict.py --captions real --split test
  python -m condcolor.protocol outputs/sd768_real
"""

import argparse, os, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from condcolor.protocol import split_ids

BASE = "runwayml/stable-diffusion-v1-5"
CONTROLNET = "latentcat/control_v1p_sd15_brightness"
IMG_DIR = Path("data/matson/images")
SPLIT = Path("splits/colour_split.csv")
GENERIC = "a colour photograph"


def captions(ids, mode):
    import csv
    with open(SPLIT, encoding="utf-8") as f:
        t = {r["id"]: (r.get("title") or "").strip() for r in csv.DictReader(f)}
    if mode == "real":
        return [t.get(i, "") or GENERIC for i in ids]
    if mode == "none":
        return [""] * len(ids)
    if mode == "generic":
        return [GENERIC] * len(ids)
    if mode == "shuffled":  # same derangement condcolor.predict uses
        sh = list(ids[1:]) + list(ids[:1])
        return [t.get(i, "") or GENERIC for i in sh]
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default="real",
                    choices=["real", "none", "generic", "shuffled"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cn", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs the gpu")
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

    name = args.name or f"sd{args.size}_{args.captions}"
    out_dir = Path("outputs") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = split_ids(args.split)
    caps = captions(ids, args.captions)
    todo = [(i, c) for i, c in zip(ids, caps) if not (out_dir / f"{i}.png").exists()]
    print(f"{name}: {len(ids)} {args.split} ids, {len(todo)} left to do")
    if not todo:
        print(f"already done. now run:  python -m condcolor.protocol outputs/{name} --split {args.split}")
        return

    net = ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE, controlnet=net, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False).to("cuda")
    pipe.enable_attention_slicing()
    try:
        pipe.vae.enable_tiling()
    except AttributeError:
        pass
    pipe.set_progress_bar_config(disable=True)

    t0 = time.time()
    for n, (iid, cap) in enumerate(todo, 1):
        src = Image.open(IMG_DIR / f"{iid}.jpg").convert("RGB")
        sq = src.resize((args.size, args.size), Image.LANCZOS)
        g = np.asarray(sq.convert("L"))
        grey = Image.fromarray(np.stack([g] * 3, axis=2))
        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        try:
            img = pipe(cap[:300], image=grey, height=args.size, width=args.size,
                       num_inference_steps=args.steps,
                       controlnet_conditioning_scale=args.cn, generator=gen).images[0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  {iid} oom, skipped")
            continue
        # protocol resizes and takes the original L, so just hand it back at source size
        img.resize(src.size, Image.BICUBIC).save(out_dir / f"{iid}.png")
        if n % 10 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {el/n:.1f}s each  ~{el/n*(len(todo)-n)/60:.0f} min left")

    print(f"\nwrote {out_dir}")
    print(f"now run:  python -m condcolor.protocol outputs/{name} --split {args.split}")


if __name__ == "__main__":
    main()
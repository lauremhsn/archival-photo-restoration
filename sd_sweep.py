"""
figures out which settings actually keep the photo's structure instead of
inventing a new picture.

python sd_sweep.py
python sd_sweep.py --n 5 --steps 25
"""

import argparse
import csv
import glob
import os
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage import color as skcolor
from skimage.metrics import structural_similarity as ssim

CONTROLNET = "latentcat/control_v1p_sd15_brightness"
BASE = "runwayml/stable-diffusion-v1-5"
OUT_DIR = "outputs/sd_sweep"

# name, mode, controlnet scale, img2img strength
CONFIGS = [
    ("txt2img cn1.0", "txt", 1.0, None),
    ("txt2img cn1.5", "txt", 1.5, None),
    ("txt2img cn2.0", "txt", 2.0, None),
    ("img2img s0.4 cn1.0", "img", 1.0, 0.4),
    ("img2img s0.6 cn1.0", "img", 1.0, 0.6),
    ("img2img s0.4 cn1.5", "img", 1.5, 0.4),
    ("img2img s0.6 cn1.5", "img", 1.5, 0.6),
]


def lab(a):
    return skcolor.rgb2lab(np.asarray(a).astype(np.float32) / 255.0)


def recombine(orig, pred):
    o, p = lab(orig), lab(pred)
    merged = np.dstack([o[:, :, 0], p[:, :, 1], p[:, :, 2]])
    return (np.clip(skcolor.lab2rgb(merged), 0, 1) * 255).astype(np.uint8)


def structure_score(orig, pred):
    """how much of the original's shape survived. 1.0 = untouched, lower = sd made stuff up."""
    a, b = lab(orig)[:, :, 0], lab(pred)[:, :, 0]
    return float(ssim(a, b, data_range=100.0))


def chroma(a):
    l = lab(a)
    return float(np.sqrt(l[:, :, 1] ** 2 + l[:, :, 2] ** 2).mean())


def label_strip(img, text, h=22):
    out = Image.new("RGB", (img.width, img.height + h), (20, 20, 20))
    out.paste(img, (0, h))
    ImageDraw.Draw(out).text((4, 5), text, fill=(235, 235, 235))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="a colour photograph")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs the gpu")

    from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                           StableDiffusionControlNetImg2ImgPipeline)

    files = sorted(glob.glob("data/matson/images/*.jpg"))[:args.n]
    if not files:
        raise SystemExit("no images in data/matson/images/")

    print(f"loading {CONTROLNET}")
    net = ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=torch.float16)
    txt = StableDiffusionControlNetPipeline.from_pretrained(
        BASE, controlnet=net, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False).to("cuda")
    img = StableDiffusionControlNetImg2ImgPipeline(**txt.components).to("cuda")
    for p in (txt, img):
        p.enable_attention_slicing()
    print(f"ready. {len(files)} images x {len(CONFIGS)} configs\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    t0 = time.time()

    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        orig = Image.open(path).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
        grey_arr = np.asarray(orig.convert("L"))
        grey = Image.fromarray(np.stack([grey_arr] * 3, axis=2))

        panels = [label_strip(grey, "original (greyscale)")]
        print(f"{stem}")

        for name, mode, cn, strength in CONFIGS:
            g = torch.Generator(device="cuda").manual_seed(args.seed)
            if mode == "txt":
                out = txt(args.prompt, image=grey, num_inference_steps=args.steps,
                          controlnet_conditioning_scale=cn, generator=g).images[0]
            else:
                out = img(args.prompt, image=grey, control_image=grey,
                          strength=strength, num_inference_steps=args.steps,
                          controlnet_conditioning_scale=cn, generator=g).images[0]

            s = structure_score(orig, out)
            fixed = recombine(orig, out)
            panels.append(label_strip(Image.fromarray(fixed), f"{name}  ssim {s:.2f}"))
            rows.append({"image": stem, "config": name, "mode": mode,
                         "cn_scale": cn, "strength": strength,
                         "structure_ssim": round(s, 4),
                         "chroma_raw": round(chroma(out), 2),
                         "chroma_fixed": round(chroma(fixed), 2)})
            print(f"  {name:<20} structure {s:.3f}  chroma {chroma(out):.1f}")

        w, h = panels[0].size
        sheet = Image.new("RGB", (w * len(panels), h), (20, 20, 20))
        for i, p in enumerate(panels):
            sheet.paste(p, (i * w, 0))
        sheet.save(f"{OUT_DIR}/{stem}_sweep.png")
        print(f"  -> {OUT_DIR}/{stem}_sweep.png\n")

    with open(f"{OUT_DIR}/sweep.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"all done in {time.time()-t0:.0f}s\n")
    print("averaged over the images:")
    print(f"{'config':<20} {'structure':>10} {'chroma':>8}")
    for name, _, _, _ in CONFIGS:
        sel = [r for r in rows if r["config"] == name]
        ms = np.mean([r["structure_ssim"] for r in sel])
        mc = np.mean([r["chroma_fixed"] for r in sel])
        print(f"{name:<20} {ms:>10.3f} {mc:>8.1f}")

    best = max(CONFIGS, key=lambda c: np.mean(
        [r["structure_ssim"] for r in rows if r["config"] == c[0]]))
    print(f"\nbest structure: {best[0]}")
    print("structure above ~0.85 means sd kept the photo. below ~0.6 it painted something else.")
    print("look at the sheets too, the number doesnt catch everything")


if __name__ == "__main__":
    main()

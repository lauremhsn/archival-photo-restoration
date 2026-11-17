"""
does giving sd a bigger latent grid fix the structure problem?
512 -> 64x64 latent. 1024 -> 128x128. four times the room for our photo's shape.

python sd_res.py
"""

import argparse, csv, glob, os, time
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage import color as skcolor
from skimage.metrics import structural_similarity as ssim

BASE = "runwayml/stable-diffusion-v1-5"
CONTROLNET = "latentcat/control_v1p_sd15_brightness"
OUT_DIR = "outputs/sd_res"


def lab(a):
    return skcolor.rgb2lab(np.asarray(a).astype(np.float32) / 255.0)


def recombine(orig, pred):
    o, p = lab(orig), lab(pred)
    return (np.clip(skcolor.lab2rgb(
        np.dstack([o[:, :, 0], p[:, :, 1], p[:, :, 2]])), 0, 1) * 255).astype(np.uint8)


def structure_score(o, p):
    return float(ssim(lab(o)[:, :, 0], lab(p)[:, :, 0], data_range=100.0))


def chroma(a):
    l = lab(a)
    return float(np.sqrt(l[:, :, 1] ** 2 + l[:, :, 2] ** 2).mean())


def delta_e_ab(a, b):
    la, lb = lab(a)[:, :, 1:], lab(b)[:, :, 1:]
    return float(np.sqrt(((la - lb) ** 2).sum(axis=2)).mean())


def strip(img, text, h=22):
    out = Image.new("RGB", (img.width, img.height + h), (20, 20, 20))
    out.paste(img, (0, h))
    ImageDraw.Draw(out).text((4, 5), text, fill=(235, 235, 235))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cn", type=float, default=1.5)
    ap.add_argument("--prompt", default="a colour photograph")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs the gpu")
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

    # colour originals only, so we can score against real ground truth
    import csv as _csv
    with open("matson.csv", encoding="utf-8") as fh:
        meta = {r["id"]: r for r in _csv.DictReader(fh)}

    def is_colour(i):
        m = meta.get(i, {}).get("medium", "").lower()
        return "color" in m and "hand-color" not in m

    files = [f for f in sorted(glob.glob("data/matson/images/*.jpg"))
             if is_colour(os.path.splitext(os.path.basename(f))[0])][:args.n]
    if not files:
        raise SystemExit("no colour originals found, check matson.csv")

    print(f"loading, {len(files)} colour originals (we have real colour to score against)")
    net = ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE, controlnet=net, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False).to("cuda")
    pipe.enable_attention_slicing()
    try:
        pipe.vae.enable_tiling()
    except AttributeError:
        pass

    SIZES = [512, 768, 1024]
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        print(f"\n{stem}  {meta.get(stem,{}).get('title','')[:50]}")
        panels = []
        for size in SIZES:
            truth = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
            g = np.asarray(truth.convert("L"))
            grey = Image.fromarray(np.stack([g] * 3, axis=2))

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            gen = torch.Generator(device="cuda").manual_seed(args.seed)
            try:
                out = pipe(args.prompt, image=grey, height=size, width=size,
                           num_inference_steps=args.steps,
                           controlnet_conditioning_scale=args.cn, generator=gen).images[0]
            except torch.cuda.OutOfMemoryError:
                print(f"  {size}px  out of memory, skipping")
                torch.cuda.empty_cache()
                continue
            secs = time.time() - t0
            vram = torch.cuda.max_memory_allocated() / 1e9

            s = structure_score(truth, out)
            fixed = recombine(truth, out)
            de = delta_e_ab(truth, fixed)
            rows.append({"image": stem, "size": size, "structure_ssim": round(s, 4),
                         "delta_e_ab": round(de, 2), "chroma_pred": round(chroma(fixed), 2),
                         "chroma_true": round(chroma(truth), 2),
                         "vram_gb": round(vram, 2), "secs": round(secs, 1)})
            print(f"  {size}px  structure {s:.3f}  dE_ab {de:5.2f}  "
                  f"chroma {chroma(fixed):5.1f} vs {chroma(truth):5.1f} true  "
                  f"{vram:.1f}GB  {secs:.0f}s")
            panels.append(strip(Image.fromarray(fixed).resize((512, 512)),
                                f"{size}px  dE {de:.1f}  ssim {s:.2f}"))

        if panels:
            t512 = Image.open(path).convert("RGB").resize((512, 512), Image.LANCZOS)
            panels = [strip(t512, "TRUE COLOUR")] + panels
            w, h = panels[0].size
            sheet = Image.new("RGB", (w * len(panels), h), (20, 20, 20))
            for i, p in enumerate(panels):
                sheet.paste(p, (i * w, 0))
            sheet.save(f"{OUT_DIR}/{stem}_res.png")

    if not rows:
        raise SystemExit("nothing completed")
    with open(f"{OUT_DIR}/res.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    print(f"\n{'size':<8} {'structure':>10} {'dE_ab':>8} {'vram':>7} {'secs':>7}")
    for size in SIZES:
        sel = [r for r in rows if r["size"] == size]
        if not sel: continue
        print(f"{str(size)+'px':<8} {np.mean([r['structure_ssim'] for r in sel]):>10.3f} "
              f"{np.mean([r['delta_e_ab'] for r in sel]):>8.2f} "
              f"{max(r['vram_gb'] for r in sel):>6.1f}G {np.mean([r['secs'] for r in sel]):>6.0f}s")
    print("\nsiggraph17 gets dE_ab 16.20 on the full set. thats the number to beat.")
    print("if dE_ab doesnt come down near that, latent diffusion just cant do this")
    print("job faithfully and thats our finding.")


if __name__ == "__main__":
    main()
